"""ML variants. V1: global LightGBM, L1 on capacity-relative target, sample weight = scale (== raw MAE), target-history lags.
V2: LightGBM Tweedie, per business-flag models, NO target lags (weather + calendar + capacity + a seasonal unit-hour profile fitted on the fit role only).
Progress/persistence only: model file + atomic history JSON (no live prints required on a headless host)."""
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import CAL, LAGCOLS, SCALE, STATIC, WEATHER


def _atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def add_profile(panel, fit_mask):
    """Seasonal profile = mean y_norm per (unit, is_consumption, month, hour) over the fit rows only."""
    g = panel[fit_mask].groupby(["prediction_unit_id", "is_consumption", "month", "hour"]).y_norm.mean().rename("profile")
    return panel.join(g, on=["prediction_unit_id", "is_consumption", "month", "hour"])


def v1_features():
    return STATIC + CAL + WEATHER + SCALE + ["price_lag24"] + LAGCOLS


def v2_features():
    return [c for c in STATIC if c != "prediction_unit_id"] + ["prediction_unit_id"] + CAL + WEATHER + SCALE + ["price_lag24", "profile"]


PARAMS_L1 = dict(objective="regression_l1", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, verbose=-1, num_threads=0)
PARAMS_TW = dict(objective="tweedie", tweedie_variance_power=1.1, learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, verbose=-1, num_threads=0)


def _dataset(df, feats, weight=True):
    X = df[feats]
    return lgb.Dataset(X, label=df.y_norm.values, weight=(df.scale.values if weight else None),
                       categorical_feature=["prediction_unit_id"] if "prediction_unit_id" in feats else "auto", free_raw_data=False)


class HistoryCallback:
    def __init__(self, path, every=25):
        self.path, self.every, self.t0 = path, every, time.time()
        self.rows = []

    def __call__(self, env):
        it = env.iteration + 1
        row = {"iter": it, "sec": round(time.time() - self.t0, 1)}
        for data, metric, val, _ in env.evaluation_result_list:
            row[f"{data}_{metric}"] = val
        self.rows.append(row)
        if it % self.every == 0:
            _atomic_json(self.path, {"rows": self.rows})


def fit_group(tr, va, feats, params, rounds, out_prefix, early=75):
    """Train one booster; early stopping on the choose role only. Returns (booster, best_iter)."""
    dtr = _dataset(tr, feats)
    sets, names = [dtr], ["fit"]
    if va is not None and len(va):
        sets.append(lgb.Dataset(va[feats], label=va.y_norm.values, weight=va.scale.values, reference=dtr,
                                categorical_feature=["prediction_unit_id"] if "prediction_unit_id" in feats else "auto"))
        names.append("choose")
    hist = HistoryCallback(out_prefix + "_history.json")
    cbs = [hist]
    if len(sets) > 1 and early:
        cbs.append(lgb.early_stopping(early, first_metric_only=True, verbose=False))
    ev = {}
    cbs.append(lgb.record_evaluation(ev))
    b = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=sets, valid_names=names, callbacks=cbs)
    _atomic_json(out_prefix + "_history.json", {"rows": hist.rows, "best_iteration": b.best_iteration or b.current_iteration()})
    b.save_model(out_prefix + "_model.txt")        # checkpoint = the model itself, written when training ends
    return b, (b.best_iteration or b.current_iteration())


SAME_SEASON = (pd.Timestamp("2022-04-02"), pd.Timestamp("2022-06-01"))   # same calendar window as the report role, one year earlier


def tune_rounds(panel, valid, g, feats, params, fit_mask, choose_mask, max_rounds, out_prefix):
    """Section 2.2 of the M5 blending paper: choose the number of rounds on SEVERAL validation windows and average, then full-train.
    Window A = last 28 days before the report role (fit-before-choose). Window B = the same season one year earlier, held out of
    that fit (leave-window-out because only 21 months of history exist). Curves are normalised by their minimum before averaging."""
    d = panel.datetime
    in_b = ((d >= SAME_SEASON[0]) & (d < SAME_SEASON[1])).values
    wins = [("last28", fit_mask & ~choose_mask & ~in_b, choose_mask), ("same_season", fit_mask & ~in_b, in_b & fit_mask)]
    curves, info = [], {}
    for name, trm, vam in wins:
        tr, va = panel[trm & g & valid], panel[vam & g & valid]
        if len(tr) == 0 or len(va) == 0:
            continue
        dtr = _dataset(tr, feats)
        dva = lgb.Dataset(va[feats], label=va.y_norm.values, weight=va.scale.values, reference=dtr,
                          categorical_feature=["prediction_unit_id"] if "prediction_unit_id" in feats else "auto")
        ev = {}
        hist = HistoryCallback(f"{out_prefix}_tune_{name}_history.json", every=50)
        lgb.train(params, dtr, num_boost_round=max_rounds, valid_sets=[dva], valid_names=["v"], callbacks=[lgb.record_evaluation(ev), hist])
        _atomic_json(f"{out_prefix}_tune_{name}_history.json", {"rows": hist.rows})
        c = np.array(next(iter(ev["v"].values())))
        curves.append(c / max(c.min(), 1e-12))
        info[name] = {"best_iter": int(c.argmin()) + 1, "rows_train": int(len(tr)), "rows_val": int(len(va))}
    mean = np.mean(np.stack(curves), 0)
    best = int(mean.argmin()) + 1
    info["chosen_rounds"] = best
    return best, info


def run_variant(panel, masks, variant, out_dir, rounds=1500, fit_all=False):
    """Returns DataFrame of predictions (raw units) for rows in masks['choose'] | masks['report'] (or all rows if fit_all)."""
    os.makedirs(out_dir, exist_ok=True)
    fit = masks["fit"] | masks["choose"] | masks["report"] if fit_all else masks["fit"]
    pred = pd.Series(np.nan, index=panel.index, dtype="float32")
    notes = {}
    tuned = variant.endswith("m")
    base = variant[:2]
    if base == "v1":
        feats, params = v1_features(), PARAMS_L1
        groups = [("c0", panel.is_consumption == 0), ("c1", panel.is_consumption == 1)]
    elif base == "v2":
        panel = add_profile(panel, fit)
        feats, params = v2_features(), PARAMS_TW
        groups = [(f"c{c}_b{b}", (panel.is_consumption == c) & (panel.is_business == b)) for c in (0, 1) for b in (0, 1)]
    else:
        raise ValueError(variant)
    valid = panel.y_norm.notna()
    for name, g in groups:
        tr = panel[fit & g & valid]
        prefix = os.path.join(out_dir, f"{variant}_{name}")
        if tuned:
            best, info = tune_rounds(panel, valid, g, feats, params, fit, masks["choose"], rounds, prefix)
            b, _ = fit_group(tr, None, feats, params, int(best * (1.1 if fit_all else 1.0)), prefix, early=0)
            best = int(best * (1.1 if fit_all else 1.0))
            notes[name] = {"rows_fit": int(len(tr)), "rounds": best, "tuning": info}
        else:
            va = None if fit_all else panel[masks["choose"] & g & valid]
            b, best = fit_group(tr, va, feats, params, rounds, prefix)
            notes[name] = {"rows_fit": int(len(tr)), "best_iter": int(best)}
        sel = g & (masks["choose"] | masks["report"]) if not fit_all else g
        pred.loc[sel[sel].index] = b.predict(panel.loc[sel, feats], num_iteration=best).astype("float32")
    out = panel[["prediction_unit_id", "is_consumption", "datetime", "scale", "y"]].copy()
    out["pred"] = (pred * panel.scale).astype("float32")
    out["pred"] = out.pred.clip(lower=0)
    _atomic_json(os.path.join(out_dir, f"{variant}_notes.json"), notes)
    return out, notes
