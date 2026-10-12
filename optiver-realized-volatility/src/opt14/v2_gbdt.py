"""V2 - LightGBM on log target with the full engineered feature table (+ same-time_id cross-stock aggregates).
Trained in chunks so progress, early stopping on the choose role (RMSPE after the optimal scalar) and the wall-clock deadline are under our control.
Run twice by run_cpu.py in separate processes: with cross-stock features and without (the ablation that measures their value)."""
import json, time
import numpy as np, pandas as pd
import lightgbm as lgb
from .common import rmspe, best_scalar
from .data import feature_columns
from .obs import History, log, peak_rss_mb, Deadline

PARAMS = dict(objective="regression", learning_rate=0.06, num_leaves=63, min_child_samples=100, feature_fraction=0.6,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=255, verbose=-1, seed=0)


def run(df, out_dir, name, with_cross, threads, budget_s, logf=None, chunk=25, patience=200, max_rounds=4000, params=None):
    cols = feature_columns(df, with_cross)
    fit, ch, rep = (df[df.role == r] for r in (0, 1, 2))
    mk = lambda d: lgb.Dataset(d[cols].to_numpy(np.float32), label=np.log(d.target.to_numpy()), free_raw_data=False)
    dtrain = mk(fit); dtrain.categorical_feature = [cols.index("stock_id")] if "stock_id" in cols else "auto"
    Xc, Xr = ch[cols].to_numpy(np.float32), rep[cols].to_numpy(np.float32)
    yc, yr = ch.target.to_numpy(), rep.target.to_numpy()
    p = dict(PARAMS, **(params or {}), num_threads=threads)
    hist = History(f"{out_dir}/{name}_history.json", {"variant": name, "cross": with_cross, "n_features": len(cols), "threads": threads, "budget_s": budget_s})
    dl, t0 = Deadline(budget_s), time.time()
    booster, best, best_it, since, rounds = None, np.inf, 0, 0, 0
    sub = fit.sample(min(40000, len(fit)), random_state=0)
    Xs, ys = sub[cols].to_numpy(np.float32), sub.target.to_numpy()
    stop = "max_rounds"
    while rounds < max_rounds:
        booster = lgb.train(p, dtrain, num_boost_round=chunk, init_model=booster, keep_training_booster=True)
        rounds += chunk
        pc = np.exp(booster.predict(Xc)); c = best_scalar(yc, pc)
        cr = rmspe(yc, c * pc)
        if cr < best - 1e-6: best, best_it, since = cr, rounds, 0; booster.save_model(f"{out_dir}/{name}_best.txt")
        else: since += chunk
        if rounds % 100 == 0 or since == 0:
            ps = np.exp(booster.predict(Xs)); cs = best_scalar(ys, ps)
            hist.add({"round": rounds, "choose_rmspe": cr, "fit_rmspe": rmspe(ys, cs * ps), "gap": cr - rmspe(ys, cs * ps), "scalar": c,
                      "report_rmspe_monitor_only": rmspe(yr, c * np.exp(booster.predict(Xr))),
                      "since_best": since, "sec": round(time.time() - t0, 1), "rounds_per_s": round(rounds / (time.time() - t0), 2), "rss_mb": round(peak_rss_mb())})
            log(logf, f"{name} round {rounds} choose RMSPE {cr:.4f} (best {best:.4f}@{best_it}) {rounds/(time.time()-t0):.1f} r/s since_best {since} left {dl.left():.0f}s")
        if since >= patience: stop = "early_stop"; break
        if dl.hit(): stop = "deadline"; break
    bst = lgb.Booster(model_file=f"{out_dir}/{name}_best.txt")
    pc = np.exp(bst.predict(Xc)); c = best_scalar(yc, pc); pr = c * np.exp(bst.predict(Xr))
    out = pd.concat([pd.DataFrame({"stock_id": d.stock_id.to_numpy(), "time_id": d.time_id.to_numpy(), "target": d.target.to_numpy(), "pred": p_, "role": r})
                     for d, p_, r in ((ch, c * pc, 1), (rep, pr, 2))])
    out.to_parquet(f"{out_dir}/{name}_predictions.parquet")
    imp = pd.Series(bst.feature_importance("gain"), index=cols).sort_values(ascending=False)
    imp.to_csv(f"{out_dir}/{name}_importance.csv")
    res = {"variant": name, "stop": stop, "best_round": best_it, "rounds": rounds, "scalar": c, "choose_rmspe": rmspe(yc, c * pc),
           "report_rmspe": rmspe(yr, pr), "seconds": round(time.time() - t0, 1), "top_features": list(imp.index[:8])}
    hist.meta.update(res); hist.flush()
    log(logf, f"{name} FINAL {json.dumps(res)}")
    return res
