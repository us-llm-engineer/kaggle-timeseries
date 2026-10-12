"""Row-level stack over the grid U-Nets: everything at row level, every fitted quantity fitted on the four
other folds and applied to the held-out fold.

Level 0  A  = snapshot blend of the grid U-Nets (all widths, epochs in the second half of training).
Level 0  B  = A + margin, margin = LightGBM on engineered absolute-frame features + A itself (target y - A).
Level 1  gate: hard per-well choice A vs B from target-free well aggregates (LightGBM classifier).
Level 2  scale direction d_s = blend(widest) - blend(narrowest), one coefficient per fold (least squares).
Level 2  seed direction  d_r = seed-average(width W) - gated base, one coefficient per fold (when seeds exist).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from multiprocessing import Pool

import tracker as lt
from wells import append_jsonl, eligible_ids, fold_of, load_wells, suffix_metrics

GRID_BS = 16


def load_grid(res: Path, folds_ids: dict) -> dict:
    """{(width, seed): {well: {epoch: block offsets}}} from all fold snapshot files."""
    out = {}
    for f in sorted(res.glob("fold*_w*_s*_snapshots.npz")):
        tag = f.name.replace("_snapshots.npz", "")
        _, w, s = tag.split("_")
        key = (int(w[1:]), int(s[1:]))
        z = np.load(f)
        d = out.setdefault(key, {})
        for k in z.files:
            if "__" not in k:
                continue
            ep, wid = k.split("__")
            d.setdefault(wid, {})[int(ep[2:])] = z[k]
    return out


def to_rows(w, block_off, bs):
    p, n = w["p"], w["n"]; L = len(block_off)
    centers = p + np.arange(L) * bs + (bs - 1) / 2
    return float(w["tvt_input"][p - 1]) + np.interp(np.arange(p, n), np.r_[p - 1, centers], np.r_[0.0, block_off])


def blend(grid, wells, wid, keys, epochs_from_half=True):
    rows = []
    for key in keys:
        eps = grid[key].get(wid)
        if not eps:
            continue
        E = max(eps)
        use = [e for e in eps if e > E // 2] if epochs_from_half else [E]
        rows += [to_rows(wells[wid], eps[e], GRID_BS) for e in use]
    return np.mean(rows, 0) if rows else None


def row_features(w, A):
    """Absolute-frame row features (no target): geometry, anchor, GR vs typewell at A, and A itself."""
    p, n = w["p"], w["n"]; r = np.arange(p, n)
    a, b, s = lt.calibrate(w)
    tw = lambda T: a + b * np.interp(T, w["tw_tvt"], w["tw_gr"], left=np.nan, right=np.nan)
    tps = float(w["tvt_input"][p - 1])
    g = w["gr"][p:]
    gm = np.convolve(np.nan_to_num(g), np.ones(31) / 31, "same")
    lo = max(0, p - 300)
    u_slope = ((w["tvt_input"][p - 1] + w["z"][p - 1]) - (w["tvt_input"][lo] + w["z"][lo])) / max(w["md"][p - 1] - w["md"][lo], 1)
    dmd = w["md"][p:] - w["md"][p - 1]; dz = w["z"][p:] - w["z"][p - 1]
    off = A - tps
    return np.c_[w["x"][p:], w["y"][p:], w["z"][p:], w["x"][p:] - w["x"][p - 1], w["y"][p:] - w["y"][p - 1], dz, dmd,
                 np.full(len(r), tps), np.full(len(r), u_slope), u_slope * dmd - dz, off, np.gradient(off),
                 (g - tw(A)) / s, (gm - tw(A)) / s, (g - tw(A + 2)) / s, (g - tw(A - 2)) / s, np.isnan(g).astype(float),
                 np.full(len(r), s), np.full(len(r), b), r - p].astype(np.float32)


def well_aggregates(w, A, B):
    """Target-free per-well summaries used by the gate."""
    p, n = w["p"], w["n"]; g = w["gr"][p:]
    a, b, s = lt.calibrate(w)
    tw = lambda T: a + b * np.interp(T, w["tw_tvt"], w["tw_gr"], left=np.nan, right=np.nan)
    rA = np.abs(g - tw(A)) / s; rB = np.abs(g - tw(B)) / s
    return np.array([n - p, p, np.mean(np.isnan(g)), np.nanmean(rA), np.nanmean(rB), np.nanmean(rA) - np.nanmean(rB),
                     np.mean(np.abs(A - B)), np.max(np.abs(A - B)), np.std(A - float(w["tvt_input"][p - 1])), s, b,
                     np.nanmedian(rA), np.nanmedian(rB)], np.float32)


def gpu_margin(X, y, rounds, depth):
    """Huber-loss gradient-boosted margin model on the local GPU (XGBoost CUDA hist)."""
    d = xgb.DMatrix(X, label=y)
    return xgb.train({"objective": "reg:pseudohubererror", "huber_slope": 5.0, "eta": 0.05, "max_depth": depth,
                      "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist",
                      "device": "cuda", "max_bin": 256}, d, rounds)


def gpu_predict(m, X):
    return m.predict(xgb.DMatrix(X)) if isinstance(m, xgb.Booster) else m.predict(X, num_threads=9)


MARGIN = "lgbm"


def fit_margin(X, y, outer: bool):
    """Margin model backend: 'lgbm' = the CPU LightGBM recipe that scored 13.16 (folds 2-5, two widths); 'xgb' = GPU."""
    if MARGIN == "xgb":
        return gpu_margin(X, y, 500, 7)
    if outer:
        return lgb.train({"objective": "huber", "alpha": 5.0, "learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 300,
                          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "num_threads": 9, "verbose": -1},
                         lgb.Dataset(X, y), 500)
    return lgb.train({"objective": "huber", "alpha": 5.0, "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 300,
                      "num_threads": 9, "verbose": -1}, lgb.Dataset(X, y), 250)


_W = {}


def _feat_job(args):
    wid, A = args
    return wid, row_features(_W["wells"][wid], A)


def lsq_coef(resid_list, dir_list):
    r = np.concatenate(resid_list); d = np.concatenate(dir_list)
    return float((r * d).sum() / max((d * d).sum(), 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="data/stage")
    ap.add_argument("--grid", default="runs/grid")
    ap.add_argument("--out", default="runs/stack")
    ap.add_argument("--seed-width", type=int, default=24)
    ap.add_argument("--margin", choices=["lgbm", "xgb"], default="lgbm")
    args = ap.parse_args()
    global MARGIN
    MARGIN = args.margin
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    wells = load_wells(args.stage); ids = eligible_ids(wells); folds = fold_of(ids)
    grid = load_grid(Path(args.grid), folds)
    widths = sorted({k[0] for k in grid}); base_keys = [k for k in grid if k[1] == 0]
    seed_keys = [k for k in grid if k[0] == args.seed_width]
    have = [i for i in ids if all(i in grid[k] for k in base_keys)]
    print(json.dumps({"keys": [list(k) for k in grid], "wells_with_all_base": len(have)}), flush=True)
    T, A, Dn, Dw, S = {}, {}, {}, {}, {}
    for wid in have:
        w = wells[wid]
        T[wid] = w["tvt"][w["p"]:]
        A[wid] = blend(grid, wells, wid, base_keys)
        Dn[wid] = blend(grid, wells, wid, [(widths[0], 0)]); Dw[wid] = blend(grid, wells, wid, [(widths[-1], 0)])
        S[wid] = blend(grid, wells, wid, seed_keys) if len(seed_keys) > 1 else None
    _W["wells"] = wells
    with Pool(9) as pool:
        F = dict(pool.map(_feat_job, [(wid, A[wid]) for wid in have], chunksize=8))
    stages = {"carry": [], "A_blend": [], "B_margin": [], "gated": [], "gated+scale": [], "final": []}
    log = []
    for k in range(5):
        tr = [i for i in have if folds[i] != k]; te = [i for i in have if folds[i] == k]
        # margin model on training folds (rows subsampled for speed)
        Xtr = np.concatenate([F[i][::4] for i in tr]); ytr = np.concatenate([(T[i] - A[i])[::4] for i in tr])
        mdl = fit_margin(Xtr, ytr, True)
        # out-of-fold margins for the training wells are needed for the gate: inner 4-fold cross-fit
        Btr = {}
        inner = {i: folds[i] for i in tr}
        for j in sorted(set(inner.values())):
            a_ = [i for i in tr if inner[i] != j]; b_ = [i for i in tr if inner[i] == j]
            st = 4 if MARGIN == 'xgb' else 6
            m_ = fit_margin(np.concatenate([F[i][::st] for i in a_]), np.concatenate([(T[i] - A[i])[::st] for i in a_]), False)
            for i in b_:
                Btr[i] = A[i] + gpu_predict(m_, F[i])
        Bte = {i: A[i] + gpu_predict(mdl, F[i]) for i in te}
        # gate: label = B better than A on the well
        Gx = np.array([well_aggregates(wells[i], A[i], Btr[i]) for i in tr])
        Gy = np.array([np.mean((Btr[i] - T[i]) ** 2) < np.mean((A[i] - T[i]) ** 2) for i in tr]).astype(int)
        gate = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 7, "min_data_in_leaf": 20, "verbose": -1, "num_threads": 9},
                         lgb.Dataset(Gx, Gy), 150)
        pick = {i: gate.predict(well_aggregates(wells[i], A[i], Bte[i])[None])[0] > 0.5 for i in te}
        gated_te = {i: (Bte[i] if pick[i] else A[i]) for i in te}
        # gated base on training wells (with their OOF B) to fit the direction coefficients
        pick_tr = dict(zip(tr, gate.predict(Gx) > 0.5))
        gated_tr = {i: (Btr[i] if pick_tr[i] else A[i]) for i in tr}
        c_s = lsq_coef([T[i] - gated_tr[i] for i in tr], [Dw[i] - Dn[i] for i in tr])
        fin_te = {i: gated_te[i] + c_s * (Dw[i] - Dn[i]) for i in te}
        c_r = None
        if S[te[0]] is not None:
            c_r = lsq_coef([T[i] - gated_tr[i] - c_s * (Dw[i] - Dn[i]) for i in tr], [S[i] - gated_tr[i] for i in tr])
            fin_te = {i: fin_te[i] + c_r * (S[i] - gated_te[i]) for i in te}
        mk = lambda P: [{"well_id": i, "err": P[i] - T[i], "carry_err": float(wells[i]["tvt_input"][wells[i]["p"] - 1]) - T[i],
                         "gr_missing": float(np.mean(np.isnan(wells[i]["gr"][wells[i]["p"]:])))} for i in te]
        carry = {i: np.full(len(T[i]), float(wells[i]["tvt_input"][wells[i]["p"] - 1])) for i in te}
        rec = {"fold": k + 1, "wells": len(te), "c_scale": c_s, "c_seed": c_r, "gate_pick_B_frac": float(np.mean(list(pick.values())))}
        for name, P in [("carry", carry), ("A_blend", {i: A[i] for i in te}), ("B_margin", Bte), ("gated", gated_te),
                        ("gated+scale", {i: gated_te[i] + c_s * (Dw[i] - Dn[i]) for i in te}), ("final", fin_te)]:
            rows = mk(P); stages[name] += rows; rec[name] = suffix_metrics(rows)["pooled_rmse"]
        log.append(rec); append_jsonl(out / "stack_log.jsonl", rec); print(json.dumps(rec), flush=True)
    tot = {"role": "pooled_all_folds", **{n: suffix_metrics(r)["pooled_rmse"] for n, r in stages.items()},
           "folds2to5": {n: suffix_metrics([x for x in r if folds[x["well_id"]] != 0])["pooled_rmse"] for n, r in stages.items()}}
    append_jsonl(out / "stack_log.jsonl", tot); print(json.dumps(tot), flush=True)
    lt.save_rows(out, "final", stages["final"])


if __name__ == "__main__":
    main()
