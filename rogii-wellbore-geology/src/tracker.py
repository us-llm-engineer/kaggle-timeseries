"""Grid tracker baseline: shared inference core + emission variants V1/V2 (no learning).

State (d, v): d = T - T_PS on a 0.25 ft grid (+-160 ft), v = slope per block in grid cells (-K..K).
Prior (T-space, fitted by least squares = Gaussian MLE on fit wells only):
    v_k = alpha * v_{k-1} + gamma * d_{k-1} + eps,  eps ~ N(0, sigma^2);   d_k = d_{k-1} + v_k.
Exact forward-backward with per-step normalisation; estimate = posterior mean (MSE-optimal for RMSE).
Emission: heavy-tailed kernel log e = -log(1 + |r| / s) (rail-localisation paper, Eq. 4), r = GR_blk - ref(d).
Only the well's prefix (rows < p), its geometry and its typewell are read at prediction time.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from wells import CHOOSE_FOLD, append_jsonl, eligible_ids, fold_of, load_wells, suffix_metrics

DT = 0.25
HALF = 160.0
D = np.arange(-HALF, HALF + DT / 2, DT)
S = len(D)


# ---------------------------------------------------------------- prior
def block_truth(w: dict, B: int) -> tuple[np.ndarray, float]:
    p, n = w["p"], w["n"]
    t = w["tvt"][p:]
    nb = int(np.ceil((n - p) / B))
    tb = np.array([t[i * B:(i + 1) * B].mean() for i in range(nb)])
    return tb, float(w["tvt_input"][p - 1])


def prefix_slope(w: dict, B: int) -> float:
    p = w["p"]
    lo = max(0, p - 1 - B)
    return float((w["tvt_input"][p - 1] - w["tvt_input"][lo]) / max(p - 1 - lo, 1) * B / DT)


def fit_prior(wells: list[dict], B: int) -> dict:
    X, Y, x0, y0, allv = [], [], [], [], []
    for w in wells:
        tb, tps = block_truth(w, B)
        d = np.r_[0.0, tb - tps]
        v = np.diff(d) / DT
        allv.append(v)
        X.append(np.c_[v[:-1], d[1:-1]]); Y.append(v[1:])
        x0.append(prefix_slope(w, B)); y0.append(v[0])
    X, Y = np.vstack(X), np.concatenate(Y)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    sigma = float(np.std(Y - X @ coef))
    x0, y0 = np.array(x0), np.array(y0)
    a0 = float(np.dot(x0, y0) / max(np.dot(x0, x0), 1e-9))
    s0 = float(np.std(y0 - a0 * x0))
    K = int(np.clip(np.ceil(np.quantile(np.abs(np.concatenate(allv)), 0.999)), 3, 16))
    return {"B": B, "alpha": float(coef[0]), "gamma": float(coef[1]), "sigma": max(sigma, 0.05), "a0": a0, "s0": max(s0, 0.3), "K": K}


def transition(prior: dict) -> np.ndarray:
    """P[d, v, v'] over the grid (shared by all wells: the grid is relative to T_PS)."""
    K = prior["K"]
    vs = np.arange(-K, K + 1, dtype=np.float64)
    mean = prior["alpha"] * vs[None, :, None] + prior["gamma"] * D[:, None, None]
    logp = -0.5 * ((vs[None, None, :] - mean) / prior["sigma"]) ** 2
    P = np.exp(logp - logp.max(axis=2, keepdims=True))
    return P / P.sum(axis=2, keepdims=True)


def init_dist(w: dict, prior: dict) -> np.ndarray:
    K = prior["K"]
    vs = np.arange(-K, K + 1)
    m = prior["a0"] * prefix_slope(w, prior["B"])
    pi = np.exp(-0.5 * ((vs - m) / prior["s0"]) ** 2)
    return pi / pi.sum()


# ---------------------------------------------------------------- forward-backward
def _shift_fwd(M: np.ndarray, K: int) -> np.ndarray:
    """out[d + s_j, j] = M[d, j] with s_j = j - K, zero fill."""
    out = np.zeros_like(M)
    for j in range(M.shape[1]):
        s = j - K
        if s >= 0:
            out[s:, j] = M[:S - s, j]
        else:
            out[:s, j] = M[-s:, j]
    return out


def _shift_bwd(X: np.ndarray, K: int) -> np.ndarray:
    """out[d, j] = X[d + s_j, j], zero outside."""
    out = np.zeros_like(X)
    for j in range(X.shape[1]):
        s = j - K
        if s >= 0:
            out[:S - s, j] = X[s:, j]
        else:
            out[-s:, j] = X[:s, j]
    return out


def forward_backward(logE: np.ndarray, P: np.ndarray, pi0: np.ndarray, K: int) -> dict:
    """logE: (L, S) log-emission per block. Returns posterior mean/std of d per block and health stats."""
    L = logE.shape[0]
    E = np.exp(logE - logE.max(axis=1, keepdims=True))
    a = np.zeros((S, 2 * K + 1)); a[S // 2] = pi0
    alphas = np.empty((L, S, 2 * K + 1)); c = np.empty(L); lost = np.empty(L)
    for k in range(L):
        m = _shift_fwd(np.einsum("dv,dvw->dw", a, P), K)
        lost[k] = 1.0 - m.sum() / a.sum()
        m *= E[k][:, None]
        c[k] = m.sum()
        if not np.isfinite(c[k]) or c[k] <= 0:  # emission killed every state: fall back to the prior step
            m = _shift_fwd(np.einsum("dv,dvw->dw", a, P), K); c[k] = m.sum()
        a = m / c[k]
        alphas[k] = a
    b = np.ones((S, 2 * K + 1))
    mean = np.empty(L); std = np.empty(L); ent = np.empty(L)
    for k in range(L - 1, -1, -1):
        g = alphas[k] * b
        pd_ = g.sum(axis=1); pd_ /= pd_.sum()
        mean[k] = pd_ @ D
        std[k] = np.sqrt(max(pd_ @ (D - mean[k]) ** 2, 0.0))
        nz = pd_[pd_ > 0]; ent[k] = -np.sum(nz * np.log(nz))
        if k:
            X = E[k][:, None] * b / c[k]
            b = np.einsum("dvj,dj->dv", P, _shift_bwd(X, K))
            b /= b.max()
    return {"mean": mean, "std": std, "entropy": ent, "edge_lost": lost}


# ---------------------------------------------------------------- emissions
def calibrate(w: dict) -> tuple[float, float, float]:
    """GR_h ~ a + b * GR_typewell(TVT) on the prefix (known TVT); robust scale of the residual."""
    p = w["p"]
    g, t = w["gr"][:p], w["tvt_input"][:p]
    m = np.isfinite(g) & np.isfinite(t) & (t >= w["tw_tvt"][0]) & (t <= w["tw_tvt"][-1])
    if m.sum() < 50:
        gs = w["gr"][np.isfinite(w["gr"])]
        return 0.0, 1.0, float(np.std(gs)) if len(gs) else 30.0
    ref = np.interp(t[m], w["tw_tvt"], w["tw_gr"])
    A = np.c_[np.ones(m.sum()), ref]
    coef, *_ = np.linalg.lstsq(A, g[m], rcond=None)
    for _ in range(2):  # Huber-style reweighting against bad prefix stretches
        r = g[m] - A @ coef
        sc = 1.4826 * np.median(np.abs(r)) + 1e-6
        wt = np.minimum(1.0, 1.5 * sc / np.maximum(np.abs(r), 1e-9))
        coef, *_ = np.linalg.lstsq(A * wt[:, None], g[m] * wt, rcond=None)
    r = g[m] - A @ coef
    return float(coef[0]), float(coef[1]), float(max(1.4826 * np.median(np.abs(r)), 1.0))


def blocks(w: dict, B: int) -> dict:
    p, n = w["p"], w["n"]
    g = w["gr"][p:]
    nb = int(np.ceil((n - p) / B))
    gb = np.full(nb, np.nan); miss = np.ones(nb)
    for i in range(nb):
        seg = g[i * B:(i + 1) * B]
        f = np.isfinite(seg)
        miss[i] = 1 - f.mean()
        if f.any():
            gb[i] = seg[f].mean()
    centers = p + np.arange(nb) * B + (np.minimum(B, n - p - np.arange(nb) * B) - 1) / 2
    return {"gr": gb, "miss": miss, "centers": centers}


def reference(w: dict, cal: tuple[float, float, float], splice: bool) -> np.ndarray:
    """Reference GR on the grid T_PS + D, in horizontal-GR units."""
    a, b, _ = cal
    tps = float(w["tvt_input"][w["p"] - 1])
    T = tps + D
    ref = a + b * np.interp(T, w["tw_tvt"], w["tw_gr"], left=np.nan, right=np.nan)
    if splice:
        p = w["p"]
        g, t = w["gr"][:p], w["tvt_input"][:p]
        m = np.isfinite(g) & np.isfinite(t)
        idx = np.round((t[m] - tps) / DT).astype(int) + S // 2
        ok = (idx >= 0) & (idx < S)
        cnt = np.bincount(idx[ok], minlength=S).astype(float)
        sm = np.bincount(idx[ok], weights=g[m][ok], minlength=S)
        ker = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
        cs, ss = np.convolve(cnt, ker, "same"), np.convolve(sm, ker, "same")
        cover = cs >= 2.0
        ref = np.where(cover, ss / np.maximum(cs, 1e-9), ref)
    return ref


def _nanmovmean(x: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return x
    f = np.isfinite(x).astype(float); v = np.where(np.isfinite(x), x, 0.0)
    ker = np.ones(k)
    num, den = np.convolve(v, ker, "same"), np.convolve(f, ker, "same")
    return np.where(den >= k / 2, num / np.maximum(den, 1e-9), np.nan)


def kernel_emission(gb: np.ndarray, ref: np.ndarray, s: float, scales=(1,)) -> np.ndarray:
    """(L, S) heavy-tailed log-emission averaged over window scales; missing block -> 0 (flat)."""
    L = len(gb)
    out = np.zeros((L, S))
    for k in scales:
        g = _nanmovmean(gb, k)
        r = _nanmovmean(ref, k) if k > 1 else ref
        r = np.where(np.isfinite(r), r, np.nanmean(ref) + 10 * s)  # outside the typewell: strongly unlikely
        e = -np.log1p(np.abs(g[:, None] - r[None, :]) / s)
        out += np.where(np.isfinite(g)[:, None], e, 0.0)
    return out / len(scales)


def variant_emission(w: dict, variant: str, B: int, mult: float) -> tuple[np.ndarray, dict]:
    cal = calibrate(w)
    bl = blocks(w, B)
    if variant == "v1":
        ref, scales = reference(w, cal, splice=False), (1,)
    elif variant == "v2":
        ref, scales = reference(w, cal, splice=True), (1, 3, 9)
    else:
        raise ValueError(variant)
    return kernel_emission(bl["gr"], ref, cal[2] * mult, scales), bl


# ---------------------------------------------------------------- per-well prediction
def expand(w: dict, centers: np.ndarray, d_mean: np.ndarray, d_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p, n = w["p"], w["n"]
    tps = float(w["tvt_input"][p - 1])
    rows = np.arange(p, n)
    x = np.r_[p - 1, centers]
    pred = tps + np.interp(rows, x, np.r_[0.0, d_mean])
    std = np.interp(rows, x, np.r_[0.0, d_std])
    return pred, std


def predict_well(w: dict, logE: np.ndarray, bl: dict, prior: dict, P: np.ndarray) -> dict:
    fb = forward_backward(logE, P, init_dist(w, prior), prior["K"])
    pred, std = expand(w, bl["centers"], fb["mean"], fb["std"])
    return {"pred": pred, "std": std, "fb": fb}


def score_row(w: dict, pred: np.ndarray, std: np.ndarray | None = None) -> dict:
    p = w["p"]
    truth = w["tvt"][p:]
    row = {"well_id": w["well_id"], "err": pred - truth, "carry_err": w["tvt_input"][p - 1] - truth,
           "gr_missing": float(np.mean(~np.isfinite(w["gr"][p:])))}
    if std is not None:
        row["std"] = std
    return row


# ---------------------------------------------------------------- multiprocessing driver
_G: dict = {}


def _init(wells, prior, P, variant, B, mult):
    _G.update(wells=wells, prior=prior, P=P, variant=variant, B=B, mult=mult)


def _job(wid: str) -> dict:
    w = _G["wells"][wid]
    logE, bl = variant_emission(w, _G["variant"], _G["B"], _G["mult"])
    out = predict_well(w, logE, bl, _G["prior"], _G["P"])
    row = score_row(w, out["pred"], out["std"])
    fb = out["fb"]
    row.update(mean_entropy=float(fb["entropy"].mean()), edge_lost=float(fb["edge_lost"].sum()),
               block_mean=fb["mean"].astype(np.float32), block_std=fb["std"].astype(np.float32))
    return row


def run_eval(wells: dict, fit_ids: list[str], eval_ids: list[str], variant: str, B: int, mult: float, workers: int) -> tuple[dict, list[dict], dict]:
    prior = fit_prior([wells[i] for i in fit_ids], B)
    P = transition(prior)
    with Pool(workers, initializer=_init, initargs=(wells, prior, P, variant, B, mult)) as pool:
        rows = pool.map(_job, eval_ids, chunksize=1)
    m = suffix_metrics(rows)
    m["mean_entropy"] = float(np.mean([r["mean_entropy"] for r in rows]))
    m["edge_lost_max"] = float(np.max([r["edge_lost"] for r in rows]))
    return m, rows, prior


def save_rows(out: Path, tag: str, rows: list[dict]) -> None:
    import pandas as pd
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"well_id": r["well_id"], "points": len(r["err"]), "rmse": float(np.sqrt(np.mean(r["err"] ** 2))),
                   "carry_rmse": float(np.sqrt(np.mean(r["carry_err"] ** 2))), "gr_missing": r["gr_missing"],
                   "mean_entropy": r.get("mean_entropy"), "edge_lost": r.get("edge_lost")} for r in rows]).to_csv(out / f"{tag}_per_well.csv", index=False)
    np.savez_compressed(out / f"{tag}_rows.npz", **{r["well_id"]: np.c_[r["err"], r["carry_err"], r.get("std", np.zeros_like(r["err"]))].astype(np.float32) for r in rows})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="data/stage")
    ap.add_argument("--variant", choices=["v1", "v2"], required=True)
    ap.add_argument("--mode", choices=["choose", "report", "smoke"], default="choose")
    ap.add_argument("--B", type=int, nargs="*", default=[5, 10, 20])
    ap.add_argument("--mult", type=float, nargs="*", default=[0.5, 1.0, 2.0])
    ap.add_argument("--workers", type=int, default=max(1, int(0.8 * (os.cpu_count() or 2))))
    ap.add_argument("--out", default="runs/tracker")
    args = ap.parse_args()
    out = Path(args.out) / args.variant
    t0 = time.time()
    wells = load_wells(args.stage)
    ids = eligible_ids(wells); folds = fold_of(ids)
    print(json.dumps({"event": "loaded", "wells": len(ids), "seconds": round(time.time() - t0, 1), "workers": args.workers}), flush=True)
    if args.mode in ("choose", "smoke"):
        fit = [i for i in ids if folds[i] != CHOOSE_FOLD]
        dev = [i for i in ids if folds[i] == CHOOSE_FOLD]
        if args.mode == "smoke":
            dev = dev[:6]
        best = None
        for B in args.B:
            for mult in args.mult:
                t1 = time.time()
                m, rows, prior = run_eval(wells, fit, dev, args.variant, B, mult, args.workers)
                rec = {"variant": args.variant, "role": "choose", "B": B, "mult": mult, "prior": prior, **m,
                       "seconds": round(time.time() - t1, 1), "wells_per_s": round(len(dev) / (time.time() - t1), 2)}
                append_jsonl(out / f"{args.mode}_log.jsonl", rec); print(json.dumps(rec), flush=True)
                if best is None or m["pooled_rmse"] < best[0]:
                    best = (m["pooled_rmse"], B, mult, rows)
        save_rows(out, f"{args.mode}_best_B{best[1]}_m{best[2]}", best[3])
        (out / f"{args.mode}_selected.json").write_text(json.dumps({"B": best[1], "mult": best[2], "pooled_rmse": best[0]}) + "\n")
    else:
        sel = json.loads((out / "choose_selected.json").read_text())
        allrows = []
        for k in range(5):
            if k == CHOOSE_FOLD:
                continue
            fit = [i for i in ids if folds[i] != k]; ev = [i for i in ids if folds[i] == k]
            m, rows, prior = run_eval(wells, fit, ev, args.variant, sel["B"], sel["mult"], args.workers)
            rec = {"variant": args.variant, "role": "report", "fold": k + 1, **sel, "prior": prior, **m}
            append_jsonl(out / "report_log.jsonl", rec); print(json.dumps(rec), flush=True)
            allrows += rows
        tot = {"variant": args.variant, "role": "report_oof_folds2to5", **sel, **suffix_metrics(allrows)}
        append_jsonl(out / "report_log.jsonl", tot); print(json.dumps(tot), flush=True)
        save_rows(out, "report_oof", allrows)


if __name__ == "__main__":
    main()
