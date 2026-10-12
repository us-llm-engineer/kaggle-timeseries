"""Coarsened (power-tempered) tracker baseline (Miller and Dunson, 2015).

Coarsened posterior (Miller & Dunson, 1506.06101, Sec. 3.1 / 6.1): the exact c-posterior is approximated by raising the
likelihood to zeta_n = alpha / (alpha + n). Here n = number of informative (GR-present) blocks of the well's suffix and
alpha = 2 sigma^2 / delta^2, where sigma is the block-level residual scale on the prefix and delta the typical calibration
drift on the suffix, both measured on FIT wells with their true T. Nothing is searched on the choose fold.
Core inference (grid, prior, forward-backward) is the V1 tracker in tracker.py.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import tracker as lt
from wells import CHOOSE_FOLD, append_jsonl, eligible_ids, fold_of, load_wells, suffix_metrics

B = 10  # fixed from V1 (not re-searched)


def misspecification(w: dict) -> tuple[float, float] | None:
    """(sigma_prefix_block, delta_suffix_drift) for one fit well, using its true T."""
    a, b, s = lt.calibrate(w)
    p, n = w["p"], w["n"]
    def block_resid(lo, hi, tvt):
        out = []
        for i in range(lo, hi, B):
            g = w["gr"][i:i + B]; t = tvt[i:i + B]
            f = np.isfinite(g) & np.isfinite(t)
            if f.sum() >= B // 2:
                out.append(g[f].mean() - (a + b * np.interp(t[f], w["tw_tvt"], w["tw_gr"])).mean())
        return np.asarray(out)
    rp = block_resid(0, p, w["tvt_input"])
    rs = block_resid(p, n, w["tvt"])
    if len(rp) < 10 or len(rs) < 10:
        return None
    sigma = 1.4826 * np.median(np.abs(rp - np.median(rp)))
    # systematic drift (mean shift) of the calibrated reference at the TRUE T over 20-block windows, not the noise
    wins = [rs[i:i + 20].mean() for i in range(0, len(rs) - 19, 20)]
    delta = np.median(np.abs(wins)) if wins else np.abs(rs.mean())
    return float(max(sigma, 1e-3)), float(max(delta, 1e-3))


def fit_alpha(wells: dict, fit: list[str]) -> dict:
    sd = [misspecification(wells[i]) for i in fit]
    sd = np.array([x for x in sd if x is not None])
    alpha_w = 2 * sd[:, 0] ** 2 / sd[:, 1] ** 2
    return {"alpha": float(np.median(alpha_w)), "alpha_q25": float(np.quantile(alpha_w, .25)), "alpha_q75": float(np.quantile(alpha_w, .75)),
            "sigma_med": float(np.median(sd[:, 0])), "delta_med": float(np.median(sd[:, 1])), "wells": int(len(sd))}


_G: dict = {}


def _init(wells, prior, P, alpha):
    _G.update(wells=wells, prior=prior, P=P, alpha=alpha)


def _job(wid):
    w = _G["wells"][wid]
    logE, bl = lt.variant_emission(w, "v1", B, 1.0)
    n_inf = int(np.isfinite(bl["gr"]).sum())
    zeta = _G["alpha"] / (_G["alpha"] + max(n_inf, 1))
    out = lt.predict_well(w, logE * zeta, bl, _G["prior"], _G["P"])
    row = lt.score_row(w, out["pred"], out["std"])
    row.update(zeta=zeta, mean_entropy=float(out["fb"]["entropy"].mean()), edge_lost=float(out["fb"]["edge_lost"].sum()))
    return row


def run(wells, fit, ev, workers):
    prior = lt.fit_prior([wells[i] for i in fit], B)
    P = lt.transition(prior)
    al = fit_alpha(wells, fit)
    with Pool(workers, initializer=_init, initargs=(wells, prior, P, al["alpha"])) as pool:
        rows = pool.map(_job, ev, chunksize=1)
    m = suffix_metrics(rows)
    z = np.array([r["zeta"] for r in rows])
    m.update(zeta_median=float(np.median(z)), zeta_q10=float(np.quantile(z, .1)), zeta_q90=float(np.quantile(z, .9)),
             mean_entropy=float(np.mean([r["mean_entropy"] for r in rows])), edge_lost_max=float(np.max([r["edge_lost"] for r in rows])))
    return m, rows, prior, al


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="data/stage")
    ap.add_argument("--mode", choices=["choose", "report", "smoke"], default="choose")
    ap.add_argument("--workers", type=int, default=max(1, int(0.8 * (os.cpu_count() or 2))))
    ap.add_argument("--out", default="runs/coarsened")
    args = ap.parse_args()
    out = Path(args.out) / "w1"
    wells = load_wells(args.stage); ids = eligible_ids(wells); folds = fold_of(ids)
    plan = [(CHOOSE_FOLD, "choose")] if args.mode in ("choose", "smoke") else [(k, "report") for k in range(5) if k != CHOOSE_FOLD]
    allrows = []
    for k, role in plan:
        fit = [i for i in ids if folds[i] != k]; ev = [i for i in ids if folds[i] == k]
        if args.mode == "smoke":
            ev = ev[:6]
        t0 = time.time()
        m, rows, prior, al = run(wells, fit, ev, args.workers)
        rec = {"variant": "w1", "role": role, "fold": k + 1, "B": B, "alpha_fit": al, "prior": prior, **m, "seconds": round(time.time() - t0, 1)}
        append_jsonl(out / f"{args.mode}_log.jsonl", rec); print(json.dumps(rec), flush=True)
        allrows += rows
    if args.mode == "report":
        tot = {"variant": "w1", "role": "report_oof_folds2to5", **suffix_metrics(allrows)}
        append_jsonl(out / "report_log.jsonl", tot); print(json.dumps(tot), flush=True)
    lt.save_rows(out, args.mode, allrows)


if __name__ == "__main__":
    main()
