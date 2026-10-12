"""Run one variant on a back-test split and persist predictions, metrics, histories, checkpoints.

python3 src/run_variant.py --variant v1|v2|v3|v4 --root data_hf --out results/backtest [--smoke]
Roles: fit (< 2023-03-05), choose (2023-03-05..04-01, early stopping / epoch choice only), report (2023-04-02.., never used to select).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enefit13 import data as D  # noqa: E402


def metrics(df, ar_grid, s_idx_of, t0):
    """MAE in raw units for the rows of df (needs columns prediction_unit_id,is_consumption,datetime,y,pred) + lag baselines."""
    d = df.dropna(subset=["y", "pred"]).copy()
    out = {"rows": int(len(d)), "mae": float((d.pred - d.y).abs().mean())}
    for c in (0, 1):
        g = d[d.is_consumption == c]
        out[f"mae_{'prod' if c == 0 else 'cons'}"] = float((g.pred - g.y).abs().mean())
    s = np.array([s_idx_of[(int(a), int(b))] for a, b in zip(d.prediction_unit_id, d.is_consumption)])
    ti = ((d.datetime - t0) / pd.Timedelta(hours=1)).astype(int).values
    for k in (48, 168):
        lag = ar_grid[s, ti - k]
        m = ~np.isnan(lag)
        out[f"baseline_lag{k}_mae"] = float(np.abs(lag[m] - d.y.values[m]).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["v1", "v2", "v3", "v4", "v1m", "v2m"])
    ap.add_argument("--root", default="data_hf")
    ap.add_argument("--out", default="results/backtest")
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--deadline-min", type=float, default=0.0, help="stop starting new work after this many minutes (0 = off)")
    a = ap.parse_args()
    t_start = time.time()
    deadline = t_start + a.deadline_min * 60 if a.deadline_min else None
    out = os.path.join(a.out, a.variant)
    os.makedirs(out, exist_ok=True)
    prog = os.path.join(out, "progress.json")

    def mark(stage, **kw):
        tmp = prog + ".tmp"
        json.dump({"stage": stage, "t": round(time.time() - t_start, 1), **kw}, open(tmp, "w"))
        os.replace(tmp, prog)

    store = D.Store(a.root)
    mark("load")
    panel = D.load_panel(store)
    if a.smoke:
        keep = sorted(panel.prediction_unit_id.unique())[:6]
        panel = panel[panel.prediction_unit_id.isin(keep)].reset_index(drop=True)
        a.rounds, a.epochs, a.seeds = 40, 2, 1
    an, ar, s, ti, sid, t0 = D.series_grid(panel)
    panel = D.add_lag_features(panel, an, s, ti)
    masks = D.roles(panel)
    mark("features", rows=int(len(panel)))
    res = {"variant": a.variant, "smoke": a.smoke}

    if a.variant in ("v1", "v2", "v1m", "v2m"):
        from enefit13 import gbdt
        pred_df, notes = gbdt.run_variant(panel, masks, a.variant, out, rounds=a.rounds)
        res["notes"] = notes
        for role in ("choose", "report"):
            res[role] = metrics(pred_df[masks[role]], ar, sid, t0)
        pred_df[masks["choose"] | masks["report"]].to_parquet(os.path.join(out, "predictions.parquet"), index=False)
    else:
        import torch
        from enefit13 import nn as N
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        td = N.build_tensors(panel, an, ar, s, ti, sid, t0)
        sidx, didx = N.day_samples(td)
        day_start = pd.Timestamp(t0) + pd.to_timedelta(didx * 24, unit="h")
        tr_m = day_start < D.CHOOSE_START
        ch_m = (day_start >= D.CHOOSE_START) & (day_start < D.REPORT_START)
        rp_m = day_start >= D.REPORT_START
        sub = lambda m: (sidx[m], didx[m])
        mode = "rel" if a.variant == "v3" else "raw"
        preds = np.zeros((len(sidx), N.H), dtype="float32")
        hist_all = []
        for seed in range(a.seeds):
            if deadline and time.time() > deadline:
                break
            mark("train", seed=seed)
            r = N.Runner(td, mode, dev, seed, out, f"{a.variant}_s{seed}")
            hist_all.append(r.train(sub(tr_m), sub(ch_m), sub(rp_m), epochs=a.epochs, deadline=deadline))
            ev = ch_m | rp_m
            preds[ev] += r.evaluate(*sub(ev))
        n = max(len(hist_all), 1)
        preds /= n
        keys = sorted(sid, key=lambda k: sid[k])
        inv = {v: k for k, v in sid.items()}
        rows = []
        ev = np.where(ch_m | rp_m)[0]
        for i in ev:
            uid, ic = inv[int(sidx[i])]
            base = int(didx[i]) * 24
            ts = pd.Timestamp(t0) + pd.to_timedelta(np.arange(base, base + N.H), unit="h")
            rows.append(pd.DataFrame({"prediction_unit_id": uid, "is_consumption": ic, "datetime": ts,
                                      "pred": preds[i], "y": ar[int(sidx[i]), base : base + N.H]}))
        pred_df = pd.concat(rows, ignore_index=True)
        pred_df["datetime"] = pd.to_datetime(pred_df.datetime)
        for role, lo, hi in (("choose", D.CHOOSE_START, D.REPORT_START), ("report", D.REPORT_START, pd.Timestamp("2100-01-01"))):
            res[role] = metrics(pred_df[(pred_df.datetime >= lo) & (pred_df.datetime < hi)], ar, sid, t0)
        res["seeds_used"] = n
        pred_df.to_parquet(os.path.join(out, "predictions.parquet"), index=False)
    res["wall_sec"] = round(time.time() - t_start, 1)
    json.dump(res, open(os.path.join(out, "metrics.json"), "w"), indent=1)
    mark("done", wall=res["wall_sec"])
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
