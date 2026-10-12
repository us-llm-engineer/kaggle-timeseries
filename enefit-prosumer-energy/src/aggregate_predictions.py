"""Aggregate saved report-window predictions into small CSVs (no competition rows are published).
Needs the per-row prediction files written by run_variant.py / run_dl_ens.py (not shipped).
python src/aggregate_predictions.py --v1 P --v3 P --v4 P --out results/post"""
import argparse, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
for k in ("v1", "v2", "v3", "v4"):
    ap.add_argument(f"--{k}", default=None)
ap.add_argument("--out", required=True); ap.add_argument("--start", default="2023-04-02")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
key = ["prediction_unit_id", "is_consumption", "datetime"]
m = None
for k in ("v1", "v2", "v3", "v4"):
    p = getattr(a, k)
    if not p: continue
    d = pd.read_parquet(p)[key + ["pred", "y"]].rename(columns={"pred": k})
    m = d if m is None else m.merge(d[key + [k]], on=key)
m = m[(m.datetime >= a.start)].dropna(subset=["y"]).copy()
ms = [k for k in ("v1", "v2", "v3", "v4") if k in m.columns]
m["blend"] = m[[k for k in ("v1", "v3", "v4") if k in ms]].mean(axis=1)
ms.append("blend")
m["hour"], m["week"] = m.datetime.dt.hour, m.datetime.dt.to_period("W").dt.start_time.dt.date
def table(g):
    rows = []
    for gv, d in m.groupby(g):
        r = dict(zip(g if isinstance(g, list) else [g], gv if isinstance(gv, tuple) else [gv]))
        r.update(rows=len(d), mean_abs_y=float(d.y.abs().mean()), sum_abs_y=float(d.y.abs().sum()))
        for k in ms:
            r[f"mae_{k}"] = float((d[k] - d.y).abs().mean()); r[f"bias_{k}"] = float((d[k] - d.y).mean())
        rows.append(r)
    return pd.DataFrame(rows)
table(["is_consumption", "hour"]).to_csv(f"{a.out}/by_hour.csv", index=False)
table(["is_consumption", "week"]).to_csv(f"{a.out}/by_week.csv", index=False)
table(["prediction_unit_id", "is_consumption"]).to_csv(f"{a.out}/by_unit.csv", index=False)
table(["is_consumption"]).to_csv(f"{a.out}/by_target.csv", index=False)
dis = m.assign(d13=(m.v1 - m.v3).abs(), d14=(m.v1 - m.v4).abs(), d34=(m.v3 - m.v4).abs()) if {"v1","v3","v4"} <= set(m.columns) else None
if dis is not None:
    dis.groupby(["is_consumption", "hour"])[["d13", "d14", "d34"]].mean().reset_index().to_csv(f"{a.out}/disagreement_by_hour.csv", index=False)
    rows = []
    for c, d in dis.groupby("is_consumption"):
        rows.append({"is_consumption": c, "corr_err_v1_v3": float(np.corrcoef(d.v1 - d.y, d.v3 - d.y)[0, 1]),
                     "corr_err_v1_v4": float(np.corrcoef(d.v1 - d.y, d.v4 - d.y)[0, 1]), "corr_err_v3_v4": float(np.corrcoef(d.v3 - d.y, d.v4 - d.y)[0, 1])})
    pd.DataFrame(rows).to_csv(f"{a.out}/error_correlation.csv", index=False)
print({k: round(float((m[k] - m.y).abs().mean()), 2) for k in ms})
