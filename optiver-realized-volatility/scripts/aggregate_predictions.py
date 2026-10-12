"""Turn the per-row prediction files of the runs into the small aggregate CSVs in results/post/ (no competition rows are published).
Needs results/cpu/*_predictions.parquet and results/gpu/v*_s*_predictions.parquet (written by the runs, not shipped).
Usage: python scripts/aggregate_predictions.py --results <dir with cpu/ and gpu/> --out results"""
import argparse, glob, json, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--results", required=True); ap.add_argument("--out", default="results"); a = ap.parse_args()
os.makedirs(f"{a.out}/post", exist_ok=True)
rmspe = lambda y, p: float(np.sqrt(np.mean(((np.asarray(y, float) - np.asarray(p, float)) / np.asarray(y, float)) ** 2)))
def scalar(y, p): q = np.asarray(p, float) / np.asarray(y, float); return float(q.sum() / (q * q).sum())

P = {os.path.basename(f).replace("_predictions.parquet", ""): pd.read_parquet(f) for f in sorted(glob.glob(f"{a.results}/cpu/*_predictions.parquet") + glob.glob(f"{a.results}/gpu/*_predictions.parquet"))}
def wide(names, role):
    b = P[names[0]]; b = b[b.role == role][["stock_id", "time_id", "target"]].reset_index(drop=True)
    for n in names:
        d = P[n]; b = b.merge(d[d.role == role][["stock_id", "time_id", "pred"]].rename(columns={"pred": n}), on=["stock_id", "time_id"])
    return b
seeds = {v: sorted(k for k in P if k.startswith(v + "_s")) for v in ("v3", "v4")}
for v, ks in seeds.items():                                  # geometric mean over seeds, scalar refitted on the choose role
    ch, rp = wide(ks, 1), wide(ks, 2)
    gc, gr = np.exp(np.log(ch[ks].to_numpy()).mean(1)), np.exp(np.log(rp[ks].to_numpy()).mean(1)); c = scalar(ch.target, gc)
    P[f"{v}_ens"] = pd.concat([pd.DataFrame({"stock_id": ch.stock_id, "time_id": ch.time_id, "target": ch.target, "pred": c * gc, "role": 1}),
                               pd.DataFrame({"stock_id": rp.stock_id, "time_id": rp.time_id, "target": rp.target, "pred": c * gr, "role": 2})])
main = [m for m in ("v1_plain", "v1_est", "v2_nocross", "v2_cross", "v3_ens", "v4_ens") if m in P]
rows = []
for m in main + sum(seeds.values(), []):
    d1, d2 = P[m][P[m].role == 1], P[m][P[m].role == 2]
    rows.append({"model": m, "choose_rmspe": rmspe(d1.target, d1.pred), "report_rmspe": rmspe(d2.target, d2.pred), "n_choose": len(d1), "n_report": len(d2)})
names = ["v2_cross", "v3_ens", "v4_ens"]; ch, rp = wide(names, 1), wide(names, 2); best = (9, None); g = np.arange(0, 1.001, 0.05)
for x in g:
    for y in g:
        if x + y > 1 + 1e-9: continue
        w = np.array([x, y, 1 - x - y]); pc = np.exp((np.log(ch[names].to_numpy()) * w).sum(1)); c = scalar(ch.target, pc); s = rmspe(ch.target, c * pc)
        if s < best[0]: best = (s, (w, c))
w, c = best[1]; pr = c * np.exp((np.log(rp[names].to_numpy()) * w).sum(1)); pc = c * np.exp((np.log(ch[names].to_numpy()) * w).sum(1))
rows.append({"model": "blend", "choose_rmspe": best[0], "report_rmspe": rmspe(rp.target, pr), "n_choose": len(ch), "n_report": len(rp),
             "weights_v2_v3_v4": ";".join(f"{x:.2f}" for x in w), "scalar": c})
P["blend"] = pd.concat([pd.DataFrame({"stock_id": rp.stock_id, "time_id": rp.time_id, "target": rp.target, "pred": pr, "role": 2})])
pd.DataFrame(rows).to_csv(f"{a.out}/metrics.csv", index=False)

rep = pd.concat([P[m][P[m].role == 2].assign(model=m) for m in main if m in ("v1_est", "v2_nocross", "v2_cross", "v3_ens", "v4_ens")])
lvl = rep.groupby("stock_id").target.median(); dec = pd.qcut(lvl, 10, labels=False); rep["stock_decile"] = rep.stock_id.map(dec)
ref = P["v2_cross"]; ref = ref[ref.role == 2]; mk = np.log(ref.target).groupby(ref.time_id).mean(); rep["regime_decile"] = pd.qcut(rep.time_id.map(mk), 10, labels=False, duplicates="drop")
for col, fn in (("stock_decile", "by_stock_level"), ("regime_decile", "by_regime")):
    rep.groupby(["model", col]).apply(lambda x: pd.Series({"rmspe": rmspe(x.target, x.pred), "n": len(x)}), include_groups=False).reset_index().to_csv(f"{a.out}/post/{fn}.csv", index=False)
res, cal = [], []
for m, d in rep.groupby("model"):
    r = np.log(d.pred / d.target); h, e = np.histogram(r, bins=120, range=(-1, 1))
    res += [{"model": m, "bin_left": float(e[i]), "count": int(h[i]), "mean_log_ratio": float(r.mean())} for i in range(len(h))]
    q = pd.qcut(d.pred, 20, labels=False); gq = d.groupby(q).agg(mean_pred=("pred", "mean"), mean_target=("target", "mean"), n=("pred", "size")).reset_index().rename(columns={"pred": "bin"})
    gq.insert(0, "model", m); cal.append(gq)
pd.DataFrame(res).to_csv(f"{a.out}/post/residual_hist.csv", index=False); pd.concat(cal).to_csv(f"{a.out}/post/calibration.csv", index=False)
mem = sum(seeds.values(), []) + ["v1_est", "v2_cross"]; W = wide(mem, 2); L = np.log(W[mem].to_numpy())
dis, err = L.std(1), np.abs(L.mean(1) - np.log(W.target.to_numpy())); q = pd.qcut(dis, 20, labels=False)
pd.DataFrame({"std": dis, "abs_err": err}).groupby(q).mean().reset_index(names="bin").to_csv(f"{a.out}/post/disagreement.csv", index=False)
cc = np.corrcoef(L.T); pd.DataFrame(cc, index=mem, columns=mem).to_csv(f"{a.out}/post/member_correlation.csv")
print(pd.DataFrame(rows)[["model", "choose_rmspe", "report_rmspe"]].round(4).to_string(index=False))
