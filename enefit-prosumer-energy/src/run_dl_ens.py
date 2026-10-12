"""Train the DL variants as batched ensembles on one device and persist predictions + metrics.
python3 src/run_dl_ens.py --prep DIR --out DIR --variants v3,v4 --drops 0.5,0.2 --seeds 4 --epochs 100 --budget-min 20"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enefit13 import data as D, nn as N, nn_ens as NE

ap = argparse.ArgumentParser()
ap.add_argument("--prep", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--variants", default="v3,v4"); ap.add_argument("--drops", default="0.5,0.2"); ap.add_argument("--seeds", type=int, default=4)
ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--budget-min", type=float, default=20.0)
ap.add_argument("--gpu-frac", type=float, default=0.8); ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
T0 = time.time()
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cuda":
    torch.cuda.set_per_process_memory_fraction(a.gpu_frac)

def mark(stage, **kw):
    tmp = os.path.join(a.out, "progress.json.tmp"); json.dump({"stage": stage, "t": round(time.time() - T0, 1), **kw}, open(tmp, "w"))
    os.replace(tmp, os.path.join(a.out, "progress.json"))

mark("load")
z = np.load(os.path.join(a.prep, "td.npz")); meta = json.load(open(os.path.join(a.prep, "td_meta.json")))
td = {k: z[k] for k in z.files}; td.update(S=meta["S"], T=meta["T"], t0=pd.Timestamp(meta["t0"]))
sid_map = {tuple(int(x) for x in k.split(",")): v for k, v in meta["sid"].items()}
inv = {v: k for k, v in sid_map.items()}
sidx, didx = N.day_samples(td)
ds = pd.Timestamp(td["t0"]) + pd.to_timedelta(didx * 24, unit="h")
tr_m, ch_m, rp_m = ds < D.CHOOSE_START, (ds >= D.CHOOSE_START) & (ds < D.REPORT_START), ds >= D.REPORT_START
if a.smoke:
    keep = np.isin(sidx, np.arange(12)); sidx, didx, ds = sidx[keep], didx[keep], ds[keep]
    tr_m, ch_m, rp_m = tr_m[keep], ch_m[keep], rp_m[keep]; a.epochs = 6
sub = lambda m: (sidx[m], didx[m])
drops = [float(x) for x in a.drops.split(",")]; seeds = list(range(a.seeds))
variants = a.variants.split(",")
per_var_budget = a.budget_min * 60 * 0.9 / len(variants)
res = {"device": dev, "variants": {}}
ar = td["ar"]

def metrics(df):
    d = df.dropna(subset=["y", "pred"])
    out = {"rows": int(len(d)), "mae": float((d.pred - d.y).abs().mean())}
    for c in (0, 1):
        g = d[d.is_consumption == c]; out["mae_" + ("prod" if c == 0 else "cons")] = float((g.pred - g.y).abs().mean())
    s = np.array([sid_map[(int(u), int(c))] for u, c in zip(d.prediction_unit_id, d.is_consumption)])
    ti = ((d.datetime - td["t0"]) / pd.Timedelta(hours=1)).astype(int).values
    for k in (48, 168):
        lag = ar[s, ti - k]; m = ~np.isnan(lag); out[f"baseline_lag{k}_mae"] = float(np.abs(lag[m] - d.y.values[m]).mean())
    return out

def frame(pred_day):
    ev = np.concatenate([np.where(ch_m)[0], np.where(rp_m)[0]]); rows = []   # same order as train_ensemble's [choose; report]
    for j, i in enumerate(ev):
        uid, ic = inv[int(sidx[i])]; base = int(didx[i]) * 24
        ts = pd.Timestamp(td["t0"]) + pd.to_timedelta(np.arange(base, base + N.H), unit="h")
        rows.append(pd.DataFrame({"prediction_unit_id": uid, "is_consumption": ic, "datetime": ts, "pred": pred_day[j], "y": ar[int(sidx[i]), base: base + N.H]}))
    df = pd.concat(rows, ignore_index=True)
    return df

for v in variants:
    mark("train", variant=v)
    mode = "rel" if v == "v3" else "raw"
    odir = os.path.join(a.out, v)
    preds, info, _model = NE.train_ensemble(td, mode, dev, drops, seeds, sub(tr_m), sub(ch_m), sub(rp_m), odir, v, epochs=a.epochs, budget_sec=per_var_budget)
    np.save(os.path.join(odir, "preds_eval.npy"), preds)
    members = info["members"]
    out = {"info": info, "groups": {}}
    n_ch = int(ch_m.sum())
    for label, idxs in [(f"drop{d}", [i for i, m in enumerate(members) if m["drop"] == d]) for d in drops] + [("all", list(range(len(members))))]:
        p = preds[idxs].mean(0)
        df = frame(p)
        for role, lo, hi in (("choose", D.CHOOSE_START, D.REPORT_START), ("report", D.REPORT_START, pd.Timestamp("2100-01-01"))):
            out["groups"].setdefault(label, {})[role] = metrics(df[(df.datetime >= lo) & (df.datetime < hi)])
        if label == "all":
            df.to_parquet(os.path.join(odir, "predictions_all.parquet"), index=False)
    res["variants"][v] = out
    json.dump(res, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
res["wall_sec"] = round(time.time() - T0, 1)
json.dump(res, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
mark("done", wall=res["wall_sec"])
print(json.dumps({v: {g: r["report"]["mae"] for g, r in o["groups"].items()} for v, o in res["variants"].items()}))
