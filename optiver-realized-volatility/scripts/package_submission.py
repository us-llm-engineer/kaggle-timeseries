"""Assemble the Kaggle weights dataset for the best model (blend of V2 + V3 ensemble + V4 ensemble).
Local run: python3 src/package_submission.py   (inputs in $OPT_SCRATCH: features.parquet, train.csv, ckpt/*.pt, v2/v2_cross_predictions.parquet)
Everything the Kaggle notebook needs is recomputed here from the fit/choose roles only; the report role is scored once at the end."""
import json, os, shutil, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opt14 import common as C, data as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCR = os.environ.get("OPT_SCRATCH", "scratch"); DST = f"{ROOT}/kaggle_submit/dataset"
os.makedirs(f"{DST}/src/opt14", exist_ok=True); os.makedirs(f"{SCR}/v2", exist_ok=True)

df = D.load_table(f"{SCR}/features.parquet", f"{SCR}/train.csv")
cols = D.feature_columns(df, True); fc = [c for c in cols if c != "stock_id"]
prep = D.Prep().fit(df.loc[df.role == 0, fc].to_numpy())
np.savez(f"{DST}/prep.npz", scale=prep.scale, mu=prep.mu, sd=prep.sd)
stock_mean = df.groupby("stock_id")["stock_logrv_mean"].first().to_dict()

# ---- V2: the booster is the one saved by the CPU run (results/cpu); verify that it reproduces the scored predictions.
import lightgbm as lgb
meta = json.load(open(f"{ROOT}/results/cpu/v2_cross_history.json"))["meta"]
res = {"scalar": meta["scalar"], "choose_rmspe": meta["choose_rmspe"], "report_rmspe": meta["report_rmspe"]}
shutil.copy(f"{ROOT}/results/cpu/v2_cross_best.txt", f"{DST}/v2_cross_best.txt")
p2 = pd.read_parquet(f"{SCR}/v2/v2_cross_predictions.parquet")
bst = lgb.Booster(model_file=f"{DST}/v2_cross_best.txt")
chk = df[df.role == 1].merge(p2[p2.role == 1][["stock_id", "time_id", "pred"]], on=["stock_id", "time_id"])
re = res["scalar"] * np.exp(bst.predict(chk[cols].to_numpy(np.float32)))
assert np.allclose(re, chk.pred.to_numpy(), rtol=1e-4), "booster does not reproduce the saved V2 predictions"
print("V2 booster reproduces the scored choose predictions; max rel diff", float(np.max(np.abs(re / chk.pred.to_numpy() - 1))))

# ---- NN seed ensembles (geometric mean, scalar fitted on the choose role) from the saved predictions of the L4 run
def seeds_preds(v):
    out = None
    for s in range(3):
        d = pd.read_parquet(f"{ROOT}/results/gpu/{v}_s{s}_predictions.parquet").rename(columns={"pred": f"p{s}"})
        out = d if out is None else out.merge(d[["stock_id", "time_id", f"p{s}"]], on=["stock_id", "time_id"])
    return out
ens, scal = {}, {}
for v in ("v3", "v4"):
    d = seeds_preds(v); g = np.exp(np.log(d[[f"p{s}" for s in range(3)]].to_numpy()).mean(1))
    ch = d.role.to_numpy() == 1; scal[v] = C.best_scalar(d.target.to_numpy()[ch], g[ch])
    ens[v] = d[["stock_id", "time_id", "target", "role"]].assign(pred=scal[v] * g)

# ---- blend weights on the choose role (grid over simplex, step 0.05), scored once on report
names = ["v2_cross", "v3", "v4"]
base = p2.rename(columns={"pred": "v2_cross"})[["stock_id", "time_id", "target", "role", "v2_cross"]]
for v in ("v3", "v4"): base = base.merge(ens[v].rename(columns={"pred": v})[["stock_id", "time_id", v]], on=["stock_id", "time_id"])
ch, rp = base[base.role == 1], base[base.role == 2]
best = (9.0, None); grid = np.arange(0, 1.001, 0.05)
for a in grid:
    for b in grid:
        if a + b > 1 + 1e-9: continue
        w = np.array([a, b, 1 - a - b]); pc = np.exp((np.log(ch[names].to_numpy()) * w).sum(1)); c = C.best_scalar(ch.target, pc); s = C.rmspe(ch.target, c * pc)
        if s < best[0]: best = (s, (w, c))
w, c_blend = best[1]
rep = C.rmspe(rp.target, c_blend * np.exp((np.log(rp[names].to_numpy()) * w).sum(1)))
art = {"feature_cols_nn": fc, "feature_cols_v2": cols, "n_feat": len(fc), "v2_scalar": res["scalar"], "v3_scalar": scal["v3"], "v4_scalar": scal["v4"],
       "blend_weights": dict(zip(names, map(float, w))), "blend_scalar": float(c_blend), "choose_rmspe_blend": best[0], "report_rmspe_blend": rep,
       "v2_choose_rmspe": res["choose_rmspe"], "v2_report_rmspe": res["report_rmspe"], "stock_logrv_mean": {str(k): float(v) for k, v in stock_mean.items()},
       "seeds": [0, 1, 2], "role_proportions": [C.SPLIT_FIT, C.SPLIT_CHOOSE]}
json.dump(art, open(f"{DST}/artifacts.json", "w"))
for f in sorted(os.listdir(f"{SCR}/ckpt")): shutil.copy(f"{SCR}/ckpt/{f}", f"{DST}/{f}")
for f in ("__init__", "common", "features", "data", "nn_models"): shutil.copy(f"{ROOT}/src/opt14/{f}.py", f"{DST}/src/opt14/{f}.py")
json.dump({"title": "optiver14 weights", "id": os.environ.get("KAGGLE_USER", "<kaggle-user>") + "/optiver14-weights", "licenses": [{"name": "other"}], "isPrivate": True}, open(f"{DST}/dataset-metadata.json", "w"))
print(json.dumps({k: art[k] for k in ("v2_scalar", "v3_scalar", "v4_scalar", "blend_weights", "blend_scalar", "choose_rmspe_blend", "report_rmspe_blend", "v2_choose_rmspe", "v2_report_rmspe")}, indent=1))
print("dataset dir:", sorted(os.listdir(DST)))
