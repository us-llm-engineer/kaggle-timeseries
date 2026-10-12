"""Kaggle inference notebook: blend of V2 (LightGBM) + V3 (CNN, 3 seeds) + V4 (set transformer, 3 seeds). CPU, no internet.
Weights come from the private dataset <kaggle-user>/optiver14-weights. Features are rebuilt from the mounted book/trade files with the same code as training.
Self-check (only when the visible test is a small stub): the same pipeline is run on the training book/trade files and the report-role RMSPE is
compared with the value measured at training time (artifacts.json)."""
import glob, json, multiprocessing as mp, os, sys, time
import numpy as np, pandas as pd, pyarrow.parquet as pq

t0 = time.time()
ART = (os.environ.get("OPT_W") or glob.glob("/kaggle/input/**/artifacts.json", recursive=True)[0]) + ("" if os.environ.get("OPT_W") is None else "/artifacts.json")
W = os.path.dirname(ART); sys.path.insert(0, f"{W}/src")
COMP = os.environ.get("OPT_COMP") or os.path.dirname(glob.glob("/kaggle/input/**/test.csv", recursive=True)[0])
KIND = os.environ.get("OPT_KIND", "test")
import torch
from opt14 import common as C, features as F, nn_models as NN
torch.set_num_threads(os.cpu_count() or 4)
A = json.load(open(ART)); FC_NN, FC_V2 = A["feature_cols_nn"], A["feature_cols_v2"]
print("weights:", W, "| competition:", COMP, "| kind:", KIND, flush=True)


def _one(args):
    kind, sid = args
    b = pq.read_table(f"{COMP}/book_{kind}.parquet/stock_id={sid}").to_pandas()
    tp = f"{COMP}/trade_{kind}.parquet/stock_id={sid}"
    t = pq.read_table(tp).to_pandas() if os.path.exists(tp) else None
    for d in (b, t):
        if d is not None and "stock_id" in d.columns: d.drop(columns="stock_id", inplace=True)
    return F.build_stock(b, t, sid, want_seq=True)


def build(kind, stocks):
    with mp.get_context("fork").Pool(min(4, os.cpu_count() or 1)) as pool:
        res = pool.map(_one, [(kind, s) for s in stocks])
    feats = pd.concat([r[0] for r in res], ignore_index=True)
    seq = np.concatenate([r[1] for r in res], axis=0)
    feats = F.add_cross(feats)
    fallback = float(np.mean(list(A["stock_logrv_mean"].values())))
    feats["stock_logrv_mean"] = feats.stock_id.map(lambda s: A["stock_logrv_mean"].get(str(int(s)), fallback)).astype(np.float32)   # training-time per-stock level
    return feats, seq


_prep = np.load(f"{W}/prep.npz")
def transform(X):
    T = np.sign(X) * np.log1p(np.abs(X) / _prep["scale"])
    return np.clip(np.nan_to_num((T - _prep["mu"]) / _prep["sd"], nan=0.0), -8, 8).astype(np.float32)


@torch.no_grad()
def predict_all(feats, seq):
    import lightgbm as lgb
    out = {}
    bst = lgb.Booster(model_file=f"{W}/v2_cross_best.txt")
    out["v2_cross"] = A["v2_scalar"] * np.exp(bst.predict(feats[FC_V2].to_numpy(np.float32)))
    X = torch.from_numpy(transform(feats[FC_NN].to_numpy(np.float64)))
    log_ref = torch.log(torch.from_numpy(feats.rv_all.to_numpy(np.float32)).clamp(min=1e-5))
    stock = torch.from_numpy(feats.stock_id.to_numpy(np.int64))
    SEQ = torch.from_numpy(seq)
    # V3 seeds
    ps = []
    for s in range(3):
        ck = torch.load(f"{W}/v3_s{s}_best.pt", map_location="cpu"); m = NN.V3(X.shape[1]); m.load_state_dict(ck["model"]); m.eval()
        p = np.concatenate([torch.exp(m(X[i:i + 4096], SEQ[i:i + 4096].float(), stock[i:i + 4096], log_ref[i:i + 4096])).numpy() for i in range(0, len(X), 4096)])
        ps.append(ck["scalar"] * p)
    out["v3"] = A["v3_scalar"] * np.exp(np.log(np.stack(ps)).mean(0))
    # V4 seeds: one set per time_id
    tid = feats.time_id.to_numpy(); order = np.argsort(tid, kind="stable"); ut, st, ct = np.unique(tid[order], return_index=True, return_counts=True)
    S = int(ct.max()); sets = np.full((len(ut), S), -1, np.int64)
    for i, (a, c) in enumerate(zip(st, ct)): sets[i, :c] = order[a:a + c]
    SETS = torch.from_numpy(sets); ps = []
    for s in range(3):
        ck = torch.load(f"{W}/v4_s{s}_best.pt", map_location="cpu"); m = NN.V4(X.shape[1]); m.load_state_dict(ck["model"]); m.eval()
        p = np.zeros(len(feats), np.float32)
        for i in range(0, len(ut), 64):
            ix = SETS[i:i + 64]; pad = ix < 0; c = ix.clamp(min=0)
            lp = m(X[c], stock[c], log_ref[c], pad); p[ix[~pad].numpy()] = torch.exp(lp)[~pad].numpy()
        ps.append(ck["scalar"] * p)
    out["v4"] = A["v4_scalar"] * np.exp(np.log(np.stack(ps)).mean(0))
    w = A["blend_weights"]
    out["blend"] = A["blend_scalar"] * np.exp(sum(w[k] * np.log(out[k]) for k in ("v2_cross", "v3", "v4")))
    return out


test = pd.read_csv(f"{COMP}/test.csv")
stocks = sorted(test.stock_id.unique())
feats, seq = build(KIND if KIND != "test" else "test", stocks)
pred = predict_all(feats, seq)
res = feats[["stock_id", "time_id"]].copy(); res["target"] = np.clip(pred["blend"], 1e-5, 1.0)
sub = test.merge(res, on=["stock_id", "time_id"], how="left")
miss = sub.target.isna().sum()
if miss:
    print("WARNING: rows without a prediction:", miss, "-> filled with the blend median", flush=True)
    sub["target"] = sub.target.fillna(float(res.target.median()))
sub[["row_id", "target"]].to_csv("submission.csv", index=False)
print("submission", sub.shape, sub.target.describe().to_dict(), f"{time.time()-t0:.0f}s", flush=True)

# ---- self-check on the training files (only for the small public stub)
if len(test) <= 1000 and os.environ.get("OPT_SELFCHECK", "1") == "1":
    tr = pd.read_csv(f"{COMP}/train.csv"); tstocks = sorted(tr.stock_id.unique())
    f2, s2 = build("train", tstocks)
    p2 = predict_all(f2, s2)
    d = f2[["stock_id", "time_id"]].copy()
    for k, v in p2.items(): d[k] = v
    d = d.merge(tr, on=["stock_id", "time_id"]); d["role"] = C.split_role(d.time_id.to_numpy())
    for r, nm in ((1, "choose"), (2, "report")):
        x = d[d.role == r]
        print(f"SELFCHECK {nm} RMSPE:", {k: round(C.rmspe(x.target, x[k]), 4) for k in ("v2_cross", "v3", "v4", "blend")}, flush=True)
    print("expected (training time): report blend", round(A["report_rmspe_blend"], 4), "choose blend", round(A["choose_rmspe_blend"], 4), f"{time.time()-t0:.0f}s total")
