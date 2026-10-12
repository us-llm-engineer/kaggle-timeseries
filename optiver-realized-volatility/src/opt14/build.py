"""Shared build: HF repo -> per-stock features (+ sequence tensors) with a process pool -> features.parquet (+ seq.npy, seq_keys.npy)."""
import multiprocessing as mp, os, time
import numpy as np, pandas as pd, pyarrow.parquet as pq
from . import common as C, features as F, obs

_G = {}


def _init(cache, want_seq, files):
    _G.update(cache=cache, want_seq=want_seq, files=files)


def _work(sid):
    cache, files = _G["cache"], _G["files"]
    bf = [f for f in files if f.startswith(f"book_train.parquet/stock_id={sid}/")][0]
    tf = [f for f in files if f.startswith(f"trade_train.parquet/stock_id={sid}/")]
    bp = C.hf_get(bf, cache); tp = C.hf_get(tf[0], cache) if tf else None
    feat, seq = F.build_stock(pq.read_table(bp).to_pandas(), pq.read_table(tp).to_pandas() if tp else None, sid, want_seq=_G["want_seq"])
    if C.is_cached_copy():
        os.remove(bp)
        if tp: os.remove(tp)
    return sid, feat, seq


def build_all(out, cache, workers, stocks=None, want_seq=False, logf=None):
    os.makedirs(out, exist_ok=True)
    files = C.list_files()
    train = pd.read_csv(C.hf_get("train.csv", cache))
    sids = sorted(train.stock_id.unique()) if not stocks else list(stocks)
    t0, parts = time.time(), {}
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(cache, want_seq, files)) as pool:
        for i, (sid, feat, seq) in enumerate(pool.imap_unordered(_work, sids)):
            parts[sid] = (feat, seq)
            if i % 8 == 0 or i == len(sids) - 1:
                el = time.time() - t0
                obs.log(logf, f"build {i+1}/{len(sids)} stocks {100*(i+1)/len(sids):.0f}% {el:.0f}s eta {el/(i+1)*(len(sids)-i-1):.0f}s")
                C.atomic_json({"stage": "build", "done": i + 1, "total": len(sids), "sec": round(el, 1)}, f"{out}/progress.json")
    order = sorted(parts)
    feats = F.add_cross(pd.concat([parts[s][0] for s in order], ignore_index=True))
    feats.to_parquet(f"{out}/features.parquet")
    if want_seq:
        np.save(f"{out}/seq.npy", np.concatenate([parts[s][1] for s in order], axis=0))
        np.save(f"{out}/seq_keys.npy", feats[["stock_id", "time_id"]].to_numpy(np.int64))
    obs.log(logf, f"features {feats.shape} built in {time.time()-t0:.0f}s (seq={want_seq})")
    return feats
