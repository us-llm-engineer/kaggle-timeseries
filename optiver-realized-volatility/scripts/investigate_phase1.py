"""Phase 1 data investigation: stream per-stock parquet from the private HF repo, build one compact
per-(stock,time_id) profile table, then print the facts that decide the methodology.
Run: python3 src/investigate_phase1.py   (cwd anywhere; writes results/phase1_profile.parquet and notes/phase1_profile.txt)
Data never sits on local disk: each stock file goes to a scratch cache and is removed after use."""
import os, sys, shutil, time, io
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from opt14 import common as C

ROOT = os.environ.get("OPT_OUT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKERS = int(os.environ.get("OPT_WORKERS", "1"))
CACHE = os.environ.get("OPT_CACHE", "cache")
os.makedirs(CACHE, exist_ok=True)


def get(path):
    return C.hf_get(path, CACHE)


def rv(x):
    return np.sqrt(np.sum(x * x))


def list_files():
    return C.list_files()


def profile_one(sid, book_f, trade_f):
    b = pq.read_table(book_f).to_pandas()
    b = b.sort_values(["time_id", "seconds_in_bucket"], kind="stable")
    wap = (b.bid_price1 * b.ask_size1 + b.ask_price1 * b.bid_size1) / (b.bid_size1 + b.ask_size1)
    b["lw"] = np.log(wap)
    b["r"] = b.groupby("time_id")["lw"].diff()
    b["r2"] = b["r"] ** 2
    late = b["seconds_in_bucket"] >= 300
    b["r2_late"] = np.where(late, b["r2"], 0.0)
    b["spread"] = (b.ask_price1 - b.bid_price1) / ((b.ask_price1 + b.bid_price1) / 2)
    b["imb"] = (b.bid_size1 - b.ask_size1) / (b.bid_size1 + b.ask_size1)
    g = b.groupby("time_id")
    out = pd.DataFrame({
        "rv_all": np.sqrt(g["r2"].sum()),
        "rv_late": np.sqrt(g["r2_late"].sum()),
        "n_book": g.size(),
        "max_sec": g["seconds_in_bucket"].max(),
        "spread": g["spread"].mean(),
        "imb": g["imb"].mean(),
        "depth1": g.apply(lambda d: (d.bid_size1 + d.ask_size1).mean(), include_groups=False),
        "px_last": g["lw"].last(),
    })
    t = pq.read_table(trade_f).to_pandas() if trade_f else None
    if t is not None and len(t):
        tg = t.groupby("time_id")
        tt = pd.DataFrame({"n_trade": tg.size(), "trade_vol": tg["size"].sum(), "trade_orders": tg["order_count"].sum()})
        out = out.join(tt)
    out["n_trade"] = out.get("n_trade", 0)
    out = out.fillna({"n_trade": 0, "trade_vol": 0, "trade_orders": 0})
    out.insert(0, "stock_id", sid)
    return out.reset_index()


def work(args):
    sid, files = args
    bf = [f for f in files if f.startswith(f"book_train.parquet/stock_id={sid}/")]
    tf = [f for f in files if f.startswith(f"trade_train.parquet/stock_id={sid}/")]
    bp = get(bf[0]); tp = get(tf[0]) if tf else None
    r = profile_one(sid, bp, tp)
    if C.is_cached_copy():
        os.remove(bp)
        if tp: os.remove(tp)
    return r


def main():
    t0 = time.time()
    files = list_files()
    print("repo files:", len(files))
    train = pd.read_csv(get("train.csv"))
    test = pd.read_csv(get("test.csv"))
    sub = pd.read_csv(get("sample_submission.csv"))
    print("train", train.shape, "test", test.shape, "sample_submission", sub.shape)
    print(test.to_string())
    sids = sorted(train.stock_id.unique())
    from multiprocessing import Pool
    with Pool(WORKERS) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(work, [(sid, files) for sid in sids])):
            rows.append(r)
            if i % 10 == 0:
                print(f"[{i+1}/{len(sids)}] elapsed {time.time()-t0:.0f}s", flush=True)
    prof = pd.concat(rows, ignore_index=True).merge(train, on=["stock_id", "time_id"], how="outer")
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    prof.to_parquet(f"{ROOT}/results/phase1_profile.parquet")
    shutil.rmtree(CACHE, ignore_errors=True)
    print("profile saved", prof.shape, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
