"""CPU host entry (Modal sandbox): build features from the HF repo with a process pool, then V1 and V2 (two LightGBM processes).
Everything is checkpointed/atomic under --out; non-blocking launch, poll with cpu_status.py. Stage budgets sum below the cap (asserted)."""
import argparse, json, multiprocessing as mp, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opt14 import common as C, data as D, obs, v1_ridge, v2_gbdt
from opt14.build import build_all

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--cache", default="cache")
ap.add_argument("--workers", type=int, default=7); ap.add_argument("--stocks", default="")
ap.add_argument("--build-min", type=float, default=6.0); ap.add_argument("--v2-min", type=float, default=8.0); ap.add_argument("--cap-min", type=float, default=20.0)
ap.add_argument("--rounds", type=int, default=4000)
a = ap.parse_args()
assert a.build_min + a.v2_min + 2 < a.cap_min, "stage budgets must sum below the cap"
os.makedirs(a.out, exist_ok=True); os.makedirs(f"{a.out}/logs", exist_ok=True)
LOG = f"{a.out}/logs/run_cpu.log"


def v2_worker(name, cross, threads, budget_s):
    df = D.load_table(f"{a.out}/features.parquet", C.hf_get("train.csv", a.cache))
    v2_gbdt.run(df, a.out, name, cross, threads, budget_s, logf=f"{a.out}/logs/{name}.log", max_rounds=a.rounds)


if __name__ == "__main__":
    t_all = time.time()
    obs.log(LOG, f"start workers={a.workers} cap={a.cap_min}min build<={a.build_min} v2<={a.v2_min}")
    build_all(a.out, a.cache, a.workers, [int(x) for x in a.stocks.split(',')] if a.stocks else None, False, LOG)
    df = D.load_table(f"{a.out}/features.parquet", C.hf_get("train.csv", a.cache))
    obs.log(LOG, f"table {df.shape}; roles {np.bincount(df.role.to_numpy())}")
    C.atomic_json({"stage": "v1", "sec": round(time.time() - t_all, 1)}, f"{a.out}/progress.json")
    v1_ridge.run_all(df, a.out, LOG)
    C.atomic_json({"stage": "v2", "sec": round(time.time() - t_all, 1)}, f"{a.out}/progress.json")
    budget = min(a.v2_min * 60, a.cap_min * 60 - (time.time() - t_all) - 90)
    t1 = max(1, int(a.workers * 0.57)); th = (t1, max(1, a.workers - t1))      # workers = 85% of cores, split over the two LightGBM processes
    ctx = mp.get_context("spawn")
    ps = [ctx.Process(target=v2_worker, args=("v2_cross", True, th[0], budget)), ctx.Process(target=v2_worker, args=("v2_nocross", False, th[1], budget))]
    [p.start() for p in ps]; [p.join() for p in ps]
    C.atomic_json({"stage": "done", "sec": round(time.time() - t_all, 1)}, f"{a.out}/progress.json")
    obs.log(LOG, f"ALL DONE in {time.time()-t_all:.0f}s")
