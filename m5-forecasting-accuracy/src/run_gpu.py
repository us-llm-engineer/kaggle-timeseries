"""Launch K ensemble members as separate processes on one L4 (memory split 0.8/K), sample GPU utilisation and memory every 5 s, then average members per variant.
Run: python3 src/run_gpu.py --members 6 --budget-min 22 --n-orig 3"""
import argparse, os, subprocess, sys, time, threading, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m5lib import *

ap = argparse.ArgumentParser(); ap.add_argument("--members", type=int, default=6); ap.add_argument("--budget-min", type=float, default=22); ap.add_argument("--n-orig", type=int, default=3)
ap.add_argument("--target-mem", type=float, default=0.8); ap.add_argument("--extra", default=""); ap.add_argument("--out", default="results"); a = ap.parse_args()
os.makedirs("logs", exist_ok=True); os.makedirs(f"{a.out}/preds", exist_ok=True)
d = load_prep()
for r in ("choose", "report"): Scorer(d, r)          # build the cached scale/weights once, before the members start
samples, stop = [], threading.Event()


def sampler():
    while not stop.is_set():
        try:
            o = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True).split(",")
            samples.append((time.time(), float(o[0]), float(o[1]), float(o[2])))
            if len(samples) % 6 == 0: atomic_json("logs/gpu_utilisation.json", samples)
        except Exception: pass
        stop.wait(5)


threading.Thread(target=sampler, daemon=True).start()
frac = a.target_mem / a.members; procs = []
for k in range(a.members):
    variant = "hl" if k % 2 == 0 else "hurdle"
    cmd = [sys.executable, "src/gpu_member.py", "--variant", variant, "--seed", str(k // 2), "--mem-frac", f"{frac:.3f}", "--budget-min", str(a.budget_min), "--n-orig", str(a.n_orig), "--out", a.out] + a.extra.split()
    procs.append(subprocess.Popen(cmd, stdout=open(f"logs/member{k}.out", "w"), stderr=subprocess.STDOUT))
codes = [p.wait() for p in procs]; stop.set(); atomic_json("logs/gpu_utilisation.json", samples)
print("member exit codes", codes)
u = np.array([s[1] for s in samples]); m = np.array([s[2] / s[3] for s in samples])
summary = dict(members=a.members, util_mean=float(u.mean()), util_median=float(np.median(u)), util_p90=float(np.percentile(u, 90)), share_ge_80=float((u >= 80).mean()), mem_peak_share=float(m.max()), mem_mean_share=float(m.mean()))
sc = {r: Scorer(d, r) for r in ("choose", "report")}; res = {"gpu": summary}
for tag, nm in (("hl", "V5_hl_mixer"), ("hurdle", "V6_hurdle_mixer")):
    import glob
    fs = sorted(glob.glob(f"{a.out}/preds/{nm}_s*_choose.npy"))
    if not fs: continue
    ens = {r: np.mean([np.load(f.replace("_choose", "_" + r)) for f in fs], 0) for r in ORIGIN}
    for r in ORIGIN: np.save(f"{a.out}/preds/{nm}_ens_{r}.npy", ens[r])
    member_scores = [sc["choose"].score(np.load(f)) for f in fs]
    res[nm] = dict(members=len(fs), member_choose=member_scores, ens_choose=sc["choose"].score(ens["choose"]), ens_report=sc["report"].score(ens["report"], per_level=True)[0],
                   ens_report_per_level=sc["report"].score(ens["report"], per_level=True)[1].round(3).tolist())
atomic_json(f"{a.out}/gpu_history.json", res); print(json.dumps(res, indent=1))
