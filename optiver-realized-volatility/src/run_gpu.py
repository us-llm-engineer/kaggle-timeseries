"""GPU host entry (Colab L4). One detached pipeline: build features + sequence tensors from the HF repo (process pool, 85% of cores),
then V3 and V4 as separate processes (own CUDA context each, GPU memory fraction 0.85/2 each) beside a GPU sampler.
Launch it without blocking; poll with gpu_status.py; every component writes its own log + atomic history under --out (Drive on Colab)."""
import argparse, json, os, shutil, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opt14 import common as C, obs
from opt14.build import build_all

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--cache", default="cache")
ap.add_argument("--workers", type=int, default=0); ap.add_argument("--stocks", default="")
ap.add_argument("--build-min", type=float, default=7.0); ap.add_argument("--nn-min", type=float, default=11.0); ap.add_argument("--cap-min", type=float, default=22.0)
ap.add_argument("--gpu-frac", type=float, default=0.85); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--max-epochs", type=int, default=400)
a = ap.parse_args()


def main():
    assert a.build_min + a.nn_min + 2.5 < a.cap_min, "stage budgets must sum below the cap"
    os.makedirs(f"{a.out}/logs", exist_ok=True); LOG = f"{a.out}/logs/run_gpu.log"
    workers = a.workers or max(1, int(os.cpu_count() * 0.85))
    t_all = time.time()
    obs.log(LOG, f"start cpu={os.cpu_count()} build_workers={workers} cap={a.cap_min}min build<={a.build_min} nn<={a.nn_min} gpu_frac={a.gpu_frac}")
    build_all(a.out, a.cache, workers, [int(x) for x in a.stocks.split(",")] if a.stocks else None, True, LOG)
    shutil.copy(C.hf_get("train.csv", a.cache), f"{a.out}/train.csv")
    left = a.cap_min * 60 - (time.time() - t_all) - 60
    budget = min(a.nn_min, left / 60.0)
    obs.log(LOG, f"build done at {time.time()-t_all:.0f}s; NN budget {budget:.1f} min each")
    stop = f"{a.out}/sampler.stop"
    if os.path.exists(stop): os.remove(stop)
    sp = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_sampler.py"), f"{a.out}/logs/gpu_utilisation.json", stop],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    here = os.path.dirname(os.path.abspath(__file__)); procs = {}
    share = {"v3": 0.6, "v4": 0.4}                        # V3 holds the sequence tensor on the GPU, V4 is small
    for v in ("v3", "v4"):
        for sd in range(a.seeds):
            frac = a.gpu_frac * share[v] / a.seeds; nm = f"{v}_s{sd}"
            lf = open(f"{a.out}/logs/{nm}.stdout", "w")
            procs[nm] = subprocess.Popen([sys.executable, os.path.join(here, "run_nn.py"), "--variant", v, "--seed", str(sd), "--prep", a.out, "--out", a.out,
                                          "--frac", f"{frac:.4f}", "--budget-min", f"{budget:.2f}", "--max-epochs", str(a.max_epochs)], stdout=lf, stderr=subprocess.STDOUT)
    C.atomic_json({"stage": "nn", "pids": {k: p.pid for k, p in procs.items()}, "t": time.time(), "budget_min": budget, "gpu_frac_total": a.gpu_frac, "seeds": a.seeds, "build_sec": round(time.time() - t_all, 1)}, f"{a.out}/pool.json")
    for p in procs.values(): p.wait()
    open(stop, "w").write("1"); sp.wait(timeout=30)
    prov = {"cpu_count": os.cpu_count(), "wall_s": round(time.time() - t_all, 1), "exit": {k: p.returncode for k, p in procs.items()}}
    try:
        import torch; prov.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception: pass
    C.atomic_json(prov, f"{a.out}/provenance.json"); C.atomic_json({"stage": "done", "sec": prov["wall_s"]}, f"{a.out}/progress.json")
    obs.log(LOG, f"ALL DONE in {prov['wall_s']}s exit={prov['exit']}")


if __name__ == "__main__":
    main()
