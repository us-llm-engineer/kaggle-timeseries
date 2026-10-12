"""Build notebooks/optiver14_L4.ipynb for the Colab L4 host. Every code cell that carries code is a byte-for-byte %%writefile of a local file
(sha256 in the cell metadata); local repo is the single source of truth. Run: python3 src/build_notebook_colab.py"""
import hashlib, json, os
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REMOTE = "/content/optiver14"
FILES = [("src/opt14/__init__.py",), ("src/opt14/common.py",), ("src/opt14/features.py",), ("src/opt14/data.py",), ("src/opt14/obs.py",),
         ("src/opt14/nn_models.py",), ("src/opt14/build.py",), ("src/run_nn.py",), ("src/run_gpu.py",), ("src/gpu_sampler.py",), ("src/gpu_status.py",),
         ("tests/synth.py",), ("tests/test_features.py",), ("tests/test_nn.py",)]
OUT = "/content/drive/MyDrive/optiver14/out"

nb = nbf.v4.new_notebook()
cells = nb.cells


def md(text, tags=()):
    c = nbf.v4.new_markdown_cell(text); c.metadata["tags"] = list(tags); cells.append(c)


def code(src, tags=(), **meta):
    c = nbf.v4.new_code_cell(src); c.metadata["tags"] = list(tags); c.metadata.update(meta); cells.append(c)


md("""# Optiver realized volatility - V3 and V4 on a Colab L4 (offloaded cells)

**Execution host:** Colab runtime with an **L4** GPU (asserted below). **Data:** streamed from the private Hugging Face dataset `<hf-user>/optiver-volatility` (set `OPT_HF_REPO`) on this host; nothing is read from the author's disk.
**Variants (this notebook):** V3 = 1-D CNN on the per-second book/trade tensor (120 bins x 13 channels) + aggregated features, one sample at a time;
V4 = set transformer over the (up to 112) stocks of one `time_id`, permutation-equivariant. V1 (noise-robust estimators, ridge) and V2 (LightGBM) run on a Modal CPU sandbox.
**Fixed split (by `time_id`, deterministic hash):** 70% fit / 15% choose (early stopping, scalar calibration) / 15% report (monitor only, never used to choose). **Metric:** RMSPE.
**Budget:** training cap 45 min for the whole project; this host: build <= 7 min, V3 and V4 (3 seeds each, 6 processes) concurrently <= 11 min each, total <= 22 min (asserted in code). GPU memory fraction 0.85 split over two processes; build uses 85% of the CPU cores.
**Observability:** one log line per epoch per worker, atomic history JSON per unit, GPU utilisation samples every 5 s, checkpoints (best and last) - all written directly to Google Drive.
**Run order:** install -> modules -> host check -> mount Drive -> Hugging Face login (text box) -> tests -> smoke -> one real unit + projection (STOP) -> `needs-approval` launch -> watch/status -> summary -> trace.
Cells tagged `offloaded` run on the cloud; `needs-approval` cells cost compute time and wait for the user.""")

code(f"""!pip -q install huggingface_hub pyarrow 2>&1 | tail -1
import os
for d in ("{REMOTE}/src/opt14", "{REMOTE}/tests"):
    os.makedirs(d, exist_ok=True)
print("directories ready")""", ["offloaded"])

for (rel,) in FILES:
    path = os.path.join(ROOT, rel)
    body = open(path).read()
    sha = hashlib.sha256(body.encode()).hexdigest()
    code(f"%%writefile {REMOTE}/{rel}\n{body}", ["offloaded"], source_file=rel, sha256=sha)

code(f"""import os, torch, multiprocessing
os.makedirs("{REMOTE}", exist_ok=True)
name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU"
print("host: Colab | gpu:", name, "| cpu cores:", os.cpu_count(), "| torch:", torch.__version__)
assert "L4" in name, "this notebook is for an L4 runtime"
import psutil; print("RAM GB:", round(psutil.virtual_memory().total/2**30,1))""", ["offloaded"])

code(f"""from google.colab import drive
drive.mount("/content/drive")
import os
OUT = "{OUT}"; os.makedirs(OUT, exist_ok=True)
open(OUT + "/.probe", "w").write("ok"); assert open(OUT + "/.probe").read() == "ok"; os.remove(OUT + "/.probe")
print("Drive writable:", OUT)""", ["offloaded"])

code("""# Hugging Face login (read key). A plain text box appears if no Colab secret HF_TOKEN exists; the token only goes to os.environ (process memory),
# is never printed or written to a file, and dies with the runtime.
import os, getpass
tok = None
try:
    from google.colab import userdata
    tok = userdata.get("HF_TOKEN")
except Exception:
    pass
if not tok:
    tok = getpass.getpass("Hugging Face token: ")
os.environ["HF_TOKEN"] = tok.strip()
from huggingface_hub import HfApi
print("logged in as", HfApi(token=os.environ["HF_TOKEN"]).whoami()["name"])""", ["offloaded"])

code(f"""!cd {REMOTE}/tests && python3 -m pytest -q test_features.py test_nn.py 2>&1 | tail -5""", ["offloaded"])

code(f"""# smoke: 3 real stocks, 20 s of NN budget per worker - proves the pipeline end to end on this host (small, seconds to ~2 min)
import subprocess, sys
r = subprocess.run([sys.executable, "{REMOTE}/src/run_gpu.py", "--out", "{OUT}_smoke", "--cache", "/content/opt_cache_smoke", "--stocks", "0,1,2",
                    "--build-min", "3", "--nn-min", "0.4", "--cap-min", "8", "--max-epochs", "6", "--seeds", "2"], capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-800:])
!python3 {REMOTE}/src/gpu_status.py {OUT}_smoke out 3""", ["offloaded"])

code(f"""# ONE real-scale unit: build 8 real stocks with the real pool, project the whole build against its stage budget, then STOP.
import subprocess, sys, time, os, math
W = max(1, int(os.cpu_count() * 0.85))
t = time.time()
r = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0,'{REMOTE}/src'); from opt14.build import build_all; "
                    "build_all('{OUT}_unit', '/content/opt_cache_unit', %d, [0,1,2,3,4,5,6,7], True, None)" % W],
                   capture_output=True, text=True, env=dict(os.environ))
el = time.time() - t
print(r.stdout[-400:], r.stderr[-600:])
rounds8 = math.ceil(8 / W); per_round = el / rounds8; full = math.ceil(112 / W) * per_round
print(f"{{W}} workers: 8 stocks in {{el:.0f}} s ({{per_round:.0f}} s per round of {{W}}) -> projected full build {{full:.0f}} s vs stage budget 420 s; NN stage 660 s; total {{(full+660)/60:.1f}} min vs cap 22 min.")
print("STOP: wait for the user's go-ahead before the launch cell.")""", ["offloaded", "needs-approval"])

code(f"""# LAUNCH (needs-approval): detached pipeline, returns immediately. build -> V3 and V4 in parallel + GPU sampler.
import subprocess, sys
p = subprocess.Popen([sys.executable, "{REMOTE}/src/run_gpu.py", "--out", "{OUT}", "--cache", "/content/opt_cache", "--build-min", "7", "--nn-min", "11", "--cap-min", "22"],
                     stdout=open("{OUT}/launcher.stdout", "w"), stderr=subprocess.STDOUT, start_new_session=True)
print("launched pid", p.pid)""", ["offloaded", "needs-approval"])

code(f"""# WATCH: prints only new log lines (merged, prefixed, untruncated), bounded in time, always returns.
import glob, os, time
OUT = "{OUT}"; pos = globals().get("_pos", {{}})
t_end = time.time() + 90
while time.time() < t_end:
    for p in ["{OUT}/logs/run_gpu.log"] + sorted(glob.glob("{OUT}/logs/v*_s*.log")):
        if os.path.exists(p):
            n = os.path.basename(p)[:-4]
            with open(p) as f:
                f.seek(pos.get(n, 0)); new = f.read(); pos[n] = f.tell()
            for line in new.splitlines(): print(f"[{{n}}] {{line}}")
    time.sleep(5)
_pos = pos""", ["offloaded"])

code(f"""!python3 {REMOTE}/src/gpu_status.py {OUT} out 2""", ["offloaded"])

code(f"""# SUMMARY from the saved files only (no recomputation): per-seed report/choose RMSPE and GPU efficiency.
import glob, json, numpy as np
OUT = "{OUT}"
for h in sorted(glob.glob(f"{{OUT}}/v*_s*_history.json")):
    d = json.load(open(h)); m = d["meta"]
    print(h.split("/")[-1][:-13], {{k: m.get(k) for k in ("stop", "best_epoch", "epochs", "choose_rmspe", "report_rmspe", "seconds", "halvings")}})
u = json.load(open(f"{{OUT}}/logs/gpu_utilisation.json")); ut = np.array([x["util"] for x in u]); mem = np.array([x["mem_mib"] for x in u])
print(f"GPU samples {{len(u)}}: mean util {{ut.mean():.0f}}%, median {{np.median(ut):.0f}}%, p90 {{np.percentile(ut,90):.0f}}%, share>=80% {{(ut>=80).mean():.2f}}, peak mem {{mem.max():.0f}} MiB / {{u[0]['mem_total_mib']:.0f}}")
print(open(f"{{OUT}}/provenance.json").read())""", ["offloaded"])

code(f"""# FINAL TRACE: one merged observability trace saved next to the results (no end-of-run copy needed: everything is already on Drive).
import glob, os
OUT = "{OUT}"
with open(f"{{OUT}}/observability_trace.txt", "w") as f:
    for p in sorted(glob.glob(f"{{OUT}}/logs/*.log")):
        f.write(f"===== {{os.path.basename(p)}}\\n" + open(p).read())
print("trace:", os.path.getsize(f"{{OUT}}/observability_trace.txt"), "bytes")""", ["offloaded"])

os.makedirs(f"{ROOT}/notebooks", exist_ok=True)
path = f"{ROOT}/notebooks/optiver14_L4.ipynb"
nbf.write(nb, path)
print("wrote", path, len(nb.cells), "cells")
