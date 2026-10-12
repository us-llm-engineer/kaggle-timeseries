"""Samples GPU utilisation + memory every 5 s beside the job; raw samples flushed every 12 samples (efficiency trace)."""
import json, os, subprocess, sys, time
out, stop = sys.argv[1], sys.argv[2]
rows = []
while not os.path.exists(stop):
    r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    try:
        u, m, t = [float(x) for x in r.stdout.strip().split(",")]
        rows.append({"t": round(time.time(), 1), "util": u, "mem_mib": m, "mem_total_mib": t})
    except Exception:
        pass
    if len(rows) % 12 == 0:
        json.dump(rows, open(out + ".tmp", "w")); os.replace(out + ".tmp", out)
    time.sleep(5)
json.dump(rows, open(out + ".tmp", "w")); os.replace(out + ".tmp", out)
