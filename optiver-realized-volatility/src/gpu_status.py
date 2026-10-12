"""Reader for the GPU run: files only. Usage: python3 gpu_status.py <out_dir> [--tail N]"""
import glob, json, os, sys, time
out = sys.argv[1]
def rd(p):
    try: return json.load(open(p))
    except Exception: return None
print("progress:", rd(f"{out}/progress.json"), "| pool:", {k: v for k, v in (rd(f"{out}/pool.json") or {}).items() if k != "pids"})
for h in sorted(glob.glob(f"{out}/v*_s*_history.json")):
    v = os.path.basename(h).replace("_history.json", ""); d = rd(h)
    if not d: print(v, "no history yet"); continue
    r = d["rows"][-1] if d["rows"] else {}
    best = min((x["choose_rmspe_ema"] for x in d["rows"]), default=None)
    print(f"{v}: epochs={len(d['rows'])} last choose={r.get('choose_rmspe')} best-ema={best} gap={r.get('gap')} {r.get('samples_per_s')}/s peak={r.get('gpu_peak_mb')}MB left={r.get('budget_left_s')}s since_write={time.time()-d['updated']:.0f}s final={d['meta'].get('report_rmspe')}")
u = rd(f"{out}/logs/gpu_utilisation.json")
if u:
    ut = [x["util"] for x in u]; print(f"gpu samples={len(u)} mean util {sum(ut)/len(ut):.0f}% last {ut[-1]:.0f}% mem {u[-1]['mem_mib']:.0f}/{u[-1]['mem_total_mib']:.0f} MiB")
for l in ["run_gpu"] + [os.path.basename(x)[:-4] for x in sorted(glob.glob(f"{out}/logs/v*_s*.log"))]:
    p = f"{out}/logs/{l}.log"
    if os.path.exists(p): print("--", l); print("".join(open(p).readlines()[-int(sys.argv[3]) if len(sys.argv) > 3 else -2:]), end="")
