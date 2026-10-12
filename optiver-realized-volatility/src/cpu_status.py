"""Reader for the CPU run: files only. Usage: python3 cpu_status.py <out_dir>"""
import json, os, sys, time, glob
out = sys.argv[1]
def rd(p):
    try: return json.load(open(p))
    except Exception: return None
pr = rd(f"{out}/progress.json"); print("progress:", pr)
for h in sorted(glob.glob(f"{out}/*_history.json")):
    d = rd(h)
    if not d: continue
    r = d["rows"][-1] if d["rows"] else {}
    age = time.time() - d["updated"]
    print(f"{os.path.basename(h):28s} rows={len(d['rows']):4d} last={json.dumps({k: (round(v,4) if isinstance(v,float) else v) for k,v in r.items() if k in ('round','alpha','choose_rmspe','fit_rmspe','since_best','sec','rounds_per_s')})} since_write={age:.0f}s final={ {k:d['meta'].get(k) for k in ('choose_rmspe','report_rmspe','stop')} if 'report_rmspe' in d['meta'] else None}")
for l in sorted(glob.glob(f"{out}/logs/*.log")):
    print("--", os.path.basename(l)); print("".join(open(l).readlines()[-2:]), end="")
