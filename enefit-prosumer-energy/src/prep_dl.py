"""Build the dense per-series tensors once (CPU) and save td.npz; workers then load it in seconds."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enefit13 import data as D, nn as N

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data_hf"); ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
store = D.Store(a.root)
panel = D.load_panel(store)
an, ar, s, ti, sid, t0 = D.series_grid(panel)
td = N.build_tensors(panel, an, ar, s, ti, sid, t0)
np.savez(os.path.join(a.out, "td.npz"), an=td["an"], ar=td["ar"], wx=td["wx"], sc=td["sc"], static=td["static"])
json.dump({"t0": str(t0), "S": int(td["S"]), "T": int(td["T"]), "sid": {f"{k[0]},{k[1]}": v for k, v in sid.items()}},
          open(os.path.join(a.out, "td_meta.json"), "w"))
print("saved", {k: getattr(v, "shape", v) for k, v in td.items() if k in ("an", "wx", "sc", "static", "S", "T")})
