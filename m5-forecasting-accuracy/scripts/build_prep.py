"""Build the compact array file used by every script from the competition files.
Run: M5_DATA_DIR=/path/to/m5-forecasting-accuracy python scripts/build_prep.py   (writes prep/m5_prep.npz, about 40 MB)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import m5lib

if __name__ == "__main__":
    m5lib.build_prep()
    d = m5lib.load_prep(); [m5lib.Scorer(d, r) for r in ("choose", "report")]      # caches the WRMSSE scale and weights
    print("series", d.n, "days", d.T, "aggregate series", d.S.shape[0])
