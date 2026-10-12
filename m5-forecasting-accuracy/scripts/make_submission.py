"""Write a Kaggle submission CSV from one variant's saved forecasts.
validation rows = forecasts from origin 1913 (days d_1914..d_1941), evaluation rows = origin 1941 (days d_1942..d_1969).
Run: python scripts/make_submission.py --variant V4_lagfree_aligned [--preds results/preds] [--out submission.csv]"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from m5lib import load_prep

ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="V4_lagfree_aligned"); ap.add_argument("--preds", default="results/preds"); ap.add_argument("--out", default=None)
a = ap.parse_args(); d = load_prep()
r, s = (np.load(f"{a.preds}/{a.variant}_{k}.npy") for k in ("report", "submit"))
df = pd.DataFrame(np.vstack([r, s]), columns=[f"F{i}" for i in range(1, 29)])
df.insert(0, "id", [i.replace("_evaluation", "_validation") for i in d.ids] + list(d.ids))
df = df.set_index("id").loc[d.sub_ids].reset_index()
df.to_csv(a.out or f"submission_{a.variant}.csv", index=False, float_format="%.5f"); print(df.shape, "negative cells:", int((df.iloc[:, 1:] < 0).sum().sum()))
