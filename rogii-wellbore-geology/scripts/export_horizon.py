"""Export RMSE by distance after the cut (100-row bins) from the stack's final_rows.npz to a small CSV.

Usage: python scripts/export_horizon.py runs/stack/final_rows.npz results/stack/horizon_rmse.csv
Each array in the npz is (rows, 3): [final error, carry-forward error, unused std].
"""
import sys

import numpy as np
import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
z = np.load(src)
edges = np.arange(0, 6001, 250)
se_f = np.zeros(len(edges) - 1); se_c = np.zeros_like(se_f); cnt = np.zeros_like(se_f)
for k in z.files:
    a = z[k]; h = np.arange(len(a))
    b = np.digitize(h, edges) - 1; ok = b < len(cnt)
    np.add.at(se_f, b[ok], a[ok, 0] ** 2); np.add.at(se_c, b[ok], a[ok, 1] ** 2); np.add.at(cnt, b[ok], 1)
m = cnt > 0
pd.DataFrame({"rows_after_cut_lo": edges[:-1][m], "rows_after_cut_hi": edges[1:][m], "points": cnt[m].astype(int),
              "final_rmse": np.sqrt(se_f[m] / cnt[m]), "carry_rmse": np.sqrt(se_c[m] / cnt[m])}).to_csv(dst, index=False)
print(dst)
