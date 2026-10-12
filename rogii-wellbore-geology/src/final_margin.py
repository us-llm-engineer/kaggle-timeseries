"""Final margin model for submission B: LightGBM (same outer recipe as the OOF stack) trained on all eligible wells,
target = TVT - A where A is each well's OUT-OF-FOLD 3-width snapshot blend (so the margin learns the blend's held-out
error pattern, as in the evaluated stack)."""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import lightgbm as lgb
import numpy as np

import stack as S
from wells import eligible_ids, fold_of, load_wells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="data/stage")
    ap.add_argument("--grid", default="runs/grid")
    ap.add_argument("--out", default="runs/final")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    wells = load_wells(args.stage); ids = eligible_ids(wells); folds = fold_of(ids)
    grid = S.load_grid(Path(args.grid), folds)
    keys = [k for k in grid if k[1] == 0]
    have = [i for i in ids if all(i in grid[k] for k in keys)]
    A = {i: S.blend(grid, wells, i, keys) for i in have}
    S._W["wells"] = wells
    with Pool(9) as pool:
        F = dict(pool.map(S._feat_job, [(i, A[i]) for i in have], chunksize=8))
    X = np.concatenate([F[i][::4] for i in have]); y = np.concatenate([(wells[i]["tvt"][wells[i]["p"]:] - A[i])[::4] for i in have])
    S.MARGIN = "lgbm"
    m = S.fit_margin(X, y, True)
    m.save_model(str(out / "margin_lgbm.txt"))
    print({"wells": len(have), "rows": len(y), "keys": keys})


if __name__ == "__main__":
    main()
