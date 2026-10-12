"""Data access: wells with their typewell, fixed fold roles, metrics.

Stage is either a local Parquet stage (data/stage, written by prepare_data.py) or an hf://datasets/... URI streamed lazily.
Fold roles are fixed in advance: fold 1 = choose, folds 2-5 = report (out-of-fold).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

TEST_IDS = {"000d7d20", "00bbac68", "00e12e8b"}
SEED = 20261011
CHOOSE_FOLD = 0  # "fold 1" in the docs (0-based here)


def _rows(stage: str, table: str, split: str):
    if stage.startswith("hf://"):
        from datasets import load_dataset
        ds = load_dataset("parquet", data_files=f"{stage}/{table}/{split}/*.parquet", streaming=True, split="train")
        yield from ds
        return
    import pyarrow.parquet as pq
    for path in sorted((Path(stage) / table / split).glob("*.parquet")):
        yield from pq.read_table(path).to_pylist()


def _arr(values) -> np.ndarray:
    return np.asarray([np.nan if v is None else v for v in values], dtype=np.float64)


def load_wells(stage: str, split: str = "train") -> dict[str, dict]:
    """All wells of a split: horizontal arrays plus typewell (tvt, gr). Test-duplicate ids are kept but flagged."""
    typewells = {}
    for row in _rows(stage, "typewell", split):
        t, g = _arr(row["tvt"]), _arr(row["gr"])
        keep = np.isfinite(t) & np.isfinite(g)
        order = np.argsort(t[keep])
        typewells[row["well_id"]] = (t[keep][order], g[keep][order])
    wells = {}
    for row in _rows(stage, "horizontal", split):
        wid = row["well_id"]
        w = {k: _arr(row[k]) for k in ("md", "x", "y", "z", "gr", "tvt", "tvt_input")}
        w["well_id"], w["p"], w["n"] = wid, int(row["prefix_end"]), int(row["n_points"])
        w["tw_tvt"], w["tw_gr"] = typewells[wid]
        wells[wid] = w
    return wells


def eligible_ids(wells: dict[str, dict]) -> list[str]:
    out = []
    for wid, w in wells.items():
        if wid in TEST_IDS or w["p"] < 8 or w["p"] >= w["n"] or np.isnan(w["tvt"]).any():
            continue
        out.append(wid)
    return sorted(out)


def fold_of(ids: list[str]) -> dict[str, int]:
    """Fixed split: sorted ids, KFold(5, shuffle, seed 20261011)."""
    f = {}
    for k, (_, va) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(ids)):
        for i in va:
            f[ids[i]] = k
    return f


def suffix_metrics(rows: list[dict]) -> dict:
    """rows: one dict per well with keys err (array over suffix rows), carry_err, gr_missing, std (optional)."""
    sse = sum(float(np.sum(r["err"] ** 2)) for r in rows)
    n = sum(len(r["err"]) for r in rows)
    carry_sse = sum(float(np.sum(r["carry_err"] ** 2)) for r in rows)
    bands = {"0_255": (0, 256), "256_1023": (256, 1024), "1024_plus": (1024, 10**9)}
    band = {}
    for name, (lo, hi) in bands.items():
        e = np.concatenate([r["err"][lo:hi] for r in rows])
        band[name] = float(np.sqrt(np.mean(e ** 2))) if len(e) else None
    rm = np.array([np.sqrt(np.mean(r["err"] ** 2)) for r in rows])
    cm = np.array([np.sqrt(np.mean(r["carry_err"] ** 2)) for r in rows])
    miss = np.array([r["gr_missing"] for r in rows])
    terc = {}
    if len(rows) >= 6:
        q = np.quantile(miss, [1 / 3, 2 / 3])
        groups = np.digitize(miss, q)
        for g in range(3):
            sel = [r for r, gg in zip(rows, groups) if gg == g]
            terc[f"gr_missing_t{g}"] = float(np.sqrt(sum(np.sum(r["err"] ** 2) for r in sel) / sum(len(r["err"]) for r in sel)))
    out = {"wells": len(rows), "points": n, "pooled_rmse": float(np.sqrt(sse / n)), "carry_pooled_rmse": float(np.sqrt(carry_sse / n)),
           "median_well_rmse": float(np.median(rm)), "horizon_rmse": band, **terc,
           "jump_rate": float(np.mean(rm > cm + 10.0)), "wells_better_than_carry": int(np.sum(rm < cm))}
    if all("std" in r for r in rows):
        s = np.concatenate([r["std"] for r in rows]); a = np.abs(np.concatenate([r["err"] for r in rows]))
        out["calib_corr_std_abs_err"] = float(np.corrcoef(s, a)[0, 1])
        out["coverage_1sigma"] = float(np.mean(a <= s))
    return out


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record, default=float) + "\n")
