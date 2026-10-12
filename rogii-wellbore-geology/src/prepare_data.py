"""Materialize a private, shardable Parquet representation of the Kaggle bundle.

Rows represent wells, preserving variable-length curves so downstream consumers can
use Hugging Face Datasets streaming instead of loading all 773 wells at once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HORIZONTAL = ("MD", "X", "Y", "Z", "GR", "TVT", "TVT_input")
TYPEWELL = ("TVT", "GR")


def well_id(path: Path) -> str:
    return path.name.removesuffix("__horizontal_well.csv").removesuffix("__typewell.csv")


def clean_list(values: pd.Series) -> list[float | None]:
    return [None if pd.isna(value) else float(value) for value in values]


def horizontal_row(path: Path, has_target: bool) -> dict[str, object]:
    frame = pd.read_csv(path)
    required = list(HORIZONTAL if has_target else tuple(x for x in HORIZONTAL if x != "TVT"))
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    observed = frame["TVT_input"].notna().to_numpy()
    prefix_end = int(np.flatnonzero(observed)[-1]) + 1 if observed.any() else 0
    if observed[prefix_end:].any():
        raise ValueError(f"{path}: TVT_input is not a prefix")
    row: dict[str, object] = {"well_id": well_id(path), "n_points": len(frame), "prefix_end": prefix_end}
    for column in HORIZONTAL:
        row[column.lower()] = clean_list(frame[column]) if column in frame else [None] * len(frame)
    return row


def typewell_row(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path)
    missing = sorted(set(TYPEWELL) - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return {"well_id": well_id(path), "n_points": len(frame), "tvt": clean_list(frame["TVT"]), "gr": clean_list(frame["GR"])}


def write_shards(rows: list[dict[str, object]], directory: Path, prefix: str, shard_size: int) -> list[dict[str, object]]:
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for number, start in enumerate(range(0, len(rows), shard_size)):
        shard = directory / f"{prefix}-{number:04d}.parquet"
        table = pa.Table.from_pylist(rows[start:start + shard_size])
        pq.write_table(table, shard, compression="zstd", row_group_size=shard_size)
        records.append({"path": str(shard), "rows": table.num_rows, "sha256": hashlib.sha256(shard.read_bytes()).hexdigest()})
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out", type=Path, default=Path("data/stage"))
    parser.add_argument("--shard-size", type=int, default=16)
    args = parser.parse_args()
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    manifests: list[dict[str, object]] = []
    for split in ("train", "test"):
        root = args.source / split
        horizontal = [horizontal_row(path, has_target=(split == "train")) for path in sorted(root.glob("*__horizontal_well.csv"))]
        typewells = [typewell_row(path) for path in sorted(root.glob("*__typewell.csv"))]
        manifests.extend(write_shards(horizontal, args.out / "horizontal" / split, "part", args.shard_size))
        manifests.extend(write_shards(typewells, args.out / "typewell" / split, "part", args.shard_size))
        for row in horizontal:
            manifests.append({"well_id": row["well_id"], "split": split, "n_points": row["n_points"], "prefix_end": row["prefix_end"]})
        print(json.dumps({"split": split, "horizontal_wells": len(horizontal), "typewells": len(typewells)}), flush=True)
    submission = pd.read_csv(args.source / "sample_submission.csv")
    pq.write_table(pa.Table.from_pandas(submission, preserve_index=False), args.out / "sample_submission.parquet", compression="zstd")
    (args.out / "manifest.json").write_text(json.dumps({"format": "well-row-parquet-v1", "shard_size": args.shard_size, "records": manifests}, indent=2) + "\n")
    (args.out / "README.md").write_text(
        "# Rogii wellbore geology prediction\n\nPrivate streaming representation of the competition data. "
        "Each Parquet row is one well with variable-length list columns. Do not redistribute.\n"
    )
    print(json.dumps({"out": str(args.out), "files": len(list(args.out.rglob("*.parquet"))), "submission_rows": len(submission)}))


if __name__ == "__main__":
    main()
