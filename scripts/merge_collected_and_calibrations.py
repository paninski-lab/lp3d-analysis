#!/usr/bin/env python3
"""Merge CollectedData_* + CollectedData_*_new and calibrations + calibrations_new without touching the source data dir.

Lightning Pose multiview expects:
  - One label CSV per view, same row count, row i aligned across views (same frame basename).
  - camera_params_file rows matching label order (see MultiviewHeatmapDataset assertion on basenames).

This script concatenates *original* rows then *_new* rows for every view and for calibrations, writing
outputs to --output-dir. Point cfg.data.data_dir at your real dataset; use absolute paths in
cfg.data.csv_file and cfg.data.camera_params_file to the merged files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from lightning_pose.utils import io as io_utils


HEADER_ROWS = [0, 1, 2]


def _read_collected(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=HEADER_ROWS, index_col=0)
    return io_utils.fix_empty_first_row(df)


def merge_collected_pair(base: Path, view: str, suffix_new: str) -> pd.DataFrame:
    a = base / f"CollectedData_{view}.csv"
    b = base / f"CollectedData_{view}{suffix_new}.csv"
    if not a.is_file():
        raise FileNotFoundError(a)
    if not b.is_file():
        raise FileNotFoundError(b)
    d1, d2 = _read_collected(a), _read_collected(b)
    if not d1.columns.equals(d2.columns):
        raise ValueError(f"Column mismatch between {a} and {b}")
    return pd.concat([d1, d2], axis=0)


def merge_calibrations_pair(base: Path, name_new: str) -> pd.DataFrame:
    a = base / "calibrations.csv"
    b = base / name_new
    if not a.is_file():
        raise FileNotFoundError(a)
    if not b.is_file():
        raise FileNotFoundError(b)
    # Match lightning_pose MultiviewHeatmapDataset
    d1 = pd.read_csv(a, index_col=0, header=[0])
    d2 = pd.read_csv(b, index_col=0, header=[0])
    if not d1.columns.equals(d2.columns):
        raise ValueError(f"Column mismatch between {a} and {b}")
    return pd.concat([d1, d2], axis=0)


def _basename(idx: object) -> str:
    return Path(str(idx)).name


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Dataset root (read-only); contains CollectedData_*, calibrations*.csv",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write merged CSVs (created if missing).",
    )
    p.add_argument(
        "--views",
        nargs="+",
        default=[f"Camera{i}" for i in range(5)],
        help="View names matching CollectedData_<view>.csv",
    )
    p.add_argument(
        "--new-suffix",
        default="_new",
        help="Suffix for second CollectedData file (default _new -> CollectedData_<view>_new.csv)",
    )
    p.add_argument(
        "--calibrations-new",
        default="calibrations_new.csv",
        help="Filename for second calibrations file",
    )
    args = p.parse_args()

    data_dir: Path = args.data_dir.resolve()
    out_dir: Path = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = args.new_suffix
    if not suffix.startswith("_"):
        suffix = "_" + suffix

    merged_views: dict[str, pd.DataFrame] = {}
    for view in args.views:
        merged_views[view] = merge_collected_pair(data_dir, view, suffix)
        n = len(merged_views[view])
        print(f"{view}: {n} rows (merged)")

    row_counts = {v: len(df) for v, df in merged_views.items()}
    if len(set(row_counts.values())) != 1:
        print("ERROR: views disagree on row counts:", row_counts, file=sys.stderr)
        sys.exit(1)

    n_total = next(iter(row_counts.values()))
    for view, df in merged_views.items():
        out_csv = out_dir / f"CollectedData_{view}.csv"
        df.to_csv(out_csv)
        print(f"Wrote {out_csv}")

    calib = merge_calibrations_pair(data_dir, args.calibrations_new)
    calib_out = out_dir / "calibrations.csv"
    calib.to_csv(calib_out)
    print(f"Wrote {calib_out} ({len(calib)} rows)")

    if len(calib) != n_total:
        print(
            f"ERROR: merged calibrations rows ({len(calib)}) != merged label rows ({n_total})",
            file=sys.stderr,
        )
        sys.exit(1)

    ref = merged_views[args.views[0]]
    for i in range(n_total):
        lb = _basename(ref.index[i])
        cb = _basename(calib.index[i])
        if lb != cb:
            print(
                f"ERROR: row {i}: label basename {lb!r} != calibrations {cb!r}",
                file=sys.stderr,
            )
            sys.exit(1)

    print("OK: row counts match and basenames align with calibrations order.")
    print()
    print("Use in your LP config (keep data_dir pointing at the dataset for images):")
    print(f'  data_dir: "{data_dir}"')
    print("  csv_file:")
    for view in args.views:
        print(f'    - "{(out_dir / f"CollectedData_{view}.csv").resolve()}"')
    print(f'  camera_params_file: "{calib_out.resolve()}"')


if __name__ == "__main__":
    main()
