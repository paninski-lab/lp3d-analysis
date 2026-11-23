#!/usr/bin/env python3
"""Put predictions from another method into the same layout as our LP3D results.

Baselines trained outside Lightning Pose (sDANNCE, DANNCE, ...) arrive with
their own filenames, e.g. ``predictions_sdannce_scratch_Camera1_new.csv``. This
writes them into ``<run_dir>/preds_original/`` next to the originals, with
exactly the names and companions every LP3D condition has, so the analysis
notebook can point at them unchanged::

    preds_original/predictions_<view>_new.csv              the predictions
    preds_original/predictions_<view>_new_pixel_error.csv  per-keypoint error
    preds_original/CollectedData_<view>{,_new}.csv         the ground truth

The input must already be in **full-frame pixels**, one row per frame, with the
combined ``<keypoint>_<animal>`` columns. Nothing is transformed. Before writing,
each view is checked against the ground truth: same frames, same order, same
keypoint columns, coordinates inside the frame, and each file scoring best
against its *own* camera's labels (a swapped view file matches another camera
better). Anything that fails stops the run.

Two things are deliberately **not** invented. External predictions carry no
``likelihood`` and no train/validation ``set``; the analysis loader treats both
as optional, and a made-up confidence or split would look like real data.

The scorer is renamed to the method name. sDANNCE writes ``mocap`` -- the ground
truth's scorer -- which makes a prediction file indistinguishable from a label
file at a glance.

Usage::

    python scripts/postprocess/format_external_predictions.py \\
        outputs/r24m_sdannce_scratch_1 \\
        --pattern 'predictions_sdannce_scratch_{view}{suffix}.csv' \\
        --scorer sdannce_scratch
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

ROOT = Path("/teamspace/studios/this_studio")
VIEWS = ["Camera1", "Camera2", "Camera3", "Camera5", "Camera6"]
FRAME_W, FRAME_H = 1328, 1048


def frame_key(rel: str) -> str:
    """Frame identity with the camera removed, so rows can be matched across views."""
    p = PurePosixPath(rel)
    session = "_".join(t for t in p.parent.name.split("_") if t not in VIEWS)
    return f"{session}/{p.name}"


def xy(df: pd.DataFrame, bodyparts: list[str]) -> np.ndarray:
    """(n_rows, n_keypoints, 2), in ``bodyparts`` order, regardless of scorer."""
    sc = df.columns.get_level_values(0)[0]
    return np.stack(
        [np.stack([df[(sc, b, "x")].to_numpy(float), df[(sc, b, "y")].to_numpy(float)], -1)
         for b in bodyparts],
        axis=1,
    )


def pixel_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Euclidean error per row and keypoint. NaN where either side is unlabelled."""
    return np.linalg.norm(pred - gt, axis=-1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--pattern", required=True,
                   help="input filename with {view} and {suffix} placeholders")
    p.add_argument("--scorer", required=True, help="scorer name for the output files")
    p.add_argument("--gt-dir", type=Path, default=ROOT / "data" / "r24m")
    p.add_argument("--views", nargs="+", default=VIEWS)
    p.add_argument("--suffixes", nargs="+", default=["_new", ""],
                   help="splits to look for; missing ones are skipped")
    args = p.parse_args()

    out = args.run_dir / "preds_original"
    problems: list[str] = []
    found: dict[str, dict[str, pd.DataFrame]] = {}

    for suffix in args.suffixes:
        files = {v: args.run_dir / args.pattern.format(view=v, suffix=suffix) for v in args.views}
        present = [v for v, f in files.items() if f.exists()]
        if not present:
            continue
        if len(present) != len(args.views):
            problems.append(f"split {suffix or 'ind'}: only {present} present")
            continue
        found[suffix] = {v: pd.read_csv(f, header=[0, 1, 2], index_col=0) for v, f in files.items()}

    if not found:
        sys.exit(f"no files matching {args.pattern} in {args.run_dir}")

    for suffix, preds in found.items():
        split = "OOD" if suffix == "_new" else "InD"
        gts = {v: pd.read_csv(args.gt_dir / f"CollectedData_{v}{suffix}.csv",
                              header=[0, 1, 2], index_col=0) for v in args.views}
        bodyparts = list(dict.fromkeys(gts[args.views[0]].columns.get_level_values(1)))
        print(f"=== {split}: {len(args.views)} views ===")

        # --- checks: all of them run before anything is written -------------
        for v in args.views:
            pr, gt = preds[v], gts[v]
            got = list(dict.fromkeys(pr.columns.get_level_values(1)))
            if got != bodyparts:
                problems.append(f"{v}{suffix}: keypoint columns differ from ground truth")
                continue
            if set(pr.columns.get_level_values(2)) - {"x", "y", "likelihood"}:
                problems.append(f"{v}{suffix}: unexpected coordinate columns")
            if list(pr.index) != list(gt.index):
                problems.append(f"{v}{suffix}: rows are not the ground-truth frames in "
                                f"ground-truth order ({len(pr)} vs {len(gt)})")
                continue
            a = xy(pr, bodyparts)
            if not (np.nanmin(a) > -FRAME_W and np.nanmax(a[..., 0]) < 2 * FRAME_W
                    and np.nanmax(a[..., 1]) < 2 * FRAME_H):
                problems.append(f"{v}{suffix}: coordinates are not full-frame pixels")

        # each view file must match its own camera best
        keys = [frame_key(i) for i in gts[args.views[0]].index]
        for v in args.views:
            if [frame_key(i) for i in preds[v].index] != keys:
                problems.append(f"{v}{suffix}: frames not aligned with other views")
        if not problems:
            scores = np.array([[np.nanmean(pixel_error(xy(preds[pv], bodyparts),
                                                       xy(gts[gv], bodyparts)))
                                for gv in args.views] for pv in args.views])
            for i, v in enumerate(args.views):
                if scores[i].argmin() != i:
                    problems.append(f"{v}{suffix}: matches {args.views[scores[i].argmin()]}'s "
                                    f"labels better than its own -- view files swapped?")
            margin = np.min([np.sort(r)[1] / r[i] for i, r in enumerate(scores)])
            print(f"  views: each file matches its own camera; nearest other camera is "
                  f"{margin:.1f}x worse")

        if problems:
            continue

        # --- write -----------------------------------------------------------
        out.mkdir(parents=True, exist_ok=True)
        for v in args.views:
            pr, gt = preds[v].copy(), gts[v]
            pr.columns = pd.MultiIndex.from_tuples(
                [(args.scorer, b, c) for _, b, c in pr.columns],
                names=["scorer", "bodyparts", "coords"])
            pr.to_csv(out / f"predictions_{v}{suffix}.csv")

            err = pixel_error(xy(pr, bodyparts), xy(gt, bodyparts))
            err_df = pd.DataFrame(err, index=pr.index, columns=bodyparts)
            err_df.to_csv(out / f"predictions_{v}{suffix}_pixel_error.csv")
            print(f"  {v}: {len(pr)} rows, {len(bodyparts)} kps -> "
                  f"pixel error mean {np.nanmean(err):.2f} px, median {np.nanmedian(err):.2f} px")

    if problems:
        print("\nNOT WRITTEN -- fix these first:")
        for m in problems:
            print(f"  - {m}")
        sys.exit(1)

    for v in args.views:
        for suffix in ("", "_new"):
            src = args.gt_dir / f"CollectedData_{v}{suffix}.csv"
            if src.exists():
                shutil.copy(src, out / src.name)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
