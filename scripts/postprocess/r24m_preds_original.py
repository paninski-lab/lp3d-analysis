#!/usr/bin/env python3
"""Give every finished r24m run a ``preds_original/`` in original full-frame space.

Six of the seven r24m conditions train on 320x320 crops, so their predictions and
Lightning Pose's own pixel-error files are in **stored-crop pixels** -- a unit
that is worth ``w / 320`` full-frame pixels and therefore varies per row by a
factor of ~10 across this dataset. Nothing in crop space is comparable across
rows, across views, or against the one condition that trains on full frames.

This puts every condition on the same footing. After it runs, each finished
model has::

    <run_dir>/predictions_<view>_new.csv                 crop pixels, per-animal rows
    <run_dir>/preds_original/predictions_<view>_new.csv  full-frame px, one row per frame

with **the same filenames** in both places -- predictions, pixel errors and
ground-truth ``CollectedData_<view>.csv`` alike. Anything that reads a model's
run directory can be pointed at ``preds_original/`` instead and keep working.

The three shapes it handles:

``native``   r24m -- trains on full frames, so its predictions are already in the
             target format. Checked and copied, not transformed.
``joint``    r24m_s, r24m_s_sam -- one model, two rows per frame. Merged.
``pair``     r24m_s-an1 + r24m_s-an2 (and the SAM pair) -- two independent
             single-animal models whose rows are disjoint. Merged together into
             one ``<condition>-combined`` run directory, because the thing worth
             comparing against the joint models is the pair *recombined*, not
             either half alone.

Safe to re-run: conditions whose inputs are not all COMPLETED are skipped.

Usage::

    python scripts/postprocess/r24m_preds_original.py
    python scripts/postprocess/r24m_preds_original.py --seed 1
    python scripts/postprocess/r24m_preds_original.py --only r24m_s
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path("/teamspace/studios/this_studio")
REPO = ROOT / "lp3d-analysis"
VIEWS = ["Camera1", "Camera2", "Camera3", "Camera5", "Camera6"]
ANIMALS = ["an1", "an2"]
SUFFIXES = ["", "_new"]

# condition -> (kind, the dataset(s) whose runs feed it)
CONDITIONS: dict[str, tuple[str, list[str]]] = {
    "r24m": ("native", ["r24m"]),
    "r24m_s": ("joint", ["r24m_s"]),
    "r24m_s_sam": ("joint", ["r24m_s_sam"]),
    "r24m_s-combined": ("pair", ["r24m_s-an1", "r24m_s-an2"]),
    "r24m_s_sam-combined": ("pair", ["r24m_s_sam-an1", "r24m_s_sam-an2"]),
}


def run_dir(dataset: str, seed: int) -> Path:
    """Where pipeline_simple.py put a run, read from that run's own pipeline config."""
    cfg = yaml.safe_load(open(REPO / "configs" / f"pipeline_{dataset}.yaml"))
    rows = cfg["train_networks"]["n_hand_labels"][0]
    return (ROOT / "outputs" / dataset / cfg["intermediate_results_dir"]
            / f"mvt_3d_loss_{rows}_{seed}")


def completed(d: Path) -> bool:
    f = d / "train_status.json"
    try:
        return json.load(open(f)).get("status") == "COMPLETED"
    except Exception:
        return False


def canonical_bodyparts() -> list[str]:
    df = pd.read_csv(ROOT / "data" / "r24m" / f"CollectedData_{VIEWS[0]}.csv",
                     header=[0, 1, 2], index_col=0)
    return list(dict.fromkeys(df.columns.get_level_values("bodyparts")))


def copy_native(src: Path, out_dir: Path) -> int:
    """r24m predictions are already what the merge would produce. Verify, then copy.

    Verified rather than assumed: the file must carry the same 24 bodyparts in the
    same order as the merged outputs, one row per frame, and its coordinates must
    land inside the full frame. A silently different column order here would put
    an1's keypoints under an2's names in every downstream comparison.
    """
    want = canonical_bodyparts()
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for suffix in SUFFIXES:
        pred = src / f"predictions_{VIEWS[0]}{suffix}.csv"
        if not pred.exists():
            continue
        for view in VIEWS:
            p = src / f"predictions_{view}{suffix}.csv"
            df = pd.read_csv(p, header=[0, 1, 2], index_col=0)
            got = [b for s, b, c in df.columns if str(s) != "set" and c == "x"]
            if got != want:
                raise AssertionError(
                    f"{p} bodyparts differ from the merged layout:\n"
                    f"  expected {want}\n  got      {got}"
                )
            xy = df.loc[:, [c for c in df.columns
                            if str(c[0]) != "set" and c[2] in ("x", "y")]].to_numpy(float)
            if not ((xy >= -50).all() and (xy[:, 0::2] <= 1400).all()
                    and (xy[:, 1::2] <= 1100).all()):
                raise AssertionError(
                    f"{p} has coordinates outside the 1328x1048 frame; these are "
                    f"not full-frame pixels"
                )
            for name in (f"predictions_{view}{suffix}.csv",
                         f"predictions_{view}{suffix}_pixel_error.csv"):
                if (src / name).exists():
                    shutil.copy(src / name, out_dir / name)
                    n += 1
    return n


def copy_ground_truth(out_dir: Path) -> int:
    """Put the uncropped labels beside the merged predictions.

    This makes ``preds_original/`` a complete drop-in for the run directory,
    which also holds its own ``CollectedData_<view>.csv``. No regeneration is
    involved: merging the crop labels back reproduces ``data/r24m`` at exactly
    0.000000 px, so the source file *is* the merged ground truth, and the merged
    predictions are row- and column-aligned with it.
    """
    n = 0
    for view in VIEWS:
        for suffix in SUFFIXES:
            src = ROOT / "data" / "r24m" / f"CollectedData_{view}{suffix}.csv"
            if src.exists():
                shutil.copy(src, out_dir / f"CollectedData_{view}{suffix}.csv")
                n += 1
    return n


def merge(model_dirs: list[Path], out_dir: Path) -> None:
    cmd = [
        sys.executable, str(REPO / "scripts/postprocess/two_mouse_merge_predictions.py"),
        *[str(d) for d in model_dirs],
        "--data-dir", str(ROOT / "data" / "r24m_s"),
        "--source", str(ROOT / "data" / "r24m"),
        "--out-dir", str(out_dir),
        "--animals", *ANIMALS,
        "--views", *VIEWS,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write("".join(f"    {l}\n" for l in r.stdout.strip().splitlines()[-3:]))
    if r.returncode != 0:
        sys.stdout.write(r.stdout[-2000:])
        sys.stderr.write(r.stderr[-2000:])
        raise RuntimeError(f"merge failed for {[d.name for d in model_dirs]}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", default=None, help="one condition name")
    p.add_argument("--force", action="store_true",
                   help="rebuild even if preds_original already exists")
    args = p.parse_args()

    conditions = ({args.only: CONDITIONS[args.only]} if args.only else CONDITIONS)
    made, skipped = 0, []

    for name, (kind, datasets) in conditions.items():
        dirs = [run_dir(d, args.seed) for d in datasets]
        not_done = [name_ for name_, ok in zip(datasets, map(completed, dirs)) if not ok]
        if not_done:
            skipped.append(f"{name}: waiting on {', '.join(not_done)}")
            continue

        if kind == "pair":
            # The pair's combined result is its own condition, so it gets its own
            # run directory rather than being duplicated into both halves'.
            cfg = yaml.safe_load(open(REPO / "configs" / f"pipeline_{datasets[0]}.yaml"))
            out = (ROOT / "outputs" / name / cfg["intermediate_results_dir"]
                   / f"mvt_3d_loss_{cfg['train_networks']['n_hand_labels'][0]}_{args.seed}"
                   / "preds_original")
        else:
            out = dirs[0] / "preds_original"

        if out.exists() and not args.force:
            skipped.append(f"{name}: preds_original already present")
            continue

        print(f"  {name} ({kind}) -> {out}")
        if kind == "native":
            n = copy_native(dirs[0], out)
            print(f"    copied {n} files (already full-frame; layout verified)")
        else:
            merge(dirs, out)
        g = copy_ground_truth(out)
        print(f"    + {g} ground-truth CollectedData files")
        made += 1

    print(f"\nbuilt {made} condition(s)")
    for s in skipped:
        print(f"  skipped {s}")


if __name__ == "__main__":
    main()
