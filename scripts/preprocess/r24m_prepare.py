#!/usr/bin/env python3
"""Organise the delivered r24m archives into LP3D-trainable datasets.

Two archives arrive under ``/teamspace/lightning_storage/rat-data``:

``r24m.tar.xz``
    Full 1328x1048 frames, one row per frame, both rats in the keypoint names
    (``HeadF_an1`` / ``HeadF_an2``): 24 keypoints, 261 in-distribution frames and
    234 out-of-distribution. This is the *combined* arrangement, the one the
    original two-mouse model was trained in.
``r24m-crop.tar.xz``
    The same frames already cropped to 320x320 per animal, one row per
    (frame, animal): 12 keypoints, 522 / 468 rows, with ``bboxes_<view>.csv``
    giving each box in original full-frame pixels.

Both are almost trainable as delivered. What they are missing is
``calibrations.csv`` -- the per-row lookup from a label row to the camera
calibration that applies to it. Lightning Pose matches it **positionally**
against the label rows, so building it is the one place this script can silently
corrupt a dataset, and the one place it checks itself hardest.

This script does not crop, does not move keypoints, and does not touch pixels. It
lays out directories, copies the label and box csvs verbatim, writes
``calibrations.csv``, and verifies the result.

Usage::

    python scripts/preprocess/r24m_prepare.py                 # both datasets
    python scripts/preprocess/r24m_prepare.py --which crop    # just r24m_s
    python scripts/preprocess/r24m_prepare.py --images copy   # don't consume staging
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path, PurePosixPath

import pandas as pd

from r24m_layout import (
    ANIMALS, HEADER_ROWS, SPLITS, VIEWS, parse_crop_dir, parse_full_dir, session_of,
)

DEFAULT_CROP_SOURCE = Path("/teamspace/studios/this_studio/data_staging/r24m-crop")
DEFAULT_FULL_SOURCE = Path("/teamspace/studios/this_studio/data_staging/r24m-full")
DEFAULT_DATA_ROOT = Path("/teamspace/studios/this_studio/data")


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# calibrations.csv
# ---------------------------------------------------------------------------

def calibration_rows(index, cropped: bool) -> pd.DataFrame:
    """Per-row calibration lookup, in label-row order.

    The index keeps the animal (a cropped row belongs to one animal) but drops
    the camera: a row is one *scene* observed from every view at once, so naming
    one camera in it would be arbitrary. ``two-mouse_s`` uses the same form.

    Lightning Pose compares only the file basenames, positionally, but staying
    close to the label path makes a misalignment visible by eye rather than
    only by assertion.
    """
    rows = []
    for rel in index:
        p = PurePosixPath(rel)
        if cropped:
            session, animal, _view = parse_crop_dir(p.parent.name)
            stem = f"{session}_{animal}"
            toml = f"calibrations/{session}_{animal}.toml"
        else:
            session = session_of(p.parent.name)
            stem = session
            toml = f"calibrations/{session}.toml"
        rows.append((f"labeled-data/{stem}/{p.name}", toml))
    df = pd.DataFrame(rows, columns=["", "file"]).set_index("")
    df.index.name = None
    return df


def verify(out_dir: Path, cropped: bool) -> dict:
    """Re-read what was written and check every invariant training depends on."""
    stats: dict = {}
    for split, suffix in SPLITS.items():
        counts, orders = set(), set()
        for view in VIEWS:
            labels = pd.read_csv(
                out_dir / f"CollectedData_{view}{suffix}.csv",
                header=HEADER_ROWS, index_col=0,
            )
            counts.add(len(labels))
            # Views must agree row-for-row on which (frame, animal) each row is;
            # the camera name is the only part allowed to differ.
            key = []
            for r in labels.index:
                p = PurePosixPath(r)
                if cropped:
                    session, animal, v = parse_crop_dir(p.parent.name)
                else:
                    session, v = parse_full_dir(p.parent.name)
                    animal = None
                if v != view:
                    raise AssertionError(
                        f"row {r} sits in CollectedData_{view}{suffix}.csv but names "
                        f"view {v}"
                    )
                key.append((session, animal, p.name))
            orders.add(tuple(key))

            for rel in labels.index:
                if not (out_dir / rel).exists():
                    raise AssertionError(f"missing image {out_dir / rel}")

            if cropped:
                boxes = pd.read_csv(
                    out_dir / f"bboxes_{view}{suffix}.csv", index_col=0)
                if list(boxes.index) != list(labels.index):
                    raise AssertionError(
                        f"bboxes_{view}{suffix}.csv is not row-aligned with its labels"
                    )

        if len(counts) != 1:
            raise AssertionError(f"views disagree on row count for {split}: {counts}")
        if len(orders) != 1:
            raise AssertionError(f"views disagree on row order for {split}")

        cal = pd.read_csv(out_dir / f"calibrations{suffix}.csv", index_col=0)
        n = counts.pop()
        if len(cal) != n:
            raise AssertionError(
                f"calibrations{suffix}.csv has {len(cal)} rows but labels have {n}"
            )
        # The assertion lightning_pose itself makes, made here where the fix is.
        ref = pd.read_csv(
            out_dir / f"CollectedData_{VIEWS[0]}{suffix}.csv",
            header=HEADER_ROWS, index_col=0,
        )
        lab_names = [PurePosixPath(i).name for i in ref.index]
        cal_names = [PurePosixPath(i).name for i in cal.index]
        if lab_names != cal_names:
            raise AssertionError(
                f"calibrations{suffix}.csv is not positionally aligned with the labels"
            )
        for f in cal.file.unique():
            if not (out_dir / f).exists():
                raise AssertionError(f"calibration {f} referenced but not present")

        stats[split] = {
            "n_rows": n,
            "n_frames": len({
                (session_of(PurePosixPath(i).parent.name), PurePosixPath(i).name)
                for i in ref.index
            }),
            "n_calibrations": int(cal.file.nunique()),
        }
    return stats


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def materialise(src: Path, dst: Path, mode: str) -> None:
    if dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        shutil.rmtree(dst)
    if mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "copy":
        shutil.copytree(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve(), target_is_directory=True)
    else:
        raise ValueError(mode)


def build(source: Path, out_dir: Path, cropped: bool, images: str) -> dict:
    if not source.exists():
        raise FileNotFoundError(source)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "videos").mkdir(exist_ok=True)

    for split, suffix in SPLITS.items():
        for view in VIEWS:
            shutil.copy(source / f"CollectedData_{view}{suffix}.csv",
                        out_dir / f"CollectedData_{view}{suffix}.csv")
            if cropped:
                shutil.copy(source / f"bboxes_{view}{suffix}.csv",
                            out_dir / f"bboxes_{view}{suffix}.csv")

        ref = pd.read_csv(source / f"CollectedData_{VIEWS[0]}{suffix}.csv",
                          header=HEADER_ROWS, index_col=0)
        calibration_rows(ref.index, cropped).to_csv(
            out_dir / f"calibrations{suffix}.csv")

    materialise(source / "calibrations", out_dir / "calibrations", images)
    materialise(source / "labeled-data", out_dir / "labeled-data", images)

    return verify(out_dir, cropped)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crop-source", type=Path, default=DEFAULT_CROP_SOURCE)
    p.add_argument("--full-source", type=Path, default=DEFAULT_FULL_SOURCE)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--which", choices=["both", "full", "crop"], default="both")
    p.add_argument("--images", choices=["move", "copy", "symlink"], default="move",
                   help="how to place labeled-data; 'move' consumes the staging copy")
    args = p.parse_args()

    targets = []
    if args.which in ("both", "full"):
        targets.append(("r24m", args.full_source, False))
    if args.which in ("both", "crop"):
        targets.append(("r24m_s", args.crop_source, True))

    for name, source, cropped in targets:
        out_dir = args.data_root / name
        print(f"=== {name} <- {source} ===", flush=True)
        stats = build(source, out_dir, cropped, args.images)
        provenance = {
            "source": str(source),
            "arrangement": "per-animal crop" if cropped else "combined full frame",
            "cropped": cropped,
            "views": VIEWS,
            "animals": ANIMALS,
            "images": args.images,
            "lp3d_analysis_sha": git_sha(),
            "splits": stats,
        }
        if cropped:
            # two_mouse_sam.py reads output_size from here.
            provenance["output_size"] = 320
            provenance["crop_ratio"] = 1.4
            provenance["crop_note"] = (
                "crops delivered pre-made; ratio 1.4 and size 320 measured from the "
                "shipped boxes and labels, not chosen here"
            )
        with open(out_dir / "provenance.json", "w") as f:
            json.dump(provenance, f, indent=2)
        for split, s in stats.items():
            print(f"  {split}: {s['n_rows']} rows, {s['n_frames']} frames, "
                  f"{s['n_calibrations']} calibrations", flush=True)
        print(f"  wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
