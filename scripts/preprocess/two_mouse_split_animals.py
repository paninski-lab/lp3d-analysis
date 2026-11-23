#!/usr/bin/env python3
"""Split a per-animal cropped dataset into one single-animal dataset per animal.

``two-mouse_s`` holds both mice as separate rows that share a frame, with identity
carried by the session directory (``..._Camera0_black``). This partitions those
rows into ``two-mouse_s-black`` and ``two-mouse_s-white``: one row per frame, 11
keypoints, the same crops.

The point is a baseline that knows the answer for free. A per-animal model never
has to work out which mouse it is looking at, because it only ever sees one kind.
Comparing it against the joint model isolates how much the identity ambiguity
costs, and against ``two-mouse_s_sam`` shows how much of that cost recolouring
recovers.

Nothing is regenerated: the crops, boxes and label values are exactly the ones
in the source, filtered by row. The ``_black`` / ``_white`` suffix is kept on the
session directories so predictions from the two models can be recombined into
the 22-keypoint format by the same path-parsing the joint pipeline uses.

Images are symlinked per session directory rather than copied, so the split costs
no disk. Use ``--copy`` for a standalone dataset that survives the source being
deleted.

The defaults are the two-mouse ones, but nothing here is specific to it: the
animal is matched as a whole underscore-separated *token* of the session
directory, so it works equally for ``..._Camera0_black`` (two-mouse, animal last)
and ``..._an1_Camera1`` (r24m, animal before the view).

Usage::

    python scripts/preprocess/two_mouse_split_animals.py
    python scripts/preprocess/two_mouse_split_animals.py --copy
    python scripts/preprocess/two_mouse_split_animals.py \
        --source /teamspace/studios/this_studio/data/r24m_s \
        --animals an1 an2 \
        --views Camera1 Camera2 Camera3 Camera5 Camera6
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path, PurePosixPath

import pandas as pd

DEFAULT_SOURCE = Path("/teamspace/studios/this_studio/data/two-mouse_s")
VIEWS = ["Camera0", "Camera1", "Camera2", "Camera3", "Camera4"]
ANIMALS = ["black", "white"]
SPLITS = {"ind": "", "ood": "_new"}
HEADER_ROWS = [0, 1, 2]


def has_animal(dir_name: str, animal: str) -> bool:
    """Whether a session directory belongs to ``animal``.

    Matched as a whole underscore-separated token rather than a suffix, because
    datasets disagree on where the animal sits in the name -- two-mouse puts it
    last, r24m puts it before the view. Token matching is also stricter than a
    substring test, which would let a session named ``..._an10`` match ``an1``.
    """
    return animal in dir_name.split("_")


def rows_for_animal(index, animal: str) -> list[bool]:
    """Row mask by session directory, the same cue the joint pipeline uses."""
    return [has_animal(PurePosixPath(p).parent.name, animal) for p in index]


def build_animal(
    source: Path, out_dir: Path, animal: str, copy: bool, views: list[str],
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "videos").mkdir(exist_ok=True)

    stats = {"rows": {}, "sessions": 0}

    for split, suffix in SPLITS.items():
        counts = set()
        for view in views:
            labels = pd.read_csv(
                source / f"CollectedData_{view}{suffix}.csv", header=HEADER_ROWS, index_col=0,
            )
            boxes = pd.read_csv(source / f"bboxes_{view}{suffix}.csv", index_col=0)

            keep = rows_for_animal(labels.index, animal)
            labels = labels[keep]
            # Reindexed by label order rather than filtered independently, so the
            # two files cannot drift apart even if their row orders ever differ.
            boxes = boxes.loc[labels.index]

            labels.to_csv(out_dir / f"CollectedData_{view}{suffix}.csv")
            boxes.to_csv(out_dir / f"bboxes_{view}{suffix}.csv")
            counts.add(len(labels))

        cal = pd.read_csv(source / f"calibrations{suffix}.csv", index_col=0)
        cal = cal[rows_for_animal(cal.index, animal)]
        cal.to_csv(out_dir / f"calibrations{suffix}.csv")

        if len(counts) != 1:
            raise AssertionError(f"views disagree on row count for {animal}{suffix}: {counts}")
        n = counts.pop()
        if len(cal) != n:
            raise AssertionError(
                f"calibrations{suffix}.csv has {len(cal)} rows but labels have {n}; "
                f"calibration is matched positionally, so a mismatch would pair "
                f"frames with the wrong camera parameters."
            )
        stats["rows"][split] = n

    link = out_dir / "calibrations"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to((source / "calibrations").resolve())

    dst_root = out_dir / "labeled-data"
    dst_root.mkdir(exist_ok=True)
    for session in sorted((source / "labeled-data").iterdir()):
        if not has_animal(session.name, animal):
            continue
        dst = dst_root / session.name
        if dst.is_symlink() or dst.exists():
            shutil.rmtree(dst) if dst.is_dir() and not dst.is_symlink() else dst.unlink()
        if copy:
            shutil.copytree(session, dst)
        else:
            dst.symlink_to(session.resolve(), target_is_directory=True)
        stats["sessions"] += 1

    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--animals", nargs="+", default=ANIMALS)
    p.add_argument("--views", nargs="+", default=VIEWS)
    p.add_argument("--copy", action="store_true",
                   help="copy image directories instead of symlinking them")
    args = p.parse_args()

    for animal in args.animals:
        out_dir = args.source.parent / f"{args.source.name}-{animal}"
        stats = build_animal(args.source, out_dir, animal, args.copy, args.views)
        with open(out_dir / "provenance.json", "w") as f:
            json.dump({
                "source": str(args.source), "animal": animal,
                "images": "copied" if args.copy else "symlinked",
                "views": args.views, **stats,
            }, f, indent=2)
        print(f"{out_dir.name}: {stats['rows']} rows, {stats['sessions']} session dirs")


if __name__ == "__main__":
    main()
