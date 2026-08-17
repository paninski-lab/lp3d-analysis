#!/usr/bin/env python3
"""Build a per-animal cropped dataset from the combined two-mouse dataset.

The crim13 arrangement, extended to multiple views. The source dataset labels
both mice in one row, identity carried by the keypoint names (``Nose_black`` /
``Nose_white``). Here each animal instead gets its own row, its own crop, and its
own session directory, with identity stripped from the keypoint names::

    labeled-data/CSDS-Day1-B_1-Defeat_Camera0_black/img00001256.jpg
    labeled-data/CSDS-Day1-B_1-Defeat_Camera0_white/img00001256.jpg

The crop for an animal in a view comes from that animal's own labelled keypoints
*in that view*: centred on the midpoint of their extent, sized from that extent
times ``--crop-ratio``. Nothing 3D is involved in choosing a box -- no
triangulation, no calibration, no centroid model.

**No labelled keypoint is ever cropped out.** Containment is structural, not a
matter of picking a generous ratio: see :func:`keypoints_to_bbox`. Generation
also asserts it per crop, so a regression fails the build rather than quietly
producing labels that sit outside their own image and cannot be learned.

Because the box is sized from the keypoints it contains, it breathes with posture:
a stretched animal gets a larger box than a curled one, so apparent scale varies
between frames. That is inherent to keypoint-extent boxes.

Beyond crim13, a multiview dataset also needs, per view:

``bboxes_<view>.csv``
    Each box in original full-frame pixels. Lightning Pose recombines this with
    the crop at load time, which is what lets a 3D loss work in full-frame
    coordinates off cropped images. Single-view crim13 never needed it.
``calibrations.csv``
    One row per label row, matched positionally.

Row order is frame-major and animal-minor, identical across every view and the
calibration index, since calibration lookup is positional.

Usage::

    python scripts/preprocess/two_mouse_crop.py
    python scripts/preprocess/two_mouse_crop.py --crop-ratio 1.5 --output ..._loose
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import pandas as pd

from lightning_pose.utils import io as lp_io

DEFAULT_SOURCE = Path("/tmp/data/two-mouse")
DEFAULT_OUTPUT = Path("/teamspace/studios/this_studio/data/two-mouse_s")
VIEWS = ["Camera0", "Camera1", "Camera2", "Camera3", "Camera4"]
ANIMALS = ["black", "white"]

DEFAULT_CROP_RATIO = 1.3
# Stored crop resolution, which is deliberately larger than the model input
# (cfg.data.image_resize_dims, 256). chickadee-crop does the same: storing above
# the model input leaves augmentation more pixels to work with and lets the input
# size be raised later without regenerating. It also matches the 320 that
# lp3d_analysis/train.py assumes when remapping video predictions out of crops.
DEFAULT_OUTPUT_SIZE = 320
HEADER_ROWS = [0, 1, 2]

# In-distribution is the unsuffixed set, out-of-distribution the "_new" set,
# following the chickadee-crop convention. Kept separate rather than merged.
SPLITS = {"ind": "", "ood": "_new"}


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------

def read_label_csv(path: Path) -> pd.DataFrame:
    """Read a label csv, repairing the dropped all-NaN first row.

    A frame whose keypoints are entirely unlabelled looks to pandas exactly like
    an index-name row, so it is silently dropped -- which would shift every row's
    alignment against the calibration index.
    """
    df = pd.read_csv(path, header=HEADER_ROWS, index_col=0)
    return lp_io.fix_empty_first_row(df)


def split_animals(df: pd.DataFrame, animals: list[str]) -> tuple[dict[str, np.ndarray], list[str]]:
    """Split a combined csv into per-animal keypoint arrays with stems.

    Returns:
        ({animal: (n_frames, n_keypoints, 2)}, keypoint stems)

    """
    bodyparts = list(dict.fromkeys(df.columns.get_level_values("bodyparts")))

    per_animal_names = {
        a: [b for b in bodyparts if b.endswith(f"_{a}")] for a in animals
    }
    stems = [b[: -len(f"_{animals[0]}")] for b in per_animal_names[animals[0]]]
    for a in animals[1:]:
        other = [b[: -len(f"_{a}")] for b in per_animal_names[a]]
        if other != stems:
            raise ValueError(
                f"Animals must share the same keypoints in the same order; "
                f"'{animals[0]}' has {stems} but '{a}' has {other}."
            )

    out = {}
    for a in animals:
        cols = [
            (scorer, bp, coord)
            for bp in per_animal_names[a]
            for scorer, b, coord in df.columns
            if b == bp and coord in ("x", "y")
        ]
        # Preserve x,y pairing per keypoint in `stems` order.
        ordered = []
        for bp in per_animal_names[a]:
            for coord in ("x", "y"):
                ordered.append(next(c for c in cols if c[1] == bp and c[2] == coord))
        out[a] = df.loc[:, ordered].to_numpy(dtype=float).reshape(len(df), len(stems), 2)

    return out, stems


# ---------------------------------------------------------------------------
# boxes
# ---------------------------------------------------------------------------

def keypoints_to_bbox(
    keypoints: np.ndarray, crop_ratio: float, min_size: int = 32,
) -> np.ndarray:
    """Box around a single animal's keypoints in one view, guaranteed to contain them.

    Side length follows ``cropzoom._calculate_bbox_size`` -- the larger of the x and
    y extents times ``crop_ratio`` -- but the box is centred on the **midpoint** of
    the keypoint extent rather than their mean. That distinction is what makes
    containment a guarantee: the side is at least each individual extent, so a box
    centred on the midpoint always reaches both ends. Centring on the mean does not,
    because the mean sits wherever the keypoints happen to cluster, and an
    asymmetric pose then pushes the far keypoint past the edge.

    Integer rounding is handled explicitly rather than assumed away. Flooring the
    top-left corner can only move the box up and left, which cannot uncover the
    low end; the side is then grown until it strictly covers the high end, with a
    pixel to spare, before being made even.

    Square boxes keep aspect ratio when the crop is resized to a square model
    input; even sides avoid trouble with video encoders downstream.

    Args:
        keypoints: (n_keypoints, 2) in original frame pixels; may contain NaN.
        crop_ratio: how much larger than the animal to crop. Must be >= 1.0, since
            below that no box sized from the extent can contain the extent.
        min_size: floor on side length, for frames with one or two labelled points
            where the extent would otherwise collapse to nothing.

    Returns:
        ``[x, y, h, w]`` with (x, y) the top-left corner, or NaN if the animal has
        no labelled keypoint in this view.

    """
    if crop_ratio < 1.0:
        raise ValueError(
            f"crop_ratio must be >= 1.0 to contain the keypoints, got {crop_ratio}"
        )

    valid = keypoints[~np.isnan(keypoints).any(axis=1)]
    if len(valid) == 0:
        return np.full(4, np.nan)

    lo, hi = valid.min(axis=0), valid.max(axis=0)
    side = max(float((hi - lo).max()) * crop_ratio, float(min_size))

    # Midpoint of the extent, so the box reaches both ends of both axes.
    center = (lo + hi) / 2.0
    x0 = float(np.floor(center[0] - side / 2.0))
    y0 = float(np.floor(center[1] - side / 2.0))

    # Grow until the far edge strictly clears the furthest keypoint. Growing only
    # extends down and right, so the low end stays covered.
    side = max(side, hi[0] - x0 + 1.0, hi[1] - y0 + 1.0)
    side = int(np.ceil(side))
    if side % 2:
        side += 1

    return np.array([x0, y0, float(side), float(side)])


def crop_and_resize(image: np.ndarray, bbox: np.ndarray, size: int) -> np.ndarray:
    """Crop ``bbox`` out of ``image`` and resize, zero-padding beyond the frame.

    Padding rather than clamping: a clamped box would shift the animal off centre
    and change its scale, breaking the correspondence between the box recorded in
    the bbox csv and what the crop actually shows.
    """
    x, y, h, w = (int(round(float(v))) for v in bbox)
    crop = np.zeros((h, w, image.shape[2]), dtype=image.dtype)

    src_x0, src_x1 = max(0, x), min(image.shape[1], x + w)
    src_y0, src_y1 = max(0, y), min(image.shape[0], y + h)
    if src_x1 > src_x0 and src_y1 > src_y0:
        dst_x0, dst_y0 = src_x0 - x, src_y0 - y
        crop[dst_y0:dst_y0 + (src_y1 - src_y0), dst_x0:dst_x0 + (src_x1 - src_x0)] = (
            image[src_y0:src_y1, src_x0:src_x1]
        )

    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def keypoints_to_crop_coords(
    keypoints: np.ndarray, bbox: np.ndarray, size: int,
) -> np.ndarray:
    """Map keypoints from original frame pixels into resized-crop pixels.

    Not clipped to the crop: a keypoint just outside the box should read as out of
    range rather than be silently pulled onto the edge, where it would look like a
    valid label.
    """
    x, y, h, w = (float(v) for v in bbox)
    out = keypoints.copy().astype(float)
    out[:, 0] = (out[:, 0] - x) / w * size
    out[:, 1] = (out[:, 1] - y) / h * size
    return out


def instance_path(img_path: str, animal: str) -> str:
    """``labeled-data/sess_Camera0/img.jpg`` -> ``labeled-data/sess_Camera0_black/img.jpg``.

    The animal goes in the directory, never the filename, so that anything parsing
    a frame index out of the filename keeps working.
    """
    p = PurePosixPath(img_path)
    return str(p.parent.with_name(f"{p.parent.name}_{animal}") / p.name)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build_split(
    source: Path,
    out_dir: Path,
    suffix: str,
    animals: list[str],
    crop_ratio: float,
    size: int,
    qc_dir: Path | None,
    qc_count: int,
) -> dict:
    dfs = {v: read_label_csv(source / f"CollectedData_{v}{suffix}.csv") for v in VIEWS}
    n = len(dfs[VIEWS[0]])

    for view, df in dfs.items():
        if len(df) != n:
            raise ValueError(f"View '{view}' has {len(df)} rows, '{VIEWS[0]}' has {n}")
        if [Path(p).name for p in df.index] != [Path(p).name for p in dfs[VIEWS[0]].index]:
            raise ValueError(f"View '{view}' filenames do not align with {VIEWS[0]}")

    cal = pd.read_csv(source / f"calibrations{suffix}.csv", index_col=0)
    if len(cal) != n:
        raise ValueError(
            f"calibrations{suffix}.csv has {len(cal)} rows but the labels have {n}; "
            f"calibration is matched positionally, so a mismatch would pair frames "
            f"with the wrong camera parameters."
        )

    per_view = {v: split_animals(dfs[v], animals) for v in VIEWS}
    stems = per_view[VIEWS[0]][1]
    n_kp = len(stems)

    # boxes[view][animal] -> (n_frames, 4)
    boxes = {
        v: {
            a: np.stack([
                keypoints_to_bbox(per_view[v][0][a][i], crop_ratio) for i in range(n)
            ])
            for a in animals
        }
        for v in VIEWS
    }

    # A (frame, animal) row is usable only if it has a box in EVERY view: the views
    # must stay row-aligned, so a row cannot be dropped from one view alone.
    usable = {
        a: np.all([np.isfinite(boxes[v][a]).all(axis=1) for v in VIEWS], axis=0)
        for a in animals
    }

    rows: list[tuple[int, str]] = [
        (i, a) for i in range(n) for a in animals if usable[a][i]
    ]
    dropped = {a: int((~usable[a]).sum()) for a in animals}

    stats = {
        "n_frames": n,
        "n_rows": len(rows),
        "n_rows_dropped": {a: dropped[a] for a in animals},
        "bbox_px": {},
        "kp_outside_crop_frac": {},
    }

    qc_written = 0
    for view_idx, view in enumerate(VIEWS):
        keypoints = per_view[view][0]
        index, values, bbs = [], [], []
        outside = 0

        for i, animal in rows:
            src_rel = dfs[view].index[i]
            image = cv2.imread(str(source / src_rel))
            if image is None:
                raise FileNotFoundError(f"Could not read {source / src_rel}")

            bbox = boxes[view][animal][i]
            out_rel = instance_path(src_rel, animal)

            out_path = out_dir / out_rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            crop = crop_and_resize(image, bbox, size)
            cv2.imwrite(str(out_path), crop)

            kp_crop = keypoints_to_crop_coords(keypoints[animal][i], bbox, size)
            valid = ~np.isnan(kp_crop).any(axis=-1)
            out_mask = np.any((kp_crop < 0) | (kp_crop > size - 1), axis=-1) & valid
            if out_mask.any():
                # Containment is a guarantee of keypoints_to_bbox, so this can only
                # fire on a bug there. Cropping a labelled keypoint away silently
                # would turn it into an unlearnable target, so refuse to write it.
                bad = [stems[k] for k in np.flatnonzero(out_mask)]
                raise AssertionError(
                    f"{bad} fall outside the crop for {instance_path(src_rel, animal)}: "
                    f"bbox={bbox.tolist()}, crop coords={kp_crop[out_mask].tolist()}"
                )
            outside += int(out_mask.sum())

            index.append(out_rel)
            values.append(kp_crop.reshape(-1))
            bbs.append(bbox)

            if qc_dir is not None and qc_written < qc_count and view_idx == 0:
                save_qc(qc_dir, out_rel, crop, kp_crop, stems)
                qc_written += 1

        columns = pd.MultiIndex.from_tuples(
            [("cropped", s, c) for s in stems for c in ("x", "y")],
            names=["scorer", "bodyparts", "coords"],
        )
        label_df = pd.DataFrame(np.stack(values), index=pd.Index(index), columns=columns)
        label_df.to_csv(out_dir / f"CollectedData_{view}{suffix}.csv")

        bbox_df = pd.DataFrame(
            np.stack(bbs).astype(np.int64), index=pd.Index(index), columns=["x", "y", "h", "w"],
        )
        bbox_df.to_csv(out_dir / f"bboxes_{view}{suffix}.csv")

        # MultiviewHeatmapDataset asserts these match; surface a mismatch here
        # rather than at the start of training.
        if not bbox_df.index.equals(label_df.index):
            raise AssertionError(f"bbox/label index mismatch for {view}")

        sides = np.stack(bbs)[:, 2]
        stats["bbox_px"][view] = {
            "p5": float(np.percentile(sides, 5)),
            "p50": float(np.percentile(sides, 50)),
            "p95": float(np.percentile(sides, 95)),
        }
        stats["kp_outside_crop_frac"][view] = round(outside / max(1, len(index) * n_kp), 5)

    pd.DataFrame(
        {"file": [cal.iloc[i].file for i, _ in rows]},
        index=pd.Index([instance_path(cal.index[i], a) for i, a in rows]),
    ).to_csv(out_dir / f"calibrations{suffix}.csv")

    return stats


def save_qc(qc_dir: Path, rel: str, crop: np.ndarray, kp: np.ndarray, stems: list[str]) -> None:
    """Draw keypoints on a crop so crop/label alignment can be eyeballed."""
    qc_dir.mkdir(parents=True, exist_ok=True)
    canvas = crop.copy()
    for (x, y), name in zip(kp, stems):
        if np.isnan(x) or np.isnan(y):
            continue
        cv2.circle(canvas, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1)
        cv2.putText(canvas, name[:6], (int(round(x)) + 4, int(round(y)) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(qc_dir / rel.replace("/", "__")), canvas)


def git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--crop-ratio", type=float, default=DEFAULT_CROP_RATIO)
    parser.add_argument("--output-size", type=int, default=DEFAULT_OUTPUT_SIZE)
    parser.add_argument("--animals", nargs="+", default=ANIMALS)
    parser.add_argument("--qc-count", type=int, default=12)
    args = parser.parse_args()

    if not args.source.is_dir():
        raise SystemExit(f"Source dataset not found at {args.source}")

    args.output.mkdir(parents=True, exist_ok=True)

    # Lightning Pose asserts cfg.data.video_dir exists whether or not it predicts
    # on video, so training dies at startup without this.
    (args.output / "videos").mkdir(exist_ok=True)

    # The calibration tomls are referenced by calibrations.csv relative to data_dir.
    link = args.output / "calibrations"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(args.source / "calibrations")

    provenance = {
        "source": str(args.source),
        "crop_ratio": args.crop_ratio,
        "output_size": args.output_size,
        "animals": list(args.animals),
        "views": VIEWS,
        "bbox_definition": "max keypoint extent * crop_ratio, square, about the keypoint mean",
        "lp3d_analysis_sha": git_sha(),
        "splits": {},
    }

    for split, suffix in SPLITS.items():
        print(f"\n=== {split} (suffix {suffix!r}) ===")
        stats = build_split(
            args.source, args.output, suffix, args.animals,
            args.crop_ratio, args.output_size,
            (args.output.parent / f"{args.output.name}_qc" / split) if args.qc_count else None,
            args.qc_count,
        )
        print(f"  {stats['n_frames']} frames -> {stats['n_rows']} rows per view "
              f"(dropped {stats['n_rows_dropped']})")
        for view in VIEWS:
            b = stats["bbox_px"][view]
            print(f"    {view}: bbox px p5/p50/p95 = {b['p5']:.0f}/{b['p50']:.0f}/{b['p95']:.0f}"
                  f"  kp outside crop = {stats['kp_outside_crop_frac'][view]:.3%}")
        provenance["splits"][split] = stats

    with open(args.output / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"\nwrote {args.output}/provenance.json")


if __name__ == "__main__":
    main()
