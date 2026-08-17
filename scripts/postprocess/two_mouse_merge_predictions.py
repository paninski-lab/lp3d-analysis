#!/usr/bin/env python3
"""Map per-animal cropped predictions back to the original two-mouse format.

``two_mouse_crop.py`` split each frame into one row per animal, cropped around
that animal's keypoints, and stripped identity from the keypoint names. This is
the inverse: it puts the two animals back on one row, restores the ``_black`` /
``_white`` suffixes, and reports coordinates in original full-frame pixels.

So for every view you end up with **both** files:

``predictions_<view>.csv``            (what the model writes)
    one row per (frame, animal), 11 keypoints, cropped session directories.
``<out>/predictions_<view>.csv``      (what this script writes)
    one row per frame -- half as many -- 22 keypoints, original image paths,
    original full-frame pixels. Same shape as the dataset you started from.

Coordinate space
----------------
Lightning Pose applies ``convert_bbox_coords`` in ``predict_step``, so labeled
predictions normally come out already in full-frame pixels. Older runs (and any
model trained without ``bbox_file``) instead leave them in stored-crop pixels.
Rather than assume, ``--pred-space auto`` decides per file by testing which
reading places the keypoints inside their own bounding box, and prints what it
chose. Pass ``crop`` or ``original`` to force it.

Usage::

    # merge a trained model's predictions
    python scripts/postprocess/two_mouse_merge_predictions.py \
        outputs/two-mouse_s/crop_1.3/mvt_3d_loss_400_0

    # self-test: merge the ground truth and diff it against the source dataset
    python scripts/postprocess/two_mouse_merge_predictions.py --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from lightning_pose.utils import io as lp_io

DEFAULT_DATA_DIR = Path("/teamspace/studios/this_studio/data/two-mouse_s")
DEFAULT_SOURCE = Path("/tmp/data/two-mouse")
VIEWS = ["Camera0", "Camera1", "Camera2", "Camera3", "Camera4"]
ANIMALS = ["black", "white"]
SUFFIXES = ["", "_new"]
HEADER_ROWS = [0, 1, 2]


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def read_multiindex_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=HEADER_ROWS, index_col=0)
    return lp_io.fix_empty_first_row(df)


def split_off_set_column(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Detach the train/val/test ``set`` column that predict_dataset appends."""
    set_cols = [c for c in df.columns if str(c[0]) == "set"]
    if not set_cols:
        return df, None
    return df.drop(columns=set_cols), df[set_cols[0]]


def keypoint_stems(df: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(df.columns.get_level_values("bodyparts")))


def to_arrays(df: pd.DataFrame, stems: list[str]) -> tuple[np.ndarray, np.ndarray | None]:
    """-> xy (n, n_kp, 2) and likelihood (n, n_kp) or None."""
    coords = set(df.columns.get_level_values("coords"))
    scorer = df.columns.get_level_values("scorer")[0]

    xy = np.stack(
        [
            np.stack([df[(scorer, s, "x")].to_numpy(float),
                      df[(scorer, s, "y")].to_numpy(float)], axis=-1)
            for s in stems
        ],
        axis=1,
    )
    lik = None
    if "likelihood" in coords:
        lik = np.stack([df[(scorer, s, "likelihood")].to_numpy(float) for s in stems], axis=1)
    return xy, lik


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

def split_instance_path(rel: str, animals: list[str]) -> tuple[str, str]:
    """``labeled-data/sess_Camera0_black/img.jpg`` -> (``.../sess_Camera0/img.jpg``, ``black``).

    Inverse of ``two_mouse_crop.instance_path``. Longest animal name first, so
    one name being a suffix of another cannot mis-split.
    """
    p = PurePosixPath(rel)
    parent = p.parent.name
    for animal in sorted(animals, key=len, reverse=True):
        tag = f"_{animal}"
        if parent.endswith(tag):
            return str(p.parent.with_name(parent[: -len(tag)]) / p.name), animal
    raise ValueError(f"'{rel}' does not end in any of {animals}")


# ---------------------------------------------------------------------------
# coordinate space
# ---------------------------------------------------------------------------

def crop_to_original(xy: np.ndarray, bbox: np.ndarray, size: int) -> np.ndarray:
    """(n, n_kp, 2) stored-crop pixels -> original full-frame pixels."""
    x, y, h, w = (bbox[:, i][:, None] for i in range(4))
    out = np.empty_like(xy)
    out[..., 0] = xy[..., 0] / size * w + x
    out[..., 1] = xy[..., 1] / size * h + y
    return out


def fraction_inside_bbox(xy_orig: np.ndarray, bbox: np.ndarray, slack: float = 0.02) -> float:
    """How many keypoints land inside their own box, reading them as full-frame."""
    x, y, h, w = (bbox[:, i][:, None] for i in range(4))
    mx, my = w * slack, h * slack
    ok = (
        (xy_orig[..., 0] >= x - mx) & (xy_orig[..., 0] <= x + w + mx)
        & (xy_orig[..., 1] >= y - my) & (xy_orig[..., 1] <= y + h + my)
    )
    valid = ~np.isnan(xy_orig).any(axis=-1)
    return float(ok[valid].mean()) if valid.any() else 0.0


def resolve_space(xy: np.ndarray, bbox: np.ndarray, size: int, requested: str) -> tuple[str, str]:
    """Decide whether ``xy`` is already full-frame or still in crop pixels.

    A prediction belongs inside the box it was cropped from, so whichever reading
    puts the keypoints in their own box is the correct one. The two readings are
    far apart -- a crop coordinate read as full-frame lands near the image origin,
    hundreds of pixels from the box -- so this is not a close call in practice.
    """
    as_orig = fraction_inside_bbox(xy, bbox)
    as_crop = fraction_inside_bbox(crop_to_original(xy, bbox, size), bbox)
    note = f"inside-bbox: as-original {as_orig:.1%}, as-crop {as_crop:.1%}"

    if requested != "auto":
        return requested, note + f" (forced '{requested}')"

    if max(as_orig, as_crop) < 0.5:
        raise ValueError(
            f"Cannot determine coordinate space -- neither reading puts keypoints "
            f"inside their bounding boxes ({note}). Predictions, bboxes and crop "
            f"size may not correspond. Force with --pred-space."
        )
    return ("original" if as_orig >= as_crop else "crop"), note


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def merge_view(
    pred_path: Path,
    bbox_path: Path,
    crop_size: int,
    animals: list[str],
    pred_space: str,
    scorer_out: str | None,
    keep_likelihood: bool,
    bodypart_order: list[str] | None,
) -> tuple[pd.DataFrame, dict]:
    df = read_multiindex_csv(pred_path)
    df, set_col = split_off_set_column(df)
    stems = keypoint_stems(df)
    xy, lik = to_arrays(df, stems)

    bbox_df = pd.read_csv(bbox_path, index_col=0)
    missing = df.index.difference(bbox_df.index)
    if len(missing):
        raise KeyError(f"{len(missing)} rows of {pred_path.name} have no bbox, e.g. {missing[0]}")
    bbox = bbox_df.loc[df.index, ["x", "y", "h", "w"]].to_numpy(float)

    space, note = resolve_space(xy, bbox, crop_size, pred_space)
    xy_orig = xy if space == "original" else crop_to_original(xy, bbox, crop_size)

    # frame -> {animal: row index}, preserving first-seen frame order.
    frames: dict[str, dict[str, int]] = {}
    for i, rel in enumerate(df.index):
        orig, animal = split_instance_path(rel, animals)
        slot = frames.setdefault(orig, {})
        if animal in slot:
            raise ValueError(f"'{orig}' has two rows for animal '{animal}'")
        slot[animal] = i

    if bodypart_order is None:
        bodypart_order = [f"{s}_{a}" for a in animals for s in stems]
    coords = ("x", "y", "likelihood") if (keep_likelihood and lik is not None) else ("x", "y")
    scorer = scorer_out or df.columns.get_level_values("scorer")[0]

    n_kp = len(stems)
    values = np.full((len(frames), len(bodypart_order) * len(coords)), np.nan)
    col_of = {bp: j for j, bp in enumerate(bodypart_order)}
    sets, split_conflicts, incomplete = [], 0, 0

    for r, (orig, slot) in enumerate(frames.items()):
        if len(slot) != len(animals):
            incomplete += 1
        for animal, i in slot.items():
            for k, stem in enumerate(stems):
                j = col_of[f"{stem}_{animal}"] * len(coords)
                values[r, j] = xy_orig[i, k, 0]
                values[r, j + 1] = xy_orig[i, k, 1]
                if len(coords) == 3:
                    values[r, j + 2] = lik[i, k]
        if set_col is not None:
            vals = {set_col.iloc[i] for i in slot.values()}
            # Both animals of a frame share a split by construction (the grouped
            # splitter); disagreement here means that guarantee has broken.
            if len(vals) > 1:
                split_conflicts += 1
            sets.append(sorted(vals)[0])

    columns = pd.MultiIndex.from_tuples(
        [(scorer, bp, c) for bp in bodypart_order for c in coords],
        names=["scorer", "bodyparts", "coords"],
    )
    out = pd.DataFrame(values, index=pd.Index(list(frames)), columns=columns)
    if set_col is not None:
        out[("set", "", "")] = sets

    stats = {
        "rows_in": len(df),
        "rows_out": len(out),
        "keypoints_out": len(bodypart_order),
        "space_detected": space,
        "space_note": note,
        "frames_missing_an_animal": incomplete,
        "frames_with_split_conflict": split_conflicts,
    }
    return out, stats


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def find_prediction_files(model_dir: Path, view: str, suffix: str) -> Path | None:
    """Locate a view's predictions, covering both layouts lp3d-analysis produces."""
    candidates = [
        model_dir / f"predictions_{view}{suffix}.csv",
        model_dir / "image_preds" / f"CollectedData_{view}{suffix}.csv" / "predictions.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def canonical_bodyparts(source: Path, view: str) -> list[str] | None:
    """Bodypart names in the source dataset's own order, if the source is present."""
    path = source / f"CollectedData_{view}.csv"
    if not path.exists():
        return None
    return keypoint_stems(read_multiindex_csv(path))


def verify_against_source(merged: pd.DataFrame, source: Path, view: str, suffix: str) -> dict:
    """Round-trip check: merging ground truth must reproduce the source labels."""
    path = source / f"CollectedData_{view}{suffix}.csv"
    if not path.exists():
        return {"status": f"source {path.name} not found"}
    src = read_multiindex_csv(path)
    common = merged.index.intersection(src.index)
    if len(common) == 0:
        return {"status": "no overlapping rows"}

    stems = keypoint_stems(src)
    m = merged.loc[common]
    s = src.loc[common]
    ms = m.columns.get_level_values("scorer")[0]
    ss = s.columns.get_level_values("scorer")[0]

    diffs = []
    for bp in stems:
        for c in ("x", "y"):
            if (ms, bp, c) not in m.columns:
                return {"status": f"merged output is missing {bp}/{c}"}
            diffs.append(np.abs(m[(ms, bp, c)].to_numpy(float) - s[(ss, bp, c)].to_numpy(float)))
    d = np.concatenate(diffs)
    d = d[~np.isnan(d)]
    return {
        "status": "ok",
        "rows_compared": len(common),
        "values_compared": int(d.size),
        "max_abs_err_px": float(d.max()) if d.size else float("nan"),
        "mean_abs_err_px": float(d.mean()) if d.size else float("nan"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir", type=Path, nargs="?",
                   help="model output dir holding predictions_<view>.csv; "
                        "omit with --verify to merge the ground truth instead")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                   help="cropped dataset, for bboxes_<view>.csv and provenance.json")
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                   help="original uncropped dataset, for canonical bodypart order")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default: <model_dir>/preds_original")
    p.add_argument("--crop-size", type=int, default=None,
                   help="stored crop size; default: read from provenance.json")
    p.add_argument("--pred-space", choices=["auto", "crop", "original"], default="auto")
    p.add_argument("--animals", nargs="+", default=ANIMALS)
    p.add_argument("--views", nargs="+", default=VIEWS)
    p.add_argument("--no-likelihood", action="store_true",
                   help="emit only x,y, exactly matching the source label layout")
    p.add_argument("--scorer", default=None, help="override the output scorer name")
    p.add_argument("--verify", action="store_true",
                   help="also merge the dataset ground truth and diff it against --source")
    args = p.parse_args()

    if args.model_dir is None and not args.verify:
        p.error("give a model_dir, or --verify to check the ground-truth round trip")

    crop_size = args.crop_size
    if crop_size is None:
        prov = args.data_dir / "provenance.json"
        if not prov.exists():
            p.error(f"no {prov}; pass --crop-size")
        crop_size = json.load(open(prov))["output_size"]
    print(f"stored crop size: {crop_size}")

    # The ground-truth merge goes to a sibling of the dataset, never inside it:
    # a CollectedData_*.csv under data_dir invites being picked up as data.
    out_dir = args.out_dir or (
        (args.model_dir / "preds_original") if args.model_dir
        else args.data_dir.parent / f"{args.data_dir.name}_merged_gt"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    n_written, problems = 0, []
    for view in args.views:
        order = canonical_bodyparts(args.source, view)
        for suffix in SUFFIXES:
            bbox_path = args.data_dir / f"bboxes_{view}{suffix}.csv"
            if not bbox_path.exists():
                continue

            jobs = []
            if args.model_dir is not None:
                pred = find_prediction_files(args.model_dir, view, suffix)
                if pred is not None:
                    jobs.append((pred, out_dir / f"predictions_{view}{suffix}.csv", False))
            if args.verify:
                gt = args.data_dir / f"CollectedData_{view}{suffix}.csv"
                if gt.exists():
                    jobs.append((gt, out_dir / f"CollectedData_{view}{suffix}.csv", True))

            for src_path, dst_path, is_gt in jobs:
                merged, stats = merge_view(
                    src_path, bbox_path, crop_size, args.animals, args.pred_space,
                    args.scorer if not is_gt else "scorer",
                    not args.no_likelihood, order,
                )
                merged.to_csv(dst_path)
                n_written += 1
                tag = "GT " if is_gt else "pred"
                print(f"  {tag} {view}{suffix}: {stats['rows_in']} rows -> {stats['rows_out']} rows, "
                      f"{stats['keypoints_out']} kps  [{stats['space_detected']}; {stats['space_note']}]")
                if stats["frames_missing_an_animal"]:
                    problems.append(f"{view}{suffix}: {stats['frames_missing_an_animal']} "
                                    f"frames missing an animal")
                if stats["frames_with_split_conflict"]:
                    problems.append(f"{view}{suffix}: {stats['frames_with_split_conflict']} "
                                    f"frames whose animals are in DIFFERENT splits")

                if is_gt:
                    v = verify_against_source(merged, args.source, view, suffix)
                    if v["status"] != "ok":
                        problems.append(f"{view}{suffix}: verify -- {v['status']}")
                    else:
                        print(f"       vs source: max {v['max_abs_err_px']:.6f} px, "
                              f"mean {v['mean_abs_err_px']:.6f} px "
                              f"over {v['values_compared']} values")
                        if v["max_abs_err_px"] > 1e-6:
                            problems.append(f"{view}{suffix}: round trip off by "
                                            f"{v['max_abs_err_px']:.4f} px")

    print(f"\nwrote {n_written} file(s) to {out_dir}")
    if problems:
        print("\nPROBLEMS:")
        for m in problems:
            print(f"  - {m}")
        sys.exit(1)


if __name__ == "__main__":
    main()
