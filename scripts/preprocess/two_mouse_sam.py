#!/usr/bin/env python3
"""Build ``two-mouse_s_sam``: ``two-mouse_s`` with the other animal painted red.

Each crop in ``two-mouse_s`` contains both mice -- the box is drawn around one
animal, but at this crop ratio the partner is usually in frame too, and nothing
in the image says which one the labels refer to. Here SAM2 segments the partner
and recolours it, so identity is carried by the pixels instead of being left for
the network to guess.

Same geometry, different pixels
-------------------------------
The boxes, labels, calibrations and row order are **copied verbatim** from
``two-mouse_s`` rather than recomputed. Only the image content differs, so the
two datasets are row-for-row comparable and any difference in results is
attributable to the recolouring alone.

Prompting
---------
SAM2 gets each animal's own labelled keypoints as positive prompts and the other
animal's as negative prompts, per view, per frame. No tracking and no propagation
-- every frame is independent, and the prompts are ground truth.

**No labelled keypoint is ever painted over.** Three layers, applied in order:
the partner's mask has the target's mask subtracted; a disc around every target
keypoint is punched out; and the result is asserted per crop. Burying a label
under paint would make it unlearnable in exactly the way cropping it away would.

Note this transform is only defined where ground-truth keypoints exist, i.e. on
labelled frames. Running the resulting model on fresh video needs a prompt
source -- first-pass predictions, or SAM2 video propagation from a seed frame.

Usage::

    python scripts/preprocess/two_mouse_sam.py
    python scripts/preprocess/two_mouse_sam.py --alpha 1.0 --limit 20
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from two_mouse_crop import (  # noqa: E402
    HEADER_ROWS, VIEWS, ANIMALS, SPLITS, crop_and_resize, instance_path, read_label_csv,
)

DEFAULT_SOURCE = Path("/tmp/data/two-mouse")
DEFAULT_GEOMETRY = Path("/teamspace/studios/this_studio/data/two-mouse_s")
DEFAULT_OUTPUT = Path("/teamspace/studios/this_studio/data/two-mouse_s_sam")
DEFAULT_MODEL = "facebook/sam2-hiera-large"
# Radius in original-frame pixels kept clear of paint around every target keypoint.
DEFAULT_KEEP_RADIUS = 6


def animal_keypoints(df: pd.DataFrame, row: int, animal: str, stems: list[str]) -> np.ndarray:
    """(n_keypoints, 2) for one animal in one frame, original full-frame pixels."""
    out = np.full((len(stems), 2), np.nan)
    for k, stem in enumerate(stems):
        name = f"{stem}_{animal}"
        for scorer, bp, coord in df.columns:
            if bp == name and coord in ("x", "y"):
                out[k, 0 if coord == "x" else 1] = df.iloc[row][(scorer, bp, coord)]
    return out


def segment_animals(
    predictor, image_rgb: np.ndarray, points: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """One boolean mask per animal, from its own keypoints against the other's."""
    predictor.set_image(image_rgb)
    masks = {}
    for animal, own in points.items():
        pos = own[~np.isnan(own).any(axis=1)]
        neg = np.concatenate(
            [p[~np.isnan(p).any(axis=1)] for a, p in points.items() if a != animal]
        ) if len(points) > 1 else np.empty((0, 2))
        if len(pos) == 0:
            masks[animal] = np.zeros(image_rgb.shape[:2], bool)
            continue
        coords = np.concatenate([pos, neg]) if len(neg) else pos
        labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            m, scores, _ = predictor.predict(
                point_coords=coords, point_labels=labels, multimask_output=True,
            )
        masks[animal] = m[int(np.argmax(scores))].astype(bool)
    return masks


def protected_mask(shape, keypoints: np.ndarray, radius: int) -> np.ndarray:
    """Region of a target that must never be painted.

    The convex hull of the target's keypoints, plus a disc around each one, grown
    by ``radius``. The hull matters more than the discs: when SAM2 mis-assigns and
    the partner's mask swallows the target, protecting only the keypoints leaves
    the target's body painted -- labels survive but the image is ruined. The hull
    approximates the target's body, so protecting it keeps the animal intact.

    Erring toward under-painting is deliberate. A partner left partly grey is a
    weaker cue; a target painted red is a corrupted training image.
    """
    keep = np.zeros(shape, np.uint8)
    valid = keypoints[~np.isnan(keypoints).any(axis=1)]
    if len(valid) >= 3:
        cv2.fillConvexPoly(keep, cv2.convexHull(valid.astype(np.int32)), 1)
    for x, y in valid:
        cv2.circle(keep, (int(round(x)), int(round(y))), radius, 1, -1)
    if radius > 0 and keep.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        keep = cv2.dilate(keep, k)
    return keep.astype(bool)


def check_fur_brightness(
    image_bgr: np.ndarray, masks: dict[str, np.ndarray],
) -> tuple[bool, dict]:
    """Sanity-check mask assignment using the one thing the names tell us.

    A black mouse is dark and a white mouse is bright, so the mask called 'black'
    must be darker than the mask called 'white'. SAM2 has no notion of which
    animal is which -- it only sees prompt points -- so this is an independent
    check on the assignment, not a restatement of it. Cheap, and it catches the
    mis-assignment that keypoint prompts alone cannot rule out.
    """
    grey = image_bgr[..., 0]
    means = {a: float(grey[m].mean()) if m.any() else float("nan") for a, m in masks.items()}
    ok = not (np.isfinite(means.get("black", np.nan)) and np.isfinite(means.get("white", np.nan))) \
        or means["black"] < means["white"]
    return ok, means


def paint_red(image_bgr: np.ndarray, paint: np.ndarray, alpha: float) -> np.ndarray:
    """Blend red over ``paint``, following the crim13 pipeline.

    ``alpha`` is the weight of pure red: 0.3 (the crim13 default) tints the
    partner clearly while leaving its texture legible, 1.0 erases it to a flat
    silhouette. Keeping texture is the gentler intervention -- the network still
    sees an animal, it just sees that this one is not the one being labelled.

    The frames are greyscale stored as three channels, so no real pixel is ever
    red and the cue cannot be confused with image content.
    """
    out = image_bgr.copy()
    if not paint.any():
        return out
    red = np.array([0.0, 0.0, 255.0])  # BGR
    out[paint] = np.clip(
        alpha * red + (1.0 - alpha) * image_bgr[paint].astype(np.float32), 0, 255
    ).astype(np.uint8)
    return out


def build_split(
    predictor, source: Path, geometry: Path, out_dir: Path, suffix: str,
    animals: list[str], size: int, keep_radius: int, alpha: float,
    qc_dir: Path | None, qc_count: int, limit: int | None,
) -> dict:
    src_dfs = {v: read_label_csv(source / f"CollectedData_{v}{suffix}.csv") for v in VIEWS}
    geo_labels = {
        v: pd.read_csv(geometry / f"CollectedData_{v}{suffix}.csv",
                       header=HEADER_ROWS, index_col=0)
        for v in VIEWS
    }
    boxes = {
        v: pd.read_csv(geometry / f"bboxes_{v}{suffix}.csv", index_col=0) for v in VIEWS
    }
    stems = list(dict.fromkeys(geo_labels[VIEWS[0]].columns.get_level_values("bodyparts")))

    # Rows to produce, taken straight from the geometry dataset so the output is
    # row-identical to it.
    rows = list(geo_labels[VIEWS[0]].index)
    src_index = {v: {p: i for i, p in enumerate(src_dfs[v].index)} for v in VIEWS}

    frames: dict[str, list[str]] = {}
    for rel in rows:
        p = Path(rel)
        base = p.parent.name.rsplit("_", 1)[0]
        frames.setdefault(f"{base}/{p.name}", []).append(rel)
    frame_keys = list(frames)
    if limit:
        frame_keys = frame_keys[:limit]

    stats = {"n_frames": len(frame_keys), "n_crops": 0, "mask_px": [],
             "paint_frac": [], "empty_masks": 0, "protected_hits": 0,
             "brightness_flags": 0, "brightness_examples": []}
    qc_written = 0

    for fi, key in enumerate(frame_keys):
        base, name = key.rsplit("/", 1)
        for view in VIEWS:
            src_rel = f"labeled-data/{base}_{view}/{name}" if not base.endswith(view) else \
                      f"labeled-data/{base}/{name}"
            # session dirs already carry the camera name; swap it for this view
            stem_base = base.rsplit("_", 1)[0]
            src_rel = f"labeled-data/{stem_base}_{view}/{name}"
            i = src_index[view].get(src_rel)
            if i is None:
                raise KeyError(f"{src_rel} not found in source {view} labels")

            image = cv2.imread(str(source / src_rel))
            if image is None:
                raise FileNotFoundError(source / src_rel)

            pts = {a: animal_keypoints(src_dfs[view], i, a, stems) for a in animals}
            masks = segment_animals(predictor, cv2.cvtColor(image, cv2.COLOR_BGR2RGB), pts)
            for a in animals:
                if not masks[a].any():
                    stats["empty_masks"] += 1
                stats["mask_px"].append(int(masks[a].sum()))
            ok, means = check_fur_brightness(image, masks)
            if not ok:
                stats["brightness_flags"] += 1
                stats["brightness_examples"].append(
                    {"frame": src_rel, "mean_intensity": {k: round(v, 1) for k, v in means.items()}}
                )

            for target in animals:
                out_rel = instance_path(src_rel, target)
                if out_rel not in boxes[view].index:
                    continue

                others = np.zeros(image.shape[:2], bool)
                for a in animals:
                    if a != target:
                        others |= masks[a]
                # 1) never paint where the target itself was segmented
                paint = others & ~masks[target]
                # 2) never paint near a labelled target keypoint
                keep = protected_mask(image.shape[:2], pts[target], keep_radius)
                overlap = int((paint & keep).sum())
                stats["protected_hits"] += 1 if overlap else 0
                paint &= ~keep

                painted = paint_red(image, paint, alpha)

                bbox = boxes[view].loc[out_rel, ["x", "y", "h", "w"]].to_numpy(float)
                crop = crop_and_resize(painted, bbox, size)

                # 3) assert: no labelled target keypoint sits on a painted pixel
                kp = geo_labels[view].loc[out_rel].to_numpy(float).reshape(-1, 2)
                for (cx, cy), stem in zip(kp, stems):
                    if np.isnan(cx) or np.isnan(cy):
                        continue
                    px, py = int(round(cx)), int(round(cy))
                    if not (0 <= px < size and 0 <= py < size):
                        continue
                    b, g, r = crop[py, px]
                    # greyscale source => R==G==B everywhere untouched, so ANY
                    # channel separation means this pixel was blended.
                    if int(r) - int(g) > 2 or int(r) - int(b) > 2:
                        raise AssertionError(
                            f"keypoint '{stem}' of {out_rel} landed on painted pixel "
                            f"(BGR={b},{g},{r})"
                        )

                dst = out_dir / out_rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dst), crop)
                stats["n_crops"] += 1
                stats["paint_frac"].append(
                    float(((crop[..., 2].astype(int) - crop[..., 1]) > 2).mean())
                )

                if qc_dir is not None and qc_written < qc_count and view == VIEWS[0]:
                    qc = crop.copy()
                    for (cx, cy) in kp:
                        if not np.isnan(cx):
                            cv2.circle(qc, (int(round(cx)), int(round(cy))), 3, (0, 255, 0), -1)
                    qc_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(qc_dir / out_rel.replace("/", "__")), qc)
                    qc_written += 1

        if (fi + 1) % 20 == 0:
            print(f"    {fi + 1}/{len(frame_keys)} frames", flush=True)

    stats["mask_px_median"] = float(np.median(stats.pop("mask_px")))
    pf = stats.pop("paint_frac")
    stats["paint_frac_mean"] = float(np.mean(pf))
    stats["paint_frac_zero"] = int(sum(1 for x in pf if x == 0))
    stats["brightness_examples"] = stats["brightness_examples"][:10]
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY,
                   help="dataset supplying boxes, labels and row order (copied verbatim)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output-size", type=int, default=None,
                   help="default: read from the geometry dataset's provenance.json")
    p.add_argument("--keep-radius", type=int, default=DEFAULT_KEEP_RADIUS)
    p.add_argument("--alpha", type=float, default=0.3,
                   help="weight of pure red over the partner; 0.3 matches the crim13 "
                        "pipeline (texture stays visible), 1.0 is opaque")
    p.add_argument("--animals", nargs="+", default=ANIMALS)
    p.add_argument("--qc-count", type=int, default=12)
    p.add_argument("--limit", type=int, default=None, help="first N frames only (smoke test)")
    args = p.parse_args()

    warnings.filterwarnings("ignore")
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    size = args.output_size
    if size is None:
        size = json.load(open(args.geometry / "provenance.json"))["output_size"]

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "videos").mkdir(exist_ok=True)

    print(f"loading {args.model} ...", flush=True)
    predictor = SAM2ImagePredictor.from_pretrained(args.model, device="cuda")

    provenance = {
        "source": str(args.source), "geometry": str(args.geometry),
        "sam2_model": args.model, "output_size": size,
        "keep_radius_px": args.keep_radius,
        "red_alpha": args.alpha,
        "animals": list(args.animals), "views": VIEWS, "splits": {},
    }

    for split, suffix in SPLITS.items():
        print(f"\n=== {split} (suffix {suffix!r}) ===", flush=True)
        stats = build_split(
            predictor, args.source, args.geometry, args.output, suffix, args.animals,
            size, args.keep_radius, args.alpha,
            (args.output.parent / f"{args.output.name}_qc" / split) if args.qc_count else None,
            args.qc_count, args.limit,
        )
        print(f"  {stats['n_frames']} frames -> {stats['n_crops']} crops; "
              f"median mask {stats['mask_px_median']:.0f} px; "
              f"mean painted area {stats['paint_frac_mean']:.2%}; "
              f"crops with no paint {stats['paint_frac_zero']}; "
              f"empty masks {stats['empty_masks']}; "
              f"protected-region saves {stats['protected_hits']}; "
              f"fur-brightness flags {stats['brightness_flags']}")
        provenance["splits"][split] = stats

        if args.limit:
            continue
        # Geometry files are copied, never regenerated: identical rows by construction.
        for v in VIEWS:
            for stem in (f"CollectedData_{v}{suffix}.csv", f"bboxes_{v}{suffix}.csv"):
                shutil.copy(args.geometry / stem, args.output / stem)
        shutil.copy(args.geometry / f"calibrations{suffix}.csv",
                    args.output / f"calibrations{suffix}.csv")

    link = args.output / "calibrations"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to((args.geometry / "calibrations").resolve())

    with open(args.output / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"\nwrote {args.output}/provenance.json")


if __name__ == "__main__":
    main()
