#!/usr/bin/env python3
"""Build ``r24m_s_sam``: ``r24m_s`` with the other rat painted red.

The r24m counterpart of ``two_mouse_sam.py``, and the same idea: each 320x320
crop is drawn around one rat, but at this crop ratio the partner is usually in
frame too, and nothing in the pixels says which rat the labels refer to. SAM2
segments the partner and recolours it, so identity is carried by the image
instead of being left for the network to guess.

Two rats, not two mice
----------------------
It matters more here. The two mice were black and white, so a network had a cue
even without help. These are two hooded rats of the same strain and markings --
to the eye they are interchangeable, and the only thing distinguishing them in a
crop is which one the box is centred on.

Same geometry, different pixels
-------------------------------
Boxes, labels, calibrations and row order are **copied verbatim** from ``r24m_s``
rather than recomputed, and the crop is taken with the same function that
produced ``r24m_s`` -- verified to reproduce its images bit-exactly. The two
datasets are therefore row-for-row and pixel-for-pixel comparable, and any
difference in results is attributable to the recolouring alone.

Prompting
---------
SAM2 gets each rat's own labelled keypoints as positive prompts and the other
rat's as negative prompts, per view, per frame. No tracking and no propagation --
every frame is independent, and the prompts are ground truth.

**No labelled keypoint is ever painted over.** Three layers, applied in order:
the partner's mask has the target's mask subtracted; the target's convex hull
plus a disc around every keypoint is punched out; and the paint mask is then
asserted against the target's keypoints directly, in full-frame coordinates,
before the crop is taken.

That last check differs from the two-mouse script, which tested whether a
keypoint's pixel had become redder than its neighbours. That worked because the
two-mouse frames were greyscale stored as three channels, so any channel
separation meant paint. These frames are genuinely colour (mean |R-G| ~ 12), so
the colour test would both miss paint and fire on bedding. Testing the mask is
exact either way.

Usage::

    python scripts/preprocess/r24m_sam.py
    python scripts/preprocess/r24m_sam.py --alpha 0.5 --limit 10
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from two_mouse_crop import crop_and_resize, read_label_csv  # noqa: E402
from two_mouse_sam import (  # noqa: E402
    animal_keypoints, paint_red, protected_mask, segment_animals,
)
from r24m_layout import (  # noqa: E402
    ANIMALS, HEADER_ROWS, KEYPOINTS, SPLITS, VIEWS, instance_path,
    parse_crop_dir,
)

DEFAULT_SOURCE = Path("/teamspace/studios/this_studio/data/r24m")
DEFAULT_GEOMETRY = Path("/teamspace/studios/this_studio/data/r24m_s")
DEFAULT_OUTPUT = Path("/teamspace/studios/this_studio/data/r24m_s_sam")
DEFAULT_MODEL = "facebook/sam2-hiera-large"
# Radius in original-frame pixels kept clear of paint around every target keypoint.
DEFAULT_KEEP_RADIUS = 6


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
    if stems != KEYPOINTS:
        raise ValueError(f"unexpected keypoints {stems}")

    # Frames to process, taken from the geometry dataset so the output is
    # row-identical to it: one entry per (session, image), both animals together.
    frames: dict[tuple[str, str], None] = {}
    for rel in geo_labels[VIEWS[0]].index:
        p = PurePosixPath(rel)
        session, _animal, _view = parse_crop_dir(p.parent.name)
        frames[(session, p.name)] = None
    frame_keys = list(frames)
    if limit:
        frame_keys = frame_keys[:limit]

    src_index = {v: {p: i for i, p in enumerate(src_dfs[v].index)} for v in VIEWS}

    stats = {"n_frames": len(frame_keys), "n_crops": 0, "mask_px": [],
             "paint_frac": [], "empty_masks": 0, "protected_hits": 0}
    qc_written = 0

    for fi, (session, name) in enumerate(frame_keys):
        for view in VIEWS:
            src_rel = f"labeled-data/{session}_{view}/{name}"
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
                if (paint & keep).any():
                    stats["protected_hits"] += 1
                paint &= ~keep

                # 3) assert, against the mask itself, that no target keypoint is
                # painted. Exact, and independent of what the frame's colours are.
                h, w = paint.shape
                for (kx, ky), stem in zip(pts[target], stems):
                    if np.isnan(kx) or np.isnan(ky):
                        continue
                    px, py = int(round(float(kx))), int(round(float(ky)))
                    if not (0 <= px < w and 0 <= py < h):
                        continue
                    if paint[py, px]:
                        raise AssertionError(
                            f"keypoint '{stem}' of {out_rel} would be painted over"
                        )

                painted = paint_red(image, paint, alpha)
                bbox = boxes[view].loc[out_rel, ["x", "y", "h", "w"]].to_numpy(float)
                crop = crop_and_resize(painted, bbox, size)

                dst = out_dir / out_rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dst), crop)
                stats["n_crops"] += 1
                # Painted area measured from the mask, not from the crop's colours:
                # these frames are colour, so a red test on pixels would count bedding.
                paint_crop = crop_and_resize(
                    np.repeat(paint[:, :, None].astype(np.uint8) * 255, 3, axis=2),
                    bbox, size,
                )
                stats["paint_frac"].append(float((paint_crop[..., 0] > 127).mean()))

                if qc_dir is not None and qc_written < qc_count and view == VIEWS[0]:
                    qc = crop.copy()
                    kp = geo_labels[view].loc[out_rel].to_numpy(float).reshape(-1, 2)
                    for (cx, cy) in kp:
                        if not np.isnan(cx):
                            cv2.circle(qc, (int(round(cx)), int(round(cy))), 3,
                                       (0, 255, 0), -1)
                    qc_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(qc_dir / out_rel.replace("/", "__")), qc)
                    qc_written += 1

        if (fi + 1) % 20 == 0:
            print(f"    {fi + 1}/{len(frame_keys)} frames", flush=True)

    stats["mask_px_median"] = float(np.median(stats.pop("mask_px")))
    pf = stats.pop("paint_frac")
    stats["paint_frac_mean"] = float(np.mean(pf))
    stats["paint_frac_zero"] = int(sum(1 for x in pf if x == 0))
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                   help="uncropped dataset supplying full frames and combined labels")
    p.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY,
                   help="dataset supplying boxes, labels and row order (copied verbatim)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output-size", type=int, default=None,
                   help="default: read from the geometry dataset's provenance.json")
    p.add_argument("--keep-radius", type=int, default=DEFAULT_KEEP_RADIUS)
    p.add_argument("--alpha", type=float, default=0.3,
                   help="weight of pure red over the partner; 0.3 matches the crim13 "
                        "pipeline and the two-mouse build, 1.0 is opaque")
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
              f"protected-region saves {stats['protected_hits']}", flush=True)
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
