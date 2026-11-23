#!/usr/bin/env python3
"""Check a multi-animal multiview dataset against everything training assumes.

Run this on any dataset before training on it. Every check here corresponds to a
failure that has actually happened in this project, and most of them are silent:
the run starts, trains to completion, and produces numbers that mean nothing.

What is checked
---------------
structure     every ``CollectedData_<view>{,_new}.csv`` present, plus boxes and
              calibrations; every image the labels reference exists
alignment     views agree on row count *and* row order; boxes are row-aligned
              with labels; calibrations are positionally aligned by basename
calibration   every referenced toml loads and names the cameras in exactly
              ``view_names`` order -- lightning_pose asserts this, and a silent
              reordering pairs frames with the wrong camera matrices
paths         the animal token parses out of every session directory, and each
              frame has exactly one row per animal
splits        in-distribution and out-of-distribution share no session
labels        NaN fraction, coordinate range, and -- for cropped datasets -- that
              no keypoint lands outside its own crop
geometry      with ``--source``: labels really are the source keypoints mapped
              through the boxes, and the stored crops really are the source
              frames cut by those boxes (sampled, bit-exact)
twin          with ``--twin``: a recoloured variant is row-for-row identical to
              its base in every file except the images, and no labelled keypoint
              of the target animal sits on a repainted pixel

Exit status is non-zero if anything fails, so it composes into a build script.

Usage::

    python scripts/preprocess/verify_multianimal_dataset.py data/r24m_s \\
        --animals an1 an2 --views Camera1 Camera2 Camera3 Camera5 Camera6 \\
        --source data/r24m

    python scripts/preprocess/verify_multianimal_dataset.py data/r24m_s_sam \\
        --animals an1 an2 --views Camera1 Camera2 Camera3 Camera5 Camera6 \\
        --twin data/r24m_s

    python scripts/preprocess/verify_multianimal_dataset.py data/two-mouse_s \\
        --animals black white --views Camera0 Camera1 Camera2 Camera3 Camera4
"""

from __future__ import annotations

import argparse
import filecmp
import sys
import tomllib
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

HEADER_ROWS = [0, 1, 2]
SPLITS = {"ind": "", "ood": "_new"}


class Report:
    """Collects check outcomes so one run reports everything, not just the first."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {label}" + (f" -- {detail}" if detail else ""))
        if not ok:
            self.failures.append(f"{label}: {detail}" if detail else label)
        return ok

    def warn(self, label: str, detail: str = "") -> None:
        print(f"  [warn] {label}" + (f" -- {detail}" if detail else ""))
        self.warnings.append(f"{label}: {detail}" if detail else label)

    def note(self, label: str) -> None:
        print(f"         {label}")


# ---------------------------------------------------------------------------
# path parsing
# ---------------------------------------------------------------------------

def parse_dir(dir_name: str, animals: list[str], views: list[str]) -> tuple[str, str | None, str]:
    """``<dir>`` -> ``(session, animal or None, view)``, animal at any position.

    Tokens are matched whole, so ``an1`` cannot match a session called
    ``..._an10_...`` and the animal may sit before or after the view.
    """
    tokens = dir_name.split("_")
    a_at = [i for i, t in enumerate(tokens) if t in animals]
    v_at = [i for i, t in enumerate(tokens) if t in views]
    if len(v_at) != 1:
        raise ValueError(f"{dir_name!r}: expected exactly one view token, found {len(v_at)}")
    if len(a_at) > 1:
        raise ValueError(f"{dir_name!r}: {len(a_at)} animal tokens")
    drop = set(v_at) | set(a_at)
    session = "_".join(t for i, t in enumerate(tokens) if i not in drop)
    return session, (tokens[a_at[0]] if a_at else None), tokens[v_at[0]]


def read_labels(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=HEADER_ROWS, index_col=0)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def check_structure(root: Path, views: list[str], cropped: bool, rep: Report) -> bool:
    missing = []
    for suffix in SPLITS.values():
        for view in views:
            missing += [
                f for f in [f"CollectedData_{view}{suffix}.csv"]
                + ([f"bboxes_{view}{suffix}.csv"] if cropped else [])
                if not (root / f).exists()
            ]
        if not (root / f"calibrations{suffix}.csv").exists():
            missing.append(f"calibrations{suffix}.csv")
    return rep.check(not missing, "all expected csv files present",
                     f"missing {missing}" if missing else "")


def check_split(root: Path, views: list[str], animals: list[str], suffix: str,
                cropped: bool, rep: Report) -> dict:
    """Per-split alignment, calibration and label sanity. Returns summary stats."""
    counts, orders, sessions = set(), set(), set()
    n_kp = None
    stats: dict = {}

    for view in views:
        labels = read_labels(root / f"CollectedData_{view}{suffix}.csv")
        counts.add(len(labels))
        n_kp = len(labels.columns) // 2

        key = []
        for rel in labels.index:
            p = PurePosixPath(rel)
            session, animal, v = parse_dir(p.parent.name, animals, views)
            if v != view:
                rep.check(False, f"{view}{suffix} row names its own view",
                          f"{rel} names {v}")
                return stats
            key.append((session, animal, p.name))
            sessions.add(session)
        orders.add(tuple(key))

        if cropped:
            boxes = pd.read_csv(root / f"bboxes_{view}{suffix}.csv", index_col=0)
            if list(boxes.index) != list(labels.index):
                rep.check(False, f"bboxes_{view}{suffix}.csv row-aligned with labels")
                return stats
            if not (boxes["w"] == boxes["h"]).all():
                rep.warn(f"bboxes_{view}{suffix}.csv has non-square boxes")

    rep.check(len(counts) == 1, f"{suffix or 'ind'}: views agree on row count",
              f"{sorted(counts)}" if len(counts) != 1 else f"{counts.copy().pop()} rows")
    rep.check(len(orders) == 1, f"{suffix or 'ind'}: views agree on row order")

    # one row per animal per frame
    ref = read_labels(root / f"CollectedData_{views[0]}{suffix}.csv")
    per_frame: dict[tuple[str, str], list[str]] = {}
    for rel in ref.index:
        p = PurePosixPath(rel)
        session, animal, _v = parse_dir(p.parent.name, animals, views)
        per_frame.setdefault((session, p.name), []).append(animal)
    sizes = {len(v) for v in per_frame.values()}
    dupes = sum(1 for v in per_frame.values() if len(set(v)) != len(v))
    rep.check(dupes == 0, f"{suffix or 'ind'}: no frame has two rows for one animal")
    rep.note(f"{len(per_frame)} frames, rows per frame {sorted(sizes)}")

    # calibrations: positional alignment + camera order inside each toml
    cal = pd.read_csv(root / f"calibrations{suffix}.csv", index_col=0)
    rep.check(len(cal) == len(ref), f"{suffix or 'ind'}: calibration row count matches labels",
              f"{len(cal)} vs {len(ref)}")
    lab_names = [PurePosixPath(i).name for i in ref.index]
    cal_names = [PurePosixPath(i).name for i in cal.index]
    rep.check(lab_names == cal_names,
              f"{suffix or 'ind'}: calibrations positionally aligned with labels")
    bad_toml = []
    for f in cal.file.unique():
        path = root / f
        if not path.exists():
            bad_toml.append(f"{f} missing")
            continue
        t = tomllib.load(open(path, "rb"))
        names = [t[k]["name"] for k in sorted(t) if k.startswith("cam_")]
        if names != views:
            bad_toml.append(f"{f} names {names}")
    rep.check(not bad_toml, f"{suffix or 'ind'}: every toml loads and matches view order",
              "; ".join(bad_toml[:3]) if bad_toml else f"{cal.file.nunique()} calibrations")

    # labels: NaN and containment
    for view in views:
        labels = read_labels(root / f"CollectedData_{view}{suffix}.csv")
        a = labels.to_numpy(float).reshape(len(labels), -1, 2)
        nan_frac = float(np.isnan(a).any(axis=-1).mean())
        if nan_frac > 0:
            rep.note(f"{view}{suffix}: {nan_frac:.1%} keypoints unlabelled")
        if cropped:
            size = _crop_size(root)
            finite = ~np.isnan(a).any(axis=-1)
            outside = ((a[..., 0] < 0) | (a[..., 0] > size - 1)
                       | (a[..., 1] < 0) | (a[..., 1] > size - 1)) & finite
            if outside.any():
                rep.check(False, f"{view}{suffix}: all keypoints inside their crop",
                          f"{outside.sum()} of {finite.sum()} outside [0,{size - 1}]")
            elif finite.any():
                # Over finite keypoints only. A plain min() here propagates the
                # NaN of any unlabelled keypoint and reports the margin as nan,
                # which reads as "no containment problem" when it means "not
                # measured" -- the exact failure this check exists to catch.
                pts = a[finite]
                margin = float(min(pts.min(), (size - 1) - pts.max()))
                stats.setdefault("margins", []).append(margin)

    # images exist
    missing_img = 0
    for view in views:
        labels = read_labels(root / f"CollectedData_{view}{suffix}.csv")
        for rel in labels.index:
            if not (root / rel).exists():
                missing_img += 1
    rep.check(missing_img == 0, f"{suffix or 'ind'}: every referenced image exists",
              f"{missing_img} missing" if missing_img else "")

    stats["sessions"] = sessions
    stats["n_rows"] = len(ref)
    stats["n_frames"] = len(per_frame)
    stats["n_keypoints"] = n_kp
    return stats


def _crop_size(root: Path) -> int:
    import json
    prov = root / "provenance.json"
    if prov.exists():
        p = json.load(open(prov))
        if "output_size" in p:
            return int(p["output_size"])
    import cv2
    for f in (root / "labeled-data").glob("*/*"):
        im = cv2.imread(str(f))
        if im is not None:
            return im.shape[0]
    raise RuntimeError("cannot determine crop size")


def check_against_source(root: Path, source: Path, views: list[str], animals: list[str],
                         n_sample: int, rep: Report) -> None:
    """Labels and pixels must both derive from the source through the boxes."""
    import cv2
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from two_mouse_crop import crop_and_resize, keypoints_to_crop_coords

    size = _crop_size(root)
    rng = np.random.default_rng(0)
    worst_lab, worst_pix, checked = 0.0, 0, 0

    for suffix in SPLITS.values():
        for view in views:
            crop = read_labels(root / f"CollectedData_{view}{suffix}.csv")
            boxes = pd.read_csv(root / f"bboxes_{view}{suffix}.csv", index_col=0)
            full = read_labels(source / f"CollectedData_{view}{suffix}.csv")
            scorer = full.columns[0][0]
            stems = list(dict.fromkeys(crop.columns.get_level_values("bodyparts")))
            fidx = {p: i for i, p in enumerate(full.index)}

            pick = rng.choice(len(crop), min(n_sample, len(crop)), replace=False)
            for i in pick:
                rel = crop.index[i]
                p = PurePosixPath(rel)
                session, animal, v = parse_dir(p.parent.name, animals, views)
                frel = f"labeled-data/{session}_{v}/{p.name}"
                if frel not in fidx:
                    rep.check(False, "source frame exists for every crop row", frel)
                    return
                j = fidx[frel]
                kp = np.array([
                    [full.iloc[j][(scorer, f"{s}_{animal}", "x")],
                     full.iloc[j][(scorer, f"{s}_{animal}", "y")]] for s in stems
                ])
                box = boxes.loc[rel, ["x", "y", "h", "w"]].to_numpy(float)
                expected = keypoints_to_crop_coords(kp, box, size)
                got = crop.loc[rel].to_numpy(float).reshape(-1, 2)
                worst_lab = max(worst_lab, float(np.nanmax(np.abs(expected - got))))

                img = cv2.imread(str(source / frel))
                regen = crop_and_resize(img, box, size)
                shipped = cv2.imread(str(root / rel))
                worst_pix = max(worst_pix,
                                int(np.abs(regen.astype(int) - shipped.astype(int)).max()))
                checked += 1

    rep.check(worst_lab < 1e-6, "labels are source keypoints mapped through the boxes",
              f"max {worst_lab:.2e} px over {checked} sampled rows")
    rep.check(worst_pix == 0, "stored crops are source frames cut by the boxes",
              f"max channel diff {worst_pix}/255 over {checked} sampled crops")


def check_twin(root: Path, twin: Path, views: list[str], animals: list[str],
               n_sample: int, rep: Report) -> None:
    """A recoloured variant must differ from its base in images only."""
    import cv2

    diffs = []
    for suffix in SPLITS.values():
        for view in views:
            for f in (f"CollectedData_{view}{suffix}.csv", f"bboxes_{view}{suffix}.csv"):
                if not filecmp.cmp(twin / f, root / f, shallow=False):
                    diffs.append(f)
        f = f"calibrations{suffix}.csv"
        if not filecmp.cmp(twin / f, root / f, shallow=False):
            diffs.append(f)
    rep.check(not diffs, "geometry, labels and calibrations byte-identical to the base",
              f"differ: {diffs}" if diffs else "")

    rels_a = {str(p.relative_to(twin)) for p in (twin / "labeled-data").glob("*/*")}
    rels_b = {str(p.relative_to(root)) for p in (root / "labeled-data").glob("*/*")}
    rep.check(rels_a == rels_b, "same set of images as the base",
              f"{len(rels_a)} vs {len(rels_b)}")

    # no labelled keypoint of the target animal may sit on a repainted pixel
    rng = np.random.default_rng(0)
    rels = sorted(rels_b & rels_a)
    pick = rng.choice(len(rels), min(n_sample, len(rels)), replace=False)
    labels = {
        (v, s): read_labels(root / f"CollectedData_{v}{s}.csv")
        for v in views for s in SPLITS.values()
    }
    violations, painted, frac = 0, 0, []
    for i in pick:
        rel = rels[i]
        a = cv2.imread(str(twin / rel)).astype(int)
        b = cv2.imread(str(root / rel)).astype(int)
        changed = np.abs(a - b).sum(axis=2) > 0
        frac.append(float(changed.mean()))
        painted += int(changed.any())
        _s, _an, v = parse_dir(PurePosixPath(rel).parent.name, animals, views)
        for suffix in SPLITS.values():
            d = labels[(v, suffix)]
            if rel in d.index:
                kp = d.loc[rel].to_numpy(float).reshape(-1, 2)
                h, w = changed.shape
                for x, y in kp:
                    if np.isnan(x):
                        continue
                    px, py = int(round(x)), int(round(y))
                    if 0 <= px < w and 0 <= py < h and changed[py, px]:
                        violations += 1
                break
    rep.check(violations == 0, "no labelled keypoint sits on a repainted pixel",
              f"{violations} violations" if violations else f"{len(pick)} crops sampled")
    rep.note(f"{painted / len(pick):.1%} of sampled crops carry paint, "
             f"mean repainted area {np.mean(frac):.2%}")


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path)
    p.add_argument("--animals", nargs="+", required=True)
    p.add_argument("--views", nargs="+", required=True)
    p.add_argument("--source", type=Path, default=None,
                   help="uncropped dataset the crops came from; enables the geometry checks")
    p.add_argument("--twin", type=Path, default=None,
                   help="base dataset this one recolours; enables the twin checks")
    p.add_argument("--sample", type=int, default=60,
                   help="rows per view/split to sample for the pixel-level checks")
    args = p.parse_args()

    root = args.dataset
    cropped = (root / f"bboxes_{args.views[0]}.csv").exists()
    rep = Report()

    print(f"=== {root} ===")
    print(f"  arrangement: {'cropped per-animal' if cropped else 'full frame'}; "
          f"animals={args.animals}; views={args.views}")

    check_structure(root, args.views, cropped, rep)

    all_sessions = {}
    for split, suffix in SPLITS.items():
        print(f"  --- {split} ---")
        stats = check_split(root, args.views, args.animals, suffix, cropped, rep)
        all_sessions[split] = stats.get("sessions", set())
        if stats.get("margins"):
            rep.note(f"smallest keypoint-to-crop-edge margin: {min(stats['margins']):.1f} px")

    overlap = all_sessions.get("ind", set()) & all_sessions.get("ood", set())
    rep.check(not overlap, "in-distribution and OOD share no session",
              f"{sorted(overlap)[:3]}" if overlap else
              f"{len(all_sessions['ind'])} ind / {len(all_sessions['ood'])} ood sessions")

    if args.source:
        print("  --- against source ---")
        check_against_source(root, args.source, args.views, args.animals, args.sample, rep)
    if args.twin:
        print("  --- against twin ---")
        check_twin(root, args.twin, args.views, args.animals, args.sample, rep)

    print()
    if rep.failures:
        print(f"FAILED {len(rep.failures)} check(s):")
        for f in rep.failures:
            print(f"  - {f}")
        sys.exit(1)
    print(f"all checks passed" + (f" ({len(rep.warnings)} warning(s))" if rep.warnings else ""))


if __name__ == "__main__":
    main()
