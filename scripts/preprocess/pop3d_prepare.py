"""Organise the shipped 3d-pop-crop archive into data/3d-pop_s (SLP3D joint crops).

The archive already is a per-animal cropped dataset: one row per (time point,
bird), 320x320 crops, boxes in original 3840x2160 pixels, one anipose toml per
(sequence, bird). What it lacks is the per-row calibration lookup Lightning Pose
reads (``calibrations.csv`` / ``calibrations_new.csv``) and a provenance record.

Nothing geometric is regenerated: labels, boxes, tomls and images are copied
verbatim. The only derived files are the two calibration lookups.

Session directories are ``<sequence>_id<tag>_<cam>``; the animal token is
``id<tag>`` (a real bird tag, consistent across sequences) and sits before the
view, as in r24m. Frames carry 2 or 5 birds.

Usage::

    python scripts/preprocess/pop3d_prepare.py \
        --src ../data_staging/3d-pop-crop/extracted --dst ../data/3d-pop_s
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path, PurePosixPath

import pandas as pd

VIEWS = ["Cam1", "Cam2", "Cam3", "Cam4"]
SPLITS = {"ind": "", "ood": "_new"}
CROP_SIZE = 320


def split_dir(dir_name: str) -> tuple[str, str, str]:
    """``Sequence13_n02_13072022_id483_Cam1`` -> (sequence, animal, view)."""
    tokens = dir_name.split("_")
    a_at = [i for i, t in enumerate(tokens) if t.startswith("id") and t[2:].isdigit()]
    v_at = [i for i, t in enumerate(tokens) if t in VIEWS]
    if len(a_at) != 1 or len(v_at) != 1:
        raise ValueError(f"{dir_name!r}: expected one animal and one view token")
    seq = "_".join(t for i, t in enumerate(tokens) if i not in (a_at[0], v_at[0]))
    return seq, tokens[a_at[0]], tokens[v_at[0]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--archive", type=str, default="data_staging/3d-pop-crop/3d-pop-crop.tar.xz")
    args = p.parse_args()
    src, dst = args.src.resolve(), args.dst.resolve()
    if dst.exists():
        raise SystemExit(f"{dst} exists; refusing to overwrite")
    dst.mkdir(parents=True)

    # verbatim copies
    for suffix in SPLITS.values():
        for view in VIEWS:
            for stem in ("CollectedData", "bboxes"):
                shutil.copy2(src / f"{stem}_{view}{suffix}.csv", dst)
    shutil.copytree(src / "calibrations", dst / "calibrations")
    shutil.copytree(src / "labeled-data", dst / "labeled-data")
    shutil.copy2(src / "README.md", dst / "README_source.md")
    (dst / "videos").mkdir()
    # lightning-pose asserts video_dir exists; empty dirs are lost on a studio machine switch
    (dst / "videos" / ".keep").write_text("placeholder so this dir persists\n")

    # calibration lookups: index = label path minus the view token, keeps the animal
    animals, splits = set(), {}
    for split, suffix in SPLITS.items():
        labels = pd.read_csv(src / f"CollectedData_{VIEWS[0]}{suffix}.csv",
                             header=[0, 1, 2], index_col=0)
        idx, files, seqs, frames = [], [], set(), set()
        for rel in labels.index:
            rp = PurePosixPath(rel)
            seq, animal, _ = split_dir(rp.parent.name)
            sess = f"{seq}_{animal}"
            toml = f"calibrations/{sess}.toml"
            if not (dst / toml).exists():
                raise FileNotFoundError(toml)
            idx.append(f"labeled-data/{sess}/{rp.name}")
            files.append(toml)
            animals.add(animal)
            seqs.add(seq)
            frames.add((seq, rp.name))
        pd.DataFrame({"file": files}, index=idx).to_csv(dst / f"calibrations{suffix}.csv")
        splits[split] = {"n_rows": len(labels), "n_frames": len(frames),
                         "n_sequences": len(seqs), "n_calibrations": len(set(files))}

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=Path(__file__).parent).stdout.strip()
    prov = {
        "source": args.archive,
        "origin": "3D-POP (Naik, Chan et al., CVPR 2023); shipped pre-cropped, see README_source.md",
        "arrangement": "per-animal crop",
        "crop_definition": "square box centred on the bird's 9-keypoint extent, side = 2.0 x larger extent, "
                           "black-padded outside the frame, resized to 320 (shipped, not rebuilt here)",
        "output_size": CROP_SIZE,
        "frame_size": [3840, 2160],
        "animals": sorted(animals, key=lambda a: int(a[2:])),
        "animal_token": "id<tag>, a bird tag number consistent across sequences; before the view",
        "animals_per_frame": [2, 5],
        "views": VIEWS,
        "split": "InD/OOD by sequence (disjoint sequences, shared birds)",
        "derived_here": ["calibrations.csv", "calibrations_new.csv"],
        "lp3d_analysis_sha": sha,
        "splits": splits,
    }
    (dst / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
