#!/usr/bin/env python
"""Generate the `camera_params_file` CSVs for the rat7m-full-crop dataset.

The full-crop tarball ships `calibrations/<session>.toml` but not the
`calibrations.csv` / `calibrations_new.csv` index that
`cfg.data.camera_params_file` points at (the older rat7m-crop release had both).
That index is pure derived data -- a per-frame map from image to the calibration
file for that frame's session -- so we rebuild it from the label CSVs.

Contract enforced by lightning_pose/data/datasets.py:469-489:
  * read with `index_col=0, header=[0]`, so exactly two columns: index + `file`
  * `[basename(i) for i in index]` must equal the dataset's image basenames
    elementwise **and in order**
  * `file` is joined onto `root_directory`, so it is relative to data_dir
  * the toml's camera `name` fields must equal cfg.data.view_names, in order

Run after staging the dataset:
    python scripts/make_rat7m_calibrations_csv.py /tmp/data/rat7m-full-crop
"""
import sys
from pathlib import Path

import pandas as pd


def build(data_dir: Path, suffix: str, view_name: str, view_names: list[str]) -> bool:
    """Build one calibration index (suffix "" for train, "_new" for OOD)."""
    label_csv = data_dir / f"CollectedData_{view_name}{suffix}.csv"
    if not label_csv.exists():
        print(f"  [skip] {label_csv.name} not found")
        return False

    df = pd.read_csv(label_csv, header=[0, 1, 2], index_col=0)

    rows = []
    missing_tomls = set()
    for img_path in df.index:
        # "labeled-data/s1-d1_camera1/img00000043.png" -> session "s1-d1"
        parts = str(img_path).split("/")
        session_dir = parts[1]
        session = session_dir[: -(len(view_name) + 1)] if session_dir.endswith(
            f"_{view_name}"
        ) else session_dir

        toml_rel = f"calibrations/{session}.toml"
        if not (data_dir / toml_rel).exists():
            missing_tomls.add(toml_rel)

        # Drop the camera suffix so the index is view-agnostic, as the older
        # release did. Keep whatever top-level dir the label csv uses (this
        # dataset says "labeled-data-cropped") so the two stay consistent; only
        # the basename is actually compared.
        rows.append((f"{parts[0]}/{session}/{parts[2]}", toml_rel))

    if missing_tomls:
        print(f"  [ERROR] referenced calibration files do not exist: {sorted(missing_tomls)}")
        return False

    out = pd.DataFrame([r[1] for r in rows], index=[r[0] for r in rows], columns=["file"])
    out.index.name = None

    # Verify the ordered-basename assertion before writing, not after.
    lhs = [str(i).split("/")[-1] for i in df.index]
    rhs = [str(i).split("/")[-1] for i in out.index]
    assert lhs == rhs, "basename order mismatch vs label csv"

    out_path = data_dir / f"calibrations{suffix}.csv"
    out.to_csv(out_path)
    per_session = pd.Series([r[1] for r in rows]).value_counts().to_dict()
    print(f"  wrote {out_path.name}: {len(out)} rows")
    for k, v in sorted(per_session.items()):
        print(f"      {k:<34} {v:>6} frames")
    return True


def main() -> int:
    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/data/rat7m-full-crop")
    view_names = [f"camera{i}" for i in range(1, 7)]

    if not data_dir.exists():
        print(f"[ERROR] data_dir does not exist: {data_dir}")
        return 1

    print(f"Building calibration indices in {data_dir}")
    ok = True
    for suffix in ("", "_new"):
        ok &= build(data_dir, suffix, view_names[0], view_names)

    # The toml camera order must line up with cfg.data.view_names or the dataset
    # raises at construction time; check it here where the message is clearer.
    try:
        from lightning_pose.data.cameras import CameraGroup

        for toml in sorted((data_dir / "calibrations").glob("*.toml")):
            names = list(CameraGroup.load(str(toml)).get_names())
            status = "OK" if names == view_names else f"MISMATCH -> {names}"
            print(f"  {toml.name:<14} camera order: {status}")
            ok &= names == view_names
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] could not verify camera order: {e}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
