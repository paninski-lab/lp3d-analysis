#!/usr/bin/env python3
"""Generate the Lightning Pose and pipeline configs for every r24m variant.

Seven datasets, five experimental conditions::

    r24m              full frames, both rats in one row, 24 suffixed keypoints
    r24m_s            320px per-animal crops, both rats as separate rows   (SLP3D)
    r24m_s_sam        the same crops with the other rat painted red        (SLP3D+SAM)
    r24m_s-an1/-an2   one model per rat, plain crops
    r24m_s_sam-an1/-an2   one model per rat, recoloured crops

They differ in five fields -- data directory, keypoint count and names, whether
rows are per-animal, and the label budget -- and must agree in every other. That
is why this is a generator rather than seven hand-maintained files: the last time
these were written by hand, a derived config silently inherited the wrong
``camera_params_file`` and training died on a calibration-row mismatch that took
a run to diagnose. Here each dataset's path is derived from its own name.

The label budget is set so that **every condition sees the same 248 labelled
frames** -- every in-distribution frame that is not held out for validation:
248 rows where a row is a frame, 496 where a row is one rat of a frame.

Usage::

    python scripts/preprocess/r24m_make_configs.py
    python scripts/preprocess/r24m_make_configs.py --seeds 0 1 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from r24m_layout import KEYPOINTS, VIEWS

DATA_ROOT = "/teamspace/studios/this_studio/data"
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

# Frames, not rows. A joint dataset spends two rows per frame.
#
# 248 is ALL of it: 261 in-distribution frames at train_prob 0.95 / val_prob 0.05
# split into 248 train and 13 val, so nothing labelled goes unused. Set from the
# measured split rather than guessed -- lightning_pose clamps an over-request
# down to the split size, so a too-large number trains on everything but silently
# stops meaning what it says, and the run directory it names becomes a lie.
FRAME_BUDGET = 248

# Derived from the median pairwise keypoint distances in the data: a head
# triangle, a spine chain, shoulders off SpineF, hips off SpineL, and the two
# trunk offset markers hung off SpineM, which is what they sit nearest to.
SKELETON = [
    ("HeadF", "HeadB"), ("HeadB", "HeadL"), ("HeadL", "HeadF"),
    ("HeadB", "SpineF"),
    ("SpineF", "SpineM"), ("SpineM", "SpineL"),
    ("SpineF", "ShoulderL"), ("SpineF", "ShoulderR"),
    ("SpineL", "HipL"), ("SpineL", "HipR"),
    ("SpineM", "Offset1"), ("SpineM", "Offset2"),
]
SKELETON_COLORS = {
    "HeadF-HeadB": "red", "HeadB-HeadL": "red", "HeadL-HeadF": "red",
    "HeadB-SpineF": "pink",
    "SpineF-SpineM": "darkred", "SpineM-SpineL": "darkred",
    "SpineF-ShoulderL": "dodgerblue", "SpineF-ShoulderR": "dodgerblue",
    "SpineL-HipL": "mediumseagreen", "SpineL-HipR": "mediumseagreen",
    "SpineM-Offset1": "goldenrod", "SpineM-Offset2": "goldenrod",
}

# name -> (data dir, animals-per-row, cropped)
VARIANTS = {
    "r24m":            (None, "combined", False),
    "r24m_s":          (None, "joint", True),
    "r24m_s_sam":      (None, "joint", True),
    "r24m_s-an1":      (None, "single", True),
    "r24m_s-an2":      (None, "single", True),
    "r24m_s_sam-an1":  (None, "single", True),
    "r24m_s_sam-an2":  (None, "single", True),
}

NOTES = {
    "combined": (
        "Full uncropped frames, one row per frame, both rats in the keypoint\n"
        "# names. This is the arrangement the original two-mouse model was trained\n"
        "# in -- the baseline the cropped variants are measured against."
    ),
    "joint": (
        "One row per (frame, rat), identity carried by the session directory and\n"
        "# stripped from the keypoint names. cfg.data.animal_names declares that, and\n"
        "# keeps both rats of a frame in the same train/val split."
    ),
    "single": (
        "One rat only. The model never has to work out which rat it is looking at,\n"
        "# which makes it the upper bound on what removing identity ambiguity can buy."
    ),
}


def yaml_list(items, indent="    ") -> str:
    return "\n".join(f"{indent}- {i}" for i in items)


def keypoint_names(kind: str) -> list[str]:
    if kind == "combined":
        return [f"{k}_{a}" for a in ("an1", "an2") for k in KEYPOINTS]
    return list(KEYPOINTS)


def skeleton_block(kind: str) -> str:
    """Skeleton and colours, duplicated per animal for the combined arrangement."""
    animals = ["an1", "an2"] if kind == "combined" else [None]
    edges, colors = [], []
    for a in animals:
        for u, v in SKELETON:
            un, vn = (f"{u}_{a}", f"{v}_{a}") if a else (u, v)
            edges.append(f"    - ['{un}', '{vn}']")
            colors.append(f"    {un}-{vn}: {SKELETON_COLORS[f'{u}-{v}']}")
    return "  skeleton:\n" + "\n".join(edges) + "\n\n  skeleton_colors:\n" + "\n".join(colors)


def make_config(name: str, kind: str, cropped: bool) -> str:
    data_dir = f"{DATA_ROOT}/{name}"
    kps = keypoint_names(kind)

    bbox = ""
    if cropped:
        bbox = (
            "  # Each box in ORIGINAL full-frame pixels, row-aligned with csv_file. This is\n"
            "  # what lets the model train on 320px crops while the 3D reprojection loss\n"
            "  # works in full-frame coordinates.\n"
            "  bbox_file:\n" + yaml_list(f"bboxes_{v}.csv" for v in VIEWS) + "\n"
        )

    animals = ""
    if kind == "joint":
        animals = (
            "  # Which suffix-or-infix token each row's session directory carries. Rows\n"
            "  # sharing a frame are kept in the same train/val split.\n"
            "  animal_names:\n    - an1\n    - an2\n"
        )

    return f"""# {name}
#
# {NOTES[kind]}
#
# Generated by scripts/preprocess/r24m_make_configs.py -- edit that, not this.
data:
  # resize dimensions to streamline model creation
  image_resize_dims:
    height: 256
    width: 256
  data_dir: {data_dir}
  video_dir: {data_dir}/videos
  # Per-row lookup from a label row to its camera calibration, matched
  # POSITIONALLY against the label rows. Always this dataset's own file: a config
  # that inherits another dataset's calibrations fails with a row-count mismatch.
  camera_params_file: {data_dir}/calibrations.csv
  # In-distribution. Switch csv_file and bbox_file together to the *_new set for OOD.
  csv_file:
{yaml_list(f"CollectedData_{v}.csv" for v in VIEWS)}
{bbox}  view_names:
{yaml_list(VIEWS)}

  # downsample heatmaps - 2 | 3
  downsample_factor: 2
  num_keypoints: {len(kps)}
  keypoint_names:
{yaml_list(kps)}
{animals}
  # this info is only used for the multiview pca loss
  mirrored_column_matches: {list(range(len(kps)))}
  columns_for_singleview_pca: []

{skeleton_block(kind)}

training:
  imgaug: dlc
  # Use the regular dlc augmentations even though camera params are present.
  imgaug_3d: True
  train_batch_size: 8
  val_batch_size: 16
  test_batch_size: 1
  train_prob: 0.95
  val_prob: 0.05
  # Overridden by the pipeline's n_hand_labels; see pipeline_{name}.yaml.
  train_frames: 1
  num_gpus: 1
  num_workers: 4
  early_stop_patience: 3
  unfreezing_epoch: 20
  # EPOCHS. Run through pipeline_simple.py these are replaced by the step-based
  # schedule in the pipeline config; standalone `litpose train` uses them as-is.
  min_epochs: 300
  max_epochs: 300
  log_every_n_steps: 10
  check_val_every_n_epoch: 5
  rng_seed_data_pt: 0
  rng_seed_model_pt: 0
  optimizer: Adam
  optimizer_params:
    learning_rate: 0.00005
  lr_scheduler: multisteplr
  lr_scheduler_params:
    multisteplr:
      milestones: [150, 200, 250]
      gamma: 0.5
  uniform_heatmaps_for_nan_keypoints: true

  # curriculum learning configuration for patch masking
  patch_mask:
    init_step: 700
    final_step: 5000
    init_ratio: 0.1
    final_ratio: 0.5

model:
  losses_to_use: []
  backbone: vits_dino
  model_type: heatmap_multiview_transformer
  head: heatmap_cnn
  heatmap_loss_type: mse
  model_name: {name}

dali:
  general:
    seed: 123456
  base:
    train:
      sequence_length: 8
    predict:
      sequence_length: 16
  context:
    train:
      batch_size: 16
    predict:
      sequence_length: 96

losses:
  pca_multiview:
    log_weight: 5.0
    components_to_keep: 3
    epsilon: null
  pca_singleview:
    log_weight: 5.0
    components_to_keep: 0.99
    epsilon: null
  temporal:
    log_weight: 5.0
    epsilon: 20.0
    prob_threshold: 0.05
  # 3D reprojection loss. Constant weight -- not gated by AnnealWeight, which
  # only scales the *unsupervised* losses.
  supervised_reprojection_heatmap_mse:
    log_weight: 1.0

eval:
  # OFF: no cropped videos exist for this dataset. Leaving it on makes a run fail
  # at the very end, after the fit has already completed.
  predict_vids_after_training: false
  test_videos_directory: ${{data.video_dir}}
  save_vids_after_training: false
  colormap: "cool"
  confidence_thresh_for_vid: 0.90
  hydra_paths: [" "]
  fiftyone:
    dataset_name: test
    model_display_names: ["test_model"]
    launch_app_from_script: false
    remote: true
    address: 127.0.0.1
    port: 5151

callbacks:
  anneal_weight:
    attr_name: total_unsupervised_importance
    init_val: 0.0
    increase_factor: 0.01
    final_val: 1.0
    freeze_until_epoch: 60
"""


def make_pipeline(name: str, kind: str, cropped: bool, seeds: list[int]) -> str:
    rows = FRAME_BUDGET * 2 if kind == "joint" else FRAME_BUDGET
    budget_note = (
        f"# training.train_frames counts ROWS. This dataset has two rows per frame\n"
        f"  # (one per rat), so {rows} rows is {FRAME_BUDGET} frames -- the same number of\n"
        f"  # frames every other r24m condition is given."
        if kind == "joint" else
        f"# training.train_frames counts ROWS. This dataset has one row per frame,\n"
        f"  # so {rows} here is {FRAME_BUDGET} frames -- the same number of frames every\n"
        f"  # other r24m condition is given."
    )
    intermediate = "crop_1.4" if cropped else "full_frame"

    return f"""# Pipeline config for {name}.
#
# Resolves to configs/config_{name}.yaml.
#
# Prefer this over `litpose train` for real runs: make_model_cfg() replaces the
# epoch-based schedule with the step-based one below and forces
# predict_vids_after_training off, which is what the rest of the project's
# results were produced with.
#
# Generated by scripts/preprocess/r24m_make_configs.py -- edit that, not this.

dataset_name: {name}

lightning_pose_config: /teamspace/studios/this_studio/lp3d-analysis/configs/config_${{dataset_name}}.yaml

pipeline_seeds: 0

outputs_dir: /teamspace/studios/this_studio/outputs/${{dataset_name}}

intermediate_results_dir: {intermediate}

train_networks:
  run: True
  overwrite: False

  ensemble_seeds:
{yaml_list(seeds, indent="    ")}

  {budget_note}
  n_hand_labels:
    - {rows}

  model_types:
    # heatmap_multiview_transformer + heatmap_cnn head, supervised, with the 3D
    # reprojection loss from cfg_lp.losses.
    - mvt_3d_loss

  # Empty: no videos exist for this dataset yet.
  inference_dirs: []

  min_steps: 5000
  max_steps: 5000
  unfreezing_step: 400
  milestone_steps: [2000, 3000, 4000]
  val_check_interval: 10
  train_check_interval: 5

# Everything below is off: ensembling and EKS need several trained seeds and, for
# the video modes, videos.
post_processing_labels:
  eks_singleview:
    run: False
    overwrite: False
  eks_multiview:
    run: False
    overwrite: False
    n_latent: 3
    non_linear: True
    output_folder_name: non_linear_eks_varinf
  ensemble_mean:
    run: False
    overwrite: False
  ensemble_median:
    run: False
    overwrite: False

post_processing_labeled_frames:
  run: False
  overwrite: False
  modes: null

post_processing_videos:
  eks_multiview:
    run: False
    overwrite: False
    n_latent: 3
    non_linear: True
    output_folder_name: non_linear_eks
  eks_singleview:
    run: False
    overwrite: False
  ensemble_median:
    run: False
    overwrite: False
  ensemble_mean:
    run: False
    overwrite: False

visualization:
  run: False
  overwrite: False
  ens_var_plots: False
  pixel_error_ens_std_plots:
    run: False
    overwrite: False
  eks_singleview_plots:
    run: False
    overwrite: False
"""


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()

    args.config_dir.mkdir(parents=True, exist_ok=True)
    for name, (_d, kind, cropped) in VARIANTS.items():
        cfg = args.config_dir / f"config_{name}.yaml"
        pipe = args.config_dir / f"pipeline_{name}.yaml"
        cfg.write_text(make_config(name, kind, cropped))
        pipe.write_text(make_pipeline(name, kind, cropped, args.seeds))
        print(f"wrote {cfg.name} and {pipe.name}")


if __name__ == "__main__":
    main()
