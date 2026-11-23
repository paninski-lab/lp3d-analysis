---
name: convert-singleanimal-multiview-dataset
description: Explore a raw single-animal multiview pose dataset in its native format, write a conversion script, and convert it to Lightning Pose format (CollectedData_<cam>.csv, labeled-data/, videos/, calibrations). Use when the user points at a new raw multiview dataset (a path, a download link, or a paper) and wants it preprocessed for lp3d-analysis.
---

# Convert a raw single-animal multiview dataset to Lightning Pose format

Scope: one animal per frame. Multi-animal datasets (e.g. 3d-pop, r24m) belong to the
`convert-multianimal-multiview-dataset` skill; if the raw data is multi-animal, use that skill
(its cropped output is what makes a multi-animal dataset single-animal for training).

Goal: given a path to a raw multiview dataset, explore it, write
`scripts/preprocess/<dataset>.py` (plus helpers in `lp3d_analysis/preprocess/<dataset>.py` if
non-trivial), register the dataset in `lp3d_analysis/dataset_info.py`, run the conversion, and
verify it visually. Existing scripts in `scripts/preprocess/` are the reference implementations;
read the one closest to the new dataset before writing anything:

| Raw data looks like | Closest reference |
|---|---|
| Hand-labeled 3D (Label3D .mat), no frame indices, must match frames to video | `chickadee.py` |
| 3D mocap + camera params in .mat (Matlab convention), huge videos | `rat7m.py` |
| No hand labels; 2D/3D predictions used as labels, filter by reprojection error | `fly-anipose.py` |
| Multi-animal, chunked videos, json calibration -> toml | `r24m.py`, `r24m-crop.py` |
| Hand-labeled 2D per camera, videos are spliced; cropping | `rat-pup.py` |

## Target format

Output goes to `/media/mattw/multiview-data/<dataset>/` (raw data lives in `.../_raw/<dataset>/`).
Confirm both paths with the user if unclear.

```
<dataset>/
  CollectedData_<cam>.csv          # one per camera, in-distribution (InD) labels
  CollectedData_<cam>_new.csv      # one per camera, OOD labels (only if there is an InD/OOD split)
  labeled-data/<session>_<cam>/img%08d.png   # labeled frames AND context frames (+-2)
  labeled-data-check/              # verification plots (plot_labeled_frames)
  videos/<session>_<cam>.mp4       # InD videos;  videos_new/ for OOD
  calibrations/<session>.toml      # anipose-format toml (or calibrations.csv for rat7m-style)
  bboxes_<cam>.csv                 # only for cropped variants: columns x, y, h, w (orig pixels)
```

Label CSV rules (these are what Lightning Pose expects; get them exactly right):
- 3-level header `(scorer, bodyparts, coords)`, coords are `x`, `y`; read with
  `header=[0, 1, 2], index_col=0`. No likelihood columns.
- Index = relative image path, `labeled-data/<session>_<cam>/img00001234.png`; the number is the
  frame index in the corresponding video, zero-padded to 8 digits.
- Unlabeled / invalid keypoints are NaN, never 0 or -1.
- **All cameras must have the same rows in the same order**, differing only by the `<cam>` suffix
  in the path. Multiview training breaks otherwise. Assert this at the end of the script.
- Keypoint order is identical across cameras and matches `dataset_info[...]['keypoints']`.

## Workflow

1. **Explore the raw data first; don't write conversion code yet.** Report findings to the user.
   - `ls`/`find` the tree, read any README/paper/download page, note file formats.
   - Load one label file and one calibration file interactively; print shapes, keys, dtypes.
   - Get video info with `ffprobe`/cv2: resolution, fps, frame count, per camera. Check frame
     counts agree across cameras and with the label array length (r24m had short final chunks
     that broke sync; decide to drop rather than guess).
   - Identify: cameras, keypoint names + skeleton, sessions/animals, 2D vs 3D labels, are labels
     hand-made or model predictions, are frame indices recorded, coordinate/axis conventions.
   - Look at 2-3 raw frames (Read the png) to confirm orientation, color, and view layout.
2. **Resolve ambiguities with the user** (AskUserQuestion, batched into one call) before
   building. Typical open questions: InD vs OOD session split; how many frames per session and
   how to select them (k-means on poses, as in rat7m/fly-anipose/r24m); filtering thresholds;
   whether to also make a cropped variant; multi-animal handling; where outputs go.
3. **Calibration.** If 3D labels or calibration exist, convert to anipose toml and sanity-check:
   project 3D points into each camera with the converted calibration and overlay on a real
   frame (see `check_calibration_reprojection` in `scripts/preprocess/r24m.py` and
   `project_points` in `lp3d_analysis/preprocess/r24m.py`). Watch Matlab row-vector conventions
   (transpose K; negate rotation vector when applying `R.T`), distortion order
   `[k1, k2, p1, p2, k3]`, and units (mm vs m). If reprojection is off by more than a few
   pixels, stop and debug; don't proceed with bad calibration.
4. **Frame selection.** Prefer diverse frames: filter out bad ones (reprojection error, bad
   mocap flags, missing keypoints), then k-means on 3D (or 2D) poses and take the frame nearest
   each center, seed fixed (`np.random.seed(0)`). If labels are 3D, project to 2D per camera and
   set keypoints outside the image or with high reprojection error to NaN.
5. **Write the script** `scripts/preprocess/<dataset>.py`:
   - Module docstring: raw data source/URL, what each step does, what manual steps remain.
   - Path config at the top (`base_dir`, `save_dir`), then step flags
     (`run_extract_frames = True`, ...) so steps can be rerun independently.
   - Reuse `lp3d_analysis.video.export_frames` (use `context_frames=2`, `reader='imageio'` if cv2
     seeking is imprecise, as with chickadee avis) and `get_frames_from_idxs`; reuse
     `export_cropped_frames` for crops. Don't reimplement them.
   - Put reusable helpers in `lp3d_analysis/preprocess/<dataset>.py`, one-off glue in the script.
6. **Register the dataset** in `lp3d_analysis/dataset_info.py`: `InD_animals`, `OOD_animals`
   (can be empty lists), `cam_names`, `keypoints`, `skeleton` (array of index pairs),
   `skeleton_colors` (one per edge), `w3d`, `crop_size`. Do this before running, since the
   script reads from it.
7. **Dry run on a small subset** (one session, a few frames). Inspect output, then run in full.
   Long jobs (video decoding of many sessions) go in the background.
8. **Verify visually.** Call `plot_labeled_frames` (from `lp3d_analysis.preprocess`; handles 4 or
   6 cameras) to write `labeled-data-check/`, then actually open several of the images with
   Read and confirm keypoints sit on the right body parts in every view. Also run a programmatic
   check: all CSVs have identical row sets modulo camera suffix, every index path exists on
   disk, NaN fraction per keypoint is sensible, label coordinates fall within image bounds.
9. **Report** to the user: what was produced (counts of sessions/frames/cameras), judgment calls
   made, NaN stats, anything suspicious, and the remaining manual steps below.

## Remaining downstream steps (not part of conversion; mention, don't silently do)

Scripts referenced from existing preprocessing docstrings: split InD/OOD
(`split_data.py`), video snippets for each OOD frame for EKS (`video_snippets_eks.py`),
unlabeled videos for unsupervised losses (`video_snippets_unlabeled.py`), and for cropped
variants, training a detector and generating cropped videos
(`video_snippets_unlabeled_cropped.py`). Some of these may not be in the repo; check first.
After conversion, a pipeline config in `configs/` (see `config_fly-anipose.yaml`) is needed to
run training.

## Cautions

- Never modify or delete anything under `_raw/`. Write only to the output dir.
- Don't overwrite an existing output dataset without asking.
- Frame-index alignment is the most common silent failure (off-by-one, dropped frames, spliced
  videos, 0- vs 1-based indexing from Matlab). Verify with images, not just shapes.
- Be honest in the report about anything not verified; don't claim a dataset is converted
  correctly on the basis of the script running without errors.
- Match repo style: pathlib, numpy-style docstrings, 100-char lines, single quotes.
