---
name: convert-multianimal-multiview-dataset
description: Explore a raw multi-animal multiview pose dataset (2+ animals per frame, possibly a different number per session) in its native format, choose train/test sequences and a diverse set of whole time points, and convert it to Lightning Pose format as a full-frame dataset plus a per-animal cropped dataset (CollectedData_<cam>.csv, bboxes, labeled-data/, calibrations). Use when the user points at a new raw multi-animal multiview dataset (a path, download link, or paper) and wants it preprocessed for lp3d-analysis.
---

# Convert a raw multi-animal multiview dataset to Lightning Pose format

For one animal per frame use `convert-singleanimal-multiview-dataset` instead. This skill builds on
it: read `.claude/skills/convert-singleanimal-multiview-dataset/SKILL.md` first for the shared
steps (explore before writing code, batch questions to the user, calibration conversion and
reprojection check, dry run, honest reporting, repo style). Only what differs for multiple animals
is spelled out here.

Goal: given a raw dataset, produce **two** datasets and the code that makes them:
- `<dataset>/`: full frames, **one label/bbox row per instance** (rows of one time point share an
  image path).
- `<dataset>-crop/`: one cropped image per instance (unique rows). This is what Lightning Pose
  trains on, because its dataloader does **not** crop by bbox: the image file is read as-is and the
  bbox is only metadata to map predictions back to the frame.

## Vocabulary

- **time point**: one (session, frame). **instance**: one animal at one time point (its frame,
  box, 2D keypoints per view). A time point with N animals has N instances.
- **session**: one recording/sequence. **animal id**: a stable identifier within a session (a tag
  number, not a slot index, whenever the raw data has one).
- Always select **whole time points**: every animal present at a selected time point is included.
  Never take one animal and drop the others at the same time. Budget in instances, convert to time
  points (`time points = instances / animals per frame`), and split the budget across animal counts
  (e.g. 250 instances from 2-animal sessions = 125 time points; 250 from 5-animal = 50).

## Reference implementations

| | `3d-pop` (default template) | `r24m` (different raw format) |
|---|---|---|
| Code | `scripts/preprocess/3d-pop.py`, `3d-pop-crop.py`, `lp3d_analysis/preprocess/pop3d.py` | `scripts/preprocess/r24m.py`, `r24m-crop.py`, `lp3d_analysis/preprocess/r24m.py` |
| Animals | 2 or 5, varies by session; ids are tags | exactly 2, anonymous slots `an1`/`an2` (real rats in `metadata.csv` Rat1/Rat2) |
| Raw labels | 2D keypoints + bboxes per camera, 3D per camera, ids per column | 3D mocap only (`markerDataset.csv`); 2D labels = projection through calibration |
| Calibration | OpenCV pickles (rvec, tvec, K, dist), mm | Matlab json (row-vector convention), needs transpose/negation |
| Video | one mp4 per camera, row index == frame index | chunked mp4s (3500 frames), short final chunk must be dropped |
| Quality flags | NaNs in filtered 2D, degenerate bboxes, per-camera 3D disagreement | `goodFrame_an1/an2` flags |
| Output format | **instance rows** (use this for new data) | older **wide** format: one row per frame, bodyparts `<kp>_<an>` (fixed N only) |

Map the new dataset onto one of these, but produce the 3d-pop-style output regardless of the raw
format. Use the r24m wide format only if the user asks for it.

## Target formats (default)

Full dataset `<dataset>/` (`/media/mattw/multiview-data/<dataset>/`; raw in `.../_raw/<dataset>/`;
confirm both with the user):
```
CollectedData_<cam>.csv        train rows (one per instance, full-frame pixels)
CollectedData_<cam>_new.csv    test/OOD rows
bboxes_<cam>[_new].csv         columns x, y, h, w (full-frame pixels), same rows/order as labels
labeled-data/<session>_<cam>/img%08d.png     full frame; number = frame index in the raw video
calibrations/<session>.toml    anipose toml, per session
labeled-data-check/            verification plots
README.md
```
Cropped dataset `<dataset>-crop/`: same files, except
```
labeled-data/<session>_id<animal>_<cam>/img%08d.png   crop resized to crop_size
CollectedData_<cam>[_new].csv  keypoints in crop pixels
bboxes_<cam>[_new].csv         the SAME boxes as the full dataset (full-frame pixels)
calibrations/<session>_id<animal>.toml   one identical copy per animal in the session
labeled-data-check/            flat directory (see below)
```
Rules (assert all of them in code, reading the files back from disk):
- 3-level header `(scorer, bodyparts, coords)`, keypoint names **without** an animal suffix; NaN
  (not 0/-1) for missing keypoints.
- Every camera has the same rows in the same order, differing only in the `<cam>` suffix of the
  path. Animals at a time point are in one fixed order (e.g. sorted by id) in every file.
- Full dataset: image paths repeat across the rows of a time point (expected). Crop dataset: no
  duplicate paths. Animal ids are not in the full-dataset CSVs; keep a selection table
  (`selection.csv` with the id list per time point) to recover them.
- Train = in-distribution = no suffix; test/OOD = `_new`. No context frames and no videos unless the
  user asks. Mention that videos (for unlabeled losses/EKS) and `split_data` are not produced.
- `plot_labeled_frames` does **not** work on the full dataset (it does `.loc[index]`, which returns
  several rows for a repeated path); write a dedicated overlay check.

## Workflow

1. **Explore the raw data** (see the single-animal skill) with these extra questions; report
   findings before writing code:
   - How are animals represented: named columns per animal (ids), anonymous slots, or tracks? Is
     the id stable across sessions (same physical animal)? How many animals per session, and does
     it vary? Get the id <-> animal mapping from metadata, and check the id's suffix (3d-pop ids
     look like `483_1307` = tag + recording date; the tag is, by inference from the metadata, the
     animal, so confirm with the user or the dataset docs that it is stable across days).
   - What exists per animal and per view: 2D labels, boxes, 3D, visibility, quality flags? Are
     2D labels hand-made, model-made, or **projections of 3D** (then reprojection error is ~0 by
     construction and says nothing about label accuracy)?
   - Quantify missingness **per view and per animal** (NaN keypoints, whole-animal dropouts,
     out-of-image points, zero/negative-size boxes). Report per-sequence rates; propose excluding
     sequences that are mostly unusable.
   - Alignment of annotation rows to video frames: compare row counts with frame counts for every
     sequence/camera, and prefer the **full-sequence** annotations/videos over pre-cut
     train/val/test chunks (3d-pop's chunk videos were each one frame short, which would have
     shifted labels). Verify the final extraction against an independent decoder (see 7).
   - Do annotation files that should agree across cameras actually agree (e.g. per-camera 3D
     files differing by about a frame of motion)? Quantify in mm and relate to animal speed.
2. **Ask the user** (one batched AskUserQuestion). Typical: which animal counts/sessions to use
   and which to exclude; instance budget per split and per animal count; train/test split rule
   (see 5); the box recipe and crop size (see 8); filters/thresholds (see 4); context frames and
   videos; output paths.
3. **Calibration**: convert to anipose toml (see single-animal skill; OpenCV pickles map directly,
   Matlab needs transpose/negation). Triangulate the 2D labels and reproject with the **written
   tomls**; assert the max error on the selected data is small (<= 2 px) and report percentiles.
4. **Candidate time points**. A time point is eligible only if the per-frame filter passes for
   **every animal**: all keypoints present in all views, valid box (positive size) in all views,
   valid 3D, and agreement between redundant sources (cross-camera 3D within a few mm). Also
   require the same for **t-1 and t+1**: neighbours with missing labels mark fast motion/blur, where
   keypoints (especially head) are least accurate. Before fixing thresholds, report how many time
   points each filter removes per sequence and whether the filter biases toward slow frames.
5. **Splits**. Split **sessions** (not frames) into train and test, seeded, stratified by animal
   count, so test sessions never overlap train sessions. Then, always:
   - Count how many distinct animals there are, how often each recurs across sessions, and how
     much of the test set contains animals seen in train. Report it; never claim animal-level
     separation unless it was enforced.
   - If the user wants an animal-disjoint split, search partitions of the animals (usable
     sessions are those whose animals all fall on one side) and show what it costs before
     committing (3d-pop: 18 birds, one connected component, ~19/30 sessions kept with a single
     5-animal session per side).
6. **Frame selection** on whole time points (reuse/generalize `pop3d.select_time_points`, `ego_pose`,
   `separation`, `kmeans`, `pca_project`; `ego_pose` uses pigeon-specific heading keypoints, so
   adapt it). Two diversity axes, per pool (split x animal count):
   - *Inter-animal distance*: stratified sampling over quantile bins (5) of one scalar per time
     point: centroid distance for 2 animals, mean nearest-neighbour distance for N > 2 (also look
     at the minimum pairwise distance).
   - *Pose*: ego-centric alignment from 3D (centroid origin, rotate about the vertical axis so the
     heading is +x, scale by body length), k-means (K~30) on all animals of all candidate time
     points, then greedy selection of time points that most reduce the gap to a square-root
     flattened target cluster histogram, with a minimum frame gap per session (60 frames) and a
     per-session cap (`ceil(1.5 * n / n_sessions)`; a 20% cap is infeasible for pools of 5 sessions).
   - Report per pool: sessions used, distance-bin counts, pose-cluster coverage vs a stratified
     random baseline, and instance counts. Fix the seed. Drop empty/outlier clusters from coverage
     claims. A rerun with different filters changes the selection a lot (greedy), so keep the old
     one (`v1_...`) when regenerating.
   - Write intermediate artifacts to `<raw>/frame_selection/` (`selection.csv` with split, animal
     count, session, frame, animal id list, separation, max reprojection error;
     `sequence_split.json`; pool and k-means pickles; README). Never overwrite without asking.
7. **Build the full dataset** (follow `scripts/preprocess/3d-pop.py`): step flags
   (`run_select`, extract, labels, calibrations, check) plus `--dry-run` (writes `<name>_dryrun`
   with a few time points per pool) and a refusal to write into a non-empty output dir.
   - Extract exactly the selected frames, colour, no context frames (`export_frames(...,
     grayscale=False, context_frames=0)`; chunked videos need chunk-aware reading as in
     `lp3d_analysis/preprocess/r24m.py`).
   - 3D-only raw data (r24m): project 3D to each camera with the converted calibration, set
     points outside the image to NaN, and note labels are projections.
   - Boxes: derived from each animal's keypoints in each view (square, centered on the tight
     keypoint extent, side = scale x larger extent; `pop3d.keypoint_bbox`) rather than raw dataset
     boxes, so they match what a detector produces and equal the crop dataset's. Pick the scale with
     the user (1.4 for rats, 2.0 for pigeons, which are small in the frame). Report the box-size
     distribution and how many boxes leave the frame (crops black-pad).
   - Verify extraction by comparing a sample of exported frames with an independent decode
     (`ffmpeg -vf select=eq(n\,IDX)`): difference must be exactly 0.
   - Register in `lp3d_analysis/dataset_info.py` (sessions as `InD_animals`/`OOD_animals`, cam_names,
     keypoints without animal suffix, skeleton, colors, `crop_size`); keep the file diff
     additive.
8. **Build the cropped dataset** (follow `scripts/preprocess/3d-pop-crop.py`): read the boxes and
   labels **from the full dataset** (so the boxes are identical by construction), cut each box out
   of the full frame (black-pad outside), resize to `crop_size` (default 320x320, bilinear), convert
   labels to crop pixels `(xy - [x, y]) / [w, h] * crop_size`, write unique rows under
   `<session>_id<animal>_<cam>`, copy the session toml once per animal. The tomls stay in original
   frame coordinates (not adjusted for the crop). Re-derive the row -> (session, frame, animal)
   mapping from the selection table and assert it matches the CSV index. Animal ids in folder names
   must be unique within a session (use the tag, not tag + date).
9. **Verify visually and programmatically** (look at the images with Read; don't trust "it ran"):
   - Full dataset: overlays of all animals' boxes and labels, one colour per animal, 4 views;
     check the same animal keeps its colour across views (instance order).
   - Frame alignment on **fast motion**: take the fastest animal in the sample, overlay labels
     from t-1, t and t+1 on the image at t in all views; t must fit best everywhere.
   - Crop dataset: a **flat** `labeled-data-check/` (no subdirectories) with one numbered image per
     selected time point, in paging order (train then test): one row per animal and one column per
     view; each cell is the crop with the coloured skeleton obtained by mapping the labels to the
     full frame through the bbox, triangulating with the copied toml, reprojecting, and mapping
     back; show the original labels as small dots and the per-view reprojection error. This
     exercises labels, boxes and calibration at once.
   - Programmatic: row counts equal expected instances; rows identical across views; no NaN
     labels; paths exist; crop bboxes equal the full bboxes exactly; keypoints inside the crop;
     worst reprojection error below the threshold.
10. **Write `README.md` in both dataset dirs** (assume a user sees only one): directory
    structure, file formats, how to map crop coordinates back (`x_frame = x + x_crop * w /
    crop_w`), calibration conventions and units, how data was filtered/selected, exclusions, and the
    caveats: no animal-level separation (with the measured overlap), labels not hand-annotated
    (propagated/projected), head accuracy under motion blur, upsampling in crops, unequal animal
    representation, what is not included (videos, context frames, split_data, config).
11. **Report** to the user: counts (sessions, time points, instances per split and per animal
    count), judgment calls, filter yields, what was verified and what was not, caveats, remaining
    steps (videos for unlabeled losses/EKS, `split_data`, pipeline config in `configs/`).

## Other raw-data situations

- **3D only** (r24m): build instances by projecting 3D per animal; anonymous slots (`an1`/`an2`)
  carry no identity across sessions, so use real identities from metadata where they exist and say
  in the README that animal-level separation can't be assessed otherwise.
- **2D only per view, no calibration**: no triangulation check and no 3D loss; tell the user before
  proceeding. If animals are tracked per view without cross-view identity, cross-view association is
  needed first (stop and ask).
- **Different keypoint visibility** (visible/occluded flags): Lightning Pose supports a
  per-keypoint visibility column; ask before adding it.
- **Single-animal sessions mixed in**: keep them out of this pipeline unless asked (one animal per
  time point is just N=1, but the distance stratification does not apply).

## Cautions

- Never modify `_raw/`. Don't overwrite existing outputs or artifacts without asking; keep the old
  ones in a versioned folder when regenerating.
- Select whole time points only; keep the animal order fixed in every file.
- Frame alignment is the silent failure: verify with an independent decode and fast-motion overlays.
- Report honestly what was not verified; sample-based visual checks are samples.
- Repo hygiene: scripts with hyphenated names can't be imported, so shared code goes in
  `lp3d_analysis/preprocess/<dataset>.py`; run in the `pose` conda env (has `toml`); lint with
  `ruff check --line-length 100 --no-fix` (pyproject sets 79 and `fix = true`, which silently
  edits files, and existing scripts use ~100); format single-quoted; long jobs in the background;
  don't commit unless asked.
