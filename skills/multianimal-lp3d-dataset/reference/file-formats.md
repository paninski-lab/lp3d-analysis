# Exact file formats

A dataset directory that Lightning Pose 3D can train on:

```
data/<dataset>/
  CollectedData_<view>.csv          labels, in-distribution
  CollectedData_<view>_new.csv      labels, out-of-distribution
  bboxes_<view>.csv                 cropped datasets only
  bboxes_<view>_new.csv
  calibrations.csv                  per-row lookup, in-distribution
  calibrations_new.csv
  calibrations/                     directory of .toml camera groups
  labeled-data/<session-dir>/<img>  the images
  videos/                           may be empty
  provenance.json                   how this was built
```

One `CollectedData_<view>.csv` and one `bboxes_<view>.csv` per view, per split.
The `_new` suffix marks out-of-distribution; it is a convention, not a flag.

---

## `CollectedData_<view>.csv`

Three header rows, then one row per labelled instance.

```
scorer,mocap,mocap,mocap,mocap,...
bodyparts,HeadF,HeadF,HeadB,HeadB,...
coords,x,y,x,y,...
labeled-data/<session>_an1_Camera1/img00000877.png,58.31,270.48,78.99,238.47,...
```

- Read with `pd.read_csv(path, header=[0,1,2], index_col=0)`.
- Two columns per keypoint (`x`, `y`) in labels; prediction files add
  `likelihood` and a trailing single-level `set` column.
- The index is the image path **relative to `data_dir`**.
- Missing keypoints are empty cells, read as NaN. A frame whose keypoints are
  *entirely* NaN looks to pandas like an index-name row and is silently dropped,
  shifting every subsequent row out of alignment with the calibration index.
  Always read through `lightning_pose.utils.io.fix_empty_first_row`.

**Combined arrangement** — one row per frame, keypoints suffixed per animal
(`HeadF_an1 ... ShoulderR_an1, HeadF_an2 ... ShoulderR_an2`). `2 x K` keypoints.

**Per-animal arrangement** — one row per (frame, animal), keypoints unsuffixed
(`HeadF ... ShoulderR`). `K` keypoints. Identity lives in the session directory
and nowhere else.

### Row order

Frame-major, animal-minor, and **identical across every view and the calibration
index**. Views are matched positionally by row, not by name. The camera name is
the only part of a row's path allowed to differ between views.

---

## `bboxes_<view>.csv`

```
,x,y,h,w
labeled-data/<session>_an1_Camera1/img00000877.png,96,242,223,223
```

- Single header row; index **exactly equal to the label csv's index, in order**.
- `x, y` is the top-left corner, `h, w` the size, all in **original full-frame
  pixels** — not crop pixels.
- Boxes are normally square (`h == w`).

This file is what lets a model train on crops while the 3D reprojection loss
works in full-frame coordinates: `convert_bbox_coords` maps predictions back
before the loss sees them.

Mapping between the two spaces:

```python
# original -> stored crop of side `size`
x_crop = (x_orig - bbox_x) / bbox_w * size
y_crop = (y_orig - bbox_y) / bbox_h * size

# stored crop -> original
x_orig = x_crop / size * bbox_w + bbox_x
y_orig = y_crop / size * bbox_h + bbox_y
```

Crops are cut with zero-padding beyond the frame edge, never by clamping the
box: a clamped box would shift the animal off centre and break the
correspondence between the recorded box and what the crop shows.

---

## `calibrations.csv`

```
,file
labeled-data/<session>_an1/img00000877.png,calibrations/<session>_an1.toml
```

- One row per label row, **in the same order**.
- The index conventionally drops the camera (a row is one scene seen from every
  view at once) but keeps the animal.
- `file` is relative to `data_dir`.

Lightning Pose matches this **positionally** and asserts only that the image
*basenames* line up:

```python
img_idxs_labels = [i.split('/')[-1] for i in dataset[view_names[0]].image_names]
img_idxs_calib  = [i.split('/')[-1] for i in cam_params_df.index]
assert np.all(img_idxs_labels == img_idxs_calib)
```

So a row-count mismatch fails loudly, but a *reordering* that preserves
basenames does not — it silently pairs frames with the wrong camera matrices.

---

## `calibrations/<name>.toml`

```toml
[cam_0]
name = "Camera1"
size = [ 1328, 1048,]
matrix = [ [ ... ],]
distortions = [ ... ]
rotation = [ ... ]
translation = [ ... ]

[cam_1]
name = "Camera2"
...
```

`cam_0 ... cam_{N-1}` in order. Lightning Pose asserts:

```python
assert np.all(camgroup.get_names() == view_names)
```

so the `name` fields must equal `cfg.data.view_names` **in exactly that order**.

Per-animal toml copies (`<session>_an1.toml`, `<session>_an2.toml`) are fine and
may be byte-identical — they just mean more camgroups get loaded.

---

## Session directory naming

```
labeled-data/<session>_<view>_<animal>/     two-mouse
labeled-data/<session>_<animal>_<view>/     r24m
labeled-data/<session>_<view>/              combined / uncropped
```

The animal may sit before or after the view. **Always match it as a whole
underscore-separated token at any position**, never as a suffix and never as a
substring:

```python
tokens = dir_name.split("_")
hits = [i for i, t in enumerate(tokens) if t == animal]
```

A substring test lets `an1` match a session called `..._an10_...`. A suffix test
works on two-mouse and raises on r24m.

Three places in this codebase depend on this and were all generalized to token
matching:
- `lightning_pose/data/datasets.py::_strip_animal_from_path` (group ids)
- `scripts/postprocess/two_mouse_merge_predictions.py::split_instance_path`
- `scripts/preprocess/two_mouse_split_animals.py::has_animal`

The filename itself never carries the animal, so anything parsing a frame index
out of it keeps working.

---

## `provenance.json`

Not read by Lightning Pose, but `output_size` is read by the SAM builder and the
merge script, and the rest is how you answer "what is this dataset" six weeks
later. Record at minimum:

```json
{
  "source": "...",
  "arrangement": "per-animal crop",
  "crop_ratio": 1.4,
  "output_size": 320,
  "animals": ["an1", "an2"],
  "views": ["Camera1", "Camera2", "Camera3", "Camera5", "Camera6"],
  "lp3d_analysis_sha": "...",
  "splits": {"ind": {"n_rows": 522, "n_frames": 261}, "ood": {...}}
}
```

Keep it truthful. A stale `bbox_definition` string describing an algorithm that
was since replaced is worse than no string at all.

---

## Config fields that must match the data

```yaml
data:
  data_dir: /abs/path/data/<dataset>
  video_dir: /abs/path/data/<dataset>/videos
  camera_params_file: /abs/path/data/<dataset>/calibrations.csv   # ITS OWN
  csv_file:   [CollectedData_<view>.csv, ...]     # view order
  bbox_file:  [bboxes_<view>.csv, ...]            # cropped only, same order
  view_names: [<view>, ...]                       # == toml name order
  num_keypoints: K            # or 2K for combined
  keypoint_names: [...]       # == the csv's bodyparts, in order
  animal_names: [a1, a2]      # joint per-animal-row datasets ONLY
  mirrored_column_matches: [0, 1, ..., num_keypoints - 1]
training:
  train_prob: 0.95
  val_prob: 0.05
  train_frames: 1             # overridden by the pipeline's n_hand_labels
```

`animal_names` present means "rows are per-animal": it builds group ids so both
animals of a frame land in the same train/val split. Set it on a single-animal
dataset and every row becomes its own group for no reason; set it on a combined
dataset and `_strip_animal_from_path` raises on the first row.
