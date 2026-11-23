# Failure modes

Fifteen traps, each of which has cost a run in this project. Sorted roughly by
how much damage they do before you notice.

The dangerous ones are not the crashes. They are the cases where training
completes and the numbers are quietly wrong.

---

## Silent — training completes, results are meaningless

### 1. A derived config inherits another dataset's `camera_params_file`

**Symptom:** `AssertionError` in `datasets.py` — calibration rows do not match
label rows (e.g. 414 vs 207). Or, if the counts happen to agree, **no error at
all** and every frame is paired with the wrong camera matrices.

**Cause:** `camera_params_file` is an absolute path, unlike the relative
`csv_file` / `bbox_file`. Copy a config to make a variant and it keeps pointing
at the original dataset.

**Fix:** derive the path from the dataset name; never copy it. Generate configs
from a script rather than hand-maintaining near-identical files. Verify with
`camera_params_file == f"{data_dir}/calibrations.csv"`.

### 2. The animal token is not where the code assumes

**Symptom:** `ValueError: Directory '..._an1_Camera1' carries none of
animal_names=['an1','an2']` — or, worse, a substring match silently groups the
wrong rows.

**Cause:** two-mouse puts the animal last, r24m puts it before the view. Suffix
matching works on one and fails on the other.

**Fix:** match whole underscore-separated tokens at any position. See
`file-formats.md` § Session directory naming for the three call sites.

### 3. Predictions come out in crop pixels, not full-frame pixels

**Symptom:** merged predictions land near the image origin; errors in the
hundreds of pixels.

**Cause:** `lightning_pose/train.py` sets `bbox_files = None` **deliberately**
before predicting on the label csvs, so labelled predictions are in stored-crop
pixels (320) even though the config has a `bbox_file`. Other prediction paths do
pass the boxes and come out already in full-frame pixels.

**Fix:** never assume. Detect per file by testing which reading places the
keypoints inside their own box — the two readings differ by hundreds of pixels,
so it is not a close call. `two_mouse_merge_predictions.py --pred-space auto`
does this and prints what it chose (typically 100% vs <3%).

### 4. The 3D reprojection loss is silently off

**Symptom:** a "3D loss" run that is indistinguishable from the control.

**Cause:** `supervised_reprojection_heatmap_mse` runs at constant weight and is
**not** gated by `AnnealWeight` — `total_unsupervised_importance` scales only
the *unsupervised* losses. But if the loss is not configured, or camera params
are absent, it simply does not appear.

**Fix:** grep the training log in the first minute for
`train_supervised_reprojection_heatmap_mse_loss` and confirm it is present and
non-zero.

### 5. Epoch-based schedules used with step-based numbers

**Symptom:** the learning rate never decays; the backbone never unfreezes.

**Cause:** `training.lr_scheduler_params.multisteplr.milestones`,
`max_epochs` and `unfreezing_epoch` are **epochs**. `pipeline_simple.py`'s
`make_model_cfg()` deletes them and substitutes step-based keys
(`min_steps`/`max_steps`/`milestone_steps`/`unfreezing_step`) from the pipeline
config. Run the same config standalone with `litpose train` and step-shaped
numbers sit in epoch-shaped fields, silently doing nothing.

**Fix:** keep epoch-shaped values in the LP config and step-shaped values in the
pipeline config. Prefer `pipeline_simple.py` for real runs.

### 6. Two runs collide in the output directory

**Symptom:** a run's results are overwritten, or mysteriously reflect a
different step budget.

**Cause:** results land in
`{outputs_dir}/{intermediate_results_dir}/{model_type}_{n_hand_labels}_{seed}`.
Two runs with the same `(model_type, n_hand_labels, seed)` but different step
budgets write to the same place.

**Fix:** make `intermediate_results_dir` unique per experiment.

### 7. Both animals of a frame land in different splits

**Symptom:** validation error suspiciously low; the joint model appears to
generalize better than it does.

**Cause:** the two rows of a frame share a scene. Splitting them independently
leaks the frame from train into val.

**Fix:** set `animal_names` so `split_indices_by_group` keeps them together.
Confirm in the log: `Grouped splits over G groups` and
`kept R rows (G complete groups)`. Note that *independent per-animal models*
legitimately draw independent splits — a split disagreement when merging two
single-animal models is expected, not a defect.

### 8. `n_hand_labels` silently means something other than it says

**Symptom:** the run directory says `..._500_0` but the model trained on 496.

**Cause:** `compute_num_train_frames` clamps `train_frames >= len_train_dataset`
down to the split size and prints only
`Warning! Requested training frames exceeds training set size; using all`.
The comparison is `>=`, so an exactly-correct value also triggers the warning.

**Fix:** measure the split size (`int(train_prob * n_rows)`) and use it exactly.
Remember `n_hand_labels` counts **rows**: a joint dataset needs twice the number
a per-animal one does to see the same frames.

---

## Loud — but easy to misread

### 9. Calibration matched positionally, checked only by basename

**Symptom:** none, until results are inexplicable.

**Cause:** Lightning Pose asserts only that the *basenames* of the label and
calibration indices line up, positionally. A reordering that preserves basenames
passes.

**Fix:** build `calibrations.csv` by iterating the label index in order, never
by joining or sorting. Verify explicitly.

### 10. Keypoints cropped out of their own image

**Symptom:** labels outside `[0, size)`; unlearnable keypoints; quietly inflated
error.

**Cause:** sizing a box from the keypoint *mean* rather than the extent, or
relying on a generous crop ratio instead of a guarantee.

**Fix:** centre on the **midpoint of the extent**, size from the extent, then
*grow the box* until it provably contains every keypoint after integer rounding,
and assert per crop at build time. Containment must be structural. Verify
afterwards: 0 outside, and report the minimum margin (for ratio `r` and size
`S`, expect `S(1 - 1/r)/2`).

### 11. A dataset variant is not actually comparable to its base

**Symptom:** a difference in results that is not attributable to the
intervention.

**Cause:** regenerating labels, boxes or calibrations for a variant instead of
copying them.

**Fix:** copy geometry verbatim; filter rows, never recompute. Verify the
variant's csvs are **byte-identical** to the base (`filecmp.cmp(..., shallow=False)`).

### 12. The "no keypoint was painted" check does not work on colour images

**Symptom:** the assert passes on a dataset where labels *were* painted, or
fires constantly on one where they were not.

**Cause:** the two-mouse check tested whether a keypoint's pixel had become
redder than its neighbours. That is valid only because those frames were
greyscale stored as three channels, so any channel separation meant paint. r24m
frames are genuinely colour (mean `|R-G| ≈ 12`): the test both misses paint and
fires on bedding.

**Fix:** assert against the **paint mask** directly, in full-frame coordinates,
before cropping. Exact regardless of what the frame's colours are.

### 13. `pkill -f <pattern>` matches its own command line

**Symptom:** the command exits with a signal status (e.g. 144) having killed
itself; or an `until ! pgrep -f ...` loop never terminates.

**Cause:** `-f` matches the full command line, and the pattern is *in* the
command line of the shell running it.

**Fix:** bracket a character — `pkill -f 'run_queue[.]sh'` — or kill by PID from
a prior `ps`.

### 14. Two copies of the experiment runner

**Symptom:** `unknown experiment '<name>'` while a queue table still prints.

**Cause:** `launch_queue.sh` executes `ROOT/scripts/run_experiments.py`, not the
source copy in `lp3d-analysis/scripts/studio/`.

**Fix:** after editing `EXPERIMENTS`, copy the file to the studio root. Check
`logs/queue.log` after launching — the detached process can fail silently.

### 15. Crop-space numbers compared as if they were pixels

**Symptom:** a cropped model looks better or worse than it is; per-view errors
that do not line up with what the images show; an ensemble-variance plot with a
suspiciously clean trend.

**Cause:** a crop pixel is worth `w / size` full-frame pixels and `w` varies per
row — ~10x across r24m. Lightning Pose's own `_pixel_error.csv` is in crop
pixels. Comparing those across rows, across views, or against a full-frame model
compares different units.

**Fix:** convert everything with the merge script before reporting. Positions
take the offset and scale; error and standard deviation take the scale;
**variance takes the scale squared**; likelihood takes none. And when
correlating ensemble std against pixel error, remember the shared per-row factor
inflates the correlation — check both spaces or control for box size.

---

## Two things that look like bugs and are not

**1 ULP differences after a CSV round trip.** Filtering rows through pandas and
re-serializing can shift a value by ~`5.7e-14` px (one ULP, `2.5e-16` relative).
Compare label values with a tolerance (`< 1e-9` px), not for exact equality.
Box and integer columns *should* match exactly.

**A `set` column in OOD prediction files.** Lightning Pose writes `set` into
out-of-distribution prediction files too. Bucketing OOD rows by it produces
nonsense "train/validation" splits of held-out data. Detach it and ignore it for
OOD.

---

## Checks that must never be allowed to return NaN

A verification that reports `nan` reads like a pass. Guard every summary
statistic against unlabelled keypoints:

```python
finite = ~np.isnan(a).any(axis=-1)
pts = a[finite]                      # not a.min(), which propagates NaN
margin = float(min(pts.min(), (size - 1) - pts.max()))
```

Same for pixel errors, mask fractions, and anything averaged over keypoints.
"Not measured" and "no problem found" must never look alike.
