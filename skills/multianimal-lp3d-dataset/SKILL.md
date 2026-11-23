---
name: multianimal-lp3d-dataset
description: Build, verify, configure and train Lightning Pose 3D datasets for multi-animal multiview recordings (two-mouse, r24m, and any new dataset in the same shape). Use when a new multi-animal dataset arrives, when creating cropped/per-animal/SAM-recoloured variants, when writing LP3D configs for them, or when predictions need mapping back to the original combined format.
---

# Multi-animal LP3D datasets

A new multi-animal dataset arrives as one or two archives. Turning it into
trainable LP3D datasets means producing up to **five conditions across seven
directories**, each of which must satisfy invariants Lightning Pose assumes but
mostly does not check. Most mistakes here are silent: the run trains to
completion and the numbers are meaningless.

Work in this order. Do not skip steps 0 or 4.

## 0. Ask what only the human knows

A handful of choices are research decisions, not facts, and guessing them
produces a dataset that builds cleanly, trains cleanly and answers the wrong
question. Get these before building — they are cheap to ask and expensive to
discover late:

1. **Which of the five conditions** are wanted (all five = 7 datasets = 7 runs
   per seed, most of a day on one GPU)
2. **Label budget** — all available, or fixed to match an earlier experiment
3. **Seeds per condition** — 1 ranks interventions separated by several pixels;
   ~1 px gaps need 3
4. **Crop ratio**, if crops are not shipped
5. **SAM recolouring strength** (`alpha` 0.3 keeps texture, 1.0 flattens)
6. **What the animal tokens mean**, and whether they are consistent across
   sessions
7. **Confirmation of the skeleton** you derived empirically
8. **Whether results must be comparable** to an earlier dataset — if so, step
   budget, batch size and label count are constrained, not free

`reference/intake-checklist.md` has a fill-in form, the reasoning behind each
question, the defaults you should *state* rather than ask, and — just as
important — the list of things to **never** ask because the data answers them.

The rule that governs step 1: **if the data can answer it, the data answers it.**
A description of a dataset, including the one that came with it, is a hypothesis
to check.

## 1. Survey before you build

Never assume the arrangement from the filename or from what you were told. Read
the archive and establish, for both the labelled and any cropped copy:

```bash
tar -tJf <archive>.tar.xz > /tmp/listing.txt   # then inspect the listing
```

Answer these, with evidence, before writing anything:

| Question | How to check |
|---|---|
| Which views? | The `CollectedData_*.csv` names. **Do not assume 0..N-1** — r24m is `Camera1,2,3,5,6` with no Camera4. |
| Suffixed or unsuffixed keypoints? | `pd.read_csv(f, header=[0,1,2]).columns.get_level_values(1)`. Suffixed (`HeadF_an1`) = combined, one row per frame. Unsuffixed = per-animal, one row per (frame, animal). |
| Where is the animal token in the directory name? | Before or after the view. two-mouse is `<session>_<view>_<animal>`; r24m is `<session>_<animal>_<view>`. **This is the single most common source of breakage.** |
| Is `calibrations.csv` present? | Usually **not**. It almost always has to be built. |
| Is the InD/OOD split by session or random? | Compare the session sets of `*.csv` and `*_new.csv`. By-session is a much harder OOD split and changes how results should be read. |
| If crops are shipped: what ratio and size? | Measure, don't guess: `320 / max_keypoint_extent` per row, and the `w` column of the bbox csv. |

Record the answers in the dataset's `provenance.json`.

## 2. Decide which of the five conditions you need

| Condition | Directory | Rows | Keypoints | Purpose |
|---|---|---|---|---|
| LP3D combined | `<ds>` | 1 per frame | `2 x K`, suffixed | Baseline: full frames, both animals in one row |
| SLP3D joint | `<ds>_s` | 2 per frame | `K`, plain | Per-animal crops, identity only in the path |
| SLP3D + SAM | `<ds>_s_sam` | 2 per frame | `K`, plain | Same crops, partner painted red |
| Per-animal plain | `<ds>_s-<a1>`, `<ds>_s-<a2>` | 1 per frame | `K`, plain | Upper bound on what removing identity ambiguity buys |
| Per-animal SAM | `<ds>_s_sam-<a1>`, `<ds>_s_sam-<a2>` | 1 per frame | `K`, plain | Does recolouring add anything on top of separation |

The per-animal pair is a *partition* of its joint dataset, never a rebuild. The
SAM dataset is its base with different pixels and **byte-identical everything
else** — that is what makes the comparison attributable to the recolouring.

## 3. Build

See `reference/file-formats.md` for the exact contents of every file.

```bash
cd /teamspace/studios/this_studio/lp3d-analysis

# a) organise the archives, build calibrations.csv, verify
python scripts/preprocess/r24m_prepare.py            # adapt paths for a new dataset

# b) if crops are NOT shipped, make them (guarantees containment structurally)
python scripts/preprocess/two_mouse_crop.py --crop-ratio 1.3

# c) partition the joint cropped dataset into one dataset per animal
python scripts/preprocess/two_mouse_split_animals.py \
    --source ../data/<ds>_s --animals <a1> <a2> --views <views...>

# d) recolour the partner with SAM2 (needs the UNCROPPED source + the crops)
python scripts/preprocess/r24m_sam.py \
    --source ../data/<ds> --geometry ../data/<ds>_s --output ../data/<ds>_s_sam

# e) partition the SAM dataset the same way as (c)
```

Rules that are not negotiable:

- **Never regenerate geometry you can copy.** The SAM build copies labels,
  boxes and calibrations verbatim from its base. The splitter filters rows, it
  does not recompute them.
- **Crops must contain their keypoints structurally, not by a generous ratio.**
  Size the box from the keypoint extent, then *grow it* until it provably
  contains every keypoint after integer rounding. Assert per crop at build time.
- **Paint may never cover a labelled keypoint.** Subtract the target's own mask,
  punch out the convex hull of its keypoints plus discs, then assert against the
  **paint mask** — not against pixel colour (see failure 12).

## 4. Verify — always, before training

```bash
python scripts/preprocess/verify_multianimal_dataset.py ../data/<ds>_s \
    --animals <a1> <a2> --views <views...> \
    --source ../data/<ds>          # enables the geometry checks

python scripts/preprocess/verify_multianimal_dataset.py ../data/<ds>_s_sam \
    --animals <a1> <a2> --views <views...> \
    --twin ../data/<ds>_s          # enables the recolouring checks
```

It exits non-zero on failure. It checks structure, cross-view row count **and
order**, bbox alignment, positional calibration alignment, camera order inside
every toml, image existence, keypoint containment, InD/OOD session disjointness,
and — given `--source` — that the labels really are the source keypoints mapped
through the boxes and the stored crops really are the source frames cut by those
boxes, bit-exactly.

Then prove the round trip *before* any model exists:

```bash
python scripts/postprocess/two_mouse_merge_predictions.py --verify \
    --data-dir ../data/<ds>_s --source ../data/<ds> \
    --animals <a1> <a2> --views <views...>
```

Merging the ground truth must reproduce the uncropped source at **0.000000 px**.
If it does not, no prediction merged later can be trusted.

## 5. Configure

Generate configs; do not hand-write seven of them (see failure 1).
`scripts/preprocess/r24m_make_configs.py` is the template — copy and adapt.

Per-dataset fields: `data_dir`, `camera_params_file` (**its own**, absolute),
`num_keypoints`, `keypoint_names`, `animal_names` (joint per-animal-row datasets
only), `bbox_file` (cropped only), `n_hand_labels`.

Everything else must be identical across conditions or the comparison is void.

**Label budget.** `n_hand_labels` counts **rows**, not frames. To give every
condition the same number of frames, a joint dataset needs twice the number a
per-animal one does. To train on *all* labelled data, set it to the measured
train-split size — `int(train_prob * n_rows)` — not to a round number above it
(see failure 8).

## 6. Train

```bash
python pipelines/pipeline_simple.py --config configs/pipeline_<ds>.yaml --for_seed 0
```

or queue every variant serially with `scripts/studio/run_r24m_queue.sh`, which
skips completed runs, waits for datasets still being written, and splits
per-animal datasets on demand.

Results land in `{outputs_dir}/{intermediate_results_dir}/{model_type}_{n_hand_labels}_{seed}`.
Two runs differing only in step budget **collide** unless
`intermediate_results_dir` differs.

Confirm in the log within the first minute:
- `Dataset splits -- train: N, val: M` matches what you intended
- for joint datasets, `Grouped splits over G groups` and
  `kept R rows (G complete groups)` — both animals of a frame must stay together
- `train_supervised_reprojection_heatmap_mse_loss` is present and non-zero, or
  the 3D loss is silently off

## 7. Map predictions back

Predictions come out in **crop pixels** (see failure 3). To get them into the
original combined format — one row per frame, `2 x K` suffixed keypoints, full
frame pixels — plus OOD pixel-error files:

```bash
# one joint model
python scripts/postprocess/two_mouse_merge_predictions.py <model_dir> \
    --data-dir ../data/<ds>_s --source ../data/<ds> \
    --animals <a1> <a2> --views <views...>

# two per-animal models recombined into one file
python scripts/postprocess/two_mouse_merge_predictions.py <dir_a1> <dir_a2> \
    --data-dir ../data/<ds>_s --source ../data/<ds> \
    --animals <a1> <a2> --views <views...>
```

Both files are kept. The merged one carries `--out-suffix` (default
`_uncropped_merged`) so it can never be confused with the cropped original.

### Everything is reported in full-frame pixels

A crop pixel is not a fixed unit. It is worth `w / size` full-frame pixels, and
`w` varies per row with how big the animal appears — on r24m the factor spans
**0.23 to 2.24, a ~10x range**, so the same 5 crop-px error is 1.1 full-frame px
in the tightest crop and 11.2 px in the loosest.

That makes crop-space numbers incomparable across rows, across views, and
against a model trained on full frames. Lightning Pose's own
`predictions_<view>_pixel_error.csv` is in **crop pixels** — useful for watching
a run converge, not for reporting. Convert before comparing anything:

| Quantity | Units | Scale leaving crop space |
|---|---|---|
| x, y | px | `x/size*w + bbox_x` (offset + scale) |
| pixel error | px | `x w/size` |
| ensemble **std** | px | `x w/size` |
| ensemble **variance** (`x_ens_var`) | px² | `x (w/size)²` |
| likelihood | — | none |

The merge script applies all of these: it carries `likelihood`, `x_ens_var` and
`y_ens_var` through, rescaling variance by the square, and flags any column it
does not recognise rather than implying it converted it.

Ensemble members share a bbox (it belongs to the dataset, not the model), so
mapping each seed out of crop space and then taking the spread is exactly
equivalent to scaling the crop-space spread. Map first — it is harder to get
wrong.

**One trap when correlating std against error.** Both get multiplied by the same
per-row `w/size`. That shared factor injects correlation driven purely by how
large the animal appeared, not by model uncertainty, and inflates how well std
appears to predict error. Report *magnitudes* in full-frame pixels — that part
is not optional — but check the std-vs-error relationship in both spaces, or
control for box size, before claiming ensemble variance is a good error proxy.

## Reference

- **`reference/intake-checklist.md`** — what to ask the human, what to state as a
  default, and what to never ask because the data answers it.
- **`reference/file-formats.md`** — exact contents of every file, both coordinate
  transforms, the animal-token rule and its call sites, config fields that must
  match the data.
- **`reference/failure-modes.md`** — fifteen traps with symptom, cause and fix,
  ordered by damage: the silent ones first, because the dangerous failures are
  not the crashes. Every one has actually cost a run here. Skim it before
  building, and consult it whenever something behaves strangely.

## Portability

These files are plain Markdown and live in the repository at
`lp3d-analysis/skills/multianimal-lp3d-dataset/`, so they are version-controlled
alongside the code they describe and readable by any tool, model or person.
`.claude/skills/multianimal-lp3d-dataset` is a symlink to that directory, which
is the only Claude-specific part; the YAML frontmatter is inert to everything
else. To use this with another assistant, point it at the directory or paste
`SKILL.md` — nothing here depends on a particular runtime.
