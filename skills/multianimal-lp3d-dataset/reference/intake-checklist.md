# Intake checklist

Most of a new dataset can be read off the archive. A handful of things cannot,
because they are research decisions rather than facts — and guessing them
produces a dataset that builds cleanly, trains cleanly, and answers the wrong
question.

Ask the items in §1 **before building**. Confirm §2 in passing. Never ask §3.

---

## 1. Must be answered by a human

Paste this and fill it in.

```
DATASET INTAKE
  name (directory stem, e.g. r24m):          ____
  archive path(s):                           ____

  1. Which conditions do you want?
     [ ] combined full-frame (LP3D baseline)
     [ ] joint per-animal crops (SLP3D)
     [ ] joint crops + SAM recolouring
     [ ] one model per animal, plain
     [ ] one model per animal, SAM
     -> all five = 7 datasets = 7 runs per seed

  2. Label budget:  [ ] all available   [ ] fixed at ____ frames
     If fixed, what is it matched to? ____

  3. Seeds per condition:  [ ] 1 (first look)  [ ] 3 (claim)  [ ] ____

  4. Crop ratio (only if crops are NOT shipped):  ____   (1.3 / 1.4 used before)

  5. SAM recolouring strength alpha:  [ ] 0.3 (keeps texture) [ ] 1.0 (flat) [ ] ____

  6. Do the animal tokens mean anything? e.g. an1 = ____, an2 = ____
     Are they consistent across sessions?  [ ] yes  [ ] no  [ ] unknown

  7. Skeleton: confirm or correct the empirically derived one.   ____

  8. Are these results meant to be compared against an earlier dataset's
     numbers?  [ ] no  [ ] yes: ____
     (if yes, step budget / batch / label count must match it)
```

### Why each one matters

**1. Which conditions.** This is the experiment. Five conditions is seven
datasets and seven runs per seed — on one GPU that is most of a day. Building
fewer is often right; building them in the wrong order wastes the most time.
Order so the headline comparison finishes first.

**2. Label budget.** `n_hand_labels` counts **rows**, so a joint dataset needs
twice the number a per-animal one does to see the same frames. "All available"
means the measured train-split size, `int(train_prob * n_rows)` — which for
r24m is 248 frames / 496 rows, not a round number. A *fixed* budget only makes
sense if it is matched to something; ask what.

**3. Seeds.** One seed is enough to rank interventions that differ by several
pixels. It is not enough to rank ones that differ by ~1 px — on two-mouse the
base seeds spread about 1 px OOD, which was the same size as the gaps between
three of the four conditions. Cost is linear, so this is purely a budget call.

**4. Crop ratio.** Tighter means more resolution on the animal and less context;
looser means the partner is in frame more often, which is exactly the ambiguity
the SAM condition exists to remove. It interacts with the experiment, so it is
not a free parameter. Measure what shipped crops used rather than assuming.

**5. SAM alpha.** 0.3 tints the partner while leaving its texture legible — the
network still sees an animal, it just sees that this one is not the labelled
one. 1.0 erases it to a flat silhouette. These are different interventions with
different failure modes; 0.3 matches the crim13 pipeline and the two-mouse
build, so it is the default when comparability matters.

**6. Animal token semantics.** `an1`/`an2` may be arbitrary per session or may
mean something (resident/intruder, implanted/control). If arbitrary, a
per-animal model is learning "whichever animal was listed first", which is not a
category — and that changes how the per-animal condition should be read.

**7. Skeleton.** Derivable from median pairwise keypoint distances, and that is a
good first pass, but it is anatomy: someone who knows the animal should confirm
it. On r24m the empirical nearest-neighbour structure put `Offset1`/`Offset2` on
the **trunk**, near SpineM — not on the head as the names might suggest. Worth a
second opinion. It affects visualization and any PCA loss, not the supervised
fit.

**8. Comparison target.** If results must sit next to an earlier dataset's, then
step budget, batch size, label count and backbone are constrained, not free. Ask
before choosing any of them.

---

## 2. Has a default — state it, don't ask

Proceed with these and mention them; only stop if something in §1 conflicts.

| Field | Default | Note |
|---|---|---|
| stored crop size | 320 | Deliberately above the model input; leaves augmentation room and lets the input grow later without regenerating |
| `image_resize_dims` | 256 x 256 | Model input |
| `train_prob` / `val_prob` | 0.95 / 0.05 | |
| steps | 5000, milestones [2000, 3000, 4000], unfreezing 400 | Step-based, via the pipeline config |
| `train_batch_size` | 8 | ~4.5 GB on an L4; 32 OOMs |
| backbone | `vits_dino` | |
| model / head | `heatmap_multiview_transformer` / `heatmap_cnn` | |
| 3D loss | `supervised_reprojection_heatmap_mse`, `log_weight: 1.0` | Constant weight, not annealed |
| patch masking | 0.1 -> 0.5 over steps 700–5000 | |
| concurrent runs | 2 | One run ≈ 2.2 of 8 cores, 4.5 GB of 23 GB, bursty GPU |

---

## 3. Never ask — read it from the data

Asking these wastes the human's time and invites a wrong answer from memory.
Every one has been mis-stated at least once in this project.

| Fact | How to read it |
|---|---|
| view names **and order** | the `CollectedData_*.csv` filenames; order from the `name` fields in a calibration toml |
| keypoint names and count | `pd.read_csv(f, header=[0,1,2]).columns.get_level_values(1)` |
| combined vs per-animal arrangement | whether keypoint names carry an animal suffix |
| where the animal token sits in the directory name | parse a session directory |
| which frames are InD vs OOD | presence in `*.csv` vs `*_new.csv` |
| whether InD/OOD split by session or at random | compare the session sets |
| crop ratio and size, if crops shipped | `size / max_keypoint_extent`, and the bbox `w` column |
| number of frames, rows, animals per frame | count them |
| whether keypoints are ever unlabelled | NaN fraction per view |
| whether frames are colour or greyscale | `mean(abs(R - G))` over a few images |

The rule: **if the data can answer it, the data answers it.** A description of a
dataset — including the one that came with it — is a hypothesis to check, not a
specification. On r24m the brief said the cropped CSVs carried `_an1`/`_an2`
suffixed keypoints; they did not. The suffixed file was in the *other* archive.
