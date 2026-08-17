# Studio setup: dataset staging into `/tmp` (GCP)

**Read this first.** This studio runs on **GCP** in `zuckerman-institute/project-lp3d`.
`rat7m-full-crop` should be present in `/tmp/data/` — automatically, on every
studio start.

The reference scripts originated in the AWS studio `nature26`. That studio (and
AWS `s3_folders` connections) are **not** fuse-mounted here. Lightning only
auto-mounts sibling studios and data connections that live in the **same cloud
cluster**. `nature26` is still reachable with the Lightning CLI / SDK:

```bash
lightning studio ls  lit://zuckerman-institute/project-lp3d/studios/nature26/
lightning studio cp  lit://zuckerman-institute/project-lp3d/studios/nature26/scripts/foo.sh ~/scripts/
python3 -c 'from lightning_sdk import Studio; s=Studio("nature26", teamspace="project-lp3d", org="zuckerman-institute"); print(s.run("ls /tmp/data"))'
```

Do not switch this studio to AWS to make the original mounts appear.

---

## What you end up with

```
/tmp/data/
└── rat7m-full-crop/         from gcs_folders/rat7m-data
```

Roughly **23 GB**. `/tmp` is node-local NVMe: fast for the random small-file
reads a dataloader does, and deliberately **not** part of the studio snapshot.
`/tmp` is wiped when the studio stops, so staging re-runs on every boot.
Everything is idempotent — re-running when the dataset is already staged costs nothing.

You do **not** need to stop/restart the studio for this to take effect. If you
*do* stop it, the next start re-extracts rat7m (~13–21 min).

---

## Step 0 — Check the mounts exist

Run this first. If any line says MISSING, stop and report it rather than working around it.

```bash
for p in /teamspace/gcs_folders/rat7m-data \
         /teamspace/lightning_storage/rat-data/rat7m-data; do
  printf "%-58s " "$p"; [ -d "$p" ] && echo OK || echo MISSING
done

ls -lh /teamspace/gcs_folders/rat7m-data/rat7m-full-crop.tar.xz
```

Expected: `rat7m-full-crop.tar.xz` ~20 GB in `gcs_folders/rat7m-data`.
`lightning_storage/rat-data` is an S3 fallback of the same archive.

---

## Step 1 — Scripts and boot hook

These should already be in place on this studio:

```
~/scripts/stage_dataset.sh
~/scripts/make_rat7m_calibrations_csv.py
~/.lightning_studio/on_start.sh
```

```bash
chmod +x ~/scripts/*.sh ~/.lightning_studio/*.sh
bash -n ~/scripts/stage_dataset.sh ~/.lightning_studio/on_start.sh
```

---

## Step 2 — Confirm the config blocks

`~/scripts/stage_dataset.sh` — the per-dataset source map:

```bash
declare -A DATASETS=(
    ["rat7m-full-crop"]="/teamspace/gcs_folders/rat7m-data/rat7m-full-crop.tar.xz|rat7m-full-crop"
)
```

---

## Step 3 — The boot hook

`~/.lightning_studio/on_start.sh` is what Lightning runs automatically at studio start.
It should read:

```bash
#!/bin/bash

nohup bash ~/scripts/stage_dataset.sh rat7m-full-crop \
    > ~/scripts/_stage_rat7m_full.log 2>&1 &
```

rat7m takes about **13–21 minutes**; the studio is usable immediately, the data
just isn't ready yet.

---

## Step 4 — Run it now, don't wait for a restart

```bash
bash ~/.lightning_studio/on_start.sh
```

Watch progress:

```bash
tail -f ~/scripts/_stage_rat7m_full.log
```

---

## Step 5 — Verify before training on any of it

```bash
ls /tmp/data/

for d in /tmp/data/*/; do
  printf "%-40s " "$d"
  [ -f "$d/.stage_complete" ] && echo COMPLETE || echo "INCOMPLETE — re-stage this one"
done

bash ~/scripts/stage_dataset.sh rat7m-full-crop --status
```

Expected for `rat7m-full-crop`:

| Check | Expected |
|---|---|
| Total size | ~22.9 GB |
| Images | 175,962 |
| Train sessions | `s1-d1` 7198 · `s2-d1` 6225 · `s2-d2` 7233 · `s3-d1` 7171 |
| Test sessions | `s4-d1` · `s5-d1` · `s5-d2`, 500 imgs each |
| Label CSVs | 12 (`CollectedData_camera{1..6}{,_new}.csv`) — 27,827 train / 300 OOD frames |
| Bbox CSVs | 12 (`bboxes_camera{1..6}{,_new}.csv`) |
| Calibrations | `calibrations/` with 7 `.toml` + `calibrations.csv` + `calibrations_new.csv` |

Then confirm every image the label CSVs reference actually resolves. This is the check
that catches the archive's internal naming inconsistency:

```bash
cd /tmp/data/rat7m-full-crop && python - <<'PY'
import pandas as pd, os
missing = 0
for view in [f"camera{i}" for i in range(1,7)]:
    for suffix in ("", "_new"):
        df = pd.read_csv(f"CollectedData_{view}{suffix}.csv", header=[0,1,2], index_col=0)
        miss = [p for p in df.index if not os.path.exists(p)]
        missing += len(miss)
        print(f"  {view}{suffix:<5}: {len(df):>6} refs, {len(miss)} missing")
print("TOTAL MISSING:", missing, "(must be 0)")
PY
```

---

## Dataset inventory

| Dataset | Source on this GCP studio | Archive | Extracted |
|---|---|---|---|
| `rat7m-full-crop` | `gcs_folders/rat7m-data` | 20 GB `.tar.xz` | **22.9 GB** |

Optional, not staged by default:

- `/teamspace/lightning_storage/rat-data/rat7m-data/rat7m-full-crop.tar.xz` — S3 copy of the same rat7m archive (fallback; `stage_dataset.sh --wait` will find it)
- `/teamspace/lightning_storage/rat-data/rat-7m/` — already unpacked, uncropped 1328×1048
- `/teamspace/lightning_storage/rat-data/rat7m.tar.xz` — 27.8 GB, the uncropped archive

---

## How `stage_dataset.sh` works

```
bash ~/scripts/stage_dataset.sh <name>            # stage it (idempotent)
bash ~/scripts/stage_dataset.sh <name> --status   # report only, extract nothing
bash ~/scripts/stage_dataset.sh <name> --force    # discard and re-extract
bash ~/scripts/stage_dataset.sh <name> --wait     # poll until the archive appears
```

Sequence, and **why each guard exists** — these were all learned the hard way, so don't
remove them:

1. **Lock** — created atomically with `set -o noclobber`, released by an `EXIT` trap, so
   two concurrent boot hooks can't stage the same dataset twice.
2. **Completion marker** — a destination directory that exists *without* `.stage_complete`
   is treated as an interrupted run and re-extracted. "Directory exists" is not a valid
   completion test; it silently accepts half-extracted data.
3. **Wait for the upload to settle** — poll the archive size until unchanged for ~60 s.
   Skipped when the tarball mtime is already ≥120 s old.
4. **Integrity check** — `xz -l` reads only the index/footer, so it costs seconds and
   fails immediately on a truncated archive. This is essential: the size-stability check
   above *cannot* distinguish "upload finished" from "upload died and stopped growing".
   This can.
5. **Free-space check** — `xz --robot -l` reports the exact uncompressed size; require it
   plus 10 % headroom. Don't estimate from the compressed size (PNG data barely compresses,
   so the ratio is near 1 — not something to assume).
6. **Streamed extraction** — `xz -dc -T0 | tar -x`, so the archive and the extracted tree
   never both occupy disk. (`-T0` only parallelises if the archive was written with
   multiple blocks; otherwise it quietly runs single-threaded.)
7. **Root auto-detection** — rat7m has *no* wrapping directory, so the extract root
   is used when `labeled-data` sits there.
8. **Move into place, then write the marker.**
9. **Post-extract fixups** (rat7m only) — see below.

---

## The rat7m archive has two defects. Both are repaired automatically.

Do not "fix" these by editing data by hand; the staging script handles them on every run.

**1. Missing calibration index.** The archive ships `calibrations/*.toml` (the actual
camera intrinsics and extrinsics) but omits `calibrations.csv`, which is what
`cfg.data.camera_params_file` points at. That file is *not* camera data — it's a
two-column lookup:

```
,file
labeled-data-cropped/s1-d1/img00000043.png,calibrations/s1-d1.toml
```

`make_rat7m_calibrations_csv.py` rebuilds it from the session name already embedded in
each image path. Nothing is invented.

**2. Directory naming mismatch.** All 12 label and bbox CSVs index images under
`labeled-data-cropped/`, but the archive ships that directory as `labeled-data/`. The
images are identical; only the name differs. The script bridges it with a symlink rather
than rewriting 168k CSV rows.

---

## Things that will bite you

- **Network mounts throw transient `Input/output error (5)` under load.** Use `rsync` for
  any bulk copy so an interrupted transfer resumes on re-run, and verify file counts at
  the end rather than trusting the exit code.
- **Don't train directly off a network mount.** Random small-file reads over fuse are far
  slower than local NVMe. That's the entire reason for staging to `/tmp`.
- **Check the boot logs.** `~/.lightning_studio/logs/on-start-*.log`. A start hook that
  exists is not a start hook that works.
- **rat7m train and test crops are different sizes** — train is 256×256, test is 320×320.
  Training and metrics are unaffected (predictions come back in each image's native
  space), but pixel errors are reported in crop space, so the two splits sit on different
  scales and can't be compared directly without converting through each frame's bbox.
- **Validation cost scales with dataset size, not with your training-frame cap.** Capping
  `train_frames` does not cap the validation set — it stays `val_prob × all labels`. On
  the 27,827-frame rat7m split that is 1,391 val frames at `val_prob: 0.05`. Check
  `val_prob` and `val_check_interval` whenever the dataset grows.
- **Cross-cloud:** GCP will not grow an `s3_folders/` tree, and AWS will not grow a
  `gcs_folders/` tree. `lightning_storage/` is the path that shows up on both.
