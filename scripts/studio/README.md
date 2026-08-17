# Studio scripts

Machine-level scripts for the Lightning studio this project runs in. They live
here rather than only in the studio because `configs/generated/*` is produced by
`run_experiments.py` and checked in — the generator and its output belong in the
same repo.

Deployed copies live outside the repo, and the studio reads them from there:

| file in repo | deployed to |
|---|---|
| `stage_dataset.sh` | `~/scripts/stage_dataset.sh` |
| `make_rat7m_calibrations_csv.py` | `~/scripts/make_rat7m_calibrations_csv.py` |
| `run_experiments.py` | `~/scripts/run_experiments.py` |
| `launch_queue.sh` | `~/scripts/launch_queue.sh` |
| `on_start.sh` | `~/.lightning_studio/on_start.sh` |
| `on_stop.sh` | `~/.lightning_studio/on_stop.sh` |

Paths inside these scripts are absolute and studio-specific (`/teamspace/...`),
same as `scripts/preprocess/chickadee.py`. Read `STUDIO_SETUP.md` first.

## What they do

- `stage_dataset.sh` — stream a dataset tarball out of a Lightning data
  connection into `/tmp/data/<name>`. `/tmp` is node-local NVMe and is wiped on
  studio stop, so staging re-runs on every boot; it is idempotent once
  `/tmp/data/<name>/.stage_complete` exists.
- `on_start.sh` — the boot hook. Backgrounds one `stage_dataset.sh --wait` per
  dataset (currently rat7m-full-crop, chickadee-crop, fly-anipose, ibl-mouse,
  two-mouse, mirror-mouse-separate). Logs to `~/scripts/_stage_<name>.log`,
  which is deliberately not tracked here.
- `run_experiments.py` — expands the `EXPERIMENTS` table into
  `configs/generated/<experiment>/{config_lp,config_pipeline}.yaml`. Those
  generated files are overwritten on every run; edit `EXPERIMENTS`, not them.
- `make_rat7m_calibrations_csv.py` — rebuilds the `calibrations.csv` index that
  `cfg.data.camera_params_file` points at. rat7m-full-crop ships the per-session
  `.toml` files but not the index, so this runs as a post-stage step.
- `launch_queue.sh` — serializes training runs on the studio's GPU.
