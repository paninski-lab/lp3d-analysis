#!/usr/bin/env python
"""Run a queue of lp3d-analysis training configurations, one after another.

Why this exists
---------------
`pipelines/pipeline_simple.py` takes exactly one pipeline yaml, and the knobs for
a run are split across *two* files:

  * pipeline yaml  -- min/max_steps, milestone_steps, unfreezing_step,
                      val_check_interval, n_hand_labels, seeds, model_types
  * lightning-pose -- train_batch_size, num_workers, patch_mask (!), train_prob,
    yaml              val_prob, learning_rate

`patch_mask` in particular is read off the LP config (pipeline_simple.py:437),
so you cannot vary the masking curriculum from the pipeline yaml alone. This
script owns both: each entry in EXPERIMENTS below is materialised as a *pair* of
generated yamls, then handed to pipeline_simple.py.

Output collision
----------------
Results land in
    {outputs_dir}/{intermediate_results_dir}/{model_type}_{n_hand_labels}_{seed}
so two runs with the same (model_type, n_hand_labels, seed) but different step
budgets WILL collide. The runner therefore forces intermediate_results_dir to
the experiment's own name.

Usage
-----
    python scripts/run_experiments.py --list          # show the queue + estimates
    python scripts/run_experiments.py --dry-run       # generate configs, run nothing
    python scripts/run_experiments.py                 # run the whole queue
    python scripts/run_experiments.py --only full_30k # run one

Detach it, since the long run outlives an ssh session. Prefer the wrapper,
which waits for the GPU and the staged dataset first:
    bash scripts/launch_queue.sh                 # detach the enabled queue
    bash scripts/launch_queue.sh --only full_30k
    bash scripts/launch_queue.sh --list

Hard constraints for this search
--------------------------------
* Learning rate stays at the base-config value (5e-5). Do not put
  training.optimizer_params.learning_rate in an experiment's lp: block.
  build_configs() will refuse the override.
* Backbone stays vits_dino until a training-config winner exists. After
  that we will add named backbone variants of the winner, still at 5e-5.
* The 200-frame / 5k baseline is *not* in this queue — it already ran on
  another studio. Drop those results under
  outputs/rat7m-crop/test_200_MVT_3d_loss_patch_masking_rebuttal/
  so full_5k_control has a paired comparison.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml
from omegaconf import OmegaConf

ROOT = Path("/teamspace/studios/this_studio")
LP3D = ROOT / "lp3d-analysis"
BASE_PIPELINE_CFG = LP3D / "configs" / "pipeline_inference_rebuttal.yaml"
GENERATED_DIR = LP3D / "configs" / "generated"
LOG_DIR = ROOT / "logs"

# Size of the training split, from data.train_prob applied to the 27,827 labeled
# frames. Passing anything >= this as n_hand_labels makes lightning-pose use the
# whole split (data/utils.py:322, prints "Requested training frames exceeds
# training set size; using all"). Kept explicit so the results directory name
# says how many frames were actually trained on.
FULL_TRAIN_SPLIT = 27271


# ---------------------------------------------------------------------------
# The queue. Each entry overrides the two base configs; anything not named here
# is inherited unchanged — including learning_rate (5e-5) and backbone
# (vits_dino). Those two are locked for this search. After a winner is picked
# we add backbone variants of that winner; we do not retune LR.
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    {
        # Control. Byte-for-byte the finished 200-frame run except for the
        # training pool. Same steps, same batch, same schedule -- so the only
        # variable is 200 labels vs 27,271. Answers "at fixed compute, what do
        # the extra labels buy?" before committing a day of GPU to full_30k.
        "name": "full_5k_control",
        "enabled": True,
        "n_hand_labels": FULL_TRAIN_SPLIT,
        "pipeline": {
            "min_steps": 5000,
            "max_steps": 5000,
            "unfreezing_step": 400,
            "milestone_steps": [2000, 3000, 4000],
            "val_check_interval": 250,
        },
        "lp": {
            "training.train_batch_size": 8,
            "training.num_workers": 8,
            "training.patch_mask.init_step": 700,
            "training.patch_mask.final_step": 5000,
        },
    },
    {
        # The headline full-data run. Batch 32 is for A100 80GB (the 5k
        # control used ~11 GB at batch 8). Keep LR at 5e-5.
        "name": "full_30k",
        "enabled": True,
        "n_hand_labels": FULL_TRAIN_SPLIT,
        "pipeline": {
            "min_steps": 30000,
            "max_steps": 30000,
            "unfreezing_step": 2000,
            "milestone_steps": [15000, 20000, 25000],
            # NOT 25. See the header note in print_queue(): at vci=25 the val
            # pass costs more than the training it interleaves with.
            "val_check_interval": 250,
        },
        "lp": {
            "training.train_batch_size": 32,
            "training.num_workers": 8,
            "training.patch_mask.init_step": 4000,
            # Must move with max_steps. Left at 5000 the ratio ramps 0.1 -> 0.5
            # over 1000 steps and then sits flat for the remaining 25,000, which
            # is not a curriculum.
            "training.patch_mask.final_step": 20000,
        },
    },
    {
        # Optional middle point for a label-efficiency curve. Disabled by
        # default -- it costs the same wall clock as full_30k.
        "name": "mid_2k_30k",
        "enabled": False,
        "n_hand_labels": 2000,
        "pipeline": {
            "min_steps": 30000,
            "max_steps": 30000,
            "unfreezing_step": 2000,
            "milestone_steps": [15000, 20000, 25000],
            "val_check_interval": 250,
        },
        "lp": {
            "training.train_batch_size": 16,
            "training.num_workers": 8,
            "training.patch_mask.init_step": 4000,
            "training.patch_mask.final_step": 20000,
        },
    },
]


# ---------------------------------------------------------------------------


def set_dotted(cfg: dict, dotted_key: str, value) -> None:
    """Set cfg['a']['b'] from the key 'a.b', erroring if the path is not there.

    Strict on purpose: a typo'd override that silently created a new key would
    be invisible until you looked at the results and wondered why nothing moved.
    """
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node:
            raise KeyError(f"no such config section: {dotted_key} (missing {part!r})")
        node = node[part]
    if parts[-1] not in node:
        raise KeyError(f"no such config key: {dotted_key}")
    node[parts[-1]] = value


def build_configs(exp: dict) -> tuple[Path, Path, Path]:
    """Materialise the pipeline + LP yaml pair for one experiment.

    Returns (pipeline_cfg_path, lp_cfg_path, results_dir).
    """
    # Resolve ${dataset_name} interpolations now, so the generated files stand
    # alone and can be read without guessing what they pointed at.
    base_pipe = OmegaConf.to_container(
        OmegaConf.load(BASE_PIPELINE_CFG), resolve=True
    )
    base_lp = yaml.safe_load(Path(base_pipe["lightning_pose_config"]).read_text())

    pipe = copy.deepcopy(base_pipe)
    lp = copy.deepcopy(base_lp)

    pipe["intermediate_results_dir"] = exp["name"]
    pipe["train_networks"]["n_hand_labels"] = [exp["n_hand_labels"]]
    pipe["train_networks"]["ensemble_seeds"] = exp.get("seeds", [0])
    if "model_types" in exp:
        pipe["train_networks"]["model_types"] = exp["model_types"]
    for key, value in exp.get("pipeline", {}).items():
        set_dotted(pipe["train_networks"], key, value)
    for key, value in exp.get("lp", {}).items():
        if key == "training.optimizer_params.learning_rate" or key.endswith(
            ".learning_rate"
        ):
            raise ValueError(
                f"{exp['name']}: learning_rate is locked at the base-config "
                f"5e-5. Do not override {key}."
            )
        set_dotted(lp, key, value)

    # Patch masking is required for this search (final_ratio 0.0 disables it
    # in lightning-pose). Do not add a no-mask experiment here.
    mask_ratio = lp.get("training", {}).get("patch_mask", {}).get("final_ratio", 0)
    if mask_ratio is None or float(mask_ratio) <= 0:
        raise ValueError(
            f"{exp['name']}: patch masking is required "
            f"(training.patch_mask.final_ratio must be > 0, got {mask_ratio})."
        )

    out_dir = GENERATED_DIR / exp["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    lp_path = out_dir / "config_lp.yaml"
    pipe_path = out_dir / "config_pipeline.yaml"
    pipe["lightning_pose_config"] = str(lp_path)

    header = (
        f"# GENERATED by scripts/run_experiments.py for experiment '{exp['name']}'.\n"
        f"# Edits here are overwritten on the next run -- change EXPERIMENTS instead.\n"
        f"# Derived from {BASE_PIPELINE_CFG.name} + {Path(base_pipe['lightning_pose_config']).name}\n"
    )
    lp_path.write_text(header + yaml.safe_dump(lp, sort_keys=False))
    pipe_path.write_text(header + yaml.safe_dump(pipe, sort_keys=False))

    model_type = pipe["train_networks"]["model_types"][0]
    seed = pipe["train_networks"]["ensemble_seeds"][0]
    results_dir = (
        Path(pipe["outputs_dir"])
        / exp["name"]
        / f"{model_type}_{exp['n_hand_labels']}_{seed}"
    )
    return pipe_path, lp_path, results_dir


def estimate_hours(exp: dict) -> float:
    """Rough wall-clock estimate.

    Calibrated against the 200-frame run on this L4: ~1.05 s per training step
    at batch 8, and ~37 s per validation pass over the 556-frame val split
    (12 batches of 48 frames x 6 views). Step cost is assumed to scale with
    batch size, which is pessimistic -- that run sat at ~17% GPU utilisation, so
    a bigger batch partly rides along for free. Treat these as upper bounds.
    """
    sec_per_step_at_bs8 = 1.05
    sec_per_val_pass = 37.0
    steps = exp["pipeline"]["max_steps"]
    batch = exp["lp"].get("training.train_batch_size", 8)
    vci = exp["pipeline"]["val_check_interval"]
    train_s = steps * sec_per_step_at_bs8 * (batch / 8)
    val_s = (steps / vci) * sec_per_val_pass
    return (train_s + val_s) / 3600


def print_queue() -> None:
    print(f"{'experiment':<18} {'labels':>7} {'steps':>7} {'batch':>6} {'vci':>5} "
          f"{'epochs':>7} {'est h':>6}  status")
    print("-" * 78)
    total = 0.0
    for exp in EXPERIMENTS:
        _, _, results_dir = build_configs(exp)
        done = (results_dir / ".done").exists()
        exists = results_dir.exists()
        status = "DONE" if done else ("PARTIAL - will skip" if exists else "queued")
        if not exp["enabled"]:
            status = "disabled"
        batch = exp["lp"].get("training.train_batch_size", 8)
        pool = min(exp["n_hand_labels"], FULL_TRAIN_SPLIT)
        epochs = exp["pipeline"]["max_steps"] / math.ceil(pool / batch)
        hours = estimate_hours(exp)
        if exp["enabled"] and not done:
            total += hours
        print(f"{exp['name']:<18} {exp['n_hand_labels']:>7} "
              f"{exp['pipeline']['max_steps']:>7} {batch:>6} "
              f"{exp['pipeline']['val_check_interval']:>5} {epochs:>7.1f} "
              f"{hours:>6.1f}  {status}")
    print("-" * 78)
    print(f"{'remaining':<18} {'':>7} {'':>7} {'':>6} {'':>5} {'':>7} {total:>6.1f} h")


def run_one(exp: dict) -> bool:
    pipe_path, _, results_dir = build_configs(exp)

    if (results_dir / ".done").exists():
        print(f"[SKIP] {exp['name']}: already finished.")
        return True
    if results_dir.exists():
        # train_and_infer() calls Model.from_dir() and skips training whenever a
        # checkpoint is found, so a half-finished run would be silently accepted
        # as complete. Refuse to guess; make the user decide.
        print(f"[SKIP] {exp['name']}: {results_dir} exists but has no .done marker.")
        print(f"       A previous attempt died partway. Delete it to retrain:")
        print(f"         rm -rf {results_dir}")
        return False

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{exp['name']}.log"
    print(f"[RUN ] {exp['name']}  (~{estimate_hours(exp):.1f} h est)")
    print(f"       config: {pipe_path}")
    print(f"       log:    {log_path}")
    start = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run(
            [sys.executable, "pipelines/pipeline_simple.py", "--config", str(pipe_path)],
            cwd=LP3D,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    elapsed = (time.time() - start) / 3600

    if proc.returncode != 0:
        print(f"[FAIL] {exp['name']}: exit {proc.returncode} after {elapsed:.2f} h. "
              f"See {log_path}")
        return False

    (results_dir / ".done").write_text(
        json.dumps({"experiment": exp["name"], "hours": round(elapsed, 3)}, indent=2)
    )
    print(f"[DONE] {exp['name']} in {elapsed:.2f} h")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="show the queue and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="generate the config pairs but launch nothing")
    parser.add_argument("--only", metavar="NAME", help="run a single experiment")
    args = parser.parse_args()

    names = [e["name"] for e in EXPERIMENTS]
    if args.only and args.only not in names:
        print(f"unknown experiment {args.only!r}; choose from {names}")
        return 2

    if args.list:
        print_queue()
        return 0

    if args.dry_run:
        for exp in EXPERIMENTS:
            pipe_path, lp_path, results_dir = build_configs(exp)
            print(f"{exp['name']}:\n  {pipe_path}\n  {lp_path}\n  -> {results_dir}")
        print()
        print_queue()
        return 0

    print_queue()
    print()
    failures = []
    for exp in EXPERIMENTS:
        if not exp["enabled"]:
            continue
        if args.only and exp["name"] != args.only:
            continue
        if not run_one(exp):
            failures.append(exp["name"])

    if failures:
        print(f"\nfinished with failures: {', '.join(failures)}")
        return 1
    print("\nall queued experiments finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
