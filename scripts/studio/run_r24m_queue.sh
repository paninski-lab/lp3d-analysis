#!/usr/bin/env bash
# Train every r24m variant, up to MAX_PARALLEL at a time on the single GPU.
#
# Seven datasets, five conditions:
#   r24m              full frames, both rats in one row      (LP3D combined)
#   r24m_s            per-animal crops, both rats as rows    (SLP3D)
#   r24m_s_sam        the same crops, other rat painted red  (SLP3D + SAM)
#   r24m_s-an1/-an2         one model per rat, plain crops
#   r24m_s_sam-an1/-an2     one model per rat, recoloured crops
#
# Ordered so the headline comparison lands first: if the queue is interrupted,
# what finished is still a usable result rather than four sevenths of a matrix.
#
# Why two at a time
# -----------------
# Measured on this L4, not assumed:
#
#   1 run   1.02 it/s   4.5 GB / 23 GB   2.2 / 8 cores   GPU util 21-100%, bursty
#   2 runs  0.53 + 0.72 = 1.25 it/s      9.0 GB, 4.2 cores, GPU util pegged ~100%
#
# So a second run buys about +22% throughput by filling the first one's idle GPU
# gaps -- worth having, but nothing like 2x. A *third* would not help: at two the
# GPU is already saturated, and memory and cores are no longer the constraint.
# Raise MAX_PARALLEL above 2 only on a bigger GPU, and re-measure before trusting
# it.
#
# Concurrency is counted over *every* matching training process, not just this
# script's children, so a run started by hand or by an earlier queue is respected
# rather than doubled. That makes the script safe to launch while work is already
# in flight, and safe to re-run after an interruption (COMPLETED runs are
# skipped).
#
# Usage:
#   bash lp3d-analysis/scripts/studio/run_r24m_queue.sh
#   MAX_PARALLEL=1 bash .../run_r24m_queue.sh          # serial
#   SEED=1 bash .../run_r24m_queue.sh
#   DATASETS="r24m_s r24m_s_sam" bash .../run_r24m_queue.sh
#   nohup bash .../run_r24m_queue.sh > logs/r24m/queue.log 2>&1 &

set -u

ROOT=/teamspace/studios/this_studio
SEED=${SEED:-0}
MAX_PARALLEL=${MAX_PARALLEL:-2}
DATASETS=${DATASETS:-"r24m_s r24m_s_sam r24m r24m_s-an1 r24m_s-an2 r24m_s_sam-an1 r24m_s_sam-an2"}
LOGDIR=$ROOT/logs/r24m
mkdir -p "$LOGDIR"

cd "$ROOT/lp3d-analysis" || exit 1

# Where pipeline_simple.py puts a run:
#   {outputs_dir}/{intermediate_results_dir}/{model_type}_{n_hand_labels}_{seed}
run_dir() {
  local ds=$1 inter rows
  inter=$(grep -E '^intermediate_results_dir:' "configs/pipeline_${ds}.yaml" | awk '{print $2}')
  rows=$(grep -A2 -E '^  n_hand_labels:' "configs/pipeline_${ds}.yaml" | grep -E '^ +- ' | head -1 | tr -d ' -')
  echo "$ROOT/outputs/${ds}/${inter}/mvt_3d_loss_${rows}_${SEED}"
}

# Bracketed so the pattern cannot match the shell that is evaluating it --
# `pgrep -f` reads full command lines, including this script's own.
#
# Counting is per *run*, not per process. A run forks its dataloader workers,
# which inherit the identical command line, so a plain `pgrep -fc` reports nine
# for one training job and the slot check below would never free a slot. A lead
# process is one whose parent is not itself a matching process.
running_pids() { pgrep -f "pipelin[e]_simple.py --config .*pipeline_r24m" 2>/dev/null; }

running_total() {
  local pids p ppid n=0
  pids=$(running_pids)
  [ -z "$pids" ] && { echo 0; return; }
  for p in $pids; do
    ppid=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    echo "$pids" | grep -qx "${ppid:-0}" || n=$((n + 1))
  done
  echo "$n"
}

running_one() { pgrep -f "pipelin[e]_simple.py --config .*pipeline_${1}\.yaml" >/dev/null 2>&1; }

dataset_ready() {
  [ -f "$ROOT/data/$1/calibrations.csv" ] && [ -f "$ROOT/data/$1/calibrations_new.csv" ]
}

# A per-animal dataset is a partition of a joint one, so what it waits on is its
# base: r24m_s_sam-an1 cannot be split until SAM2 has finished writing
# r24m_s_sam. Capped, so a build that died does not hang the queue behind it.
wait_ready() {
  local ds=$1 waited=0 base watch
  base=${ds%-an[12]}
  watch=$ds
  [ "$base" != "$ds" ] && watch=$base
  while ! dataset_ready "$watch"; do
    if [ $waited -ge 3600 ]; then
      echo "[$(date '+%F %T')] NOT READY $ds after 60m -- skipping"
      return 1
    fi
    [ $waited -eq 0 ] && echo "[$(date '+%F %T')] WAIT  $ds (dataset still being written)"
    sleep 60
    waited=$((waited + 60))
  done
  [ $waited -gt 0 ] && echo "[$(date '+%F %T')] READY $watch after $((waited / 60))m"
  if [ "$base" != "$ds" ] && ! dataset_ready "$ds"; then
    echo "[$(date '+%F %T')] SPLIT $base -> ${base}-an1, ${base}-an2"
    python scripts/preprocess/two_mouse_split_animals.py \
        --source "$ROOT/data/$base" --animals an1 an2 \
        --views Camera1 Camera2 Camera3 Camera5 Camera6 || return 1
  fi
  return 0
}

# Map every finished run into original full-frame space. Called after each run
# because the paired single-animal conditions only become mergeable once *both*
# halves are done -- the driver itself skips whatever is not ready yet, so
# calling it unconditionally is the simplest thing that is also correct.
#
# Serialised with an mkdir lock: runs finish concurrently, and two builds of the
# same output directory would race.
preds_original() {
  local ds=$1 lock="$ROOT/outputs/.preds_original.lock" waited=0
  while ! mkdir "$lock" 2>/dev/null; do
    sleep 5
    waited=$((waited + 5))
    [ $waited -ge 600 ] && { echo "  (preds_original lock stuck, skipping)"; return 1; }
  done
  python scripts/postprocess/r24m_preds_original.py --seed "$SEED" 2>&1 \
    | sed 's/^/  /' | grep -vE "skipped .*already present|^  $"
  rmdir "$lock" 2>/dev/null
}

train_one() {
  local ds=$1 log=$2 start=$SECONDS rc mins
  python pipelines/pipeline_simple.py \
      --config "$ROOT/lp3d-analysis/configs/pipeline_${ds}.yaml" \
      --for_seed "$SEED" > "$log" 2>&1
  rc=$?
  mins=$(( (SECONDS - start) / 60 ))
  if [ $rc -eq 0 ]; then
    echo "[$(date '+%F %T')] DONE  $ds in ${mins}m"
    preds_original "$ds"
  else
    # Keep going: one dataset failing should not cost the rest of the queue.
    echo "[$(date '+%F %T')] FAIL  $ds after ${mins}m (exit $rc) -- see $log"
    tail -20 "$log"
  fi
}

echo "=== r24m queue, seed ${SEED}, up to ${MAX_PARALLEL} at a time ==="
for ds in $DATASETS; do
  printf '  %-18s -> %s\n' "$ds" "$(run_dir "$ds")"
done
echo

for ds in $DATASETS; do
  dir=$(run_dir "$ds")
  if [ -f "$dir/train_status.json" ] && grep -q COMPLETED "$dir/train_status.json"; then
    echo "[$(date '+%F %T')] SKIP  $ds (already COMPLETED)"
    continue
  fi
  if running_one "$ds"; then
    echo "[$(date '+%F %T')] SKIP  $ds (already training in another process)"
    continue
  fi
  wait_ready "$ds" || continue

  while [ "$(running_total)" -ge "$MAX_PARALLEL" ]; do sleep 30; done

  log="$LOGDIR/${ds}_seed${SEED}.log"
  echo "[$(date '+%F %T')] START $ds  (log: $log)"
  train_one "$ds" "$log" &
  # Let the new process appear to pgrep before the slot check runs again;
  # without this the next iteration reads a stale count and oversubscribes.
  sleep 90
done

wait
# Catch anything the per-run calls could not finish -- in particular a paired
# condition whose second half completed in a different queue invocation.
preds_original final
echo "[$(date '+%F %T')] queue finished"
