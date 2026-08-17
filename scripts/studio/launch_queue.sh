#!/bin/bash
# Thin launcher around run_experiments.py.
#
# This script does NOT own the experiment list, the yaml generation, or the
# skip/.done logic. Those live in scripts/run_experiments.py. This only:
#   1. refuses to start without a GPU
#   2. waits for /tmp/data/rat7m-full-crop to finish staging
#   3. detaches the Python queue so it survives a studio disconnect
#
# usage:
#   bash scripts/launch_queue.sh                 # detach the enabled queue
#   bash scripts/launch_queue.sh --only full_30k
#   bash scripts/launch_queue.sh --list          # print the queue, run nothing
#   bash scripts/launch_queue.sh --dry-run       # generate yamls, run nothing
#   bash scripts/launch_queue.sh --fg            # foreground (e.g. inside tmux)
#   bash scripts/launch_queue.sh --fg --only full_5k_control

set -euo pipefail

ROOT="/teamspace/studios/this_studio"
LP3D="$ROOT/lp3d-analysis"
RUNNER="$ROOT/scripts/run_experiments.py"
PYTHON="${PYTHON:-/home/zeus/miniconda3/envs/cloudspace/bin/python}"
LOG_DIR="$ROOT/logs"
STAGE_MARKER="/tmp/data/rat7m-full-crop/.stage_complete"
STAGE_LOG="$ROOT/scripts/_stage_rat7m_full.log"
PID_FILE="$LOG_DIR/queue.pid"

FOREGROUND=0
PASSTHROUGH=()
for arg in "$@"; do
    if [ "$arg" = "--fg" ]; then
        FOREGROUND=1
    else
        PASSTHROUGH+=("$arg")
    fi
done

need_gpu() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[ERROR] nvidia-smi not found. Attach a GPU machine before launching."
        exit 1
    fi
    if ! nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; then
        echo "[ERROR] nvidia-smi failed. No usable GPU."
        exit 1
    fi
}

wait_for_data() {
    if [ -f "$STAGE_MARKER" ]; then
        echo "[OK  ] dataset staged: $STAGE_MARKER"
        return 0
    fi
    echo "[WAIT] $STAGE_MARKER not present. Staging can take 13–21 min."
    if [ -f "$STAGE_LOG" ]; then
        echo "       tail of $STAGE_LOG:"
        tail -n 8 "$STAGE_LOG" || true
    else
        echo "       no stage log yet; starting stage_dataset.sh in the background."
        nohup bash "$ROOT/scripts/stage_dataset.sh" rat7m-full-crop \
            > "$STAGE_LOG" 2>&1 &
    fi
    local waited=0
    while [ ! -f "$STAGE_MARKER" ]; do
        sleep 30
        waited=$((waited + 30))
        if [ "$waited" -ge 2400 ]; then
            echo "[ERROR] dataset still not staged after 40 min. See $STAGE_LOG"
            exit 1
        fi
        echo "       still waiting for dataset... (${waited}s)"
    done
    echo "[OK  ] dataset staged after ${waited}s"
}

mkdir -p "$LOG_DIR"

# --list / --dry-run never need a GPU or a detach.
case "${PASSTHROUGH[*]:-}" in
    *--list*|*--dry-run*)
        exec "$PYTHON" "$RUNNER" "${PASSTHROUGH[@]}"
        ;;
esac

need_gpu
wait_for_data

echo "[INFO] queue:"
"$PYTHON" "$RUNNER" --list

if [ "$FOREGROUND" -eq 1 ]; then
    echo "[RUN ] foreground. Ctrl-C stops the current experiment."
    exec "$PYTHON" "$RUNNER" "${PASSTHROUGH[@]}"
fi

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "[ERROR] queue already running as pid $(cat "$PID_FILE")."
    echo "        logs: $LOG_DIR/queue.log"
    exit 1
fi

echo "[RUN ] detaching to $LOG_DIR/queue.log"
cd "$LP3D"
setsid nohup "$PYTHON" "$RUNNER" "${PASSTHROUGH[@]}" \
    > "$LOG_DIR/queue.log" 2>&1 &
echo $! > "$PID_FILE"
echo "[OK  ] pid $(cat "$PID_FILE")"
echo "       tail -f $LOG_DIR/queue.log"
echo "       per-run logs: $LOG_DIR/<experiment>.log"
