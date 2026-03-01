#!/bin/bash

# Sweep over supervised_reprojection_heatmap_mse log_weight values for ibl-mouse MVT 3D loss
# For each log_weight, trains 3 seeds and saves to Lightning Storage
#
# Usage:
#   ./run_logweight_sweep.sh           # All log_weights on CUDA device 0
#   ./run_logweight_sweep.sh 1         # All log_weights on CUDA device 1
#   ./run_logweight_sweep.sh 0 0.5,1   # Only specific log_weights

set -e

CUDA_DEVICE="${1:-0}"
FILTER_WEIGHTS="${2:-}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LP3D_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="${LP3D_DIR}/configs"
PIPELINE_CONFIG="${CONFIG_DIR}/pipeline_inference_logweight_sweep_ibl-mouse.yaml"
LIGHTNING_CONFIG="${CONFIG_DIR}/config_ibl-mouse_logweight_sweep.yaml"

for f in "$PIPELINE_CONFIG" "$LIGHTNING_CONFIG"; do
    if [ ! -f "$f" ]; then
        echo "Error: Config not found: $f"
        exit 1
    fi
done

OUTPUT_BASE="/teamspace/lightning_storage/Nature_results/outputs/ibl-mouse"
mkdir -p "$OUTPUT_BASE"

# Backup configs
TIMESTAMP=$(date +%Y%m%d_%H%M%S_%N)
BACKUP_PIPELINE="${PIPELINE_CONFIG}.backup_${TIMESTAMP}"
BACKUP_LIGHTNING="${LIGHTNING_CONFIG}.backup_${TIMESTAMP}"
cp "$PIPELINE_CONFIG" "$BACKUP_PIPELINE"
cp "$LIGHTNING_CONFIG" "$BACKUP_LIGHTNING"

restore_config() {
    echo ""
    echo "Restoring original config files..."
    [ -f "$BACKUP_PIPELINE" ] && cp "$BACKUP_PIPELINE" "$PIPELINE_CONFIG"
    [ -f "$BACKUP_LIGHTNING" ] && cp "$BACKUP_LIGHTNING" "$LIGHTNING_CONFIG"
    echo "  Done"
}
trap restore_config EXIT INT TERM

# Log weight values and their directory-safe names
ALL_WEIGHTS=("0.5" "1" "2" "3" "5")
ALL_NAMES=("0_5" "1"  "2" "3" "5")

# Apply filter if provided
WEIGHTS=()
NAMES=()
for i in "${!ALL_WEIGHTS[@]}"; do
    w="${ALL_WEIGHTS[$i]}"
    if [ -z "$FILTER_WEIGHTS" ]; then
        WEIGHTS+=("$w")
        NAMES+=("${ALL_NAMES[$i]}")
    else
        IFS=',' read -ra FW <<< "$FILTER_WEIGHTS"
        for fw in "${FW[@]}"; do
            if [ "$(echo "$fw" | xargs)" = "$w" ]; then
                WEIGHTS+=("$w")
                NAMES+=("${ALL_NAMES[$i]}")
            fi
        done
    fi
done

if [ ${#WEIGHTS[@]} -eq 0 ]; then
    echo "Error: No matching log_weights for filter: $FILTER_WEIGHTS"
    echo "Available: ${ALL_WEIGHTS[*]}"
    exit 1
fi

TOTAL=${#WEIGHTS[@]}

echo "=========================================="
echo "Log Weight Sweep - ibl-mouse (MVT 3D loss)"
echo "=========================================="
echo "  CUDA device:   $CUDA_DEVICE"
echo "  Log weights:   ${WEIGHTS[*]}"
echo "  Seeds:         0, 1, 2 (per weight)"
echo "  Total runs:    $TOTAL log_weights x 3 seeds"
echo "  Output base:   $OUTPUT_BASE"
echo "=========================================="
echo ""

COUNTER=0
FAILED=()

for i in "${!WEIGHTS[@]}"; do
    w="${WEIGHTS[$i]}"
    name="${NAMES[$i]}"
    COUNTER=$((COUNTER + 1))
    RESULTS_NAME="test_200_MVT_3d_loss_heat_(${name})"

    echo "----------------------------------------"
    echo "[$COUNTER/$TOTAL] log_weight=${w}  ->  ${RESULTS_NAME}"
    echo "----------------------------------------"

    # Update log_weight in lightning config
    sed -i "/supervised_reprojection_heatmap_mse:/,/^[^ ]/{s/    log_weight:.*/    log_weight: ${w}/}" "$LIGHTNING_CONFIG"

    # Update intermediate_results_dir in pipeline config
    sed -i "s/^intermediate_results_dir:.*/intermediate_results_dir: ${RESULTS_NAME}/" "$PIPELINE_CONFIG"

    echo "  log_weight: $(grep -A1 'supervised_reprojection_heatmap_mse:' "$LIGHTNING_CONFIG" | tail -1)"
    echo "  intermediate_results_dir: $(grep '^intermediate_results_dir:' "$PIPELINE_CONFIG")"

    cd "$LP3D_DIR"

    if python pipelines/pipeline_simple.py --config "configs/$(basename "$PIPELINE_CONFIG")"; then
        echo "  Completed [$COUNTER/$TOTAL] log_weight=${w}"
    else
        echo "  FAILED [$COUNTER/$TOTAL] log_weight=${w}"
        FAILED+=("$w")
    fi

    # Restore configs before next iteration
    cp "$BACKUP_PIPELINE" "$PIPELINE_CONFIG"
    cp "$BACKUP_LIGHTNING" "$LIGHTNING_CONFIG"

    echo ""
done

echo "=========================================="
echo "Log weight sweep complete!"
echo "=========================================="
echo "  Total:   $TOTAL"
echo "  Failed:  ${#FAILED[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    for f in "${FAILED[@]}"; do echo "    - $f"; done
fi
echo ""
echo "Results saved to: $OUTPUT_BASE"
for i in "${!WEIGHTS[@]}"; do
    echo "  test_200_MVT_3d_loss_heat_(${NAMES[$i]})/"
done
echo "=========================================="

# ./lp3d-analysis/runs/run_logweight_sweep.sh 1          # all weights, GPU 0
# ./lp3d-analysis/runs/run_logweight_sweep.sh 0 0.5,1    # only specific weights