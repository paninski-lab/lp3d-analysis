#!/bin/bash

# Sweep over patch_mask init_step x final_ratio for ibl-mouse (MVT 3D loss)
# Each combination trains 3 seeds and saves to Lightning Storage
#
# Usage:
#   ./run_patchmask_sweep.sh                                    # All combos on GPU 0
#   ./run_patchmask_sweep.sh 1                                  # All combos on GPU 1
#   ./run_patchmask_sweep.sh 0 --init-steps 700,1000            # Only specific init_steps
#   ./run_patchmask_sweep.sh 0 --final-ratios 0.1,0.25          # Only specific final_ratios

set -e

CUDA_DEVICE="${1:-0}"
FILTER_INIT_STEPS=""
FILTER_FINAL_RATIOS=""

shift 2>/dev/null || true

while [[ $# -gt 0 ]]; do
    case $1 in
        --init-steps)
            FILTER_INIT_STEPS="$2"; shift 2 ;;
        --final-ratios)
            FILTER_FINAL_RATIOS="$2"; shift 2 ;;
        *)
            echo "Usage: $0 [cuda_device] [--init-steps v1,v2,...] [--final-ratios v1,v2,...]"
            exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LP3D_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="${LP3D_DIR}/configs"
PIPELINE_CONFIG="${CONFIG_DIR}/pipeline_inference_patchmask_sweep_ibl-mouse.yaml"
LIGHTNING_CONFIG="${CONFIG_DIR}/config_ibl-mouse_patchmask_sweep.yaml"

for f in "$PIPELINE_CONFIG" "$LIGHTNING_CONFIG"; do
    if [ ! -f "$f" ]; then echo "Error: Config not found: $f"; exit 1; fi
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

# Parameter grid
ALL_INIT_STEPS=(0 700 1000 2000)
ALL_FINAL_RATIOS=(0.1 0.25 0.5 0.75)
FINAL_STEP=5000
INIT_RATIO=0.1

# Apply filters
value_in_filter() {
    local value="$1" filter="$2"
    [ -z "$filter" ] && return 0
    IFS=',' read -ra vals <<< "$filter"
    for v in "${vals[@]}"; do [ "$(echo "$v" | xargs)" = "$value" ] && return 0; done
    return 1
}

INIT_STEPS=()
for v in "${ALL_INIT_STEPS[@]}"; do
    value_in_filter "$v" "$FILTER_INIT_STEPS" && INIT_STEPS+=("$v")
done
FINAL_RATIOS=()
for v in "${ALL_FINAL_RATIOS[@]}"; do
    value_in_filter "$v" "$FILTER_FINAL_RATIOS" && FINAL_RATIOS+=("$v")
done

[ ${#INIT_STEPS[@]} -eq 0 ] && echo "Error: No matching init_steps. Available: ${ALL_INIT_STEPS[*]}" && exit 1
[ ${#FINAL_RATIOS[@]} -eq 0 ] && echo "Error: No matching final_ratios. Available: ${ALL_FINAL_RATIOS[*]}" && exit 1

TOTAL=$((${#INIT_STEPS[@]} * ${#FINAL_RATIOS[@]}))

echo "=========================================="
echo "Patch Mask Sweep - ibl-mouse (MVT 3D loss)"
echo "=========================================="
echo "  CUDA device:    $CUDA_DEVICE"
echo "  init_steps:     ${INIT_STEPS[*]}"
echo "  final_ratios:   ${FINAL_RATIOS[*]}"
echo "  Constant:       final_step=$FINAL_STEP  init_ratio=$INIT_RATIO"
echo "  Seeds:          0, 1, 2 (per combination)"
echo "  Total combos:   $TOTAL"
echo "  Output base:    $OUTPUT_BASE"
echo "=========================================="
echo ""

COUNTER=0
FAILED=()

for init_step in "${INIT_STEPS[@]}"; do
    for final_ratio in "${FINAL_RATIOS[@]}"; do
        COUNTER=$((COUNTER + 1))

        final_ratio_fmt=$(echo "$final_ratio" | tr '.' '_')
        RESULTS_NAME="test_200_MVT_patch_masking_(${init_step})_(${final_ratio_fmt})"

        echo "----------------------------------------"
        echo "[$COUNTER/$TOTAL] init_step=${init_step}  final_ratio=${final_ratio}"
        echo "  -> ${RESULTS_NAME}"
        echo "----------------------------------------"

        # Update patch_mask in lightning config
        sed -i "/^  patch_mask:/,/^[A-Za-z]/ {
            s/^    init_step:.*/    init_step: ${init_step}/
            s/^    final_step:.*/    final_step: ${FINAL_STEP}/
            s/^    init_ratio:.*/    init_ratio: ${INIT_RATIO}/
            s/^    final_ratio:.*/    final_ratio: ${final_ratio}/
        }" "$LIGHTNING_CONFIG"

        # Update intermediate_results_dir in pipeline config
        sed -i "s/^intermediate_results_dir:.*/intermediate_results_dir: ${RESULTS_NAME}/" "$PIPELINE_CONFIG"

        echo "  patch_mask:"
        grep -A4 "patch_mask:" "$LIGHTNING_CONFIG" | head -5
        echo "  intermediate_results_dir: $(grep '^intermediate_results_dir:' "$PIPELINE_CONFIG")"

        cd "$LP3D_DIR"

        if python pipelines/pipeline_simple.py --config "configs/$(basename "$PIPELINE_CONFIG")"; then
            echo "  Completed [$COUNTER/$TOTAL] init_step=${init_step} final_ratio=${final_ratio}"
        else
            echo "  FAILED [$COUNTER/$TOTAL] init_step=${init_step} final_ratio=${final_ratio}"
            FAILED+=("${init_step}_${final_ratio}")
        fi

        # Restore configs before next iteration
        cp "$BACKUP_PIPELINE" "$PIPELINE_CONFIG"
        cp "$BACKUP_LIGHTNING" "$LIGHTNING_CONFIG"

        echo ""
    done
done

echo "=========================================="
echo "Patch mask sweep complete!"
echo "=========================================="
echo "  Total combos: $TOTAL"
echo "  Failed:       ${#FAILED[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    for f in "${FAILED[@]}"; do echo "    - $f"; done
fi
echo ""
echo "Results saved to: $OUTPUT_BASE"
for is in "${INIT_STEPS[@]}"; do
    for fr in "${FINAL_RATIOS[@]}"; do
        fr_fmt=$(echo "$fr" | tr '.' '_')
        echo "  test_200_MVT_patch_masking_(${is})_(${fr_fmt})/"
    done
done
echo "=========================================="

# How to run the sweep:
# ./lp3d-analysis/runs/run_patchmask_sweep.sh 2  # all 12 combos
# ./lp3d-analysis/runs/run_patchmask_sweep.sh 0 --init-steps 700         # only init_step=700
# ./lp3d-analysis/runs/run_patchmask_sweep.sh 0 --final-ratios 0.1,0.25  # only those ratios