#!/bin/bash

# Sweep over MVT backbones for two-mouse (multiview_transformer, no 3D loss)
# Each backbone trains 3 seeds and saves to Lightning Storage
#
# Usage:
#   ./run_mvt_backbone_sweep_two-mouse.sh                                    # All backbones on GPU 0
#   ./run_mvt_backbone_sweep_two-mouse.sh 1                                  # All backbones on GPU 1
#   ./run_mvt_backbone_sweep_two-mouse.sh 0 --backbones vitb_dino,vits_dinov2  # Specific backbones

set -e

CUDA_DEVICE="${1:-0}"
FILTER_BACKBONES=""

shift 2>/dev/null || true

while [[ $# -gt 0 ]]; do
    case $1 in
        --backbones)
            FILTER_BACKBONES="$2"; shift 2 ;;
        *)
            echo "Usage: $0 [cuda_device] [--backbones b1,b2,...]"
            exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LP3D_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="${LP3D_DIR}/configs"
PIPELINE_CONFIG="${CONFIG_DIR}/pipeline_inference_mvt_backbone_sweep_two-mouse.yaml"
LIGHTNING_CONFIG="${CONFIG_DIR}/config_two-mouse_mvt_backbone_sweep.yaml"

for f in "$PIPELINE_CONFIG" "$LIGHTNING_CONFIG"; do
    if [ ! -f "$f" ]; then echo "Error: Config not found: $f"; exit 1; fi
done

OUTPUT_BASE="/teamspace/lightning_storage/Nature_results/outputs/two-mouse"
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

# Backbones to sweep
ALL_BACKBONES=(vitb_dino vits_dinov2 vitb_dinov2)

value_in_filter() {
    local value="$1" filter="$2"
    [ -z "$filter" ] && return 0
    IFS=',' read -ra vals <<< "$filter"
    for v in "${vals[@]}"; do [ "$(echo "$v" | xargs)" = "$value" ] && return 0; done
    return 1
}

BACKBONES=()
for val in "${ALL_BACKBONES[@]}"; do
    value_in_filter "$val" "$FILTER_BACKBONES" && BACKBONES+=("$val")
done
[ ${#BACKBONES[@]} -eq 0 ] && echo "Error: No matching backbones. Available: ${ALL_BACKBONES[*]}" && exit 1

TOTAL=${#BACKBONES[@]}

echo "=========================================="
echo "MVT Backbone Sweep - two-mouse (multiview_transformer)"
echo "=========================================="
echo "  CUDA device:  $CUDA_DEVICE"
echo "  Backbones:    ${BACKBONES[*]}"
echo "  Seeds:        0, 1, 2 (per backbone)"
echo "  Total runs:   $TOTAL backbones x 3 seeds each"
echo "  Output base:  $OUTPUT_BASE"
echo "=========================================="
echo ""

COUNTER=0
FAILED=()

for backbone in "${BACKBONES[@]}"; do
    COUNTER=$((COUNTER + 1))
    RESULTS_NAME="test_100_MVT_${backbone}"

    echo "----------------------------------------"
    echo "[$COUNTER/$TOTAL] backbone=${backbone}"
    echo "  -> ${RESULTS_NAME}/"
    echo "----------------------------------------"

    # Update backbone in lightning config
    sed -i "s/^  backbone:.*/  backbone: ${backbone}/" "$LIGHTNING_CONFIG"

    # Update intermediate_results_dir in pipeline config
    sed -i "s/^intermediate_results_dir:.*/intermediate_results_dir: ${RESULTS_NAME}/" "$PIPELINE_CONFIG"

    echo "  backbone: $(grep '  backbone:' "$LIGHTNING_CONFIG")"
    echo "  intermediate_results_dir: $(grep '^intermediate_results_dir:' "$PIPELINE_CONFIG")"

    cd "$LP3D_DIR"

    if python pipelines/pipeline_simple.py --config "configs/$(basename "$PIPELINE_CONFIG")"; then
        echo "  Completed [$COUNTER/$TOTAL] backbone=${backbone} (all 3 seeds)"
    else
        echo "  FAILED [$COUNTER/$TOTAL] backbone=${backbone}"
        FAILED+=("${backbone}")
    fi

    # Restore configs before next iteration
    cp "$BACKUP_PIPELINE" "$PIPELINE_CONFIG"
    cp "$BACKUP_LIGHTNING" "$LIGHTNING_CONFIG"

    echo ""
done

echo "=========================================="
echo "MVT backbone sweep complete!"
echo "=========================================="
echo "  Total backbones: $TOTAL"
echo "  Failed:          ${#FAILED[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    for f in "${FAILED[@]}"; do echo "    - $f"; done
fi
echo ""
echo "Results saved to: $OUTPUT_BASE"
for backbone in "${BACKBONES[@]}"; do
    echo "  test_100_MVT_${backbone}/"
    echo "    multiview_transformer_100_0/"
    echo "    multiview_transformer_100_1/"
    echo "    multiview_transformer_100_2/"
done
echo "=========================================="
