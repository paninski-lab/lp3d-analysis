#!/bin/bash

# Script to compare different single-view transformer backbones on two-mouse dataset
# Runs each backbone with 3 seeds (0, 1, 2) and saves results to Lightning Storage
#
# Usage:
#   ./run_backbone_sweep_two-mouse.sh                          # Run all backbones on CUDA device 0
#   ./run_backbone_sweep_two-mouse.sh 1                        # Run all backbones on CUDA device 1
#   ./run_backbone_sweep_two-mouse.sh 0 --backbones vitb_dino,vits_dinov2   # Run specific backbones
#   ./run_backbone_sweep_two-mouse.sh 0 --seeds 0,1            # Run specific seeds only

set -e

# ==========================================
# Parse arguments
# ==========================================
CUDA_DEVICE="${1:-0}"
FILTER_BACKBONES=""
FILTER_SEEDS=""

shift 2>/dev/null || true

while [[ $# -gt 0 ]]; do
    case $1 in
        --backbones)
            if [ $# -lt 2 ]; then
                echo "Error: --backbones requires a value"
                exit 1
            fi
            FILTER_BACKBONES="$2"
            shift 2
            ;;
        --seeds)
            if [ $# -lt 2 ]; then
                echo "Error: --seeds requires a value"
                exit 1
            fi
            FILTER_SEEDS="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown option: $1"
            echo "Usage: $0 [cuda_device] [--backbones b1,b2,...] [--seeds s1,s2,...]"
            exit 1
            ;;
    esac
done

if ! [[ "$CUDA_DEVICE" =~ ^[0-9]+$ ]]; then
    echo "Error: CUDA device must be a number (e.g., 0, 1, 2)"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

# ==========================================
# Define paths
# ==========================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LP3D_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="${LP3D_DIR}/configs"
PIPELINE_CONFIG="${CONFIG_DIR}/pipeline_inference_backbone_sweep_two-mouse.yaml"
LIGHTNING_CONFIG="${CONFIG_DIR}/config_two-mouse_backbone_sweep.yaml"

if [ ! -f "$PIPELINE_CONFIG" ]; then
    echo "Error: Pipeline config not found: $PIPELINE_CONFIG"
    exit 1
fi
if [ ! -f "$LIGHTNING_CONFIG" ]; then
    echo "Error: Lightning config not found: $LIGHTNING_CONFIG"
    exit 1
fi

# ==========================================
# Create output directory
# ==========================================
OUTPUT_BASE="/teamspace/lightning_storage/Nature_results/outputs/two-mouse"
mkdir -p "$OUTPUT_BASE"

# ==========================================
# Backups
# ==========================================
HOSTNAME=$(hostname | cut -d. -f1)
TIMESTAMP=$(date +%Y%m%d_%H%M%S_%N)
BACKUP_PIPELINE="${PIPELINE_CONFIG}.backup_${HOSTNAME}_${TIMESTAMP}"
BACKUP_LIGHTNING="${LIGHTNING_CONFIG}.backup_${HOSTNAME}_${TIMESTAMP}"
cp "$PIPELINE_CONFIG" "$BACKUP_PIPELINE"
cp "$LIGHTNING_CONFIG" "$BACKUP_LIGHTNING"

restore_config() {
    echo ""
    echo "Restoring original config files..."
    [ -f "$BACKUP_PIPELINE" ] && cp "$BACKUP_PIPELINE" "$PIPELINE_CONFIG" && echo "  Restored pipeline config"
    [ -f "$BACKUP_LIGHTNING" ] && cp "$BACKUP_LIGHTNING" "$LIGHTNING_CONFIG" && echo "  Restored lightning config"
    echo "  Done"
}
trap restore_config EXIT INT TERM

# ==========================================
# Define backbones and seeds
# ==========================================
ALL_BACKBONES=(vitb_dino vits_dinov2 vitb_dinov2 vitb_imagenet vitb_sam)
ALL_SEEDS=(0 1 2)

value_in_filter() {
    local value="$1"
    local filter_string="$2"
    if [ -z "$filter_string" ]; then
        return 0
    fi
    IFS=',' read -ra FILTER_VALS <<< "$filter_string"
    for filter_val in "${FILTER_VALS[@]}"; do
        filter_val=$(echo "$filter_val" | xargs)
        if [ "$value" = "$filter_val" ]; then
            return 0
        fi
    done
    return 1
}

BACKBONES=()
for val in "${ALL_BACKBONES[@]}"; do
    if value_in_filter "$val" "$FILTER_BACKBONES"; then
        BACKBONES+=("$val")
    fi
done
if [ ${#BACKBONES[@]} -eq 0 ]; then
    echo "Error: No matching backbones for filter: $FILTER_BACKBONES"
    echo "Available: ${ALL_BACKBONES[*]}"
    exit 1
fi

SEEDS=()
for val in "${ALL_SEEDS[@]}"; do
    if value_in_filter "$val" "$FILTER_SEEDS"; then
        SEEDS+=("$val")
    fi
done
if [ ${#SEEDS[@]} -eq 0 ]; then
    echo "Error: No matching seeds for filter: $FILTER_SEEDS"
    echo "Available: ${ALL_SEEDS[*]}"
    exit 1
fi

TOTAL=${#BACKBONES[@]}

# Build the seeds YAML block (e.g. "    - 0\n    - 1\n    - 2")
SEEDS_YAML=""
for s in "${SEEDS[@]}"; do
    SEEDS_YAML="${SEEDS_YAML}    - ${s}\n"
done

echo "=========================================="
echo "Backbone Sweep - two-mouse (Single View)"
echo "=========================================="
echo "  CUDA device:  $CUDA_DEVICE"
echo "  Backbones:    ${BACKBONES[*]}"
echo "  Seeds:        ${SEEDS[*]} (all seeds run per backbone)"
echo "  Total runs:   $TOTAL backbones x ${#SEEDS[@]} seeds each"
echo "  Output base:  $OUTPUT_BASE"
echo "=========================================="
echo ""

# ==========================================
# Main loop — one pipeline call per backbone, all seeds at once
# ==========================================
COUNTER=0
FAILED=()

for backbone in "${BACKBONES[@]}"; do
    COUNTER=$((COUNTER + 1))
    RESULTS_NAME="test_100_SV_${backbone}"

    echo "----------------------------------------"
    echo "[$COUNTER/$TOTAL] backbone=${backbone}  seeds=[${SEEDS[*]}]"
    echo "  results dir: ${OUTPUT_BASE}/${RESULTS_NAME}/"
    echo "----------------------------------------"

    # Update backbone in lightning config
    sed -i "s/^  backbone:.*/  backbone: ${backbone}/" "$LIGHTNING_CONFIG"

    # Update intermediate_results_dir in pipeline config
    sed -i "s/^intermediate_results_dir:.*/intermediate_results_dir: ${RESULTS_NAME}/" "$PIPELINE_CONFIG"

    # Update ensemble_seeds to include all requested seeds
    sed -i '/^  ensemble_seeds:/,/^  [a-z]/{
        /^  ensemble_seeds:/!{
            /^  [a-z]/!d
        }
    }' "$PIPELINE_CONFIG"
    sed -i "s/^  ensemble_seeds:.*/  ensemble_seeds:\n${SEEDS_YAML}/" "$PIPELINE_CONFIG"

    echo "  Config updated:"
    echo "    backbone: $(grep '  backbone:' "$LIGHTNING_CONFIG")"
    echo "    intermediate_results_dir: $(grep '^intermediate_results_dir:' "$PIPELINE_CONFIG")"
    echo "    seeds: ${SEEDS[*]}"

    cd "$LP3D_DIR"

    PIPELINE_CONFIG_RELATIVE="configs/$(basename "$PIPELINE_CONFIG")"
    echo "  Running: python pipelines/pipeline_simple.py --config $PIPELINE_CONFIG_RELATIVE"
    echo ""

    if python pipelines/pipeline_simple.py --config "$PIPELINE_CONFIG_RELATIVE"; then
        echo ""
        echo "  Completed [$COUNTER/$TOTAL] backbone=${backbone} (all ${#SEEDS[@]} seeds)"
    else
        echo ""
        echo "  FAILED [$COUNTER/$TOTAL] backbone=${backbone}"
        FAILED+=("${backbone}")
    fi

    # Restore configs from backup before next iteration
    cp "$BACKUP_PIPELINE" "$PIPELINE_CONFIG"
    cp "$BACKUP_LIGHTNING" "$LIGHTNING_CONFIG"

    echo ""
done

echo "=========================================="
echo "Backbone sweep complete!"
echo "=========================================="
echo "  Total backbones: $TOTAL"
echo "  Seeds per backbone: ${#SEEDS[@]} (${SEEDS[*]})"
echo "  Failed:          ${#FAILED[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  Failed list:"
    for f in "${FAILED[@]}"; do
        echo "    - $f"
    done
fi
echo ""
echo "Results saved to: $OUTPUT_BASE"
echo "  Subdirectories:"
for backbone in "${BACKBONES[@]}"; do
    echo "    test_100_SV_${backbone}/"
    for s in "${SEEDS[@]}"; do
        echo "      supervised_100_${s}/"
    done
done
echo ""
echo "Config files have been restored to their original state."
echo "Backup files kept for safety:"
echo "  $BACKUP_PIPELINE"
echo "  $BACKUP_LIGHTNING"
echo "=========================================="

# this is how we run this:
# ./lp3d-analysis/runs/run_backbone_sweep_two-mouse.sh 0
