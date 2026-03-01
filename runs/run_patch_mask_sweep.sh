#!/bin/bash

# Script to run pipeline with different patch_mask parameter combinations
# Usage: ./run_patch_mask_sweep.sh [dataset_name] [cuda_device] [--init-steps VAL1,VAL2,...] [--final-ratios VAL1,VAL2,...]
# Example: ./run_patch_mask_sweep.sh fly-anipose          # Uses CUDA device 1 (default), runs all combinations
# Example: ./run_patch_mask_sweep.sh fly-anipose 0 --init-steps 2000    # Runs init_step=2000 with all final_ratios on GPU 0
# Example: ./run_patch_mask_sweep.sh fly-anipose 1 --init-steps 700,1000 --final-ratios 0.1,0.25    # Runs specific combinations on GPU 1
# Example: ./run_patch_mask_sweep.sh chickadee-crop 2    # Uses CUDA device 2, runs all combinations

set -e  # Exit on error

# Parse arguments
DATASET_NAME="${1:-fly-anipose}"
CUDA_DEVICE="${2:-1}"
FILTER_INIT_STEPS=""
FILTER_FINAL_RATIOS=""

# Parse optional filter arguments
# Shift past dataset and cuda_device if they were provided
if [ $# -gt 0 ]; then shift; fi  # Shift past dataset_name
if [ $# -gt 0 ] && [[ ! "$1" =~ ^-- ]]; then shift; fi  # Shift past cuda_device if it's not an option

while [[ $# -gt 0 ]]; do
    case $1 in
        --init-steps)
            if [ $# -lt 2 ]; then
                echo "Error: --init-steps requires a value"
                exit 1
            fi
            FILTER_INIT_STEPS="$2"
            shift 2
            ;;
        --final-ratios)
            if [ $# -lt 2 ]; then
                echo "Error: --final-ratios requires a value"
                exit 1
            fi
            FILTER_FINAL_RATIOS="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown option: $1"
            echo "Usage: $0 [dataset_name] [cuda_device] [--init-steps VAL1,VAL2,...] [--final-ratios VAL1,VAL2,...]"
            exit 1
            ;;
    esac
done

# Validate CUDA device is a number
if ! [[ "$CUDA_DEVICE" =~ ^[0-9]+$ ]] && ! [[ "$CUDA_DEVICE" =~ ^[0-9,]+$ ]]; then
    echo "Error: CUDA device must be a number or comma-separated numbers (e.g., 0, 1, 2, or '0,1')"
    exit 1
fi

# Set CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

# Enable CUDA error debugging (helps identify index out of bounds errors)
# Set to 1 to get synchronous error reporting (slower but more accurate)
# Default to 1 for better error messages when debugging CUDA index errors
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

echo "=========================================="
echo "CUDA Configuration:"
echo "  CUDA device(s): $CUDA_DEVICE"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  CUDA_LAUNCH_BLOCKING: $CUDA_LAUNCH_BLOCKING"
echo "=========================================="
echo ""

# Define paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/configs"
# Use the _param_sweep versions to avoid conflicts with other runs
# Try dataset-specific pipeline config first, fall back to generic one
PIPELINE_CONFIG_DATASET_SPECIFIC="${CONFIG_DIR}/pipeline_inference_param_sweep_${DATASET_NAME}.yaml"
PIPELINE_CONFIG_GENERIC="${CONFIG_DIR}/pipeline_inference_param_sweep.yaml"

if [ -f "$PIPELINE_CONFIG_DATASET_SPECIFIC" ]; then
    PIPELINE_CONFIG="$PIPELINE_CONFIG_DATASET_SPECIFIC"
    echo "Using dataset-specific pipeline config: $PIPELINE_CONFIG"
else
    PIPELINE_CONFIG="$PIPELINE_CONFIG_GENERIC"
    echo "Using generic pipeline config: $PIPELINE_CONFIG"
fi

LIGHTNING_CONFIG="${CONFIG_DIR}/config_${DATASET_NAME}_param_sweep.yaml"

# Check if config files exist
if [ ! -f "$LIGHTNING_CONFIG" ]; then
    echo "Error: Config file not found: $LIGHTNING_CONFIG"
    exit 1
fi

if [ ! -f "$PIPELINE_CONFIG" ]; then
    echo "Error: Pipeline config file not found: $PIPELINE_CONFIG"
    echo "Tried: $PIPELINE_CONFIG_DATASET_SPECIFIC"
    echo "Tried: $PIPELINE_CONFIG_GENERIC"
    exit 1
fi

# Note: Lock file creation moved to after parameter combinations are determined
# This allows different parameter combinations to run in parallel
HOSTNAME=$(hostname | cut -d. -f1)

# Create backup of original configs IMMEDIATELY (before any modifications)
# Each run creates its own timestamped backup to avoid conflicts between different machines/runs
TIMESTAMP=$(date +%Y%m%d_%H%M%S_%N)
BACKUP_CONFIG="${LIGHTNING_CONFIG}.backup_${HOSTNAME}_${TIMESTAMP}"
BACKUP_PIPELINE_CONFIG="${PIPELINE_CONFIG}.backup_${HOSTNAME}_${TIMESTAMP}"
cp "$LIGHTNING_CONFIG" "$BACKUP_CONFIG"
cp "$PIPELINE_CONFIG" "$BACKUP_PIPELINE_CONFIG"
echo "Created backups: $BACKUP_CONFIG and $BACKUP_PIPELINE_CONFIG"
echo "IMPORTANT: Config files will be restored to these backups when the script completes."
echo "NOTE: Each run on each machine creates its own backup to prevent conflicts."
echo ""


# Update dataset_name in pipeline config if needed (before main loop)
CURRENT_DATASET=$(grep "^dataset_name:" "$PIPELINE_CONFIG" | sed 's/^dataset_name: *//' | sed 's/ *#.*$//' | xargs)
if [ "$CURRENT_DATASET" != "$DATASET_NAME" ]; then
    echo "Updating dataset_name in pipeline config from '$CURRENT_DATASET' to '$DATASET_NAME'"
    sed -i "s/^dataset_name:.*/dataset_name: $DATASET_NAME/" "$PIPELINE_CONFIG"
    # Update backup to include this change
    cp "$PIPELINE_CONFIG" "$BACKUP_PIPELINE_CONFIG"
fi

# Store the original base intermediate_results_dir (before any modifications)
BASE_INTERMEDIATE_RESULTS_DIR=$(grep "^intermediate_results_dir:" "$BACKUP_PIPELINE_CONFIG" | sed 's/^intermediate_results_dir: *//' | sed 's/ *#.*$//' | xargs)
echo "Base intermediate_results_dir: $BASE_INTERMEDIATE_RESULTS_DIR"

# Define parameter arrays (full set)
INIT_STEPS_ALL=(700 1000 2000)
# INIT_STEPS_ALL=(1000)
FINAL_RATIOS_ALL=(0.1 0.25 0.5 0.75)

# Function to check if a value is in a comma-separated filter string
value_in_filter() {
    local value="$1"
    local filter_string="$2"
    
    if [ -z "$filter_string" ]; then
        return 0  # No filter means include all
    fi
    
    IFS=',' read -ra FILTER_VALS <<< "$filter_string"
    for filter_val in "${FILTER_VALS[@]}"; do
        # Remove whitespace and compare
        filter_val=$(echo "$filter_val" | xargs)
        value_trimmed=$(echo "$value" | xargs)
        if [ "$value_trimmed" = "$filter_val" ]; then
            return 0  # Found match
        fi
    done
    return 1  # No match
}

# Apply filters if provided
if [ -n "$FILTER_INIT_STEPS" ]; then
    INIT_STEPS=()
    for val in "${INIT_STEPS_ALL[@]}"; do
        if value_in_filter "$val" "$FILTER_INIT_STEPS"; then
            INIT_STEPS+=("$val")
        fi
    done
    if [ ${#INIT_STEPS[@]} -eq 0 ]; then
        echo "Error: No matching init_steps found for filter: $FILTER_INIT_STEPS"
        echo "Available values: ${INIT_STEPS_ALL[*]}"
        exit 1
    fi
else
    INIT_STEPS=("${INIT_STEPS_ALL[@]}")
fi

if [ -n "$FILTER_FINAL_RATIOS" ]; then
    FINAL_RATIOS=()
    for val in "${FINAL_RATIOS_ALL[@]}"; do
        if value_in_filter "$val" "$FILTER_FINAL_RATIOS"; then
            FINAL_RATIOS+=("$val")
        fi
    done
    if [ ${#FINAL_RATIOS[@]} -eq 0 ]; then
        echo "Error: No matching final_ratios found for filter: $FILTER_FINAL_RATIOS"
        echo "Available values: ${FINAL_RATIOS_ALL[*]}"
        exit 1
    fi
else
    FINAL_RATIOS=("${FINAL_RATIOS_ALL[@]}")
fi

# Keep final_step and init_ratio constant (from original config)
FINAL_STEP=5000
INIT_RATIO=0.1

# Counter for tracking progress
COUNTER=0
TOTAL_COMBINATIONS=$((${#INIT_STEPS[@]} * ${#FINAL_RATIOS[@]}))

echo "=========================================="
echo "Running patch_mask parameter sweep"
echo "Dataset: $DATASET_NAME"
echo "Total combinations: $TOTAL_COMBINATIONS"
echo ""
if [ -n "$FILTER_INIT_STEPS" ] || [ -n "$FILTER_FINAL_RATIOS" ]; then
    echo "Filters applied:"
    [ -n "$FILTER_INIT_STEPS" ] && echo "  --init-steps: $FILTER_INIT_STEPS"
    [ -n "$FILTER_FINAL_RATIOS" ] && echo "  --final-ratios: $FILTER_FINAL_RATIOS"
    echo ""
fi
echo "Parameter combinations to test:"
echo "  init_step values: ${INIT_STEPS[*]}"
echo "  final_ratio values: ${FINAL_RATIOS[*]}"
echo ""
echo "Constant parameters:"
echo "  final_step: $FINAL_STEP"
echo "  init_ratio: $INIT_RATIO"
echo "=========================================="
echo ""

# Create lock file based on parameter combination set (allows different combinations to run in parallel)
# Create a hash/signature of the parameter combinations being run
LOCK_SIGNATURE=$(echo "${INIT_STEPS[*]}" "${FINAL_RATIOS[*]}" | tr ' ' '_' | tr '.' '_' | sed 's/[^a-zA-Z0-9_]//g')
LOCK_FILE="${SCRIPT_DIR}/.patch_mask_sweep_${DATASET_NAME}_${LOCK_SIGNATURE}.lock"
LOCK_CONTENT="${HOSTNAME}:$$:$(date +%s):${INIT_STEPS[*]}:${FINAL_RATIOS[*]}"

if [ -f "$LOCK_FILE" ]; then
    echo "Warning: Lock file exists: $LOCK_FILE"
    LOCK_INFO=$(cat "$LOCK_FILE" 2>/dev/null)
    echo "Lock info: $LOCK_INFO"
    echo "Another instance may be running the same parameter combination set."
    
    if [ -t 0 ]; then
        # Interactive terminal - ask user
        read -p "Continue anyway? This may cause conflicts if another instance is running. (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        # Non-interactive - check if the process is still running
        LOCK_HOST=$(echo "$LOCK_INFO" | cut -d: -f1)
        LOCK_PID=$(echo "$LOCK_INFO" | cut -d: -f2)
        
        # If lock is from same host, check if process exists
        if [ "$LOCK_HOST" = "$HOSTNAME" ]; then
            if [ -n "$LOCK_PID" ] && kill -0 "$LOCK_PID" 2>/dev/null; then
                echo "ERROR: Another instance appears to be running the same parameter combination set on this machine (PID: $LOCK_PID)"
                echo "If you want to run different parameter combinations, use --init-steps and/or --final-ratios to specify different values."
                exit 1
            else
                echo "Previous lock file found but process is not running. Removing old lock and continuing..."
                rm -f "$LOCK_FILE"
            fi
        else
            echo "WARNING: Lock file is from a different machine ($LOCK_HOST)."
            echo "If you're running on different machines with shared filesystem, they may interfere!"
            echo "Consider using separate config file copies or running sequentially."
            read -p "Continue anyway? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    fi
fi
echo "$LOCK_CONTENT" > "$LOCK_FILE"
echo "Created lock file: $LOCK_FILE (Host: $HOSTNAME, PID: $$)"
echo ""

# Function to update patch_mask parameters in YAML
# Only modifies lines within the patch_mask section, leaving everything else unchanged
update_patch_mask() {
    local config_file="$1"
    local init_step="$2"
    local final_step="$3"
    local init_ratio="$4"
    local final_ratio="$5"
    
    # Use sed with range addressing to ONLY modify lines within the patch_mask section
    # Range: from "  patch_mask:" to the next top-level key (starts with a letter, no leading spaces)
    # The substitutions only match lines with 4 spaces (parameter lines), ensuring we don't modify
    # the section header or any other part of the file
    sed -i \
        -e "/^  patch_mask:/,/^[A-Za-z]/ {
            /^    init_step:/s/^    init_step:.*/    init_step: $init_step/
            /^    final_step:/s/^    final_step:.*/    final_step: $final_step/
            /^    init_ratio:/s/^    init_ratio:.*/    init_ratio: $init_ratio/
            /^    final_ratio:/s/^\(    final_ratio: \)[0-9.]*\(.*\)/\1$final_ratio\2/
        }" \
        "$config_file"
}

# Function to update intermediate_results_dir in pipeline config
update_intermediate_results_dir() {
    local pipeline_config="$1"
    local init_step="$2"
    local final_ratio="$3"
    
    # Convert final_ratio to use underscores instead of dots (0.25 -> 0_25)
    local final_ratio_formatted=$(echo "$final_ratio" | tr '.' '_')
    
    # Use the original base directory stored at script start
    local new_dir="${BASE_INTERMEDIATE_RESULTS_DIR}_(${init_step})_(${final_ratio_formatted})"
    
    # Update the config file
    sed -i "s/^intermediate_results_dir:.*/intermediate_results_dir: ${new_dir}/" "$pipeline_config"
    
    echo "Updated intermediate_results_dir to: ${new_dir}"
}

# Function to verify backups exist and are valid
verify_backups() {
    if [ ! -f "$BACKUP_CONFIG" ]; then
        echo "ERROR: Backup file missing: $BACKUP_CONFIG"
        exit 1
    fi
    if [ ! -f "$BACKUP_PIPELINE_CONFIG" ]; then
        echo "ERROR: Backup file missing: $BACKUP_PIPELINE_CONFIG"
        exit 1
    fi
}

# Function to restore original configs
restore_config() {
    echo ""
    echo "Restoring original config files from backups..."
    if [ -f "$BACKUP_CONFIG" ]; then
        cp "$BACKUP_CONFIG" "$LIGHTNING_CONFIG"
        echo "✓ Restored original lightning config: $LIGHTNING_CONFIG"
    fi
    if [ -f "$BACKUP_PIPELINE_CONFIG" ]; then
        cp "$BACKUP_PIPELINE_CONFIG" "$PIPELINE_CONFIG"
        echo "✓ Restored original pipeline config: $PIPELINE_CONFIG"
    fi
    # Remove lock file
    rm -f "$LOCK_FILE"
    echo "✓ Removed lock file"
}

# Trap to restore config on exit (even if script is interrupted)
trap restore_config EXIT INT TERM

# Loop through all combinations
for init_step in "${INIT_STEPS[@]}"; do
    for final_ratio in "${FINAL_RATIOS[@]}"; do
        COUNTER=$((COUNTER + 1))
        
        echo "----------------------------------------"
        echo "Combination $COUNTER/$TOTAL_COMBINATIONS"
        echo "init_step: $init_step"
        echo "final_ratio: $final_ratio"
        echo "----------------------------------------"
        
        # Verify backups still exist (safety check)
        verify_backups
        
        # Update the patch_mask parameters in lightning config
        update_patch_mask "$LIGHTNING_CONFIG" "$init_step" "$FINAL_STEP" "$INIT_RATIO" "$final_ratio"
        
        # Update the intermediate_results_dir in pipeline config
        update_intermediate_results_dir "$PIPELINE_CONFIG" "$init_step" "$final_ratio"
        
        # Verify the updates
        echo "Updated configs. Current patch_mask settings:"
        grep -A 4 "patch_mask:" "$LIGHTNING_CONFIG" | head -5
        echo "Current intermediate_results_dir:"
        grep "^intermediate_results_dir:" "$PIPELINE_CONFIG"
        
        # Change to lp3d-analysis directory (where the script is located)
        cd "$SCRIPT_DIR"
        
        # Verify we're in the correct directory
        if [ ! -f "pipelines/pipeline_simple.py" ] || [ ! -f "configs/pipeline_inference.yaml" ]; then
            echo "Error: Not in lp3d-analysis directory or files not found!"
            echo "Current directory: $(pwd)"
            exit 1
        fi
        
        # Run the pipeline (exact same command as usual, but with param_sweep config)
        # Use the pipeline config filename (relative to configs directory)
        # CUDA_VISIBLE_DEVICES is already set at the top of the script
        PIPELINE_CONFIG_RELATIVE="configs/$(basename "$PIPELINE_CONFIG")"
        echo "Running pipeline from: $(pwd)"
        echo "Using CUDA device: $CUDA_DEVICE (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
        echo "Command: python pipelines/pipeline_simple.py --config $PIPELINE_CONFIG_RELATIVE"
        python pipelines/pipeline_simple.py --config "$PIPELINE_CONFIG_RELATIVE"
        
        echo "Completed combination $COUNTER/$TOTAL_COMBINATIONS"
        echo ""
    done
done

# Restore original configs (trap will also handle this, but we do it explicitly here)
restore_config

# Keep backup files by default (user can delete them manually if needed)
# This ensures safety - if something goes wrong, backups are still available
# Each run creates timestamped backups, so multiple runs won't overwrite each other's backups
echo ""
echo "Backup files are kept for safety (timestamped to avoid conflicts):"
echo "  $BACKUP_CONFIG"
echo "  $BACKUP_PIPELINE_CONFIG"
echo ""
echo "To clean up old backups, you can run:"
echo "  rm -f ${CONFIG_DIR}/*.backup_*"

echo "=========================================="
echo "All combinations completed!"
echo "Config files have been restored to their original state."
echo "=========================================="

