#!/bin/bash
# Stage a dataset tarball from a Lightning data connection into local /tmp scratch.
#
# Why: the studio image should stay light (data connections are network-backed and
# slow for the random small-file reads a dataloader does), but training wants the
# data on the local NVMe. So on studio startup we stream the tarball out of the
# mounted data connection straight into /tmp, which is local and is NOT part of the
# studio snapshot.
#
# usage:
#   bash stage_dataset.sh <dataset_name>          # extract (idempotent)
#   bash stage_dataset.sh <dataset_name> --status # report only, no extraction
#   bash stage_dataset.sh <dataset_name> --force  # re-extract from scratch
#
# Extraction is streamed (xz -> tar) so we never hold both the tarball and the
# extracted tree on disk at once.

set -o pipefail

# --- Configuration ---------------------------------------------------------

# dataset_name -> "<absolute path to tarball>|<final directory name under FINAL_DIR>"
#
# This studio runs on GCP. rat7m uses the GCP gcs_folders connection;
# the other Nature datasets are tarballs on lightning_storage (copied once
# from nature26 s3_folders/data_nature, which is not mounted here).
# --wait will also find a matching basename under lightning_storage / gcs.
declare -A DATASETS=(
    ["rat7m-full-crop"]="/teamspace/gcs_folders/rat7m-data/rat7m-full-crop.tar.xz|rat7m-full-crop"
    ["chickadee-crop"]="/teamspace/lightning_storage/Nature_results/data_nature/chickadee-crop.tar|chickadee-crop"
    ["fly-anipose"]="/teamspace/lightning_storage/Nature_results/data_nature/fly-anipose.tar|fly-anipose"
    ["ibl-mouse"]="/teamspace/lightning_storage/Nature_results/data_nature/ibl-mouse.tar|ibl-mouse"
    ["two-mouse"]="/teamspace/lightning_storage/Nature_results/data_nature/two-mouse.tar|two-mouse"
    ["mirror-mouse-separate"]="/teamspace/lightning_storage/Nature_results/data_nature/mirror-mouse-separate.tar|mirror-mouse-separate"
)

readonly FINAL_DIR="/tmp/data"        # where training reads from
readonly UNTAR_DIR="/tmp/data_untar"  # staging area; contents are always disposable
readonly LOCK_DIR="/tmp/data_locks"
readonly MARKER=".stage_complete"     # written into the final dir once extraction succeeds

# --- End of Configuration --------------------------------------------------

usage() {
    echo "Usage: $0 <dataset_name> [--status|--force|--wait]"
    echo "  --status  report on what is staged, extract nothing"
    echo "  --force   discard any existing staged copy and re-extract"
    echo "  --wait    poll until the source tarball appears, then stage it"
    echo "Available datasets:"
    for d in "${!DATASETS[@]}"; do
        echo "  - $d  <- ${DATASETS[$d]%%|*}"
    done
}

# Print a human-readable picture of what actually landed on disk, so you can tell
# at a glance whether the dataset is complete and sane before launching training.
report_dataset() {
    local root="$1"

    echo
    echo "================ DATASET REPORT: $root ================"
    if [ ! -d "$root" ]; then
        echo "  [MISSING] directory does not exist"
        echo "======================================================="
        return 1
    fi

    if [ -f "$root/$MARKER" ]; then
        echo "  status      : COMPLETE ($(cat "$root/$MARKER"))"
    else
        echo "  status      : INCOMPLETE (no $MARKER marker -- extraction did not finish)"
    fi
    echo "  total size  : $(du -sh "$root" 2>/dev/null | cut -f1)"
    echo "  free on /tmp: $(df -h /tmp | awk 'NR==2{print $4}')"

    echo
    echo "  --- top level ---"
    ls -1 "$root" | head -40 | sed 's/^/    /'
    local n_top
    n_top=$(ls -1 "$root" | wc -l)
    [ "$n_top" -gt 40 ] && echo "    ... ($n_top entries total)"

    # Label CSVs: row count minus the 3 header rows == number of labeled frames.
    echo
    echo "  --- label csvs (frames = rows - 3 header rows) ---"
    local found_csv=0
    for f in "$root"/CollectedData_*.csv; do
        [ -e "$f" ] || continue
        found_csv=1
        local rows frames
        rows=$(wc -l < "$f")
        frames=$((rows - 3))
        printf "    %-45s %7d frames\n" "$(basename "$f")" "$frames"
    done
    [ "$found_csv" -eq 0 ] && echo "    (none at top level)"

    echo
    echo "  --- bbox / calibration files ---"
    for pat in "bboxes_*.csv" "calibrations*.csv" "calibrations"; do
        for f in "$root"/$pat; do
            [ -e "$f" ] || continue
            if [ -d "$f" ]; then
                echo "    $(basename "$f")/  ($(ls -1 "$f" | wc -l) files: $(ls -1 "$f" | head -8 | tr '\n' ' '))"
            else
                printf "    %-45s %7d rows\n" "$(basename "$f")" "$(wc -l < "$f")"
            fi
        done
    done

    # Per-session image counts. This is the check that matters most: it tells you
    # whether every session has every camera, and whether the frame counts match.
    if [ -d "$root/labeled-data" ]; then
        echo
        echo "  --- labeled-data: images per session dir ---"
        local total=0
        for d in "$root"/labeled-data/*/; do
            [ -d "$d" ] || continue
            local n
            n=$(find "$d" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) | wc -l)
            total=$((total + n))
            printf "    %-40s %7d imgs\n" "$(basename "$d")" "$n"
        done
        echo "    ----------------------------------------------------"
        printf "    %-40s %7d imgs\n" "TOTAL" "$total"
    fi

    if [ -d "$root/videos" ]; then
        echo
        echo "  --- videos ---"
        echo "    $(find "$root/videos" -maxdepth 1 -type f | wc -l) files"
        find "$root/videos" -maxdepth 1 -type f -printf '    %f\n' 2>/dev/null | head -10
    fi

    echo "======================================================="
    echo
}

# --- Argument parsing ------------------------------------------------------

if [ $# -lt 1 ]; then
    echo "[ERROR] Please provide a dataset name."
    usage
    exit 1
fi

DATASET_NAME="$1"
MODE="${2:-extract}"

if [ -z "${DATASETS[$DATASET_NAME]}" ]; then
    echo "[ERROR] Unknown dataset: $DATASET_NAME"
    usage
    exit 1
fi

entry="${DATASETS[$DATASET_NAME]}"
SOURCE_TAR="${entry%%|*}"
OUTPUT_NAME="${entry##*|}"
FINAL_PATH="$FINAL_DIR/$OUTPUT_NAME"
STAGE_PATH="$UNTAR_DIR/$OUTPUT_NAME.partial"
LOCK_FILE="$LOCK_DIR/$DATASET_NAME.lock"

mkdir -p "$UNTAR_DIR" "$FINAL_DIR" "$LOCK_DIR"

if [ "$MODE" == "--status" ]; then
    report_dataset "$FINAL_PATH"
    exit 0
fi

# --- Pre-flight ------------------------------------------------------------

echo "--- Staging '$DATASET_NAME' ---"
echo "[INFO] source : $SOURCE_TAR"
echo "[INFO] target : $FINAL_PATH"

if [ "$MODE" == "--force" ]; then
    echo "[INFO] --force: removing existing '$FINAL_PATH'"
    rm -rf "$FINAL_PATH"
fi

# Only the marker proves a previous run finished; a bare directory could be a
# half-extracted tree from an interrupted run, which we must not train on.
if [ -f "$FINAL_PATH/$MARKER" ]; then
    echo "[INFO] Already staged. Nothing to do (use --force to re-extract)."
    report_dataset "$FINAL_PATH"
    exit 0
fi
if [ -d "$FINAL_PATH" ]; then
    echo "[WARN] '$FINAL_PATH' exists but has no $MARKER -- it is incomplete."
    echo "[WARN] Removing it and re-extracting."
    rm -rf "$FINAL_PATH"
fi

# noclobber makes lock creation atomic, so two startup hooks can't race.
# A lock whose owning process is gone is stale -- a hard kill (SIGKILL, studio
# stop) never runs the EXIT trap, and the leftover file would then block this
# dataset forever while claiming another process holds it.
if [ -f "$LOCK_FILE" ]; then
    LOCK_PID=$(cat "$LOCK_FILE" 2>/dev/null)
    if [ -z "$LOCK_PID" ] || ! kill -0 "$LOCK_PID" 2>/dev/null; then
        echo "[INFO] Clearing stale lock (pid ${LOCK_PID:-unknown} is gone)."
        rm -f "$LOCK_FILE"
    fi
fi

if (set -o noclobber; echo "$$" > "$LOCK_FILE") 2>/dev/null; then
    trap 'rm -f "$LOCK_FILE"' EXIT
else
    echo "[INFO] Another process (pid $(cat "$LOCK_FILE" 2>/dev/null)) is already staging this dataset. Exiting."
    exit 0
fi

if [ ! -f "$SOURCE_TAR" ] && [ "$MODE" == "--wait" ]; then
    # Arm-and-forget: park here until the tarball shows up, then fall through and
    # stage it. We don't insist on the configured path -- the file may get dropped
    # into whichever data connection is convenient -- so also sweep the mounted
    # connection roots for the same basename and take the first match.
    TAR_BASENAME=$(basename "$SOURCE_TAR")
    echo "[WAIT] Source not present yet. Polling every 60s for '$TAR_BASENAME'"
    echo "[WAIT]   preferred : $SOURCE_TAR"
    echo "[WAIT]   fallback  : any /teamspace/{s3_folders,lightning_storage,gcs_folders}/*/ (incl. one level down)"
    waited=0
    while true; do
        [ -f "$SOURCE_TAR" ] && break
        found=$(find /teamspace/s3_folders /teamspace/lightning_storage /teamspace/gcs_folders \
                     -maxdepth 3 -name "$TAR_BASENAME" -type f 2>/dev/null | head -1)
        if [ -n "$found" ]; then
            echo "[WAIT] Found it somewhere else: $found"
            SOURCE_TAR="$found"
            break
        fi
        sleep 60
        waited=$((waited + 1))
        [ $((waited % 10)) -eq 0 ] && echo "      still waiting... (${waited} min)"
    done
    echo "[WAIT] Source appeared after ${waited} min: $SOURCE_TAR"
fi

if [ ! -f "$SOURCE_TAR" ]; then
    echo "[ERROR] Source tarball not found: $SOURCE_TAR"
    echo "[ERROR] If this is a Lightning data connection, make sure it is mounted:"
    echo "[ERROR]   ls $(dirname "$(dirname "$SOURCE_TAR")")"
    echo "[ERROR] (Re-run with --wait to poll until it shows up.)"
    exit 1
fi

echo "[INFO] tarball size: $(du -h "$SOURCE_TAR" | cut -f1)"
echo "[INFO] free on /tmp: $(df -h /tmp | awk 'NR==2{print $4}')"

# An upload in progress looks exactly like a complete file, just shorter -- and a
# truncated .tar.xz will happily decompress for 19GB before dying. So refuse to
# start until the size has stopped changing. Skip that wait when the file is
# already old (typical after a one-time copy into lightning_storage).
mtime=$(stat -c %Y "$SOURCE_TAR" 2>/dev/null || echo 0)
now=$(date +%s)
age=$((now - mtime))
if [ "$age" -ge 120 ]; then
    echo "[INFO] tarball mtime is ${age}s old -- skip settle wait"
else
    echo "[RUN] Checking the upload has settled (size stable for 60s)..."
    prev=-1
    stable=0
    while [ $stable -lt 2 ]; do
        cur=$(stat -c %s "$SOURCE_TAR" 2>/dev/null || echo 0)
        if [ "$cur" == "$prev" ] && [ "$cur" != "0" ]; then
            stable=$((stable + 1))
        else
            [ "$prev" != "-1" ] && echo "      still growing: $(numfmt --to=iec "$cur" 2>/dev/null || echo "$cur") (was $(numfmt --to=iec "$prev" 2>/dev/null || echo "$prev"))"
            stable=0
        fi
        prev="$cur"
        [ $stable -lt 2 ] && sleep 30
    done
    echo "[INFO] size stable at $(numfmt --to=iec "$prev" 2>/dev/null || echo "$prev") -- treating upload as complete."
fi

# An .xz footer records the uncompressed size, so we can know exactly how much
# room extraction needs instead of guessing -- worth checking, since running /tmp
# out of space mid-extract wastes the whole (slow) decompression pass.
NEED_BYTES=0
case "$SOURCE_TAR" in
    *.tar.xz|*.txz)
        # Reading the index/footer also proves the archive is whole: a truncated or
        # corrupt .xz fails here in seconds. That matters because the size-stability
        # check above cannot tell "upload finished" from "upload died and stopped
        # growing" -- this can, and it costs a footer read rather than a full pass.
        if ! xz -l "$SOURCE_TAR" >/dev/null 2>&1; then
            echo "[ERROR] '$SOURCE_TAR' is not a readable/complete .xz archive."
            echo "[ERROR] It is most likely truncated -- an interrupted or partial upload."
            echo "[ERROR] Re-upload it and run this again; nothing has been extracted."
            exit 1
        fi
        NEED_BYTES=$(xz --robot -l "$SOURCE_TAR" 2>/dev/null | awk '$1=="totals"{print $5}')
        ;;
    *) NEED_BYTES=$(stat -c %s "$SOURCE_TAR" 2>/dev/null) ;;
esac
if [ -n "$NEED_BYTES" ] && [ "$NEED_BYTES" -gt 0 ] 2>/dev/null; then
    AVAIL_BYTES=$(( $(df -k /tmp | awk 'NR==2{print $4}') * 1024 ))
    echo "[INFO] extracted size will be : $(numfmt --to=iec "$NEED_BYTES")"
    echo "[INFO] free space on /tmp     : $(numfmt --to=iec "$AVAIL_BYTES")"
    # 10% head-room so we don't fill the disk the training run also writes to.
    if [ "$AVAIL_BYTES" -lt $(( NEED_BYTES + NEED_BYTES / 10 )) ]; then
        echo "[ERROR] Not enough free space on /tmp (need ~$(numfmt --to=iec $(( NEED_BYTES + NEED_BYTES / 10 ))) incl. head-room)."
        exit 1
    fi
fi

# --- Extract ---------------------------------------------------------------

rm -rf "$STAGE_PATH"
mkdir -p "$STAGE_PATH"

# Pick the decompressor from the extension. xz -T0 only actually parallelises if
# the archive was written with multiple blocks (xz -T>1); otherwise it silently
# runs single-threaded, which is the slow case for a ~20GB archive.
case "$SOURCE_TAR" in
    *.tar.xz|*.txz) DECOMP=(xz -dc -T0) ;;
    *.tar.gz|*.tgz) DECOMP=(pigz -dc) ;;
    *.tar)          DECOMP=(cat) ;;
    *) echo "[ERROR] Unrecognised archive extension: $SOURCE_TAR"; exit 1 ;;
esac

echo "[RUN] Streaming (${DECOMP[*]} | tar) into $STAGE_PATH ..."
START=$(date +%s)

# Progress reporter: the pipeline gives no output, so sample the growing tree.
( while sleep 30; do
      [ -d "$STAGE_PATH" ] || break
      echo "      ... extracted so far: $(du -sh "$STAGE_PATH" 2>/dev/null | cut -f1) (elapsed $(( ($(date +%s) - START) / 60 ))m)"
  done ) &
PROGRESS_PID=$!

"${DECOMP[@]}" "$SOURCE_TAR" | tar -xf - -C "$STAGE_PATH"
STATUS=$?

kill "$PROGRESS_PID" 2>/dev/null
wait "$PROGRESS_PID" 2>/dev/null

ELAPSED=$(( $(date +%s) - START ))
if [ $STATUS -ne 0 ]; then
    echo "[ERROR] Extraction failed (exit $STATUS) after ${ELAPSED}s. Check tarball integrity / disk space."
    rm -rf "$STAGE_PATH"
    exit 1
fi
echo "[SUCCESS] Extracted in $((ELAPSED / 60))m $((ELAPSED % 60))s -> $(du -sh "$STAGE_PATH" | cut -f1)"

# --- Locate the dataset root inside the extracted tree ---------------------

# Wrapped tars (chickadee, fly, ibl, two-mouse) have a single top-level
# directory named after the dataset. Prefer that over drilling into the first
# `labeled-data` find hits -- two-mouse has nested copies under lp_Cohort_*,
# and using one of those drops calibrations, videos, and the real label CSVs.
DATA_ROOT=""
if [ -d "$STAGE_PATH/$OUTPUT_NAME" ]; then
    DATA_ROOT="$STAGE_PATH/$OUTPUT_NAME"
elif [ -d "$STAGE_PATH/labeled-data" ] || [ -d "$STAGE_PATH/labeled_data" ]; then
    # rat7m-full-crop: no wrapping directory, dataset files sit at archive root.
    DATA_ROOT="$STAGE_PATH"
else
    DATA_ROOT=$(find "$STAGE_PATH" -maxdepth 3 -type d \( -name "labeled-data" -o -name "labeled_data" \) -print -quit 2>/dev/null)
    [ -n "$DATA_ROOT" ] && DATA_ROOT=$(dirname "$DATA_ROOT")
fi

if [ -z "$DATA_ROOT" ]; then
    echo "[WARN] Could not find a dataset root in the extracted tree."
    echo "[WARN] Falling back to the single top-level directory, if there is one."
    n_entries=$(ls -1 "$STAGE_PATH" | wc -l)
    if [ "$n_entries" -eq 1 ] && [ -d "$STAGE_PATH/$(ls -1 "$STAGE_PATH")" ]; then
        DATA_ROOT="$STAGE_PATH/$(ls -1 "$STAGE_PATH")"
    else
        DATA_ROOT="$STAGE_PATH"
    fi
    echo "[WARN] Using: $DATA_ROOT"
    echo "[WARN] Extracted tree looks like:"
    find "$STAGE_PATH" -maxdepth 2 | head -30 | sed 's/^/         /'
fi

echo "[INFO] dataset root inside tarball: ${DATA_ROOT#$STAGE_PATH/}"

# --- Move into place -------------------------------------------------------

echo "[RUN] Moving into place -> $FINAL_PATH"
if mv "$DATA_ROOT" "$FINAL_PATH"; then
    echo "extracted $(date -u '+%Y-%m-%dT%H:%M:%SZ') from $SOURCE_TAR" > "$FINAL_PATH/$MARKER"
    rm -rf "$STAGE_PATH"
    find "$UNTAR_DIR" -mindepth 1 -type d -empty -delete 2>/dev/null || true
    echo "[COMPLETE] '$DATASET_NAME' is staged at $FINAL_PATH"

    # rat7m-full-crop ships calibrations/*.toml but not the calibrations.csv index
    # that cfg.data.camera_params_file points at. It is derived data, so rebuild it
    # here -- otherwise every re-stage (i.e. every studio restart) would need this
    # done by hand and training would die with a FileNotFoundError.
    if [ "$OUTPUT_NAME" == "rat7m-full-crop" ]; then
        # The label/bbox csvs index images under "labeled-data-cropped/", but the
        # tarball ships that directory as "labeled-data/". The images are the same
        # (they are already 256x256 crops) -- it is purely a naming mismatch -- so
        # bridge it with a symlink rather than rewriting 12 csvs or renaming data.
        if [ -d "$FINAL_PATH/labeled-data" ] && [ ! -e "$FINAL_PATH/labeled-data-cropped" ]; then
            ln -sfn labeled-data "$FINAL_PATH/labeled-data-cropped"
            echo "[INFO] linked labeled-data-cropped -> labeled-data (csvs reference the former)"
        fi
        if [ ! -f "$FINAL_PATH/calibrations.csv" ]; then
            echo "[RUN] Generating calibration index CSVs..."
            python "$(dirname "${BASH_SOURCE[0]}")/make_rat7m_calibrations_csv.py" "$FINAL_PATH" \
                || echo "[WARN] calibration index generation failed; run it by hand before training."
        fi
    fi
else
    echo "[ERROR] Failed to move dataset into place."
    exit 1
fi

report_dataset "$FINAL_PATH"
