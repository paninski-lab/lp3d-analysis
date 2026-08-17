#!/bin/bash

# Stage Nature datasets into /tmp on every studio start (GCP).
# /tmp is node-local NVMe and is wiped when the studio stops, so this
# re-extracts from lightning_storage / gcs_folders. Each job is idempotent
# once /tmp/data/<name>/.stage_complete exists.
#
# Backgrounded so the studio is usable immediately. Watch
# ~/scripts/_stage_<name>.log. --wait parks until the tarball is visible
# (first boot after a copy, or a slow fuse mount).

DATASETS=(
    rat7m-full-crop
    chickadee-crop
    fly-anipose
    ibl-mouse
    two-mouse
    mirror-mouse-separate
)

for name in "${DATASETS[@]}"; do
    nohup bash ~/scripts/stage_dataset.sh "$name" --wait \
        > ~/scripts/_stage_${name}.log 2>&1 &
done
