#!/usr/bin/env bash
# Launch TensorBoard over a results tree.
#
#   ./tensorboard.sh                      # outputs/two-mouse_s on :6006
#   ./tensorboard.sh <logdir> <port>
#
# Pointed at the results *root* rather than a single run, so every experiment and
# seed appears as its own comparable curve, and runs started later show up without
# a restart.
#
# View it at (substitute the port):
#   https://lightning.ai/zuckerman-institute/project-lp3d/studios/test-rat-nature/web-ui?port=6006
#
# Note: tensorboard 2.20 imports pkg_resources, which setuptools removed in 81.
# This studio has setuptools 80.10.2, so it works as-is -- but a setuptools upgrade
# will break it. If that happens, pin a copy for this process alone rather than
# downgrading underneath a running sweep:
#
#   pip install --target /teamspace/studios/this_studio/.tbdeps "setuptools<81"
#
# then prepend PYTHONPATH="/teamspace/studios/this_studio/.tbdeps" to the command
# below.

set -euo pipefail

LOGDIR="${1:-/teamspace/studios/this_studio/outputs/two-mouse_s}"
PORT="${2:-6006}"

mkdir -p "$LOGDIR"
echo "TensorBoard: $LOGDIR on :$PORT"
echo "https://lightning.ai/zuckerman-institute/project-lp3d/studios/test-rat-nature/web-ui?port=$PORT"

exec python -m tensorboard.main --logdir "$LOGDIR" --port "$PORT" --bind_all
