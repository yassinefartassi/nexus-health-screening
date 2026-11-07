#!/usr/bin/env bash
# Deploys this project to the Raspberry Pi 5 over SSH via rsync.
#
# Usage:
#   scripts/deploy_to_rpi.sh <user@rpi-host> [remote-path]
#
# Example (RPi5 reachable on the Ubuntu machine's hotspot):
#   scripts/deploy_to_rpi.sh pi@192.168.4.2 ~/nexus_pfe_raspberrypi
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <user@rpi-host> [remote-path]" >&2
  exit 1
fi

REMOTE="$1"
REMOTE_PATH="${2:-~/nexus_pfe_raspberrypi}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Deploying $LOCAL_DIR -> $REMOTE:$REMOTE_PATH"

rsync -avz --progress \
  --exclude 'venv/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache/' \
  --exclude '.streamlit/' \
  --exclude 'fl/logs/*.csv' \
  --exclude 'fl/logs/*.png' \
  "$LOCAL_DIR"/ "$REMOTE:$REMOTE_PATH"/

cat <<EOF

Done. On the RPi5:
  ssh $REMOTE
  cd $REMOTE_PATH
  python3 -m venv --system-site-packages venv && source venv/bin/activate
  pip install -r requirements.txt -r requirements-fl.txt -r requirements-rpi5.txt
  # (see requirements-rpi5.txt for the picamera2/libcamera apt step)
EOF
