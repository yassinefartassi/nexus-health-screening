#!/usr/bin/env bash
# Starts the FedAvg server (fl/server.py) on this machine (the Ubuntu box).
# Run once; both clients (Ubuntu + RPi5) then dial into it -- see
# docs/deployment.md for the full 2-client walkthrough.
#
# Usage: scripts/start_fl_server.sh [rounds] [min_clients]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROUNDS="${1:-${FL_ROUNDS:-5}}"
MIN_CLIENTS="${2:-${FL_MIN_CLIENTS:-2}}"
ADDRESS="${FL_SERVER_ADDRESS:-0.0.0.0:8080}"

echo "Starting FL server on $ADDRESS for $ROUNDS round(s), waiting for $MIN_CLIENTS client(s)..."
python3 -m fl.server --address "$ADDRESS" --rounds "$ROUNDS" --min-clients "$MIN_CLIENTS"
