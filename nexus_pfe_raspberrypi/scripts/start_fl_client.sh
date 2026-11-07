#!/usr/bin/env bash
# Starts one FL client (fl/client.py). Run on the Ubuntu machine AND on the
# RPi5, each with a different <client_id>, both pointed at the same server
# address -- see docs/deployment.md for the full 2-client walkthrough.
#
# Usage: scripts/start_fl_client.sh <client_id> <server_host:port> [num_clients]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ $# -lt 2 ]; then
  echo "Usage: $0 <client_id> <server_host:port> [num_clients]" >&2
  exit 1
fi

CLIENT_ID="$1"
SERVER="$2"
NUM_CLIENTS="${3:-2}"

echo "Starting FL client $CLIENT_ID -> server $SERVER (of $NUM_CLIENTS client(s))"
python3 -m fl.client --client-id "$CLIENT_ID" --server "$SERVER" --num-clients "$NUM_CLIENTS"
