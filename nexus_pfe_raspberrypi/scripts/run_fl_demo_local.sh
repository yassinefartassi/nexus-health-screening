#!/usr/bin/env bash
# Quick local smoke test: runs the FedAvg server + BOTH clients on this one
# machine (2 simulated devices, synthetic data shards), so you can confirm
# the FL wiring works before doing the real 2-machine (Ubuntu + RPi5) round
# -- see docs/deployment.md for that.
#
# Usage: scripts/run_fl_demo_local.sh [rounds]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROUNDS="${1:-2}"
ADDRESS="127.0.0.1:8080"

# Pre-warm the EfficientNetV2-S ImageNet weights cache (~/.keras/models/) with
# a single sequential download BEFORE the two clients start concurrently.
# Discovered the hard way: on a fresh machine, both clients call
# build_dermatology_model(weights="imagenet") for the first time at nearly
# the same moment, and Keras's get_file() isn't safe against two processes
# racing to download+cache the same file -- one wins, the other crashes with
# a PermissionError (worse on Windows, where an in-use file can't be
# os.remove()'d by the loser). Pre-warming here makes both clients' first
# build_fn() call a cheap local cache hit instead of a race.
python3 -c "import tensorflow as tf; tf.keras.applications.EfficientNetV2S(include_top=False, weights='imagenet', input_shape=(224,224,3))"

python3 -m fl.server --address "$ADDRESS" --rounds "$ROUNDS" --min-clients 2 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" "${CLIENT_PIDS[@]:-}" 2>/dev/null' EXIT
sleep 3  # let the server bind before clients dial in

CLIENT_PIDS=()
python3 -m fl.client --client-id 0 --server "$ADDRESS" --num-clients 2 &
CLIENT_PIDS+=($!)
python3 -m fl.client --client-id 1 --server "$ADDRESS" --num-clients 2 &
CLIENT_PIDS+=($!)

wait "$SERVER_PID"
echo "Demo round(s) complete -- see fl/logs/ for the convergence CSV."
