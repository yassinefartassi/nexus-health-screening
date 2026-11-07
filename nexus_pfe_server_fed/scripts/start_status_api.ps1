# Starts the read-only FL status API on this Windows machine -- the JSON
# endpoint nexus_pfe's "Apprentissage federe" tab and nexus_pfe_mobile's
# "Federated Learning" screen poll for real round/accuracy history. This is
# NOT the FedAvg server (that's start_server.ps1, gRPC, port 8080) -- it's a
# separate, independent, stdlib-only process that just reads
# models\<module>\manifest.json and serves it as JSON. Safe to start/stop any
# time, independently of whether a FedAvg round is currently running.
#
# Usage: scripts\start_status_api.ps1 [-Port 8090]
param(
    [int]$Port = 8090
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "Starting FL status API on 0.0.0.0:$Port (GET /status, /status/<module>)..."
python -m fl_server.status_api --port $Port
