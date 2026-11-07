"""
*** LOCAL-ONLY DEMO/SMOKE-TEST SCAFFOLD -- NOT THE PRODUCTION FL SERVER ***

nexus_pfe_server_fed/fl_server/server.py is the authoritative NEXUS_PFE FL
server: it supports both dermatology (TF/Keras) and ocular (PyTorch) via a
FederatedModule abstraction, requires a real pretrained checkpoint, does
mandatory per-round centralized held-out evaluation, model-registry
versioning, and automatic ONNX re-export every round. fl/client.py in THIS
repo is the real client and is weight-compatible with that server.

This file is a smaller, single-framework (dermatology-only), no-versioning,
no-registry FedAvg server kept around for fully local single-repo smoke
testing (see scripts/run_fl_demo_local.sh) -- it predates the split into
nexus_pfe_server_fed and was never deleted because it's still a handy quick
sanity check that doesn't need the other repo checked out. Do not point real
clients at this for anything beyond local smoke testing.

Flower FedAvg server for the NEXUS_PFE federated learning proof of concept.

Runs on the Ubuntu machine; the Ubuntu desktop and the RPi5 each run one
fl/client.py pointed at this server's address (see docs/deployment.md for a
full 2-client walkthrough). The server only ever aggregates model weight
arrays and small scalar metrics sent up by clients -- it never sees raw
images (see fl/client.py's privacy note for the full argument).

Logs one row per round (train + eval loss/accuracy, weighted by each
client's sample count) to fl/logs/<model>_fedavg_<timestamp>.csv, so
convergence can be plotted afterwards with fl/plot_convergence.py -- the
thesis deliverable this whole PoC is meant to produce.

After the final round, the server also saves the aggregated weights as a
.keras checkpoint (fl/logs/<model>_fedavg_<timestamp>.keras). That file is
NOT automatically picked up by the Streamlit inference tab -- ONNX is
inference-only and src/inference/dermatology_infer.py loads a static .onnx
file, so it will silently keep serving the OLD weights until you explicitly
re-export the checkpoint with `python -m fl.export_to_onnx`. See
docs/deployment.md step 4 for that explicit step.

Proof-of-concept scope: plain FedAvg, no secure aggregation or differential
privacy (see docs/deployment.md's "Future work" section).
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr  # not aliased to "fl" -- this package is itself named fl
from flwr.common import Metrics, Parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from fl.models import MODEL_REGISTRY

LOGS_DIR = Path(__file__).resolve().parent / "logs"

DEFAULT_ROUNDS = 5
DEFAULT_MIN_CLIENTS = 2


def _weighted_average(client_metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregates a list of (num_examples, metrics_dict) tuples -- one per
    participating client -- into a single examples-weighted average. Passed
    to FedAvg for both fit and evaluate metrics."""
    total_examples = sum(n for n, _ in client_metrics)
    if total_examples == 0:
        return {}
    averaged: Dict[str, float] = {}
    for key, value in client_metrics[0][1].items():
        if isinstance(value, (int, float)):
            averaged[key] = sum(n * m[key] for n, m in client_metrics) / total_examples
    return averaged


class LoggingFedAvg(FedAvg):
    """FedAvg that additionally appends one CSV row per round so
    convergence can be plotted -- see fl/plot_convergence.py."""

    def __init__(self, *args, log_path: Path, **kwargs):
        super().__init__(
            *args,
            fit_metrics_aggregation_fn=_weighted_average,
            evaluate_metrics_aggregation_fn=_weighted_average,
            **kwargs,
        )
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["round", "train_loss", "train_accuracy", "eval_loss", "eval_accuracy", "n_clients"]
            )
        self._pending_fit_metrics: Metrics = {}
        self.latest_parameters: Optional[Parameters] = None

    def aggregate_fit(self, server_round, results, failures):
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        self._pending_fit_metrics = metrics
        self.latest_parameters = parameters
        print(f"[server] round {server_round} fit  -> {metrics} ({len(results)} client(s))")
        return parameters, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        print(f"[server] round {server_round} eval -> loss={loss} metrics={metrics} ({len(results)} client(s))")

        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    server_round,
                    self._pending_fit_metrics.get("loss", ""),
                    self._pending_fit_metrics.get("accuracy", ""),
                    loss if loss is not None else "",
                    metrics.get("accuracy", ""),
                    len(results),
                ]
            )
        self._pending_fit_metrics = {}
        return loss, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="NEXUS_PFE federated learning server (FedAvg)")
    parser.add_argument("--address", type=str, default=os.environ.get("FL_SERVER_ADDRESS", "0.0.0.0:8080"))
    parser.add_argument("--rounds", type=int, default=int(os.environ.get("FL_ROUNDS", str(DEFAULT_ROUNDS))))
    parser.add_argument(
        "--min-clients", type=int, default=int(os.environ.get("FL_MIN_CLIENTS", str(DEFAULT_MIN_CLIENTS)))
    )
    parser.add_argument("--model", type=str, default=os.environ.get("FL_MODEL", "dermatology"))
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"{args.model}_fedavg_{timestamp}.csv"

    strategy = LoggingFedAvg(
        log_path=log_path,
        min_available_clients=args.min_clients,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
    )

    print(f"[server] starting FedAvg on {args.address} for {args.rounds} rounds, logging to {log_path}")
    flwr.server.start_server(
        server_address=args.address,
        config=flwr.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    print(f"[server] done. Plot convergence with: python -m fl.plot_convergence {log_path}")

    if strategy.latest_parameters is None:
        print("[server] no aggregated parameters were produced (0 successful rounds) -- nothing to save.")
        return

    spec = MODEL_REGISTRY[args.model]
    model = spec.build_fn(len(spec.class_names))
    model.set_weights(parameters_to_ndarrays(strategy.latest_parameters))
    checkpoint_path = LOGS_DIR / f"{args.model}_fedavg_{timestamp}.keras"
    model.save(checkpoint_path)
    print(f"[server] saved final aggregated model to {checkpoint_path}")
    print(
        "[server] IMPORTANT: this checkpoint is NOT used by the Streamlit app yet -- "
        "src/inference/dermatology_infer.py still loads the static models/dermnet_model.onnx "
        "file until you re-export it:\n"
        f"    python -m fl.export_to_onnx {checkpoint_path}\n"
        "See docs/deployment.md step 4."
    )


if __name__ == "__main__":
    main()
