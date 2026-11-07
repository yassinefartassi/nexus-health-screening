"""
Flower FedAvg federated-learning SERVER for NEXUS_PFE.

Runs on THIS Windows 11 machine only. This machine is ONLY the FL server --
it never runs inference and is never itself a client. Two clients dial into
this process over the LAN: an Ubuntu desktop and a Raspberry Pi 5 (see
docs/fl_setup.md, including the Windows Firewall rule you need or clients
will just hang with no useful error).

--------------------------------------------------------------------------
Modular, framework-agnostic design
--------------------------------------------------------------------------
Each federated module (dermatology, ocular) gets its own independently-
versioned global model (see model_registry.py) and, crucially, its own ML
framework: dermatology is TensorFlow/Keras, ocular is PyTorch (see
modules/base.py's FederatedModule for why and how). This file is a single
generic round loop written ENTIRELY in terms of that interface -- it never
imports tensorflow or torch, and never branches on which framework the
`--model` argument selected. One server process federates ONE module at a
time (pick with --model); to run dermatology and ocular concurrently, run
two server processes on two ports.

rPPG is deterministic CHROM/POS signal processing with no trainable weights
and is NEVER federated -- it has no entry in MODULE_REGISTRY.

--------------------------------------------------------------------------
PRIVACY NOTE (core thesis argument) -- read this before touching networking
--------------------------------------------------------------------------
Flower's client/server protocol only ever exchanges model *weight arrays*
(a client's fit()/evaluate() return values -- see modules/base.py's
get_weights/set_weights) and small scalar metrics (loss, accuracy, sample
counts). Raw images NEVER cross the network in either direction: each
client trains locally on its own device and only the resulting parameter
update is sent here to be FedAvg-aggregated. This server never receives,
stores, or has access to a single patient image, for either module.

Differential privacy and secure aggregation are intentionally NOT
implemented here -- this is a proof-of-concept scope decision, not an
oversight. See docs/fl_setup.md's "Future work" section.

--------------------------------------------------------------------------
The round loop, explicitly spelled out
--------------------------------------------------------------------------
Each round: clients fit() locally -> FedAvg aggregates the weight updates
here (aggregate_fit) -> the aggregated model is, in this order:
  1. evaluated on a held-out test set (centralized, server-side, via
     module.evaluate() -- this is what produces the accuracy-vs-round
     convergence curve, the thesis deliverable),
  2. versioned to models/<module>/global_v{round}.<ext> via
     model_registry.py (module.save_checkpoint()),
  3. re-exported to ONNX via module.export_onnx().
Step 3 is NOT optional -- see onnx_export.py's module docstring for exactly
what silently breaks if you skip it.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import flwr  # not aliased to "fl" -- the package itself is named flwr
from flwr.common import Metrics, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from fl_server.config import CHECKPOINTS_DIR, DEFAULT_HOST, DEFAULT_MIN_CLIENTS, DEFAULT_PORT, DEFAULT_ROUNDS, LOGS_DIR
from fl_server.datasets.dermatology import load_holdout_set as load_dermatology_holdout
from fl_server.datasets.ocular import load_holdout_set as load_ocular_holdout
from fl_server.model_registry import ModelRegistry
from fl_server.modules import MODULE_REGISTRY
from fl_server.modules.base import FederatedModule

# Each module supplies its own held-out loader (different preprocessing
# conventions -- see datasets/dermatology.py vs. datasets/ocular.py) behind
# a common (data_dir) -> (images, labels) signature.
HOLDOUT_LOADERS = {
    "dermatology": load_dermatology_holdout,
    "ocular": load_ocular_holdout,
}
# Env var each module's held-out directory is read from, if --test-dir isn't passed.
HOLDOUT_ENV_VARS = {
    "dermatology": "DERMNET_TEST_DIR",
    "ocular": "OCULAR_TEST_DIR",
}


def _weighted_average(client_metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Examples-weighted average of client-reported fit metrics (their local
    train loss/accuracy) -- purely informational logging, not the accuracy
    number the thesis convergence curve is built from (that's the
    server-side held-out evaluation below)."""
    total = sum(n for n, _ in client_metrics)
    if total == 0:
        return {}
    out: Dict[str, float] = {}
    for key, value in client_metrics[0][1].items():
        if isinstance(value, (int, float)):
            out[key] = sum(n * m[key] for n, m in client_metrics) / total
    return out


class FederatingFedAvg(FedAvg):
    """FedAvg plus the per-round housekeeping this server exists to do:
    version the aggregated model, re-export it to ONNX, and log
    accuracy/loss/client-count/communication-size -- all delegated to
    `module` (a FederatedModule), so this class is identical for
    dermatology and ocular rounds."""

    def __init__(
        self,
        *args,
        module: FederatedModule,
        registry: ModelRegistry,
        eval_model,
        csv_path: Path,
        data_source: str = "unknown",
        **kwargs,
    ):
        super().__init__(*args, fit_metrics_aggregation_fn=_weighted_average, **kwargs)
        self.module = module
        self.registry = registry
        self.eval_model = eval_model
        self.csv_path = csv_path
        # "real" (a real curated test-dir was found) or "synthetic" (fell
        # back to the seeded smoke-test set) -- see model_registry.py's
        # module docstring. Decided once in main() from whether --test-dir
        # resolved to a real directory; every round this server process runs
        # uses the same held-out set, so this doesn't change per round.
        self.data_source = data_source
        self._last_num_clients = 0
        self._last_comm_bytes = 0

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["round", "timestamp", "eval_loss", "eval_accuracy", "n_clients", "comm_bytes", "data_source"]
            )

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        self._last_num_clients = len(results)
        if aggregated_parameters is not None:
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            weights_bytes = sum(arr.nbytes for arr in ndarrays)
            # Approximation: each participating client both downloaded the
            # previous global model and uploaded its own update, each the
            # same size as the aggregated weights (architecture is fixed
            # across rounds/clients). Good enough for a PoC communication-
            # cost number; not a packet-level measurement.
            self._last_comm_bytes = weights_bytes * len(results) * 2
        print(f"[server] round {server_round} fit  -> {metrics} ({len(results)} client(s))")
        return aggregated_parameters, metrics

    def evaluate(self, server_round: int, parameters: Parameters):
        """Centralized, server-side evaluation on the held-out test set.
        Flower calls this once with the initial parameters before round 1
        (server_round == 0 -- the pretrained baseline) and again after every
        round's aggregate_fit. This is the single trigger point for (1)
        versioning the model via model_registry and (2) the ONNX re-export
        -- see module docstring."""
        result = super().evaluate(server_round, parameters)
        if result is None:
            return None
        loss, metrics = result

        ndarrays = parameters_to_ndarrays(parameters)
        self.module.set_weights(self.eval_model, ndarrays)

        num_clients = self._last_num_clients if server_round > 0 else 0
        comm_bytes = self._last_comm_bytes if server_round > 0 else 0

        checkpoint_path = self.registry.save_round(
            round_num=server_round,
            module=self.module,
            model=self.eval_model,
            num_clients=num_clients,
            accuracy=metrics.get("accuracy"),
            loss=loss,
            comm_bytes=comm_bytes,
            data_source=self.data_source,
        )
        onnx_path = self.module.export_onnx(self.eval_model, checkpoint_path.with_suffix(".onnx"))
        print(
            f"[server] round {server_round} eval -> loss={loss:.4f} acc={metrics.get('accuracy'):.4f} "
            f"| data_source={self.data_source} | saved {checkpoint_path.name} | exported {onnx_path.name}"
        )

        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    server_round,
                    time.strftime("%Y-%m-%dT%H:%M:%S"),
                    loss,
                    metrics.get("accuracy"),
                    num_clients,
                    comm_bytes,
                    self.data_source,
                ]
            )
        return result


def _make_evaluate_fn(module: FederatedModule, eval_model, holdout_images, holdout_labels):
    def evaluate_fn(server_round: int, parameters_ndarrays, config):
        module.set_weights(eval_model, parameters_ndarrays)
        loss, metrics = module.evaluate(eval_model, holdout_images, holdout_labels)
        return loss, metrics

    return evaluate_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="NEXUS_PFE federated learning server (FedAvg)")
    parser.add_argument("--host", type=str, default=os.environ.get("FL_SERVER_HOST", DEFAULT_HOST))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("FL_SERVER_PORT", str(DEFAULT_PORT)))
    )
    parser.add_argument(
        "--rounds", type=int, default=int(os.environ.get("FL_ROUNDS", str(DEFAULT_ROUNDS)))
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        default=int(os.environ.get("FL_MIN_CLIENTS", str(DEFAULT_MIN_CLIENTS))),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("FL_MODEL", "dermatology"),
        choices=list(MODULE_REGISTRY),
        help="Which federated module this server process aggregates. rPPG has no entry -- "
        "it's deterministic signal processing with no trainable weights, never federated.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.environ.get("FL_INIT_CHECKPOINT"),
        help="Pretrained checkpoint used as round-0 starting weights. Defaults to "
        "models/checkpoints/<model>_pretrained<ext> (.keras for dermatology, .pt for ocular). "
        "See docs/fl_setup.md.",
    )
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Start from random weights if --checkpoint is missing. For pipeline smoke-testing "
        "ONLY -- never use this for real thesis convergence numbers.",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help="Class-per-subfolder held-out test directory for server-side evaluation. Falls back "
        "to this module's env var (DERMNET_TEST_DIR / OCULAR_TEST_DIR), then a seeded synthetic set.",
    )
    args = parser.parse_args()

    module = MODULE_REGISTRY[args.model]
    print(f"[server] module: {module.name} ({len(module.class_names)} classes, "
          f"framework checkpoint format: {module.checkpoint_extension})")

    template_model = module.build()

    checkpoint_arg = args.checkpoint or os.environ.get("FL_INIT_CHECKPOINT")
    checkpoint_path = Path(
        checkpoint_arg or (CHECKPOINTS_DIR / f"{module.name}_pretrained{module.checkpoint_extension}")
    )

    if checkpoint_path.exists():
        pretrained_model = module.load_checkpoint(checkpoint_path)
        initial_weights = module.get_weights(pretrained_model)
        expected = module.get_weights(template_model)
        if len(initial_weights) != len(expected):
            raise ValueError(
                f"Checkpoint {checkpoint_path} has {len(initial_weights)} weight arrays, but "
                f"{module.name}'s architecture expects {len(expected)}. Has modules/{module.name}.py "
                "diverged from whatever produced this checkpoint?"
            )
        print(f"[server] round-0 weights loaded from pretrained checkpoint: {checkpoint_path}")
    elif args.allow_random_init:
        initial_weights = module.get_weights(template_model)
        print(
            f"[server] WARNING: checkpoint {checkpoint_path} not found -- starting from RANDOM "
            "weights (--allow-random-init). Fine for a pipeline smoke test, meaningless for the "
            "thesis convergence curve."
        )
    else:
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {checkpoint_path}\n"
            f"Place {module.name}'s centrally-trained checkpoint there (see docs/fl_setup.md "
            "step 2), pass --checkpoint <path>, or explicitly pass --allow-random-init to run a "
            "smoke test with random initial weights."
        )

    test_dir = args.test_dir or os.environ.get(HOLDOUT_ENV_VARS[module.name])
    # Mirrors HOLDOUT_LOADERS[...]'s own real-vs-synthetic decision (see
    # datasets/dermatology.py / datasets/ocular.py) so the manifest/CSV tag
    # always matches what was actually evaluated on -- computed once here,
    # not re-derived per round, since this server process's held-out set
    # never changes across rounds.
    data_source = "real" if test_dir and Path(test_dir).is_dir() else "synthetic"
    holdout_images, holdout_labels = HOLDOUT_LOADERS[module.name](data_dir=test_dir)
    print(f"[server] held-out test set: {len(holdout_labels)} sample(s) (data_source={data_source})")

    registry = ModelRegistry(module_name=module.name)
    eval_model = module.build()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = LOGS_DIR / f"{module.name}_fedavg_{timestamp}.csv"

    strategy = FederatingFedAvg(
        module=module,
        registry=registry,
        eval_model=eval_model,
        csv_path=csv_path,
        data_source=data_source,
        initial_parameters=ndarrays_to_parameters(initial_weights),
        evaluate_fn=_make_evaluate_fn(module, eval_model, holdout_images, holdout_labels),
        fraction_fit=1.0,
        fraction_evaluate=0.0,  # server-side centralized eval only, see evaluate() above
        min_fit_clients=args.min_clients,
        min_available_clients=args.min_clients,
        min_evaluate_clients=0,
    )

    address = f"{args.host}:{args.port}"
    print(
        f"[server] starting FedAvg ({module.name}) on {address} for {args.rounds} round(s), "
        f"min {args.min_clients} client(s)"
    )
    print(f"[server] logging to {csv_path}")
    print(f"[server] versioned models -> models/{module.name}/global_v<round>{module.checkpoint_extension} (+ .onnx)")

    flwr.server.start_server(
        server_address=address,
        config=flwr.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    print(f"[server] done. Plot convergence with: python -m fl_server.plot_convergence {csv_path}")


if __name__ == "__main__":
    main()
