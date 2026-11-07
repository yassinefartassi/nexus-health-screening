"""
Flower client wrapping ONE federated module at a time -- dermatology
(TF/Keras) or ocular (PyTorch), selected via --model, see fl_client/models.py's
MODEL_REGISTRY. NexusFLClient branches on `spec.framework` because Keras and
PyTorch models don't share a get_weights()/set_weights()/fit()/evaluate()
calling convention: the Keras path calls those methods directly on the
model, the torch path goes through fl_client/torch_ops.py instead.

This is nexus_pfe's (the Streamlit web app's) FL client -- a second, separate
client implementation from nexus_pfe_raspberrypi/fl/client.py, but built to
the same FederatedModule-shaped contract and the same non-IID partitioning
recipe (fl_client/data_partition.py), so a round can mix clients from both
apps and nexus_pfe_server_fed's FedAvg aggregation still lines weight arrays
up correctly. Point it at the server with --server <ip>:<port>:

    # dermatology, client 0
    python -m fl_client.client --client-id 0 --server 192.168.1.10:8080 --model dermatology --num-clients 2

    # ocular, client 1
    python -m fl_client.client --client-id 1 --server 192.168.1.10:8081 --model ocular --num-clients 2

This module is intentionally a standalone CLI, not something the Streamlit
UI calls in-process: local training can take minutes and Streamlit's
single-threaded rerun model isn't a good fit for a long-lived blocking
training loop. app.py's "Apprentissage federe" tab documents this command
and shows real round-by-round results once a run has happened (see
nexus_pfe_server_fed's status_api.py) -- it does not launch training itself.

--------------------------------------------------------------------------
PRIVACY NOTE -- read this before touching the networking below
--------------------------------------------------------------------------
Flower's client/server protocol only ever exchanges model *weight arrays*
(via get_parameters()/fit()/evaluate() below -- see fl_client/torch_ops.py
and tf.keras.Model.get_weights()/set_weights()) and small scalar metrics
(loss, accuracy, sample counts). The image tensors in self.shard
(fl_client/data_partition.py) never leave this process: training happens
locally, in this Python process, on this machine, and only the RESULTING
PARAMETER UPDATE -- never a single pixel of a training image -- is sent to
nexus_pfe_server_fed to be FedAvg-aggregated with the other client's update.
That is the whole privacy argument for federated learning in this project.
"""
from __future__ import annotations

import argparse
import os

import flwr  # not aliased to "fl" -- the package itself is named flwr

from fl_client import torch_ops
from fl_client.models import MODEL_REGISTRY

DEFAULT_SERVER_ADDRESS = "127.0.0.1:8080"
DEFAULT_LOCAL_EPOCHS = 1
DEFAULT_BATCH_SIZE = 8


class NexusFLClient(flwr.client.NumPyClient):
    """Generic client: which model + which non-IID shard it wraps is decided
    entirely by (model_name, client_id) via fl_client.models.MODEL_REGISTRY."""

    def __init__(
        self,
        model_name: str,
        client_id: int,
        num_clients: int,
        local_epochs: int = DEFAULT_LOCAL_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        spec = MODEL_REGISTRY[model_name]
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.framework = spec.framework

        print(f"[web client {client_id}] loading non-IID '{model_name}' shard (simulated partition, not real user data)...")
        self.shard = spec.load_shard_fn(client_id=client_id, num_clients=num_clients)
        print(
            f"[web client {client_id}] shard ready: "
            f"{len(self.shard.train_labels)} train / {len(self.shard.val_labels)} val samples, "
            f"classes present: {sorted(set(self.shard.train_labels.tolist()))}"
        )

        self.model = spec.build_fn(len(spec.class_names))

    def get_parameters(self, config):
        if self.framework == "torch":
            return torch_ops.get_weights(self.model)
        return self.model.get_weights()

    def fit(self, parameters, config):
        # `parameters` is the current global model's weight arrays, nothing
        # else -- see module docstring. Only self.shard's images (already
        # loaded locally, never transmitted) are used for training below.
        if self.framework == "torch":
            torch_ops.set_weights(self.model, parameters)
            metrics = torch_ops.fit(
                self.model,
                self.shard.train_images,
                self.shard.train_labels,
                epochs=self.local_epochs,
                batch_size=self.batch_size,
            )
            print(f"[web client {self.client_id}] fit -> loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f}")
            return torch_ops.get_weights(self.model), len(self.shard.train_labels), metrics

        self.model.set_weights(parameters)
        history = self.model.fit(
            self.shard.train_images,
            self.shard.train_labels,
            epochs=self.local_epochs,
            batch_size=self.batch_size,
            verbose=0,
        )
        metrics = {
            "accuracy": float(history.history["accuracy"][-1]),
            "loss": float(history.history["loss"][-1]),
        }
        print(f"[web client {self.client_id}] fit -> loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f}")
        # Only the updated weight arrays + sample count + scalar metrics
        # cross back out -- see module docstring.
        return self.model.get_weights(), len(self.shard.train_labels), metrics

    def evaluate(self, parameters, config):
        if self.framework == "torch":
            torch_ops.set_weights(self.model, parameters)
            loss, metrics = torch_ops.evaluate(self.model, self.shard.val_images, self.shard.val_labels)
            print(f"[web client {self.client_id}] eval -> loss={loss:.4f} acc={metrics['accuracy']:.4f}")
            return loss, len(self.shard.val_labels), metrics

        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(
            self.shard.val_images, self.shard.val_labels, batch_size=self.batch_size, verbose=0
        )
        print(f"[web client {self.client_id}] eval -> loss={loss:.4f} acc={accuracy:.4f}")
        return float(loss), len(self.shard.val_labels), {"accuracy": float(accuracy)}


def main() -> None:
    parser = argparse.ArgumentParser(description="NEXUS_PFE federated learning client (web app)")
    parser.add_argument("--client-id", type=int, default=int(os.environ.get("FL_CLIENT_ID", "0")))
    parser.add_argument("--num-clients", type=int, default=int(os.environ.get("FL_NUM_CLIENTS", "2")))
    parser.add_argument(
        "--server", type=str, default=os.environ.get("FL_SERVER_ADDRESS", DEFAULT_SERVER_ADDRESS)
    )
    parser.add_argument(
        "--model", type=str, default=os.environ.get("FL_MODEL", "dermatology"), choices=list(MODEL_REGISTRY)
    )
    parser.add_argument("--local-epochs", type=int, default=DEFAULT_LOCAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    client = NexusFLClient(
        model_name=args.model,
        client_id=args.client_id,
        num_clients=args.num_clients,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
    )
    flwr.client.start_client(server_address=args.server, client=client.to_client())


if __name__ == "__main__":
    main()
