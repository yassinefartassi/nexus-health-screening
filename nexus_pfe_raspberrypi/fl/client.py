"""
Flower client wrapping ONE federated module at a time -- dermatology
(TF/Keras) or ocular (PyTorch), selected via --model, see fl/models.py's
MODEL_REGISTRY. The two frameworks don't share a get_weights()/set_weights()/
fit()/evaluate() calling convention, so NexusFLClient branches on
`spec.framework`: the Keras path calls those methods directly on the model,
the torch path goes through fl/torch_ops.py instead.

Usable unchanged on the Ubuntu desktop or the RPi5: it only needs a
--client-id (which non-IID shard to load, see fl/data_partition.py) and a
--server address to dial. Nothing about this file depends on which physical
machine it runs on.

    # Ubuntu (client 0), dermatology
    python -m fl.client --client-id 0 --server 192.168.1.10:8080 --model dermatology

    # RPi5 (client 1), ocular, reached over SSH per docs/deployment.md
    python -m fl.client --client-id 1 --server 192.168.1.10:8081 --model ocular

--------------------------------------------------------------------------
PRIVACY NOTE -- read this before touching the networking below
--------------------------------------------------------------------------
Flower's client/server protocol only ever exchanges model *weight arrays*
(via get_parameters()/set_weights() below) and small scalar metrics (loss,
accuracy, sample counts). Raw images never leave this process -- each client
trains locally on its own shard (fl/data_partition.py) and only the
resulting parameter updates cross the network to be averaged by
fl/server.py's FedAvg strategy. That's the core privacy argument for
federated learning here: patient images stay on the device that captured
them.
"""
from __future__ import annotations

import argparse
import os

import flwr  # not aliased to "fl" -- this package is itself named fl, aliasing would shadow it

from fl import torch_ops
from fl.models import MODEL_REGISTRY

DEFAULT_SERVER_ADDRESS = "127.0.0.1:8080"
DEFAULT_LOCAL_EPOCHS = 1
DEFAULT_BATCH_SIZE = 8


class NexusFLClient(flwr.client.NumPyClient):
    """Generic client: which model + which non-IID shard it wraps is decided
    entirely by (model_name, client_id) via fl.models.MODEL_REGISTRY, so
    adding a second federated module doesn't require a second client class."""

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

        print(f"[client {client_id}] loading non-IID '{model_name}' shard...")
        self.shard = spec.load_shard_fn(client_id=client_id, num_clients=num_clients)
        print(
            f"[client {client_id}] shard ready: "
            f"{len(self.shard.train_labels)} train / {len(self.shard.val_labels)} val samples, "
            f"classes present: {sorted(set(self.shard.train_labels.tolist()))}"
        )

        self.model = spec.build_fn(len(spec.class_names))

    def get_parameters(self, config):
        if self.framework == "torch":
            return torch_ops.get_weights(self.model)
        return self.model.get_weights()

    def fit(self, parameters, config):
        if self.framework == "torch":
            torch_ops.set_weights(self.model, parameters)
            metrics = torch_ops.fit(
                self.model,
                self.shard.train_images,
                self.shard.train_labels,
                epochs=self.local_epochs,
                batch_size=self.batch_size,
            )
            print(f"[client {self.client_id}] fit -> loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f}")
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
        print(f"[client {self.client_id}] fit -> loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f}")
        return self.model.get_weights(), len(self.shard.train_labels), metrics

    def evaluate(self, parameters, config):
        if self.framework == "torch":
            torch_ops.set_weights(self.model, parameters)
            loss, metrics = torch_ops.evaluate(self.model, self.shard.val_images, self.shard.val_labels)
            print(f"[client {self.client_id}] eval -> loss={loss:.4f} acc={metrics['accuracy']:.4f}")
            return loss, len(self.shard.val_labels), metrics

        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(
            self.shard.val_images, self.shard.val_labels, batch_size=self.batch_size, verbose=0
        )
        print(f"[client {self.client_id}] eval -> loss={loss:.4f} acc={accuracy:.4f}")
        return float(loss), len(self.shard.val_labels), {"accuracy": float(accuracy)}


def main() -> None:
    parser = argparse.ArgumentParser(description="NEXUS_PFE federated learning client")
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
