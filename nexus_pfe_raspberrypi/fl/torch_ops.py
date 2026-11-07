"""
PyTorch model-ops for the ocular federated module, client-side.

fl/client.py's NexusFLClient is otherwise Keras-only (it calls
model.get_weights()/set_weights()/fit()/evaluate() directly, which only
exist on tf.keras.Model). Ocular is a torchvision ResNet-18, which has none
of those methods, so this module provides the equivalent free functions --
mirroring nexus_pfe_server_fed/fl_server/modules/ocular.py::OcularModule
method-for-method, since weight arrays produced here must be positionally
compatible with what that server-side module's get_weights()/set_weights()
expect (same architecture, same state_dict() key order).

Input convention: ImageNet-normalized float32 NCHW -- see
fl/data_partition.py::load_ocular_shard. This is a DIFFERENT convention from
dermatology's raw-[0,255] NHWC; do not mix them up.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import numpy as np


def build_ocular_model(num_classes: int) -> Any:
    """ResNet-18 backbone (frozen) + fc = Sequential(Dropout(0.4), Linear(512,
    num_classes)) -- identical architecture to
    fl_server/modules/ocular.py::OcularModule.build(), required so FedAvg
    weight arrays line up positionally between server and this client."""
    import torch.nn as nn
    import torchvision

    model = torchvision.models.resnet18(weights=None)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def get_weights(model: Any) -> List[np.ndarray]:
    return [tensor.cpu().numpy() for tensor in model.state_dict().values()]


def set_weights(model: Any, weights: List[np.ndarray]) -> None:
    import torch

    keys = model.state_dict().keys()
    state_dict = OrderedDict((key, torch.tensor(value)) for key, value in zip(keys, weights))
    model.load_state_dict(state_dict, strict=True)


def fit(
    model: Any, train_images: np.ndarray, train_labels: np.ndarray, epochs: int, batch_size: int
) -> Dict[str, float]:
    import torch
    import torch.nn.functional as F

    model.train()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    images_t = torch.from_numpy(train_images)
    labels_t = torch.from_numpy(train_labels).long()

    last_loss, last_acc = 0.0, 0.0
    n = len(labels_t)
    for _ in range(epochs):
        permutation = torch.randperm(n)
        total_loss, total_correct = 0.0, 0
        for start in range(0, n, batch_size):
            idx = permutation[start : start + batch_size]
            batch_images, batch_labels = images_t[idx], labels_t[idx]

            optimizer.zero_grad()
            logits = model(batch_images)
            loss = F.cross_entropy(logits, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)
            total_correct += (logits.argmax(dim=1) == batch_labels).sum().item()
        last_loss = total_loss / n
        last_acc = total_correct / n
    return {"loss": float(last_loss), "accuracy": float(last_acc)}


def evaluate(model: Any, images: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict[str, float]]:
    import torch
    import torch.nn.functional as F

    model.eval()
    with torch.no_grad():
        images_t = torch.from_numpy(images)
        labels_t = torch.from_numpy(labels).long()
        logits = model(images_t)
        loss = F.cross_entropy(logits, labels_t).item()
        accuracy = (logits.argmax(dim=1) == labels_t).float().mean().item()
    return float(loss), {"accuracy": float(accuracy)}
