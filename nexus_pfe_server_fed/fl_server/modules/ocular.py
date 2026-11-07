"""
Ocular's FederatedModule implementation -- PyTorch.

Architecture: a ResNet-18 backbone (frozen) + `fc = Sequential(Dropout,
Linear(512, 5))` over the 5 classes from
nexus_pfe/models/eye_disease_class_names.json (Cataract, Conjunctivitis,
Eyelid, Normal, Uveitis). This matches models/checkpoints/ocular_pretrained.pt
(a PyTorch export of the currently-deployed ocular model,
nexus_pfe/models/eye_disease_model.onnx) key-for-key -- confirmed from that
checkpoint's state_dict keys (layer1-4 with 2 BasicBlocks each and
downsample only on layer2/3/4 block 0 -> ResNet-18, not -34/-50; final
`fc.1.weight` shaped (5, 512) -> a Linear sits at index 1 of an `fc`
Sequential, index 0 being a parameter-free Dropout). The exact dropout
probability isn't recoverable from the checkpoint (Dropout has no learned
weights) -- 0.4 here is a placeholder matching dermatology's head; it only
affects local training (fit()), not checkpoint loading or inference.

If dropout rate or the ocular training pipeline changes later, only this
file and its checkpoint need to change -- server.py, model_registry.py, and
onnx_export.py are untouched (that's the point of the FederatedModule
abstraction, see modules/base.py).

Input convention: ImageNet-normalized float32 NCHW, matching the
preprocessing the currently-deployed ocular ONNX model already expects
(src/preprocessing/image_utils.py, PREPROCESS_MODE="imagenet") -- see
datasets/ocular.py. Unlike dermatology's EfficientNetV2-S, PyTorch does NOT
bake this normalization into the exported graph, so it must stay applied in
the data pipeline both for local training (client-side) and here in
evaluate() (server-side). This is a DIFFERENT convention from dermatology's
raw-[0,255] input -- do not mix them up.

`torch`/`torchvision` are imported inside each method, not at module level,
on purpose: MODULE_REGISTRY (modules/__init__.py) constructs a
DermatologyModule and an OcularModule up front regardless of which --model
a given server run actually uses, so this keeps a dermatology-only run
working even on a machine where torch isn't installed. requirements.txt
still installs both -- this is a defensive fallback, not the intended setup.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from fl_server.config import OCULAR_CLASS_NAMES_PATH
from fl_server.modules.base import FederatedModule
from fl_server.onnx_export import export_torch_to_onnx

with open(OCULAR_CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    OCULAR_CLASS_NAMES: List[str] = json.load(f)


class OcularModule(FederatedModule):
    name = "ocular"
    class_names = OCULAR_CLASS_NAMES
    checkpoint_extension = ".pt"

    def build(self) -> Any:
        import torch.nn as nn
        import torchvision

        # weights=None, same rationale as dermatology's build(): every
        # weight is immediately overwritten via set_weights() from either
        # the pretrained checkpoint (round 0) or an aggregated update
        # (round > 0), so there's no reason to fetch ImageNet weights here.
        model = torchvision.models.resnet18(weights=None)
        for param in model.parameters():
            param.requires_grad = False
        # fc.0 = Dropout (no weights, so its keys never show up in a
        # state_dict), fc.1 = Linear -- this two-element Sequential, not a
        # bare Linear, is what models/checkpoints/ocular_pretrained.pt's
        # state_dict keys (fc.1.weight, fc.1.bias) require; see module
        # docstring.
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.fc.in_features, len(self.class_names)),
        )
        # Only the new head trains locally on clients -- the frozen backbone
        # still needs its (untrained-here) weights included in
        # get_weights()/set_weights() so FedAvg has a consistent, complete
        # weight list to aggregate every round.
        return model

    def load_checkpoint(self, path: Path) -> Any:
        import torch

        model = self.build()
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    def save_checkpoint(self, model: Any, path_stem: Path) -> Path:
        import torch

        pt_path = Path(path_stem).with_suffix(self.checkpoint_extension)
        torch.save(model.state_dict(), pt_path)
        return pt_path

    def get_weights(self, model: Any) -> List[np.ndarray]:
        return [tensor.cpu().numpy() for tensor in model.state_dict().values()]

    def set_weights(self, model: Any, weights: List[np.ndarray]) -> None:
        import torch

        keys = model.state_dict().keys()
        state_dict = OrderedDict((key, torch.tensor(value)) for key, value in zip(keys, weights))
        model.load_state_dict(state_dict, strict=True)

    def fit(
        self, model: Any, train_images: np.ndarray, train_labels: np.ndarray, epochs: int, batch_size: int
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

    def evaluate(self, model: Any, images: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict[str, float]]:
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

    def export_onnx(self, model: Any, onnx_path: Path) -> Path:
        return export_torch_to_onnx(model, onnx_path)
