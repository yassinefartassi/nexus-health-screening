"""
FederatedModule -- the framework-agnostic contract every federated module
(dermatology, ocular, and whatever comes after) implements.

--------------------------------------------------------------------------
Why this exists (a deliberate architectural point of the thesis)
--------------------------------------------------------------------------
NEXUS_PFE's two trainable modules use DIFFERENT ML frameworks:

  - dermatology: TensorFlow/Keras (.keras checkpoints), see modules/dermatology.py
  - ocular:      PyTorch (.pt checkpoints),            see modules/ocular.py

Flower itself is framework-agnostic: its client/server protocol only ever
exchanges a `List[numpy.ndarray]` (see get_weights/set_weights below) and
small scalar metrics. It does not know or care whether those arrays came
from `model.get_weights()` (Keras) or `[t.numpy() for t in
model.state_dict().values()]` (PyTorch). This class is where that fact gets
turned into a concrete design: fl_server/server.py, model_registry.py, and
onnx_export.py are written ONCE, entirely in terms of this interface, and
never import `tensorflow` or `torch` directly, or branch on which framework
a module uses. Each module supplies its own framework-specific
implementation behind this same interface; the server-side pipeline is
otherwise identical for both.

Do NOT try to convert one framework's weights into the other's -- that's
not what this abstraction is for. A dermatology round and an ocular round
each aggregate weights only among clients running that same module, using
that module's own architecture. The abstraction's job is to let the SAME
generic server code (round loop, versioning, ONNX export, evaluation
logging) run either one, not to make the two interchangeable.

--------------------------------------------------------------------------
fit() and this repo
--------------------------------------------------------------------------
This repo is the FL SERVER ONLY -- it never trains, so fit() is never
called from fl_server/server.py. It's part of this interface anyway
because it's the same per-module contract the FL CLIENT side must
implement (see nexus_pfe_raspberrypi/fl/) to actually produce local weight
updates. Keeping fit() here, even unused server-side, is what makes this
one contract shared by both sides of the network instead of two documents
that can drift apart.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


class FederatedModule(ABC):
    name: str
    class_names: List[str]
    checkpoint_extension: str  # ".keras" or ".pt" -- so model_registry.py never hardcodes a framework

    @abstractmethod
    def build(self) -> Any:
        """Returns a fresh, framework-native model instance with this
        module's architecture (uninitialized/framework-default weights --
        callers immediately overwrite via set_weights() or load_checkpoint())."""

    @abstractmethod
    def load_checkpoint(self, path: Path) -> Any:
        """Loads a saved checkpoint from disk and returns a framework-native
        model instance (used for round-0 pretrained starting weights)."""

    @abstractmethod
    def save_checkpoint(self, model: Any, path_stem: Path) -> Path:
        """Saves a framework-native model instance to disk, appending this
        module's checkpoint_extension to path_stem. Returns the actual path
        written. The save-side counterpart of load_checkpoint(); used every
        round by model_registry.py to version the aggregated model."""

    @abstractmethod
    def get_weights(self, model: Any) -> List[np.ndarray]:
        """The ONLY thing that crosses the Flower client/server boundary.
        Must return weights in a stable, deterministic order for a given
        architecture (Keras: model.get_weights() order; PyTorch:
        state_dict() insertion order) so aggregation lines arrays up
        correctly across clients."""

    @abstractmethod
    def set_weights(self, model: Any, weights: List[np.ndarray]) -> None:
        """Inverse of get_weights() -- loads an aggregated weight list back
        into a framework-native model instance, in place."""

    @abstractmethod
    def fit(
        self, model: Any, train_images: np.ndarray, train_labels: np.ndarray, epochs: int, batch_size: int
    ) -> Dict[str, float]:
        """Runs local training for `epochs` epochs, in place on `model`.
        Returns a metrics dict (at least {"loss": ..., "accuracy": ...}).
        NOT called by fl_server/server.py -- see module docstring."""

    @abstractmethod
    def evaluate(self, model: Any, images: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Evaluates `model` on (images, labels), returns (loss, metrics).
        Called server-side every round on the held-out test set -- this is
        what produces the accuracy-vs-round convergence curve."""

    @abstractmethod
    def export_onnx(self, model: Any, onnx_path: Path) -> Path:
        """Re-exports `model` to ONNX at onnx_path. Called every round,
        right after save_checkpoint() -- see onnx_export.py's module
        docstring for why this step is not optional."""
