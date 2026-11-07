"""
Shared paths/constants for the FL server. Kept in one place because
model_registry.py, onnx_export.py, the datasets/ and modules/ packages, and
server.py all need to agree on where models/logs live and what a "module"
(dermatology, ocular) is called on disk.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
DEFAULT_ROUNDS = 5
DEFAULT_MIN_CLIENTS = 2

# Both federated modules happen to use 224x224 input -- coincidence, not a
# shared assumption. Each module is free to define its own if that changes.
IMAGE_SIZE = (224, 224)

DERMATOLOGY_CLASS_NAMES_PATH = Path(__file__).resolve().parent / "dermatology_class_names.json"
OCULAR_CLASS_NAMES_PATH = Path(__file__).resolve().parent / "ocular_class_names.json"
