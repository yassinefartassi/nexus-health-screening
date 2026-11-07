"""
Loads the server-side HELD-OUT test set used for centralized, per-round
evaluation of the aggregated ocular global model (see modules/ocular.py and
server.py's evaluate_fn).

Mirrors datasets/dermatology.py's structure (real class-per-subfolder
directory, else a seeded synthetic fallback) but with a DIFFERENT
preprocessing convention, because ocular is a different framework with a
different deployed-model convention:

  - dermatology (TF/Keras): raw [0, 255] float32 NHWC -- EfficientNetV2-S
    bakes its own rescaling into the model graph (see datasets/dermatology.py).
  - ocular (PyTorch): ImageNet-normalized float32 NCHW -- PyTorch does NOT
    bake preprocessing into the exported graph the way the Keras
    application does, so normalization has to happen here, in the data
    pipeline, to match how the currently-deployed ocular ONNX model
    (nexus_pfe/src/inference/ocular_infer.py, PREPROCESS_MODE="imagenet")
    expects its input. Do not swap these two conventions between modules.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import numpy as np

from fl_server.config import IMAGE_SIZE, OCULAR_CLASS_NAMES_PATH

with open(OCULAR_CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    OCULAR_CLASS_NAMES: List[str] = json.load(f)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Different from dermatology's HOLDOUT_SEED (999) and the clients' shard
# seed (42) -- keeps every synthetic fallback set in this project distinct.
HOLDOUT_SEED = 2024
DEFAULT_SYNTHETIC_PER_CLASS = 15


def _load_image_normalized(path: str) -> np.ndarray:
    """Resize, scale to [0,1], ImageNet-normalize, transpose to CHW --
    matches src/preprocessing/image_utils.py's "imagenet" mode, the
    convention the deployed ocular ONNX model already uses."""
    from PIL import Image

    image = Image.open(path).convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
    array = np.asarray(image).astype(np.float32) / 255.0  # HWC, [0, 1]
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(array, (2, 0, 1))  # CHW


def _index_real_directory(data_dir: str) -> Tuple[List[str], List[int]]:
    paths: List[str] = []
    labels: List[int] = []
    for label_idx, class_name in enumerate(OCULAR_CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(class_dir, fname))
                labels.append(label_idx)
    return paths, labels


def _make_synthetic_holdout(per_class: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(HOLDOUT_SEED)
    num_classes = len(OCULAR_CLASS_NAMES)
    images = np.empty((num_classes * per_class, 3, *IMAGE_SIZE), dtype=np.float32)
    labels = np.empty(num_classes * per_class, dtype=np.int64)

    i = 0
    for label_idx in range(num_classes):
        base = rng.uniform(-1.5, 1.5, size=3).astype(np.float32)  # roughly normalized range
        for _ in range(per_class):
            noise = rng.normal(0, 0.3, size=(3, *IMAGE_SIZE)).astype(np.float32)
            img = base[:, None, None] + noise
            images[i] = img
            labels[i] = label_idx
            i += 1
    return images, labels


def load_holdout_set(
    data_dir: Optional[str] = None, per_class: int = DEFAULT_SYNTHETIC_PER_CLASS
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (images, labels) for ocular's server-side held-out test set,
    images already ImageNet-normalized NCHW float32. Falls back to the
    OCULAR_TEST_DIR env var, then to the seeded synthetic set above."""
    data_dir = data_dir or os.environ.get("OCULAR_TEST_DIR")

    if data_dir and os.path.isdir(data_dir):
        paths, label_list = _index_real_directory(data_dir)
        if not paths:
            raise FileNotFoundError(
                f"OCULAR_TEST_DIR={data_dir} exists but no class subfolders matching "
                f"{OCULAR_CLASS_NAMES} were found."
            )
        images = np.stack([_load_image_normalized(p) for p in paths])
        labels = np.array(label_list, dtype=np.int64)
        return images, labels

    print(
        "[fl_server] OCULAR_TEST_DIR not set/found -- evaluating ocular on a seeded synthetic "
        "held-out set. Fine for a pipeline smoke test; the resulting accuracy numbers are "
        "meaningless for the thesis until a real held-out directory is provided."
    )
    return _make_synthetic_holdout(per_class)
