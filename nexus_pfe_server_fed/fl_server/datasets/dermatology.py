"""
Loads the server-side HELD-OUT test set used for centralized, per-round
evaluation of the aggregated dermatology global model (see
modules/dermatology.py and server.py's evaluate_fn) -- this is what
produces the accuracy-vs-round convergence curve, the key thesis
deliverable.

This is deliberately separate from client training data: clients never send
their data here (see server.py's privacy note), so this test set must be
either a small curated held-out directory placed on this machine, or, absent
that, a seeded synthetic fallback so the FL pipeline can be smoke-tested
end-to-end before real held-out data exists.

Two modes, picked automatically:
  - Real data: point DERMNET_TEST_DIR (or --test-dir) at a class-per-subfolder
    image directory, same layout as tf.keras.utils.image_dataset_from_directory
    expects and the same layout dermnet_v2s_curated.ipynb was trained on.
  - Synthetic fallback: seeded per-class color+noise images, just structured
    enough that accuracy/loss actually move round to round. NOT meaningful
    for real thesis numbers -- only for exercising the pipeline.

Returns raw [0, 255] float32 NHWC arrays -- dermatology's EfficientNetV2-S
bakes its own rescaling in as its first layers (see modules/dermatology.py).
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import numpy as np

from fl_server.config import DERMATOLOGY_CLASS_NAMES_PATH, IMAGE_SIZE

with open(DERMATOLOGY_CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    DERMATOLOGY_CLASS_NAMES: List[str] = json.load(f)

# Deliberately different from the clients' shard seed (42, see
# nexus_pfe_raspberrypi/fl/data_partition.py) so the synthetic fallback here
# never coincides with synthetic client training data.
HOLDOUT_SEED = 999
DEFAULT_SYNTHETIC_PER_CLASS = 15


def _load_image_raw(path: str) -> np.ndarray:
    from PIL import Image

    image = Image.open(path).convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
    return np.asarray(image).astype(np.float32)  # HWC, [0, 255]


def _index_real_directory(data_dir: str) -> Tuple[List[str], List[int]]:
    paths: List[str] = []
    labels: List[int] = []
    for label_idx, class_name in enumerate(DERMATOLOGY_CLASS_NAMES):
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
    num_classes = len(DERMATOLOGY_CLASS_NAMES)
    images = np.empty((num_classes * per_class, *IMAGE_SIZE, 3), dtype=np.float32)
    labels = np.empty(num_classes * per_class, dtype=np.int64)

    i = 0
    for label_idx in range(num_classes):
        base_color = rng.uniform(30, 220, size=3)
        for _ in range(per_class):
            noise = rng.normal(0, 25, size=(*IMAGE_SIZE, 3))
            img = np.clip(base_color[None, None, :] + noise, 0, 255).astype(np.float32)
            images[i] = img
            labels[i] = label_idx
            i += 1
    return images, labels


def load_holdout_set(
    data_dir: Optional[str] = None, per_class: int = DEFAULT_SYNTHETIC_PER_CLASS
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (images, labels) for dermatology's server-side held-out test
    set. data_dir: class-per-subfolder directory. Falls back to the
    DERMNET_TEST_DIR env var, then to the seeded synthetic set above."""
    data_dir = data_dir or os.environ.get("DERMNET_TEST_DIR")

    if data_dir and os.path.isdir(data_dir):
        paths, label_list = _index_real_directory(data_dir)
        if not paths:
            raise FileNotFoundError(
                f"DERMNET_TEST_DIR={data_dir} exists but no class subfolders matching "
                f"{DERMATOLOGY_CLASS_NAMES} were found."
            )
        images = np.stack([_load_image_raw(p) for p in paths])
        labels = np.array(label_list, dtype=np.int64)
        return images, labels

    print(
        "[fl_server] DERMNET_TEST_DIR not set/found -- evaluating dermatology on a seeded "
        "synthetic held-out set. Fine for a pipeline smoke test; the resulting accuracy numbers "
        "are meaningless for the thesis until a real held-out directory is provided."
    )
    return _make_synthetic_holdout(per_class)
