"""
Splits the dermatology and ocular datasets into non-IID shards, one per
federated client, to simulate heterogeneous devices (Ubuntu desktop, RPi5,
...) since no real multi-device user data exists yet for this proof of
concept.

Two modes, picked automatically per model:
  - Real data: point DERMNET_DATA_DIR / OCULAR_DATA_DIR at a class-per-subfolder
    image directory (the same layout tf.keras.utils.image_dataset_from_directory
    expects -- the layout the EfficientNetV2-S model was trained on, see
    dermnet_v2s_curated.ipynb).
  - Synthetic fallback: if no directory is configured/found, generates small
    seeded synthetic per-class images so the FL pipeline (partitioning,
    client training, server aggregation) can be exercised end-to-end today,
    without waiting on a real curated dataset drop.

Non-IID skew: for each class, samples are split across clients using a
Dirichlet(alpha) draw, so most clients see a lopsided mix of classes rather
than a uniform random split -- the standard way to simulate heterogeneous
federated devices (Hsu, Qi & Brown, 2019, "Measuring the Effects of
Non-Identical Data Distribution for Federated Visual Classification").
Lower alpha = more skewed/non-IID; alpha -> inf approaches IID.
`_dirichlet_non_iid_split` is generic over any labels array, so both
load_dermatology_shard and load_ocular_shard share it.

Everything here is seeded (numpy.random.default_rng(seed)) so the same
(client_id, num_clients, alpha, seed) always produces the same shard --
needed so FL rounds are reproducible and comparable across machines.

Dermatology (TF/Keras) and ocular (PyTorch) use DIFFERENT preprocessing
conventions -- raw [0,255] NHWC vs. ImageNet-normalized NCHW, respectively
-- mirroring nexus_pfe_server_fed/fl_server/datasets/{dermatology,ocular}.py.
Do not mix them up.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

FL_DIR = Path(__file__).resolve().parent
CLASS_NAMES_PATH = FL_DIR / "dermatology_class_names.json"
OCULAR_CLASS_NAMES_PATH = FL_DIR / "ocular_class_names.json"

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    DERMATOLOGY_CLASS_NAMES: List[str] = json.load(f)

with open(OCULAR_CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    OCULAR_CLASS_NAMES: List[str] = json.load(f)

IMAGE_SIZE = (224, 224)  # matches dermnet_v2s_curated.ipynb's IMG_SIZE

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_ALPHA = 0.5  # Dirichlet concentration: lower = more non-IID
DEFAULT_SEED = 42
DEFAULT_SYNTHETIC_PER_CLASS = 40  # samples/class in the synthetic fallback

# Different from dermatology's DEFAULT_SEED (42) and the server-side holdout
# seeds -- keeps every synthetic fallback set in this project distinct.
OCULAR_DEFAULT_SEED = 43


@dataclass
class Shard:
    """One client's local dermatology data shard, already split train/val."""

    client_id: int
    train_images: np.ndarray  # (N, H, W, 3) float32, raw [0, 255] range
    train_labels: np.ndarray  # (N,) int64
    val_images: np.ndarray
    val_labels: np.ndarray
    class_names: List[str]


def _index_real_directory(data_dir: str, class_names: List[str]) -> Tuple[List[str], List[int]]:
    """Scans a class-per-subfolder directory into (filepath, label) lists,
    ordered/matched against `class_names`. Shared by both dermatology and
    ocular sharding."""
    paths: List[str] = []
    labels: List[int] = []
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(class_dir, fname))
                labels.append(label_idx)
    return paths, labels


def _load_image_raw(path: str) -> np.ndarray:
    """Resize + cast to float32, no rescaling. EfficientNetV2-S's Keras
    application bakes its own Rescaling/Normalization in as the first
    layers, so it expects raw [0, 255] pixel values fed straight in.

    Do NOT apply the "imagenet" mean/std mode from
    src/preprocessing/image_utils.py here -- that mode belongs to the ocular
    model, not dermatology. As of the 2026-07 hardening pass, the deployed
    dermatology model (models/dermnet_model.onnx) is this same
    EfficientNetV2-S architecture and already expects raw [0, 255] input, so
    this convention now matches deployment too -- but mixing the two modes up
    still silently produces wrong predictions/gradients, with no error
    raised, so the distinction stays load-bearing.
    """
    from PIL import Image

    image = Image.open(path).convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
    return np.asarray(image).astype(np.float32)  # HWC, [0, 255]


def _load_image_normalized(path: str) -> np.ndarray:
    """Resize, scale to [0,1], ImageNet-normalize, transpose to CHW -- matches
    src/preprocessing/image_utils.py's "imagenet" mode and
    fl_server/datasets/ocular.py::_load_image_normalized, the convention the
    deployed ocular ONNX model already uses. Unlike dermatology's raw-[0,255]
    NHWC, do NOT feed this straight in without normalizing."""
    from PIL import Image

    image = Image.open(path).convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
    array = np.asarray(image).astype(np.float32) / 255.0  # HWC, [0, 1]
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(array, (2, 0, 1))  # CHW


def _make_synthetic_index(rng: np.random.Generator, per_class: int) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic per-class synthetic images (a color/noise bias per
    class) standing in for real DermNet data until a curated dataset
    directory is wired in via DERMNET_DATA_DIR. Enough structure for the
    model to actually learn something (so the FL loop's loss/accuracy moves),
    without pretending to be real skin images."""
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


def _dirichlet_non_iid_split(
    labels: np.ndarray, num_clients: int, alpha: float, seed: int
) -> List[np.ndarray]:
    """Returns, for each client, an array of sample indices into `labels`,
    dividing each class's samples across clients via a Dirichlet(alpha)
    draw (see module docstring for the non-IID simulation recipe)."""
    rng = np.random.default_rng(seed)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for label_idx in np.unique(labels):
        class_indices = np.where(labels == label_idx)[0]
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        for client_id, part in enumerate(np.split(class_indices, split_points)):
            client_indices[client_id].extend(part.tolist())

    return [np.array(sorted(idxs)) for idxs in client_indices]


def load_dermatology_shard(
    client_id: int,
    num_clients: int = 2,
    data_dir: Optional[str] = None,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
    val_fraction: float = 0.2,
) -> Shard:
    """Builds this client's non-IID dermatology shard.

    data_dir: class-per-subfolder DermNet-style directory. Falls back to the
    DERMNET_DATA_DIR env var, then to a seeded synthetic dataset if neither
    is set or the path doesn't exist -- see module docstring.
    """
    if client_id < 0 or client_id >= num_clients:
        raise ValueError(f"client_id {client_id} out of range for num_clients={num_clients}")

    data_dir = data_dir or os.environ.get("DERMNET_DATA_DIR")

    if data_dir and os.path.isdir(data_dir):
        paths, label_list = _index_real_directory(data_dir, DERMATOLOGY_CLASS_NAMES)
        if not paths:
            raise FileNotFoundError(
                f"DERMNET_DATA_DIR={data_dir} exists but no class subfolders matching "
                f"{DERMATOLOGY_CLASS_NAMES} were found."
            )
        labels = np.array(label_list, dtype=np.int64)
        client_splits = _dirichlet_non_iid_split(labels, num_clients, alpha, seed)
        my_indices = client_splits[client_id]
        images = np.stack([_load_image_raw(paths[i]) for i in my_indices])
        client_labels = labels[my_indices]
    else:
        rng = np.random.default_rng(seed)
        all_images, all_labels = _make_synthetic_index(rng, DEFAULT_SYNTHETIC_PER_CLASS)
        client_splits = _dirichlet_non_iid_split(all_labels, num_clients, alpha, seed)
        my_indices = client_splits[client_id]
        images = all_images[my_indices]
        client_labels = all_labels[my_indices]

    if len(client_labels) < 2:
        raise RuntimeError(
            f"Client {client_id} received only {len(client_labels)} sample(s) after the "
            f"non-IID split -- raise `alpha` (less skew) or DEFAULT_SYNTHETIC_PER_CLASS."
        )

    # Deterministic per-client train/val split (separate seed stream so it
    # doesn't disturb the cross-client sharding above).
    split_rng = np.random.default_rng(seed + 1000 + client_id)
    order = split_rng.permutation(len(client_labels))
    n_val = max(1, int(len(order) * val_fraction))
    val_idx, train_idx = order[:n_val], order[n_val:]

    return Shard(
        client_id=client_id,
        train_images=images[train_idx],
        train_labels=client_labels[train_idx],
        val_images=images[val_idx],
        val_labels=client_labels[val_idx],
        class_names=DERMATOLOGY_CLASS_NAMES,
    )


def _make_synthetic_index_ocular(rng: np.random.Generator, per_class: int) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic per-class synthetic images in roughly-ImageNet-normalized
    CHW range, standing in for real ocular data until a curated dataset
    directory is wired in via OCULAR_DATA_DIR -- mirrors
    fl_server/datasets/ocular.py::_make_synthetic_holdout's recipe so the
    scale is consistent with what the server's synthetic holdout set uses."""
    num_classes = len(OCULAR_CLASS_NAMES)
    images = np.empty((num_classes * per_class, 3, *IMAGE_SIZE), dtype=np.float32)
    labels = np.empty(num_classes * per_class, dtype=np.int64)

    i = 0
    for label_idx in range(num_classes):
        base = rng.uniform(-1.5, 1.5, size=3).astype(np.float32)
        for _ in range(per_class):
            noise = rng.normal(0, 0.3, size=(3, *IMAGE_SIZE)).astype(np.float32)
            img = base[:, None, None] + noise
            images[i] = img
            labels[i] = label_idx
            i += 1
    return images, labels


def load_ocular_shard(
    client_id: int,
    num_clients: int = 2,
    data_dir: Optional[str] = None,
    alpha: float = DEFAULT_ALPHA,
    seed: int = OCULAR_DEFAULT_SEED,
    val_fraction: float = 0.2,
) -> Shard:
    """Builds this client's non-IID ocular shard. Same Dirichlet-split recipe
    as load_dermatology_shard, but ImageNet-normalized NCHW images instead of
    raw [0,255] NHWC -- see module docstring.

    data_dir: class-per-subfolder directory (Cataract/Conjunctivitis/Eyelid/
    Normal/Uveitis subfolders). Falls back to the OCULAR_DATA_DIR env var,
    then to a seeded synthetic dataset if neither is set or the path doesn't
    exist.
    """
    if client_id < 0 or client_id >= num_clients:
        raise ValueError(f"client_id {client_id} out of range for num_clients={num_clients}")

    data_dir = data_dir or os.environ.get("OCULAR_DATA_DIR")

    if data_dir and os.path.isdir(data_dir):
        paths, label_list = _index_real_directory(data_dir, OCULAR_CLASS_NAMES)
        if not paths:
            raise FileNotFoundError(
                f"OCULAR_DATA_DIR={data_dir} exists but no class subfolders matching "
                f"{OCULAR_CLASS_NAMES} were found."
            )
        labels = np.array(label_list, dtype=np.int64)
        client_splits = _dirichlet_non_iid_split(labels, num_clients, alpha, seed)
        my_indices = client_splits[client_id]
        images = np.stack([_load_image_normalized(paths[i]) for i in my_indices])
        client_labels = labels[my_indices]
    else:
        rng = np.random.default_rng(seed)
        all_images, all_labels = _make_synthetic_index_ocular(rng, DEFAULT_SYNTHETIC_PER_CLASS)
        client_splits = _dirichlet_non_iid_split(all_labels, num_clients, alpha, seed)
        my_indices = client_splits[client_id]
        images = all_images[my_indices]
        client_labels = all_labels[my_indices]

    if len(client_labels) < 2:
        raise RuntimeError(
            f"Client {client_id} received only {len(client_labels)} sample(s) after the "
            f"non-IID split -- raise `alpha` (less skew) or DEFAULT_SYNTHETIC_PER_CLASS."
        )

    split_rng = np.random.default_rng(seed + 1000 + client_id)
    order = split_rng.permutation(len(client_labels))
    n_val = max(1, int(len(order) * val_fraction))
    val_idx, train_idx = order[:n_val], order[n_val:]

    return Shard(
        client_id=client_id,
        train_images=images[train_idx],
        train_labels=client_labels[train_idx],
        val_images=images[val_idx],
        val_labels=client_labels[val_idx],
        class_names=OCULAR_CLASS_NAMES,
    )
