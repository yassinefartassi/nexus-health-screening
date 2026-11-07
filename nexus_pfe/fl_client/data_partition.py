"""
Splits the dermatology and ocular training data into non-IID shards, one per
federated client, to simulate multiple heterogeneous devices holding data
this web app's single machine doesn't actually have -- there is no real
multi-user database behind this app (see app.py), so this is a **simulated**
partition, not a live user base.

Mirrors nexus_pfe_raspberrypi/fl/data_partition.py's recipe exactly (same
Dirichlet-split algorithm, same seed convention) so that a dermatology round
run from this app and one run from the Raspberry Pi client are shards of the
*same* simulated population, not two unrelated datasets -- both are
weight-compatible clients of the same nexus_pfe_server_fed server. Class
names are loaded directly from this project's own models/ directory (the
single source of truth already used by src/inference/) rather than a
duplicated copy.

Two modes, picked automatically per model:
  - Real data: point DERMNET_DATA_DIR / OCULAR_DATA_DIR at a class-per-subfolder
    image directory (same layout dermnet_v2s_curated.ipynb was trained on).
  - Synthetic fallback: seeded per-class color+noise images, just structured
    enough that local training actually moves loss/accuracy, so the FL
    pipeline can be exercised end-to-end before a curated dataset exists.

Non-IID skew: for each class, samples are split across clients using a
Dirichlet(alpha) draw, so most clients see a lopsided mix of classes rather
than a uniform random split (Hsu, Qi & Brown, 2019, "Measuring the Effects of
Non-Identical Data Distribution for Federated Visual Classification"). Lower
alpha = more skewed/non-IID.

Everything here is seeded (numpy.random.default_rng(seed)) so the same
(client_id, num_clients, alpha, seed) always produces the same shard --
reproducible across runs and machines.

Dermatology (TF/Keras) and ocular (PyTorch) use DIFFERENT preprocessing
conventions -- raw [0,255] NHWC vs. ImageNet-normalized NCHW, respectively,
matching src/preprocessing/image_utils.py's "raw255" and "imagenet" modes.
Do not mix them up.
"""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DERMATOLOGY_CLASS_NAMES_PATH = MODELS_DIR / "dermnet_class_names.pkl"
OCULAR_CLASS_NAMES_PATH = MODELS_DIR / "eye_disease_class_names.json"

with open(DERMATOLOGY_CLASS_NAMES_PATH, "rb") as f:
    DERMATOLOGY_CLASS_NAMES: List[str] = pickle.load(f)

with open(OCULAR_CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    OCULAR_CLASS_NAMES: List[str] = json.load(f)

IMAGE_SIZE = (224, 224)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_ALPHA = 0.5  # Dirichlet concentration: lower = more non-IID
# Same seed as nexus_pfe_raspberrypi/fl/data_partition.py's dermatology seed,
# on purpose -- this app's client 0/1 and the RPi client's client 0/1 draw
# from the identical simulated population, they're just different named
# clients within it.
DEFAULT_SEED = 42
OCULAR_DEFAULT_SEED = 43
DEFAULT_SYNTHETIC_PER_CLASS = 40


@dataclass
class Shard:
    """One client's local data shard, already split train/val. Only ever
    held in this process's memory -- see client.py's privacy note, nothing
    here is sent anywhere."""

    client_id: int
    train_images: np.ndarray
    train_labels: np.ndarray
    val_images: np.ndarray
    val_labels: np.ndarray
    class_names: List[str]


def _index_real_directory(data_dir: str, class_names: List[str]) -> Tuple[List[str], List[int]]:
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
    """Raw [0, 255] HWC float32 -- dermatology's convention, matches
    src/preprocessing/image_utils.py's "raw255" mode. EfficientNetV2-S bakes
    its own rescaling into the model graph."""
    from PIL import Image

    image = Image.open(path).convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
    return np.asarray(image).astype(np.float32)


def _load_image_normalized(path: str) -> np.ndarray:
    """ImageNet-normalized CHW float32 -- ocular's convention, matches
    src/preprocessing/image_utils.py's "imagenet" mode."""
    from PIL import Image

    image = Image.open(path).convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
    array = np.asarray(image).astype(np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(array, (2, 0, 1))


def _make_synthetic_index(
    rng: np.random.Generator, class_names: List[str], per_class: int, chw: bool
) -> Tuple[np.ndarray, np.ndarray]:
    num_classes = len(class_names)
    shape = (num_classes * per_class, 3, *IMAGE_SIZE) if chw else (num_classes * per_class, *IMAGE_SIZE, 3)
    images = np.empty(shape, dtype=np.float32)
    labels = np.empty(num_classes * per_class, dtype=np.int64)

    i = 0
    for label_idx in range(num_classes):
        if chw:
            base = rng.uniform(-1.5, 1.5, size=3).astype(np.float32)
        else:
            base = rng.uniform(30, 220, size=3)
        for _ in range(per_class):
            if chw:
                noise = rng.normal(0, 0.3, size=(3, *IMAGE_SIZE)).astype(np.float32)
                img = base[:, None, None] + noise
            else:
                noise = rng.normal(0, 25, size=(*IMAGE_SIZE, 3))
                img = np.clip(base[None, None, :] + noise, 0, 255).astype(np.float32)
            images[i] = img
            labels[i] = label_idx
            i += 1
    return images, labels


def _dirichlet_non_iid_split(
    labels: np.ndarray, num_clients: int, alpha: float, seed: int
) -> List[np.ndarray]:
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


def _load_shard(
    *,
    client_id: int,
    num_clients: int,
    class_names: List[str],
    data_dir: Optional[str],
    data_dir_env: str,
    seed: int,
    alpha: float,
    val_fraction: float,
    load_image_fn,
    chw: bool,
) -> Shard:
    if client_id < 0 or client_id >= num_clients:
        raise ValueError(f"client_id {client_id} out of range for num_clients={num_clients}")

    data_dir = data_dir or os.environ.get(data_dir_env)

    if data_dir and os.path.isdir(data_dir):
        paths, label_list = _index_real_directory(data_dir, class_names)
        if not paths:
            raise FileNotFoundError(
                f"{data_dir_env}={data_dir} exists but no class subfolders matching {class_names} were found."
            )
        labels = np.array(label_list, dtype=np.int64)
        client_splits = _dirichlet_non_iid_split(labels, num_clients, alpha, seed)
        my_indices = client_splits[client_id]
        images = np.stack([load_image_fn(paths[i]) for i in my_indices])
        client_labels = labels[my_indices]
    else:
        rng = np.random.default_rng(seed)
        all_images, all_labels = _make_synthetic_index(rng, class_names, DEFAULT_SYNTHETIC_PER_CLASS, chw)
        client_splits = _dirichlet_non_iid_split(all_labels, num_clients, alpha, seed)
        my_indices = client_splits[client_id]
        images = all_images[my_indices]
        client_labels = all_labels[my_indices]

    if len(client_labels) < 2:
        raise RuntimeError(
            f"Client {client_id} received only {len(client_labels)} sample(s) after the non-IID "
            "split -- raise `alpha` (less skew) or DEFAULT_SYNTHETIC_PER_CLASS."
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
        class_names=class_names,
    )


def load_dermatology_shard(
    client_id: int,
    num_clients: int = 2,
    data_dir: Optional[str] = None,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
    val_fraction: float = 0.2,
) -> Shard:
    return _load_shard(
        client_id=client_id,
        num_clients=num_clients,
        class_names=DERMATOLOGY_CLASS_NAMES,
        data_dir=data_dir,
        data_dir_env="DERMNET_DATA_DIR",
        seed=seed,
        alpha=alpha,
        val_fraction=val_fraction,
        load_image_fn=_load_image_raw,
        chw=False,
    )


def load_ocular_shard(
    client_id: int,
    num_clients: int = 2,
    data_dir: Optional[str] = None,
    alpha: float = DEFAULT_ALPHA,
    seed: int = OCULAR_DEFAULT_SEED,
    val_fraction: float = 0.2,
) -> Shard:
    return _load_shard(
        client_id=client_id,
        num_clients=num_clients,
        class_names=OCULAR_CLASS_NAMES,
        data_dir=data_dir,
        data_dir_env="OCULAR_DATA_DIR",
        seed=seed,
        alpha=alpha,
        val_fraction=val_fraction,
        load_image_fn=_load_image_normalized,
        chw=True,
    )
