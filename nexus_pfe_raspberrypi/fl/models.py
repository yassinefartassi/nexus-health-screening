"""
Per-module model builders for federated learning, registered by name so
fl/client.py and fl/server.py never need to change when a new module is
federated -- add a build_fn + load_shard_fn pair and one MODEL_REGISTRY
entry, that's it.

Two frameworks are federated here: dermatology (TF/Keras, ModelSpec.framework
= "keras") and ocular (PyTorch, framework = "torch"). fl/client.py's
NexusFLClient branches on this field since Keras and PyTorch models don't
share a get_weights()/set_weights()/fit()/evaluate() calling convention --
the torch path goes through fl/torch_ops.py instead of calling model methods
directly. This mirrors nexus_pfe_server_fed/fl_server/modules/base.py's
FederatedModule abstraction on the server side.

rPPG is deliberately absent here: it's deterministic CHROM/POS signal
processing (src/inference/rppg_algorithms.py) with no trainable weights, so
there is nothing to federate -- it stays out of fl/ entirely.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import tensorflow as tf

from fl.data_partition import (
    DERMATOLOGY_CLASS_NAMES,
    OCULAR_CLASS_NAMES,
    Shard,
    load_dermatology_shard,
    load_ocular_shard,
)
from fl.torch_ops import build_ocular_model


def build_dermatology_model(num_classes: int) -> tf.keras.Model:
    """EfficientNetV2-S backbone (frozen, ImageNet-pretrained) + a small
    trainable head -- mirrors dermnet_v2s_curated.ipynb's architecture, so
    weights produced by FL rounds here are compatible with that notebook's
    model once exported to ONNX for the Streamlit inference tab.

    Input convention: raw [0, 255] float32 images, NOT ImageNet-normalized.
    EfficientNetV2-S's Keras application bakes its own rescaling/
    normalization in as the model's first layers (see
    fl/data_partition.py's _load_image_raw docstring for why this matters).
    """
    base_model = tf.keras.applications.EfficientNetV2S(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


@dataclass
class ModelSpec:
    build_fn: Callable[[int], object]
    load_shard_fn: Callable[..., Shard]
    class_names: List[str]
    framework: str = "keras"  # "keras" | "torch" -- see module docstring


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "dermatology": ModelSpec(
        build_fn=build_dermatology_model,
        load_shard_fn=load_dermatology_shard,
        class_names=DERMATOLOGY_CLASS_NAMES,
        framework="keras",
    ),
    "ocular": ModelSpec(
        build_fn=build_ocular_model,
        load_shard_fn=load_ocular_shard,
        class_names=OCULAR_CLASS_NAMES,
        framework="torch",
    ),
}
