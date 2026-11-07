"""
Per-module model builders for federated learning, registered by name so
fl_client/client.py never needs to change when a new module is federated --
add a build_fn + load_shard_fn pair and one MODEL_REGISTRY entry.

Two frameworks are federated here: dermatology (TF/Keras, ModelSpec.framework
= "keras") and ocular (PyTorch, framework = "torch"). fl_client/client.py's
NexusFLClient branches on this field since Keras and PyTorch models don't
share a get_weights()/set_weights()/fit()/evaluate() calling convention --
the torch path goes through fl_client/torch_ops.py. Mirrors
nexus_pfe_server_fed/fl_server/modules/base.py's FederatedModule abstraction
on the server side, and nexus_pfe_raspberrypi/fl/models.py's registry on the
other reference client -- all three must build the identical layer graph per
module or FedAvg's get_weights()/set_weights() lists stop lining up
array-for-array.

rPPG is deliberately absent here: it's deterministic CHROM/POS signal
processing (src/inference/rppg_algorithms.py) with no trainable weights, so
there is nothing to federate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import tensorflow as tf

from fl_client.data_partition import (
    DERMATOLOGY_CLASS_NAMES,
    OCULAR_CLASS_NAMES,
    Shard,
    load_dermatology_shard,
    load_ocular_shard,
)
from fl_client.torch_ops import build_ocular_model


def build_dermatology_model(num_classes: int) -> tf.keras.Model:
    """EfficientNetV2-S backbone + a small trainable head -- must match,
    layer-for-layer, nexus_pfe_server_fed/fl_server/modules/dermatology.py's
    build() so FedAvg weight arrays line up positionally with the server and
    every other client.

    weights="imagenet" here (unlike the server's weights=None) only matters
    for the very first local fit() before this client has received the
    server's round-0 pretrained parameters via set_weights() -- Flower
    always calls set_weights(parameters) before fit()/evaluate(), so
    whatever this starts with is immediately overwritten in practice; kept
    as "imagenet" simply because it's a more sensible default for anyone
    running this file standalone/interactively outside a live FL round.

    Input convention: raw [0, 255] float32 images, NOT ImageNet-normalized
    -- EfficientNetV2-S's Keras application bakes its own rescaling in as
    its first layers (see fl_client/data_partition.py's _load_image_raw).
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
    framework: str = "keras"  # "keras" | "torch"


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
