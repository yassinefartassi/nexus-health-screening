"""
Dermatology's FederatedModule implementation -- TensorFlow/Keras.

Architecture: EfficientNetV2-S backbone + a small trainable head. Must
match, layer-for-layer, both:

  - dermnet_v2s_curated.ipynb, the notebook that produced the centrally
    pretrained checkpoint this server loads as round-0 starting weights
    (see server.py), and
  - nexus_pfe_raspberrypi/fl/models.py's build_dermatology_model(), the
    client-side copy of this same architecture.

FedAvg only works because every participant (this server + each client)
builds the identical layer graph, so get_weights()/set_weights() lists line
up array-for-array by position. If you change this architecture, change the
client's copy to match, or aggregation will silently produce garbage (wrong
array assigned to wrong layer) or crash on a shape mismatch.

Input convention: raw [0, 255] float32 NHWC, NOT ImageNet-normalized --
EfficientNetV2-S's Keras application bakes its own rescaling in as its
first layers. This is a DIFFERENT convention from ocular's (see
modules/ocular.py) -- do not mix them up.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf

from fl_server.config import DERMATOLOGY_CLASS_NAMES_PATH, IMAGE_SIZE
from fl_server.modules.base import FederatedModule
from fl_server.onnx_export import export_keras_to_onnx

with open(DERMATOLOGY_CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    DERMATOLOGY_CLASS_NAMES: List[str] = json.load(f)


class DermatologyModule(FederatedModule):
    name = "dermatology"
    class_names = DERMATOLOGY_CLASS_NAMES
    checkpoint_extension = ".keras"

    def build(self) -> tf.keras.Model:
        # weights=None: this server never trains from an ImageNet-initialized
        # backbone -- every weight array is overwritten via set_weights()
        # immediately after this returns, either from the pretrained
        # checkpoint (round 0) or a client-aggregated update (round > 0).
        # Fetching ImageNet weights here would be wasted network I/O on a
        # machine that only aggregates.
        base_model = tf.keras.applications.EfficientNetV2S(
            include_top=False, weights=None, input_shape=(*IMAGE_SIZE, 3)
        )
        base_model.trainable = False

        inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        outputs = tf.keras.layers.Dense(len(self.class_names), activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        return model

    def load_checkpoint(self, path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def save_checkpoint(self, model: tf.keras.Model, path_stem: Path) -> Path:
        keras_path = Path(path_stem).with_suffix(self.checkpoint_extension)
        model.save(keras_path)
        return keras_path

    def get_weights(self, model: tf.keras.Model) -> List[np.ndarray]:
        return model.get_weights()

    def set_weights(self, model: tf.keras.Model, weights: List[np.ndarray]) -> None:
        model.set_weights(weights)

    def fit(
        self,
        model: tf.keras.Model,
        train_images: np.ndarray,
        train_labels: np.ndarray,
        epochs: int,
        batch_size: int,
    ) -> Dict[str, float]:
        history = model.fit(
            train_images, train_labels, epochs=epochs, batch_size=batch_size, verbose=0
        )
        return {
            "loss": float(history.history["loss"][-1]),
            "accuracy": float(history.history["accuracy"][-1]),
        }

    def evaluate(
        self, model: tf.keras.Model, images: np.ndarray, labels: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        loss, accuracy = model.evaluate(images, labels, verbose=0)
        return float(loss), {"accuracy": float(accuracy)}

    def export_onnx(self, model: tf.keras.Model, onnx_path: Path) -> Path:
        return export_keras_to_onnx(model, onnx_path)
