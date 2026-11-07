"""
Re-exports an FL-aggregated Keras checkpoint (fl/server.py's
fl/logs/<model>_fedavg_<timestamp>.keras) to ONNX.

This step is NOT automatic. ONNX Runtime is inference-only -- it can't run
the FedAvg training loop -- so fl/ trains in Keras/TF and produces a .keras
checkpoint each round. Nothing on the inference side (src/inference/, this
Streamlit app) watches that checkpoint or reloads automatically: the ONNX
file in models/ is what actually gets served, and it stays exactly as it was
until someone runs this script and swaps it in. Skipping this step means
inference silently keeps using stale, pre-FL weights -- no error, no warning,
just an ONNX file that never changed.

Usage:
    python -m fl.export_to_onnx fl/logs/dermatology_fedavg_20260710_143000.keras
    python -m fl.export_to_onnx fl/logs/dermatology_fedavg_20260710_143000.keras \\
        -o models/dermnet_v2s_model.onnx

The output is NOT written over models/dermnet_model.onnx by default. As of
the 2026-07 hardening pass, that file is now the same architecture the FL
package trains -- EfficientNetV2-S, 11 classes, raw [0, 255] preprocessing --
so a freshly-exported checkpoint from this script is architecture-compatible
with what's deployed. It's still not swapped in automatically, though:
wiring the exported file into src/inference/dermatology_infer.py (updating
MODEL_PATH and CLASS_NAMES_PATH) remains a separate, deliberate manual step,
not something this script does for you -- see docs/deployment.md step 4.
"""
from __future__ import annotations

import argparse
import os

import tensorflow as tf
import tf2onnx


def export(keras_path: str, onnx_path: str) -> None:
    model = tf.keras.models.load_model(keras_path)
    input_shape = model.inputs[0].shape  # (None, 224, 224, 3) for EfficientNetV2-S here
    spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)
    tf2onnx.convert.from_keras(model, input_signature=spec, output_path=onnx_path)
    print(f"[export_to_onnx] wrote {onnx_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an FL-trained Keras checkpoint to ONNX")
    parser.add_argument("keras_path", help="e.g. fl/logs/dermatology_fedavg_20260710_143000.keras")
    parser.add_argument("-o", "--output", default=None, help="defaults to <keras_path> with a .onnx extension")
    args = parser.parse_args()

    onnx_path = args.output or os.path.splitext(args.keras_path)[0] + ".onnx"
    export(args.keras_path, onnx_path)


if __name__ == "__main__":
    main()
