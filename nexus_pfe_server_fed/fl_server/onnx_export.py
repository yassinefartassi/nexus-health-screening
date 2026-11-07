"""
Framework-specific ONNX export helpers, called by each module's
export_onnx() (see modules/dermatology.py, modules/ocular.py) right after
model_registry.save_round() every round. This is NOT optional.

Why this matters: FL rounds update models/<module>/global_v{round}.<ext>
(.keras for dermatology, .pt for ocular). Anything that actually serves
inference (the Streamlit app, mobile app, Raspberry Pi client, see
nexus_pfe/src/inference/) loads a .onnx file, not a native checkpoint. If
this export step were skipped, the FL loop would happily aggregate and
improve weights round after round while every deployed inference path
silently kept using stale ONNX weights -- no error, no warning. Every
round's checkpoint save is paired 1:1 with a .onnx export so that can't
happen, for either framework.

Two entirely separate conversion paths, per the mixed-frameworks design
(see modules/base.py): dermatology goes through tf2onnx (TensorFlow/Keras),
ocular goes through torch.onnx.export (PyTorch). Neither path knows the
other exists -- do not try to unify them beyond sharing this file.
"""
from __future__ import annotations

from pathlib import Path

DEFAULT_OPSET = 13


def export_keras_to_onnx(model, onnx_path: Path, opset: int = DEFAULT_OPSET) -> Path:
    """Dermatology's export path: tf2onnx. `model` is a tf.keras.Model."""
    import tensorflow as tf
    import tf2onnx

    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    input_signature = [tf.TensorSpec(model.input_shape, tf.float32, name="input")]
    tf2onnx.convert.from_keras(
        model, input_signature=input_signature, opset=opset, output_path=str(onnx_path)
    )
    return onnx_path


def export_torch_to_onnx(
    model, onnx_path: Path, input_shape: tuple = (1, 3, 224, 224), opset: int = DEFAULT_OPSET
) -> Path:
    """Ocular's export path: torch.onnx.export. `model` is a torch.nn.Module.
    input_shape is NCHW (PyTorch convention), unlike dermatology's NHWC --
    see modules/ocular.py for why ocular's preprocessing is not baked into
    the model graph the way EfficientNetV2-S's is."""
    import torch

    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.zeros(*input_shape, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        # dynamo=False: torch>=2.6 defaults to the new dynamo-based exporter,
        # which (a) needs the extra `onnxscript` package and (b) prints a
        # unicode checkmark that crashes outright on Windows' default cp1252
        # console encoding (UnicodeEncodeError, not caught anywhere in
        # torch.onnx internals). The legacy TorchScript-based exporter this
        # code was written against has neither problem and is still fully
        # supported. Found by actually running a local ocular FL round
        # end-to-end (2026-07 hardening pass) -- every round's export step
        # crashed the server outright until this was pinned.
        dynamo=False,
    )
    return onnx_path
