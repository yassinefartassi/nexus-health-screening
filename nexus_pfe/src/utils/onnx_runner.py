"""
Shared ONNX Runtime session loader used by every inference module
(ocular_infer.py, dermatology_infer.py, ...).

Input resolution/layout isn't hard-coded: it's read from the ONNX graph
itself at load time via `session.get_inputs()`, so the same loader works
for both the ocular model (NCHW) and the dermatology model (NHWC) without
per-model branching in calling code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime as ort


@dataclass
class OnnxModelInfo:
    input_name: str
    output_name: str
    input_shape: list
    height: int
    width: int
    channels_first: bool  # True: NCHW, False: NHWC


def load_session(model_path: str) -> ort.InferenceSession:
    """Load an ONNX model, raising a clear error if the file is missing."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Make sure the .onnx (and matching .onnx.data, if any) file is in models/."
        )
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def resolve_input_layout(shape: list, default: int = 224) -> Tuple[int, int, bool]:
    """Infer (height, width, channels_first) from an ONNX input shape,
    falling back to default x default NCHW if dimensions are dynamic/unreadable."""
    dims = [d if isinstance(d, int) and d > 0 else None for d in shape]
    if len(dims) == 4:
        # NCHW: [N, 3, H, W]
        if dims[1] in (1, 3):
            return dims[2] or default, dims[3] or default, True
        # NHWC: [N, H, W, 3]
        if dims[3] in (1, 3):
            return dims[1] or default, dims[2] or default, False
    return default, default, True


def describe_model(session: ort.InferenceSession) -> OnnxModelInfo:
    model_input = session.get_inputs()[0]
    height, width, channels_first = resolve_input_layout(model_input.shape)
    return OnnxModelInfo(
        input_name=model_input.name,
        output_name=session.get_outputs()[0].name,
        input_shape=model_input.shape,
        height=height,
        width=width,
        channels_first=channels_first,
    )


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def run_classifier(
    session: ort.InferenceSession,
    info: OnnxModelInfo,
    input_tensor: np.ndarray,
) -> np.ndarray:
    """Run the model and return a 1D probability vector (softmax applied if
    the raw output doesn't already look like a probability distribution)."""
    raw = session.run([info.output_name], {info.input_name: input_tensor})[0]
    logits = np.squeeze(raw).flatten().astype(np.float32)
    if 0.99 <= logits.sum() <= 1.01 and logits.min() >= 0:
        return logits
    return softmax(logits)
