"""Resize/normalize helpers shared by the ocular and dermatology ONNX models."""
from __future__ import annotations

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

PreprocessMode = str  # "imagenet" | "0-1" | "-1-1" | "raw255"


def preprocess_image(
    image: Image.Image,
    height: int,
    width: int,
    channels_first: bool,
    mode: PreprocessMode = "imagenet",
) -> np.ndarray:
    """Resize + normalize a PIL image into a single-sample NCHW/NHWC float32 tensor."""
    image = image.convert("RGB").resize((width, height), Image.BILINEAR)

    if mode == "raw255":
        # No rescale, no normalization: the model's own graph (e.g. EfficientNetV2's
        # baked-in Rescaling layer) expects raw [0, 255] pixel values. Do NOT run
        # this through the /255 branch below or the model gets normalized twice.
        arr = np.asarray(image).astype(np.float32)  # HWC, [0, 255]
        if channels_first:
            arr = np.transpose(arr, (2, 0, 1))  # CHW
        return np.expand_dims(arr, axis=0).astype(np.float32)

    arr = np.asarray(image).astype(np.float32) / 255.0  # HWC, [0, 1]

    if mode == "imagenet":
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    elif mode == "-1-1":
        arr = (arr - 0.5) / 0.5
    # "0-1" -> leave as is

    if channels_first:
        arr = np.transpose(arr, (2, 0, 1))  # CHW

    return np.expand_dims(arr, axis=0).astype(np.float32)
