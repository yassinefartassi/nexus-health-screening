"""ONNX Runtime inference wrapper for the dermatology (skin disease) model.

11 curated DermNet classes (10 non-cancerous conditions + "normal"),
EfficientNetV2-S backbone, ~61.6% test accuracy (dermatho-assets/classification_report.txt).
Model input: NHWC float32, 224x224 (auto-detected from the ONNX graph). The
EfficientNetV2-S Rescaling layer is baked into the ONNX graph itself, so this
wrapper must feed raw [0, 255] pixels (PREPROCESS_MODE = "raw255") -- do NOT
apply ImageNet normalization here or the input gets normalized twice.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import List

from PIL import Image

from src.preprocessing.image_utils import preprocess_image
from src.utils.onnx_runner import describe_model, load_session, run_classifier

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "dermnet_model.onnx")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "dermnet_class_names.pkl")
PREPROCESS_MODE = "raw255"


@dataclass
class Prediction:
    label: str
    confidence: float


class DermatologyClassifier:
    def __init__(self, model_path: str = MODEL_PATH, class_names_path: str = CLASS_NAMES_PATH):
        self.session = load_session(model_path)
        self.info = describe_model(self.session)
        with open(class_names_path, "rb") as f:
            self.class_names: List[str] = pickle.load(f)

    def predict(self, image: Image.Image, top_k: int = 5) -> List[Prediction]:
        tensor = preprocess_image(
            image, self.info.height, self.info.width, self.info.channels_first, PREPROCESS_MODE
        )
        probs = run_classifier(self.session, self.info, tensor)
        n = min(len(probs), len(self.class_names))
        order = probs[:n].argsort()[::-1][:top_k]
        return [Prediction(label=self.class_names[i], confidence=float(probs[i])) for i in order]


_classifier: DermatologyClassifier | None = None


def get_classifier() -> DermatologyClassifier:
    global _classifier
    if _classifier is None:
        _classifier = DermatologyClassifier()
    return _classifier


def predict_skin_condition(image: Image.Image, top_k: int = 5) -> List[Prediction]:
    return get_classifier().predict(image, top_k=top_k)
