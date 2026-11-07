"""ONNX Runtime inference wrapper for the ocular (eye disease) model.

5 classes: Cataract, Conjunctivitis, Eyelid, Normal, Uveitis.
Model input: NCHW float32, 224x224 (auto-detected from the ONNX graph).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List

from PIL import Image

from src.preprocessing.image_utils import preprocess_image
from src.utils.onnx_runner import describe_model, load_session, run_classifier

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "eye_disease_model.onnx")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "eye_disease_class_names.json")
PREPROCESS_MODE = "imagenet"


@dataclass
class Prediction:
    label: str
    confidence: float


class OcularClassifier:
    def __init__(self, model_path: str = MODEL_PATH, class_names_path: str = CLASS_NAMES_PATH):
        self.session = load_session(model_path)
        self.info = describe_model(self.session)
        with open(class_names_path, "r", encoding="utf-8") as f:
            self.class_names: List[str] = json.load(f)

    def predict(self, image: Image.Image, top_k: int = 5) -> List[Prediction]:
        tensor = preprocess_image(
            image, self.info.height, self.info.width, self.info.channels_first, PREPROCESS_MODE
        )
        probs = run_classifier(self.session, self.info, tensor)
        n = min(len(probs), len(self.class_names))
        order = probs[:n].argsort()[::-1][:top_k]
        return [Prediction(label=self.class_names[i], confidence=float(probs[i])) for i in order]


_classifier: OcularClassifier | None = None


def get_classifier() -> OcularClassifier:
    global _classifier
    if _classifier is None:
        _classifier = OcularClassifier()
    return _classifier


def predict_eye_condition(image: Image.Image, top_k: int = 5) -> List[Prediction]:
    return get_classifier().predict(image, top_k=top_k)
