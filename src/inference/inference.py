"""ONNX inference for face anti-spoofing."""

import numpy as np
import onnxruntime as ort
from typing import List, Dict
from src.inference.preprocess import preprocess_batch


def process_with_logits(raw_logits: np.ndarray, threshold: float) -> Dict:
    """Convert raw logits to real/spoof classification."""
    real_logit = float(raw_logits[0])
    spoof_logit = float(raw_logits[1])
    logit_diff = real_logit - spoof_logit
    is_real = logit_diff >= threshold

    return {
        "is_real": is_real,
        "status": "real" if is_real else "spoof",
        "logit_diff": logit_diff,
        "real_logit": real_logit,
        "spoof_logit": spoof_logit,
        "confidence": abs(logit_diff),
    }


def infer(
    face_crops: List[np.ndarray],
    ort_session: ort.InferenceSession,
    input_name: str,
    model_img_size: int,
) -> List[np.ndarray]:
    """Run batch inference on cropped face images. Return list of logits per face."""
    if not face_crops:
        return []

    batch_input = preprocess_batch(face_crops, model_img_size)
    logits = ort_session.run(None, {input_name: batch_input})[0]
    return list(logits)
