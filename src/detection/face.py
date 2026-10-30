"""Face detection."""

import cv2
import numpy as np
from typing import List, Dict
from pathlib import Path


def load_detector(
    model_path: str,
    input_size: tuple,
    confidence_threshold: float = 0.8,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
):
    """Load YuNet face detector. Raises if the model can't be loaded."""
    if not Path(model_path).exists():
        raise FileNotFoundError(model_path)
    return cv2.FaceDetectorYN.create(
        str(model_path), "", input_size, confidence_threshold, nms_threshold, top_k
    )


def detect(
    image: np.ndarray, detector, min_face_size: int = 60, margin: int = 5
) -> List[Dict]:
    """Detect faces. Filter by min size and edge margin. Return list of {bbox, confidence}."""
    img_h, img_w = image.shape[:2]
    detector.setInputSize((img_w, img_h))
    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        return []

    detections = []
    for face in faces:
        x, y, w, h = face[:4].astype(int)
        conf = float(face[14])

        dist_left = x
        dist_right = img_w - (x + w)
        dist_top = y
        dist_bottom = img_h - (y + h)
        if min(dist_left, dist_right, dist_top, dist_bottom) < margin:
            continue

        if w >= min_face_size and h >= min_face_size:
            detections.append(
                {
                    "bbox": {
                        "x": float(x),
                        "y": float(y),
                        "width": float(w),
                        "height": float(h),
                    },
                    "confidence": conf,
                }
            )

    return detections
