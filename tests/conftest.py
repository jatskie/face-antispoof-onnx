"""Pytest fixtures for face-antispoof tests."""

from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
FACE_IMAGE = ROOT / "assets/docs/face.jpg"
DETECTOR_MODEL = MODELS_DIR / "detector_quantized.onnx"
LIVENESS_MODEL = MODELS_DIR / "best/98.20/best_model_quantized.onnx"


@pytest.fixture(scope="session")
def liveness_session():
    """Load liveness model once for all tests."""
    from src.inference import load_model

    return load_model(str(LIVENESS_MODEL))


@pytest.fixture(scope="session")
def face_detector():
    """Load face detector once for all tests."""
    from src.detection import load_detector

    return load_detector(str(DETECTOR_MODEL), (640, 480))


@pytest.fixture
def dummy_face_crop():
    """Create a dummy 128x128 RGB image."""
    return np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)


@pytest.fixture
def dummy_frame():
    """Create a dummy 480x640 RGB frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def face_batch(face_detector):
    """The single real face in FACE_IMAGE, cropped and preprocessed: (1, 3, 128, 128)."""
    from src.detection import detect
    from src.inference import crop, preprocess_batch

    img = cv2.cvtColor(cv2.imread(str(FACE_IMAGE)), cv2.COLOR_BGR2RGB)
    faces = detect(img, face_detector)
    assert len(faces) == 1
    b = faces[0]["bbox"]
    face = crop(img, (b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]), 1.5)
    return preprocess_batch([face], 128)
