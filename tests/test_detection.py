"""Tests for detection module."""

import pytest

from src.detection import detect, load_detector


class TestDetector:
    def test_detect_returns_list(self, face_detector, dummy_frame):
        """Detect returns a list."""
        results = detect(dummy_frame, face_detector)
        assert isinstance(results, list)

    def test_missing_model_raises(self):
        with pytest.raises(FileNotFoundError):
            load_detector("does_not_exist.onnx", (320, 320))
