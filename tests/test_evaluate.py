"""Evaluation script on a tiny prepared test split."""

import json

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")  # data loading lives in the training package

from scripts.evaluate import evaluate  # noqa: E402
from tests.conftest import MODELS_DIR  # noqa: E402


def test_evaluate(tmp_path, face_batch):
    (tmp_path / "test").mkdir()
    # face_batch is the preprocessed real face (1, 3, 128, 128) in [0, 1] RGB.
    face = (face_batch[0].transpose(1, 2, 0) * 255).round().astype(np.uint8)
    cv2.imwrite(str(tmp_path / "test/real.png"), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
    rng = np.random.default_rng(0)
    for i in range(3):
        noise = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"test/noise{i}.png"), noise)

    labels = {f"Data/test/noise{i}.png": [0] * 40 + [2, 0, 0, 0] for i in range(3)}
    labels["Data/test/real.png"] = [0] * 44
    (tmp_path / "metas/labels").mkdir(parents=True)
    (tmp_path / "metas/labels/test_label.json").write_text(json.dumps(labels))

    metrics = evaluate(MODELS_DIR / "best/98.20/best_model_quantized.onnx", tmp_path)

    assert metrics["validation_samples"] == 4
    assert np.array(metrics["confusion_matrix"]).sum() == 4
    assert metrics["confusion_matrix"][0] == [1, 0]  # the real face is classified real
    assert set(metrics) == {
        "accuracy",
        "roc_auc",
        "average_precision",
        "validation_samples",
        "confusion_matrix",
    }
