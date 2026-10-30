"""Golden-output tests on a real face: catch any refactor that changes model results."""

import numpy as np
import pytest

from src.inference import load_model
from tests.conftest import MODELS_DIR

MODEL_DIR = MODELS_DIR / "best/98.20"

# Recorded logits [real, spoof] for FACE_IMAGE before the refactor.
EXPECTED_FP32 = np.array([4.40475, -4.40492])
EXPECTED_INT8 = np.array([4.48281, -4.48115])


@pytest.mark.parametrize(
    ("model_file", "expected"),
    [("best_model.onnx", EXPECTED_FP32), ("best_model_quantized.onnx", EXPECTED_INT8)],
)
def test_onnx_golden(face_batch, model_file, expected):
    session, input_name = load_model(str(MODEL_DIR / model_file))
    logits = session.run(None, {input_name: face_batch})[0][0]
    np.testing.assert_allclose(logits, expected, atol=1e-3)


def test_torch_matches_onnx(face_batch):
    torch = pytest.importorskip("torch")
    from src.minifasv2.model import MultiFTNet, load_weights

    model = MultiFTNet(conv6_kernel=(8, 8))
    load_weights(model, MODEL_DIR / "best_model.pth")
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(face_batch)).numpy()[0]
    np.testing.assert_allclose(logits, EXPECTED_FP32, atol=1e-3)
