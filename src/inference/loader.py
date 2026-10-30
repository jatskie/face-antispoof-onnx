"""ONNX model loader with provider auto-detection."""

import onnxruntime as ort

PREFERRED_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def load_model(model_path: str) -> tuple[ort.InferenceSession, str]:
    """Load ONNX model. Return (session, input_name). Raises if the model can't be loaded."""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    available = ort.get_available_providers()
    providers = [p for p in PREFERRED_PROVIDERS if p in available] or available

    session = ort.InferenceSession(str(model_path), sess_options=sess_options, providers=providers)
    return session, session.get_inputs()[0].name
