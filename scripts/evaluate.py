"""Evaluate an ONNX model on the prepared test split; writes metrics.json.

Same schema as assets/results/metrics/*/metrics.json. Class 0 = real, 1 = spoof;
the prediction is argmax (real iff real_logit >= spoof_logit).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from src.inference import infer, load_model
from src.minifasv2.config import DATA_PATH, SPOOF_CATEGORIES
from src.minifasv2.data import load_image, load_labels


def evaluate(model_path, data_dir, input_size=128, batch_size=256):
    labels = load_labels(
        Path(data_dir) / "metas/labels/test_label.json", data_dir, SPOOF_CATEGORIES, "test"
    )
    if labels.empty:
        raise SystemExit(f"No test images found under {data_dir}/test")
    session, input_name = load_model(model_path)

    logits = []
    for start in range(0, len(labels), batch_size):
        paths = labels["path"].iloc[start : start + batch_size]
        crops = [load_image(str(Path(data_dir) / "test" / p)) for p in paths]
        logits.extend(infer(crops, session, input_name, input_size))
    logits = np.array(logits)

    y_true = labels["spoof_type"].to_numpy()
    y_pred = logits.argmax(axis=1)
    # Softmax probability of "real", scored with real as the positive class.
    real_score = 1 / (1 + np.exp(logits[:, 1] - logits[:, 0]))
    is_real = y_true == 0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "accuracy": {
            "overall": round(100 * float(np.mean(y_pred == y_true)), 2),
            "real": round(100 * cm[0, 0] / max(cm[0].sum(), 1), 2),
            "spoof": round(100 * cm[1, 1] / max(cm[1].sum(), 1), 2),
        },
        "roc_auc": round(float(roc_auc_score(is_real, real_score)), 4),
        "average_precision": round(float(average_precision_score(is_real, real_score)), 4),
        "validation_samples": len(labels),
        "confusion_matrix": cm.tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Path to .onnx model")
    parser.add_argument(
        "--crop_dir",
        default="data",
        help="Prepared dataset dir under $DATA_PATH, with test/ and metas/labels/",
    )
    parser.add_argument("--input_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output", type=Path, help="Write metrics JSON here (default: print)")
    args = parser.parse_args()

    metrics = evaluate(
        args.model, Path(DATA_PATH) / args.crop_dir, args.input_size, args.batch_size
    )
    text = json.dumps(metrics, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
