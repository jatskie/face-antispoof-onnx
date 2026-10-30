"""Real-time face anti-spoofing demo (webcam or image)."""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src.detection import detect, load_detector
from src.inference import crop, infer, load_model, process_with_logits

MODELS_DIR = Path(__file__).parent / "models"
GREEN, RED, CYAN = (0, 255, 0), (0, 0, 255), (255, 255, 0)


def annotate(frame, detector, session, input_name, logit_threshold, args):
    """Detect faces in a BGR frame, classify each, and draw labeled boxes in place."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = [
        tuple(int(face["bbox"][k]) for k in ("x", "y", "width", "height"))
        for face in detect(frame_rgb, detector, margin=args.margin)
    ]
    crops = [
        crop(frame_rgb, (x, y, x + w, y + h), args.bbox_expansion_factor)
        for x, y, w, h in boxes
    ]
    for (x, y, w, h), logits in zip(
        boxes, infer(crops, session, input_name, args.model_img_size)
    ):
        result = process_with_logits(logits, logit_threshold)
        color = GREEN if result["is_real"] else RED
        label = f"{result['status'].upper()}: {result['logit_diff']:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def run_camera(args, annotate_frame, provider):
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Controls: 'q' quit, 'i' toggle info")
    show_info = True
    frame_times = deque(maxlen=30)
    while True:
        start = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        annotate_frame(frame)
        frame_times.append(time.perf_counter() - start)

        if show_info:
            fps = len(frame_times) / sum(frame_times)
            for i, text in enumerate([f"FPS: {fps:.1f}", f"Provider: {provider}"]):
                cv2.putText(
                    frame, text, (5, 25 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1
                )

        cv2.imshow("Liveness Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("i"):
            show_info = not show_info

    cap.release()
    cv2.destroyAllWindows()


def run_image(args, annotate_frame):
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not load image: {args.image}")
    annotate_frame(image)
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="Path to image file (default: use camera)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--model_img_size", type=int, default=128)
    parser.add_argument("--bbox_expansion_factor", type=float, default=1.5)
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Real-probability threshold (0-1)"
    )
    parser.add_argument("--margin", type=int, default=5, help="Min px from frame edge")
    parser.add_argument(
        "--detector_model", default=str(MODELS_DIR / "detector_quantized.onnx")
    )
    parser.add_argument(
        "--liveness_model", default=str(MODELS_DIR / "best/98.20/best_model_quantized.onnx")
    )
    args = parser.parse_args()

    p = np.clip(args.threshold, 1e-6, 1 - 1e-6)
    logit_threshold = np.log(p / (1 - p))

    detector = load_detector(args.detector_model, (320, 320))
    session, input_name = load_model(args.liveness_model)

    def annotate_frame(frame):
        annotate(frame, detector, session, input_name, logit_threshold, args)

    if args.image:
        run_image(args, annotate_frame)
    else:
        run_camera(args, annotate_frame, session.get_providers()[0])


if __name__ == "__main__":
    main()
