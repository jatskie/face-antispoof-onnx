"""Headless check of the demo's per-frame pipeline."""

from argparse import Namespace

import cv2
import numpy as np

from demo import GREEN, annotate
from tests.conftest import FACE_IMAGE


def test_annotate_marks_real_face_green(face_detector, liveness_session):
    frame = cv2.imread(str(FACE_IMAGE))
    args = Namespace(margin=5, bbox_expansion_factor=1.5, model_img_size=128)
    annotate(frame, face_detector, *liveness_session, logit_threshold=0.0, args=args)
    assert np.all(frame == GREEN, axis=-1).sum() > 100
