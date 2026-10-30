"""Training smoke test on a tiny synthetic dataset."""

import json

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")

from src.minifasv2.config import TrainConfig  # noqa: E402
from src.minifasv2.main import Trainer  # noqa: E402

NUM_IMAGES = 16


@pytest.fixture
def dataset(tmp_path):
    """CelebA-Spoof layout: train/*.jpg + metas/labels/train_label.json (label at index 40)."""
    rng = np.random.default_rng(0)
    (tmp_path / "train").mkdir()
    labels = {}
    for i in range(NUM_IMAGES):
        image = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "train" / f"{i}.jpg"), image)
        row = [0] * 44
        row[40] = 0 if i % 2 else 2
        labels[f"Data/train/{i}.jpg"] = row
    # Listed but missing on disk: must be filtered out, not crash the loader.
    labels["Data/train/missing.jpg"] = [0] * 44

    (tmp_path / "metas/labels").mkdir(parents=True)
    (tmp_path / "metas/labels/train_label.json").write_text(json.dumps(labels))
    return tmp_path


def make_config(dataset, epochs):
    config = TrainConfig(batch_size=4, crop_dir=str(dataset), output_dir=str(dataset / "out"))
    config.epochs = epochs
    config.set_job("test")
    return config


def test_train_and_resume(dataset):
    trainer = Trainer(make_config(dataset, epochs=1))
    trainer.train_model()
    steps_per_epoch = len(trainer.train_loader)
    assert trainer.step == steps_per_epoch

    config = make_config(dataset, epochs=2)
    resumed = Trainer(config, resume_from=f"{config.model_path}/checkpoint_latest.pth")
    resumed.train_model()
    # Resuming after epoch 1 must run only epoch 2, not repeat epoch 1.
    assert resumed.step == 2 * steps_per_epoch
