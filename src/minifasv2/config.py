"""Training configuration."""

import os

import torch

DATA_PATH = os.environ.get("DATA_PATH", "./data")
SPOOF_CATEGORIES = ((0,), (1, 2, 3, 7, 8, 9))  # CelebA-Spoof: live vs print/replay


def get_kernel(height, width):
    """conv_6_dw kernel: the feature-map size after the backbone's 16x downsampling."""
    return ((height + 15) // 16, (width + 15) // 16)


class TrainConfig:
    def __init__(
        self,
        input_size=128,
        batch_size=256,
        spoof_categories=SPOOF_CATEGORIES,
        crop_dir="data",
        output_dir="./output",
    ):
        self.lr = 1e-1
        self.milestones = [10, 15, 22, 30]
        self.gamma = 0.1
        self.epochs = 50
        self.momentum = 0.9
        self.batch_size = batch_size
        self.valid_size = 0.2
        self.output_dir = output_dir

        self.input_size = input_size
        self.data_dir = os.path.join(DATA_PATH, crop_dir)
        self.train_path = os.path.join(self.data_dir, "train")
        self.labels_path = os.path.join(self.data_dir, "metas", "labels", "train_label.json")
        self.spoof_categories = spoof_categories

        self.num_classes = len(spoof_categories)
        self.kernel_size = get_kernel(input_size, input_size)
        self.fourier_size = tuple(2 * s for s in self.kernel_size)

        self.log_loss_per_epoch = 10

    def set_job(self, name, device_id=0):
        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        self.model_path = os.path.join(self.output_dir, name)
        self.log_path = os.path.join(self.model_path, "logs")
        os.makedirs(self.log_path, exist_ok=True)
        print(f"Output directory: {self.model_path}")
