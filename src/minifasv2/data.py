"""Training data: CelebA-Spoof-style labels, augmentation, and Fourier targets."""

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
LABEL_COLUMN = 40  # CelebA-Spoof: index of the spoof-type code in each label row


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def generate_FT(image):
    """Normalized log-magnitude spectrum: the FT branch's training target."""
    # NOTE: `image` is RGB, so BGR2GRAY swaps the R/B weights. The released models
    # were trained with this target; keep it unless you intend to retrain.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fimg = np.log(np.abs(np.fft.fftshift(np.fft.fft2(image))) + 1)
    lo, hi = fimg.min(), fimg.max()
    return (fimg - lo + 1) / (hi - lo + 1)


class LivenessDataset(Dataset):
    """Yields (image, label), plus the image's Fourier target when fourier_size is set."""

    def __init__(self, root, labels, transform, fourier_size=None):
        self.root = root
        self.labels = labels
        self.transform = transform
        self.fourier_size = fourier_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path, target = self.labels.iloc[idx]
        image = load_image(os.path.join(self.root, path))
        sample = self.transform(image)
        if self.fourier_size is None:
            return sample, target

        # Computed on the un-augmented image, as the released models were trained.
        fourier = cv2.resize(generate_FT(image), self.fourier_size)
        return sample, torch.from_numpy(fourier).float().unsqueeze(0), target


class RandomRotationWithReflect:
    """Random rotation that fills corners by reflection instead of black borders."""

    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, img):
        angle = T.RandomRotation.get_params([-self.degrees, self.degrees])
        img = np.array(img, dtype=np.uint8)
        h, w = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(
            img,
            matrix,
            (w, h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return Image.fromarray(rotated, "RGB")


def load_labels(json_path, data_dir, spoof_categories, split="train"):
    """Return a DataFrame of (path, class index) for images present on disk.

    Rows whose spoof type is in spoof_categories[i] get class i; others are dropped.
    """
    labels = pd.read_json(json_path, orient="index")[LABEL_COLUMN]
    class_of = {t: i for i, category in enumerate(spoof_categories) for t in category}
    labels = labels[labels.isin(class_of)]

    split_dir = Path(data_dir) / split
    on_disk = {
        p.relative_to(split_dir).as_posix()
        for p in split_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    }
    paths = labels.index.str.replace("Data/", "", n=1).str.removeprefix(f"{split}/")
    df = pd.DataFrame({"path": paths, "spoof_type": labels.map(class_of).to_numpy()})
    return df[df["path"].isin(on_disk)].reset_index(drop=True)


def get_train_valid(config):
    train_transform = T.Compose(
        [
            T.ToPILImage(),
            T.RandomResizedCrop(size=(config.input_size, config.input_size), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            RandomRotationWithReflect(90),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]
    )
    valid_transform = T.Compose([T.ToPILImage(), T.ToTensor()])

    labels = load_labels(config.labels_path, config.data_dir, config.spoof_categories)
    train_labels, valid_labels = train_test_split(
        labels, test_size=config.valid_size, random_state=20220826
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        LivenessDataset(
            config.train_path,
            train_labels,
            train_transform,
            fourier_size=config.fourier_size,
        ),
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        LivenessDataset(config.train_path, valid_labels, valid_transform),
        batch_size=config.batch_size,
        pin_memory=pin_memory,
    )
    return train_loader, valid_loader
