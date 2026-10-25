"""Unlabelled AI01 Z-stacks. No segmentation masks or label metadata are read."""

from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


# A1_Refractor averages the three RGB processor statistics into one channel.
DEFAULT_PREPROCESSING = {
    "image_size": 256,
    "num_channels": 11,
    "channel_order": list(range(11)),
    "crop": "center",
    "rescale_factor": 1.0 / 255.0,
    "mean": float(np.mean([0.485, 0.456, 0.406])),
    "std": float(np.mean([0.229, 0.224, 0.225])),
}


def read_manifest(path):
    """One .npy path per line, relative to the manifest or absolute; # comments."""
    path = Path(path).resolve()
    files = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        item = Path(line)
        files.append((path.parent / item).resolve())
    return files


def load_zstack(path, preprocessing=None, array_layout="auto"):
    """Return normalized CHW float32. Input intensities must be raw [0, 255]."""
    spec = dict(DEFAULT_PREPROCESSING if preprocessing is None else preprocessing)
    image = np.load(path, allow_pickle=False, mmap_mode="r")
    channels = spec["num_channels"]
    if array_layout == "auto":
        if image.ndim == 4 and image.shape[0] == channels and image.shape[-1] == 1:
            array_layout = "zhwc"
        elif image.ndim == 3 and image.shape[0] == channels:
            array_layout = "zhw"
        elif image.ndim == 3 and image.shape[-1] == channels:
            array_layout = "hwz"
        else:
            raise ValueError(f"{path}: expected 11-channel Z-stack, got {image.shape}")
    if array_layout == "zhwc" and image.ndim == 4 and image.shape[-1] == 1:
        image = image[..., 0]
    elif array_layout == "hwz" and image.ndim == 3:
        image = image.transpose(2, 0, 1)
    elif array_layout != "zhw":
        raise ValueError(f"{path}: invalid layout {array_layout} for {image.shape}")
    if image.ndim != 3 or image.shape[0] != channels:
        raise ValueError(f"{path}: expected [{channels}, H, W], got {image.shape}")
    size = spec["image_size"]
    height, width = image.shape[-2:]
    if min(height, width) < size:
        raise ValueError(f"{path}: image {height}x{width} is smaller than crop {size}")
    top, left = (height - size) // 2, (width - size) // 2
    image = np.array(image[:, top:top + size, left:left + size], dtype=np.float32, copy=True)
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 255:
        raise ValueError(f"{path}: expected finite raw intensities in [0, 255]")
    if spec["std"] <= 0:
        raise ValueError("Normalization std must be positive")
    image = (image * spec["rescale_factor"] - spec["mean"]) / spec["std"]
    return torch.from_numpy(image)


class AI01UnlabelledDataset(Dataset):
    def __init__(self, data_path=None, manifest=None, preprocessing=None,
                 array_layout="auto", augment=True):
        if (data_path is None) == (manifest is None):
            raise ValueError("Provide exactly one of data_path or manifest")
        if manifest is not None:
            files = read_manifest(manifest)
        else:
            root = Path(data_path)
            if not root.is_dir():
                raise ValueError(f"Not a data directory: {root}")
            files = list(root.rglob("*.npy"))
        self.files = sorted(set(Path(file).resolve() for file in files))
        if not self.files:
            raise ValueError("No .npy images found")
        for file in self.files:
            if file.suffix.lower() != ".npy" or not file.is_file():
                raise ValueError(f"Missing or invalid .npy image: {file}")
        self.preprocessing = dict(DEFAULT_PREPROCESSING if preprocessing is None else preprocessing)
        self.array_layout = array_layout
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = load_zstack(self.files[index], self.preprocessing, self.array_layout)
        # Transform the whole stack together, never shuffle Z or augment Z independently.
        if self.augment:
            if random.random() < 0.5:
                image = image.flip(-1)
            if random.random() < 0.5:
                image = image.flip(-2)
            image = torch.rot90(image, random.randrange(4), (-2, -1))
        return image.contiguous()

    def well_balanced_sampler(self, generator=None):
        """Assumes one parent directory per well, as in prepared_image_path/frame.npy."""
        from collections import Counter

        counts = Counter(file.parent for file in self.files)
        weights = [1.0 / counts[file.parent] for file in self.files]
        return WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)


def ensure_disjoint_wells(train_dataset, validation_dataset):
    overlap = {file.parent for file in train_dataset.files} & {
        file.parent for file in validation_dataset.files
    }
    if overlap:
        raise ValueError(f"Train/validation share well directories: {sorted(map(str, overlap))[:5]}")
