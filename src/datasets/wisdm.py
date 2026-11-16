"""WISDM Dataset / DataLoaders. Reads the file produced by data/preprocess.py.

Convention used by ALL models: a batch x has shape [B, 200, 12]
(time-major). Models that need channels-first (Conv1d) permute inside
their own forward().
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

DEFAULT_NPZ = Path(__file__).resolve().parents[2] / "data" / "processed" / "wisdm_windows.npz"


class WISDMDataset(Dataset):
    def __init__(self, npz_path, split: str, augment=None):
        assert split in ("train", "val", "test")
        with np.load(npz_path) as d:
            self.X = torch.from_numpy(d[f"X_{split}"])  # [N, 200, 12] float32
            self.y = torch.from_numpy(d[f"y_{split}"])  # [N] int64
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment is not None:
            x = self.augment(x)
        return x, self.y[i]


def get_dataloaders(npz_path=DEFAULT_NPZ, batch_size=64, augmentation="none",
                    num_workers=0, seed=0):
    """Return {'train','val','test'} loaders. Augmentation only on train.

    Note: the test loader exists for evaluate.py. train.py must NOT use it.
    """
    augment = None
    if augmentation != "none":
        from .augmentation import get_augmentation  # only needed when augmenting
        augment = get_augmentation(augmentation)

    g = torch.Generator()
    g.manual_seed(seed)
    train_ds = WISDMDataset(npz_path, "train", augment)
    val_ds = WISDMDataset(npz_path, "val")
    test_ds = WISDMDataset(npz_path, "test")
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, generator=g),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }


def class_weights(npz_path=DEFAULT_NPZ, num_classes=18) -> torch.Tensor:
    """Inverse-frequency weights from the TRAIN labels (for weighted loss)."""
    with np.load(npz_path) as d:
        y = d["y_train"]
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    w = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


def load_meta(npz_path=DEFAULT_NPZ) -> dict:
    """meta.json written next to the npz by preprocess.py (label names, split, ...)."""
    return json.loads((Path(npz_path).parent / "meta.json").read_text(encoding="utf-8"))