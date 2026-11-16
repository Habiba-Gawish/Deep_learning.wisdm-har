"""Time-series augmentations for the data-strategy study.

Each augmentation takes ONE window x of shape [200, 12] (already z-scored)
and returns a tensor of the same shape. They are applied to the TRAIN set only.

Select with `augmentation:` in the config:
    none | jitter | scaling | jitter_scaling
"""
import torch


def jitter(x: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    """Add Gaussian noise (sigma is in units of std, since data is z-scored)."""
    return x + sigma * torch.randn_like(x)


def scaling(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """Multiply each channel by a random factor ~ N(1, sigma)."""
    factor = 1.0 + sigma * torch.randn(1, x.shape[1])
    return x * factor


def jitter_scaling(x: torch.Tensor) -> torch.Tensor:
    return jitter(scaling(x))


_AUGMENTATIONS = {
    "none": None,
    "jitter": jitter,
    "scaling": scaling,
    "jitter_scaling": jitter_scaling,
}


def get_augmentation(name: str):
    if name not in _AUGMENTATIONS:
        raise ValueError(f"Unknown augmentation '{name}'. Options: {list(_AUGMENTATIONS)}")
    return _AUGMENTATIONS[name]