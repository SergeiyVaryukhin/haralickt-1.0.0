"""haralickt - GPU-accelerated Haralick texture features via PyTorch.

Provides 13 standard Haralick texture features computed from
Gray-Level Co-occurrence Matrices (GLCM) with 4-direction averaging.

Quick start:
    >>> import torch
    >>> from haralickt import haralick
    >>> image = torch.randint(0, 32, (256, 256))
    >>> features = haralick(image, num_levels=32, window_size=32, stride=8)
    >>> print(features.shape)  # torch.Size([29, 29, 13])
"""

from .api import haralick, haralick_from_numpy
from .config import HaralickConfig
from . import features

from .device import get_device_info, resolve_device
from .engine import HaralickEngine

__version__ = "1.0.0"

__all__ = [
    "haralick",
    "features",
    "haralick_from_numpy",
    "HaralickConfig",
    "HaralickEngine",
    "resolve_device",
    "get_device_info",
    "__version__",
]