"""Public API for Haralick texture feature extraction.

Provides a simple functional interface on top of the HaralickEngine.

Example:
    >>> import torch
    >>> from haralickt import haralick
    >>> image = torch.randint(0, 32, (256, 256))
    >>> features = haralick(image, num_levels=32, window_size=32, stride=8)
    >>> print(features.shape)  # torch.Size([29, 29, 13])
"""

from __future__ import annotations

import torch

from .config import HaralickConfig
from .device import resolve_device
from .engine import HaralickEngine


def haralick(
    image: torch.Tensor,
    num_levels: int = 32,
    window_size: int = 32,
    stride: int = 8,
    ignore_zeros: bool = True,
    tile_size: int = 1024,
    device: str | None = None,
    distance: int | None = None,
) -> torch.Tensor:
    """Compute 13 Haralick texture features for a 2D image.

    Args:
        image: 2D tensor of shape (H, W) with integer or float values in [0, num_levels - 1].
        num_levels: Number of gray levels (L). Default: 32.
        window_size: Spatial window size for local GLCM. Default: 32.
        stride: Step between windows. Default: 8.
        ignore_zeros: Exclude pixel pairs involving value 0. Default: True.
        tile_size: Spatial tile size for OOM prevention. Default: 1024.
        device: Compute device ("cuda", "cpu", "mps"). Auto-detected if None.
        distance: GLCM pixel-pair distance. Default: None (auto = max(1, window_size // 8)).

    Returns:
        Tensor of shape (H_out, W_out, 13) where:
            H_out = (H - window_size) // stride + 1
            W_out = (W - window_size) // stride + 1

        Feature channels:
            0  - Angular Second Moment (ASM)
            1  - Contrast
            2  - Correlation
            3  - Variance
            4  - Inverse Difference Moment
            5  - Sum Average
            6  - Sum Variance
            7  - Sum Entropy
            8  - Entropy
            9  - Difference Variance
            10 - Difference Entropy
            11 - Information Measure of Correlation 1
            12 - Information Measure of Correlation 2

    Raises:
        TypeError: If image is not a torch.Tensor.
        ValueError: If image dimensions or values are invalid.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = HaralickConfig(
        num_levels=num_levels,
        window_size=window_size,
        stride=stride,
        ignore_zeros=ignore_zeros,
        tile_size=tile_size,
        device=device,
        distance=distance,
    )

    engine = HaralickEngine(config)
    return engine.compute(image)


def haralick_from_numpy(
    image,
    num_levels: int = 32,
    window_size: int = 32,
    stride: int = 8,
    ignore_zeros: bool = True,
    tile_size: int = 1024,
    device: str | None = None,
    distance: int | None = None,
) -> torch.Tensor:
    """Compute Haralick features from a NumPy array.

    Convenience wrapper that converts NumPy to torch.Tensor automatically.

    Args:
        image: 2D NumPy array of shape (H, W).
        num_levels: Number of gray levels. Default: 32.
        window_size: Spatial window size. Default: 32.
        stride: Step between windows. Default: 8.
        ignore_zeros: Exclude zero pairs. Default: True.
        tile_size: Tile size for OOM prevention. Default: 1024.
        device: Compute device. Auto-detected if None.
        distance: GLCM pixel-pair distance. Default: None (auto).

    Returns:
        Tensor of shape (H_out, W_out, 13).
    """
    import numpy as np

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")

    tensor = torch.from_numpy(image)
    return haralick(
        tensor,
        num_levels=num_levels,
        window_size=window_size,
        stride=stride,
        ignore_zeros=ignore_zeros,
        tile_size=tile_size,
        device=device,
        distance=distance,
    )