"""Input validation for Haralick feature extraction."""

from __future__ import annotations

import torch

from .config import HaralickConfig


def validate_image(image: torch.Tensor, config: HaralickConfig) -> None:
    """Validate input image tensor against configuration.

    Raises:
        TypeError: If image is not a torch.Tensor.
        ValueError: If image dimensions, dtype, or value range are invalid.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(image).__name__}")

    if image.dim() != 2:
        raise ValueError(f"Expected 2D image (H, W), got {image.dim()}D tensor with shape {image.shape}")

    H, W = image.shape
    if H < config.window_size or W < config.window_size:
        raise ValueError(
            f"Image size ({H}x{W}) must be >= window_size ({config.window_size}x{config.window_size})"
        )

    if image.dtype not in (torch.uint8, torch.int16, torch.int32, torch.int64, torch.float32, torch.float64):
        raise ValueError(f"Unsupported dtype: {image.dtype}. Use uint8, int16/32/64, or float32/64.")

    # Check value range
    if image.is_floating_point():
        if image.min() < 0 or image.max() >= config.num_levels:
            raise ValueError(
                f"Float image values must be in [0, {config.num_levels - 1}], "
                f"got range [{image.min().item():.2f}, {image.max().item():.2f}]"
            )
    else:
        if image.min() < 0 or image.max() >= config.num_levels:
            raise ValueError(
                f"Image values must be in [0, {config.num_levels - 1}], "
                f"got range [{image.min().item()}, {image.max().item()}]"
            )


def validate_config(config: HaralickConfig) -> None:
    """Validate configuration parameters for consistency.

    Raises:
        ValueError: If configuration parameters are inconsistent.
    """
    if config.stride > config.window_size:
        raise ValueError(
            f"stride ({config.stride}) must be <= window_size ({config.window_size})"
        )

    if config.tile_size < config.window_size:
        raise ValueError(
            f"tile_size ({config.tile_size}) should be >= window_size ({config.window_size}) "
            f"for efficient processing"
        )