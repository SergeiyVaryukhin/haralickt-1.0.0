"""Core computation engine for Haralick texture features.

Orchestrates the full pipeline:
    1. Input validation
    2. Spatial tiling
    3. Per-tile: window extraction -> GLCM -> features
    4. Assembly into final feature map
"""

from __future__ import annotations

import torch

from .config import HaralickConfig
from .device import clear_cache, move_to_device
from .feature_math import compute_features
from .glcm import build_glcm
from .tiling import compute_output_shape, generate_tiles
from .validation import validate_config, validate_image


class HaralickEngine:
    """GPU/CPU engine for computing 13 Haralick texture features.

    Usage:
        >>> config = HaralickConfig(num_levels=32, window_size=32, stride=8)
        >>> engine = HaralickEngine(config)
        >>> features = engine.compute(image_tensor)  # shape: (H_out, W_out, 13)
    """

    def __init__(self, config: HaralickConfig | None = None) -> None:
        self.config = config or HaralickConfig()
        validate_config(self.config)

    def compute(self, image: torch.Tensor) -> torch.Tensor:
        """Compute Haralick feature map for the input image.

        Args:
            image: 2D tensor of shape (H, W) with values in [0, num_levels - 1].

        Returns:
            Tensor of shape (H_out, W_out, 13) with 13 Haralick features
            at each spatial position. Empty windows (all zeros) produce NaN.
        """
        validate_image(image, self.config)

        device = torch.device(self.config.device)
        image = move_to_device(image, device)

        H, W = image.shape
        H_out, W_out = compute_output_shape(H, W, self.config)

        if H_out == 0 or W_out == 0:
            return torch.zeros((0, 0, 13), dtype=torch.float32, device=device)

        final_features = torch.zeros(
            (H_out, W_out, 13), dtype=torch.float32, device=device
        )

        tiles = generate_tiles(H, W, self.config)

        for tile in tiles:
            tile_img = image[tile.r_start : tile.r_end, tile.c_start : tile.c_end]

            try:
                windows = tile_img.unfold(0, self.config.window_size, self.config.stride).unfold(
                    1, self.config.window_size, self.config.stride
                )
                t_H_out = windows.shape[0]
                t_W_out = windows.shape[1]
                windows = windows.contiguous()
            except RuntimeError:
                continue

            if t_H_out == 0 or t_W_out == 0:
                continue

            # Build GLCM and compute features
            P, sum_glcm = build_glcm(windows, self.config, device)
            features = compute_features(P, self.config, device)

            # Mask empty windows (all-zero regions)
            is_empty = sum_glcm.sum(dim=(0, -2, -1)) == 0
            if is_empty.any():
                features[is_empty] = float("nan")

            # Write tile results into the final feature map
            r0 = tile.out_r_start
            c0 = tile.out_c_start
            final_features[r0 : r0 + t_H_out, c0 : c0 + t_W_out] = features

            # Free GPU memory between tiles
            clear_cache(device)

        return final_features