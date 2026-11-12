"""Spatial tiling iteration for block-wise Haralick computation.

Generates tile boundaries that cover the full image with proper overlap
so that all output windows are computed exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import HaralickConfig


@dataclass(frozen=True)
class TileBounds:
    """Spatial bounds of a single tile in the input image."""

    r_start: int
    r_end: int
    c_start: int
    c_end: int
    out_r_start: int
    out_c_start: int


def generate_tiles(H: int, W: int, config: HaralickConfig) -> list[TileBounds]:
    """Generate all tile boundaries for the image.

    The tiling strategy ensures:
    - Each tile covers a region of at most tile_size x tile_size pixels.
    - Tiles overlap by (window_size - stride) pixels to avoid gaps.
    - Every output window position is covered by exactly one tile.

    Args:
        H: Image height.
        W: Image width.
        config: Haralick configuration.

    Returns:
        List of TileBounds covering the entire image.
    """
    tiles: list[TileBounds] = []
    ws = config.window_size
    stride = config.stride
    tile_size = config.tile_size

    # Number of output positions
    H_out = (H - ws) // stride + 1
    W_out = (W - ws) // stride + 1

    if H_out <= 0 or W_out <= 0:
        return tiles

    for r_start in range(0, H - ws + 1, tile_size):
        r_end = min(r_start + tile_size + ws - stride, H)
        out_r_start = r_start // stride

        for c_start in range(0, W - ws + 1, tile_size):
            c_end = min(c_start + tile_size + ws - stride, W)
            out_c_start = c_start // stride

            tiles.append(
                TileBounds(
                    r_start=r_start,
                    r_end=r_end,
                    c_start=c_start,
                    c_end=c_end,
                    out_r_start=out_r_start,
                    out_c_start=out_c_start,
                )
            )

    return tiles


def compute_output_shape(H: int, W: int, config: HaralickConfig) -> tuple[int, int]:
    """Compute the output spatial dimensions (H_out, W_out)."""
    H_out = (H - config.window_size) // config.stride + 1
    W_out = (W - config.window_size) // config.stride + 1
    return max(0, H_out), max(0, W_out)