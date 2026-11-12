"""Configuration dataclass for Haralick feature extraction."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HaralickConfig:
    """Immutable configuration for the Haralick feature engine.

    Attributes:
        num_levels: Number of gray levels (L). Image values must be in [0, L-1].
        window_size: Spatial window size (square) for local GLCM computation.
        stride: Step between consecutive windows.
        ignore_zeros: If True, pixel pairs involving value 0 are excluded from GLCM.
        tile_size: Spatial tile size for block-wise processing (OOM prevention).
        device: Target compute device string (e.g. "cuda", "cpu").
        eps: Numerical epsilon for division safety.
    """

    num_levels: int = 32
    window_size: int = 32
    stride: int = 8
    ignore_zeros: bool = True
    tile_size: int = 1024
    device: str = "cuda"
    eps: float = 1e-10
    distance: int | None = None  # None = auto-derive: max(1, window_size // 8)

    def __post_init__(self) -> None:
        if self.num_levels < 2:
            raise ValueError(f"num_levels must be >= 2, got {self.num_levels}")
        if self.window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {self.window_size}")
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        if self.tile_size < 1:
            raise ValueError(f"tile_size must be >= 1, got {self.tile_size}")
        if self.eps <= 0:
            raise ValueError(f"eps must be > 0, got {self.eps}")
        if self.distance is not None and self.distance < 1:
            raise ValueError(f"distance must be >= 1, got {self.distance}")
        if self.distance is not None and self.distance >= self.window_size:
            raise ValueError(
                f"distance ({self.distance}) must be < window_size ({self.window_size})"
            )

    @property
    def resolved_distance(self) -> int:
        """GLCM pixel-pair distance.

        Returns the explicit ``distance`` if set, otherwise auto-derives
        as ``max(1, window_size // 8)``.
        """
        if self.distance is not None:
            return self.distance
        return max(1, self.window_size // 8)

    @property
    def num_features(self) -> int:
        """Number of Haralick features (always 13)."""
        return 13

    def output_shape(self, H: int, W: int) -> tuple[int, int, int]:
        """Compute output tensor shape (H_out, W_out, 13) for input of size (H, W)."""
        H_out = (H - self.window_size) // self.stride + 1
        W_out = (W - self.window_size) // self.stride + 1
        if H_out <= 0 or W_out <= 0:
            return (0, 0, self.num_features)
        return (H_out, W_out, self.num_features)