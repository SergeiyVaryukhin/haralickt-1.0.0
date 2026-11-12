"""GLCM (Gray-Level Co-occurrence Matrix) construction for a batch of windows.

Builds 4 directional GLCMs (horizontal, diagonal-135, vertical, diagonal-45)
using vectorized bincount over all windows simultaneously.
"""

from __future__ import annotations

import torch

from .config import HaralickConfig


def build_glcm(
    windows: torch.Tensor,
    config: HaralickConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build symmetrized, normalized GLCM for all windows in a tile.

    Args:
        windows: Tensor of shape (t_H_out, t_W_out, window_size, window_size)
                 containing all spatial windows for this tile.
        config: Haralick configuration.
        device: Target device.

    Returns:
        P: Normalized probability matrix, shape (4, t_H_out, t_W_out, L, L).
        sum_glcm: Raw GLCM sums, shape (4, t_H_out, t_W_out, 1, 1).
                  Used for empty-window detection.
    """
    L = config.num_levels
    d = config.resolved_distance
    t_H_out, t_W_out = windows.shape[0], windows.shape[1]

    # Precompute batch offsets for parallel bincount layout
    grid_offsets = (
        torch.arange(t_H_out, device=device).unsqueeze(1) * t_W_out
        + torch.arange(t_W_out, device=device).unsqueeze(0)
    ).view(t_H_out, t_W_out, 1, 1)
    batch_offsets = grid_offsets * (L * L)
    min_len = t_H_out * t_W_out * L * L

    # --- Direction 0: Horizontal (0 degrees) ---
    left = windows[:, :, :, :-d].long()
    right = windows[:, :, :, d:].long()
    if config.ignore_zeros:
        mask = (left != 0) & (right != 0)
    else:
        mask = torch.ones_like(left, dtype=torch.bool)

    glcm0 = _bincount_glcm(left, right, mask, batch_offsets, min_len, t_H_out, t_W_out, L, device)

    # --- Direction 1: Diagonal 135 degrees ---
    left = windows[:, :, :-d, :-d].long()
    right = windows[:, :, d:, d:].long()
    if config.ignore_zeros:
        mask = (left != 0) & (right != 0)
    else:
        mask = torch.ones_like(left, dtype=torch.bool)

    glcm1 = _bincount_glcm(left, right, mask, batch_offsets, min_len, t_H_out, t_W_out, L, device)

    # --- Direction 2: Vertical (90 degrees) ---
    left = windows[:, :, :-d, :].long()
    right = windows[:, :, d:, :].long()
    if config.ignore_zeros:
        mask = (left != 0) & (right != 0)
    else:
        mask = torch.ones_like(left, dtype=torch.bool)

    glcm2 = _bincount_glcm(left, right, mask, batch_offsets, min_len, t_H_out, t_W_out, L, device)

    # --- Direction 3: Diagonal 45 degrees ---
    left = windows[:, :, d:, :-d].long()
    right = windows[:, :, :-d, d:].long()
    if config.ignore_zeros:
        mask = (left != 0) & (right != 0)
    else:
        mask = torch.ones_like(left, dtype=torch.bool)

    glcm3 = _bincount_glcm(left, right, mask, batch_offsets, min_len, t_H_out, t_W_out, L, device)

    # Stack into 5D tensor: [4, t_H_out, t_W_out, L, L]
    glcm = torch.stack([glcm0, glcm1, glcm2, glcm3], dim=0)

    # Symmetrize: GLCM(i,j) += GLCM(j,i)
    glcm = glcm + glcm.transpose(-2, -1)

    # Normalize to probability matrix P
    sum_glcm = glcm.sum(dim=(-2, -1), keepdim=True)
    safe_sum = torch.where(sum_glcm == 0, torch.ones_like(sum_glcm), sum_glcm)
    P = glcm / safe_sum

    return P, sum_glcm


def _bincount_glcm(
    left: torch.Tensor,
    right: torch.Tensor,
    mask: torch.Tensor,
    batch_offsets: torch.Tensor,
    min_len: int,
    t_H_out: int,
    t_W_out: int,
    L: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute GLCM via bincount for one direction.

    Args:
        left: Left pixel values, shape (t_H_out, t_W_out, H', W').
        right: Right pixel values, same shape.
        mask: Boolean mask for valid pairs.
        batch_offsets: Per-window offset for parallel bincount.
        min_len: Minimum length for bincount output.
        t_H_out: Number of output rows in tile.
        t_W_out: Number of output columns in tile.
        L: Number of gray levels.
        device: Target device.

    Returns:
        GLCM tensor of shape (t_H_out, t_W_out, L, L).
    """
    if not mask.any():
        return torch.zeros((t_H_out, t_W_out, L, L), dtype=torch.float32, device=device)

    idx = (left * L + right) + batch_offsets
    return (
        torch.bincount(idx[mask], minlength=min_len)
        .view(t_H_out, t_W_out, L, L)
        .float()
    )