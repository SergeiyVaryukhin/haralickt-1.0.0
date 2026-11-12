"""Mahotas-compatible Haralick feature extraction.

Provides a drop-in replacement for ``mahotas.features.haralick``.

Example:
    >>> import numpy as np
    >>> from haralickt import features
    >>> image = np.random.randint(0, 32, (64, 64), dtype=np.uint8)
    >>> result = features.haralick(image, ignore_zeros=True, distance=1)
    >>> print(result.shape)  # (4, 13)
"""

from __future__ import annotations

import math

import numpy as np
import torch


def haralick(
    image: np.ndarray,
    ignore_zeros: bool = True,
    distance: int = 1,
) -> np.ndarray:
    """Compute 13 Haralick texture features for a 2D image.

    Drop-in replacement for ``mahotas.features.haralick``.

    Args:
        image: 2D numpy array of shape (H, W) with non-negative integer values.
        ignore_zeros: Exclude pixel pairs involving value 0. Default: True.
        distance: GLCM pixel-pair distance. Default: 1.

    Returns:
        numpy array of shape (4, 13) with one row per direction:
            row 0: 0° (horizontal)
            row 1: 45° (diagonal)
            row 2: 90° (vertical)
            row 3: 135° (diagonal)

        Feature columns (matching mahotas):
            0  - Angular Second Moment (ASM)
            1  - Contrast
            2  - Correlation
            3  - Variance
            4  - Inverse Difference Moment (IDM)
            5  - Sum Average
            6  - Sum Variance
            7  - Sum Entropy
            8  - Entropy
            9  - Difference Variance
            10 - Difference Entropy
            11 - Information Measure of Correlation 1 (IMC1)
            12 - Information Measure of Correlation 2 (IMC2)

    Raises:
        TypeError: If image is not a numpy array.
        ValueError: If image is not 2D or too small for the given distance.
    """
    # --- Validation ---
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")
    if image.ndim != 2:
        raise ValueError(f"Expected 2D array, got {image.ndim}D")
    H, W = image.shape
    if H < 2 or W < 2:
        raise ValueError(f"Image too small: {image.shape}, need at least (2, 2)")
    if distance < 1:
        raise ValueError(f"distance must be >= 1, got {distance}")
    if H <= distance and W <= distance:
        raise ValueError(
            f"Image too small for distance={distance}: shape={image.shape}"
        )

    # --- Determine num_levels from image content ---
    max_val = int(image.max()) if image.size > 0 else 0
    L = max(max_val + 1, 2)

    # --- Convert to torch tensor ---
    tensor = torch.from_numpy(image).long()
    device = torch.device("cpu")

    # --- Build 4 directional GLCMs (mahotas order: 0°, 45°, 90°, 135°) ---
    glcms: list[torch.Tensor] = []

    # 0°: Horizontal
    glcms.append(_bincount_glcm(tensor[:, :-distance], tensor[:, distance:], L, ignore_zeros, device))

    # 45°: Diagonal (top-left to bottom-right)
    glcms.append(_bincount_glcm(tensor[:-distance, :-distance], tensor[distance:, distance:], L, ignore_zeros, device))

    # 90°: Vertical
    glcms.append(_bincount_glcm(tensor[:-distance, :], tensor[distance:, :], L, ignore_zeros, device))

    # 135°: Diagonal (top-right to bottom-left)
    glcms.append(_bincount_glcm(tensor[distance:, :-distance], tensor[:-distance, distance:], L, ignore_zeros, device))

    # Stack: (4, L, L)
    glcm = torch.stack(glcms, dim=0)

    # Symmetrize: GLCM(i,j) += GLCM(j,i)
    glcm = glcm + glcm.transpose(-2, -1)

    # Normalize to probability matrix
    total = glcm.sum(dim=(-2, -1), keepdim=True)
    safe_total = torch.where(total == 0, torch.ones_like(total), total)
    P = glcm / safe_total

    # --- Compute 13 features per direction ---
    result = _compute_features_4dir(P, L, device)

    return result.numpy()


def _bincount_glcm(
    left: torch.Tensor,
    right: torch.Tensor,
    L: int,
    ignore_zeros: bool,
    device: torch.device,
) -> torch.Tensor:
    """Build GLCM for one direction from pixel pairs.

    Args:
        left: Left pixel values, shape (H', W').
        right: Right pixel values, same shape.
        L: Number of gray levels.
        ignore_zeros: Exclude pairs where either pixel is 0.
        device: Target device.

    Returns:
        GLCM tensor of shape (L, L).
    """
    left = left.long()
    right = right.long()

    if ignore_zeros:
        mask = (left != 0) & (right != 0)
    else:
        mask = torch.ones_like(left, dtype=torch.bool)

    if not mask.any():
        return torch.zeros((L, L), dtype=torch.float32, device=device)

    idx = left[mask] * L + right[mask]
    return torch.bincount(idx, minlength=L * L).view(L, L).float()


def _compute_features_4dir(
    P: torch.Tensor,
    L: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute 13 Haralick features for 4 directional GLCMs.

    Args:
        P: Normalized probability matrix, shape (4, L, L).
        L: Number of gray levels.
        device: Target device.

    Returns:
        Tensor of shape (4, 13) with features per direction.
    """
    eps = 1e-10
    log_base = math.log(2.0)

    # Index grids
    i_idx = torch.arange(L, dtype=torch.float32, device=device).view(1, L, 1)
    j_idx = torch.arange(L, dtype=torch.float32, device=device).view(1, 1, L)
    k_idx = torch.arange(L, dtype=torch.float32, device=device).view(1, L)

    features = torch.zeros((4, 13), dtype=torch.float32, device=device)

    # 1. Angular Second Moment (ASM)
    features[:, 0] = (P**2).sum(dim=(1, 2))

    # 2. Contrast
    diff_matrix = (i_idx - j_idx) ** 2
    features[:, 1] = (diff_matrix * P).sum(dim=(1, 2))

    # 3. Correlation
    P_x = P.sum(dim=2)  # (4, L)
    P_y = P.sum(dim=1)  # (4, L)
    mean_x = (k_idx * P_x).sum(dim=1, keepdim=True)  # (4, 1)
    mean_y = (k_idx * P_y).sum(dim=1, keepdim=True)  # (4, 1)
    std_x = torch.sqrt(((k_idx - mean_x) ** 2 * P_x).sum(dim=1) + eps)  # (4,)
    std_y = torch.sqrt(((k_idx - mean_y) ** 2 * P_y).sum(dim=1) + eps)  # (4,)

    cov = ((i_idx - mean_x.unsqueeze(2)) * (j_idx - mean_y.unsqueeze(1)) * P).sum(dim=(1, 2))
    features[:, 2] = cov / (std_x * std_y + eps)

    # 4. Variance
    features[:, 3] = ((i_idx - mean_x.unsqueeze(2)) ** 2 * P).sum(dim=(1, 2))

    # 5. Inverse Difference Moment (IDM)
    features[:, 4] = (P / (1.0 + diff_matrix)).sum(dim=(1, 2))

    # Prepare P_x_plus_y and P_x_minus_y
    _i = torch.arange(L, device=device)
    idx_plus = (_i[:, None] + _i[None, :]).flatten()  # [L*L], values [0, 2L-2]
    idx_minus = (_i[:, None] - _i[None, :]).abs().flatten()  # [L*L], values [0, L-1]

    P_flat = P.reshape(4, L * L)

    P_x_plus_y = torch.zeros(4, 2 * L - 1, dtype=torch.float32, device=device)
    P_x_plus_y.scatter_add_(1, idx_plus.view(1, -1).expand(4, -1), P_flat)

    P_x_minus_y = torch.zeros(4, L, dtype=torch.float32, device=device)
    P_x_minus_y.scatter_add_(1, idx_minus.view(1, -1).expand(4, -1), P_flat)

    k_plus = torch.arange(2 * L - 1, dtype=torch.float32, device=device).view(1, -1)

    # 6. Sum Average
    features[:, 5] = (k_plus * P_x_plus_y).sum(dim=1)

    # 7. Sum Variance
    features[:, 6] = (((k_plus - features[:, 5:6]) ** 2) * P_x_plus_y).sum(dim=1)

    # 8. Sum Entropy
    features[:, 7] = -(P_x_plus_y * torch.log(P_x_plus_y + eps)).sum(dim=1) / log_base

    # 9. Entropy
    features[:, 8] = -(P * torch.log(P + eps)).sum(dim=(1, 2)) / log_base

    # 10. Difference Variance
    _mean_minus = P_x_minus_y.mean(dim=1)
    features[:, 9] = ((P_x_minus_y - _mean_minus.unsqueeze(1)) ** 2).mean(dim=1)

    # 11. Difference Entropy
    features[:, 10] = -(P_x_minus_y * torch.log(P_x_minus_y + eps)).sum(dim=1) / log_base

    # 12 & 13. IMC1, IMC2
    HX = -(P_x * torch.log(P_x + eps)).sum(dim=1) / log_base
    HY = -(P_y * torch.log(P_y + eps)).sum(dim=1) / log_base
    HXY = features[:, 8]

    log_PxPy = torch.log(P_x.unsqueeze(2) * P_y.unsqueeze(1) + eps) / log_base
    HXY1 = -(P * log_PxPy).sum(dim=(1, 2))
    HXY2 = -((P_x.unsqueeze(2) * P_y.unsqueeze(1)) * log_PxPy).sum(dim=(1, 2))

    max_HX_HY = torch.max(HX, HY)
    features[:, 11] = (HXY - HXY1) / (max_HX_HY + eps)
    features[:, 12] = torch.sqrt(torch.clamp(1.0 - torch.exp(-2.0 * (HXY2 - HXY)), min=0.0))

    return features