"""Computation of 13 Haralick texture features from normalized GLCM.

Feature indices (matching mahotas.features.haralick):
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
"""

from __future__ import annotations

import math

import torch

from .config import HaralickConfig


def compute_features(
    P: torch.Tensor,
    config: HaralickConfig,
    device: torch.device,
) -> torch.Tensor:
    """Compute all 13 Haralick features for 4 directional GLCMs.

    Args:
        P: Normalized probability matrix, shape (4, t_H_out, t_W_out, L, L).
        config: Haralick configuration.
        device: Target device.

    Returns:
        features: Tensor of shape (t_H_out, t_W_out, 13) with features
                  averaged over 4 directions.
    """
    L = config.num_levels
    eps = config.eps
    log_base = math.log(2.0)

    # Index grids for vectorized formulas
    i_idx = torch.arange(L, dtype=torch.float32, device=device).view(1, 1, 1, L, 1)
    j_idx = torch.arange(L, dtype=torch.float32, device=device).view(1, 1, 1, 1, L)
    k_idx = torch.arange(L, dtype=torch.float32, device=device).view(1, 1, 1, L)

    t_H_out = P.shape[1]
    t_W_out = P.shape[2]

    features_4dir = torch.zeros((4, t_H_out, t_W_out, 13), dtype=torch.float32, device=device)

    # --- 1. Angular Second Moment (ASM) ---
    features_4dir[:, :, :, 0] = (P**2).sum(dim=(3, 4))

    # --- 2. Contrast ---
    diff_matrix = (i_idx - j_idx) ** 2
    features_4dir[:, :, :, 1] = (diff_matrix * P).sum(dim=(3, 4))

    # --- 3. Correlation ---
    P_x = P.sum(dim=4)  # Marginal P(x)
    P_y = P.sum(dim=3)  # Marginal P(y)
    mean_x = (k_idx * P_x).sum(dim=3, keepdim=True)
    mean_y = (k_idx * P_y).sum(dim=3, keepdim=True)
    std_x = torch.sqrt(((k_idx - mean_x) ** 2 * P_x).sum(dim=3) + eps)
    std_y = torch.sqrt(((k_idx - mean_y) ** 2 * P_y).sum(dim=3) + eps)

    cov = ((i_idx - mean_x.unsqueeze(-1)) * (j_idx - mean_y.unsqueeze(-1)) * P).sum(dim=(3, 4))
    features_4dir[:, :, :, 2] = cov / (std_x * std_y + eps)

    # --- 4. Variance ---
    features_4dir[:, :, :, 3] = ((i_idx - mean_x.unsqueeze(-1)) ** 2 * P).sum(dim=(3, 4))

    # --- 5. Inverse Difference Moment (IDM) ---
    features_4dir[:, :, :, 4] = (P / (1.0 + diff_matrix)).sum(dim=(3, 4))

    # --- Prepare P_x_plus_y and P_x_minus_y via scatter_add_ ---
    _i = torch.arange(L, device=device)
    idx_plus = (_i[:, None] + _i[None, :]).flatten()  # [L*L], values [0, 2L-2]
    idx_minus = (_i[:, None] - _i[None, :]).abs().flatten()  # [L*L], values [0, L-1]

    P_flat = P.reshape(4, t_H_out, t_W_out, L * L)

    P_x_plus_y = torch.zeros(4, t_H_out, t_W_out, 2 * L - 1, dtype=torch.float32, device=device)
    P_x_plus_y.scatter_add_(3, idx_plus.view(1, 1, 1, -1).expand(4, t_H_out, t_W_out, -1), P_flat)

    P_x_minus_y = torch.zeros(4, t_H_out, t_W_out, L, dtype=torch.float32, device=device)
    P_x_minus_y.scatter_add_(3, idx_minus.view(1, 1, 1, -1).expand(4, t_H_out, t_W_out, -1), P_flat)

    k_plus = torch.arange(2 * L - 1, dtype=torch.float32, device=device).view(1, 1, 1, -1)
    k_minus = torch.arange(L, dtype=torch.float32, device=device).view(1, 1, 1, -1)

    # --- 6. Sum Average ---
    features_4dir[:, :, :, 5] = (k_plus * P_x_plus_y).sum(dim=3)

    # --- 7. Sum Variance ---
    features_4dir[:, :, :, 6] = (((k_plus - features_4dir[:, :, :, 5].unsqueeze(3)) ** 2) * P_x_plus_y).sum(dim=3)

    # --- 8. Sum Entropy ---
    features_4dir[:, :, :, 7] = -(P_x_plus_y * torch.log(P_x_plus_y + eps)).sum(dim=3) / log_base

    # --- 9. Entropy ---
    features_4dir[:, :, :, 8] = -(P * torch.log(P + eps)).sum(dim=(3, 4)) / log_base

    # --- 11. Difference Entropy (index 10) ---
    features_4dir[:, :, :, 10] = -(P_x_minus_y * torch.log(P_x_minus_y + eps)).sum(dim=3) / log_base

    # --- 10. Difference Variance (index 9) ---
    _mean_minus = P_x_minus_y.mean(dim=3)
    features_4dir[:, :, :, 9] = ((P_x_minus_y - _mean_minus.unsqueeze(3)) ** 2).mean(dim=3)

    # --- 12 & 13. IMC1, IMC2 ---
    HX = -(P_x * torch.log(P_x + eps)).sum(dim=3) / log_base
    HY = -(P_y * torch.log(P_y + eps)).sum(dim=3) / log_base
    HXY = features_4dir[:, :, :, 8]

    log_PxPy = torch.log(P_x.unsqueeze(4) * P_y.unsqueeze(3) + eps) / log_base
    HXY1 = -(P * log_PxPy).sum(dim=(3, 4))
    HXY2 = -((P_x.unsqueeze(4) * P_y.unsqueeze(3)) * log_PxPy).sum(dim=(3, 4))

    max_HX_HY = torch.max(HX, HY)
    features_4dir[:, :, :, 11] = (HXY - HXY1) / (max_HX_HY + eps)
    features_4dir[:, :, :, 12] = torch.sqrt(torch.clamp(1.0 - torch.exp(-2.0 * (HXY2 - HXY)), min=0.0))

    # Average over 4 directions -> [t_H_out, t_W_out, 13]
    features = features_4dir.mean(dim=0)

    return features