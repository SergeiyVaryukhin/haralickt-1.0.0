"""Device management utilities for Haralick feature extraction."""

from __future__ import annotations

import torch


def resolve_device(preferred: str = "cuda") -> torch.device:
    """Resolve the best available device.

    Args:
        preferred: Preferred device string ("cuda", "cpu", "mps", etc.).

    Returns:
        A torch.device instance. Falls back to CPU if preferred is unavailable.
    """
    if preferred == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if preferred == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preferred)


def get_device_info(device: torch.device) -> str:
    """Return a human-readable device description."""
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        return f"CUDA:{idx} ({name}, {mem_gb:.1f} GB)"
    if device.type == "mps":
        return "Apple MPS"
    return "CPU"


def move_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move tensor to target device (no-op if already there)."""
    if tensor.device != device:
        return tensor.to(device)
    return tensor


def clear_cache(device: torch.device) -> None:
    """Clear device memory cache if applicable."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()