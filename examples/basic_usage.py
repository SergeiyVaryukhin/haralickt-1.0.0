"""Basic usage example for haralickt.

Demonstrates:
    1. Computing Haralick features on a synthetic image
    2. Using the engine API for repeated computation
    3. NumPy input support
    4. Device selection

Run:
    python examples/basic_usage.py
"""

from __future__ import annotations

import time

import numpy as np
import torch

from haralickt import HaralickConfig, HaralickEngine, get_device_info, haralick, resolve_device


def example_basic():
    """Basic usage: compute features on a random image."""
    print("=" * 60)
    print("1. Basic usage")
    print("=" * 60)

    # Create a synthetic image with texture
    torch.manual_seed(42)
    H, W = 256, 256
    image = torch.randint(0, 32, (H, W))

    # Add some structure (horizontal stripes)
    for i in range(0, H, 16):
        image[i : i + 8, :] = 16

    # Compute features
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    start = time.perf_counter()
    features = haralick(
        image,
        num_levels=32,
        window_size=32,
        stride=8,
        device=device,
    )
    elapsed = time.perf_counter() - start

    print(f"Input shape:  {image.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Time:         {elapsed:.3f}s")
    print(f"\nFeature means:")
    feature_names = [
        "ASM", "Contrast", "Correlation", "Variance", "IDM",
        "Sum Avg", "Sum Var", "Sum Entropy", "Entropy",
        "Diff Var", "Diff Entropy", "IMC1", "IMC2",
    ]
    for i, name in enumerate(feature_names):
        print(f"  {name:15s}: {features[:, :, i].mean():.6f}")


def example_engine():
    """Engine API for repeated computation with the same config."""
    print("\n" + "=" * 60)
    print("2. Engine API (reusable)")
    print("=" * 60)

    config = HaralickConfig(
        num_levels=32,
        window_size=32,
        stride=8,
        tile_size=128,
        device="cpu",
    )
    engine = HaralickEngine(config)

    # Process multiple images
    for size in [(64, 64), (128, 128), (256, 256)]:
        image = torch.randint(1, 32, size)
        start = time.perf_counter()
        features = engine.compute(image)
        elapsed = time.perf_counter() - start
        print(f"  {size[0]}x{size[1]} -> {features.shape} in {elapsed:.3f}s")


def example_numpy():
    """NumPy input support."""
    print("\n" + "=" * 60)
    print("3. NumPy input")
    print("=" * 60)

    from haralickt import haralick_from_numpy

    # Simulate a grayscale image
    image_np = np.random.randint(0, 32, (128, 128), dtype=np.uint8)

    features = haralick_from_numpy(
        image_np,
        num_levels=32,
        window_size=32,
        stride=8,
        device="cpu",
    )

    print(f"  Input:  {image_np.shape} {image_np.dtype}")
    print(f"  Output: {features.shape} {features.dtype}")


def example_device_info():
    """Show available devices."""
    print("\n" + "=" * 60)
    print("4. Device info")
    print("=" * 60)

    if torch.cuda.is_available():
        cuda_device = resolve_device("cuda")
        print(f"  CUDA: {get_device_info(cuda_device)}")
    else:
        print("  CUDA: not available")

    cpu_device = resolve_device("cpu")
    print(f"  CPU:  {get_device_info(cpu_device)}")


def example_tiling():
    """Demonstrate tiling for large images."""
    print("\n" + "=" * 60)
    print("5. Tiling for large images")
    print("=" * 60)

    H, W = 512, 512
    image = torch.randint(1, 32, (H, W))

    for tile_size in [128, 256, 512]:
        start = time.perf_counter()
        features = haralick(
            image,
            num_levels=32,
            window_size=32,
            stride=8,
            tile_size=tile_size,
            device="cpu",
        )
        elapsed = time.perf_counter() - start
        print(f"  tile_size={tile_size:4d}: {features.shape} in {elapsed:.3f}s")


if __name__ == "__main__":
    example_basic()
    example_engine()
    example_numpy()
    example_device_info()
    example_tiling()
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)