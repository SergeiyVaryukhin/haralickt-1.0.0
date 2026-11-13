"""Benchmark haralickt vs mahotas across different image sizes.

Uses middle-range parameters (window=16, stride=4, distance=1)
where speedup is close to the average.

Image sizes: 256x256, 512x512, 1024x1024, 2048x2048

Usage:
    python examples/benchmark_by_size.py
"""

from __future__ import annotations

import time

import numpy as np
import torch

from haralickt import haralick

# ─── Parameters (middle range → average speedup) ────────────────────────────
WINDOW = 16
STRIDE = 4
DISTANCE = 1
NUM_LEVELS = 32
SIZES = [256, 512, 1024, 2048]


def _extract_features(image: np.ndarray, distance: int) -> np.ndarray:
    """Compute 13 Haralick features for a single window via mahotas."""
    import mahotas
    try:
        textures = mahotas.features.haralick(image, ignore_zeros=True, distance=distance)
    except ValueError:
        textures = np.zeros((4, 13)) + np.nan
    return textures.mean(axis=0)


def mahotas_haralick(
    gray_img: np.ndarray,
    window_size: int,
    stride: int,
    distance: int,
    threads: int = 10,
) -> np.ndarray:
    """Per-window computation via mahotas + multiprocessing."""
    from multiprocessing import Pool

    pool = Pool(threads)
    har_list: list[np.ndarray] = []

    for r in range(0, gray_img.shape[0] - window_size + 1, stride):
        sub_imgs = [
            gray_img[r: r + window_size, c: c + window_size]
            for c in range(0, gray_img.shape[1] - window_size + 1, stride)
        ]
        har_row = pool.starmap(_extract_features, [(img, distance) for img in sub_imgs])
        har_list.append(np.array(har_row))

    pool.close()
    pool.join()
    return np.array(har_list).astype("float32")


def benchmark_size(size: int, device: str, mahotas_threads: int) -> dict:
    """Run benchmark for one image size."""
    np.random.seed(42)
    np_img = np.random.randint(0, NUM_LEVELS, size=(size, size), dtype=np.uint8)
    torch_img = torch.from_numpy(np_img)

    # Warmup
    _ = haralick(torch_img, num_levels=NUM_LEVELS, window_size=WINDOW,
                 stride=STRIDE, device=device)

    # Time haralickt
    t0 = time.perf_counter()
    result = haralick(torch_img, num_levels=NUM_LEVELS, window_size=WINDOW,
                      stride=STRIDE, device=device)
    t_haralickt = time.perf_counter() - t0

    # Time mahotas
    t0 = time.perf_counter()
    result_m = mahotas_haralick(np_img, WINDOW, STRIDE, DISTANCE, mahotas_threads)
    t_mahotas = time.perf_counter() - t0

    speedup = t_mahotas / t_haralickt if t_haralickt > 0 else float("inf")

    return {
        "size": size,
        "t_mahotas": t_mahotas,
        "t_haralickt": t_haralickt,
        "speedup": speedup,
    }


def main():
    print(f"Parameters: window={WINDOW}, stride={STRIDE}, distance={DISTANCE}, levels={NUM_LEVELS}")
    print(f"Image sizes: {SIZES}")
    print()

    # ─── CPU scenario ────────────────────────────────────────────────────────
    print(f"{'=' * 70}")
    print(f"  CPU: PyTorch CPU vs mahotas (1 thread)")
    print(f"{'=' * 70}")
    print(f"  {'Size':>10} | {'mahotas':>10} | {'haralickt':>10} | {'Speedup':>8}")
    print(f"  {'-' * 50}")

    cpu_results = []
    for size in SIZES:
        r = benchmark_size(size, device="cpu", mahotas_threads=1)
        cpu_results.append(r)
        print(f"  {size:>4}x{size:<4} | {r['t_mahotas']:>8.3f}s | {r['t_haralickt']:>8.3f}s | {r['speedup']:>6.1f}x")

    # ─── GPU scenario ────────────────────────────────────────────────────────
    gpu_results = []
    if torch.cuda.is_available():
        print(f"\n{'=' * 70}")
        print(f"  GPU: PyTorch CUDA vs mahotas (10 threads)")
        print(f"{'=' * 70}")
        print(f"  {'Size':>10} | {'mahotas':>10} | {'haralickt':>10} | {'Speedup':>8}")
        print(f"  {'-' * 50}")

        for size in SIZES:
            r = benchmark_size(size, device="cuda", mahotas_threads=10)
            gpu_results.append(r)
            print(f"  {size:>4}x{size:<4} | {r['t_mahotas']:>8.3f}s | {r['t_haralickt']:>8.3f}s | {r['speedup']:>6.1f}x")
    else:
        print("\n[!] CUDA not available — GPU scenario skipped.")

    # ─── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    if cpu_results:
        avg_sp = np.mean([r["speedup"] for r in cpu_results])
        print(f"  CPU avg speedup: {avg_sp:.1f}x  (range: {min(r['speedup'] for r in cpu_results):.1f}x – {max(r['speedup'] for r in cpu_results):.1f}x)")
    if gpu_results:
        avg_sp = np.mean([r["speedup"] for r in gpu_results])
        print(f"  GPU avg speedup: {avg_sp:.1f}x  (range: {min(r['speedup'] for r in gpu_results):.1f}x – {max(r['speedup'] for r in gpu_results):.1f}x)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()