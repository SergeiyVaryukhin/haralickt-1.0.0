"""Сравнение haralickt (PyTorch) vs mahotas (CPU) по всем 13 признакам.

Два режима:
    1. PyTorch CPU vs mahotas CPU (1 поток) — базовое сравнение
    2. PyTorch GPU vs mahotas CPU (10 потоков) — реальное преимущество GPU


Перебирает все комбинации:
    окна:    8, 16, 32
    шаги:    2, 4, 8
    дистанции: 1, 2, 4

Входное изображение: 256x256, 32 градации.

Запуск:
    python examples/compare_with_mahotas.py
"""

from __future__ import annotations

import time
from itertools import product

import numpy as np
import torch

from haralickt import haralick

# ====================================================================
# 1. MAHOTAS REFERENCE (CPU, multiprocessing)
# ====================================================================


def _extract_features(image: np.ndarray, distance: int) -> np.ndarray:
    """Вычислить 13 признаков Харалика для одного окна через mahotas."""
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
    """Попиксельный расчёт через mahotas + multiprocessing (эталон)."""
    from multiprocessing import Pool

    pool = Pool(threads)
    har_list: list[np.ndarray] = []

    for r in range(0, gray_img.shape[0] - window_size + 1, stride):
        sub_imgs = [
            gray_img[r : r + window_size, c : c + window_size]
            for c in range(0, gray_img.shape[1] - window_size + 1, stride)
        ]
        har_row = pool.starmap(_extract_features, [(img, distance) for img in sub_imgs])
        har_list.append(np.array(har_row))

    pool.close()
    pool.join()
    return np.array(har_list).astype("float32")


# ====================================================================
# 2. НАЗВАНИЯ ПРИЗНАКОВ
# ====================================================================

FEATURE_NAMES = [
    "ASM (Угловой второй момент)",
    "Contrast (Контраст)",
    "Correlation (Корреляция)",
    "Dissimilarity (Диссимилярность)",
    "Entropy (Энтропия)",
    "Inverse Difference Moment (Обратный момент разницы)",
    "Inverse Difference Moment 2 (Обратный момент разницы 2)",
    "Max Probability (Макс. вероятность)",
    "Sum Average (Среднее суммы)",
    "Sum Variance (Дисперсия суммы)",
    "Sum Entropy (Энтропия суммы)",
    "Variance (Дисперсия)",
    "Standard Deviation (Стандартное отклонение)",
    "Diff Entropy (Энтропия разницы)",
]


# ====================================================================
# 3. БЕНЧМАРК (параметризованный)
# ====================================================================


def run_benchmark(np_img, torch_img, device, mahotas_threads, label):
    """Запустить бенчмарк для одного сценария."""
    num_levels = 32
    windows = [8, 16, 32]
    strides = [2, 4, 8]
    distances = [1, 2, 4]

    print(f"\n{'=' * 78}")
    print(f"  СЦЕНАРИЙ: {label}")
    print(f"  PyTorch device: {device} | mahotas threads: {mahotas_threads}")
    print(f"{'=' * 78}")

    all_results = []
    total = len(windows) * len(strides) * len(distances)

    for i, (ws, st, dist) in enumerate(product(windows, strides, distances), 1):
        print(f"\n  [{i}/{total}]  окно={ws}, шаг={st}, дистанция={dist}")

        t0 = time.perf_counter()
        cpu_result = mahotas_haralick(np_img, ws, st, dist, threads=mahotas_threads)
        t_cpu = time.perf_counter() - t0

        t0 = time.perf_counter()
        torch_result = haralick(
            torch_img, window_size=ws, stride=st,
            num_levels=num_levels, distance=dist, device=device,
        )
        t_torch = time.perf_counter() - t0
        torch_np = torch_result.cpu().numpy()

        if cpu_result.shape != torch_np.shape:
            print(f"  [!] SHAPE MISMATCH: {cpu_result.shape} vs {torch_np.shape}")
            all_results.append({"window": ws, "stride": st, "distance": dist,
                                "shape_ok": False, "max_diff": float("nan"),
                                "mean_diff": float("nan"), "t_cpu": t_cpu, "t_torch": t_torch})
            continue

        mask = ~np.isnan(cpu_result) & ~np.isnan(torch_np)
        all_diffs = np.abs(cpu_result[mask] - torch_np[mask])
        max_diff = float(all_diffs.max()) if all_diffs.size > 0 else 0.0
        mean_diff = float(all_diffs.mean()) if all_diffs.size > 0 else 0.0

        sp = t_cpu / t_torch if t_torch > 0 else float("inf")
        print(f"  [+] mahotas: {t_cpu:.4f}s | haralickt: {t_torch:.4f}s | speedup: {sp:.2f}x")
        print(f"  [+] max_diff: {max_diff:.6f} | mean_diff: {mean_diff:.8f}")

        all_results.append({"window": ws, "stride": st, "distance": dist,
                            "shape_ok": True, "max_diff": max_diff,
                            "mean_diff": mean_diff, "t_cpu": t_cpu, "t_torch": t_torch})

    # --- Сводная таблица ---
    print(f"\n  СВОДНАЯ ТАБЛИЦА: {label}")
    print(f"  {'Окно':>5} | {'Шаг':>4} | {'Дист':>5} | {'Max diff':>12} | {'mahotas':>8} | {'torch':>8} | {'Уск.':>5}")
    print("  " + "-" * 65)
    for r in all_results:
        if r["shape_ok"]:
            sp = r["t_cpu"] / r["t_torch"] if r["t_torch"] > 0 else float("inf")
            print(f"  {r['window']:5d} | {r['stride']:4d} | {r['distance']:5d} | "
                  f"{r['max_diff']:12.6f} | {r['t_cpu']:7.3f}s | {r['t_torch']:7.3f}s | {sp:4.1f}x")
        else:
            print(f"  {r['window']:5d} | {r['stride']:4d} | {r['distance']:5d} | "
                  f"{'FAIL':>12} | {r['t_cpu']:7.3f}s | {r['t_torch']:7.3f}s |    N/A")
    print("  " + "-" * 65)
    valid = [r for r in all_results if r["shape_ok"]]
    if valid:
        gm = max(r["max_diff"] for r in valid)
        print(f"  ГЛОБАЛЬНЫЙ MAX: {gm:.8f}")

    return all_results


# ====================================================================
# 4. ГЛАВНЫЙ БЛОК
# ====================================================================

if __name__ == "__main__":
    num_levels = 32
    test_size = 256

    print(f"[*] Генерация изображения {test_size}x{test_size} ({num_levels} градаций)...")
    np.random.seed(42)
    np_img = np.random.randint(0, num_levels, size=(test_size, test_size), dtype=np.uint8)
    torch_img = torch.from_numpy(np_img)

    # --- Сценарий 1: CPU vs CPU (1 поток) ---
    results_cpu = run_benchmark(
        np_img, torch_img,
        device="cpu",
        mahotas_threads=1,
        label="PyTorch CPU vs mahotas CPU (1 поток)",
    )

    # --- Сценарий 2: GPU vs CPU (10 потоков) ---
    results_gpu = None
    if torch.cuda.is_available():
        results_gpu = run_benchmark(
            np_img, torch_img,
            device="cuda",
            mahotas_threads=10,
            label="PyTorch GPU vs mahotas CPU (10 потоков)",
        )
    else:
        print("\n[!] CUDA недоступна — сценарий GPU пропущен.")

    # --- Итоговое сравнение ---
    if results_gpu is not None:
        print(f"\n\n{'=' * 78}")
        print("  ИТОГОВОЕ СРАВНЕНИЕ")
        print(f"{'=' * 78}")
        print(f"  {'Сценарий':<45} | {'mahotas (ср.)':>14} | {'torch (ср.)':>12} | {'Уск.':>6}")
        print("  " + "-" * 85)
        for label, results in [("CPU vs CPU (1 thr)", results_cpu), ("GPU vs CPU (10 thr)", results_gpu)]:
            valid = [r for r in results if r["shape_ok"]]
            if valid:
                avg_c = sum(r["t_cpu"] for r in valid) / len(valid)
                avg_t = sum(r["t_torch"] for r in valid) / len(valid)
                sp = avg_c / avg_t if avg_t > 0 else float("inf")
                print(f"  {label:<45} | {avg_c:13.3f}s | {avg_t:11.3f}s | {sp:5.1f}x")
        print("  " + "-" * 85)
        print(f"{'=' * 78}")
