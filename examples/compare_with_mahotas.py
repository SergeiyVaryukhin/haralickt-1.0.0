"""Сравнение haralickt (PyTorch) vs mahotas (CPU) по всем 13 признакам.

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
    threads: int = 7,
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
    "Variance (Дисперсия)",
    "IDM (Обратный момент)",
    "Sum Average (Суммарное среднее)",
    "Sum Variance (Суммарная дисперсия)",
    "Sum Entropy (Суммарная энтропия)",
    "Entropy (Энтропия)",
    "Diff Variance (Дисперсия разности)",
    "Diff Entropy (Энтропия разности)",
    "IMC1 (Инф. мера корреляции 1)",
    "IMC2 (Инф. мера корреляции 2)",
]


# ====================================================================
# 3. ГЛАВНЫЙ БЛОК СРАВНЕНИЯ
# ====================================================================

if __name__ == "__main__":
    print("=" * 78)
    print("  СРАВНЕНИЕ haralickt (PyTorch) vs mahotas (CPU)")
    print("  Изображение: 256x256, 32 градации")
    print("  Комбинации: окна {8,16,32} x шаги {2,4,8} x дистанции {1,2,4}")
    print("=" * 78)

    num_levels = 32
    test_size = 256

    print(f"\n[*] Генерация изображения {test_size}x{test_size} ({num_levels} градаций)...")
    np.random.seed(42)
    np_img = np.random.randint(0, num_levels, size=(test_size, test_size), dtype=np.uint8)
    torch_img = torch.from_numpy(np_img)

    # --- Перебор всех комбинаций ---
    windows = [8, 16, 32]
    strides = [2, 4, 8]
    distances = [1, 2, 4]

    all_results: list[dict] = []
    total_combos = len(windows) * len(strides) * len(distances)
    combo_idx = 0

    for ws, st, dist in product(windows, strides, distances):
        combo_idx += 1
        print(f"\n{'─' * 78}")
        print(f"  [{combo_idx}/{total_combos}]  окно={ws}, шаг={st}, дистанция={dist}")
        print(f"{'─' * 78}")

        # --- Mahotas (CPU) ---
        t_start_cpu = time.perf_counter()
        cpu_result = mahotas_haralick(
            np_img, window_size=ws, stride=st, distance=dist, threads=7
        )
        t_cpu = time.perf_counter() - t_start_cpu

        # --- PyTorch (haralickt) ---
        t_start_torch = time.perf_counter()
        torch_result = haralick(
            torch_img,
            num_levels=num_levels,
            window_size=ws,
            stride=st,
            distance=dist,
            device="cpu",
        )
        t_torch = time.perf_counter() - t_start_torch

        torch_result_np = torch_result.numpy()

        # --- Проверка размерностей ---
        shape_ok = cpu_result.shape == torch_result_np.shape
        if not shape_ok:
            print(f"  [!] РАЗМЕРНОСТИ РАЗЛИЧАЮТСЯ!")
            print(f"      mahotas      : {cpu_result.shape}")
            print(f"      haralickt: {torch_result_np.shape}")
            all_results.append({
                "window": ws, "stride": st, "distance": dist,
                "shape_ok": False, "max_diff": float("inf"),
                "t_cpu": t_cpu, "t_torch": t_torch,
            })
            continue

        # --- Попривнакный анализ ---
        mask = ~np.isnan(cpu_result) & ~np.isnan(torch_result_np)
        max_diffs: list[float] = []

        for idx in range(13):
            c_feat = cpu_result[:, :, idx]
            t_feat = torch_result_np[:, :, idx]
            m = mask[:, :, idx]
            max_d = np.max(np.abs(c_feat[m] - t_feat[m])) if np.any(m) else 0.0
            max_diffs.append(max_d)

        overall_max = max(max_diffs)
        all_diffs = np.abs(cpu_result[mask] - torch_result_np[mask])
        mean_diff = float(all_diffs.mean()) if all_diffs.size > 0 else 0.0
        p99 = float(np.percentile(all_diffs, 99)) if all_diffs.size > 0 else 0.0

        # --- Вывод ---
        print(f"  [+] Размерности совпадают: {cpu_result.shape}")
        print(f"  [+] mahotas: {t_cpu:.4f} сек | haralickt: {t_torch:.4f} сек")
        if t_torch > 0:
            print(f"  [+] Ускорение: {t_cpu / t_torch:.2f}x")

        print(f"\n  {'Индекс':>6} | {'Название признака':<35} | {'Макс. расхождение':>16}")
        print("  " + "-" * 65)
        for idx in range(13):
            print(f"  {idx:6d} | {FEATURE_NAMES[idx]:<35} | {max_diffs[idx]:>16.6f}")
        print("  " + "-" * 65)
        print(f"  {'':>6} | {'MAX (все признаки)':<35} | {overall_max:>16.6f}")

        print(f"\n  Статистика по всем {mask.sum():,} значениям:")
        print(f"    Среднее расхождение : {mean_diff:.8f}")
        print(f"    99-й перцентиль     : {p99:.8f}")
        print(f"    Максимум            : {overall_max:.8f}")

        all_results.append({
            "window": ws, "stride": st, "distance": dist,
            "shape_ok": True, "max_diff": overall_max,
            "mean_diff": mean_diff, "p99": p99,
            "t_cpu": t_cpu, "t_torch": t_torch,
        })

    # ====================================================================
    # 4. СВОДНАЯ ТАБЛИЦА
    # ====================================================================
    print(f"\n\n{'=' * 78}")
    print("  СВОДНАЯ ТАБЛИЦА ВСЕХ КОМБИНАЦИЙ")
    print(f"{'=' * 78}")
    print(f"  {'Окно':>5} | {'Шаг':>4} | {'Дист':>5} | {'Размерность':>12} | {'Max diff':>12} | {'Среднее':>12} | {'mahotas':>8} | {'torch':>8} | {'Уск.':>5}")
    print("  " + "-" * 95)

    for r in all_results:
        if r["shape_ok"]:
            speedup = r["t_cpu"] / r["t_torch"] if r["t_torch"] > 0 else float("inf")
            print(
                f"  {r['window']:5d} | {r['stride']:4d} | {r['distance']:5d} | "
                f"{'OK':>12} | {r['max_diff']:>12.6f} | {r['mean_diff']:>12.8f} | "
                f"{r['t_cpu']:7.3f}s | {r['t_torch']:7.3f}s | {speedup:4.1f}x"
            )
        else:
            print(
                f"  {r['window']:5d} | {r['stride']:4d} | {r['distance']:5d} | "
                f"{'FAIL':>12} | {'N/A':>12} | {'N/A':>12} | "
                f"{r['t_cpu']:7.3f}s | {r['t_torch']:7.3f}s |    N/A"
            )

    print("  " + "-" * 95)

    # --- Итог ---
    valid = [r for r in all_results if r["shape_ok"]]
    if valid:
        global_max = max(r["max_diff"] for r in valid)
        global_mean = max(r["mean_diff"] for r in valid)
        print(f"\n  ГЛОБАЛЬНЫЙ МАКСИМУМ расхождения: {global_max:.8f}")
        print(f"  ГЛОБАЛЬНОЕ СРЕДНЕЕ расхождение: {global_mean:.8f}")
        if global_max < 1e-4:
            print(f"\n  [+] ВСЕ КОМБИНАЦИИ СОВПАДАЮТ с mahotas (max < 1e-4)")
        else:
            print(f"\n  [!] ЕСТЬ РАСХОЖДЕНИЯ > 1e-4 — требуется анализ")

    print(f"\n{'=' * 78}")