"""Tests for haralickt package."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from haralickt import HaralickConfig, HaralickEngine, haralick


class TestConfig:
    """Tests for HaralickConfig."""

    def test_default_config(self):
        cfg = HaralickConfig()
        assert cfg.num_levels == 32
        assert cfg.window_size == 32
        assert cfg.stride == 8
        assert cfg.ignore_zeros is True
        assert cfg.tile_size == 1024
        assert cfg.resolved_distance == 4  # max(1, 32 // 8)
        assert cfg.num_features == 13

    def test_custom_config(self):
        cfg = HaralickConfig(num_levels=16, window_size=16, stride=4)
        assert cfg.resolved_distance == 2  # max(1, 16 // 8)

    def test_invalid_num_levels(self):
        with pytest.raises(ValueError, match="num_levels"):
            HaralickConfig(num_levels=1)

    def test_invalid_window_size(self):
        with pytest.raises(ValueError, match="window_size"):
            HaralickConfig(window_size=1)

    def test_invalid_stride(self):
        with pytest.raises(ValueError, match="stride"):
            HaralickConfig(stride=0)

    def test_output_shape(self):
        cfg = HaralickConfig(window_size=32, stride=8)
        shape = cfg.output_shape(256, 256)
        assert shape == (29, 29, 13)


class TestValidation:
    """Tests for input validation."""

    def test_reject_3d_input(self):
        image = torch.randint(0, 32, (4, 64, 64))
        with pytest.raises(ValueError, match="2D"):
            haralick(image, device="cpu")

    def test_reject_too_small_image(self):
        image = torch.randint(0, 32, (16, 16))
        with pytest.raises(ValueError, match="window_size"):
            haralick(image, window_size=32, device="cpu")

    def test_reject_out_of_range_values(self):
        image = torch.randint(0, 64, (64, 64))
        with pytest.raises(ValueError, match="values must be in"):
            haralick(image, num_levels=32, device="cpu")

    def test_reject_non_tensor(self):
        with pytest.raises(TypeError, match="torch.Tensor"):
            haralick(np.zeros((64, 64), dtype=np.uint8), device="cpu")


class TestHaralickBasic:
    """Basic functional tests."""

    def test_output_shape(self):
        image = torch.randint(1, 32, (64, 64))
        features = haralick(image, num_levels=32, window_size=32, stride=8, device="cpu")
        expected_h = (64 - 32) // 8 + 1  # 5
        expected_w = (64 - 32) // 8 + 1  # 5
        assert features.shape == (expected_h, expected_w, 13)

    def test_output_dtype(self):
        image = torch.randint(1, 32, (64, 64))
        features = haralick(image, num_levels=32, window_size=32, stride=8, device="cpu")
        assert features.dtype == torch.float32

    def test_uniform_image_zero_contrast(self):
        """A uniform image should have zero contrast and zero variance."""
        image = torch.full((64, 64), 16, dtype=torch.uint8)
        features = haralick(image, num_levels=32, window_size=32, stride=8, device="cpu")
        # Contrast (index 1) should be ~0
        assert features[:, :, 1].abs().max() < 1e-5
        # Variance (index 3) should be ~0
        assert features[:, :, 3].abs().max() < 1e-5

    def test_ascending_gradient_high_contrast(self):
        """An ascending gradient should have high contrast."""
        x = torch.arange(32, dtype=torch.uint8)
        image = x.unsqueeze(0).repeat(64, 1)  # (64, 32) — horizontal gradient
        image = image[:, :64] if image.shape[1] >= 64 else torch.cat([image, image[:, :64 - image.shape[1]]], dim=1)
        image = image[:64, :64]
        features = haralick(image, num_levels=32, window_size=32, stride=8, device="cpu")
        # Contrast (index 1) should be high
        assert features[:, :, 1].abs().max() > 1.0


class TestHaralickTiling:
    """Tests for tiling behavior."""

    def test_large_image_tiling(self):
        """Large images should use tiling and produce valid output."""
        image = torch.randint(1, 32, (256, 256))
        features = haralick(image, num_levels=32, window_size=32, stride=8, device="cpu")
        expected_h = (256 - 32) // 8 + 1  # 29
        expected_w = (256 - 32) // 8 + 1  # 29
        assert features.shape == (expected_h, expected_w, 13)

    def test_tiling_matches_untiled(self):
        """Tiled computation should match untiled for small images."""
        image = torch.randint(1, 32, (64, 64))
        features_small = haralick(image, num_levels=32, window_size=32, stride=8, device="cpu")
        features_tiled = haralick(
            image, num_levels=32, window_size=32, stride=8, tile_size=32, device="cpu"
        )
        assert torch.allclose(features_small, features_tiled, atol=1e-5)


class TestHaralickEdgeCases:
    """Edge case tests."""

    def test_single_pixel_window(self):
        """Window size equal to image size."""
        image = torch.randint(1, 32, (32, 32))
        features = haralick(image, num_levels=32, window_size=32, stride=1, device="cpu")
        assert features.shape == (1, 1, 13)

    def test_stride_equals_window(self):
        """Stride equal to window size."""
        image = torch.randint(1, 32, (64, 64))
        features = haralick(image, num_levels=32, window_size=32, stride=32, device="cpu")
        expected_h = (64 - 32) // 32 + 1  # 2
        expected_w = (64 - 32) // 32 + 1  # 2
        assert features.shape == (expected_h, expected_w, 13)


class TestHaralickEngine:
    """Tests for HaralickEngine."""

    def test_engine_basic(self):
        engine = HaralickEngine(HaralickConfig(device="cpu"))
        image = torch.randint(1, 32, (64, 64))
        features = engine.compute(image)
        assert features.shape == (5, 5, 13)

    def test_engine_reuse(self):
        engine = HaralickEngine(HaralickConfig(device="cpu"))
        image1 = torch.randint(1, 32, (64, 64))
        image2 = torch.randint(1, 32, (64, 64))
        f1 = engine.compute(image1)
        f2 = engine.compute(image2)
        assert f1.shape == f2.shape


class TestHaralickFromNumpy:
    """Tests for haralick_from_numpy."""

    def test_basic(self):
        from haralickt import haralick_from_numpy

        image = np.random.randint(1, 32, (64, 64), dtype=np.uint8)
        features = haralick_from_numpy(image, num_levels=32, window_size=32, stride=8, device="cpu")
        assert features.shape == (5, 5, 13)
        assert features.dtype == torch.float32

    def test_reject_non_ndarray(self):
        from haralickt import haralick_from_numpy

        with pytest.raises(TypeError, match="numpy.ndarray"):
            haralick_from_numpy(torch.zeros(64, 64), device="cpu")


class TestHaralickConsistency:
    """Consistency tests between different interfaces."""

    def test_api_matches_engine(self):
        image = torch.randint(1, 32, (64, 64))
        f_api = haralick(image, num_levels=32, window_size=32, stride=8, device="cpu")
        engine = HaralickEngine(HaralickConfig(device="cpu"))
        f_engine = engine.compute(image)
        assert torch.allclose(f_api, f_engine, atol=1e-5)

    def test_numpy_matches_torch(self):
        from haralickt import haralick_from_numpy

        image_np = np.random.randint(1, 32, (64, 64), dtype=np.uint8)
        f_numpy = haralick_from_numpy(image_np, num_levels=32, window_size=32, stride=8, device="cpu")
        f_torch = haralick(torch.from_numpy(image_np), num_levels=32, window_size=32, stride=8, device="cpu")
        assert torch.allclose(f_numpy, f_torch, atol=1e-5)


class TestFeaturesMahotasCompat:
    """Tests for the mahotas-compatible features.haralick interface."""

    def test_output_shape(self):
        """Should return (4, 13) array."""
        from haralickt import features

        image = np.random.randint(1, 32, (64, 64), dtype=np.uint8)
        result = features.haralick(image, ignore_zeros=True, distance=1)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 13)

    def test_output_dtype(self):
        """Should return float32 numpy array."""
        from haralickt import features

        image = np.random.randint(1, 32, (64, 64), dtype=np.uint8)
        result = features.haralick(image)
        assert result.dtype == np.float32

    def test_reject_non_ndarray(self):
        """Should reject non-numpy input."""
        from haralickt import features

        with pytest.raises(TypeError, match="numpy.ndarray"):
            features.haralick(torch.zeros(64, 64))

    def test_reject_3d_input(self):
        """Should reject 3D input."""
        from haralickt import features

        with pytest.raises(ValueError, match="2D"):
            features.haralick(np.zeros((4, 64, 64), dtype=np.uint8))

    def test_reject_too_small(self):
        """Should reject images smaller than 2x2."""
        from haralickt import features

        with pytest.raises(ValueError, match="too small"):
            features.haralick(np.zeros((1, 5), dtype=np.uint8))

    def test_reject_invalid_distance(self):
        """Should reject distance < 1."""
        from haralickt import features

        image = np.random.randint(1, 32, (64, 64), dtype=np.uint8)
        with pytest.raises(ValueError, match="distance"):
            features.haralick(image, distance=0)

    def test_uniform_image_zero_contrast(self):
        """Uniform image should have zero contrast and variance."""
        from haralickt import features

        image = np.full((64, 64), 16, dtype=np.uint8)
        result = features.haralick(image, ignore_zeros=True, distance=1)
        # Contrast (index 1) should be ~0
        assert np.abs(result[:, 1]).max() < 1e-5
        # Variance (index 3) should be ~0
        assert np.abs(result[:, 3]).max() < 1e-5

    def test_matches_mahotas(self):
        """Should closely match mahotas.features.haralick output."""
        from haralickt import features

        np.random.seed(42)
        image = np.random.randint(1, 32, (64, 64), dtype=np.uint8)

        our_result = features.haralick(image, ignore_zeros=True, distance=1)

        try:
            import mahotas
        except ImportError:
            pytest.skip("mahotas not installed")

        ref = mahotas.features.haralick(image, ignore_zeros=True, distance=1)

        assert our_result.shape == ref.shape

        # Compare where both are finite
        mask = np.isfinite(our_result) & np.isfinite(ref)
        assert mask.any(), "No finite values to compare"

        for idx in range(13):
            m = mask[:, idx]
            if m.any():
                max_diff = np.max(np.abs(our_result[m, idx] - ref[m, idx]))
                assert max_diff < 1e-3, (
                    f"Feature {idx} max difference {max_diff:.6f} exceeds tolerance 1e-3"
                )

    def test_distance_2(self):
        """Should work with distance=2."""
        from haralickt import features

        image = np.random.randint(1, 32, (64, 64), dtype=np.uint8)
        result = features.haralick(image, ignore_zeros=True, distance=2)
        assert result.shape == (4, 13)
        assert np.isfinite(result).all()

    def test_ignore_zeros_false(self):
        """Should work with ignore_zeros=False."""
        from haralickt import features

        image = np.random.randint(0, 32, (64, 64), dtype=np.uint8)
        result = features.haralick(image, ignore_zeros=False, distance=1)
        assert result.shape == (4, 13)