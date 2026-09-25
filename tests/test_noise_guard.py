from __future__ import annotations

import json
import numpy as np
import pytest
import torch
from pathlib import Path
from PIL import Image

from fmad.methods.reference_matching import ReferenceMatching, immerkaer_sigma
from fmad.protocol import MethodContext
from tests.test_reference_matching import TinyLateBackbone, TinyPairBackbone
from tests.test_psma_mr_v2 import _write_image


class TestImmerkaerSigma:
    """Test the Immerkaer noise estimation function."""

    def test_constant_image_has_zero_noise(self):
        """Constant images should have zero noise."""
        image = np.ones((64, 64, 3), dtype=np.uint8) * 128
        sigma = immerkaer_sigma(image)
        assert sigma == 0.0

    def test_scipy_reference_on_random_image(self):
        """Test against scipy.signal.convolve2d reference."""
        from scipy.signal import convolve2d

        # Create a random uint8 RGB image
        np.random.seed(42)
        image = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)

        # Our implementation
        sigma = immerkaer_sigma(image)

        # Reference implementation
        gray = (0.299 * image[..., 0] +
                0.587 * image[..., 1] +
                0.114 * image[..., 2]).astype(np.float64)
        M = np.array([[1, -2, 1],
                      [-2, 4, -2],
                      [1, -2, 1]], dtype=np.float64)
        conv = convolve2d(gray, M, mode='valid')
        h, w = gray.shape
        sigma_ref = np.sqrt(np.pi / 2.0) / (6.0 * (h - 2) * (w - 2)) * np.sum(np.abs(conv))

        # Should match to high precision
        np.testing.assert_allclose(sigma, sigma_ref, rtol=1e-9)

    def test_rejects_non_uint8(self):
        """Should reject non-uint8 input."""
        image_float = np.ones((64, 64, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="must be uint8"):
            immerkaer_sigma(image_float)

    def test_rejects_wrong_shape(self):
        """Should reject non-RGB images."""
        image_2d = np.ones((64, 64), dtype=np.uint8)
        with pytest.raises(ValueError, match="must be HxWx3"):
            immerkaer_sigma(image_2d)


class TestNoiseGuardConfigValidation:
    """Test configuration validation for noise_guard parameters."""

    def test_non_bool_noise_guard_rejected(self):
        """noise_guard must be a bool."""
        with pytest.raises(ValueError, match="noise_guard must be a bool"):
            ReferenceMatching(TinyLateBackbone(), {
                "variant": "R0",
                "noise_guard": "yes"
            })

    def test_noise_guard_k_must_be_gt_1(self):
        """noise_guard_k must be > 1.0."""
        with pytest.raises(ValueError, match="noise_guard_k must be.*> 1.0"):
            ReferenceMatching(TinyLateBackbone(), {
                "variant": "R0",
                "noise_guard": True,
                "noise_guard_k": 0.5
            })
        with pytest.raises(ValueError, match="noise_guard_k must be.*> 1.0"):
            ReferenceMatching(TinyLateBackbone(), {
                "variant": "R0",
                "noise_guard": True,
                "noise_guard_k": 1.0
            })

    def test_invalid_noise_guard_action(self):
        """noise_guard_action must be one of the allowed values."""
        with pytest.raises(ValueError, match="noise_guard_action must be one of"):
            ReferenceMatching(TinyLateBackbone(), {
                "variant": "R0",
                "noise_guard": True,
                "noise_guard_action": "invalid"
            })

    def test_noise_guard_resolutions_not_subset(self):
        """noise_guard_resolutions must be a subset of resolutions."""
        with pytest.raises(ValueError, match="must be a subset"):
            ReferenceMatching(TinyLateBackbone(), {
                "variant": "R0",
                "resolutions": [8, 12],
                "noise_guard": True, "noise_guard_resolutions": [8],
                "noise_guard_resolutions": [16, 24]
            })

    def test_noise_guard_nlm_scale_must_be_positive(self):
        """noise_guard_nlm_scale must be > 0."""
        with pytest.raises(ValueError, match="noise_guard_nlm_scale must be.*> 0"):
            ReferenceMatching(TinyLateBackbone(), {
                "variant": "R0",
                "noise_guard": True,
                "noise_guard_nlm_scale": 0.0
            })


class TestNoiseGuardBehavior:
    """Test the actual noise guard behavior during fit/predict."""

    def test_noise_guard_false_matches_key_absent(self, tmp_path):
        """When noise_guard is False or absent, predictions should be identical."""
        paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
        context = MethodContext("can", 0, 8, False, False)
        config = {"resolutions": [8, 12], "bank_tokens_per_source": 4, "query_chunk_size": 4}

        model_no_key = ReferenceMatching(TinyLateBackbone(), {**config, "variant": "R0"})
        model_explicit_false = ReferenceMatching(TinyLateBackbone(), {
            **config, "variant": "R0", "noise_guard": False
        })

        model_no_key.fit(paths, context)
        model_explicit_false.fit(paths, context)

        result_no_key, score_no_key = model_no_key.predict(paths[0])
        result_explicit_false, score_explicit_false = model_explicit_false.predict(paths[0])

        np.testing.assert_array_equal(result_no_key, result_explicit_false)
        assert score_no_key == score_explicit_false

    def test_clean_query_does_not_fire_noise_guard(self, tmp_path):
        """A clean query should not fire the noise guard."""
        paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
        context = MethodContext("can", 0, 8, False, False)
        config = {
            "resolutions": [8, 12],
            "bank_tokens_per_source": 4,
            "query_chunk_size": 4,
            "noise_guard": True, "noise_guard_resolutions": [8],
            "noise_guard_k": 2.0
        }

        model = ReferenceMatching(TinyLateBackbone(), {**config, "variant": "R0"})
        model.fit(paths, context)

        result, score = model.predict(paths[0])

        assert len(model.noise_guard_log) == 1
        assert model.noise_guard_log[0]["fired"] is False
        assert np.isfinite(result).all()
        assert np.isfinite(score)

    def test_clean_query_output_matches_disabled(self, tmp_path):
        """When noise_guard doesn't fire, output should match noise_guard=False."""
        paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
        context = MethodContext("can", 0, 8, False, False)
        config = {"resolutions": [8, 12], "bank_tokens_per_source": 4, "query_chunk_size": 4}

        model_guard = ReferenceMatching(TinyLateBackbone(), {
            **config, "variant": "R0", "noise_guard": True, "noise_guard_k": 100.0,
            "noise_guard_resolutions": [8]
        })
        model_no_guard = ReferenceMatching(TinyLateBackbone(), {
            **config, "variant": "R0", "noise_guard": False
        })

        model_guard.fit(paths, context)
        model_no_guard.fit(paths, context)

        result_guard, score_guard = model_guard.predict(paths[0])
        result_no_guard, score_no_guard = model_no_guard.predict(paths[0])

        np.testing.assert_array_equal(result_guard, result_no_guard)
        assert score_guard == score_no_guard

    def test_zero_reference_noise_handles_clean_and_noisy_queries(self, tmp_path):
        """Zero over zero stays quiet; positive over zero fires the guard."""
        support = []
        for index in range(3):
            path = tmp_path / f"constant_{index}.png"
            Image.fromarray(np.full((12, 16, 3), 128, dtype=np.uint8)).save(path)
            support.append(str(path))
        model = ReferenceMatching(TinyLateBackbone(), {
            "variant": "R0", "resolutions": [8, 12],
            "bank_tokens_per_source": 4, "query_chunk_size": 4,
            "noise_guard": True, "noise_guard_resolutions": [8],
            "noise_guard_action": "lowres", "noise_guard_k": 2.0})
        model.fit(support, MethodContext("can", 0, 8, False, False))
        assert model.noise_reference_sigma == 0.0

        clean_map, clean_score = model.predict(support[0])
        assert model.noise_guard_log[-1]["sigma"] == 0.0
        assert model.noise_guard_log[-1]["ratio"] == 1.0
        assert model.noise_guard_log[-1]["fired"] is False
        assert model.last_resolutions == (8, 12)
        assert np.isfinite(clean_map).all() and np.isfinite(clean_score)

        query = tmp_path / "positive_noise.png"
        Image.fromarray(np.random.default_rng(42).integers(
            0, 256, (12, 16, 3), dtype=np.uint8)).save(query)
        noisy_map, noisy_score = model.predict(str(query))
        assert model.noise_guard_log[-1]["sigma"] > 0.0
        assert model.noise_guard_log[-1]["ratio"] == pytest.approx(
            model.noise_guard_log[-1]["sigma"] / 1e-12)
        assert np.isfinite(model.noise_guard_log[-1]["ratio"])
        assert model.noise_guard_log[-1]["fired"] is True
        assert model.last_resolutions == (8,)
        assert np.isfinite(noisy_map).all() and np.isfinite(noisy_score)
        summary = model.object_metadata()["noise_guard_summary"]
        assert all(np.isfinite(summary[key]) for key in ("ratio_min", "ratio_median", "ratio_max"))
        json.dumps(model.noise_guard_log, allow_nan=False)
        json.dumps(model.object_metadata(), allow_nan=False)

    def test_noisy_query_fires_noise_guard(self, tmp_path):
        """A sufficiently noisy query should fire the noise guard."""
        paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
        context = MethodContext("can", 0, 8, False, False)
        config = {
            "resolutions": [8, 12],
            "bank_tokens_per_source": 4,
            "query_chunk_size": 4,
            "noise_guard": True, "noise_guard_resolutions": [8],
            "noise_guard_k": 1.5
        }

        model = ReferenceMatching(TinyLateBackbone(), {**config, "variant": "R0"})
        model.fit(paths, context)

        clean_path = _write_image(tmp_path / "clean.png", 99)
        clean_image = np.asarray(Image.open(clean_path))
        noisy_image = clean_image.copy()
        np.random.seed(42)
        noise = np.random.normal(0, 30, size=noisy_image.shape)
        noisy_image = np.clip(noisy_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        noisy_path = tmp_path / "noisy.png"
        Image.fromarray(noisy_image).save(noisy_path)

        result, score = model.predict(str(noisy_path))

        assert len(model.noise_guard_log) >= 1
        assert any(entry["fired"] for entry in model.noise_guard_log)
        assert np.isfinite(result).all()
        assert np.isfinite(score)

    def test_lowres_action_uses_subset_of_resolutions(self, tmp_path):
        """The 'lowres' action should use only noise_guard_resolutions."""
        paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
        context = MethodContext("can", 0, 8, False, False)
        config = {
            "resolutions": [8, 12],
            "bank_tokens_per_source": 4,
            "query_chunk_size": 4,
            "noise_guard": True, "noise_guard_resolutions": [8],
            "noise_guard_k": 1.5,
            "noise_guard_action": "lowres",
            "noise_guard_resolutions": [8]
        }

        model_with_guard = ReferenceMatching(TinyLateBackbone(), {**config, "variant": "R0"})
        model_with_guard.fit(paths, context)

        clean_path = _write_image(tmp_path / "clean.png", 99)
        clean_image = np.asarray(Image.open(clean_path))
        noisy_image = clean_image.copy()
        np.random.seed(42)
        noise = np.random.normal(0, 30, size=noisy_image.shape)
        noisy_image = np.clip(noisy_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        noisy_path = tmp_path / "noisy.png"
        Image.fromarray(noisy_image).save(noisy_path)

        result_with_guard, _ = model_with_guard.predict(str(noisy_path))

        assert len(model_with_guard.noise_guard_log) >= 1
        assert any(entry["fired"] for entry in model_with_guard.noise_guard_log)
        assert np.isfinite(result_with_guard).all()

    def test_deep_only_action_with_r1(self, tmp_path):
        """The 'deep_only' action should suppress the shallow residual spike."""
        paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
        context = MethodContext("can", 0, 8, False, False)
        config = {
            "resolutions": [8, 12],
            "bank_tokens_per_source": 4,
            "query_chunk_size": 4,
            "feature_layers": [11, 23],
            "layer_fusion": "residual_spike",
            "layer_fusion_tau": 1.2,
            "layer_fusion_tau_mode": "support_quantile"
        }

        model_with_gamma_0 = ReferenceMatching(TinyPairBackbone(), {
            **config, "variant": "R1",
            "layer_fusion_gamma": 0.0
        })
        model_with_guard = ReferenceMatching(TinyPairBackbone(), {
            **config, "variant": "R1",
            "noise_guard": True,
            "noise_guard_k": 1.5,
            "noise_guard_action": "deep_only"
        })

        model_with_gamma_0.fit(paths, context)
        model_with_guard.fit(paths, context)

        clean_path = _write_image(tmp_path / "clean.png", 99)
        clean_image = np.asarray(Image.open(clean_path))
        noisy_image = clean_image.copy()
        np.random.seed(42)
        noise = np.random.normal(0, 30, size=noisy_image.shape)
        noisy_image = np.clip(noisy_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        noisy_path = tmp_path / "noisy.png"
        Image.fromarray(noisy_image).save(noisy_path)

        result_gamma_0, _ = model_with_gamma_0.predict(str(noisy_path))
        result_with_guard, _ = model_with_guard.predict(str(noisy_path))

        assert model_with_guard.noise_guard_log[-1]["fired"]
        np.testing.assert_allclose(result_with_guard, result_gamma_0, rtol=1e-5, atol=1e-5)
        assert model_with_guard._noise_suppress_shallow is False


def _noisy_query(tmp_path, seed=42, std=30.0):
    clean = np.asarray(Image.open(_write_image(tmp_path / "clean.png", 99)))
    rng = np.random.default_rng(seed)
    noisy = np.clip(clean.astype(np.float32) + rng.normal(0, std, clean.shape), 0, 255).astype(np.uint8)
    path = tmp_path / "noisy.png"
    Image.fromarray(noisy).save(path)
    return str(path)


R1_SPIKE = {"bank_tokens_per_source": 4, "query_chunk_size": 4, "feature_layers": [11, 23],
            "layer_fusion": "residual_spike", "layer_fusion_tau": 1.2,
            "layer_fusion_tau_mode": "support_quantile", "variant": "R1"}


class TestNoiseGuardActionsExact:
    """Fired actions must equal a model configured directly for that remedy."""

    def test_lowres_equals_direct_lowres_model(self, tmp_path):
        paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
        context = MethodContext("can", 0, 8, False, False)
        guarded = ReferenceMatching(TinyPairBackbone(), {
            **R1_SPIKE, "resolutions": [8, 12], "noise_guard": True, "noise_guard_resolutions": [8], "noise_guard_k": 1.5,
            "noise_guard_action": "lowres", "noise_guard_resolutions": [8]})
        direct = ReferenceMatching(TinyPairBackbone(), {**R1_SPIKE, "resolutions": [8]})
        guarded.fit(paths, context)
        direct.fit(paths, context)
        query = _noisy_query(tmp_path)
        result_guarded, _ = guarded.predict(query)
        result_direct, _ = direct.predict(query)
        assert guarded.noise_guard_log[-1]["fired"]
        np.testing.assert_allclose(result_guarded, result_direct, rtol=1e-6, atol=1e-7)

    def test_lowres_deep_only_does_not_denoise(self, tmp_path, monkeypatch):
        import fmad.methods.reference_matching as module
        if module.cv2 is not None:
            def forbidden(*args, **kwargs):
                raise AssertionError("lowres_deep_only must not run NLM denoising")
            monkeypatch.setattr(module.cv2, "fastNlMeansDenoisingColored", forbidden)
        paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
        context = MethodContext("can", 0, 8, False, False)
        guarded = ReferenceMatching(TinyPairBackbone(), {
            **R1_SPIKE, "resolutions": [8, 12], "noise_guard": True, "noise_guard_resolutions": [8], "noise_guard_k": 1.5,
            "noise_guard_action": "lowres_deep_only", "noise_guard_resolutions": [8]})
        direct = ReferenceMatching(TinyPairBackbone(), {
            **R1_SPIKE, "resolutions": [8], "layer_fusion_gamma": 0.0})
        guarded.fit(paths, context)
        direct.fit(paths, context)
        query = _noisy_query(tmp_path)
        result_guarded, _ = guarded.predict(query)
        result_direct, _ = direct.predict(query)
        assert guarded.noise_guard_log[-1]["fired"]
        np.testing.assert_allclose(result_guarded, result_direct, rtol=1e-5, atol=1e-5)

    def test_denoise_nlm_uses_bgr_and_sigma_strength(self, tmp_path, monkeypatch):
        import fmad.methods.reference_matching as module
        if module.cv2 is None:
            pytest.skip("OpenCV not installed")
        calls = []
        def record(src, dst, h, h_color, template, search):
            calls.append((src.copy(), h, h_color, template, search))
            return src
        monkeypatch.setattr(module.cv2, "fastNlMeansDenoisingColored", record)
        paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
        context = MethodContext("can", 0, 8, False, False)
        guarded = ReferenceMatching(TinyPairBackbone(), {
            **R1_SPIKE, "resolutions": [8, 12], "noise_guard": True, "noise_guard_resolutions": [8], "noise_guard_k": 1.5,
            "noise_guard_action": "denoise_nlm", "noise_guard_nlm_scale": 0.5})
        guarded.fit(paths, context)
        query = _noisy_query(tmp_path)
        guarded.predict(query)
        rgb = np.asarray(Image.open(query).convert("RGB"))
        src, h, h_color, template, search = calls[0]
        np.testing.assert_array_equal(src, rgb[..., ::-1])
        assert h == h_color == pytest.approx(0.5 * guarded.noise_guard_log[-1]["sigma"])
        assert isinstance(h, float) and (template, search) == (7, 21)

    def test_deep_only_through_fusion_factorial_gated(self, tmp_path):
        from fmad.methods.fusion_factorial import FusionFactorial
        from tests.test_fusion_factorial import config as ff_config
        from tests.test_reference_matching import TinyTripleBackbone
        paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
        context = MethodContext("can", 0, 8, False, False)
        guarded = FusionFactorial(TinyTripleBackbone(), {
            **ff_config("D"), "noise_guard": True, "noise_guard_k": 1.5,
            "noise_guard_action": "deep_only"})
        deep = FusionFactorial(TinyTripleBackbone(), ff_config("E"))
        guarded.fit(paths, context)
        deep.fit(paths, context)
        query = _noisy_query(tmp_path)
        result_guarded, _ = guarded.predict(query)
        result_deep, _ = deep.predict(query)
        assert guarded.noise_guard_log[-1]["fired"]
        np.testing.assert_allclose(result_guarded, result_deep, rtol=1e-5, atol=1e-5)

    def test_fusion_factorial_batch_with_fired_lowres(self, tmp_path):
        # Regression: the batch audit required one gate grid per CONFIGURED resolution,
        # so every fired lowres query raised 'incomplete fusion audit'.
        from fmad.methods.fusion_factorial import FusionFactorial
        from tests.test_fusion_factorial import config as ff_config
        from tests.test_reference_matching import TinyTripleBackbone
        paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
        context = MethodContext("can", 0, 8, False, False)
        guarded = FusionFactorial(TinyTripleBackbone(), {
            **ff_config("D"), "noise_guard": True, "noise_guard_k": 1.5,
            "noise_guard_action": "lowres", "noise_guard_resolutions": [8]})
        direct = FusionFactorial(TinyTripleBackbone(), {**ff_config("D"), "resolutions": [8]})
        guarded.fit(paths, context)
        direct.fit(paths, context)
        query = _noisy_query(tmp_path)
        fired, = guarded.predict_batch([query])
        expected, = direct.predict_batch([query])
        assert guarded.noise_guard_log[-1]["fired"]
        assert len(fired.diagnostics["fusion_audit"]["gate_grids"]) == 1
        np.testing.assert_allclose(fired.score_map, expected.score_map, rtol=1e-6, atol=1e-7)
        clean, = guarded.predict_batch([paths[0]])
        assert not guarded.noise_guard_log[-1]["fired"]
        assert len(clean.diagnostics["fusion_audit"]["gate_grids"]) == 2

    def test_default_resolutions_are_448_672(self):
        model = ReferenceMatching(TinyLateBackbone(), {"variant": "R0", "resolutions": [8, 12]})
        assert model.noise_guard_resolutions == (448, 672)


class TestNoiseGuardMetadata:
    """Test metadata generation when noise_guard is enabled/disabled."""

    def test_metadata_unchanged_when_disabled(self, tmp_path):
        """Metadata should be unchanged when noise_guard is False."""
        paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
        context = MethodContext("can", 0, 8, False, False)
        config = {"resolutions": [8, 12], "bank_tokens_per_source": 4, "query_chunk_size": 4}

        model = ReferenceMatching(TinyLateBackbone(), {**config, "variant": "R0"})
        model.fit(paths, context)
        model.predict(paths[0])

        metadata = model.object_metadata()

        assert "noise_guard" not in metadata
        assert "noise_guard_summary" not in metadata
        assert "_noise_guard_" not in metadata["score_definition"]

    def test_metadata_contains_noise_guard_when_enabled(self, tmp_path):
        """Metadata should contain noise_guard info when enabled."""
        paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
        context = MethodContext("can", 0, 8, False, False)
        config = {
            "resolutions": [8, 12],
            "bank_tokens_per_source": 4,
            "query_chunk_size": 4,
            "noise_guard": True, "noise_guard_resolutions": [8],
            "noise_guard_k": 2.0,
            "noise_guard_action": "lowres",
            "noise_guard_resolutions": [8]
        }

        model = ReferenceMatching(TinyLateBackbone(), {**config, "variant": "R0"})
        model.fit(paths, context)
        model.predict(paths[0])

        metadata = model.object_metadata()

        assert "noise_guard" in metadata
        assert metadata["noise_guard"]["k"] == 2.0
        assert metadata["noise_guard"]["action"] == "lowres"
        assert metadata["noise_guard"]["resolutions"] == [8]
        assert "reference_sigma" in metadata["noise_guard"]
        assert "support_sigmas" in metadata["noise_guard"]

        assert "noise_guard_summary" in metadata
        assert metadata["noise_guard_summary"]["queries"] >= 1
        assert "fired" in metadata["noise_guard_summary"]

        assert "_noise_guard_lowres" in metadata["score_definition"]

    def test_noise_guard_summary_statistics(self, tmp_path):
        """noise_guard_summary should have correct min/median/max ratios."""
        paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
        context = MethodContext("can", 0, 8, False, False)
        config = {
            "resolutions": [8, 12],
            "bank_tokens_per_source": 4,
            "query_chunk_size": 4,
            "noise_guard": True, "noise_guard_resolutions": [8],
            "noise_guard_k": 10.0
        }

        model = ReferenceMatching(TinyLateBackbone(), {**config, "variant": "R0"})
        model.fit(paths, context)

        for i in range(3):
            model.predict(paths[i])

        metadata = model.object_metadata()
        summary = metadata["noise_guard_summary"]

        assert summary["queries"] == 3
        ratios = [entry["ratio"] for entry in model.noise_guard_log]
        np.testing.assert_allclose(summary["ratio_min"], np.min(ratios), rtol=1e-9)
        np.testing.assert_allclose(summary["ratio_median"], np.median(ratios), rtol=1e-9)
        np.testing.assert_allclose(summary["ratio_max"], np.max(ratios), rtol=1e-9)
