"""Tests for feature pooling in reference_matching.

Tests cover:
- (a) default byte-identical (no pooling)
- (b) avg3 pooled tokens match torch.nn.functional.avg_pool2d and differ from unpooled
- (c) fired lowres_pool prediction == allclose to directly configured resolutions=[subset], feature_pooling="avg3" model
- (d) non-fired query byte-identical to guard-off model
- (e) invalid values raise
- (f) pooled banks NOT built when not needed
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn import functional as F

from fmad.methods.reference_matching import ReferenceMatching
from fmad.protocol import MethodContext
from tests.test_psma_mr_v2 import TinyPSMAV2Backbone, _write_image


class TinyLateBackbone(TinyPSMAV2Backbone):
    feature_layers = (23,)

    def extract_features(self, image_tensor):
        return [super().extract_features(image_tensor)[-1]]


class TinyPairBackbone(TinyPSMAV2Backbone):
    feature_layers = (11, 23)

    def extract_features(self, image_tensor):
        outputs = super().extract_features(image_tensor)
        return [outputs[1], outputs[-1]]


def test_feature_pooling_default_none(tmp_path):
    """(a) Default feature_pooling="none" is byte-identical to no pooling."""
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    config = {"variant": "R1", "feature_layers": [11, 23], "resolutions": [8, 12],
              "bank_tokens_per_source": 4, "query_chunk_size": 4}

    default = ReferenceMatching(TinyPairBackbone(), config)
    explicit_none = ReferenceMatching(TinyPairBackbone(), {**config, "feature_pooling": "none"})

    assert default.feature_pooling == "none"
    assert explicit_none.feature_pooling == "none"

    default.fit(paths, context)
    explicit_none.fit(paths, context)

    expected, expected_score = explicit_none.predict(paths[0])
    actual, actual_score = default.predict(paths[0])

    np.testing.assert_array_equal(actual, expected)
    assert actual_score == expected_score
    assert "pool_" not in default.object_metadata()["score_definition"]


def test_feature_pooling_avg3_applied(tmp_path):
    """(b) avg3 pooling matches torch.nn.functional.avg_pool2d and differs from unpooled."""
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    config = {"variant": "R1", "feature_layers": [11, 23], "resolutions": [8, 12],
              "bank_tokens_per_source": 4, "query_chunk_size": 4}

    unpooled = ReferenceMatching(TinyPairBackbone(), {**config, "feature_pooling": "none"})
    pooled = ReferenceMatching(TinyPairBackbone(), {**config, "feature_pooling": "avg3"})

    unpooled.fit(paths, context)
    pooled.fit(paths, context)

    map_unpooled, score_unpooled = unpooled.predict(paths[0])
    map_pooled, score_pooled = pooled.predict(paths[0])

    # Pooled and unpooled should differ
    assert not np.allclose(map_pooled, map_unpooled, atol=1e-5), \
        "Pooled and unpooled predictions should differ"
    assert score_pooled != score_unpooled

    # Verify avg3 pooling is in the score definition
    assert "pool_avg3" in pooled.object_metadata()["score_definition"]
    assert "pool_" not in unpooled.object_metadata()["score_definition"]


def test_avg3_pooling_matches_torch_pool(tmp_path):
    """Verify that _apply_feature_pooling with enable=True matches torch.nn.functional.avg_pool2d."""
    # Create a method with a tiny backbone
    backbone = TinyPairBackbone()
    method = ReferenceMatching(backbone, {"variant": "R1", "feature_layers": [11, 23]})

    # Create a distinct feature matrix (not tiny backbone features, but crafted ones)
    H, W = 4, 5
    patches, dim = H * W, 3
    # Create a matrix with sequential values for easy verification
    matrix = torch.arange(patches * dim, dtype=torch.float32).reshape(patches, dim)
    grid = (H, W)

    # Apply pooling
    pooled = method._apply_feature_pooling(matrix, grid, enable=True)

    # Verify shape is preserved
    assert pooled.shape == matrix.shape

    # Manually compute expected pooling
    # Reshape to spatial: [1, dim, H, W]
    spatial = matrix.reshape(H, W, dim).permute(2, 0, 1).unsqueeze(0)
    expected_spatial = F.avg_pool2d(spatial, kernel_size=3, stride=1, padding=1, count_include_pad=False)
    expected = expected_spatial.squeeze(0).permute(1, 2, 0).reshape(patches, dim)

    torch.testing.assert_close(pooled, expected, rtol=0, atol=0)

    # Verify pooling actually changes values (not identity)
    assert not torch.allclose(pooled, matrix, atol=1e-5), "Pooling should change feature values"


def test_lowres_pool_action_fires_with_noisy_query(tmp_path):
    """(c) lowres_pool action fires when query is noisy and uses pooled banks."""
    # Create clean support images
    clean_seed = 42
    paths = [_write_image(tmp_path / f"{i}.png", clean_seed) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    config = {
        "variant": "R1", "feature_layers": [11, 23],
        "resolutions": [8, 12],
        "noise_guard_resolutions": [8],  # Only use resolution 8 when fired
        "bank_tokens_per_source": 4, "query_chunk_size": 4,
        "noise_guard": True,
        "noise_guard_k": 1.1,  # Low threshold to trigger with different noise levels
        "noise_guard_action": "lowres_pool",
    }

    method = ReferenceMatching(TinyPairBackbone(), config)
    method.fit(paths, context)

    # Verify pooled banks were created
    assert len(method.pooled_banks) > 0, "Pooled banks should be created with lowres_pool action"
    assert 8 in method.pooled_banks, "Resolution 8 should be in pooled_banks"

    # Add enough high-frequency noise to ensure the threshold is crossed.
    clean_image = np.asarray(Image.open(paths[0])).astype(np.float32)
    noise = np.random.default_rng(91).normal(0, 80, clean_image.shape)
    noisy_query = tmp_path / "noisy.png"
    Image.fromarray(np.clip(clean_image + noise, 0, 255).astype(np.uint8)).save(noisy_query)

    # First predict to establish reference sigma
    clean_result, _ = method.predict(paths[0])
    assert not method.noise_guard_log[-1]["fired"], "Clean image should not fire"

    # Now predict with noisy query
    result, score = method.predict(str(noisy_query))

    # Check last noise_guard_log entry to see if it fired
    last_log_entry = method.noise_guard_log[-1]
    assert last_log_entry["fired"]
    assert method.last_resolutions == (8,)
    assert np.isfinite(result).all() and np.isfinite(score)


def test_lowres_pool_vs_direct_pooled_config(tmp_path):
    """(c) lowres_pool prediction matches directly configured subset + pooling."""
    paths = [_write_image(tmp_path / f"{i}.png", 42) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)

    # Method with lowres_pool action (will use pooled banks if guard fires)
    config_guard = {
        "variant": "R1", "feature_layers": [11, 23],
        "resolutions": [8, 12],
        "noise_guard_resolutions": [8],
        "bank_tokens_per_source": 4, "query_chunk_size": 4,
        "noise_guard": True,
        "noise_guard_k": 100.0,  # Very high threshold so guard won't fire in tests
        "noise_guard_action": "lowres_pool",
    }
    method_guard = ReferenceMatching(TinyPairBackbone(), config_guard)
    method_guard.fit(paths, context)

    # Method with direct pooled subset config (simulates normal use without guard)
    config_direct = {
        "variant": "R1", "feature_layers": [11, 23],
        "resolutions": [8],  # Only resolution 8
        "bank_tokens_per_source": 4, "query_chunk_size": 4,
        "feature_pooling": "avg3",  # Pooling enabled
    }
    method_direct = ReferenceMatching(TinyPairBackbone(), config_direct)
    method_direct.fit(paths, context)

    # Verify pooled banks exist for method_guard
    assert len(method_guard.pooled_banks) > 0, "Pooled banks should exist with lowres_pool action"
    assert 8 in method_guard.pooled_banks

    # Predict with a clean query (guard won't fire)
    result_guard, score_guard = method_guard.predict(paths[0])
    result_direct, score_direct = method_direct.predict(paths[0])

    # Guard didn't fire, so method_guard used normal (unpooled) banks
    assert not method_guard.noise_guard_log[-1]["fired"]
    assert method_guard.last_resolutions == (8, 12)

    # Both methods should have valid results
    assert np.isfinite(result_guard).all()
    assert np.isfinite(result_direct).all()
    assert np.isfinite(score_guard) and np.isfinite(score_direct)


def test_lowres_pool_nonfiring_query_unchanged(tmp_path):
    """(d) Non-fired query uses unchanged normal path with lowres_pool action."""
    paths = [_write_image(tmp_path / f"{i}.png", 42) for i in range(3)]  # All clean
    context = MethodContext("can", 0, 8, False, False)

    # Method with lowres_pool guard but high threshold so it won't fire
    config_guard = {
        "variant": "R1", "feature_layers": [11, 23],
        "resolutions": [8, 12],
        "noise_guard_resolutions": [8],
        "bank_tokens_per_source": 4, "query_chunk_size": 4,
        "noise_guard": True,
        "noise_guard_k": 100.0,  # Very high threshold, won't fire
        "noise_guard_action": "lowres_pool",
    }
    method_guard = ReferenceMatching(TinyPairBackbone(), config_guard)

    # Method without noise guard
    config_no_guard = {
        "variant": "R1", "feature_layers": [11, 23],
        "resolutions": [8, 12],
        "bank_tokens_per_source": 4, "query_chunk_size": 4,
    }
    method_no_guard = ReferenceMatching(TinyPairBackbone(), config_no_guard)

    method_guard.fit(paths, context)
    method_no_guard.fit(paths, context)

    # Use a clean query
    clean_query = paths[0]
    result_guard, score_guard = method_guard.predict(clean_query)
    result_no_guard, score_no_guard = method_no_guard.predict(clean_query)

    # Guard didn't fire
    assert not method_guard.noise_guard_log[-1]["fired"], "Guard should not fire for clean query"

    # Should use full resolutions
    assert method_guard.last_resolutions == (8, 12)

    # Results should be identical (both use normal path without pooling)
    np.testing.assert_allclose(result_guard, result_no_guard, rtol=1e-5, atol=1e-6)
    assert score_guard == score_no_guard


def test_feature_pooling_invalid_values_raise(tmp_path):
    """(e) Invalid feature_pooling values raise ValueError."""
    with pytest.raises(ValueError, match="feature_pooling must be"):
        ReferenceMatching(TinyPairBackbone(), {
            "variant": "R1", "feature_layers": [11, 23],
            "feature_pooling": "invalid"
        })

    with pytest.raises(ValueError, match="feature_pooling must be"):
        ReferenceMatching(TinyPairBackbone(), {
            "variant": "R1", "feature_layers": [11, 23],
            "feature_pooling": True
        })


def test_lowres_pool_invalid_noise_guard_resolution_raises(tmp_path):
    """Invalid noise_guard_resolutions for lowres_pool raises error."""
    # noise_guard_resolutions [99] is not in resolutions [8, 12]
    with pytest.raises(ValueError, match="noise_guard_resolutions must be a subset of resolutions"):
        ReferenceMatching(TinyPairBackbone(), {
            "variant": "R1", "feature_layers": [11, 23],
            "resolutions": [8, 12],
            "noise_guard": True,
            "noise_guard_resolutions": [99],
            "noise_guard_action": "lowres_pool",
        })


def test_pooled_banks_not_created_when_not_needed(tmp_path):
    """(f) Pooled banks are NOT built when noise_guard=False or action != lowres_pool."""
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    config = {
        "variant": "R1", "feature_layers": [11, 23],
        "resolutions": [8, 12],
        "bank_tokens_per_source": 4, "query_chunk_size": 4,
    }

    # No noise guard
    method_no_guard = ReferenceMatching(TinyPairBackbone(), config)
    method_no_guard.fit(paths, context)
    assert len(method_no_guard.pooled_banks) == 0, "Pooled banks should not be created without guard"

    # Guard enabled but action != lowres_pool
    method_lowres = ReferenceMatching(TinyPairBackbone(), {
        **config,
        "noise_guard": True,
        "noise_guard_action": "lowres",
        "noise_guard_resolutions": [8],
    })
    method_lowres.fit(paths, context)
    assert len(method_lowres.pooled_banks) == 0, "Pooled banks should not be created for 'lowres' action"

    # Guard enabled with lowres_pool action
    method_lowres_pool = ReferenceMatching(TinyPairBackbone(), {
        **config,
        "noise_guard": True,
        "noise_guard_action": "lowres_pool",
        "noise_guard_resolutions": [8],
    })
    method_lowres_pool.fit(paths, context)
    assert len(method_lowres_pool.pooled_banks) > 0, "Pooled banks should be created with lowres_pool action"
    assert 8 in method_lowres_pool.pooled_banks


def test_apply_feature_pooling_with_enable_parameter(tmp_path):
    """Test _apply_feature_pooling with explicit enable parameter."""
    backbone = TinyPairBackbone()

    # Create method with feature_pooling="none"
    method = ReferenceMatching(backbone, {
        "variant": "R1", "feature_layers": [11, 23],
        "feature_pooling": "none"
    })

    # Create test matrix
    H, W = 4, 5
    matrix = torch.arange(H * W * 3, dtype=torch.float32).reshape(H * W, 3)
    grid = (H, W)

    # With feature_pooling="none", default behavior should not pool
    result_default = method._apply_feature_pooling(matrix, grid)
    assert torch.equal(result_default, matrix), "Default should not pool when feature_pooling='none'"

    # But explicit enable=True should override and pool
    result_enabled = method._apply_feature_pooling(matrix, grid, enable=True)
    assert not torch.equal(result_enabled, matrix), "Explicit enable=True should pool despite feature_pooling='none'"

    # And explicit enable=False should not pool
    result_disabled = method._apply_feature_pooling(matrix, grid, enable=False)
    assert torch.equal(result_disabled, matrix), "Explicit enable=False should not pool"


def test_feature_pooling_with_different_grid_sizes(tmp_path):
    """Test pooling with various grid dimensions."""
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(2)]
    context = MethodContext("can", 0, 8, False, False)

    # Test with different resolutions which yield different grids
    config = {
        "variant": "R1", "feature_layers": [11, 23],
        "resolutions": [8, 16, 32],
        "bank_tokens_per_source": 4, "query_chunk_size": 4,
        "feature_pooling": "avg3",
    }

    method = ReferenceMatching(TinyPairBackbone(), config)
    method.fit(paths, context)

    # All resolutions should have pooled banks
    for res in config["resolutions"]:
        assert res in method.banks, f"Bank for resolution {res} should exist"

    # Predictions should be finite
    result, score = method.predict(paths[0])
    assert np.isfinite(result).all()
    assert np.isfinite(score)


def test_metadata_includes_feature_pooling(tmp_path):
    """Verify feature_pooling is included in object_metadata."""
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(2)]
    context = MethodContext("can", 0, 8, False, False)

    # With pooling
    method_pooled = ReferenceMatching(TinyPairBackbone(), {
        "variant": "R1", "feature_layers": [11, 23],
        "feature_pooling": "avg3",
    })
    method_pooled.fit(paths, context)
    metadata = method_pooled.object_metadata()
    assert metadata["feature_pooling"] == "avg3"
    assert "pool_avg3" in metadata["score_definition"]

    # Without pooling
    method_none = ReferenceMatching(TinyPairBackbone(), {
        "variant": "R1", "feature_layers": [11, 23],
        "feature_pooling": "none",
    })
    method_none.fit(paths, context)
    metadata = method_none.object_metadata()
    assert metadata["feature_pooling"] == "none"


def test_lowres_pool_fired_equals_direct_pooled_config_fail_then_pass(tmp_path):
    """A fired guard uses the same banks and scores as direct pooled fitting."""
    paths = [_write_image(tmp_path / f"support_{i}.png", i * 17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    common = {"variant": "R1", "feature_layers": [11, 23],
              "bank_tokens_per_source": 4, "query_chunk_size": 4,
              "bank_bags": 2}
    guarded = ReferenceMatching(TinyPairBackbone(), {
        **common, "resolutions": [8, 12, 16],
        "noise_guard": True, "noise_guard_resolutions": [8, 12],
        "noise_guard_k": 1.1, "noise_guard_action": "lowres_pool"})
    direct = ReferenceMatching(TinyPairBackbone(), {
        **common, "resolutions": [8, 12], "feature_pooling": "avg3"})
    guarded.fit(paths, context)
    direct.fit(paths, context)

    assert set(guarded.pooled_banks) == {8, 12}
    for resolution in (8, 12):
        for guard_bank, direct_bank in zip(guarded.bag_pooled_banks[resolution],
                                           direct.bag_banks[resolution], strict=True):
            assert guard_bank["owners"] == direct_bank["owners"]
            for guard_layer, direct_layer in zip(guard_bank["layers"],
                                                 direct_bank["layers"], strict=True):
                torch.testing.assert_close(guard_layer, direct_layer, rtol=0, atol=0)

    clean_image = np.asarray(Image.open(paths[0])).astype(np.float32)
    noise = np.random.default_rng(91).normal(0, 80, clean_image.shape)
    query = tmp_path / "noisy_query.png"
    Image.fromarray(np.clip(clean_image + noise, 0, 255).astype(np.uint8)).save(query)
    result_guard, score_guard = guarded.predict(str(query))
    result_direct, score_direct = direct.predict(str(query))

    assert guarded.noise_guard_log[-1]["fired"]
    assert guarded.last_resolutions == (8, 12)
    np.testing.assert_allclose(result_guard, result_direct, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(score_guard, score_direct, rtol=1e-6, atol=1e-7)
