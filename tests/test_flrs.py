from __future__ import annotations

import numpy as np
import pytest

from flow_tte.flrs import (
    combine_layer_scores,
    fit_robust_envelope,
    fit_signature_envelope,
    invert_permuted_map,
    local_relation_signature,
    normalize_feature_map,
    permute_feature_grid,
    radius1_relation_signature,
    random_neighbor_signature,
    score_robust_energy,
)


def feature_grid(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(7, 9, 6)).astype(np.float32)


def test_relation_signature_is_finite_and_fixed_width() -> None:
    signature = local_relation_signature(feature_grid())
    assert signature.shape == (7, 9, 4)
    assert np.all(np.isfinite(signature))
    assert radius1_relation_signature(feature_grid()).shape == (7, 9, 3)
    assert random_neighbor_signature(feature_grid()).shape == (7, 9, 4)


def test_relation_signature_is_invariant_to_channel_orthogonal_transform() -> None:
    features = feature_grid()
    orthogonal, _ = np.linalg.qr(
        np.random.default_rng(2).normal(size=(features.shape[-1], features.shape[-1])),
    )
    transformed = features @ orthogonal.astype(np.float32)
    np.testing.assert_allclose(
        local_relation_signature(features),
        local_relation_signature(transformed),
        rtol=1e-5,
        atol=1e-5,
    )


def test_robust_envelope_scores_center_at_zero() -> None:
    values = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    envelope = fit_robust_envelope(values)
    score = score_robust_energy(np.array([[0.0, 1.0]], dtype=np.float32), envelope)
    np.testing.assert_allclose(score, 0.0)
    assert np.all(envelope.scale >= 1e-3)


def test_signature_envelope_accepts_different_grid_shapes() -> None:
    first = local_relation_signature(feature_grid())
    second = local_relation_signature(feature_grid()[:5, :6])
    envelope = fit_signature_envelope((first, second))
    assert envelope.center.shape == (4,)
    assert envelope.scale.shape == (4,)


def test_spatial_permutation_round_trip_for_scores() -> None:
    features = feature_grid()
    permuted, inverse = permute_feature_grid(features, seed=7)
    scores = np.arange(features.shape[0] * features.shape[1], dtype=np.float32).reshape(
        features.shape[:2],
    )
    permutation_scores = scores.reshape(-1)[
        np.random.default_rng(7).permutation(scores.size)
    ].reshape(scores.shape)
    np.testing.assert_array_equal(invert_permuted_map(permutation_scores, inverse), scores)
    assert permuted.shape == features.shape


def test_layer_score_combination_rejects_shape_mismatch() -> None:
    first = np.ones((3, 4), dtype=np.float32)
    second = np.zeros((3, 4), dtype=np.float32)
    np.testing.assert_array_equal(combine_layer_scores((first, second)), 0.5)
    with pytest.raises(ValueError, match="same shape"):
        combine_layer_scores((first, np.zeros((4, 3), dtype=np.float32)))


def test_normalization_rejects_zero_direction() -> None:
    with pytest.raises(ValueError, match="zero direction"):
        normalize_feature_map(np.zeros((2, 3, 4), dtype=np.float32))
