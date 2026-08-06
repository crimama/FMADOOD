from __future__ import annotations

import numpy as np
import pytest

from flow_tte.mechanism_diagnostics import (
    cosine_displacement,
    derive_logdet,
    dvt_variance_removed,
    effective_rank,
    neighbor_diagnostics,
)


def test_neighbor_diagnostics_tracks_identity_margin_and_position() -> None:
    clean_indices = np.array([[0, 1], [5, 4], [10, 9], [15, 14]])
    shifted_indices = np.array([[0, 1], [6, 5], [10, 11], [12, 13]])
    clean_distances = np.array([[1, 2], [1, 3], [2, 5], [1, 2]], dtype=np.float32)
    shifted_distances = np.array([[1, 2], [2, 4], [2, 3], [1, 1]], dtype=np.float32)
    result = neighbor_diagnostics(
        clean_indices,
        clean_distances,
        shifted_indices,
        shifted_distances,
        patches_per_source=4,
        grid=(2, 2),
        source_count=4,
    )
    assert result.identity_retention == pytest.approx(0.5)
    assert result.clean_margin_mean == pytest.approx(1.75)
    assert result.shifted_margin_mean == pytest.approx(1.0)
    assert result.tie_rate == pytest.approx(0.25)
    assert 0.0 <= result.source_entropy <= 1.0
    assert 0.0 <= result.source_gini <= 1.0
    assert 0.0 <= result.normalized_position_displacement <= 1.0


def test_derive_logdet_inverts_patch_nll_formula() -> None:
    z = np.array([[1.0, 2.0], [0.5, -0.5]], dtype=np.float32)
    expected = np.array([0.75, -0.25], dtype=np.float32)
    base = 0.5 * np.sum(np.square(z), axis=1) + np.log(2.0 * np.pi)
    nll = base - expected
    np.testing.assert_allclose(derive_logdet(z, nll), expected, atol=1e-6)


def test_effective_rank_and_cosine_displacement_are_bounded() -> None:
    identity = np.eye(4, dtype=np.float32)
    assert effective_rank(identity) == pytest.approx(3.0)
    assert cosine_displacement(identity, identity) == pytest.approx(0.0)
    assert cosine_displacement(identity, -identity) == pytest.approx(2.0)


def test_dvt_variance_removed_uses_matched_sequences() -> None:
    raw = [np.zeros((2, 2, 2), dtype=np.float32), np.ones((2, 2, 2), dtype=np.float32)]
    denoised = [np.full((2, 2, 2), 0.5, dtype=np.float32)] * 2
    assert dvt_variance_removed(raw, denoised) == pytest.approx(1.0)


def test_neighbor_diagnostics_rejects_inconsistent_bank_shape() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        neighbor_diagnostics(
            [[8, 0]],
            [[1.0, 2.0]],
            [[0, 1]],
            [[1.0, 2.0]],
            patches_per_source=4,
            grid=(2, 2),
            source_count=2,
        )
