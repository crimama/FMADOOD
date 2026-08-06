from __future__ import annotations

import numpy as np
import pytest

from flow_tte.semantic_atlas_metric import (
    SemanticAtlasConfig,
    assign_clusters,
    basis_diagnostics,
    fit_global_rank_metric,
    fit_semantic_atlas,
    fit_unorthogonalized_atlas,
    hard_neighbor_retention,
    loo_distances,
    pair_distance_spearman,
    projected_energy_ratio,
    random_atlas_like,
    semantic_atlas_1nn,
    shuffled_atlas,
)


def _toy_support(
    *,
    rows_per_cluster: int = 48,
    dim: int = 6,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    centers = np.asarray(
        [
            [-3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    clean_rows = []
    shifted_rows = []
    sources = []
    shift_dirs = (
        np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    )
    for cluster, center in enumerate(centers):
        for row in range(rows_per_cluster):
            source = row % 8
            semantic = rng.normal(0.0, 0.20, size=dim).astype(np.float32)
            semantic[2 + cluster] *= 0.03
            clean = center + semantic
            shift = np.float32(0.35) * shift_dirs[cluster]
            clean_rows.append(clean)
            shifted_rows.append(clean + shift)
            sources.append(f"img_{source:02d}")
    return np.stack(clean_rows), np.stack(shifted_rows), sources


def test_atlas_fit_is_deterministic_and_valid() -> None:
    clean, shifted, sources = _toy_support()
    config = SemanticAtlasConfig(
        n_clusters=2,
        min_cluster_rows=24,
        min_cluster_sources=4,
        semantic_tangent_rank=2,
        seed=123,
    )

    first = fit_semantic_atlas(clean, shifted, sources, config)
    second = fit_semantic_atlas(clean, shifted, sources, config)

    assert np.allclose(first.centers, second.centers)
    assert [basis.valid for basis in first.bases] == [True, True]
    assert first.valid_cluster_count == 2
    for left, right in zip(first.bases, second.bases):
        assert np.allclose(left.shift_basis, right.shift_basis)
        assert left.reason == "valid"
    diagnostics = basis_diagnostics(first, row_labels=assign_clusters(clean, first.centers))
    assert diagnostics["finite"] is True
    assert diagnostics["min_metric_eigenvalue"] == pytest.approx(0.75)
    assert diagnostics["max_orthonormal_error"] < 1e-5
    assert diagnostics["disabled_patch_mass"] == pytest.approx(0.0)


def test_memory_conditioned_distance_shrinks_only_cluster_shift_axis() -> None:
    clean, shifted, sources = _toy_support()
    config = SemanticAtlasConfig(
        n_clusters=2,
        beta=0.25,
        min_cluster_rows=24,
        min_cluster_sources=4,
        semantic_tangent_rank=0,
        seed=123,
    )
    atlas = fit_semantic_atlas(clean, shifted, sources, config)

    baseline_dist, _ = semantic_atlas_1nn(shifted, clean, None)
    atlas_dist, _ = semantic_atlas_1nn(shifted, clean, atlas)

    assert float(atlas_dist.mean()) < float(baseline_dist.mean())
    assert float(atlas_dist.mean()) == pytest.approx(0.75 * float(baseline_dist.mean()), rel=0.08)
    assert loo_distances(clean, atlas).shape == (clean.shape[0],)


def test_orthogonalization_disables_cluster_when_shift_is_semantic_tangent() -> None:
    rng = np.random.default_rng(9)
    clean = rng.normal(size=(80, 4)).astype(np.float32)
    clean[:, 0] = np.linspace(-2.0, 2.0, clean.shape[0], dtype=np.float32)
    clean[:, 1:] *= 0.02
    shifted = clean.copy()
    shifted[:, 0] += np.float32(0.20)
    sources = [f"img_{idx % 8}" for idx in range(clean.shape[0])]
    config = SemanticAtlasConfig(
        n_clusters=1,
        min_cluster_rows=32,
        min_cluster_sources=4,
        semantic_tangent_rank=1,
        orthogonal_residual_min_ratio=0.25,
        seed=4,
    )

    atlas = fit_semantic_atlas(clean, shifted, sources, config)
    raw = fit_unorthogonalized_atlas(clean, shifted, sources, config)

    assert atlas.bases[0].valid is False
    assert atlas.bases[0].reason == "orthogonal_residual_too_small"
    assert raw.bases[0].valid is True
    atlas_dist, _ = semantic_atlas_1nn(shifted, clean, atlas)
    raw_dist, _ = semantic_atlas_1nn(shifted, clean, raw)
    assert float(raw_dist.mean()) < float(atlas_dist.mean())


def test_invalid_small_cluster_falls_back_to_identity_distance() -> None:
    clean, shifted, sources = _toy_support(rows_per_cluster=12)
    config = SemanticAtlasConfig(
        n_clusters=2,
        min_cluster_rows=512,
        min_cluster_sources=6,
        seed=1,
    )
    atlas = fit_semantic_atlas(clean, shifted, sources, config)

    base_dist, base_idx = semantic_atlas_1nn(shifted, clean, None)
    atlas_dist, atlas_idx = semantic_atlas_1nn(shifted, clean, atlas)

    assert [basis.valid for basis in atlas.bases] == [False, False]
    assert np.allclose(atlas_dist, base_dist)
    assert np.array_equal(atlas_idx, base_idx)


def test_global_shuffle_and_random_controls_are_well_formed() -> None:
    clean, shifted, sources = _toy_support()
    config = SemanticAtlasConfig(
        n_clusters=2,
        min_cluster_rows=24,
        min_cluster_sources=4,
        semantic_tangent_rank=0,
        seed=123,
    )
    atlas = fit_semantic_atlas(clean, shifted, sources, config)
    global_metric = fit_global_rank_metric(clean, shifted, rank=2, beta=0.25)
    shuffled = shuffled_atlas(atlas, seed=99)
    random = random_atlas_like(atlas, seed=101)

    assert global_metric.config.n_clusters == 1
    assert global_metric.bases[0].shift_basis.shape == (clean.shape[1], 2)
    assert shuffled.bases[0].cluster == 0
    assert not np.allclose(shuffled.bases[0].shift_basis, atlas.bases[0].shift_basis)
    for basis in random.bases:
        if basis.valid:
            assert np.allclose(basis.shift_basis.T @ basis.shift_basis, np.eye(1), atol=1e-5)


def test_energy_and_geometry_helpers() -> None:
    clean, shifted, _ = _toy_support()
    residuals = shifted - clean
    direction = np.zeros((clean.shape[1], 1), dtype=np.float32)
    direction[2, 0] = 1.0

    capture = projected_energy_ratio(residuals[:48], direction)
    retention = hard_neighbor_retention(clean, clean.copy(), k=5)
    spearman = pair_distance_spearman(clean, clean.copy(), seed=11)

    assert capture > 0.95
    assert retention == pytest.approx(1.0)
    assert spearman == pytest.approx(1.0)


def test_validation_rejects_misaligned_inputs() -> None:
    clean, shifted, sources = _toy_support()

    with pytest.raises(ValueError, match="share shape"):
        fit_semantic_atlas(clean, shifted[:-1], sources)
    with pytest.raises(ValueError, match="one entry per row"):
        fit_semantic_atlas(clean, shifted, sources[:-1])
    with pytest.raises(ValueError, match="feature dimensions"):
        semantic_atlas_1nn(clean[:, :3], clean, None)
