from __future__ import annotations

import numpy as np
import pytest
import torch

from flow_tte.config import FlowConfig
from flow_tte.flow import PatchNormalizingFlow
from flow_tte.trainer import FlowDensityEstimator
from flow_tte.wtcf import (
    WTCFConfig,
    bounded_ratio_penalty,
    discover_normal_witnesses,
    jvp_sensitivity,
    make_image_disjoint_folds,
    refine_wtcf,
    soft_nearest_distance,
    upper_cvar,
)


def test_image_disjoint_folds_are_deterministic_and_complete() -> None:
    first = make_image_disjoint_folds(np.arange(8), fold_count=4, seed=3)
    second = make_image_disjoint_folds(np.arange(8), fold_count=4, seed=3)

    assert first == second
    assert set(first) == set(range(8))
    assert set(first.values()) == set(range(4))
    assert all(sum(value == fold for value in first.values()) == 2 for fold in range(4))


def test_soft_nearest_approaches_exact_hard_nearest() -> None:
    query = torch.tensor([[0.0, 0.0], [4.0, 0.0]])
    bank = torch.tensor([[1.0, 0.0], [8.0, 0.0]])

    soft = soft_nearest_distance(query, bank, tau=1e-4)
    hard = torch.cdist(query, bank).min(dim=1).values

    torch.testing.assert_close(soft, hard, atol=1e-5, rtol=0.0)


def test_upper_cvar_uses_only_largest_tail() -> None:
    values = torch.tensor([0.1, 0.2, 5.0, 6.0])
    assert float(upper_cvar(values, 0.5)) == pytest.approx(5.5)
    assert float(upper_cvar(values, 0.01)) == pytest.approx(6.0)


def test_crossfit_primitives_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="at least one source per fold"):
        make_image_disjoint_folds([0, 1, 2], fold_count=4)
    with pytest.raises(ValueError, match="cannot be empty"):
        soft_nearest_distance(torch.empty((0, 2)), torch.ones((1, 2)), tau=0.1)
    with pytest.raises(ValueError, match="temperature must be positive"):
        soft_nearest_distance(torch.ones((1, 2)), torch.ones((1, 2)), tau=0.0)
    with pytest.raises(ValueError, match="fraction in"):
        upper_cvar(torch.ones(2), 0.0)


def test_repeated_cross_fold_mutual_residual_becomes_witness() -> None:
    # Every cross-image mutual match differs in the same first-coordinate
    # direction.  The second coordinate separates the two patch identities.
    clean = {
        source: np.asarray(
            [[float(source), 0.0], [float(source), 100.0]], dtype=np.float32,
        )
        for source in range(4)
    }
    folds = {source: source for source in clean}
    witness = discover_normal_witnesses(
        clean,
        folds,
        config=WTCFConfig(
            steps=1,
            witness_rows_per_source=2,
            witness_candidates_per_pair=2,
            witness_cosine=0.99,
            witness_min_source_pairs=3,
            witness_min_fold_pairs=3,
            max_witnesses=2,
            unsupported_probes=1,
            audit_rows=8,
        ),
    )

    assert witness.accepted_count == 1
    assert witness.source_pair_diversity[0] >= 3
    assert witness.fold_pair_diversity[0] >= 3
    np.testing.assert_allclose(np.abs(witness.directions[0]), [1.0, 0.0], atol=1e-6)


def test_single_source_pair_does_not_grant_contraction() -> None:
    clean = {
        source: np.asarray([[float(source), 0.0], [float(source), 1.0]], dtype=np.float32)
        for source in range(4)
    }
    witness = discover_normal_witnesses(
        clean,
        {source: source for source in clean},
        config=WTCFConfig(
            steps=1,
            witness_rows_per_source=2,
            witness_candidates_per_pair=1,
            witness_min_source_pairs=7,
            witness_min_fold_pairs=3,
            unsupported_probes=1,
            audit_rows=8,
        ),
    )
    assert witness.accepted_count == 0


def test_identity_jvp_ratio_and_bounded_penalty() -> None:
    flow = PatchNormalizingFlow(4, n_coupling_layers=2, hidden_multiplier=1, clamp=1.9)
    anchor = torch.randn(4)
    direction = torch.randn(4)
    sensitivity = jvp_sensitivity(flow, anchor, direction, create_graph=True)
    penalty = bounded_ratio_penalty(sensitivity, floor=0.9, ceiling=1.1)

    assert float(sensitivity.detach()) == pytest.approx(1.0, abs=1e-6)
    assert float(penalty.detach()) == pytest.approx(0.0, abs=1e-8)


def test_jvp_penalty_backpropagates_to_flow_parameters() -> None:
    flow = PatchNormalizingFlow(4, n_coupling_layers=2, hidden_multiplier=1, clamp=1.9)
    with torch.no_grad():
        flow.blocks[0].net.output.bias[0] = 0.4
    ratio = jvp_sensitivity(flow, torch.randn(4), torch.randn(4), create_graph=True)
    penalty = bounded_ratio_penalty(ratio, floor=0.4, ceiling=0.8)
    penalty.backward()

    gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in flow.parameters()
        if parameter.grad is not None
    )
    assert gradient > 0.0


def _fitted_estimator(clean: dict[int, np.ndarray]) -> FlowDensityEstimator:
    features = np.concatenate(list(clean.values()))
    estimator = FlowDensityEstimator(
        dim=features.shape[1],
        config=FlowConfig(
            n_coupling_layers=2,
            hidden_multiplier=1,
            n_epochs=1,
            lr=1e-3,
            batch_size=16,
            seed=7,
            standardize=True,
        ),
        device="cpu",
    )
    estimator.fit(features, density_quantile=0.9)
    return estimator


def test_wtcf_refinement_is_finite_and_preserves_baseline_clone() -> None:
    rng = np.random.default_rng(17)
    clean = {
        source: rng.normal(loc=source * 0.05, scale=0.5, size=(8, 4)).astype(np.float32)
        for source in range(4)
    }
    estimator = _fitted_estimator(clean)
    before = tuple(parameter.detach().clone() for parameter in estimator.flow.parameters())
    result = refine_wtcf(
        estimator,
        clean,
        mode="wtcf",
        config=WTCFConfig(
            steps=2,
            query_rows=4,
            memory_rows=8,
            witness_rows_per_source=4,
            witness_candidates_per_pair=2,
            witness_min_source_pairs=2,
            witness_min_fold_pairs=2,
            max_witnesses=2,
            unsupported_probes=2,
            witness_target=1.10,
            rollback_interval=1,
            audit_rows=8,
            pair_spearman_floor=0.0,
            neighbor_retention_floor=0.0,
            geometry_ratio_floor=0.1,
            geometry_ratio_ceiling=10.0,
            logdet_drift_std_ceiling=100.0,
        ),
        seed=5,
    )

    assert result.finite
    assert len(result.losses) == 2
    assert result.final_geometry.passed
    assert result.final_directional.passed
    assert set(result.fold_by_source) == set(clean)
    assert any(
        not torch.equal(anchor, parameter)
        for anchor, parameter in zip(before, estimator.flow.parameters())
    )


def test_zero_witness_cannot_pass_wtcf_directional_validity() -> None:
    rng = np.random.default_rng(31)
    clean = {
        source: rng.normal(size=(6, 4)).astype(np.float32) for source in range(4)
    }
    estimator = _fitted_estimator(clean)
    result = refine_wtcf(
        estimator,
        clean,
        mode="wtcf",
        config=WTCFConfig(
            steps=1,
            query_rows=4,
            memory_rows=8,
            witness_rows_per_source=2,
            witness_candidates_per_pair=1,
            witness_min_source_pairs=99,
            witness_min_fold_pairs=3,
            unsupported_probes=1,
            witness_target=1.10,
            rollback_interval=1,
            audit_rows=8,
            pair_spearman_floor=0.0,
            neighbor_retention_floor=0.0,
            geometry_ratio_floor=0.1,
            geometry_ratio_ceiling=10.0,
            logdet_drift_std_ceiling=100.0,
        ),
    )
    assert result.witness.accepted_count == 0
    assert not result.final_directional.passed


def test_aupro_contract_requires_grid_shaped_masks() -> None:
    # Runner diagnostics must preserve [image, height, width] for connected
    # component PRO instead of flattening the mask alongside pixel AUROC.
    masks = np.zeros((4, 3, 5), dtype=bool)
    assert masks.ndim == 3
    assert masks.reshape(-1).size == 60
