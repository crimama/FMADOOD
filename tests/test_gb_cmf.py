from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from flow_tte.config import FlowConfig, ScoreConfig
from flow_tte.gb_cmf import (
    GBCMFCell,
    GBCMFConfig,
    audit_flow_geometry,
    audit_proxy_fidelity,
    clone_estimator,
    refine_flow,
    score_null,
    stable_seed,
    train_score_null,
)
from flow_tte.pipeline import FlowTTE
from flow_tte.scoring import ScoreCalibration
from flow_tte.trainer import FlowDensityEstimator


@dataclass(frozen=True)
class _ComponentCell:
    clean: np.ndarray
    shifted_clean: np.ndarray
    defect: np.ndarray
    shifted_defect: np.ndarray
    defect_mask: np.ndarray
    background_mask: np.ndarray


def _fitted_estimator(features: np.ndarray) -> FlowDensityEstimator:
    estimator = FlowDensityEstimator(
        dim=features.shape[1],
        config=FlowConfig(
            n_coupling_layers=2,
            hidden_multiplier=1,
            n_epochs=1,
            lr=1e-3,
            batch_size=8,
            seed=7,
        ),
        device="cpu",
    )
    estimator.fit(features, density_quantile=0.9)
    return estimator


def _factorial_fixture() -> tuple[dict[int, np.ndarray], list[GBCMFCell]]:
    rng = np.random.default_rng(11)
    clean = {
        source: rng.normal(loc=0.1 * source, scale=0.3, size=(8, 4)).astype(np.float32)
        for source in range(3)
    }
    mask = np.asarray([True, True, False, False, False, False, False, False])
    background = ~mask
    cells: list[GBCMFCell] = []
    for source, values in clean.items():
        for axis_index, axis in enumerate(("exposure", "gamma")):
            for family_index, family in enumerate(("cutpaste", "blur")):
                shifted = values + np.float32(0.01 * (axis_index + 1))
                defect = values.copy()
                defect[mask, family_index] += np.float32(0.8)
                shifted_defect = defect + np.float32(0.01 * (axis_index + 1))
                cells.append(
                    GBCMFCell(
                        source_id=source,
                        clean=values,
                        shifted_clean=shifted,
                        defect=defect,
                        shifted_defect=shifted_defect,
                        defect_mask=mask,
                        background_mask=background,
                        axis=axis,
                        family=family,
                    ),
                )
    return clean, cells


def test_clone_estimator_preserves_default_exactly() -> None:
    clean, _ = _factorial_fixture()
    features = np.concatenate(list(clean.values()))
    estimator = _fitted_estimator(features)
    clone = clone_estimator(estimator)

    original = estimator.evaluate(features)
    copied = clone.evaluate(features)
    assert torch.equal(original.z, copied.z)
    assert torch.equal(original.nll, copied.nll)
    assert all(
        torch.equal(left, right)
        for left, right in zip(estimator.flow.parameters(), clone.flow.parameters())
    )


def test_refine_flow_is_finite_and_updates_only_clone() -> None:
    clean, cells = _factorial_fixture()
    features = np.concatenate(list(clean.values()))
    estimator = _fitted_estimator(features)
    candidate = clone_estimator(estimator)
    m0 = estimator.transform(features)
    calibration = ScoreCalibration.fit(m0, ScoreConfig(query_chunk_size=16))
    before = tuple(parameter.detach().clone() for parameter in estimator.flow.parameters())

    result = refine_flow(
        candidate,
        clean,
        cells,
        calibration,
        mode="gb_cmf",
        config=GBCMFConfig(
            steps=4,
            background_count=4,
            interior_cap=2,
            memory_rows=8,
            dual_update_interval=2,
        ),
        seed=5,
    )

    assert result.finite
    assert len(result.losses) == 4
    assert result.max_parameter_delta > 0.0
    assert result.final_tail_constraints
    assert result.final_drop_constraints
    assert set(result.final_tail_duals) == {"exposure", "gamma"}
    assert set(result.final_drop_duals) == {
        "exposure:cutpaste",
        "exposure:blur",
        "gamma:cutpaste",
        "gamma:blur",
    }
    assert all(
        torch.equal(anchor, parameter)
        for anchor, parameter in zip(before, estimator.flow.parameters())
    )


def test_score_nulls_are_deterministic_and_nonconstant() -> None:
    components = np.asarray(
        [[0.0, 0.0], [0.2, 0.1], [1.0, 0.3], [0.8, 0.4]],
        dtype=np.float32,
    )
    mask = np.asarray([False, False, True, True])
    cell = _ComponentCell(components, components + 0.01, components, components + 0.02, mask, ~mask)

    for kind in ("affine", "nonlinear"):
        first = train_score_null([cell], kind=kind, steps=3, learning_rate=1e-3, seed=3)
        second = train_score_null([cell], kind=kind, steps=3, learning_rate=1e-3, seed=3)
        first_scores = score_null(first, components)
        second_scores = score_null(second, components)
        np.testing.assert_array_equal(first_scores, second_scores)
        assert float(np.std(first_scores)) > 0.0


def test_identical_flow_passes_geometry_audit() -> None:
    clean, _ = _factorial_fixture()
    features = np.concatenate(list(clean.values()))
    estimator = _fitted_estimator(features)
    audit = audit_flow_geometry(estimator, clone_estimator(estimator), features, sample_rows=16)

    assert audit.passed
    assert audit.latent_rms_ratio == 1.0
    assert audit.pair_distance_spearman == 1.0
    assert audit.effective_rank_ratio == 1.0
    assert audit.inverse_max_abs_error <= 1e-5


def test_small_tau_proxy_matches_hard_nearest_neighbor() -> None:
    clean, _ = _factorial_fixture()
    features = np.concatenate(list(clean.values()))
    estimator = _fitted_estimator(features)
    bank_z = estimator.transform(features)
    calibration = ScoreCalibration.fit(bank_z, ScoreConfig(query_chunk_size=16))

    audit = audit_proxy_fidelity(
        estimator,
        calibration,
        features[:16],
        features[16:],
        tau=1e-4,
        top_fraction=0.25,
    )

    assert audit.passed
    assert audit.spearman == 1.0
    assert audit.top_fraction_overlap == 1.0
    assert audit.q995_difference <= 0.01


def test_refresh_static_memory_preserves_frozen_calibration() -> None:
    clean, _ = _factorial_fixture()
    features = np.concatenate(list(clean.values()))
    pipeline = FlowTTE()
    pipeline.fit(features)
    calibration = pipeline.score_calibration
    old_memory = pipeline.memory
    old_bank = old_memory.bank.features.detach().clone()  # type: ignore[union-attr]

    pipeline.refresh_static_memory(features)

    assert pipeline.score_calibration is calibration
    assert pipeline.memory is not old_memory
    assert pipeline.memory is not None
    assert pipeline.memory.bank.size() == old_bank.shape[0]
    assert torch.equal(pipeline.memory.bank.features, old_bank)


def test_offline_pipeline_clone_is_exact_and_parameter_independent() -> None:
    clean, _ = _factorial_fixture()
    features = np.concatenate(list(clean.values()))
    pipeline = FlowTTE()
    pipeline.fit(features)

    clone = pipeline.clone_offline()

    assert clone.score_calibration is pipeline.score_calibration
    assert clone.memory is not pipeline.memory
    assert clone.memory is not None
    assert pipeline.memory is not None
    assert torch.equal(clone.memory.bank.features, pipeline.memory.bank.features)
    assert clone.estimator is not None
    assert pipeline.estimator is not None
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            clone.estimator.flow.parameters(),
            pipeline.estimator.flow.parameters(),
        )
    )
    with torch.no_grad():
        next(clone.estimator.flow.parameters()).add_(1.0)
    assert not torch.equal(
        next(clone.estimator.flow.parameters()),
        next(pipeline.estimator.flow.parameters()),
    )


def test_stable_seed_is_repeatable_and_separates_parts() -> None:
    assert stable_seed("can", 0, "fold") == stable_seed("can", 0, "fold")
    assert stable_seed("can", 0, "fold") != stable_seed("can", 1, "fold")
