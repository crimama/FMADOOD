from __future__ import annotations

import numpy as np
import pytest
import torch

from flow_tte.config import ExpansionConfig, FlowConfig, FlowTTEConfig, ScoreConfig
from flow_tte.pipeline import FlowTTE
from flow_tte.waff import (
    WaffConfig,
    WaffFactorialCell,
    WaffScorer,
    audit_weight_simplex_feasibility,
    score_waff_components,
    source_excluded_nearest_distance,
    train_waff_scorer,
    waff_constraint_values,
)


def _cell(axis: str, family: str, offset: float = 0.0) -> WaffFactorialCell:
    clean = np.asarray(
        [[0.0, 0.1], [0.1, 0.0], [0.2, 0.1], [0.1, 0.2], [0.0, 0.0], [0.2, 0.2]],
        dtype=np.float32,
    )
    shifted_clean = clean + np.asarray([0.25 + offset, 0.10], dtype=np.float32)
    defect = clean + np.asarray([0.9, 0.5], dtype=np.float32)
    shifted_defect = defect + np.asarray([-0.35 - offset, -0.10], dtype=np.float32)
    return WaffFactorialCell(
        clean=clean,
        shifted_clean=shifted_clean,
        defect=defect,
        shifted_defect=shifted_defect,
        defect_mask=np.asarray([True, True, False, False, False, False]),
        background_mask=np.asarray([False, False, False, True, True, True]),
        axis=axis,
        family=family,
    )


def test_source_excluded_distance_rejects_self_source_matches() -> None:
    bank = torch.tensor([[0.0], [0.1], [2.0], [2.1]], dtype=torch.float32)
    source_ids = torch.tensor([0, 0, 1, 1])

    distances = source_excluded_nearest_distance(
        bank,
        bank,
        source_ids,
        source_ids,
        chunk_size=2,
    )

    torch.testing.assert_close(distances, torch.tensor([2.0, 1.9, 1.9, 2.0]))
    assert bool((distances > 0.0).all())


def test_source_excluded_distance_requires_another_source() -> None:
    features = torch.zeros((3, 2), dtype=torch.float32)
    source_ids = torch.zeros(3, dtype=torch.long)
    with pytest.raises(ValueError, match="different-source"):
        source_excluded_nearest_distance(features, features, source_ids, source_ids)


def test_disabled_pipeline_is_byte_identical_and_static() -> None:
    rng = np.random.default_rng(91)
    support = rng.normal(size=(2, 5, 4)).astype(np.float32)
    query = rng.normal(size=(2, 3, 4)).astype(np.float32)
    pipeline = FlowTTE(
        FlowTTEConfig(
            flow=FlowConfig(n_coupling_layers=2, n_epochs=1, batch_size=8, seed=91),
            expansion=ExpansionConfig(budget=1.0, density_quantile=0.9),
            score=ScoreConfig(density_weight=0.25, top_percent=0.25),
            device="cpu",
        ),
    )
    pipeline.fit(support)
    before = pipeline.memory.bank.features.detach().clone()  # type: ignore[union-attr]

    first = pipeline.score_static(query)
    second = pipeline.score_static(query)

    np.testing.assert_array_equal(first.patch_scores, second.patch_scores)
    np.testing.assert_array_equal(first.image_scores, second.image_scores)
    torch.testing.assert_close(pipeline.memory.bank.features, before)  # type: ignore[union-attr]
    assert pipeline.waff_scorer is None
    assert first.memory_size_before == first.memory_size_after


def test_factorial_components_use_cross_source_clean_loo_without_mutation() -> None:
    support = np.asarray(
        [[[0.0, 0.0], [0.1, 0.0]], [[2.0, 0.0], [2.1, 0.0]]],
        dtype=np.float32,
    )
    pipeline = FlowTTE(
        FlowTTEConfig(
            flow=FlowConfig(
                transform_mode="identity",
                n_coupling_layers=2,
                n_epochs=1,
                batch_size=4,
            ),
            expansion=ExpansionConfig(budget=1.0, density_quantile=0.9),
            score=ScoreConfig(loo_standardize=False, query_chunk_size=2),
            device="cpu",
        ),
    )
    pipeline.fit(support)
    source_ids = np.asarray([0, 0, 1, 1], dtype=np.int64)
    before = pipeline.memory.bank.features.detach().clone()  # type: ignore[union-attr]

    ordinary = pipeline.factorial_score_components(support)
    excluded = pipeline.factorial_score_components(
        support,
        query_source_ids=source_ids,
        bank_source_ids=source_ids,
    )

    assert np.all(excluded[:, 0] > ordinary[:, 0])
    assert np.all(excluded[:, 0] > 1.8)
    np.testing.assert_allclose(excluded[:, 0], excluded[::-1, 0], atol=1e-6)
    torch.testing.assert_close(pipeline.memory.bank.features, before)  # type: ignore[union-attr]


def test_constraints_have_nonzero_gradients() -> None:
    weight = torch.tensor([1.0, 0.25], requires_grad=True)
    cells = (_cell("exposure", "cutpaste"), _cell("blur", "chromatic", offset=0.1))

    defect, tail = waff_constraint_values(cells, weight)
    defect.backward(retain_graph=True)
    defect_gradient = weight.grad.detach().clone()
    weight.grad.zero_()
    tail.backward()

    assert float(defect_gradient.abs().sum()) > 0.0
    assert float(weight.grad.abs().sum()) > 0.0


def test_waff_training_is_finite_positive_and_changes_the_scorer() -> None:
    cells = (
        _cell("exposure", "cutpaste"),
        _cell("exposure", "chromatic", offset=0.05),
        _cell("blur", "cutpaste", offset=0.10),
        _cell("blur", "chromatic", offset=0.15),
    )

    scorer = train_waff_scorer(
        cells,
        config=WaffConfig(steps=30, learning_rate=0.03, constraint_update_interval=5),
    )
    scores = score_waff_components(scorer, cells[0].defect)

    assert scorer.weight.shape == (2,)
    assert np.all(np.isfinite(scorer.weight))
    assert np.all(scorer.weight > 0.0)
    assert float(scorer.weight.sum()) == pytest.approx(1.25)
    assert not np.array_equal(scorer.weight, np.asarray([1.0, 0.25], dtype=np.float32))
    assert np.all(np.isfinite(scores))
    assert scorer.training_cells == 4


@pytest.mark.parametrize("mode", ["nonfactorial", "mean_factorial", "waff"])
def test_matched_objective_modes_preserve_frozen_score_scale(mode: str) -> None:
    cells = (
        _cell("exposure", "cutpaste"),
        _cell("exposure", "chromatic", offset=0.05),
        _cell("blur", "cutpaste", offset=0.10),
        _cell("blur", "chromatic", offset=0.15),
    )

    scorer = train_waff_scorer(
        cells,
        config=WaffConfig(
            objective_mode=mode,
            steps=5,
            constraint_update_interval=1,
        ),
    )

    assert float(scorer.weight.sum()) == pytest.approx(1.25)


def test_invalid_objective_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="objective_mode"):
        train_waff_scorer(
            (_cell("exposure", "cutpaste"), _cell("blur", "chromatic")),
            config=WaffConfig(objective_mode="unknown", steps=1),
        )


def test_simplex_feasibility_audit_is_deterministic_and_scale_fixed() -> None:
    cells = (
        _cell("exposure", "cutpaste"),
        _cell("blur", "chromatic", offset=0.1),
    )

    first = audit_weight_simplex_feasibility(cells, grid_points=11)
    second = audit_weight_simplex_feasibility(cells, grid_points=11)

    assert first.grid_points == second.grid_points
    assert first.min_joint_violation == second.min_joint_violation
    np.testing.assert_array_equal(first.best_joint_weight, second.best_joint_weight)
    assert first.grid_points == 11
    assert float(first.best_joint_weight.sum()) == pytest.approx(1.25)
    assert np.isfinite(first.min_joint_violation)


def test_reported_constraints_match_returned_final_weight() -> None:
    cells = (
        _cell("exposure", "cutpaste"),
        _cell("exposure", "chromatic", offset=0.05),
        _cell("blur", "cutpaste", offset=0.10),
        _cell("blur", "chromatic", offset=0.15),
    )
    config = WaffConfig(steps=7, constraint_update_interval=1)

    scorer = train_waff_scorer(cells, config=config)
    defect, tail = waff_constraint_values(
        cells,
        torch.as_tensor(scorer.weight),
        config=config,
    )

    assert scorer.defect_constraint == pytest.approx(float(defect))
    assert scorer.normal_tail_constraint == pytest.approx(float(tail))


def test_active_scorer_changes_scores_but_keeps_memory_static() -> None:
    rng = np.random.default_rng(93)
    support = rng.normal(size=(2, 5, 4)).astype(np.float32)
    query = rng.normal(size=(1, 4, 4)).astype(np.float32)
    pipeline = FlowTTE(
        FlowTTEConfig(
            flow=FlowConfig(n_coupling_layers=2, n_epochs=1, batch_size=8, seed=93),
            expansion=ExpansionConfig(budget=1.0, density_quantile=0.9),
            score=ScoreConfig(density_weight=0.25, top_percent=0.25),
            device="cpu",
        ),
    )
    pipeline.fit(support)
    default = pipeline.score_static(query)
    pipeline.set_waff_scorer(
        WaffScorer(
            weight=np.asarray([0.4, 1.2], dtype=np.float32),
            initial_loss=1.0,
            final_loss=0.5,
            defect_constraint=0.0,
            normal_tail_constraint=0.0,
            training_cells=4,
        ),
    )
    before = pipeline.memory.bank.features.detach().clone()  # type: ignore[union-attr]

    active = pipeline.score_static(query)

    assert not np.array_equal(active.patch_scores, default.patch_scores)
    torch.testing.assert_close(pipeline.memory.bank.features, before)  # type: ignore[union-attr]
    assert active.memory_size_before == active.memory_size_after


def test_misaligned_factorial_cell_is_rejected() -> None:
    cell = _cell("exposure", "cutpaste")
    broken = WaffFactorialCell(
        clean=cell.clean[:-1],
        shifted_clean=cell.shifted_clean,
        defect=cell.defect,
        shifted_defect=cell.shifted_defect,
        defect_mask=cell.defect_mask,
        background_mask=cell.background_mask,
        axis=cell.axis,
        family=cell.family,
    )
    with pytest.raises(ValueError, match="aligned"):
        train_waff_scorer(
            (broken, _cell("blur", "chromatic")),
            config=WaffConfig(steps=1),
        )
