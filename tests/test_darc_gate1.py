from __future__ import annotations

import numpy as np

from flow_tte.darc_gate1 import (
    Gate1Thresholds,
    SourceConditions,
    SourceEvaluationInput,
    decide_gate1,
    evaluate_source,
)


def _source_input(*, broad_score: float = 1.0, clean_score: float = 0.0) -> SourceEvaluationInput:
    clean = np.zeros((8, 8), dtype=np.float32)
    cue_mask = np.zeros((8, 8), dtype=np.bool_)
    cue_mask[3:5, 3:5] = True
    broad_mask = np.zeros((8, 8), dtype=np.bool_)
    broad_mask[2:6, 2:6] = True
    masks = (
        np.zeros_like(cue_mask),
        cue_mask,
        cue_mask,
        broad_mask,
    )
    low_maps = tuple(clean.copy() for _ in masks)
    null_maps = tuple(clean.copy() for _ in masks)
    high_maps = (
        np.full_like(clean, clean_score),
        cue_mask.astype(np.float32),
        cue_mask.astype(np.float32),
        broad_mask.astype(np.float32) * broad_score,
    )
    reference = np.zeros(128, dtype=np.float32)
    calibration = tuple(np.zeros_like(clean) for _ in range(4))
    return SourceEvaluationInput(
        object_name="bottle",
        source_id="train/good/000.png",
        seed=0,
        fold_index=0,
        masks=masks,
        low=SourceConditions(
            query_maps=low_maps,
            reference_scores=reference,
            calibration_maps=calibration,
        ),
        bilinear_null=SourceConditions(
            query_maps=null_maps,
            reference_scores=reference,
            calibration_maps=calibration,
        ),
        high=SourceConditions(
            query_maps=high_maps,
            reference_scores=reference,
            calibration_maps=calibration,
        ),
    )


def test_evaluate_source_reports_resolution_gain_without_shape_resampling() -> None:
    # Given
    inputs = _source_input()

    # When
    result = evaluate_source(inputs)

    # Then
    assert result.high.ap > result.low.ap
    assert result.high.component_recall == 1.0
    assert result.low.component_recall == 0.0


def test_decide_gate1_passes_registered_component_branch() -> None:
    # Given
    results = tuple(evaluate_source(_source_input()) for _ in range(4))
    thresholds = Gate1Thresholds(bootstrap_replicates=200, bootstrap_seed=7)

    # When
    decision = decide_gate1(results, thresholds)

    # Then
    assert decision.passed
    assert decision.component_gain >= 0.10
    assert decision.component_ci_lower > 0.0


def test_broad_control_and_clean_map_do_not_change_thin_profile_ap() -> None:
    # Given
    baseline = _source_input(broad_score=1.0, clean_score=0.0)
    changed_controls = _source_input(broad_score=100.0, clean_score=100.0)

    # When
    baseline_metric = evaluate_source(baseline)
    changed_metric = evaluate_source(changed_controls)

    # Then
    assert changed_metric.high.ap == baseline_metric.high.ap


def test_fixed_threshold_uses_all_four_heldout_clean_calibration_maps() -> None:
    # Given
    inputs = _source_input()
    high_calibration = list(inputs.high.calibration_maps)
    high_calibration[-1] = np.ones_like(high_calibration[-1])
    changed = SourceEvaluationInput(
        object_name=inputs.object_name,
        source_id=inputs.source_id,
        seed=inputs.seed,
        fold_index=inputs.fold_index,
        masks=inputs.masks,
        low=inputs.low,
        bilinear_null=inputs.bilinear_null,
        high=SourceConditions(
            query_maps=inputs.high.query_maps,
            reference_scores=inputs.high.reference_scores,
            calibration_maps=tuple(high_calibration),
        ),
    )

    # When
    metric = evaluate_source(changed)

    # Then
    assert metric.high.fixed_threshold > 0.0


def test_threshold_stability_rotates_three_calibration_maps_and_flags_outlier() -> None:
    # Given: one of four held-out clean maps has a shifted score distribution.
    inputs = _source_input()
    high_calibration = list(inputs.high.calibration_maps)
    high_calibration[-1] = np.ones_like(high_calibration[-1])
    changed = SourceEvaluationInput(
        object_name=inputs.object_name,
        source_id=inputs.source_id,
        seed=inputs.seed,
        fold_index=inputs.fold_index,
        masks=inputs.masks,
        low=inputs.low,
        bilinear_null=inputs.bilinear_null,
        high=SourceConditions(
            query_maps=inputs.high.query_maps,
            reference_scores=inputs.high.reference_scores,
            calibration_maps=tuple(high_calibration),
        ),
    )

    # When
    metric = evaluate_source(changed)

    # Then: the primary threshold still uses all four, while diagnostics use 3/1 rotations.
    assert metric.high.fixed_threshold > 0.0
    assert tuple(row.heldout_index for row in metric.high.stability.rotations) == (0, 1, 2, 3)
    assert metric.high.stability.maximum_fpr == 1.0
    assert not metric.high.stability.stable


def test_threshold_stability_limits_are_frozen_from_registered_method() -> None:
    # Given / When
    thresholds = Gate1Thresholds()

    # Then
    assert thresholds.maximum_stability_median_fpr == 2e-4
    assert thresholds.maximum_stability_fpr == 1e-3
    assert thresholds.maximum_stability_iqr_ratio == 0.25
