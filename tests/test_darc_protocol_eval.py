from __future__ import annotations

import math

import numpy as np
import pytest

from flow_tte.darc_protocol_eval import (
    MorphologyConfig,
    Population,
    ProtocolConfig,
    ProtocolInputError,
    ProtocolInputs,
    component_recall,
    evaluate_protocol,
    shared_morphology,
)


def _inputs(
    scores: np.ndarray,
    masks: np.ndarray,
    populations: tuple[Population, ...],
    calibration: np.ndarray,
) -> ProtocolInputs:
    return ProtocolInputs(
        score_maps=np.asarray(scores, dtype=np.float32),
        gt_masks=np.asarray(masks, dtype=np.bool_),
        image_ids=tuple(f"test-{index}" for index in range(len(scores))),
        populations=populations,
        calibration_score_maps=np.asarray(calibration, dtype=np.float32),
        calibration_image_ids=("calibration-0",),
    )


def test_evaluate_protocol_uses_only_attainable_tied_score_thresholds() -> None:
    # Given: all test pixels share one score, with one anomalous pixel.
    inputs = _inputs(
        scores=np.full((2, 1, 2), 0.5, dtype=np.float32),
        masks=np.asarray([[[0, 0]], [[1, 0]]], dtype=np.bool_),
        populations=(Population.GOOD, Population.BAD),
        calibration=np.zeros((1, 1, 2), dtype=np.float32),
    )

    # When: standardized all-test and bad-only views are evaluated.
    report = evaluate_protocol(inputs)

    # Then: oracle F1 does not split equal scores into an unattainable prefix.
    assert math.isclose(report.all_test.oracle_f1, 0.4)
    assert math.isclose(report.bad_only.oracle_f1, 2.0 / 3.0)
    assert math.isclose(report.all_test.p_auroc_005, 0.5)
    assert math.isclose(report.all_test.p_ap, 0.25)


def test_evaluate_protocol_uses_the_higher_empirical_quantile() -> None:
    # Given: an even calibration sample and a test positive at the higher median.
    inputs = _inputs(
        scores=np.asarray([[[0.29, 0.30]], [[0.0, 0.0]]], dtype=np.float32),
        masks=np.asarray([[[0, 1]], [[0, 0]]], dtype=np.bool_),
        populations=(Population.BAD, Population.GOOD),
        calibration=np.asarray([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32),
    )

    # When: the 50th-percentile fixed threshold is selected conservatively.
    report = evaluate_protocol(inputs, ProtocolConfig(normal_quantile=0.5))

    # Then: the threshold is the observed higher order statistic, not an interpolation.
    assert math.isclose(report.fixed_threshold, 0.3, rel_tol=1e-6)
    assert report.bad_only.fixed_f1 == 1.0


def test_shared_morphology_closes_a_fixed_one_pixel_gap() -> None:
    # Given: two positive pixels separated by one interior pixel.
    mask = np.zeros((7, 7), dtype=np.bool_)
    mask[3, 2] = True
    mask[3, 4] = True

    # When: the shared fixed 3x3 closing profile is applied.
    processed = shared_morphology(mask, MorphologyConfig(closing_size=3))

    # Then: the one-pixel gap is connected.
    assert np.all(processed[3, 2:5])


def test_component_recall_counts_ground_truth_components_hit_at_least_once() -> None:
    # Given: two disjoint ground-truth components and a prediction hitting one.
    masks = np.zeros((1, 5, 5), dtype=np.bool_)
    masks[0, 1, 1:3] = True
    masks[0, 3, 3:5] = True
    predictions = np.zeros_like(masks)
    predictions[0, 1, 1] = True

    # When: 8-connected component recall is measured.
    recall = component_recall(masks, predictions)

    # Then: one of two components is recalled.
    assert recall == 0.5


def test_evaluate_protocol_reports_raw_and_shared_morphology_fixed_f1() -> None:
    # Given: a bad-image score mask with a closable gap inside the ground truth.
    scores = np.zeros((2, 7, 7), dtype=np.float32)
    scores[0, 3, 2] = 1.0
    scores[0, 3, 4] = 1.0
    masks = np.zeros((2, 7, 7), dtype=np.bool_)
    masks[0, 3, 2:5] = True
    inputs = _inputs(
        scores=scores,
        masks=masks,
        populations=(Population.BAD, Population.GOOD),
        calibration=np.full((1, 2, 2), 0.5, dtype=np.float32),
    )

    # When: raw and shared-morphology fixed predictions are evaluated together.
    report = evaluate_protocol(inputs)

    # Then: morphology is explicit and improves only its separately named metric.
    assert report.bad_only.fixed_f1 == 0.8
    assert report.bad_only.morphology_fixed_f1 == 1.0
    assert report.bad_only.component_recall == 1.0


def test_evaluate_protocol_rejects_population_mask_mismatch() -> None:
    # Given: an image declared good despite a positive ground-truth pixel.
    inputs = _inputs(
        scores=np.asarray([[[0.1, 0.2]], [[0.3, 0.4]]], dtype=np.float32),
        masks=np.asarray([[[1, 0]], [[0, 1]]], dtype=np.bool_),
        populations=(Population.GOOD, Population.BAD),
        calibration=np.zeros((1, 1, 2), dtype=np.float32),
    )

    # When / Then: the protocol boundary refuses the inconsistent population.
    with pytest.raises(ProtocolInputError, match="good image"):
        evaluate_protocol(inputs)


def test_evaluate_protocol_rejects_duplicate_or_leaked_image_ids() -> None:
    # Given: otherwise valid maps whose test identities are duplicated.
    inputs = ProtocolInputs(
        score_maps=np.asarray([[[0.1, 0.2]], [[0.3, 0.4]]], dtype=np.float32),
        gt_masks=np.asarray([[[0, 0]], [[0, 1]]], dtype=np.bool_),
        image_ids=("same", "same"),
        populations=(Population.GOOD, Population.BAD),
        calibration_score_maps=np.zeros((1, 1, 2), dtype=np.float32),
        calibration_image_ids=("same",),
    )

    # When / Then: identity ambiguity is rejected before metric computation.
    with pytest.raises(ProtocolInputError, match="unique"):
        evaluate_protocol(inputs)
