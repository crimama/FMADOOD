from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
import struct
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pytest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from flow_tte.darc_rank_metrics import RankedMetrics, rank_binary_views

BoolArray = npt.NDArray[np.bool_]
Float32Array = npt.NDArray[np.float32]
Float64Array = npt.NDArray[np.float64]


@dataclass(frozen=True)  # noqa: RUF100 -- Python 3.8; # noqa: SLOTS_OK
class RankingFixture:
    labels: BoolArray
    scores: Float32Array
    bad_mask: BoolArray


def _reference(labels: BoolArray, scores: Float32Array) -> RankedMetrics:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    attainable_precision: Float64Array = np.asarray(precision[:-1], dtype=np.float64)
    attainable_recall: Float64Array = np.asarray(recall[:-1], dtype=np.float64)
    f1: Float64Array = np.asarray(
        np.divide(
            2.0 * attainable_precision * attainable_recall,
            attainable_precision + attainable_recall,
            out=np.zeros_like(attainable_precision),
            where=(attainable_precision + attainable_recall) > 0.0,
        ),
        dtype=np.float64,
    )
    best = int(np.argmax(f1))
    oracle_f1: Tuple[float] = struct.unpack("=d", f1[best : best + 1].tobytes())
    return RankedMetrics(
        p_auroc_005=float(roc_auc_score(labels, scores, max_fpr=0.05)),
        p_ap=float(average_precision_score(labels, scores)),
        oracle_f1=oracle_f1[0],
        oracle_threshold=float(thresholds[best]),
    )


def _random_fixture(seed: int) -> RankingFixture:
    rng = np.random.default_rng(seed)
    scores = np.asarray(rng.integers(-31, 32, size=20_003) / 16.0, dtype=np.float32)
    labels = np.asarray(rng.random(scores.size) < 0.017, dtype=np.bool_)
    bad_mask = np.asarray(rng.random(scores.size) < 0.61, dtype=np.bool_)
    labels[0] = True
    labels[1] = False
    bad_mask[:2] = True
    return RankingFixture(labels=labels, scores=scores, bad_mask=bad_mask)


@pytest.mark.parametrize("seed", range(12))
def test_rank_binary_views_matches_sklearn_for_random_float32_ties(seed: int) -> None:
    # Given: a seeded, tied float32 ranking problem and an induced bad-only subset.
    fixture = _random_fixture(seed)

    # When: both views are derived from one shared stable ordering.
    actual = rank_binary_views(fixture.labels, fixture.scores, fixture.bad_mask)

    # Then: all metrics preserve the prior public-sklearn semantics.
    for result, labels, scores in (
        (actual.all_test, fixture.labels, fixture.scores),
        (
            actual.bad_only,
            fixture.labels[fixture.bad_mask],
            fixture.scores[fixture.bad_mask],
        ),
    ):
        expected = _reference(labels, scores)
        assert result.p_ap == expected.p_ap
        assert result.oracle_f1 == expected.oracle_f1
        assert result.oracle_threshold == expected.oracle_threshold
        assert result.p_auroc_005 == pytest.approx(expected.p_auroc_005, abs=2e-15)


def test_rank_binary_views_matches_sklearn_under_extreme_imbalance() -> None:
    # Given: one positive among 250,001 tied float32 samples in both views.
    sample_count = 250_001
    scores = np.asarray(np.arange(sample_count) % 257, dtype=np.float32)
    labels = np.zeros(sample_count, dtype=np.bool_)
    labels[-1] = True
    bad_mask = np.ones(sample_count, dtype=np.bool_)

    # When: the one-sort implementation evaluates the imbalanced ranking.
    actual = rank_binary_views(labels, scores, bad_mask)

    # Then: sklearn parity includes AP, attainable oracle threshold/F1, and pAUC.
    expected = _reference(labels, scores)
    assert actual.all_test.p_ap == expected.p_ap
    assert actual.all_test.oracle_f1 == expected.oracle_f1
    assert actual.all_test.oracle_threshold == expected.oracle_threshold
    assert actual.all_test.p_auroc_005 == pytest.approx(expected.p_auroc_005, abs=2e-15)
    assert actual.bad_only == actual.all_test


def test_rank_binary_views_matches_sklearn_for_dense_float32_scores() -> None:
    # Given: mostly unique float32 scores plus sparse ties omitted from bad-only.
    rng = np.random.default_rng(20260710)
    scores = np.asarray(rng.random(200_003), dtype=np.float32)
    scores[::97] = np.float32(0.25)
    labels = np.asarray(rng.random(scores.size) < 0.013, dtype=np.bool_)
    bad_mask = np.asarray(rng.random(scores.size) < 0.69, dtype=np.bool_)
    labels[:2] = [True, False]
    bad_mask[:2] = True

    # When: bad-only groups are compacted from the shared global ordering.
    actual = rank_binary_views(labels, scores, bad_mask)

    # Then: group omission does not perturb sklearn's AP reduction order.
    for result, view_mask in (
        (actual.all_test, np.ones(scores.size, dtype=np.bool_)),
        (actual.bad_only, bad_mask),
    ):
        expected = _reference(labels[view_mask], scores[view_mask])
        assert result.p_ap == expected.p_ap
        assert result.oracle_f1 == expected.oracle_f1
        assert result.oracle_threshold == expected.oracle_threshold
        assert result.p_auroc_005 == pytest.approx(expected.p_auroc_005, abs=2e-15)


def test_rank_binary_views_preserves_extreme_finite_float32_thresholds() -> None:
    # Given: finite float32 extrema and adjacent representable tied values.
    maximum = np.finfo(np.float32).max
    near_maximum_values: Tuple[float] = struct.unpack(
        "=f",
        np.asarray(
            [np.nextafter(maximum, np.float32(0.0), dtype=np.float32)],
            dtype=np.float32,
        ).tobytes(),
    )
    near_maximum = near_maximum_values[0]
    scores = np.asarray(
        [maximum, near_maximum, near_maximum, 0.0, -near_maximum, -maximum],
        dtype=np.float32,
    )
    labels = np.asarray([False, True, False, True, False, False], dtype=np.bool_)
    bad_mask = np.asarray([True, True, True, True, False, False], dtype=np.bool_)

    # When: the scores are ranked without changing their dtype or tie groups.
    actual = rank_binary_views(labels, scores, bad_mask)

    # Then: attainable thresholds are bit-preserving float32-to-float64 conversions.
    for result, view_mask in (
        (actual.all_test, np.ones(scores.size, dtype=np.bool_)),
        (actual.bad_only, bad_mask),
    ):
        expected = _reference(labels[view_mask], scores[view_mask])
        assert result == expected
