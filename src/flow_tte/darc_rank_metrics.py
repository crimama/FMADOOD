"""One-sort exact binary ranking metrics for the DARC common evaluator."""

from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
import struct
from dataclasses import dataclass
from typing import Final, Tuple

import numpy as np
import numpy.typing as npt
from scipy.integrate import trapezoid

BoolArray = npt.NDArray[np.bool_]
Float32Array = npt.NDArray[np.float32]
Float64Array = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]
_MAX_FPR: Final = 0.05


@dataclass(frozen=True)  # noqa: RUF100 -- Python 3.8; # noqa: SLOTS_OK
class RankedMetrics:
    p_auroc_005: float
    p_ap: float
    oracle_f1: float
    oracle_threshold: float


@dataclass(frozen=True)  # noqa: RUF100 -- Python 3.8; # noqa: SLOTS_OK
class RankedViews:
    all_test: RankedMetrics
    bad_only: RankedMetrics


@dataclass(frozen=True)  # noqa: RUF100 -- Python 3.8; # noqa: SLOTS_OK
class _RankCurve:
    tps: Float64Array
    fps: Float64Array
    ordered_scores: Float32Array
    starts: IntArray


def rank_binary_views(
    labels: BoolArray,
    scores: Float32Array,
    bad_mask: BoolArray,
) -> RankedViews:
    """Compute all-test and bad-only metrics from one stable score ordering."""
    descending: IntArray = np.asarray(
        np.argsort(scores, kind="mergesort")[::-1],
        dtype=np.intp,
    )
    ordered_scores: Float32Array = np.asarray(scores[descending], dtype=np.float32)
    ordered_labels: BoolArray = np.asarray(labels[descending], dtype=np.bool_)
    ordered_bad: BoolArray = np.asarray(bad_mask[descending], dtype=np.bool_)
    starts: IntArray = np.asarray(
        np.concatenate(
            (
                np.zeros(1, dtype=np.intp),
                np.flatnonzero(np.diff(ordered_scores)) + 1,
            ),
        ),
        dtype=np.intp,
    )

    all_tps: Float64Array = np.asarray(
        np.add.reduceat(ordered_labels, starts, dtype=np.float64),
        dtype=np.float64,
    )
    np.cumsum(all_tps, out=all_tps)
    all_fps = _all_false_positives(starts, scores.size, all_tps)
    all_test = _summarize(_RankCurve(all_tps, all_fps, ordered_scores, starts))

    bad_group_sizes: Float64Array = np.asarray(
        np.add.reduceat(ordered_bad, starts, dtype=np.float64),
        dtype=np.float64,
    )
    present: BoolArray = np.asarray(bad_group_sizes > 0.0, dtype=np.bool_)
    np.cumsum(bad_group_sizes, out=bad_group_sizes)
    bad_tps: Float64Array = np.asarray(
        np.add.reduceat(ordered_labels & ordered_bad, starts, dtype=np.float64),
        dtype=np.float64,
    )
    np.cumsum(bad_tps, out=bad_tps)
    bad_fps: Float64Array = np.asarray(bad_group_sizes - bad_tps, dtype=np.float64)
    bad_tps = np.asarray(bad_tps[present], dtype=np.float64)
    bad_fps = np.asarray(bad_fps[present], dtype=np.float64)
    bad_starts = np.asarray(starts[present], dtype=np.intp)
    bad_only = _summarize(_RankCurve(bad_tps, bad_fps, ordered_scores, bad_starts))
    return RankedViews(all_test=all_test, bad_only=bad_only)


def _all_false_positives(
    starts: IntArray,
    sample_count: int,
    tps: Float64Array,
) -> Float64Array:
    cumulative_sizes = np.empty(starts.size, dtype=np.float64)
    cumulative_sizes[:-1] = starts[1:]
    cumulative_sizes[-1] = sample_count
    cumulative_sizes -= tps
    return cumulative_sizes


def _summarize(curve: _RankCurve) -> RankedMetrics:
    tps = curve.tps
    fps = curve.fps
    p_auroc = _partial_auroc(tps, fps)
    total_positives = _float64_at(tps, tps.size - 1)
    fps += tps
    positive_predictions: BoolArray = np.asarray(np.not_equal(fps, 0.0), dtype=np.bool_)
    np.divide(tps, fps, out=fps, where=positive_predictions)
    np.divide(tps, total_positives, out=tps)

    ap_terms = np.empty_like(tps)
    np.subtract(tps[-2::-1], tps[:0:-1], out=ap_terms[:-1])
    ap_terms[-1] = -tps[0]
    np.multiply(ap_terms, fps[::-1], out=ap_terms)
    p_ap = -float(np.sum(ap_terms))

    denominator: Float64Array = np.asarray(fps + tps, dtype=np.float64)
    np.multiply(tps, fps, out=tps)
    tps *= 2.0
    np.divide(tps, denominator, out=tps, where=denominator > 0.0)
    reverse_index = int(np.argmax(tps[::-1]))
    best_index = tps.size - 1 - reverse_index
    threshold_value: Float64Array = np.asarray(
        curve.ordered_scores[curve.starts[best_index : best_index + 1]],
        dtype=np.float64,
    )
    return RankedMetrics(
        p_auroc_005=p_auroc,
        p_ap=p_ap,
        oracle_f1=_float64_at(tps, best_index),
        oracle_threshold=_float64_at(threshold_value, 0),
    )


def _partial_auroc(tps: Float64Array, fps: Float64Array) -> float:
    max_fpr = _MAX_FPR
    total_negatives = _float64_at(fps, fps.size - 1)
    total_positives = _float64_at(tps, tps.size - 1)
    lower = 0
    upper = fps.size
    while lower < upper:
        middle = (lower + upper) // 2
        if fps[middle] / total_negatives <= max_fpr:
            lower = middle + 1
        else:
            upper = middle
    upper = lower
    fpr: Float64Array = np.hstack((0.0, fps[: upper + 1] / total_negatives))
    tpr: Float64Array = np.hstack((0.0, tps[: upper + 1] / total_positives))
    interpolation = float(
        np.interp(
            max_fpr,
            fpr[upper : upper + 2],
            tpr[upper : upper + 2],
        ),
    )
    partial_fpr: Float64Array = np.hstack((fpr[: upper + 1], max_fpr))
    partial_tpr: Float64Array = np.hstack((tpr[: upper + 1], interpolation))
    partial_auc = _float64_at(
        np.asarray([trapezoid(partial_tpr, partial_fpr)], dtype=np.float64),
        0,
    )
    min_area = 0.5 * max_fpr**2
    return 0.5 * (1.0 + (partial_auc - min_area) / (max_fpr - min_area))


def _float64_at(values: Float64Array, index: int) -> float:
    normalized: Float64Array = np.asarray(values[index : index + 1], dtype=np.float64)
    unpacked: Tuple[float] = struct.unpack("=d", normalized.tobytes())
    return unpacked[0]
