"""Raw-map utilities for the DARC AD2 performance pilot."""

from __future__ import annotations

from typing import List, NamedTuple, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from flow_tte.darc_feature_stream import FloatArray, stitch_score_grids
from flow_tte.darc_gate2_pipeline_types import BoolArray, QueryLadderResult

Int64Array = npt.NDArray[np.int64]


class PilotMapError(ValueError):
    reason: str

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Invalid DARC AD2 pilot map input: {reason}")


class RawRungMaps(NamedTuple):
    g0: FloatArray
    l0: FloatArray
    l1: FloatArray
    r1: FloatArray


class LadderCoverage(NamedTuple):
    token_count: int
    nonfallback_count: int
    fallback_fraction: float
    registration_count: int
    accepted_registration_count: int
    l0_support_histogram: Tuple[int, ...]
    l1_support_histogram: Tuple[int, ...]
    shared_support_histogram: Tuple[int, ...]
    r1_support_histogram: Tuple[int, ...]


def raw_rung_maps(result: QueryLadderResult) -> RawRungMaps:
    """Stitch every raw ladder residual onto the query's native image grid."""
    crops = tuple(item.crop for item in result.crops)
    if not crops:
        raise PilotMapError("a query must contain at least one scored crop")
    return RawRungMaps(
        g0=_stitch(result, tuple(item.scores.g0 for item in result.crops)),
        l0=_stitch(result, tuple(item.scores.l0 for item in result.crops)),
        l1=_stitch(result, tuple(item.scores.l1 for item in result.crops)),
        r1=_stitch(result, tuple(item.scores.r1 for item in result.crops)),
    )


def mean_rung_maps(folds: Sequence[RawRungMaps]) -> RawRungMaps:
    """Average fold maps in float64 and cast each completed arm once to float32."""
    ordered = tuple(folds)
    if not ordered:
        raise PilotMapError("at least one fold map is required")
    return RawRungMaps(
        g0=_mean_arm(tuple(item.g0 for item in ordered)),
        l0=_mean_arm(tuple(item.l0 for item in ordered)),
        l1=_mean_arm(tuple(item.l1 for item in ordered)),
        r1=_mean_arm(tuple(item.r1 for item in ordered)),
    )


def ladder_coverage(result: QueryLadderResult) -> LadderCoverage:
    """Compact label-free fallback and support-validity diagnostics for one fold."""
    if not result.crops:
        raise PilotMapError("coverage requires at least one scored crop")
    fallback = result.concatenate_fallback_mask()
    token_count = len(fallback)
    support_count = len(result.selected_support_ids)
    if token_count < 1 or support_count < 1:
        raise PilotMapError("coverage populations must be non-empty")
    support_arrays = (
        _concatenate_support_validity(
            tuple(crop.scores.support_validity.l0 for crop in result.crops),
        ),
        _concatenate_support_validity(
            tuple(crop.scores.support_validity.l1 for crop in result.crops),
        ),
        _concatenate_support_validity(
            tuple(crop.scores.support_validity.shared for crop in result.crops),
        ),
        _concatenate_support_validity(
            tuple(crop.scores.support_validity.r1 for crop in result.crops),
        ),
    )
    if any(values.shape != (token_count, support_count) for values in support_arrays):
        raise PilotMapError("support-validity population does not match scorer tokens")
    histograms = tuple(
        _support_histogram(values, support_count) for values in support_arrays
    )
    nonfallback_count = int(np.count_nonzero(~fallback))
    return LadderCoverage(
        token_count=token_count,
        nonfallback_count=nonfallback_count,
        fallback_fraction=(token_count - nonfallback_count) / token_count,
        registration_count=len(result.registration_audit),
        accepted_registration_count=sum(item.accepted for item in result.registration_audit),
        l0_support_histogram=histograms[0],
        l1_support_histogram=histograms[1],
        shared_support_histogram=histograms[2],
        r1_support_histogram=histograms[3],
    )


def _stitch(result: QueryLadderResult, values: Tuple[FloatArray, ...]) -> FloatArray:
    grids: List[FloatArray] = []
    for crop, scores in zip(result.crops, values):
        expected = crop.token_shape.height * crop.token_shape.width
        if len(scores) != expected:
            raise PilotMapError("rung score population does not match its token grid")
        grids.append(
            np.asarray(
                scores.reshape(crop.token_shape.height, crop.token_shape.width),
                dtype=np.float32,
            ),
        )
    return stitch_score_grids(
        result.native_size,
        tuple(item.crop for item in result.crops),
        tuple(grids),
    )


def _mean_arm(values: Tuple[FloatArray, ...]) -> FloatArray:
    shape = values[0].shape
    if any(value.shape != shape for value in values):
        raise PilotMapError("fold maps must have identical shapes")
    stacked: np.ndarray = np.asarray(np.stack(values, axis=0), dtype=np.float64)
    return np.asarray(np.mean(stacked, axis=0, dtype=np.float64), dtype=np.float32)


def _concatenate_support_validity(values: Tuple[BoolArray, ...]) -> BoolArray:
    return np.asarray(np.concatenate(values, axis=0), dtype=np.bool_)


def _support_histogram(values: BoolArray, support_count: int) -> Tuple[int, ...]:
    counts: Int64Array = np.asarray(np.sum(values, axis=1, dtype=np.int64), dtype=np.int64)
    histogram: List[int] = []
    for valid_count in range(support_count + 1):
        matches: BoolArray = np.asarray(counts == valid_count, dtype=np.bool_)
        histogram.append(int(np.count_nonzero(matches)))
    return tuple(histogram)
