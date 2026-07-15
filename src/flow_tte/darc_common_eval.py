"""Exact common-grid metrics and image-disjoint fixed diagnostics for DARC."""

from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
import math
import struct
from dataclasses import dataclass
from typing import Final, Mapping, Tuple

import cv2
import numpy as np
import numpy.typing as npt
from typing_extensions import override

from flow_tte.darc_map_io import BoolArray, FloatArray, ImageRecord, ObjectMapSet, Population
from flow_tte.darc_rank_metrics import RankedMetrics, rank_binary_views

try:
    from flow_tte.superadd_morphology import MorphologyConfig, postprocess_binary
except ModuleNotFoundError as error:
    message = "flow_tte.superadd_morphology is required for shared-superadd-v1 evaluation"
    raise RuntimeError(message) from error

_FOLD_COUNT: Final = 4
_NORMAL_QUANTILE: Final = 0.9999


@dataclass(frozen=True)  # noqa: RUF100 -- Python 3.8; # noqa: SLOTS_OK
class CommonEvaluationError(ValueError):
    reason: str

    @override
    def __str__(self) -> str:
        return f"Invalid common evaluation: {self.reason}"


@dataclass(frozen=True)  # noqa: RUF100 -- Python 3.8; # noqa: SLOTS_OK
class ViewMetrics:
    p_auroc_005: float
    p_ap: float
    oracle_f1: float
    oracle_threshold: float
    oracle_component_recall: float
    fixed_raw_f1: float
    fixed_morphology_f1: float


@dataclass(frozen=True)  # noqa: RUF100 -- Python 3.8; # noqa: SLOTS_OK
class ObjectEvaluation:
    object_name: str
    good_count: int
    bad_count: int
    fixed_thresholds: Tuple[float, float, float, float]
    all_test: ViewMetrics
    bad_only: ViewMetrics


@dataclass(frozen=True)  # noqa: RUF100 -- Python 3.8; # noqa: SLOTS_OK
class _FixedPredictions:
    raw: Tuple[BoolArray, ...]
    morphology: Tuple[BoolArray, ...]


def evaluate_object(maps: ObjectMapSet) -> ObjectEvaluation:
    """Evaluate one object while retaining at most that object's maps in memory."""
    good = tuple(record for record in maps.records if record.audit.population is Population.GOOD)
    bad = tuple(record for record in maps.records if record.audit.population is Population.BAD)
    if len(good) < 2 or not bad:
        raise CommonEvaluationError("evaluation requires at least two good and one bad image")
    thresholds, fold_by_id = _crossfit_thresholds(good, bad)
    bad_indices = tuple(
        index
        for index, record in enumerate(maps.records)
        if record.audit.population is Population.BAD
    )
    labels = np.concatenate(tuple(record.gt_mask.reshape(-1) for record in maps.records))
    scores = np.concatenate(tuple(record.score_map.reshape(-1) for record in maps.records))
    bad_pixel_mask = np.concatenate(
        tuple(
            np.full(
                record.score_map.size,
                record.audit.population is Population.BAD,
                dtype=np.bool_,
            )
            for record in maps.records
        ),
    )
    positive_count = int(np.count_nonzero(labels))
    bad_pixel_count = int(np.count_nonzero(bad_pixel_mask))
    bad_positive_count = int(np.count_nonzero(labels & bad_pixel_mask))
    if positive_count in {0, labels.size} or bad_positive_count in {0, bad_pixel_count}:
        raise CommonEvaluationError("each continuous metric view requires both pixel classes")
    if not np.all(np.isfinite(scores)):
        raise CommonEvaluationError("continuous metrics require finite score maps")
    ranked = rank_binary_views(labels, scores, bad_pixel_mask)
    del labels, scores, bad_pixel_mask
    raw_predictions: Tuple[BoolArray, ...] = tuple(
        np.asarray(
            record.score_map > thresholds[fold_by_id[record.audit.image_id]],
            dtype=np.bool_,
        )
        for record in maps.records
    )
    morphology_predictions: Tuple[BoolArray, ...] = tuple(
        np.asarray(
            postprocess_binary(
                record.score_map,
                thresholds[fold_by_id[record.audit.image_id]],
                MorphologyConfig(),
            )
            > 0,
            dtype=np.bool_,
        )
        for record in maps.records
    )
    return ObjectEvaluation(
        object_name=maps.object_name,
        good_count=len(good),
        bad_count=len(bad),
        fixed_thresholds=thresholds,
        all_test=_evaluate_view(
            maps.records,
            _FixedPredictions(raw_predictions, morphology_predictions),
            ranked.all_test,
        ),
        bad_only=_evaluate_view(
            tuple(maps.records[index] for index in bad_indices),
            _FixedPredictions(
                tuple(raw_predictions[index] for index in bad_indices),
                tuple(morphology_predictions[index] for index in bad_indices),
            ),
            ranked.bad_only,
        ),
    )


def _crossfit_thresholds(
    good: Tuple[ImageRecord, ...],
    bad: Tuple[ImageRecord, ...],
) -> tuple[Tuple[float, float, float, float], Mapping[str, int]]:
    sorted_good = tuple(sorted(good, key=lambda record: record.audit.image_id))
    sorted_bad = tuple(sorted(bad, key=lambda record: record.audit.image_id))
    fold_by_id = {
        record.audit.image_id: index % _FOLD_COUNT
        for records in (sorted_good, sorted_bad)
        for index, record in enumerate(records)
    }
    values: list[float] = []
    for fold in range(_FOLD_COUNT):
        calibration = np.concatenate(
            tuple(
                record.score_map.reshape(-1)
                for record in sorted_good
                if fold_by_id[record.audit.image_id] != fold
            ),
        )
        values.append(_higher_quantile(calibration, _NORMAL_QUANTILE))
    return (values[0], values[1], values[2], values[3]), fold_by_id


def _higher_quantile(values: npt.NDArray[np.float32], quantile: float) -> float:
    index = math.ceil(quantile * (values.size - 1))
    partitioned: FloatArray = np.asarray(
        np.partition(np.asarray(values, dtype=np.float32), index),
        dtype=np.float32,
    )
    unpacked: Tuple[float] = struct.unpack(
        "=f",
        partitioned[index : index + 1].tobytes(),
    )
    return unpacked[0]


def _evaluate_view(
    records: Tuple[ImageRecord, ...],
    fixed: _FixedPredictions,
    ranked: RankedMetrics,
) -> ViewMetrics:
    oracle_predictions: Tuple[BoolArray, ...] = tuple(
        np.asarray(record.score_map >= ranked.oracle_threshold, dtype=np.bool_)
        for record in records
    )
    return ViewMetrics(
        p_auroc_005=ranked.p_auroc_005,
        p_ap=ranked.p_ap,
        oracle_f1=ranked.oracle_f1,
        oracle_threshold=ranked.oracle_threshold,
        oracle_component_recall=_component_recall(records, oracle_predictions),
        fixed_raw_f1=_binary_f1(records, fixed.raw),
        fixed_morphology_f1=_binary_f1(records, fixed.morphology),
    )


def _binary_f1(
    records: Tuple[ImageRecord, ...],
    predictions: Tuple[BoolArray, ...],
) -> float:
    labels = np.concatenate(tuple(record.gt_mask.reshape(-1) for record in records))
    predicted = np.concatenate(tuple(mask.reshape(-1) for mask in predictions))
    true_positive = int(np.count_nonzero(labels & predicted))
    false_positive = int(np.count_nonzero(~labels & predicted))
    false_negative = int(np.count_nonzero(labels & ~predicted))
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else (2.0 * true_positive) / denominator


def _component_recall(
    records: Tuple[ImageRecord, ...],
    predictions: Tuple[BoolArray, ...],
) -> float:
    hit_count = 0
    component_total = 0
    for record, prediction in zip(records, predictions):
        components: Tuple[int, npt.NDArray[np.int32]] = cv2.connectedComponents(
            record.gt_mask.astype(np.uint8),
            connectivity=8,
        )
        component_count, labels = components
        for component_index in range(1, int(component_count)):
            component_total += 1
            hit_count += int(np.any(prediction & (labels == component_index)))
    if component_total == 0:
        raise CommonEvaluationError("component recall requires a positive ground-truth component")
    return hit_count / component_total
