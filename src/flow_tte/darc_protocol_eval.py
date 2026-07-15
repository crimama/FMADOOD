from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from flow_tte.metrics import MetricInputError as ProtocolInputError

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


class Population(str, Enum):
    GOOD = "good"
    BAD = "bad"


@dataclass(frozen=True)  # noqa: SLOTS_OK -- Python 3.8 has no dataclass slots.
class MorphologyConfig:
    closing_size: int = 3

    def __post_init__(self) -> None:
        if self.closing_size < 1 or self.closing_size % 2 == 0:
            raise ProtocolInputError("closing_size must be a positive odd integer")


@dataclass(frozen=True)  # noqa: SLOTS_OK -- Python 3.8 has no dataclass slots.
class ProtocolConfig:
    max_fpr: float = 0.05
    normal_quantile: float = 0.9999
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)

    def __post_init__(self) -> None:
        if not 0.0 < self.max_fpr <= 1.0:
            raise ProtocolInputError("max_fpr must be in (0, 1]")
        if not 0.0 <= self.normal_quantile <= 1.0:
            raise ProtocolInputError("normal_quantile must be in [0, 1]")


@dataclass(frozen=True)  # noqa: SLOTS_OK -- Python 3.8 has no dataclass slots.
class ProtocolInputs:
    score_maps: FloatArray
    gt_masks: BoolArray
    image_ids: Tuple[str, ...]
    populations: Tuple[Population, ...]
    calibration_score_maps: FloatArray
    calibration_image_ids: Tuple[str, ...]


@dataclass(frozen=True)  # noqa: SLOTS_OK -- Python 3.8 has no dataclass slots.
class ProtocolViewMetrics:
    p_auroc_005: float
    p_ap: float
    oracle_f1: float
    fixed_f1: float
    morphology_fixed_f1: float
    component_recall: float
    morphology_component_recall: float


@dataclass(frozen=True)  # noqa: SLOTS_OK -- Python 3.8 has no dataclass slots.
class ProtocolReport:
    fixed_threshold: float
    all_test: ProtocolViewMetrics
    bad_only: ProtocolViewMetrics


@dataclass(frozen=True)  # noqa: SLOTS_OK -- Python 3.8 has no dataclass slots.
class _ViewInputs:
    score_maps: FloatArray
    gt_masks: BoolArray
    fixed_threshold: float


def evaluate_protocol(
    inputs: ProtocolInputs,
    config: Optional[ProtocolConfig] = None,
) -> ProtocolReport:
    active = config if config is not None else ProtocolConfig()
    scores: FloatArray = np.asarray(inputs.score_maps, dtype=np.float32)
    masks: BoolArray = np.asarray(inputs.gt_masks, dtype=np.bool_)
    calibration: FloatArray = np.asarray(inputs.calibration_score_maps, dtype=np.float32)
    if scores.ndim != 3 or masks.shape != scores.shape or len(scores) == 0:
        raise ProtocolInputError("score_maps and gt_masks require matching non-empty (N,H,W)")
    if calibration.ndim != 3 or len(calibration) == 0:
        raise ProtocolInputError("calibration_score_maps must be non-empty (N,H,W)")
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(calibration)):
        raise ProtocolInputError("score maps must contain only finite values")
    _validate_identities(inputs, len(scores), len(calibration))
    _validate_populations(inputs, masks)
    quantile_index = math.ceil(active.normal_quantile * (calibration.size - 1))
    ordered: FloatArray = np.asarray(
        np.partition(calibration.reshape(-1), quantile_index),
        dtype=np.float32,
    )
    threshold = float(np.min(ordered[quantile_index : quantile_index + 1]))
    bad_indices: BoolArray = np.asarray(
        [population == Population.BAD for population in inputs.populations],
        dtype=np.bool_,
    )
    return ProtocolReport(
        fixed_threshold=threshold,
        all_test=_evaluate_view(
            _ViewInputs(score_maps=scores, gt_masks=masks, fixed_threshold=threshold),
            active,
        ),
        bad_only=_evaluate_view(
            _ViewInputs(
                score_maps=scores[bad_indices],
                gt_masks=masks[bad_indices],
                fixed_threshold=threshold,
            ),
            active,
        ),
    )


def shared_morphology(
    mask: BoolArray,
    config: Optional[MorphologyConfig] = None,
) -> BoolArray:
    active = config if config is not None else MorphologyConfig()
    source = np.asarray(mask, dtype=np.bool_)
    if source.ndim != 2:
        raise ProtocolInputError("shared_morphology expects a 2D mask")
    kernel = np.ones((active.closing_size, active.closing_size), dtype=np.uint8)
    closed = cv2.morphologyEx(source.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return np.asarray(closed > 0, dtype=np.bool_)


def component_recall(gt_masks: BoolArray, predictions: BoolArray) -> float:
    masks: BoolArray = np.asarray(gt_masks, dtype=np.bool_)
    predicted: BoolArray = np.asarray(predictions, dtype=np.bool_)
    if masks.ndim != 3 or masks.shape != predicted.shape:
        raise ProtocolInputError("component recall expects matching (N,H,W) arrays")
    recalled = 0
    total = 0
    for image_index in range(len(masks)):
        mask: BoolArray = np.asarray(masks[image_index], dtype=np.bool_)
        prediction: BoolArray = np.asarray(predicted[image_index], dtype=np.bool_)
        component_count, raw_labels = cv2.connectedComponents(
            mask.astype(np.uint8),
            connectivity=8,
        )
        labels: npt.NDArray[np.int32] = np.asarray(raw_labels, dtype=np.int32)
        for component_index in range(1, int(component_count)):
            component: BoolArray = np.asarray(labels == component_index, dtype=np.bool_)
            total += 1
            recalled += int(np.any(prediction & component))
    return float("nan") if total == 0 else recalled / total


def _evaluate_view(inputs: _ViewInputs, config: ProtocolConfig) -> ProtocolViewMetrics:
    labels = inputs.gt_masks.reshape(-1)
    scores = inputs.score_maps.reshape(-1)
    fixed: BoolArray = np.asarray(inputs.score_maps >= inputs.fixed_threshold, dtype=np.bool_)
    morphology: BoolArray = np.asarray(
        np.stack(
            tuple(
                shared_morphology(np.asarray(fixed[index], dtype=np.bool_), config.morphology)
                for index in range(len(fixed))
            ),
            axis=0,
        ),
        dtype=np.bool_,
    )
    curve = precision_recall_curve(labels, scores)
    precision: npt.NDArray[np.float64] = np.asarray(curve[0], dtype=np.float64)
    recall: npt.NDArray[np.float64] = np.asarray(curve[1], dtype=np.float64)
    f1_values: npt.NDArray[np.float64] = np.asarray(np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0.0,
    ), dtype=np.float64)
    label_count = np.unique(labels).size
    p_auroc = (
        float("nan")
        if label_count < 2
        else float(roc_auc_score(labels, scores, max_fpr=config.max_fpr))
    )
    return ProtocolViewMetrics(
        p_auroc_005=p_auroc,
        p_ap=float(average_precision_score(labels, scores)),
        oracle_f1=float(np.max(f1_values)),
        fixed_f1=_binary_f1(labels, fixed.reshape(-1)),
        morphology_fixed_f1=_binary_f1(labels, morphology.reshape(-1)),
        component_recall=component_recall(inputs.gt_masks, fixed),
        morphology_component_recall=component_recall(inputs.gt_masks, morphology),
    )


def _validate_identities(
    inputs: ProtocolInputs,
    image_count: int,
    calibration_count: int,
) -> None:
    if len(inputs.image_ids) != image_count or len(inputs.populations) != image_count:
        raise ProtocolInputError("image IDs and populations must match the test population")
    if len(set(inputs.image_ids)) != image_count:
        raise ProtocolInputError("test image IDs must be unique")
    if len(inputs.calibration_image_ids) != calibration_count:
        raise ProtocolInputError("calibration image IDs must match calibration maps")
    if len(set(inputs.calibration_image_ids)) != len(inputs.calibration_image_ids):
        raise ProtocolInputError("calibration image IDs must be unique")
    if set(inputs.image_ids) & set(inputs.calibration_image_ids):
        raise ProtocolInputError("test and calibration image IDs must be disjoint")


def _validate_populations(inputs: ProtocolInputs, masks: BoolArray) -> None:
    if Population.GOOD not in inputs.populations or Population.BAD not in inputs.populations:
        raise ProtocolInputError("all-test evaluation requires good and bad images")
    for index, population in enumerate(inputs.populations):
        mask: BoolArray = np.asarray(masks[index], dtype=np.bool_)
        if population == Population.GOOD and np.any(mask):
            reason = f"good image {inputs.image_ids[index]} has a positive mask"
            raise ProtocolInputError(reason)
        if population == Population.BAD and not np.any(mask):
            reason = f"bad image {inputs.image_ids[index]} has an empty mask"
            raise ProtocolInputError(reason)


def _binary_f1(labels: BoolArray, predictions: BoolArray) -> float:
    true_positive = int(np.count_nonzero(labels & predictions))
    false_positive = int(np.count_nonzero(~labels & predictions))
    false_negative = int(np.count_nonzero(labels & ~predictions))
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else (2.0 * true_positive) / denominator
