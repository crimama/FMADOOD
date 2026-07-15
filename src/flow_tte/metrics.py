from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
from typing_extensions import final, override

from flow_tte.aupro import aupro_score

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
Float64Array = npt.NDArray[np.float64]

TTE_METRIC_KEYS: Tuple[str, ...] = (
    "I-AUROC",
    "I-AP",
    "I-F1_max",
    "P-AUROC",
    "P-AP",
    "P-F1_max",
    "AUPRO",
)


@dataclass(frozen=True)
class MetricInputError(ValueError):
    reason: str

    @override
    def __str__(self) -> str:
        return f"Invalid metric inputs: {self.reason}"


@dataclass(frozen=True)
@final
class MetricConfig:
    aupro_max_fpr: float = 0.30
    aupro_thresholds: int = 200

    def __post_init__(self) -> None:
        if not 0.0 < self.aupro_max_fpr <= 1.0:
            raise MetricInputError("aupro_max_fpr must be in (0, 1]")
        if self.aupro_thresholds <= 1:
            raise MetricInputError("aupro_thresholds must be greater than 1")


@dataclass(frozen=True)
@final
class MetricInputs:
    image_scores: FloatArray
    image_labels: BoolArray
    pixel_scores: Optional[FloatArray] = None
    pixel_masks: Optional[BoolArray] = None


@dataclass(frozen=True)
@final
class MetricScores:
    i_auroc: float
    i_ap: float
    i_f1_max: float
    p_auroc: float
    p_ap: float
    p_f1_max: float
    aupro: float

    def as_tte_dict(self) -> Dict[str, float]:
        return {
            "I-AUROC": self.i_auroc,
            "I-AP": self.i_ap,
            "I-F1_max": self.i_f1_max,
            "P-AUROC": self.p_auroc,
            "P-AP": self.p_ap,
            "P-F1_max": self.p_f1_max,
            "AUPRO": self.aupro,
        }

    def as_tte_vector(self) -> Tuple[float, ...]:
        values = self.as_tte_dict()
        return tuple(values[key] for key in TTE_METRIC_KEYS)


def compute_ad_metrics(
    inputs: MetricInputs,
    config: Optional[MetricConfig] = None,
) -> MetricScores:
    metric_config = config if config is not None else MetricConfig()
    image_scores = _float_array(inputs.image_scores).reshape(-1)
    image_labels = _bool_array(inputs.image_labels).reshape(-1)
    if len(image_scores) != len(image_labels):
        raise MetricInputError("image_scores and image_labels must have the same length")

    pixel_scores = inputs.pixel_scores
    pixel_masks = inputs.pixel_masks
    if pixel_scores is None or pixel_masks is None:
        return MetricScores(
            i_auroc=binary_auroc(image_labels, image_scores),
            i_ap=average_precision(image_labels, image_scores),
            i_f1_max=f1_score_max(image_labels, image_scores),
            p_auroc=float("nan"),
            p_ap=float("nan"),
            p_f1_max=float("nan"),
            aupro=float("nan"),
        )

    px_scores = _normalize_pixel_scores(pixel_scores)
    px_masks = _normalize_pixel_masks(pixel_masks)
    if px_scores.shape != px_masks.shape:
        raise MetricInputError("pixel_scores and pixel_masks must have the same shape")
    return MetricScores(
        i_auroc=binary_auroc(image_labels, image_scores),
        i_ap=average_precision(image_labels, image_scores),
        i_f1_max=f1_score_max(image_labels, image_scores),
        p_auroc=binary_auroc(px_masks.reshape(-1), px_scores.reshape(-1)),
        p_ap=average_precision(px_masks.reshape(-1), px_scores.reshape(-1)),
        p_f1_max=f1_score_max(px_masks.reshape(-1), px_scores.reshape(-1)),
        aupro=aupro_score(
            px_masks,
            px_scores,
            metric_config.aupro_max_fpr,
            metric_config.aupro_thresholds,
        ),
    )


def binary_auroc(labels: BoolArray, scores: FloatArray) -> float:
    labels = _bool_array(labels).reshape(-1)
    scores = _float_array(scores).reshape(-1)
    positives = _true_count(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    positive_rank_sum = _float64_sum(ranks[labels])
    numerator = positive_rank_sum - (positives * (positives + 1) / 2.0)
    return numerator / float(positives * negatives)


def average_precision(labels: BoolArray, scores: FloatArray) -> float:
    labels = _bool_array(labels).reshape(-1)
    scores = _float_array(scores).reshape(-1)
    positives = _true_count(labels)
    if positives == 0:
        return float("nan")
    order: npt.NDArray[np.intp] = np.argsort(-scores, kind="mergesort")
    sorted_labels: BoolArray = labels[order]
    sorted_scores: FloatArray = scores[order]
    true_positives: Float64Array = np.cumsum(sorted_labels.astype(np.float64))
    # Evaluate only after each equal-score group, matching sklearn/VisionAD AP.
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(scores) - 1]
    group_tp = true_positives[group_ends]
    group_precision = group_tp / (group_ends.astype(np.float64) + 1.0)
    group_positive = np.diff(np.r_[0.0, group_tp])
    return _float64_sum(group_positive * group_precision) / positives


def f1_score_max(labels: BoolArray, scores: FloatArray) -> float:
    labels = _bool_array(labels).reshape(-1)
    scores = _float_array(scores).reshape(-1)
    positives = _true_count(labels)
    if positives == 0:
        return float("nan")
    order: npt.NDArray[np.intp] = np.argsort(-scores, kind="mergesort")
    sorted_labels: BoolArray = labels[order]
    true_positives: Float64Array = np.cumsum(sorted_labels.astype(np.float64))
    false_positives: Float64Array = np.cumsum((~sorted_labels).astype(np.float64))
    precision: Float64Array = true_positives / np.maximum(
        true_positives + false_positives,
        1.0,
    )
    recall: Float64Array = true_positives / float(positives)
    f1: Float64Array = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(np.max(f1))


def _average_ranks(scores: FloatArray) -> Float64Array:
    order: npt.NDArray[np.intp] = np.argsort(scores, kind="mergesort")
    sorted_scores: FloatArray = scores[order]
    ranks: Float64Array = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _normalize_pixel_scores(scores: FloatArray) -> FloatArray:
    array = _float_array(scores)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 3:
        raise MetricInputError("pixel_scores must have shape (N, H, W) or (N, 1, H, W)")
    return array


def _normalize_pixel_masks(masks: BoolArray) -> BoolArray:
    array = _bool_array(masks)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 3:
        raise MetricInputError("pixel_masks must have shape (N, H, W) or (N, 1, H, W)")
    return array


def _float_array(values: npt.ArrayLike) -> FloatArray:
    array: FloatArray = np.asarray(values, dtype=np.float32)
    return array


def _bool_array(values: npt.ArrayLike) -> BoolArray:
    array: BoolArray = np.asarray(values, dtype=np.bool_)
    return array


def _true_count(values: BoolArray) -> int:
    return int(np.count_nonzero(values))


def _float64_sum(values: Float64Array) -> float:
    return float(np.sum(values))
