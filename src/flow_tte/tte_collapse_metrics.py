"""Patch-grid mechanism metrics for the TTE collapse decomposition.

These are deliberately NOT the frozen evaluation contract.  They are computed on
the DINOv3 patch grid rather than at original resolution, so absolute values are
not comparable with any recorded anchor.  Only within-run arm deltas are
interpretable, and every consumer must state that.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]

MAX_FPR = 0.05


class MetricError(ValueError):
    """Raised when a metric receives a degenerate population."""


@dataclass(frozen=True)
class ArmMetrics:
    patch_pauroc_005: float
    patch_best_f1: float
    shifted_good_inflation: float
    bad_good_separation: float
    stream_drift: float
    defect_score_mean: float
    normal_score_mean: float

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration across the numpy 1.x/2.x rename."""
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # numpy < 2.0
        integrate = np.trapz  # noqa: NPY201
    return float(integrate(y, x))


def _roc_counts(labels: BoolArray, scores: FloatArray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative true/false positive counts at each distinct descending score."""
    order = np.argsort(-scores.astype(np.float64), kind="mergesort")
    ranked_labels = labels[order].astype(np.float64)
    ranked_scores = scores[order].astype(np.float64)
    distinct = np.where(np.diff(ranked_scores))[0]
    cut = np.r_[distinct, ranked_labels.size - 1]
    true_positive = np.cumsum(ranked_labels)[cut]
    false_positive = (1.0 + cut) - true_positive
    return true_positive, false_positive


def partial_auroc(labels: BoolArray, scores: FloatArray) -> float:
    """McClish-standardized partial AUROC over FPR `[0, 0.05]`.

    Reproduces `sklearn.metrics.roc_auc_score(..., max_fpr=0.05)`, which the
    frozen evaluator uses, without adding a scikit-learn dependency to this
    diagnostic path.
    """
    if labels.size != scores.size:
        raise MetricError("labels and scores must align")
    positives = int(labels.sum())
    if positives in (0, labels.size):
        raise MetricError("partial AUROC needs both classes")
    true_positive, false_positive = _roc_counts(labels, scores)
    tpr = np.r_[0.0, true_positive / true_positive[-1]]
    fpr = np.r_[0.0, false_positive / false_positive[-1]]
    stop = int(np.searchsorted(fpr, MAX_FPR, side="right"))
    interpolated = np.interp(MAX_FPR, fpr[stop - 1 : stop + 1], tpr[stop - 1 : stop + 1])
    partial_tpr = np.r_[tpr[:stop], interpolated]
    partial_fpr = np.r_[fpr[:stop], MAX_FPR]
    area = float(_trapezoid(partial_tpr, partial_fpr))
    minimum = 0.5 * MAX_FPR * MAX_FPR
    return float(0.5 * (1.0 + (area - minimum) / (MAX_FPR - minimum)))


def best_f1(labels: BoolArray, scores: FloatArray) -> float:
    """Pooled oracle F1 over a single global threshold."""
    if int(labels.sum()) == 0:
        raise MetricError("oracle F1 needs at least one positive")
    true_positive, false_positive = _roc_counts(labels, scores)
    positives = float(labels.sum())
    precision = true_positive / np.maximum(true_positive + false_positive, 1.0)
    recall = true_positive / positives
    denominator = precision + recall
    f1 = np.zeros_like(precision)
    valid = denominator > 0
    f1[valid] = 2.0 * precision[valid] * recall[valid] / denominator[valid]
    return float(f1.max())


def compute_arm_metrics(
    scores: Sequence[FloatArray],
    defects: Sequence[BoolArray],
    conditions: Sequence[str],
    is_good: Sequence[bool],
    order: Sequence[int],
) -> ArmMetrics:
    """Pool every patch of every image, then derive the mechanism terms."""
    if not (len(scores) == len(defects) == len(conditions) == len(is_good)):
        raise MetricError("per-image sequences must align")
    pooled_scores = np.concatenate([value.reshape(-1) for value in scores])
    pooled_labels = np.concatenate([value.reshape(-1) for value in defects])
    good_regular: list[FloatArray] = []
    good_shifted: list[FloatArray] = []
    normal_good: list[FloatArray] = []
    per_image_normal_mean: list[float] = []
    for index, score in enumerate(scores):
        flat = score.reshape(-1)
        normal = flat[~defects[index].reshape(-1)]
        per_image_normal_mean.append(float(normal.mean()) if normal.size else float("nan"))
        if not is_good[index]:
            continue
        normal_good.append(flat)
        if conditions[index] == "regular":
            good_regular.append(flat)
        else:
            good_shifted.append(flat)
    defect_scores = pooled_scores[pooled_labels]
    normal_reference = np.concatenate(normal_good) if normal_good else pooled_scores[~pooled_labels]
    return ArmMetrics(
        patch_pauroc_005=partial_auroc(pooled_labels, pooled_scores),
        patch_best_f1=best_f1(pooled_labels, pooled_scores),
        shifted_good_inflation=_mean_gap(good_shifted, good_regular),
        bad_good_separation=float(defect_scores.mean() - normal_reference.mean()),
        stream_drift=_drift(per_image_normal_mean, order),
        defect_score_mean=float(defect_scores.mean()),
        normal_score_mean=float(normal_reference.mean()),
    )


def _mean_gap(left: Sequence[FloatArray], right: Sequence[FloatArray]) -> float:
    if not left or not right:
        return float("nan")
    return float(np.concatenate(left).mean() - np.concatenate(right).mean())


def _drift(per_image_normal_mean: Sequence[float], order: Sequence[int]) -> float:
    """Pearson correlation between stream position and per-image normal score.

    A static bank gives a value near zero.  A growing, contaminated bank makes
    later images systematically cheaper, which is what destroys a single pooled
    threshold.
    """
    values = np.asarray(per_image_normal_mean, dtype=np.float64)
    positions = np.empty(len(order), dtype=np.float64)
    for rank, image_index in enumerate(order):
        positions[image_index] = float(rank)
    finite = np.isfinite(values)
    if int(finite.sum()) < 3:
        return float("nan")
    values = values[finite]
    positions = positions[finite]
    if float(values.std()) == 0.0 or float(positions.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(positions, values)[0, 1])
