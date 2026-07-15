from __future__ import annotations

# pyright: reportMissingImports=false
import math
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from sklearn.metrics import average_precision_score, roc_auc_score

from flow_tte.darc_gate1 import (
    ConditionMetric,
    DarcGate1Error,
    Gate1Thresholds,
    SourceConditions,
)
from flow_tte.darc_gate1_stability import ThresholdRotation, ThresholdStability
from flow_tte.darc_protocol_eval import component_recall

FloatArray = npt.NDArray[np.float32]


def evaluate_condition(
    condition: SourceConditions,
    masks: Tuple[npt.NDArray[np.bool_], ...],
    thresholds: Gate1Thresholds,
) -> ConditionMetric:
    """Evaluate thin cues while reserving clean and broad maps for controls."""
    reference = np.asarray(condition.reference_scores, dtype=np.float32).reshape(-1)
    ordered = reference if np.all(reference[:-1] <= reference[1:]) else np.sort(reference)
    evidence_maps = tuple(
        _upper_tail_from_ordered(ordered, np.asarray(score_map, dtype=np.float32))
        for score_map in condition.query_maps
    )
    calibration_evidence = tuple(
        _upper_tail_from_ordered(ordered, np.asarray(score_map, dtype=np.float32))
        for score_map in condition.calibration_maps
    )
    calibration_pixels = np.concatenate(
        tuple(score_map.reshape(-1) for score_map in calibration_evidence),
    )
    fixed_threshold = _higher_quantile(calibration_pixels, thresholds.normal_quantile)
    stability = aggregate_stability(
        (
            _rotation_stability(calibration_evidence, thresholds),
        ),
        thresholds,
    )
    thin_labels = np.concatenate(tuple(mask.reshape(-1) for mask in masks[1:3]))
    thin_scores = np.concatenate(tuple(score_map.reshape(-1) for score_map in evidence_maps[1:3]))
    thin_predictions = tuple(score_map > fixed_threshold for score_map in evidence_maps[1:3])
    broad_labels = masks[3].reshape(-1)
    broad_scores = evidence_maps[3].reshape(-1)
    return ConditionMetric(
        ap=float(average_precision_score(thin_labels, thin_scores)),
        p_auroc_005=float(roc_auc_score(broad_labels, broad_scores, max_fpr=0.05)),
        component_recall=component_recall(
            np.stack(masks[1:3], axis=0),
            np.stack(thin_predictions, axis=0),
        ),
        fixed_threshold=fixed_threshold,
        clean_fpr=float(np.mean(evidence_maps[0] > fixed_threshold)),
        stability=stability,
    )


def aggregate_stability(
    diagnostics: Tuple[ThresholdStability, ...],
    thresholds: Gate1Thresholds,
) -> ThresholdStability:
    """Apply the frozen stability rule across unique folds and seeds."""
    rotations = tuple(row for diagnostic in diagnostics for row in diagnostic.rotations)
    if not rotations:
        raise DarcGate1Error("threshold stability requires at least one rotation")
    fprs = np.asarray([row.heldout_clean_fpr for row in rotations], dtype=np.float64)
    values = np.asarray([row.threshold for row in rotations], dtype=np.float64)
    quartiles = np.quantile(values, (0.25, 0.5, 0.75))
    threshold_median = float(quartiles[1])
    threshold_iqr = float(quartiles[2] - quartiles[0])
    epsilon = np.finfo(np.float64).eps
    if threshold_median > epsilon:
        ratio = threshold_iqr / threshold_median
    else:
        ratio = 0.0 if threshold_iqr <= epsilon else float(np.finfo(np.float64).max)
    median_fpr = float(np.median(fprs))
    maximum_fpr = float(np.max(fprs))
    stable = bool(
        median_fpr <= thresholds.maximum_stability_median_fpr
        and maximum_fpr <= thresholds.maximum_stability_fpr
        and ratio <= thresholds.maximum_stability_iqr_ratio,
    )
    criterion = (
        "unstable iff median heldout FPR > 2e-4, any heldout FPR > 1e-3, "
        "or threshold IQR/median > 0.25"
    )
    return ThresholdStability(
        rotations=rotations,
        median_fpr=median_fpr,
        maximum_fpr=maximum_fpr,
        threshold_median=threshold_median,
        threshold_iqr=threshold_iqr,
        threshold_iqr_ratio=float(ratio),
        stable=stable,
        criterion=criterion,
    )


def _rotation_stability(
    evidence_maps: Tuple[FloatArray, ...],
    thresholds: Gate1Thresholds,
) -> ThresholdStability:
    rotations = []
    for heldout_index, heldout in enumerate(evidence_maps):
        calibration = np.concatenate(
            tuple(
                score_map.reshape(-1)
                for index, score_map in enumerate(evidence_maps)
                if index != heldout_index
            ),
        )
        threshold = _higher_quantile(calibration, thresholds.normal_quantile)
        rotations.append(
            ThresholdRotation(
                heldout_index=heldout_index,
                threshold=threshold,
                heldout_clean_fpr=float(np.mean(heldout > threshold)),
            ),
        )
    placeholder = ThresholdStability(
        rotations=tuple(rotations),
        median_fpr=0.0,
        maximum_fpr=0.0,
        threshold_median=0.0,
        threshold_iqr=0.0,
        threshold_iqr_ratio=0.0,
        stable=True,
        criterion="",
    )
    return aggregate_stability((placeholder,), thresholds)


def bootstrap_lower(
    values: npt.NDArray[np.float64],
    strata: Tuple[Tuple[str, int], ...],
    thresholds: Gate1Thresholds,
) -> float:
    grouped: Dict[Tuple[str, int], List[int]] = {}
    for index, key in enumerate(strata):
        grouped.setdefault(key, []).append(index)
    generator = np.random.default_rng(thresholds.bootstrap_seed)
    replicates = np.empty(thresholds.bootstrap_replicates, dtype=np.float64)
    for replicate in range(thresholds.bootstrap_replicates):
        means = []
        for indices in grouped.values():
            sampled = generator.choice(indices, size=len(indices), replace=True)
            means.append(float(np.mean(values[sampled])))
        replicates[replicate] = float(np.mean(means))
    return float(np.quantile(replicates, 0.025))


def _higher_quantile(values: FloatArray, quantile: float) -> float:
    index = math.ceil(quantile * (values.size - 1))
    return float(np.partition(values, index)[index])


def _upper_tail_from_ordered(reference: FloatArray, values: FloatArray) -> FloatArray:
    lower_indices = np.searchsorted(reference, values, side="left")
    upper_counts = reference.size - lower_indices
    probabilities = (1.0 + upper_counts) / float(reference.size + 1)
    return np.asarray(-np.log(probabilities), dtype=np.float32)
