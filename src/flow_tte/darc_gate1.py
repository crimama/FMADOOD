from __future__ import annotations

# pyright: reportMissingImports=false
from dataclasses import dataclass
from typing import Final, Optional, Sequence, Tuple, TypedDict

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from flow_tte.darc_gate1_stability import ThresholdStability, ThresholdStabilityManifest

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]

GATE1_METHOD_VERSION: Final = "darc-gate1-v1"


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class DarcGate1Error(ValueError):
    reason: str

    @override
    def __str__(self) -> str:
        return f"Invalid DARC Gate1 input: {self.reason}"


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class SourceConditions:
    query_maps: Tuple[FloatArray, ...]
    reference_scores: FloatArray
    calibration_maps: Tuple[FloatArray, ...]


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class SourceEvaluationInput:
    object_name: str
    source_id: str
    seed: int
    fold_index: int
    masks: Tuple[BoolArray, ...]
    low: SourceConditions
    bilinear_null: SourceConditions
    high: SourceConditions


class ConditionMetricManifest(TypedDict):
    ap: float
    p_auroc_005: float
    component_recall: float
    fixed_threshold: float
    clean_fpr: float
    stability: ThresholdStabilityManifest


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class ConditionMetric:
    ap: float
    p_auroc_005: float
    component_recall: float
    fixed_threshold: float
    clean_fpr: float
    stability: ThresholdStability

    def to_manifest(self) -> ConditionMetricManifest:
        return {
            "ap": self.ap,
            "p_auroc_005": self.p_auroc_005,
            "component_recall": self.component_recall,
            "fixed_threshold": self.fixed_threshold,
            "clean_fpr": self.clean_fpr,
            "stability": self.stability.to_manifest(),
        }


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class SourceMetric:
    object_name: str
    source_id: str
    seed: int
    fold_index: int
    low: ConditionMetric
    bilinear_null: ConditionMetric
    high: ConditionMetric


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class Gate1Thresholds:
    ap_absolute_gain: float = 0.005
    ap_relative_gain: float = 0.50
    component_gain: float = 0.10
    maximum_control_pauroc_loss: float = 0.005
    normal_quantile: float = 0.9999
    bootstrap_replicates: int = 10000
    bootstrap_seed: int = 20260710
    maximum_stability_median_fpr: float = 2e-4
    maximum_stability_fpr: float = 1e-3
    maximum_stability_iqr_ratio: float = 0.25

    def __post_init__(self) -> None:
        gains_valid = (
            self.ap_absolute_gain >= 0.0
            and self.ap_relative_gain >= 0.0
            and self.component_gain >= 0.0
            and self.maximum_control_pauroc_loss >= 0.0
            and self.maximum_stability_median_fpr >= 0.0
            and self.maximum_stability_fpr >= 0.0
            and self.maximum_stability_iqr_ratio >= 0.0
        )
        if not gains_valid or not 0.0 <= self.normal_quantile <= 1.0:
            raise DarcGate1Error("gain and quantile thresholds are outside their ranges")
        if self.bootstrap_replicates < 1 or self.bootstrap_seed < 0:
            raise DarcGate1Error("bootstrap settings must be positive")


class Gate1DecisionManifest(TypedDict):
    passed: bool
    source_count: int
    ap_gain: float
    ap_relative_gain: float
    ap_ci_lower: float
    component_gain: float
    component_ci_lower: float
    control_pauroc_gain: float
    high_vs_null_ap_gain: float
    threshold_stability: ThresholdStabilityManifest
    deployable_fixed_f1_allowed: bool


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class Gate1Decision:
    passed: bool
    source_count: int
    ap_gain: float
    ap_relative_gain: float
    ap_ci_lower: float
    component_gain: float
    component_ci_lower: float
    control_pauroc_gain: float
    high_vs_null_ap_gain: float
    threshold_stability: ThresholdStability
    deployable_fixed_f1_allowed: bool

    def to_manifest(self) -> Gate1DecisionManifest:
        return {
            "passed": self.passed,
            "source_count": self.source_count,
            "ap_gain": self.ap_gain,
            "ap_relative_gain": self.ap_relative_gain,
            "ap_ci_lower": self.ap_ci_lower,
            "component_gain": self.component_gain,
            "component_ci_lower": self.component_ci_lower,
            "control_pauroc_gain": self.control_pauroc_gain,
            "high_vs_null_ap_gain": self.high_vs_null_ap_gain,
            "threshold_stability": self.threshold_stability.to_manifest(),
            "deployable_fixed_f1_allowed": self.deployable_fixed_f1_allowed,
        }


def evaluate_source(
    inputs: SourceEvaluationInput,
    thresholds: Optional[Gate1Thresholds] = None,
) -> SourceMetric:
    """Evaluate paired conditions for one held-out normal source image."""
    active = thresholds if thresholds is not None else Gate1Thresholds()
    _validate_source(inputs)
    from flow_tte.darc_gate1_metrics import evaluate_condition  # noqa: PLC0415

    return SourceMetric(
        object_name=inputs.object_name,
        source_id=inputs.source_id,
        seed=inputs.seed,
        fold_index=inputs.fold_index,
        low=evaluate_condition(inputs.low, inputs.masks, active),
        bilinear_null=evaluate_condition(inputs.bilinear_null, inputs.masks, active),
        high=evaluate_condition(inputs.high, inputs.masks, active),
    )


def decide_gate1(
    metrics: Sequence[SourceMetric],
    thresholds: Optional[Gate1Thresholds] = None,
) -> Gate1Decision:
    """Apply the registered paired resolution decision with a source bootstrap."""
    active = thresholds if thresholds is not None else Gate1Thresholds()
    from flow_tte.darc_gate1_metrics import aggregate_stability, bootstrap_lower  # noqa: PLC0415

    if not metrics:
        raise DarcGate1Error("at least one source metric is required")
    ap_low = np.asarray([item.low.ap for item in metrics], dtype=np.float64)
    ap_delta = np.asarray([item.high.ap - item.low.ap for item in metrics], dtype=np.float64)
    component_delta = np.asarray(
        [item.high.component_recall - item.low.component_recall for item in metrics],
        dtype=np.float64,
    )
    control_delta = np.asarray(
        [item.high.p_auroc_005 - item.low.p_auroc_005 for item in metrics],
        dtype=np.float64,
    )
    null_delta = np.asarray(
        [item.high.ap - item.bilinear_null.ap for item in metrics],
        dtype=np.float64,
    )
    ap_gain = float(np.mean(ap_delta))
    ap_relative = float(ap_gain / max(float(np.mean(ap_low)), np.finfo(np.float64).eps))
    component_gain = float(np.mean(component_delta))
    control_gain = float(np.mean(control_delta))
    strata = tuple((item.object_name, item.seed) for item in metrics)
    ap_lower = bootstrap_lower(ap_delta, strata, active)
    component_lower = bootstrap_lower(component_delta, strata, active)
    ap_branch = (
        ap_gain >= active.ap_absolute_gain
        and ap_relative >= active.ap_relative_gain
        and ap_lower > 0.0
    )
    component_branch = component_gain >= active.component_gain and component_lower > 0.0
    passed = bool(
        (ap_branch or component_branch)
        and control_gain >= -active.maximum_control_pauroc_loss,
    )
    stability_by_fold = {}
    for item in metrics:
        key = (item.object_name, item.seed, item.fold_index)
        prior = stability_by_fold.setdefault(key, item.high.stability)
        if prior != item.high.stability:
            raise DarcGate1Error("one fold has inconsistent threshold-stability rotations")
    threshold_stability = aggregate_stability(tuple(stability_by_fold.values()), active)
    return Gate1Decision(
        passed=passed,
        source_count=len(metrics),
        ap_gain=ap_gain,
        ap_relative_gain=ap_relative,
        ap_ci_lower=ap_lower,
        component_gain=component_gain,
        component_ci_lower=component_lower,
        control_pauroc_gain=control_gain,
        high_vs_null_ap_gain=float(np.mean(null_delta)),
        threshold_stability=threshold_stability,
        deployable_fixed_f1_allowed=threshold_stability.stable,
    )


def _validate_source(inputs: SourceEvaluationInput) -> None:
    identity_valid = bool(inputs.object_name and inputs.source_id and inputs.seed >= 0)
    if len(inputs.masks) != 4 or not identity_valid:
        raise DarcGate1Error("one clean and three cue masks plus an identity are required")
    shape = inputs.masks[0].shape
    conditions = (inputs.low, inputs.bilinear_null, inputs.high)
    valid_masks = all(mask.dtype == np.bool_ and mask.shape == shape for mask in inputs.masks)
    valid_maps = all(
        len(condition.query_maps) == len(inputs.masks)
        and all(score_map.shape == shape for score_map in condition.query_maps)
        and condition.reference_scores.size > 0
        and len(condition.calibration_maps) == 4
        and all(score_map.size > 0 for score_map in condition.calibration_maps)
        for condition in conditions
    )
    if not valid_masks or not valid_maps or np.any(inputs.masks[0]):
        raise DarcGate1Error("source masks, maps, and normal reference shapes are inconsistent")
