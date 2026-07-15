"""Pure source-map evaluation for the frozen DARC Gate 2 protocol."""

from __future__ import annotations

# NumPy 1.x bundled typing leaves reductions and indexing unknown on Python 3.8.
# pyright: reportMissingImports=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
import math
from typing import Final, Tuple, Union

import numpy as np

import flow_tte.darc_gate2_metrics_types as metrics
from flow_tte.darc_gate2_evaluation_types import (
    BoolMask,
    FloatMap,
    FoldCleanCalibration,
    FoldComponentThresholds,
    ProfileResponse,
    RungSourceMaps,
    SourceEvaluationInput,
    SourceEvaluationManifest,
    SourceEvaluationResult,
    SourceMasks,
)
from flow_tte.darc_protocol_eval import component_recall
from flow_tte.darc_rank_metrics import rank_binary_views

_COMPONENT_QUANTILE: Final = 0.9999
# Frozen extraction uses 512/1024 maps; registered thin masks peak at 242 pixels.
_MAX_NATIVE_PIXELS: Final = 1024 * 1024
_MAX_THIN_MASK_PIXELS: Final = 256
RuntimeScalar = Union[str, int, float, bool]
RuntimeAudit = Union[metrics.SourceAudit, Tuple[str, ...]]


def compute_fold_component_thresholds(
    calibration: FoldCleanCalibration,
) -> FoldComponentThresholds:
    """Compute L0/L1 higher-q=.9999 thresholds from exactly four clean maps."""
    l0, l1 = _validated_calibration(calibration)
    _require_identical_shapes((*l0, *l1))
    return FoldComponentThresholds(
        l0=_higher_quantile(l0),
        l1=_higher_quantile(l1),
    )


def evaluate_source_maps(source: SourceEvaluationInput) -> SourceEvaluationResult:
    """Evaluate one paired L0/L1/R1 source without mutation or post-processing."""
    _validate_identity(source)
    l0 = _validated_rung("L0", source.maps.l0)
    l1 = _validated_rung("L1", source.maps.l1)
    r1 = _validated_rung("R1", source.maps.r1)
    masks = _validated_masks(source.masks)
    calibration_l0, calibration_l1 = _validated_calibration(source.calibration)
    _require_identical_shapes(
        (
            *_rung_arrays(l0),
            *_rung_arrays(l1),
            *_rung_arrays(r1),
            *masks.thin_profiles,
            masks.broad,
            *calibration_l0,
            *calibration_l1,
        ),
    )
    _validate_binary_populations(masks)
    thresholds = FoldComponentThresholds(
        l0=_higher_quantile(calibration_l0),
        l1=_higher_quantile(calibration_l1),
    )

    l0_ap = _thin_ap(l0, masks)
    l1_ap = _thin_ap(l1, masks)
    l0_component = _thin_component_recall(l0, masks, thresholds.l0)
    l1_component = _thin_component_recall(l1, masks, thresholds.l1)
    l1_profiles = _profile_responses(l1, masks)
    r1_profiles = _profile_responses(r1, masks)
    broad_l0 = _broad_pauroc(l0, masks.broad)
    broad_r1 = _broad_pauroc(r1, masks.broad)
    broad_delta = broad_r1 - broad_l0

    metric = metrics.SourceMetric(
        object_name=source.object_name,
        seed=source.seed,
        fold_index=source.fold_index,
        source_id=source.source_id,
        d_ap=l1_ap - l0_ap,
        d_component_recall=l1_component - l0_component,
        l1_responses=(l1_profiles[0].response, l1_profiles[1].response),
        r1_responses=(r1_profiles[0].response, r1_profiles[1].response),
        broad_pauroc_delta=broad_delta,
        l0_audit=source.audit,
        l1_audit=source.audit,
        r1_audit=source.audit,
    )
    manifest = SourceEvaluationManifest(
        thresholds=thresholds,
        l0_ap=l0_ap,
        l1_ap=l1_ap,
        l0_component_recall=l0_component,
        l1_component_recall=l1_component,
        l1_profiles=l1_profiles,
        r1_profiles=r1_profiles,
        broad_l0_pauroc_005=broad_l0,
        broad_r1_pauroc_005=broad_r1,
        broad_pauroc_delta=broad_delta,
    )
    return SourceEvaluationResult(metric=metric, manifest=manifest)


def _validated_rung(name: str, rung: RungSourceMaps) -> RungSourceMaps:
    if len(rung.thin_profiles) != 2:
        reason = f"{name} requires exactly two thin profiles"
        raise metrics.InvalidGateInput(reason)
    return RungSourceMaps(
        clean=_validated_map(f"{name} clean", rung.clean),
        thin_profiles=(
            _validated_map(f"{name} thin 0", rung.thin_profiles[0]),
            _validated_map(f"{name} thin 1", rung.thin_profiles[1]),
        ),
        broad=_validated_map(f"{name} broad", rung.broad),
    )


def _validated_calibration(
    calibration: FoldCleanCalibration,
) -> Tuple[Tuple[FloatMap, ...], Tuple[FloatMap, ...]]:
    if len(calibration.l0) != 4 or len(calibration.l1) != 4:
        raise metrics.InvalidGateInput("L0 and L1 each require exactly four clean maps")
    l0 = tuple(
        _validated_map(f"L0 calibration {index}", value)
        for index, value in enumerate(calibration.l0)
    )
    l1 = tuple(
        _validated_map(f"L1 calibration {index}", value)
        for index, value in enumerate(calibration.l1)
    )
    return l0, l1


def _validated_masks(masks: SourceMasks) -> SourceMasks:
    if len(masks.thin_profiles) != 2:
        raise metrics.InvalidGateInput("masks require exactly two thin profiles")
    return SourceMasks(
        thin_profiles=(
            _validated_mask("thin mask 0", masks.thin_profiles[0]),
            _validated_mask("thin mask 1", masks.thin_profiles[1]),
        ),
        broad=_validated_mask("broad mask", masks.broad),
    )


def _validated_map(name: str, value: FloatMap) -> FloatMap:
    raw = np.asarray(value)
    _validate_extent(name, raw, "map")
    if raw.dtype.kind not in ("f", "i", "u"):
        reason = f"{name} must have a real numeric dtype"
        raise metrics.InvalidGateInput(reason)
    array: FloatMap = np.asarray(raw, dtype=np.float32)
    if not bool(np.all(np.isfinite(array))):
        reason = f"{name} must contain only finite values"
        raise metrics.InvalidGateInput(reason)
    return array


def _validated_mask(name: str, value: BoolMask) -> BoolMask:
    raw = np.asarray(value)
    _validate_extent(name, raw, "mask")
    if raw.dtype != np.dtype(np.bool_):
        reason = f"{name} must have boolean dtype"
        raise metrics.InvalidGateInput(reason)
    mask: BoolMask = np.asarray(raw, dtype=np.bool_)
    if not bool(np.any(mask)):
        reason = f"{name} must contain a positive pixel"
        raise metrics.InvalidGateInput(reason)
    return mask


def _validate_extent(name: str, value: np.ndarray, kind: str) -> None:
    if value.ndim != 2 or value.size == 0:
        reason = f"{name} must be a non-empty 2D {kind}"
        raise metrics.InvalidGateInput(reason)
    if value.size > _MAX_NATIVE_PIXELS:
        reason = f"{name} exceeds the 1024-by-1024 native-map limit"
        raise metrics.InvalidGateInput(reason)


def _validate_identity(source: SourceEvaluationInput) -> None:
    if not _is_nonempty_string(source.object_name) or not _is_nonempty_string(source.source_id):
        raise metrics.InvalidGateInput("object and source identities must be non-empty")
    integers_valid = _is_plain_integer(source.seed) and _is_plain_integer(source.fold_index)
    if not integers_valid or source.seed < 0 or source.fold_index not in range(4):
        raise metrics.InvalidGateInput("seed must be non-negative and fold must be in [0, 3]")
    if not _is_source_audit(source.audit):
        raise metrics.InvalidGateInput("source audit must use the registered audit type")
    if any(not _is_nonempty_string(value) for value in source.audit):
        raise metrics.InvalidGateInput("source audit identifiers must be non-empty")


def _is_nonempty_string(value: RuntimeScalar) -> bool:
    return isinstance(value, str) and bool(value)


def _is_plain_integer(value: RuntimeScalar) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_source_audit(value: RuntimeAudit) -> bool:
    return isinstance(value, metrics.SourceAudit)


def _validate_binary_populations(masks: SourceMasks) -> None:
    thin = np.concatenate(tuple(mask.reshape(-1) for mask in masks.thin_profiles))
    if bool(np.all(thin)):
        raise metrics.InvalidGateInput("thin masks require background pixels for AP")
    if any(int(np.count_nonzero(mask)) > _MAX_THIN_MASK_PIXELS for mask in masks.thin_profiles):
        raise metrics.InvalidGateInput("thin masks exceed the registered 256-pixel cue limit")
    if bool(np.all(masks.broad)):
        raise metrics.InvalidGateInput("broad mask requires background pixels for pAUROC")


def _require_identical_shapes(arrays: Tuple[np.ndarray, ...]) -> None:
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise metrics.InvalidGateInput(
            "all source, mask, and calibration maps require identical shapes",
        )


def _rung_arrays(rung: RungSourceMaps) -> Tuple[FloatMap, ...]:
    return (rung.clean, *rung.thin_profiles, rung.broad)


def _higher_quantile(clean_maps: Tuple[FloatMap, ...]) -> float:
    values = np.concatenate(tuple(score_map.reshape(-1) for score_map in clean_maps))
    index = math.ceil(_COMPONENT_QUANTILE * (values.size - 1))
    partitioned: FloatMap = np.asarray(np.partition(values, index), dtype=np.float32)
    normalized: FloatMap = np.asarray(partitioned[index : index + 1], dtype=np.float32)
    return float(normalized.item())


def _thin_ap(rung: RungSourceMaps, masks: SourceMasks) -> float:
    labels: BoolMask = np.concatenate(tuple(mask.reshape(-1) for mask in masks.thin_profiles))
    scores: FloatMap = np.concatenate(
        tuple(score_map.reshape(-1) for score_map in rung.thin_profiles),
    )
    population: BoolMask = np.ones(labels.shape, dtype=np.bool_)
    return rank_binary_views(labels, scores, population).all_test.p_ap


def _thin_component_recall(
    rung: RungSourceMaps,
    masks: SourceMasks,
    threshold: float,
) -> float:
    predictions: BoolMask = np.asarray(
        np.stack(tuple(score_map > threshold for score_map in rung.thin_profiles), axis=0),
        dtype=np.bool_,
    )
    exact_masks: BoolMask = np.asarray(np.stack(masks.thin_profiles, axis=0), dtype=np.bool_)
    return component_recall(exact_masks, predictions)


def _profile_responses(
    rung: RungSourceMaps,
    masks: SourceMasks,
) -> Tuple[ProfileResponse, ProfileResponse]:
    return (
        _profile_response(rung.clean, rung.thin_profiles[0], masks.thin_profiles[0]),
        _profile_response(rung.clean, rung.thin_profiles[1], masks.thin_profiles[1]),
    )


def _profile_response(clean: FloatMap, cue: FloatMap, mask: BoolMask) -> ProfileResponse:
    clean_mean = float(np.mean(clean[mask], dtype=np.float64))
    cue_mean = float(np.mean(cue[mask], dtype=np.float64))
    return ProfileResponse(
        clean_mean=clean_mean,
        cue_mean=cue_mean,
        response=cue_mean - clean_mean,
    )


def _broad_pauroc(rung: RungSourceMaps, mask: BoolMask) -> float:
    labels: BoolMask = mask.reshape(-1)
    scores: FloatMap = rung.broad.reshape(-1)
    population: BoolMask = np.ones(labels.shape, dtype=np.bool_)
    return rank_binary_views(labels, scores, population).all_test.p_auroc_005
