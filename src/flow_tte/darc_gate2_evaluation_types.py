"""Typed source-map inputs and manifests for DARC Gate 2 evaluation."""

from __future__ import annotations

# pyright: reportMissingImports=false
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
import numpy.typing as npt

from flow_tte.darc_gate2_metrics_types import JsonValue, SourceAudit, SourceMetric

FloatMap = npt.NDArray[np.float32]
BoolMask = npt.NDArray[np.bool_]


class RungSourceMaps(NamedTuple):
    """One rung's native clean, two-thin, and broad source maps."""

    clean: FloatMap
    thin_profiles: Tuple[FloatMap, ...]
    broad: FloatMap


class SourceMapBundle(NamedTuple):
    l0: RungSourceMaps
    l1: RungSourceMaps
    r1: RungSourceMaps


class SourceMasks(NamedTuple):
    thin_profiles: Tuple[BoolMask, ...]
    broad: BoolMask


class FoldCleanCalibration(NamedTuple):
    """The four held-out-clean native maps for each thresholded rung."""

    l0: Tuple[FloatMap, ...]
    l1: Tuple[FloatMap, ...]


class FoldComponentThresholds(NamedTuple):
    l0: float
    l1: float

    def to_manifest(self) -> Dict[str, JsonValue]:
        return {"l0": self.l0, "l1": self.l1}


class ProfileResponse(NamedTuple):
    clean_mean: float
    cue_mean: float
    response: float

    def to_manifest(self) -> Dict[str, JsonValue]:
        return {
            "clean_mean": self.clean_mean,
            "cue_mean": self.cue_mean,
            "response": self.response,
        }


class SourceEvaluationManifest(NamedTuple):
    thresholds: FoldComponentThresholds
    l0_ap: float
    l1_ap: float
    l0_component_recall: float
    l1_component_recall: float
    l1_profiles: Tuple[ProfileResponse, ProfileResponse]
    r1_profiles: Tuple[ProfileResponse, ProfileResponse]
    broad_l0_pauroc_005: float
    broad_r1_pauroc_005: float
    broad_pauroc_delta: float

    def to_manifest(self) -> Dict[str, JsonValue]:
        ap: Dict[str, JsonValue] = {"l0": self.l0_ap, "l1": self.l1_ap}
        component: Dict[str, JsonValue] = {
            "l0": self.l0_component_recall,
            "l1": self.l1_component_recall,
        }
        broad: Dict[str, JsonValue] = {
            "l0": self.broad_l0_pauroc_005,
            "r1": self.broad_r1_pauroc_005,
            "delta_r1_l0": self.broad_pauroc_delta,
        }
        l1_profiles: List[JsonValue] = [row.to_manifest() for row in self.l1_profiles]
        r1_profiles: List[JsonValue] = [row.to_manifest() for row in self.r1_profiles]
        return {
            "thresholds": self.thresholds.to_manifest(),
            "absolute_ap": ap,
            "absolute_component_recall": component,
            "l1_profiles": l1_profiles,
            "r1_profiles": r1_profiles,
            "broad_pauroc_005": broad,
        }


class SourceEvaluationInput(NamedTuple):
    object_name: str
    seed: int
    fold_index: int
    source_id: str
    maps: SourceMapBundle
    masks: SourceMasks
    calibration: FoldCleanCalibration
    audit: SourceAudit


class SourceEvaluationResult(NamedTuple):
    metric: SourceMetric
    manifest: SourceEvaluationManifest
