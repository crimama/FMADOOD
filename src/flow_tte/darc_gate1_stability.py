from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, TypedDict


class ThresholdRotationManifest(TypedDict):
    heldout_index: int
    threshold: float
    heldout_clean_fpr: float


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class ThresholdRotation:
    heldout_index: int
    threshold: float
    heldout_clean_fpr: float

    def to_manifest(self) -> ThresholdRotationManifest:
        return {
            "heldout_index": self.heldout_index,
            "threshold": self.threshold,
            "heldout_clean_fpr": self.heldout_clean_fpr,
        }


class ThresholdStabilityManifest(TypedDict):
    rotations: Tuple[ThresholdRotationManifest, ...]
    median_fpr: float
    maximum_fpr: float
    threshold_median: float
    threshold_iqr: float
    threshold_iqr_ratio: float
    stable: bool
    criterion: str


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class ThresholdStability:
    rotations: Tuple[ThresholdRotation, ...]
    median_fpr: float
    maximum_fpr: float
    threshold_median: float
    threshold_iqr: float
    threshold_iqr_ratio: float
    stable: bool
    criterion: str

    def to_manifest(self) -> ThresholdStabilityManifest:
        return {
            "rotations": tuple(row.to_manifest() for row in self.rotations),
            "median_fpr": self.median_fpr,
            "maximum_fpr": self.maximum_fpr,
            "threshold_median": self.threshold_median,
            "threshold_iqr": self.threshold_iqr,
            "threshold_iqr_ratio": self.threshold_iqr_ratio,
            "stable": self.stable,
            "criterion": self.criterion,
        }
