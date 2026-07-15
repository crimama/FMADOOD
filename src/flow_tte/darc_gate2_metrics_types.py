"""Frozen value objects and manifests for the DARC Gate 2 decision."""

from __future__ import annotations

from enum import Enum
from typing import Dict, Final, List, NamedTuple, Optional, Tuple, Union

from typing_extensions import Literal, final, override

JsonScalar = Union[str, int, float, bool, None]
JsonValue = Union[JsonScalar, Dict[str, "JsonValue"], List["JsonValue"]]

AD1_OBJECTS: Final = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)


class Gate2FailureCode(str, Enum):
    NORMAL_RESIDUAL = "NORMAL_RESIDUAL"
    SIGNAL_AP = "SIGNAL_AP"
    SIGNAL_COMPONENT_RECALL = "SIGNAL_COMPONENT_RECALL"
    R1_RETENTION = "R1_RETENTION"


@final
class InvalidGateInput(ValueError):  # noqa: N818 -- exact registered protocol status
    __slots__ = ("reason",)

    reason: str

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(str(self))

    @override
    def __str__(self) -> str:
        return f"INVALID_GATE_INPUT: {self.reason}"


class SourceAudit(NamedTuple):
    population_sha256: str
    support_sha256: str
    fallback_sha256: str
    mask_sha256: str

    def to_manifest(self) -> Dict[str, JsonValue]:
        return {
            "population_sha256": self.population_sha256,
            "support_sha256": self.support_sha256,
            "fallback_sha256": self.fallback_sha256,
            "mask_sha256": self.mask_sha256,
        }


class SourceMetric(NamedTuple):
    object_name: str
    seed: int
    fold_index: int
    source_id: str
    d_ap: float
    d_component_recall: float
    l1_responses: Tuple[float, float]
    r1_responses: Tuple[float, float]
    broad_pauroc_delta: float
    l0_audit: SourceAudit
    l1_audit: SourceAudit
    r1_audit: SourceAudit

    def to_manifest(self) -> Dict[str, JsonValue]:
        return {
            "object": self.object_name,
            "seed": self.seed,
            "fold": self.fold_index,
            "source": self.source_id,
            "d_ap": self.d_ap,
            "d_component_recall": self.d_component_recall,
            "l1_responses": list(self.l1_responses),
            "r1_responses": list(self.r1_responses),
            "broad_pauroc_delta": self.broad_pauroc_delta,
            "l0_audit": self.l0_audit.to_manifest(),
            "l1_audit": self.l1_audit.to_manifest(),
            "r1_audit": self.r1_audit.to_manifest(),
        }


class GroupResidual(NamedTuple):
    object_name: str
    seed: int
    source_ids: Tuple[str, ...]
    fold_indices: Tuple[int, ...]
    l0_p999: float
    l1_p999: float
    l0_population_sha256: str
    l1_population_sha256: str
    l0_residual_sha256: str
    l1_residual_sha256: str
    quantile: float = 0.999
    quantile_method: Literal["higher"] = "higher"

    def to_manifest(self) -> Dict[str, JsonValue]:
        return {
            "object": self.object_name,
            "seed": self.seed,
            "source_ids": list(self.source_ids),
            "fold_indices": list(self.fold_indices),
            "l0_p999": self.l0_p999,
            "l1_p999": self.l1_p999,
            "l0_population_sha256": self.l0_population_sha256,
            "l1_population_sha256": self.l1_population_sha256,
            "l0_residual_sha256": self.l0_residual_sha256,
            "l1_residual_sha256": self.l1_residual_sha256,
            "quantile": self.quantile,
            "quantile_method": self.quantile_method,
        }


class Gate2Config(NamedTuple):
    bootstrap_replicates: int = 10000
    bootstrap_seed: int = 20260710


class BootstrapLower(NamedTuple):
    d_ap: float
    d_component_recall: float


class Gate2Decision(NamedTuple):
    status: Literal["PASS", "FAIL"]
    passed: bool
    source_count: int
    group_count: int
    q_l0: float
    q_l1: float
    residual_ratio: Optional[float]
    ap_gain: float
    ap_ci_lower: float
    component_gain: float
    component_ci_lower: float
    s_l1: float
    s_r1: float
    retention_ratio: Optional[float]
    broad_pauroc_delta: float
    residual_pass: bool
    signal_ap_pass: bool
    signal_component_recall_pass: bool
    retention_pass: bool
    failure_codes: Tuple[Gate2FailureCode, ...]

    def to_manifest(self) -> Dict[str, JsonValue]:
        return {
            "status": self.status,
            "passed": self.passed,
            "source_count": self.source_count,
            "group_count": self.group_count,
            "q_l0": self.q_l0,
            "q_l1": self.q_l1,
            "residual_ratio": self.residual_ratio,
            "ap_gain": self.ap_gain,
            "ap_ci_lower": self.ap_ci_lower,
            "component_gain": self.component_gain,
            "component_ci_lower": self.component_ci_lower,
            "s_l1": self.s_l1,
            "s_r1": self.s_r1,
            "retention_ratio": self.retention_ratio,
            "broad_pauroc_delta": self.broad_pauroc_delta,
            "residual_pass": self.residual_pass,
            "signal_ap_pass": self.signal_ap_pass,
            "signal_component_recall_pass": self.signal_component_recall_pass,
            "retention_pass": self.retention_pass,
            "failure_codes": [code.value for code in self.failure_codes],
        }
