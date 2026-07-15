"""Strict paired metrics and decision contract for DARC Gate 2."""

from __future__ import annotations

# NumPy 1.x bundled typing leaves reduction/index dtypes unknown under Python 3.8.
# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false
import math
from typing import Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

import numpy as np
import numpy.typing as npt

from flow_tte.darc_gate2_metrics_types import (
    AD1_OBJECTS,
    BootstrapLower,
    Gate2Config,
    Gate2Decision,
    Gate2FailureCode,
    GroupResidual,
    InvalidGateInput,
    SourceAudit,
    SourceMetric,
)

__all__ = (
    "AD1_OBJECTS",
    "BootstrapLower",
    "Gate2Config",
    "Gate2Decision",
    "Gate2FailureCode",
    "GroupResidual",
    "InvalidGateInput",
    "SourceAudit",
    "SourceMetric",
    "decide_gate2",
    "higher_p999",
    "paired_bootstrap_lower",
)


class _ValidatedPopulation(NamedTuple):
    d_ap: npt.NDArray[np.float64]
    d_component_recall: npt.NDArray[np.float64]
    q_l0: npt.NDArray[np.float64]
    q_l1: npt.NDArray[np.float64]


def higher_p999(values: npt.NDArray[np.float64]) -> float:
    residuals: npt.NDArray[np.float64] = np.asarray(values, dtype=np.float64).reshape(-1)
    if residuals.size == 0 or not bool(np.all(np.isfinite(residuals))):
        raise InvalidGateInput("raw residual population is empty or non-finite")
    index = int(np.ceil(0.999 * (residuals.size - 1)))
    return float(np.partition(residuals, index)[index])


def paired_bootstrap_lower(
    d_ap: npt.NDArray[np.float64],
    d_component_recall: npt.NDArray[np.float64],
    config: Gate2Config,
) -> BootstrapLower:
    if config.bootstrap_replicates < 1 or config.bootstrap_seed < 0:
        raise InvalidGateInput("bootstrap configuration must be positive")
    if d_ap.shape != (45, 16) or d_component_recall.shape != (45, 16):
        raise InvalidGateInput("bootstrap matrices must have shape (45, 16)")
    if not bool(np.all(np.isfinite(d_ap))) or not bool(np.all(np.isfinite(d_component_recall))):
        raise InvalidGateInput("bootstrap matrices contain non-finite values")
    generator = np.random.Generator(np.random.PCG64(config.bootstrap_seed))
    shape = (config.bootstrap_replicates, 45, 16)
    indices: npt.NDArray[np.int64] = generator.integers(0, 16, size=shape, dtype=np.int64)
    group_indices: npt.NDArray[np.int64] = np.arange(45, dtype=np.int64)[None, :, None]
    sampled_ap: npt.NDArray[np.float64] = d_ap[group_indices, indices]
    ap_replicates: npt.NDArray[np.float64] = np.mean(sampled_ap, axis=(1, 2))
    component_replicates: npt.NDArray[np.float64] = np.mean(
        d_component_recall[group_indices, indices],
        axis=(1, 2),
    )
    return BootstrapLower(
        d_ap=_linear_lower(ap_replicates),
        d_component_recall=_linear_lower(component_replicates),
    )


def decide_gate2(
    sources: Sequence[SourceMetric],
    groups: Sequence[GroupResidual],
    config: Optional[Gate2Config] = None,
) -> Gate2Decision:
    """Validate the complete frozen population and apply every Gate 2 conjunct."""
    active = config if config is not None else Gate2Config()
    population = _validate_population(sources, groups)
    lower = paired_bootstrap_lower(population.d_ap, population.d_component_recall, active)
    q_l0 = math.fsum(population.q_l0) / 45.0
    q_l1 = math.fsum(population.q_l1) / 45.0
    s_l1 = math.fsum(math.fsum(row.l1_responses) / 2.0 for row in sources) / 720.0
    s_r1 = math.fsum(math.fsum(row.r1_responses) / 2.0 for row in sources) / 720.0
    residual_pass = bool(q_l0 > 0.0 and q_l1 <= 0.80 * q_l0)
    signal_ap_pass = bool(lower.d_ap > 0.0)
    signal_component_pass = bool(lower.d_component_recall > 0.0)
    retention_pass = bool(s_l1 > 0.0 and s_r1 >= 0.90 * s_l1)
    checks = (
        (residual_pass, Gate2FailureCode.NORMAL_RESIDUAL),
        (signal_ap_pass, Gate2FailureCode.SIGNAL_AP),
        (signal_component_pass, Gate2FailureCode.SIGNAL_COMPONENT_RECALL),
        (retention_pass, Gate2FailureCode.R1_RETENTION),
    )
    failure_codes = tuple(code for passed, code in checks if not passed)
    passed = not failure_codes
    return Gate2Decision(
        status="PASS" if passed else "FAIL",
        passed=passed,
        source_count=len(sources),
        group_count=len(groups),
        q_l0=q_l0,
        q_l1=q_l1,
        residual_ratio=q_l1 / q_l0 if q_l0 > 0.0 else None,
        ap_gain=math.fsum(row.d_ap for row in sources) / 720.0,
        ap_ci_lower=lower.d_ap,
        component_gain=math.fsum(row.d_component_recall for row in sources) / 720.0,
        component_ci_lower=lower.d_component_recall,
        s_l1=s_l1,
        s_r1=s_r1,
        retention_ratio=s_r1 / s_l1 if s_l1 > 0.0 else None,
        broad_pauroc_delta=math.fsum(row.broad_pauroc_delta for row in sources) / 720.0,
        residual_pass=residual_pass,
        signal_ap_pass=signal_ap_pass,
        signal_component_recall_pass=signal_component_pass,
        retention_pass=retention_pass,
        failure_codes=failure_codes,
    )


def _validate_population(
    sources: Sequence[SourceMetric],
    groups: Sequence[GroupResidual],
) -> _ValidatedPopulation:
    if len(sources) != 720 or len(groups) != 45:
        raise InvalidGateInput("expected exactly 720 source rows and 45 residual groups")
    expected_groups = tuple(
        (object_name, seed) for object_name in AD1_OBJECTS for seed in (0, 1, 2)
    )
    sources_by_group = _source_groups(sources)
    groups_by_key = _residual_groups(groups)
    expected_keys = set(expected_groups)
    if set(sources_by_group) != expected_keys or set(groups_by_key) != expected_keys:
        raise InvalidGateInput("object names or seeds differ from the frozen AD1 population")
    ordered_sources: List[List[SourceMetric]] = []
    ordered_groups: List[GroupResidual] = []
    for key in expected_groups:
        group = groups_by_key[key]
        rows = sorted(sources_by_group[key], key=lambda row: row.source_id)
        _validate_group(group, rows)
        ordered_sources.append(rows)
        ordered_groups.append(group)
    return _ValidatedPopulation(
        d_ap=np.asarray(
            [[row.d_ap for row in rows] for rows in ordered_sources],
            dtype=np.float64,
        ),
        d_component_recall=np.asarray(
            [[row.d_component_recall for row in rows] for rows in ordered_sources],
            dtype=np.float64,
        ),
        q_l0=np.asarray([group.l0_p999 for group in ordered_groups], dtype=np.float64),
        q_l1=np.asarray([group.l1_p999 for group in ordered_groups], dtype=np.float64),
    )


def _linear_lower(values: npt.NDArray[np.float64]) -> float:
    ordered: npt.NDArray[np.float64] = np.sort(values)
    position = 0.025 * (ordered.size - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    lower_value = float(ordered[lower])
    return lower_value + (float(ordered[upper]) - lower_value) * weight


def _source_groups(
    sources: Sequence[SourceMetric],
) -> Dict[Tuple[str, int], List[SourceMetric]]:
    sources_by_group: Dict[Tuple[str, int], List[SourceMetric]] = {}
    source_keys: Set[Tuple[str, int, str]] = set()
    for source in sources:
        key = (source.object_name, source.seed, source.source_id)
        if key in source_keys:
            raise InvalidGateInput("source rows contain a duplicate identity")
        source_keys.add(key)
        _validate_source(source)
        sources_by_group.setdefault((source.object_name, source.seed), []).append(source)
    return sources_by_group


def _validate_source(source: SourceMetric) -> None:
    if len(source.l1_responses) != 2 or len(source.r1_responses) != 2:
        raise InvalidGateInput("each source requires exactly two thin-profile responses")
    numeric = (
        source.d_ap,
        source.d_component_recall,
        source.broad_pauroc_delta,
        *source.l1_responses,
        *source.r1_responses,
    )
    if not bool(np.all(np.isfinite(np.asarray(numeric, dtype=np.float64)))):
        raise InvalidGateInput("source metrics contain non-finite values")
    audits = (source.l0_audit, source.l1_audit, source.r1_audit)
    if source.l0_audit != source.l1_audit or source.l1_audit != source.r1_audit:
        raise InvalidGateInput("paired rung population/support/fallback/mask audits differ")
    if not all(value for audit in audits for value in audit):
        raise InvalidGateInput("source audit identifiers must be non-empty")


def _residual_groups(groups: Sequence[GroupResidual]) -> Dict[Tuple[str, int], GroupResidual]:
    groups_by_key: Dict[Tuple[str, int], GroupResidual] = {}
    for group in groups:
        key = (group.object_name, group.seed)
        if key in groups_by_key:
            raise InvalidGateInput("residual groups contain a duplicate identity")
        groups_by_key[key] = group
    return groups_by_key


def _validate_group(group: GroupResidual, rows: Sequence[SourceMetric]) -> None:
    expected_membership = set(zip(group.source_ids, group.fold_indices))
    actual_membership = {(row.source_id, row.fold_index) for row in rows}
    counts_valid = all(group.fold_indices.count(fold) == 4 for fold in range(4))
    valid_group = len(rows) == len(group.source_ids) == len(group.fold_indices) == 16
    valid_group &= len(set(group.source_ids)) == 16 and counts_valid
    valid_group &= expected_membership == actual_membership
    if not valid_group:
        raise InvalidGateInput("source membership or fold assignment differs from selection")
    residual_values: npt.NDArray[np.float64] = np.asarray(
        (group.l0_p999, group.l1_p999),
        dtype=np.float64,
    )
    if not bool(np.all(np.isfinite(residual_values))):
        raise InvalidGateInput("group residual quantiles contain non-finite values")
    if group.quantile != 0.999 or group.quantile_method != "higher":
        raise InvalidGateInput("group residual quantile contract is not p99.9 higher")
    populations_match = group.l0_population_sha256 == group.l1_population_sha256
    if not group.l0_population_sha256 or not populations_match:
        raise InvalidGateInput("ordered L0/L1 residual token populations differ")
    if not group.l0_residual_sha256 or not group.l1_residual_sha256:
        raise InvalidGateInput("raw residual digests must be non-empty")
