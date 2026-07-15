"""Typed JSON boundaries for strict DARC Gate 2 aggregation."""

from __future__ import annotations

# pyright: reportMissingImports=false
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Final, FrozenSet, List, Mapping, NoReturn, Tuple

from flow_tte.darc_gate2_metrics import (
    Gate2Decision,
    GroupResidual,
    InvalidGateInput,
    SourceAudit,
    SourceMetric,
)
from flow_tte.darc_gate2_provenance import JsonValue, ModelProvenance, SeedProvenance, decode_json

SOURCE_KEYS: Final[FrozenSet[str]] = frozenset(
    {
        "object",
        "seed",
        "fold",
        "source",
        "d_ap",
        "d_component_recall",
        "l1_responses",
        "r1_responses",
        "broad_pauroc_delta",
        "l0_audit",
        "l1_audit",
        "r1_audit",
        "evaluation",
        "provenance_sha256",
    },
)
GROUP_KEYS: Final[FrozenSet[str]] = frozenset(
    {
        "object",
        "seed",
        "source_ids",
        "fold_indices",
        "l0_p999",
        "l1_p999",
        "l0_population_sha256",
        "l1_population_sha256",
        "l0_residual_sha256",
        "l1_residual_sha256",
        "quantile",
        "quantile_method",
        "provenance_sha256",
    },
)
PROVENANCE_KEYS: Final[FrozenSet[str]] = frozenset(
    {
        "method_sha256",
        "code_config_sha256",
        "dataset_inventory_sha256",
        "split_inventory_sha256",
        "selection_sha256",
        "model",
    },
)
MODEL_KEYS: Final[FrozenSet[str]] = frozenset(
    {"model_id", "revision", "config_sha256", "weights_sha256"},
)


# Python 3.8 dataclasses do not support slots=True.
@dataclass(frozen=True)
class AggregateInput:
    sources: Tuple[SourceMetric, ...]
    groups: Tuple[GroupResidual, ...]
    method_sha256: str
    code_config_sha256: str
    model: ModelProvenance


# Python 3.8 dataclasses do not support slots=True.
@dataclass(frozen=True)
class AggregateResult:
    aggregate: AggregateInput
    decision: Gate2Decision

    def to_manifest(self) -> Dict[str, JsonValue]:
        return {
            "schema": "darc-gate2-aggregate-v1",
            "method_sha256": self.aggregate.method_sha256,
            "code_config_sha256": self.aggregate.code_config_sha256,
            "model": self.aggregate.model.to_manifest(),
            **self.decision.to_manifest(),
        }


@dataclass(frozen=True)
class SeedIdentity:
    object_name: str
    seed: int


def fail(reason: str) -> NoReturn:
    raise InvalidGateInput(reason)


def read_json(path: Path) -> Dict[str, JsonValue]:
    try:
        return json_object(decode_json(path.read_text(encoding="utf-8")), path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        fail(f"could not read JSON artifact: {path}")


def read_jsonl(path: Path) -> Tuple[Dict[str, JsonValue], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return tuple(json_object(decode_json(line), path) for line in lines)
    except (OSError, UnicodeError, json.JSONDecodeError):
        fail(f"could not read JSONL artifact: {path}")


def json_object(payload: JsonValue, path: Path) -> Dict[str, JsonValue]:
    if not isinstance(payload, dict):
        fail(f"expected JSON object: {path}")
    return payload


def exact_object(
    payload: JsonValue,
    path: Path,
    keys: FrozenSet[str],
) -> Dict[str, JsonValue]:
    values = json_object(payload, path)
    if set(values) != set(keys):
        fail(f"JSON fields differ from registered schema: {path}")
    return values


def string(values: Mapping[str, JsonValue], key: str, path: Path) -> str:
    value = values[key]
    if not isinstance(value, str) or not value:
        fail(f"{key} must be a non-empty string: {path}")
    return value


def integer(values: Mapping[str, JsonValue], key: str, path: Path) -> int:
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{key} must be an integer: {path}")
    return value


def number(values: Mapping[str, JsonValue], key: str, path: Path) -> float:
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{key} must be numeric: {path}")
    result = float(value)
    if not math.isfinite(result):
        fail(f"{key} must be finite: {path}")
    return result


def boolean(values: Mapping[str, JsonValue], key: str, path: Path) -> bool:
    value = values[key]
    if not isinstance(value, bool):
        fail(f"{key} must be boolean: {path}")
    return value


def sha256(values: Mapping[str, JsonValue], key: str, path: Path) -> str:
    value = string(values, key, path)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        fail(f"{key} must be a lowercase SHA256 digest: {path}")
    return value


def parse_provenance(payload: JsonValue, path: Path) -> SeedProvenance:
    values = exact_object(payload, path, PROVENANCE_KEYS)
    model_values = exact_object(values["model"], path, MODEL_KEYS)
    model = ModelProvenance(
        model_id=string(model_values, "model_id", path),
        revision=string(model_values, "revision", path),
        config_sha256=sha256(model_values, "config_sha256", path),
        weights_sha256=sha256(model_values, "weights_sha256", path),
    )
    return SeedProvenance(
        method_sha256=sha256(values, "method_sha256", path),
        code_config_sha256=sha256(values, "code_config_sha256", path),
        dataset_inventory_sha256=sha256(values, "dataset_inventory_sha256", path),
        split_inventory_sha256=sha256(values, "split_inventory_sha256", path),
        selection_sha256=sha256(values, "selection_sha256", path),
        model=model,
    )


def parse_source(
    payload: JsonValue,
    identity: SeedIdentity,
    provenance_sha256: str,
) -> SourceMetric:
    path = Path("source_rows.jsonl")
    values = exact_object(payload, path, SOURCE_KEYS)
    if (
        string(values, "object", path) != identity.object_name
        or integer(values, "seed", path) != identity.seed
        or sha256(values, "provenance_sha256", path) != provenance_sha256
    ):
        fail("source row identity or provenance differs from completion")
    _evaluation(values["evaluation"], path)
    fold = integer(values, "fold", path)
    if fold not in (0, 1, 2, 3):
        fail("source fold must be in 0..3")
    return SourceMetric(
        object_name=identity.object_name,
        seed=identity.seed,
        fold_index=fold,
        source_id=string(values, "source", path),
        d_ap=number(values, "d_ap", path),
        d_component_recall=number(values, "d_component_recall", path),
        l1_responses=_number_pair(values["l1_responses"], path),
        r1_responses=_number_pair(values["r1_responses"], path),
        broad_pauroc_delta=number(values, "broad_pauroc_delta", path),
        l0_audit=_audit(values["l0_audit"], path),
        l1_audit=_audit(values["l1_audit"], path),
        r1_audit=_audit(values["r1_audit"], path),
    )


def parse_group(
    payload: JsonValue,
    identity: SeedIdentity,
    provenance_sha256: str,
) -> GroupResidual:
    path = Path("group_residual.json")
    values = exact_object(payload, path, GROUP_KEYS)
    method = string(values, "quantile_method", path)
    if (
        string(values, "object", path) != identity.object_name
        or integer(values, "seed", path) != identity.seed
        or sha256(values, "provenance_sha256", path) != provenance_sha256
        or method != "higher"
    ):
        fail("group identity, provenance, or quantile method differs from registration")
    return GroupResidual(
        object_name=identity.object_name,
        seed=identity.seed,
        source_ids=tuple(strings(values["source_ids"], path)),
        fold_indices=tuple(integers(values["fold_indices"], path)),
        l0_p999=number(values, "l0_p999", path),
        l1_p999=number(values, "l1_p999", path),
        l0_population_sha256=sha256(values, "l0_population_sha256", path),
        l1_population_sha256=sha256(values, "l1_population_sha256", path),
        l0_residual_sha256=sha256(values, "l0_residual_sha256", path),
        l1_residual_sha256=sha256(values, "l1_residual_sha256", path),
        quantile=number(values, "quantile", path),
        quantile_method="higher",
    )


def _audit(payload: JsonValue, path: Path) -> SourceAudit:
    keys = frozenset({"population_sha256", "support_sha256", "fallback_sha256", "mask_sha256"})
    values = exact_object(payload, path, keys)
    return SourceAudit(
        population_sha256=sha256(values, "population_sha256", path),
        support_sha256=sha256(values, "support_sha256", path),
        fallback_sha256=sha256(values, "fallback_sha256", path),
        mask_sha256=sha256(values, "mask_sha256", path),
    )


def _evaluation(payload: JsonValue, path: Path) -> None:
    keys = frozenset(
        {
            "thresholds",
            "absolute_ap",
            "absolute_component_recall",
            "l1_profiles",
            "r1_profiles",
            "broad_pauroc_005",
        },
    )
    values = exact_object(payload, path, keys)
    pair_keys = frozenset({"l0", "l1"})
    for field in ("thresholds", "absolute_ap", "absolute_component_recall"):
        pair = exact_object(values[field], path, pair_keys)
        number(pair, "l0", path)
        number(pair, "l1", path)
    profile_keys = frozenset({"clean_mean", "cue_mean", "response"})
    for field in ("l1_profiles", "r1_profiles"):
        profiles = json_list(values[field], path)
        if len(profiles) != 2:
            fail(f"{field} must contain exactly two thin-profile rows: {path}")
        for profile in profiles:
            row = exact_object(profile, path, profile_keys)
            for metric in profile_keys:
                number(row, metric, path)
    broad_keys = frozenset({"l0", "r1", "delta_r1_l0"})
    broad = exact_object(values["broad_pauroc_005"], path, broad_keys)
    for metric in broad_keys:
        number(broad, metric, path)


def json_list(payload: JsonValue, path: Path) -> List[JsonValue]:
    if not isinstance(payload, list):
        fail(f"expected JSON array: {path}")
    return payload


def strings(payload: JsonValue, path: Path) -> List[str]:
    output: List[str] = []
    for value in json_list(payload, path):
        if not isinstance(value, str) or not value:
            fail(f"expected non-empty string array: {path}")
        output.append(value)
    return output


def integers(payload: JsonValue, path: Path) -> List[int]:
    output: List[int] = []
    for value in json_list(payload, path):
        if isinstance(value, bool) or not isinstance(value, int):
            fail(f"expected integer array: {path}")
        output.append(value)
    return output


def _number_pair(payload: JsonValue, path: Path) -> Tuple[float, float]:
    values = json_list(payload, path)
    if len(values) != 2:
        fail(f"expected numeric pair: {path}")
    wrapper: Dict[str, JsonValue] = {"first": values[0], "second": values[1]}
    return number(wrapper, "first", path), number(wrapper, "second", path)
