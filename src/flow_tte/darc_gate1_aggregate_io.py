from __future__ import annotations

# pyright: reportMissingImports=false
import json
from pathlib import Path
from typing import Dict, List, Mapping, NoReturn, Tuple

from flow_tte.darc_gate1 import ConditionMetric
from flow_tte.darc_gate1_provenance import JsonValue
from flow_tte.darc_gate1_stability import ThresholdRotation, ThresholdStability


class DarcAggregateError(RuntimeError):
    pass


def fail(reason: str) -> NoReturn:
    raise DarcAggregateError(reason)


def read_json(path: Path) -> Dict[str, JsonValue]:
    try:
        return json_dict(json.loads(path.read_text(encoding="utf-8")), path)
    except (OSError, json.JSONDecodeError):
        fail(f"could not read JSON artifact: {path}")


def read_json_lines(path: Path) -> Tuple[Dict[str, JsonValue], ...]:
    try:
        return tuple(
            json_dict(json.loads(line), path)
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    except (OSError, json.JSONDecodeError):
        fail(f"could not read JSONL artifact: {path}")


def json_dict(payload: JsonValue | None, path: Path) -> Dict[str, JsonValue]:
    if not isinstance(payload, dict):
        fail(f"expected JSON object: {path}")
    return payload


def json_list(payload: JsonValue | None, path: Path) -> List[JsonValue]:
    if not isinstance(payload, list):
        fail(f"expected JSON array: {path}")
    return payload


def json_string(payload: Mapping[str, JsonValue], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        fail(f"{key} must be a string")
    return value


def json_integer(payload: Mapping[str, JsonValue], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{key} must be an integer")
    return value


def json_number(payload: Mapping[str, JsonValue], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{key} must be numeric")
    return float(value)


def json_boolean(payload: Mapping[str, JsonValue], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        fail(f"{key} must be boolean")
    return value


def parse_condition(payload: JsonValue | None) -> ConditionMetric:
    path = Path("source_metrics.json")
    values = json_dict(payload, path)
    stability_values = json_dict(values.get("stability"), path)
    rotations = tuple(
        ThresholdRotation(
            heldout_index=json_integer(json_dict(row, path), "heldout_index"),
            threshold=json_number(json_dict(row, path), "threshold"),
            heldout_clean_fpr=json_number(json_dict(row, path), "heldout_clean_fpr"),
        )
        for row in json_list(stability_values.get("rotations"), path)
    )
    if tuple(row.heldout_index for row in rotations) != (0, 1, 2, 3):
        fail("each condition requires four ordered stability rotations")
    stability = ThresholdStability(
        rotations=rotations,
        median_fpr=json_number(stability_values, "median_fpr"),
        maximum_fpr=json_number(stability_values, "maximum_fpr"),
        threshold_median=json_number(stability_values, "threshold_median"),
        threshold_iqr=json_number(stability_values, "threshold_iqr"),
        threshold_iqr_ratio=json_number(stability_values, "threshold_iqr_ratio"),
        stable=json_boolean(stability_values, "stable"),
        criterion=json_string(stability_values, "criterion"),
    )
    return ConditionMetric(
        ap=json_number(values, "ap"),
        p_auroc_005=json_number(values, "p_auroc_005"),
        component_recall=json_number(values, "component_recall"),
        fixed_threshold=json_number(values, "fixed_threshold"),
        clean_fpr=json_number(values, "clean_fpr"),
        stability=stability,
    )
