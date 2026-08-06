#!/usr/bin/env python3
"""Evaluate the preregistered DSCR paired acquisition interaction audit."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import tifffile as tiff
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from flow_tte_gap_decomposition import load_gt, oracle_f1, parse_condition  # noqa: E402
from flow_tte_phase3_scorer_suite import (  # noqa: E402
    NativeMapRecord,
    evaluate_native_records,
)

ARMS = ("G0", "D0")
CONDITIONS = ("regular", "overexposed", "shift_1", "shift_2")
SHIFTED_CONDITIONS = ("overexposed", "shift_1", "shift_2")
POPULATIONS = ("good", "bad")
METRICS = ("seg_AUROC_0.05", "seg_F1", "pixel_AP")


@dataclass(frozen=True)
class LoadedMap:
    split: str
    stem: str
    condition: str
    score: np.ndarray
    gt: np.ndarray
    map_path: Path
    source_dtype: str


def _source_path(data_root: Path, object_name: str, population: str, stem: str) -> Path:
    matches = sorted((data_root / object_name / "test_public" / population).glob(f"{stem}.*"))
    if len(matches) != 1:
        message = f"expected one source for {population}/{stem}, found {len(matches)}"
        raise RuntimeError(message)
    return matches[0]


def _load_map(
    data_root: Path,
    object_name: str,
    arm: str,
    population: str,
    map_path: Path,
) -> LoadedMap:
    try:
        condition = parse_condition(map_path.stem)
    except ValueError as error:
        message = f"unexpected {arm}/{population} map stem: {map_path.stem}"
        raise RuntimeError(message) from error
    if condition not in CONDITIONS:
        message = f"unexpected {arm}/{population} condition {condition!r}: {map_path.stem}"
        raise RuntimeError(message)
    raw_score = np.asarray(tiff.imread(map_path))
    if raw_score.ndim != 2:
        message = f"invalid non-2D map: {map_path}"
        raise RuntimeError(message)
    if not np.issubdtype(raw_score.dtype, np.floating):
        message = f"invalid non-floating map: {map_path} ({raw_score.dtype})"
        raise RuntimeError(message)
    if not bool(np.all(np.isfinite(raw_score))):
        message = f"non-finite map: {map_path}"
        raise RuntimeError(message)
    if float(np.max(raw_score)) <= float(np.min(raw_score)):
        message = f"constant map: {map_path}"
        raise RuntimeError(message)
    source = _source_path(data_root, object_name, population, map_path.stem)
    score = np.asarray(raw_score, dtype=np.float32)
    gt = load_gt(data_root, object_name, population, source.stem, score.shape)
    return LoadedMap(
        split=population,
        stem=source.stem,
        condition=condition,
        score=score,
        gt=np.asarray(gt, dtype=np.bool_),
        map_path=map_path,
        source_dtype=str(raw_score.dtype),
    )


def _load_arm_maps(
    data_root: Path,
    cell_root: Path,
    object_name: str,
    arm: str,
) -> Mapping[str, tuple[LoadedMap, ...]]:
    grouped: dict[str, list[LoadedMap]] = {condition: [] for condition in CONDITIONS}
    for population in POPULATIONS:
        directory = cell_root / "arms" / arm / "anomaly_maps" / object_name / "test" / population
        map_paths = sorted(directory.glob("*.tiff"))
        for map_path in map_paths:
            loaded = _load_map(data_root, object_name, arm, population, map_path)
            grouped[loaded.condition].append(loaded)

    for condition, records in grouped.items():
        counts = {
            population: sum(row.split == population for row in records)
            for population in POPULATIONS
        }
        if counts != {"good": 1, "bad": 1}:
            message = (
                f"{arm}/{condition} requires exactly one good and one bad map; observed={counts}"
            )
            raise RuntimeError(message)
    return {condition: tuple(records) for condition, records in grouped.items()}


def _logical_ids(grouped: Mapping[str, Sequence[LoadedMap]]) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted((record.split, record.stem) for records in grouped.values() for record in records),
    )


def _native_records(records: Sequence[LoadedMap]) -> tuple[NativeMapRecord, ...]:
    return tuple(
        NativeMapRecord(
            split=record.split,
            stem=record.stem,
            score=np.asarray(record.score, dtype=np.float16),
            gt=record.gt,
        )
        for record in records
    )


def _float16_metrics(records: Sequence[LoadedMap]) -> Mapping[str, float]:
    metrics = evaluate_native_records(_native_records(records))
    return {
        "seg_AUROC_0.05": float(metrics["pooled_pauroc_0.05_float16"]),
        "seg_F1": float(metrics["pooled_oracle_f1_float16"]),
        "pixel_AP": float(metrics["pooled_pixel_ap_float16"]),
    }


def _float32_metrics(records: Sequence[LoadedMap]) -> Mapping[str, float]:
    labels = np.concatenate([np.asarray(row.gt, dtype=np.uint8).ravel() for row in records])
    scores = np.concatenate([np.asarray(row.score, dtype=np.float32).ravel() for row in records])
    if not bool(np.any(labels)) or not bool(np.any(labels == 0)):
        raise ValueError("condition metrics require both anomalous and good pixels")
    return {
        "seg_AUROC_0.05": float(roc_auc_score(labels, scores, max_fpr=0.05)),
        "seg_F1": float(oracle_f1(labels, scores, cast_float16=False)["f1"]),
        "pixel_AP": float(average_precision_score(labels, scores)),
    }


def _condition_arm_payload(records: Sequence[LoadedMap]) -> Mapping[str, Any]:
    float32_available = all(record.source_dtype == "float32" for record in records)
    metrics: dict[str, Mapping[str, float]] = {"float16": _float16_metrics(records)}
    if float32_available:
        metrics["float32"] = _float32_metrics(records)
    return {
        "metrics": metrics,
        "validity": {
            "finite": True,
            "nonconstant": True,
            "map_gt_shape_match": True,
            "float32_tiff_available": float32_available,
            "source_dtypes": sorted({record.source_dtype for record in records}),
        },
        "coverage": {
            "expected_query_count": 2,
            "observed_query_count": len(records),
            "good_count": sum(record.split == "good" for record in records),
            "bad_count": sum(record.split == "bad" for record in records),
            "complete": len(records) == 2,
        },
        "query_ids": [
            {"split": record.split, "stem": record.stem}
            for record in sorted(records, key=lambda row: row.split)
        ],
    }


def _delta(
    d0: Mapping[str, Mapping[str, float]],
    g0: Mapping[str, Mapping[str, float]],
) -> Mapping[str, Mapping[str, float]]:
    common_precisions = tuple(
        precision for precision in ("float16", "float32") if precision in d0 and precision in g0
    )
    return {
        precision: {
            metric: float(d0[precision][metric]) - float(g0[precision][metric])
            for metric in METRICS
        }
        for precision in common_precisions
    }


def _interaction(
    condition_payloads: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    common_precisions = tuple(
        precision
        for precision in ("float16", "float32")
        if all(
            precision in condition_payloads[condition]["delta_vs_g0"] for condition in CONDITIONS
        )
    )
    values: dict[str, Any] = {}
    for precision in common_precisions:
        values[precision] = {}
        for metric in METRICS:
            regular_delta = float(condition_payloads["regular"]["delta_vs_g0"][precision][metric])
            shifted_delta = float(
                np.mean(
                    [
                        condition_payloads[condition]["delta_vs_g0"][precision][metric]
                        for condition in SHIFTED_CONDITIONS
                    ],
                ),
            )
            values[precision][metric] = {
                "regular_delta": regular_delta,
                "mean_shifted_delta": shifted_delta,
                "value": shifted_delta - regular_delta,
            }
    return {
        "definition": "mean_shifted_delta_minus_regular_delta",
        "shifted_conditions": list(SHIFTED_CONDITIONS),
        **values,
    }


def evaluate_condition_audit(
    data_root: Path,
    cell_root: Path,
    object_name: str,
) -> Mapping[str, Any]:
    arm_maps = {arm: _load_arm_maps(data_root, cell_root, object_name, arm) for arm in ARMS}
    arm_ids = {arm: _logical_ids(grouped) for arm, grouped in arm_maps.items()}
    if arm_ids["D0"] != arm_ids["G0"]:
        raise RuntimeError("arms do not contain identical logical query IDs")

    conditions: dict[str, Any] = {}
    for condition in CONDITIONS:
        arms = {arm: _condition_arm_payload(arm_maps[arm][condition]) for arm in ARMS}
        conditions[condition] = {
            "arms": arms,
            "delta_vs_g0": _delta(
                arms["D0"]["metrics"],
                arms["G0"]["metrics"],
            ),
        }

    expected_per_arm = len(CONDITIONS) * len(POPULATIONS)
    float32_complete = all(
        conditions[condition]["arms"][arm]["validity"]["float32_tiff_available"]
        for condition in CONDITIONS
        for arm in ARMS
    )
    return {
        "schema": "dscr-density-condition-audit-v1",
        "object": object_name,
        "conditions": conditions,
        "interaction": _interaction(conditions),
        "arm_id_parity": {
            "match": True,
            "logical_ids": [{"split": split, "stem": stem} for split, stem in arm_ids["G0"]],
        },
        "validity": {
            "finite": True,
            "nonconstant": True,
            "map_gt_shape_match": True,
            "float32_tiff_complete": float32_complete,
        },
        "query_coverage": {
            "expected_logical_query_count_per_arm": expected_per_arm,
            "observed_logical_query_count_per_arm": len(arm_ids["G0"]),
            "expected_total_arm_map_count": expected_per_arm * len(ARMS),
            "observed_total_arm_map_count": sum(len(value) for value in arm_ids.values()),
            "complete": len(arm_ids["G0"]) == expected_per_arm,
        },
        "precisions_evaluated": ["float16", *(["float32"] if float32_complete else [])],
        "comparable": False,
        "scope": "paired-acquisition-interaction-forensic-diagnostic",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cell-root", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = evaluate_condition_audit(args.data_root, args.cell_root, args.object)
    output = args.output or args.cell_root / "condition_audit_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
