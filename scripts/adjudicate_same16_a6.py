#!/usr/bin/env python3
"""Select the frozen detector branch and decide whether shot scaling may start."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from scripts.aggregate_a6_fusion_public_shadow import (
    ARM_CONTRACTS,
    ARMS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    COMPARISONS,
    build_comparison_payload,
    build_gates,
)
from scripts.aggregate_same16_baselines import FROZEN_SUPPORT_SHA256, METRICS, OBJECTS

BASELINES = ("superad", "rn_fmlk")
COLLAPSE_BOUNDS = {
    "seg_AUROC_0.05": -0.02,
    "seg_F1_raw": -0.03,
    "seg_AUPRO_0.05": -0.02,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args(argv)


def read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        message = f"expected JSON object: {path}"
        raise TypeError(message)
    return payload


def _validate_summary_headers(
    baseline: Dict[str, Any],
    mechanism: Dict[str, Any],
) -> None:
    if baseline.get("support_manifest_sha256") != FROZEN_SUPPORT_SHA256:
        raise ValueError("baseline summary has the wrong support manifest SHA-256")
    if mechanism.get("support_manifest_sha256") != FROZEN_SUPPORT_SHA256:
        raise ValueError("mechanism summary has the wrong support manifest SHA-256")
    if baseline.get("objects") != list(OBJECTS):
        raise ValueError("baseline summary has the wrong object contract")
    if mechanism.get("objects") != list(OBJECTS):
        raise ValueError("mechanism summary has the wrong object contract")
    if baseline.get("status") != "COMPLETE_COMPARABLE_PUBLIC_SHADOW":
        raise ValueError("baseline summary has the wrong completion status")
    if mechanism.get("status") != "COMPLETE_PUBLIC_SHADOW":
        raise ValueError("mechanism summary has the wrong completion status")
    if baseline.get("comparable") is not True:
        raise ValueError("baseline summary is not comparable")


def _validate_declared_contract(
    baseline: Dict[str, Any],
    mechanism: Dict[str, Any],
) -> None:
    if baseline.get("metric_contract") != mechanism.get("metric_contract"):
        raise ValueError("baseline and mechanism metric contracts differ")
    metric_contract = baseline.get("metric_contract")
    if not isinstance(metric_contract, dict) or set(metric_contract) != set(METRICS):
        raise ValueError("summary has the wrong metric contract")
    if mechanism.get("arm_contracts") != ARM_CONTRACTS:
        raise ValueError("mechanism summary has the wrong arm contracts")
    expected_bootstrap = {
        "unit": "object",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
    }
    if mechanism.get("bootstrap") != expected_bootstrap:
        raise ValueError("mechanism summary has the wrong bootstrap contract")


def _validate_metric_table(
    rows_by_variant: object,
    expected_variants: set[str],
    family: str,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    if not isinstance(rows_by_variant, dict) or set(rows_by_variant) != expected_variants:
        message = f"{family} summary has the wrong variant set"
        raise ValueError(message)
    for variant, rows in rows_by_variant.items():
        if not isinstance(rows, dict) or set(rows) != set(OBJECTS):
            message = f"{family}/{variant}: wrong object rows"
            raise ValueError(message)
        for object_name, row in rows.items():
            if not isinstance(row, dict):
                message = f"{family}/{variant}/{object_name}: invalid metric row"
                raise TypeError(message)
            for metric in METRICS:
                value = row.get(metric)
                if not isinstance(value, (float, int)) or not 0.0 <= value <= 1.0:
                    message = f"{family}/{variant}/{object_name}: invalid {metric}"
                    raise ValueError(message)
            if row.get("prediction_map_count", 0.0) <= 0.0:
                message = f"{family}/{variant}/{object_name}: missing coverage"
                raise ValueError(message)
    return rows_by_variant


def _validate_mechanism_evidence(
    mechanism: Dict[str, Any],
    per_arm: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    if mechanism.get("arms") != list(ARMS):
        raise ValueError("mechanism summary has the wrong ordered arm contract")
    expected_comparisons = {
        name: build_comparison_payload(left, right, per_arm)
        for name, (left, right) in COMPARISONS.items()
    }
    if mechanism.get("comparisons") != expected_comparisons:
        raise ValueError("mechanism comparison evidence is stale or inconsistent")
    if mechanism.get("gates") != build_gates(expected_comparisons):
        raise ValueError("mechanism gates are stale or inconsistent")


def validate_shared_contract(
    baseline: Dict[str, Any],
    mechanism: Dict[str, Any],
) -> None:
    _validate_summary_headers(baseline, mechanism)
    _validate_declared_contract(baseline, mechanism)
    per_method = _validate_metric_table(
        baseline.get("per_method"),
        {"basic", *BASELINES},
        "baseline",
    )
    per_arm = _validate_metric_table(
        mechanism.get("per_arm"),
        set(ARMS),
        "mechanism",
    )
    if per_method["basic"] != per_arm["basic"]:
        raise ValueError("baseline and mechanism Basic rows differ")
    _validate_mechanism_evidence(mechanism, per_arm)


def select_detector(mechanism: Dict[str, Any]) -> Dict[str, Any]:
    gates = mechanism.get("gates")
    if not isinstance(gates, dict):
        raise TypeError("mechanism summary is missing gates")
    a6_promoted = (
        gates.get("a6_simplification_safe") is True
        and gates.get("flow_without_dvt_supported") is True
    )
    if not a6_promoted:
        return {
            "selected_detector": "basic",
            "mechanism_verdict": "A6_NOT_PROMOTED_RETAIN_BASIC",
        }
    if gates.get("dominant_l17_repair_supported") is True:
        return {
            "selected_detector": "a6_l17",
            "mechanism_verdict": "A6_L17_PROMOTED",
        }
    return {
        "selected_detector": "a6",
        "mechanism_verdict": "A6_PROMOTED_L17_REJECTED",
    }


def compare_with_baseline(
    candidate_rows: Dict[str, Dict[str, float]],
    baseline_rows: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    per_object = {
        object_name: {
            metric: candidate_rows[object_name][metric] - baseline_rows[object_name][metric]
            for metric in METRICS
        }
        for object_name in OBJECTS
    }
    macro = {
        metric: sum(per_object[object_name][metric] for object_name in OBJECTS) / len(OBJECTS)
        for metric in METRICS
    }
    positive_objects = {
        metric: sum(per_object[object_name][metric] > 0.0 for object_name in OBJECTS)
        for metric in METRICS
    }
    collapsed_objects = {
        metric: [
            object_name
            for object_name in OBJECTS
            if per_object[object_name][metric] < COLLAPSE_BOUNDS[metric]
        ]
        for metric in METRICS
    }
    passed = (
        all(macro[metric] > 0.0 for metric in METRICS)
        and all(positive_objects[metric] >= 5 for metric in METRICS)
        and not any(collapsed_objects.values())
    )
    return {
        "macro_delta": macro,
        "positive_object_count": positive_objects,
        "collapsed_objects": collapsed_objects,
        "passed": passed,
    }


def adjudicate(root: Path) -> Dict[str, Any]:
    baseline = read_json(root / "summary.json")
    mechanism = read_json(root / "a6_fusion_summary.json")
    validate_shared_contract(baseline, mechanism)
    selection = select_detector(mechanism)
    selected_detector = selection["selected_detector"]
    if selected_detector == "basic":
        candidate_rows = baseline["per_method"]["basic"]
    else:
        candidate_rows = mechanism["per_arm"][selected_detector]
    comparisons = {
        baseline_name: compare_with_baseline(
            candidate_rows,
            baseline["per_method"][baseline_name],
        )
        for baseline_name in BASELINES
    }
    shot_curve_authorized = all(comparison["passed"] for comparison in comparisons.values())
    return {
        "status": "COMPLETE_PRE_REGISTERED_DECISION",
        "claim_scope": "MVTec AD2 test_public public-shadow development evidence",
        "support_manifest_sha256": FROZEN_SUPPORT_SHA256,
        **selection,
        "baseline_comparisons": comparisons,
        "shot_curve_authorized": shot_curve_authorized,
        "next_action": (
            "FREEZE_AND_RUN_RANDOM16" if shot_curve_authorized else "KILL_FOR_CLAIM_NO_SHOT_SCALING"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = adjudicate(args.root)
    output = args.root / "post_a6_decision.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    authorized_marker = args.root / "shot_scaling_authorized.txt"
    blocked_marker = args.root / "shot_scaling_blocked.txt"
    if payload["shot_curve_authorized"]:
        if blocked_marker.exists():
            raise RuntimeError("conflicting shot-scaling marker already exists")
        authorized_marker.write_text(
            f"selected_detector={payload['selected_detector']}\n",
            encoding="utf-8",
        )
    else:
        if authorized_marker.exists():
            raise RuntimeError("conflicting shot-scaling marker already exists")
        blocked_marker.write_text(
            f"next_action={payload['next_action']}\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
