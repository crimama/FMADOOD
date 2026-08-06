#!/usr/bin/env python3
"""Evaluate the frozen Stage 2 acquisition-routed support-bank controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from evaluate_flow_tte_default_shift_audit import (  # noqa: E402
    OBJECTS,
    condition_rows,
    continuous_metrics,
    finite_nonconstant,
    macro_mean,
    object_arm_root,
    parse_objects,
    verify_parity,
)

from flow_tte_gap_decomposition import load_records  # noqa: E402

ARMS = ("identity", "union", "patchwise_min", "routed")


def object_harm(delta_auroc: float, delta_f1: float) -> bool:
    return delta_auroc < -0.02 or delta_f1 < -0.03


def continuation_gate(
    identity: dict[str, float],
    routed: dict[str, float],
    *,
    has_object_harm: bool,
    valid_object_count: int,
    expected_object_count: int,
) -> dict[str, Any]:
    delta_auroc = routed["seg_AUROC_0.05"] - identity["seg_AUROC_0.05"]
    delta_f1 = routed["seg_F1"] - identity["seg_F1"]
    baseline_inflation = identity["shifted_good_p95_inflation"]
    inflation_reduction = (
        baseline_inflation - routed["shifted_good_p95_inflation"]
    ) / max(abs(baseline_inflation), 1e-12)
    performance = delta_auroc >= 0.005 or delta_f1 >= 0.005
    tail = inflation_reduction >= 0.15
    separation = (
        routed["bad_good_p95_separation"] >= identity["bad_good_p95_separation"]
    )
    valid = valid_object_count == expected_object_count
    continuation = performance and tail and separation and not has_object_harm and valid
    return {
        "delta_seg_AUROC_0.05": delta_auroc,
        "delta_seg_F1": delta_f1,
        "shifted_good_inflation_relative_reduction": inflation_reduction,
        "performance_gate": performance,
        "tail_reduction_gate": tail,
        "separation_gate": separation,
        "no_object_harm_gate": not has_object_harm,
        "validity_gate": valid,
        "continuation": continuation,
        "decision": "PROMISING_DIAGNOSTIC" if continuation else "NO_CONTINUE",
    }


def arm_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        **continuous_metrics(records),
        "coverage": finite_nonconstant(records),
        "conditions": condition_rows(records),
    }


def macro_metrics(per_object: dict[str, Any], objects: Sequence[str], arm: str) -> dict[str, Any]:
    rows = [per_object[name]["arms"][arm] for name in objects]
    inflation = [
        float(condition["good_p95_inflation_vs_regular"])
        for row in rows
        for condition_name, condition in row["conditions"].items()
        if condition_name != "regular"
    ]
    separation = [
        float(condition["bad_good_p95_separation"])
        for row in rows
        for condition in row["conditions"].values()
    ]
    return {
        "seg_AUROC_0.05": macro_mean(rows, "seg_AUROC_0.05"),
        "seg_F1": macro_mean(rows, "seg_F1"),
        "shifted_good_p95_inflation": float(np.mean(inflation)),
        "bad_good_p95_separation": float(np.mean(separation)),
        "valid_object_count": sum(bool(row["coverage"]["pass"]) for row in rows),
    }


def evaluate_object(root: Path, data_root: Path, object_name: str) -> dict[str, Any]:
    records = {
        arm: load_records(object_arm_root(root, object_name, arm), data_root, object_name)
        for arm in ARMS
    }
    verify_parity(records)
    arms = {arm: arm_metrics(rows) for arm, rows in records.items()}
    identity = arms["identity"]
    routed = arms["routed"]
    delta_auroc = routed["seg_AUROC_0.05"] - identity["seg_AUROC_0.05"]
    delta_f1 = routed["seg_F1"] - identity["seg_F1"]
    routing_path = root / "objects" / object_name / "routing.json"
    routing = json.loads(routing_path.read_text(encoding="utf-8"))
    return {
        "arms": arms,
        "routed_vs_identity": {
            "seg_AUROC_0.05": delta_auroc,
            "seg_F1": delta_f1,
            "object_harm": object_harm(delta_auroc, delta_f1),
        },
        "routing": {
            "support_routing": routing["support_routing"],
            "confidence_threshold": routing["confidence_threshold"],
            "selected_bank_counts": {
                bank: sum(row["selected_bank"] == bank for row in routing["routes"])
                for bank in routing["bank_names"]
            },
            "identity_fallback_count": sum(
                bool(row["identity_fallback"]) for row in routing["routes"]
            ),
            "query_count": len(routing["routes"]),
            "identity_scorer_max_abs_parity_error": routing[
                "identity_scorer_max_abs_parity_error"
            ],
        },
    }


def summarize(per_object: dict[str, Any], objects: Sequence[str]) -> dict[str, Any]:
    macro = {arm: macro_metrics(per_object, objects, arm) for arm in ARMS}
    harm_objects = [
        name for name in objects if per_object[name]["routed_vs_identity"]["object_harm"]
    ]
    verdict = continuation_gate(
        macro["identity"],
        macro["routed"],
        has_object_harm=bool(harm_objects),
        valid_object_count=int(macro["routed"]["valid_object_count"]),
        expected_object_count=len(objects),
    )
    verdict["object_harm"] = harm_objects
    return {
        "target_dataset": "MVTec AD2 single-image",
        "evaluation_role": "test_public_shadow_support_bank_routing",
        "claim_boundary": "diagnostic only; no query transform, TTA, or memory update",
        "arms": list(ARMS),
        "macro": macro,
        "per_object": per_object,
        "verdict": verdict,
    }


def evaluate(root: Path, data_root: Path, objects: Sequence[str]) -> dict[str, Any]:
    per_object = {
        object_name: evaluate_object(root, data_root, object_name)
        for object_name in objects
    }
    return summarize(per_object, objects)


def aggregate_object_summaries(root: Path, objects: Sequence[str]) -> dict[str, Any]:
    per_object: dict[str, Any] = {}
    for object_name in objects:
        path = root / "objects" / object_name / "stage2_metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if list(payload["per_object"]) != [object_name]:
            message = f"invalid object summary identity: {path}"
            raise ValueError(message)
        per_object[object_name] = payload["per_object"][object_name]
    return summarize(per_object, objects)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--objects", default=",".join(OBJECTS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--from-object-summaries", action="store_true")
    args = parser.parse_args(argv)
    objects = parse_objects(args.objects)
    if args.from_object_summaries:
        payload = aggregate_object_summaries(Path(args.root), objects)
    else:
        payload = evaluate(Path(args.root), Path(args.data_root), objects)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"macro": payload["macro"], "verdict": payload["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
