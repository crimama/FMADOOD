#!/usr/bin/env python3
"""Validity-first aggregation and frozen adjudication for WTCF Stage -1."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

OBJECTS = ("can", "fruit_jelly", "rice", "fabric", "sheet_metal")
SEEDS = (0, 1, 2)
ARMS = (
    "a6",
    "a7_identity",
    "pca_residual",
    "shrinkage_mahalanobis",
    "crossfit_only",
    "global_isometry",
    "random_witness",
    "witness_only",
    "wtcf",
)
METRICS = ("pAUROC_0.05", "fixed_F1", "AUPRO_0.05")
NONINFERIOR = {"pAUROC_0.05": -0.005, "fixed_F1": -0.010, "AUPRO_0.05": -0.010}
MATERIAL = {"pAUROC_0.05": 0.005, "fixed_F1": 0.010, "AUPRO_0.05": 0.010}
COLLAPSE = {"pAUROC_0.05": -0.020, "fixed_F1": -0.030, "AUPRO_0.05": -0.030}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    return parser.parse_args(list(argv))


def _read_cases(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    cases: dict[tuple[str, int], dict[str, Any]] = {}
    for object_name in OBJECTS:
        for seed in SEEDS:
            path = root / "cases" / f"{object_name}_seed{seed}" / "summary.json"
            if not path.is_file():
                message = f"missing frozen case: {path}"
                raise FileNotFoundError(message)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("object") != object_name or payload.get("support_seed") != seed:
                message = f"case identity mismatch: {path}"
                raise RuntimeError(message)
            if set(payload.get("arms", {})) != set(ARMS):
                message = f"arm coverage mismatch: {path}"
                raise RuntimeError(message)
            cases[(object_name, seed)] = payload
    return cases


def _wtcf_valid(case: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    access = case["access_ledger"]
    if access["coverage"] != 1.0 or access["public_private_test_ground_truth_attempts"] != 0:
        reasons.append("ACCESS_CUSTODY")
    if case.get("synthetic_generation_after_all_arm_states_frozen") is not True:
        reasons.append("SYNTHETIC_ORDER")
    reasons.extend(
        f"MAP_VALIDITY:{arm_name}"
        for arm_name in ARMS
        if case["arms"][arm_name]["finite_nonconstant_complete"] is not True
    )
    proxy_arms = (
        "a6",
        "crossfit_only",
        "global_isometry",
        "random_witness",
        "witness_only",
        "wtcf",
    )
    reasons.extend(
        f"SOFT_HARD_PROXY:{arm_name}"
        for arm_name in proxy_arms
        if case["arms"][arm_name]["proxy"]["passed"] is not True
    )
    arm = case["arms"]["wtcf"]
    training = arm["training"]
    if training["finite"] is not True:
        reasons.append("TRAINING_FINITE")
    if training["final_geometry"]["passed"] is not True:
        reasons.append("FUNCTION_GEOMETRY")
    if training["final_directional"]["passed"] is not True:
        reasons.append("JVP_DIRECTIONAL")
    witness = training["witness"]
    if witness["accepted_count"] <= 0:
        reasons.append("ZERO_WITNESS_NO_MECHANISM")
    elif (
        min(witness["source_pair_diversity"]) < 3
        or min(witness["fold_pair_diversity"]) < 3
        or min(witness["angular_stability"]) < 0.85
    ):
        reasons.append("WITNESS_DIVERSITY_OR_STABILITY")
    return not reasons, reasons


def _macro(cases: dict[tuple[str, int], dict[str, Any]], arm: str) -> dict[str, float]:
    keys = ("held_normal_q995_a6_units", *METRICS)
    return {
        key: float(np.mean([case["arms"][arm]["metrics"][key] for case in cases.values()]))
        for key in keys
    }


def _delta(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {key: left[key] - right[key] for key in left}


def _joint_dominates(wtcf: dict[str, float], control: dict[str, float]) -> bool:
    delta = _delta(wtcf, control)
    return bool(
        delta["held_normal_q995_a6_units"] <= -0.005
        and all(delta[key] >= NONINFERIOR[key] for key in METRICS)
        and sum(delta[key] >= MATERIAL[key] for key in METRICS) >= 2,
    )


def aggregate(root: Path) -> dict[str, Any]:
    cases = _read_cases(root)
    validity: dict[str, Any] = {}
    all_valid = True
    for key, case in cases.items():
        passed, reasons = _wtcf_valid(case)
        validity[f"{key[0]}_seed{key[1]}"] = {"passed": passed, "reasons": reasons}
        all_valid = all_valid and passed
    macro = {arm: _macro(cases, arm) for arm in ARMS}
    wtcf_delta = _delta(macro["wtcf"], macro["a6"])
    null_dominance = {
        arm: _joint_dominates(macro["wtcf"], macro[arm])
        for arm in ARMS
        if arm not in {"a6", "wtcf"}
    }
    object_no_collapse: dict[str, bool] = {}
    seed_agreement: dict[str, bool] = {}
    for object_name in OBJECTS:
        object_deltas: list[dict[str, float]] = []
        seed_signs: list[bool] = []
        for seed in SEEDS:
            case = cases[(object_name, seed)]
            wtcf_metrics = case["arms"]["wtcf"]["metrics"]
            a6_metrics = case["arms"]["a6"]["metrics"]
            delta = {
                key: float(wtcf_metrics[key] - a6_metrics[key])
                for key in ("held_normal_q995_a6_units", *METRICS)
            }
            object_deltas.append(delta)
            seed_signs.append(
                delta["held_normal_q995_a6_units"] < 0.0
                and sum(delta[key] >= 0.0 for key in METRICS) >= 2,
            )
        means = {
            key: float(np.mean([row[key] for row in object_deltas]))
            for key in object_deltas[0]
        }
        object_no_collapse[object_name] = all(means[key] >= COLLAPSE[key] for key in METRICS)
        seed_agreement[object_name] = sum(seed_signs) >= 2
    gates = {
        "all_valid": all_valid,
        "a6_tail_reduction": wtcf_delta["held_normal_q995_a6_units"] <= -0.01,
        "a6_detector_noninferiority": all(wtcf_delta[key] >= NONINFERIOR[key] for key in METRICS),
        "a6_two_material_detector_gains": (
            sum(wtcf_delta[key] >= MATERIAL[key] for key in METRICS) >= 2
        ),
        "all_nulls_jointly_dominated": all(null_dominance.values()),
        "object_no_collapse": all(object_no_collapse.values()),
        "seed_direction_agreement": all(seed_agreement.values()),
    }
    keep = all(gates.values())
    return {
        "schema": "wtcf-stage-minus1-summary-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": len(cases),
        "validity": validity,
        "macro": macro,
        "wtcf_minus_a6": wtcf_delta,
        "null_dominance": null_dominance,
        "object_no_collapse": object_no_collapse,
        "seed_direction_agreement": seed_agreement,
        "gates": gates,
        "verdict": "KEEP_FOR_PUBLIC_STAGE_0" if keep else "KILL_FOR_CLAIM_NO_PUBLIC_ACCESS",
        "public_access_authorized": bool(keep),
    }


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve(strict=True)
    payload = aggregate(root)
    (root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = ["arm\theld_q995\tpAUROC_0.05\tfixed_F1\tAUPRO_0.05"]
    for arm in ARMS:
        row = payload["macro"][arm]
        lines.append(
            "\t".join(
                [arm, *(f"{row[key]:.9f}" for key in ("held_normal_q995_a6_units", *METRICS))],
            ),
        )
    (root / "summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "verdict.txt").write_text(str(payload["verdict"]) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
