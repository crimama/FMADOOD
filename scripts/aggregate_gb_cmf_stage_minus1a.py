#!/usr/bin/env python3
# ruff: noqa: C901
"""Aggregate the four frozen GB-CMF Stage -1A support-only cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROTOCOL_ID = "gb-cmf-stage-minus1a-support-only-20260719"
SCHEMA = "gb-cmf-stage-minus1a-case-v1"
SUMMARY_SCHEMA = "gb-cmf-stage-minus1a-summary-v1"
SUPPORT_SHA256 = "9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5"
ARMS = ("default", "nonfactorial", "gb_cmf", "affine_null", "nonlinear_head_null")
HARD_NULLS = ("affine_null", "nonlinear_head_null")
METRICS = ("pAUROC_0.05", "fixed_F1", "raw_pAUROC_0.05", "raw_fixed_F1")
BOOTSTRAPS = 2000


@dataclass(frozen=True)
class FrozenCase:
    gpu: int
    object_name: str
    seed: int
    fold: int
    family: str
    axis: str
    sources: tuple[str, str, str, str]


EXPECTED_CASES = (
    FrozenCase(
        0,
        "rice",
        0,
        0,
        "cutpaste",
        "exposure",
        ("128_regular.png", "205_regular.png", "064_regular.png", "060_regular.png"),
    ),
    FrozenCase(
        1,
        "can",
        1,
        1,
        "chromatic_blotch",
        "gamma",
        ("322_regular.png", "107_regular.png", "262_regular.png", "217_regular.png"),
    ),
    FrozenCase(
        2,
        "rice",
        1,
        2,
        "gaussian_blur",
        "contrast",
        ("273_regular.png", "091_regular.png", "093_regular.png", "036_regular.png"),
    ),
    FrozenCase(
        3,
        "can",
        0,
        3,
        "cutpaste",
        "white_balance",
        ("105_regular.png", "211_regular.png", "020_regular.png", "289_regular.png"),
    ),
)
EXPECTED_BY_ID = {(case.object_name, case.seed): case for case in EXPECTED_CASES}

KILL_ORDER = (
    "SHARD_SCHEMA_INVALID",
    "PROTOCOL_MISMATCH",
    "SUPPORT_MANIFEST_MISMATCH",
    "CASE_PARITY_FAILURE",
    "CASE_ASSIGNMENT_MISMATCH",
    "FORBIDDEN_ACCESS_ATTEMPT",
    "FORBIDDEN_STAGE_TRANSITION",
    "FIT_PARITY_FAILURE",
    "ARM_INCOMPLETE",
    "SOURCE_COVERAGE_FAILURE",
    "METRIC_INVALID",
    "JOINT_CONSTRAINT_INFEASIBLE",
    "CONSTRAINT_BASELINE_NOT_BEATEN",
    "GEOMETRY_GUARD_FAILURE",
    "PROXY_FIDELITY_FAILURE",
    "MAP_ACCESS_COVERAGE_FAILURE",
    "COMPONENT_COLLAPSE",
    "HARD_NULL_SUPERIORITY_FAILURE",
    "RAW_NONINFERIORITY_FAILURE",
    "SEED_DIRECTION_DISAGREEMENT",
)
INVALID_RUN_CODES = frozenset(
    {
        "SHARD_SCHEMA_INVALID",
        "PROTOCOL_MISMATCH",
        "SUPPORT_MANIFEST_MISMATCH",
        "CASE_PARITY_FAILURE",
        "CASE_ASSIGNMENT_MISMATCH",
        "FORBIDDEN_ACCESS_ATTEMPT",
        "FORBIDDEN_STAGE_TRANSITION",
        "FIT_PARITY_FAILURE",
        "ARM_INCOMPLETE",
        "SOURCE_COVERAGE_FAILURE",
        "METRIC_INVALID",
    },
)


class ShardError(RuntimeError):
    """A named, non-retryable Stage -1A shard failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    return parser.parse_args(list(argv))


def _finite(value: object, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ShardError("METRIC_INVALID", f"{context}: not numeric") from error
    if not math.isfinite(result):
        raise ShardError("METRIC_INVALID", f"{context}: not finite")
    return result


def _bounded_metric(value: object, context: str) -> float:
    result = _finite(value, context)
    if not 0.0 <= result <= 1.0:
        raise ShardError("METRIC_INVALID", f"{context}: outside [0,1]")
    return result


def _basename(value: object) -> str:
    return str(value).replace("\\", "/").rsplit("/", maxsplit=1)[-1]


def _require_access_custody(payload: Mapping[str, Any], context: str) -> None:
    access = payload.get("access_attempts")
    expected_keys = {"public", "test", "private", "ground_truth"}
    if not isinstance(access, Mapping) or set(access) != expected_keys:
        raise ShardError("SHARD_SCHEMA_INVALID", f"{context}: access-attempt schema")
    if (
        payload.get("query_or_test_access") is not False
        or payload.get("public_metrics_computed") is not False
        or any(access[key] != 0 for key in expected_keys)
    ):
        raise ShardError("FORBIDDEN_ACCESS_ATTEMPT", f"{context}: non-support access")
    if payload.get("stage_minus1b_authorized") is not False:
        raise ShardError(
            "FORBIDDEN_STAGE_TRANSITION",
            f"{context}: Stage -1B must remain unauthorized",
        )


def _normalize_source_row(
    value: Mapping[str, Any],
    *,
    case: FrozenCase,
    arm: str,
) -> dict[str, Any]:
    source = _basename(value.get("source"))
    context = f"{case.object_name}/seed{case.seed}/{arm}/{source}"
    family = str(value.get("family"))
    if source not in case.sources or family != case.family:
        raise ShardError("SOURCE_COVERAGE_FAILURE", f"{context}: source/family mismatch")
    return {
        "source": source,
        "family": family,
        **{metric: _bounded_metric(value.get(metric), f"{context}/{metric}") for metric in METRICS},
        "g_tail": _finite(value.get("g_tail"), f"{context}/g_tail"),
        "g_drop": _finite(value.get("g_drop"), f"{context}/g_drop"),
        "geometry_pass": value.get("geometry_pass") is True,
        "proxy_pass": value.get("proxy_pass") is True,
        "map_coverage": _finite(value.get("map_coverage"), f"{context}/map_coverage"),
        "access_coverage": _finite(
            value.get("access_coverage"),
            f"{context}/access_coverage",
        ),
        "component_std_ratio_min": _finite(
            value.get("component_std_ratio_min"),
            f"{context}/component_std_ratio_min",
        ),
        "total_score_std_ratio": _finite(
            value.get("total_score_std_ratio"),
            f"{context}/total_score_std_ratio",
        ),
    }


def _normalize_arm(
    value: Mapping[str, Any],
    *,
    case: FrozenCase,
    arm: str,
) -> dict[str, Any]:
    context = f"{case.object_name}/seed{case.seed}/{arm}"
    if value.get("status") != "complete":
        raise ShardError("ARM_INCOMPLETE", f"{context}: arm is not complete")
    source_values = value.get("source_metrics")
    if not isinstance(source_values, list) or len(source_values) != 4:
        raise ShardError("SOURCE_COVERAGE_FAILURE", f"{context}: expected four sources")
    if not all(isinstance(item, Mapping) for item in source_values):
        raise ShardError("SHARD_SCHEMA_INVALID", f"{context}: invalid source row")
    rows = [_normalize_source_row(item, case=case, arm=arm) for item in source_values]
    sources = [row["source"] for row in rows]
    if len(set(sources)) != 4 or set(sources) != set(case.sources):
        raise ShardError("SOURCE_COVERAGE_FAILURE", f"{context}: source parity failed")
    rows.sort(key=lambda row: case.sources.index(row["source"]))

    macro_value = value.get("macro_metrics")
    if not isinstance(macro_value, Mapping) or set(macro_value) != set(METRICS):
        raise ShardError("SHARD_SCHEMA_INVALID", f"{context}: macro-metric schema")
    macro = {
        metric: _bounded_metric(macro_value[metric], f"{context}/macro/{metric}")
        for metric in METRICS
    }
    for metric in METRICS:
        recomputed = sum(float(row[metric]) for row in rows) / 4.0
        if not math.isclose(macro[metric], recomputed, rel_tol=0.0, abs_tol=1e-9):
            raise ShardError("METRIC_INVALID", f"{context}: macro {metric} mismatch")
    return {"status": "complete", "source_metrics": rows, "macro_metrics": macro}


def _normalize_case(payload: Mapping[str, Any]) -> dict[str, Any]:
    object_name = str(payload.get("object"))
    try:
        seed = int(payload.get("seed"))
    except (TypeError, ValueError) as error:
        raise ShardError("SHARD_SCHEMA_INVALID", "missing object/seed identity") from error
    context = f"{object_name}/seed{seed}"
    if payload.get("schema") != SCHEMA:
        raise ShardError("SHARD_SCHEMA_INVALID", f"{context}: schema mismatch")
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ShardError("PROTOCOL_MISMATCH", f"{context}: protocol mismatch")
    if payload.get("support_manifest_sha256") != SUPPORT_SHA256:
        raise ShardError("SUPPORT_MANIFEST_MISMATCH", f"{context}: support hash mismatch")
    case = EXPECTED_BY_ID.get((object_name, seed))
    if case is None:
        raise ShardError("CASE_ASSIGNMENT_MISMATCH", f"{context}: unknown frozen case")
    _require_access_custody(payload, context)

    held_sources = payload.get("held_sources")
    if not isinstance(held_sources, list):
        raise ShardError("CASE_ASSIGNMENT_MISMATCH", f"{context}: held source schema")
    assignment_ok = (
        int(payload.get("gpu", -1)) == case.gpu
        and int(payload.get("held_fold", -1)) == case.fold
        and str(payload.get("held_family")) == case.family
        and str(payload.get("held_axis")) == case.axis
        and [_basename(item) for item in held_sources] == list(case.sources)
    )
    if not assignment_ok:
        raise ShardError("CASE_ASSIGNMENT_MISMATCH", f"{context}: frozen assignment mismatch")
    if int(payload.get("fit_count", -1)) != 5:
        raise ShardError("FIT_PARITY_FAILURE", f"{context}: fit count must equal five")

    arms = payload.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(ARMS):
        raise ShardError("FIT_PARITY_FAILURE", f"{context}: five-arm parity failed")
    if not all(isinstance(arms[arm], Mapping) for arm in ARMS):
        raise ShardError("SHARD_SCHEMA_INVALID", f"{context}: invalid arm payload")
    return {
        "object": object_name,
        "seed": seed,
        "gpu": case.gpu,
        "held_fold": case.fold,
        "held_family": case.family,
        "held_axis": case.axis,
        "held_sources": list(case.sources),
        "arms": {arm: _normalize_arm(arms[arm], case=case, arm=arm) for arm in ARMS},
    }


def _source_lookup(case: Mapping[str, Any], arm: str) -> dict[str, Mapping[str, Any]]:
    return {row["source"]: row for row in case["arms"][arm]["source_metrics"]}


def _case_gates(case: Mapping[str, Any]) -> dict[str, bool]:
    candidate = _source_lookup(case, "gb_cmf")
    default = _source_lookup(case, "default")
    nonfactorial = _source_lookup(case, "nonfactorial")
    all_rows = [
        row
        for arm in ARMS
        for row in case["arms"][arm]["source_metrics"]
    ]
    joint = all(row["g_tail"] <= 0.0 and row["g_drop"] <= 0.0 for row in candidate.values())
    constraints = all(
        candidate[source][constraint] < default[source][constraint]
        and candidate[source][constraint] < nonfactorial[source][constraint]
        for source in candidate
        for constraint in ("g_tail", "g_drop")
    )
    geometry = all(row["geometry_pass"] for row in all_rows)
    proxy = all(row["proxy_pass"] for row in all_rows)
    coverage = all(
        math.isclose(row["map_coverage"], 1.0, abs_tol=1e-12)
        and math.isclose(row["access_coverage"], 1.0, abs_tol=1e-12)
        for row in all_rows
    )
    component = all(
        row["component_std_ratio_min"] >= 0.50
        and 0.80 <= row["total_score_std_ratio"] <= 1.25
        for row in all_rows
    )
    candidate_macro = case["arms"]["gb_cmf"]["macro_metrics"]
    null_macros = [case["arms"][arm]["macro_metrics"] for arm in HARD_NULLS]
    hard_null = (
        candidate_macro["pAUROC_0.05"]
        >= max(row["pAUROC_0.05"] for row in null_macros) + 0.005
        and candidate_macro["fixed_F1"]
        >= max(row["fixed_F1"] for row in null_macros) + 0.010
    )
    raw = all(
        candidate[source]["raw_pAUROC_0.05"]
        >= default[source]["raw_pAUROC_0.05"] - 0.005
        and candidate[source]["raw_fixed_F1"] >= default[source]["raw_fixed_F1"] - 0.010
        for source in candidate
    )
    return {
        "joint_constraint_feasibility": joint,
        "constraint_superiority_vs_default_and_nonfactorial": constraints,
        "geometry_100pct": geometry,
        "proxy_fidelity_100pct": proxy,
        "map_and_access_coverage_100pct": coverage,
        "component_noncollapse": component,
        "hard_null_superiority": hard_null,
        "raw_noninferiority": raw,
    }


def _sign(value: float) -> int:
    if value > 1e-12:
        return 1
    if value < -1e-12:
        return -1
    return 0


def _seed_directions(cases: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], bool]:
    directions: dict[str, Any] = {}
    consistent = True
    for object_name in ("can", "rice"):
        object_cases = sorted(
            (case for case in cases if case["object"] == object_name),
            key=lambda case: int(case["seed"]),
        )
        vectors: dict[int, dict[str, int]] = {}
        for case in object_cases:
            candidate = case["arms"]["gb_cmf"]["macro_metrics"]
            null_envelope = {
                metric: max(case["arms"][arm]["macro_metrics"][metric] for arm in HARD_NULLS)
                for metric in ("pAUROC_0.05", "fixed_F1")
            }
            vector: dict[str, int] = {}
            for metric in ("pAUROC_0.05", "fixed_F1"):
                for comparison in ("default", "nonfactorial"):
                    baseline = case["arms"][comparison]["macro_metrics"][metric]
                    vector[f"{metric}_vs_{comparison}"] = _sign(candidate[metric] - baseline)
                vector[f"{metric}_vs_hard_null"] = _sign(
                    candidate[metric] - null_envelope[metric],
                )
            vectors[int(case["seed"])] = vector
        object_consistent = set(vectors) == {0, 1} and vectors[0] == vectors[1]
        directions[object_name] = {
            "by_seed": {str(seed): value for seed, value in sorted(vectors.items())},
            "consistent": object_consistent,
        }
        consistent = consistent and object_consistent
    return directions, consistent


def _linear_quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _bootstrap(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seed = int.from_bytes(
        hashlib.sha256(f"{PROTOCOL_ID}\x1fpaired-source-bootstrap".encode()).digest()[:8],
        "big",
    )
    rng = random.Random(seed)  # noqa: S311 - fixed scientific bootstrap stream
    comparisons = ("default", "nonfactorial", "hard_null")
    metrics = ("pAUROC_0.05", "fixed_F1")
    values = {
        f"{metric}_delta_vs_{comparison}": []
        for comparison in comparisons
        for metric in metrics
    }
    for _ in range(BOOTSTRAPS):
        accumulator = {key: [] for key in values}
        for case in cases:
            sampled = [rng.randrange(4) for _ in range(4)]
            candidate_rows = case["arms"]["gb_cmf"]["source_metrics"]
            arm_means = {
                arm: {
                    metric: sum(
                        case["arms"][arm]["source_metrics"][index][metric]
                        for index in sampled
                    )
                    / 4.0
                    for metric in metrics
                }
                for arm in ARMS
            }
            for metric in metrics:
                for comparison in ("default", "nonfactorial"):
                    accumulator[f"{metric}_delta_vs_{comparison}"].append(
                        sum(candidate_rows[index][metric] for index in sampled) / 4.0
                        - arm_means[comparison][metric],
                    )
                hard_null = max(arm_means[arm][metric] for arm in HARD_NULLS)
                accumulator[f"{metric}_delta_vs_hard_null"].append(
                    sum(candidate_rows[index][metric] for index in sampled) / 4.0 - hard_null,
                )
        for key, items in values.items():
            items.append(sum(accumulator[key]) / len(accumulator[key]))
    intervals = {
        key: {
            "q025": _linear_quantile(items, 0.025),
            "q500": _linear_quantile(items, 0.500),
            "q975": _linear_quantile(items, 0.975),
        }
        for key, items in values.items()
    }
    return {
        "replicates": BOOTSTRAPS,
        "seed": seed,
        "unit": "four held source images independently within each frozen case",
        "descriptive_only": True,
        "patch_bootstrap_used": False,
        "intervals": intervals,
    }


def _kill_codes(case_gates: Sequence[Mapping[str, bool]], seed_consistent: bool) -> list[str]:
    mapping = (
        ("joint_constraint_feasibility", "JOINT_CONSTRAINT_INFEASIBLE"),
        (
            "constraint_superiority_vs_default_and_nonfactorial",
            "CONSTRAINT_BASELINE_NOT_BEATEN",
        ),
        ("geometry_100pct", "GEOMETRY_GUARD_FAILURE"),
        ("proxy_fidelity_100pct", "PROXY_FIDELITY_FAILURE"),
        ("map_and_access_coverage_100pct", "MAP_ACCESS_COVERAGE_FAILURE"),
        ("component_noncollapse", "COMPONENT_COLLAPSE"),
        ("hard_null_superiority", "HARD_NULL_SUPERIORITY_FAILURE"),
        ("raw_noninferiority", "RAW_NONINFERIORITY_FAILURE"),
    )
    codes = [code for gate, code in mapping if not all(item[gate] for item in case_gates)]
    if not seed_consistent:
        codes.append("SEED_DIRECTION_DISAGREEMENT")
    return codes


def _attempt_normalize(path: Path) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            return None, {
                "path": str(path),
                "code": "SHARD_SCHEMA_INVALID",
                "message": "payload is not an object",
            }
        return _normalize_case(payload), None
    except (json.JSONDecodeError, OSError) as error:
        return None, {
            "path": str(path),
            "code": "SHARD_SCHEMA_INVALID",
            "message": str(error),
        }
    except ShardError as error:
        return None, {"path": str(path), "code": error.code, "message": str(error)}


def aggregate(run_root: Path) -> dict[str, Any]:
    paths = sorted((run_root / "chunks").glob("*/summary.json"))
    cases: dict[tuple[str, int], dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    input_codes: set[str] = set()
    for path in paths:
        case, failure = _attempt_normalize(path)
        if failure is not None:
            input_codes.add(failure["code"])
            failures.append(failure)
            continue
        if case is None:
            code = "SHARD_SCHEMA_INVALID"
            input_codes.add(code)
            failures.append(
                {"path": str(path), "code": code, "message": "empty normalized case"},
            )
            continue
        identity = (case["object"], case["seed"])
        if identity in cases:
            code = "CASE_PARITY_FAILURE"
            input_codes.add(code)
            failures.append(
                {"path": str(path), "code": code, "message": f"duplicate frozen case {identity}"},
            )
            continue
        cases[identity] = case

    if set(cases) != set(EXPECTED_BY_ID) or len(paths) != 4:
        input_codes.add("CASE_PARITY_FAILURE")
    ordered_cases = [cases[key] for key in EXPECTED_BY_ID if key in cases]
    case_gate_rows = [_case_gates(case) for case in ordered_cases]
    directions, direction_consistent = _seed_directions(ordered_cases)
    scientific_codes = (
        _kill_codes(case_gate_rows, direction_consistent)
        if len(ordered_cases) == 4 and not input_codes
        else []
    )
    all_codes = input_codes.union(scientific_codes)
    kill_codes = [code for code in KILL_ORDER if code in all_codes]
    gate_pass = not kill_codes

    per_case = []
    for case, gates in zip(ordered_cases, case_gate_rows):
        per_case.append(
            {
                **{
                    key: case[key]
                    for key in ("object", "seed", "held_fold", "held_family", "held_axis")
                },
                "macro_metrics": {
                    arm: case["arms"][arm]["macro_metrics"] for arm in ARMS
                },
                "gates": gates,
            },
        )
    return {
        "schema": SUMMARY_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "support_manifest_sha256": SUPPORT_SHA256,
        "case_count": len(ordered_cases),
        "fit_count": len(ordered_cases) * len(ARMS),
        "expected_fit_count": 20,
        "support_only": True,
        "query_or_test_access": False,
        "public_metrics_computed": False,
        "public_access_authorized": False,
        "stage_minus1b_authorized": False,
        "selection_credit": False,
        "case_results": per_case,
        "seed_directions": directions,
        "bootstrap": _bootstrap(ordered_cases) if len(ordered_cases) == 4 else None,
        "input_failures": failures,
        "kill_codes": kill_codes,
        "gate_pass": gate_pass,
        "claim_verdict": "KILL_FOR_CLAIM",
        "diagnostic_verdict": (
            "STAGE_MINUS1A_SUPPORT_ONLY_PASS_POST_RESULT_AUDIT_REQUIRED"
            if gate_pass
            else "KILL_FOR_CLAIM_NO_PUBLIC_ACCESS"
        ),
        "next_gate": "POST_RESULT_AUDIT_REQUIRED" if gate_pass else "NO_PUBLIC_ACCESS",
    }


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    run_root = Path(args.run_root).resolve(strict=True)
    summary = aggregate(run_root)
    (run_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (run_root / "verdict.txt").write_text(
        "\n".join(
            (
                f"claim={summary['claim_verdict']}",
                f"diagnostic={summary['diagnostic_verdict']}",
                "public_access_authorized=false",
                "stage_minus1b_authorized=false",
                f"kill_codes={','.join(summary['kill_codes'])}",
            ),
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 2 if INVALID_RUN_CODES.intersection(summary["kill_codes"]) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
