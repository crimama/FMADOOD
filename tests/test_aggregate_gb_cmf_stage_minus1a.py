from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable

import pytest

from scripts.aggregate_gb_cmf_stage_minus1a import (
    ARMS,
    BOOTSTRAPS,
    EXPECTED_CASES,
    PROTOCOL_ID,
    SCHEMA,
    SUPPORT_SHA256,
    aggregate,
    main,
)

if TYPE_CHECKING:
    from pathlib import Path


def _arm_payload(case_index: int, arm: str) -> dict[str, Any]:
    case = EXPECTED_CASES[case_index]
    values = {
        "default": (0.700, 0.600, 0.20, 0.20),
        "nonfactorial": (0.705, 0.605, 0.10, 0.10),
        "gb_cmf": (0.730, 0.640, -0.10, -0.10),
        "affine_null": (0.710, 0.610, 0.15, 0.15),
        "nonlinear_head_null": (0.715, 0.615, 0.15, 0.15),
    }
    pauc, f1, g_tail, g_drop = values[arm]
    rows = [
        {
            "source": f"{case.object_name}/train/good/{source}",
            "family": case.family,
            "pAUROC_0.05": pauc,
            "fixed_F1": f1,
            "raw_pAUROC_0.05": 0.72,
            "raw_fixed_F1": 0.62,
            "g_tail": g_tail,
            "g_drop": g_drop,
            "geometry_pass": True,
            "proxy_pass": True,
            "map_coverage": 1.0,
            "access_coverage": 1.0,
            "component_std_ratio_min": 0.8,
            "total_score_std_ratio": 1.0,
        }
        for source in case.sources
    ]
    return {
        "status": "complete",
        "source_metrics": rows,
        "macro_metrics": {
            "pAUROC_0.05": pauc,
            "fixed_F1": f1,
            "raw_pAUROC_0.05": 0.72,
            "raw_fixed_F1": 0.62,
        },
    }


def _case_payload(case_index: int) -> dict[str, Any]:
    case = EXPECTED_CASES[case_index]
    return {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "support_manifest_sha256": SUPPORT_SHA256,
        "object": case.object_name,
        "seed": case.seed,
        "gpu": case.gpu,
        "held_fold": case.fold,
        "held_family": case.family,
        "held_axis": case.axis,
        "held_sources": list(case.sources),
        "query_or_test_access": False,
        "public_metrics_computed": False,
        "access_attempts": {"public": 0, "test": 0, "private": 0, "ground_truth": 0},
        "stage_minus1b_authorized": False,
        "fit_count": 5,
        "arms": {arm: _arm_payload(case_index, arm) for arm in ARMS},
    }


def _write_cases(root: Path, payloads: list[dict[str, Any]] | None = None) -> None:
    values = payloads or [_case_payload(index) for index in range(4)]
    for payload in values:
        chunk = root / "chunks" / f"{payload['object']}_seed{payload['seed']}"
        chunk.mkdir(parents=True)
        (chunk / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def _refresh_macro(arm: dict[str, Any]) -> None:
    arm["macro_metrics"] = {
        metric: sum(float(row[metric]) for row in arm["source_metrics"]) / 4.0
        for metric in (
            "pAUROC_0.05",
            "fixed_F1",
            "raw_pAUROC_0.05",
            "raw_fixed_F1",
        )
    }


def test_strict_pass_remains_support_only_and_requires_a_new_audit(tmp_path: Path) -> None:
    _write_cases(tmp_path)

    result = aggregate(tmp_path)

    assert result["gate_pass"] is True
    assert result["kill_codes"] == []
    assert result["case_count"] == 4
    assert result["fit_count"] == 20
    assert result["bootstrap"]["replicates"] == BOOTSTRAPS
    assert result["bootstrap"]["patch_bootstrap_used"] is False
    assert result["claim_verdict"] == "KILL_FOR_CLAIM"
    assert result["public_access_authorized"] is False
    assert result["stage_minus1b_authorized"] is False
    assert result["next_gate"] == "POST_RESULT_AUDIT_REQUIRED"


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (
            lambda payload: payload.update(query_or_test_access=True),
            "FORBIDDEN_ACCESS_ATTEMPT",
        ),
        (
            lambda payload: payload.update(stage_minus1b_authorized=True),
            "FORBIDDEN_STAGE_TRANSITION",
        ),
        (
            lambda payload: payload.update(held_axis="gamma"),
            "CASE_ASSIGNMENT_MISMATCH",
        ),
        (
            lambda payload: payload["arms"].pop("affine_null"),
            "FIT_PARITY_FAILURE",
        ),
    ],
)
def test_custody_and_exact_parity_fail_with_named_kill_codes(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    expected_code: str,
) -> None:
    payloads = [_case_payload(index) for index in range(4)]
    mutate(payloads[0])
    _write_cases(tmp_path, payloads)

    result = aggregate(tmp_path)

    assert expected_code in result["kill_codes"]
    assert result["gate_pass"] is False
    assert result["public_access_authorized"] is False
    assert result["stage_minus1b_authorized"] is False
    assert result["next_gate"] == "NO_PUBLIC_ACCESS"


def test_scientific_failures_are_not_averaged_away(tmp_path: Path) -> None:
    payloads = [_case_payload(index) for index in range(4)]
    candidate = payloads[0]["arms"]["gb_cmf"]
    candidate["source_metrics"][0]["g_tail"] = 0.001
    candidate["source_metrics"][1]["geometry_pass"] = False
    candidate["source_metrics"][2]["proxy_pass"] = False
    candidate["source_metrics"][3]["map_coverage"] = 0.99
    candidate["source_metrics"][0]["component_std_ratio_min"] = 0.49
    candidate["source_metrics"][1]["raw_pAUROC_0.05"] = 0.70
    _refresh_macro(candidate)
    affine = payloads[0]["arms"]["affine_null"]
    for row in affine["source_metrics"]:
        row["pAUROC_0.05"] = 0.73
    _refresh_macro(affine)
    _write_cases(tmp_path, payloads)

    result = aggregate(tmp_path)

    assert {
        "JOINT_CONSTRAINT_INFEASIBLE",
        "GEOMETRY_GUARD_FAILURE",
        "PROXY_FIDELITY_FAILURE",
        "MAP_ACCESS_COVERAGE_FAILURE",
        "COMPONENT_COLLAPSE",
        "HARD_NULL_SUPERIORITY_FAILURE",
        "RAW_NONINFERIORITY_FAILURE",
    }.issubset(result["kill_codes"])
    assert result["gate_pass"] is False


def test_candidate_must_beat_both_constraint_baselines_per_source(tmp_path: Path) -> None:
    payloads = [_case_payload(index) for index in range(4)]
    candidate = payloads[2]["arms"]["gb_cmf"]["source_metrics"][0]
    candidate["g_drop"] = 0.10
    _write_cases(tmp_path, payloads)

    result = aggregate(tmp_path)

    assert "CONSTRAINT_BASELINE_NOT_BEATEN" in result["kill_codes"]


def test_both_seed_metric_directions_must_agree(tmp_path: Path) -> None:
    payloads = [_case_payload(index) for index in range(4)]
    can_seed1 = next(
        payload for payload in payloads if payload["object"] == "can" and payload["seed"] == 1
    )
    default = can_seed1["arms"]["default"]
    for row in default["source_metrics"]:
        row["pAUROC_0.05"] = 0.735
        row["fixed_F1"] = 0.645
    _refresh_macro(default)
    _write_cases(tmp_path, payloads)

    result = aggregate(tmp_path)

    assert "SEED_DIRECTION_DISAGREEMENT" in result["kill_codes"]
    assert result["seed_directions"]["can"]["consistent"] is False


def test_missing_case_is_a_named_parity_kill(tmp_path: Path) -> None:
    _write_cases(tmp_path, [_case_payload(index) for index in range(3)])

    result = aggregate(tmp_path)

    assert result["case_count"] == 3
    assert "CASE_PARITY_FAILURE" in result["kill_codes"]
    assert result["bootstrap"] is None


def test_cli_exit_distinguishes_scientific_kill_from_invalid_run(tmp_path: Path) -> None:
    payloads = [_case_payload(index) for index in range(4)]
    payloads[0]["arms"]["gb_cmf"]["source_metrics"][0]["g_tail"] = 0.001
    _write_cases(tmp_path, payloads)

    assert main(("--run-root", str(tmp_path))) == 0

    invalid_root = tmp_path / "invalid"
    invalid_root.mkdir()
    invalid_payloads = [_case_payload(index) for index in range(4)]
    invalid_payloads[0]["query_or_test_access"] = True
    _write_cases(invalid_root, invalid_payloads)

    assert main(("--run-root", str(invalid_root))) == 2
