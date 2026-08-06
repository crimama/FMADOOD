from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts.adjudicate_same16_a6 import adjudicate, main
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

if TYPE_CHECKING:
    from pathlib import Path


def _rows(value: float) -> dict[str, dict[str, float]]:
    return {
        object_name: {
            **dict.fromkeys(METRICS, value),
            "prediction_map_count": 2.0,
        }
        for object_name in OBJECTS
    }


def _write_summaries(
    root: Path,
    *,
    values: dict[str, float],
) -> None:
    metric_contract = {metric: f"frozen {metric}" for metric in METRICS}
    baseline = {
        "status": "COMPLETE_COMPARABLE_PUBLIC_SHADOW",
        "objects": list(OBJECTS),
        "support_manifest_sha256": FROZEN_SUPPORT_SHA256,
        "metric_contract": metric_contract,
        "comparable": True,
        "per_method": {
            "basic": _rows(values["basic"]),
            "superad": _rows(values["superad"]),
            "rn_fmlk": _rows(values["rn_fmlk"]),
        },
    }
    per_arm = {arm: _rows(values[arm]) for arm in ARMS}
    comparisons = {
        name: build_comparison_payload(left, right, per_arm)
        for name, (left, right) in COMPARISONS.items()
    }
    mechanism = {
        "status": "COMPLETE_PUBLIC_SHADOW",
        "objects": list(OBJECTS),
        "support_manifest_sha256": FROZEN_SUPPORT_SHA256,
        "metric_contract": metric_contract,
        "arms": list(ARMS),
        "arm_contracts": ARM_CONTRACTS,
        "bootstrap": {
            "unit": "object",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "per_arm": per_arm,
        "comparisons": comparisons,
        "gates": build_gates(comparisons),
    }
    (root / "summary.json").write_text(json.dumps(baseline), encoding="utf-8")
    (root / "a6_fusion_summary.json").write_text(
        json.dumps(mechanism),
        encoding="utf-8",
    )


def test_adjudicator_selects_l17_and_authorizes_random16(tmp_path: Path) -> None:
    _write_summaries(
        tmp_path,
        values={
            "basic": 0.60,
            "superad": 0.61,
            "rn_fmlk": 0.62,
            "a6": 0.64,
            "a7": 0.60,
            "a6_l17": 0.66,
            "l17": 0.61,
        },
    )

    assert main(["--root", str(tmp_path)]) == 0
    result = json.loads(
        (tmp_path / "post_a6_decision.json").read_text(encoding="utf-8"),
    )

    assert result["selected_detector"] == "a6_l17"
    assert result["shot_curve_authorized"] is True
    assert result["next_action"] == "FREEZE_AND_RUN_RANDOM16"
    assert (tmp_path / "shot_scaling_authorized.txt").is_file()
    assert not (tmp_path / "shot_scaling_blocked.txt").exists()


def test_adjudicator_requires_flow_gate_for_a6_promotion(tmp_path: Path) -> None:
    _write_summaries(
        tmp_path,
        values={
            "basic": 0.63,
            "superad": 0.61,
            "rn_fmlk": 0.62,
            "a6": 0.66,
            "a7": 0.66,
            "a6_l17": 0.68,
            "l17": 0.67,
        },
    )

    result = adjudicate(tmp_path)

    assert result["selected_detector"] == "basic"
    assert result["mechanism_verdict"] == "A6_NOT_PROMOTED_RETAIN_BASIC"


def test_adjudicator_blocks_shots_on_object_collapse(tmp_path: Path) -> None:
    _write_summaries(
        tmp_path,
        values={
            "basic": 0.60,
            "superad": 0.61,
            "rn_fmlk": 0.62,
            "a6": 0.66,
            "a7": 0.60,
            "a6_l17": 0.68,
            "l17": 0.61,
        },
    )
    baseline_path = tmp_path / "summary.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline["per_method"]["basic"]["can"]["seg_AUROC_0.05"] = 0.50
    baseline["per_method"]["superad"]["can"]["seg_AUROC_0.05"] = 0.70
    baseline["per_method"]["rn_fmlk"]["can"]["seg_AUROC_0.05"] = 0.70
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    mechanism_path = tmp_path / "a6_fusion_summary.json"
    mechanism = json.loads(mechanism_path.read_text(encoding="utf-8"))
    mechanism["per_arm"]["basic"]["can"]["seg_AUROC_0.05"] = 0.50
    mechanism["per_arm"]["a7"]["can"]["seg_AUROC_0.05"] = 0.50
    mechanism["per_arm"]["a6"]["can"]["seg_AUROC_0.05"] = 0.51
    mechanism["per_arm"]["a6_l17"]["can"]["seg_AUROC_0.05"] = 0.52
    mechanism["comparisons"] = {
        name: build_comparison_payload(left, right, mechanism["per_arm"])
        for name, (left, right) in COMPARISONS.items()
    }
    mechanism["gates"] = build_gates(mechanism["comparisons"])
    mechanism_path.write_text(json.dumps(mechanism), encoding="utf-8")

    assert main(["--root", str(tmp_path)]) == 0
    result = json.loads(
        (tmp_path / "post_a6_decision.json").read_text(encoding="utf-8"),
    )

    assert result["shot_curve_authorized"] is False
    assert result["next_action"] == "KILL_FOR_CLAIM_NO_SHOT_SCALING"
    assert (tmp_path / "shot_scaling_blocked.txt").is_file()
    assert not (tmp_path / "shot_scaling_authorized.txt").exists()


def test_adjudicator_rejects_stale_mechanism_gate(tmp_path: Path) -> None:
    _write_summaries(
        tmp_path,
        values={
            "basic": 0.60,
            "superad": 0.61,
            "rn_fmlk": 0.62,
            "a6": 0.64,
            "a7": 0.60,
            "a6_l17": 0.66,
            "l17": 0.61,
        },
    )
    mechanism_path = tmp_path / "a6_fusion_summary.json"
    mechanism = json.loads(mechanism_path.read_text(encoding="utf-8"))
    mechanism["gates"]["flow_without_dvt_supported"] = False
    mechanism_path.write_text(json.dumps(mechanism), encoding="utf-8")

    with pytest.raises(ValueError, match="stale or inconsistent"):
        adjudicate(tmp_path)


def test_adjudicator_rejects_missing_coverage_evidence(tmp_path: Path) -> None:
    _write_summaries(
        tmp_path,
        values={
            "basic": 0.60,
            "superad": 0.61,
            "rn_fmlk": 0.62,
            "a6": 0.64,
            "a7": 0.60,
            "a6_l17": 0.66,
            "l17": 0.61,
        },
    )
    baseline_path = tmp_path / "summary.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    del baseline["per_method"]["superad"]["can"]["prediction_map_count"]
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    with pytest.raises(ValueError, match="missing coverage"):
        adjudicate(tmp_path)
