from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from scripts.aggregate_same16_baselines import OBJECTS
from scripts.aggregate_same16_dvt_density_factorial import (
    ARM_CONTRACTS,
    ARMS,
    COMMON_CONTRACT,
    aggregate,
    arm_run_dir,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_fixture(
    factorial_root: Path,
    baseline_root: Path,
    support: dict[str, list[str]],
    *,
    arm: str,
    object_name: str,
    value: float,
) -> None:
    run_dir = arm_run_dir(factorial_root, baseline_root, arm, object_name)
    run_dir.mkdir(parents=True)
    row = {
        "seg_AUROC": value,
        "seg_AUROC_0.05": value,
        "seg_F1_raw": value,
        "seg_F1": value,
        "seg_AUPRO_0.05": value,
        "coverage": {
            "prediction_map_count": 2,
            "ground_truth_map_count": 2,
            "shape_match_count": 2,
            "nonconstant_prediction_map_count": 2,
            "anomalous_region_count": 1,
            "non_finite_prediction_count": 0,
        },
    }
    manifest = {
        **COMMON_CONTRACT,
        **ARM_CONTRACTS[arm],
        "support_manifest_sha256": hashlib.sha256(
            json.dumps(support).encode(),
        ).hexdigest(),
        "object_diagnostics": [
            {
                "selected_support_paths": support[object_name],
            },
        ],
    }
    (run_dir / "metrics.json").write_text(
        json.dumps({object_name: row}),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def _support() -> dict[str, list[str]]:
    return {
        object_name: [f"/data/{object_name}/train/good/{index:03d}.png" for index in range(16)]
        for object_name in OBJECTS
    }


def test_factorial_credits_dvt_density_and_interaction(tmp_path: Path) -> None:
    support = _support()
    support_path = tmp_path / "support.json"
    support_path.write_text(json.dumps(support), encoding="utf-8")
    support_sha256 = hashlib.sha256(support_path.read_bytes()).hexdigest()
    factorial_root = tmp_path / "factorial"
    baseline_root = tmp_path / "baseline"
    values = {
        "a6": 0.50,
        "dvt_only": 0.53,
        "density_only": 0.52,
        "basic": 0.58,
    }
    for arm in ARMS:
        for object_name in OBJECTS:
            _write_fixture(
                factorial_root,
                baseline_root,
                support,
                arm=arm,
                object_name=object_name,
                value=values[arm],
            )

    result = aggregate(
        factorial_root,
        baseline_root,
        support_path,
        expected_support_sha256=support_sha256,
    )

    assert result["means"]["basic"]["seg_AUROC_0.05"] == pytest.approx(0.58)
    assert result["effects"]["dvt_average_main_effect"]["macro"]["seg_AUROC_0.05"] == (
        pytest.approx(0.045)
    )
    assert result["effects"]["density_average_main_effect"]["macro"]["seg_F1_raw"] == (
        pytest.approx(0.035)
    )
    assert result["effects"]["interaction"]["macro"]["seg_AUPRO_0.05"] == pytest.approx(0.03)
    assert result["factor_credit"]["dvt"]["credited"] is True
    assert result["factor_credit"]["density"]["credited"] is True
    assert result["factor_credit"]["interaction"]["supported"] is True
    assert result["attribution_verdict"] == "DVT_DENSITY_INTERACTION"
    assert result["strict_method_claim"] is False
    assert result["diagnostic_only"] is True


def test_factorial_rejects_contract_drift(tmp_path: Path) -> None:
    support = _support()
    support_path = tmp_path / "support.json"
    support_path.write_text(json.dumps(support), encoding="utf-8")
    support_sha256 = hashlib.sha256(support_path.read_bytes()).hexdigest()
    factorial_root = tmp_path / "factorial"
    baseline_root = tmp_path / "baseline"
    for arm in ARMS:
        for object_name in OBJECTS:
            _write_fixture(
                factorial_root,
                baseline_root,
                support,
                arm=arm,
                object_name=object_name,
                value=0.6,
            )
    manifest_path = (
        arm_run_dir(factorial_root, baseline_root, "density_only", "can")
        / "run_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dvt_denoise_mode"] = "position_mean"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="dvt_denoise_mode"):
        aggregate(
            factorial_root,
            baseline_root,
            support_path,
            expected_support_sha256=support_sha256,
        )


def test_factor_credit_rejects_cross_metric_collapse(tmp_path: Path) -> None:
    support = _support()
    support_path = tmp_path / "support.json"
    support_path.write_text(json.dumps(support), encoding="utf-8")
    support_sha256 = hashlib.sha256(support_path.read_bytes()).hexdigest()
    factorial_root = tmp_path / "factorial"
    baseline_root = tmp_path / "baseline"
    values = {
        "a6": 0.50,
        "dvt_only": 0.53,
        "density_only": 0.50,
        "basic": 0.53,
    }
    for arm in ARMS:
        for object_name in OBJECTS:
            _write_fixture(
                factorial_root,
                baseline_root,
                support,
                arm=arm,
                object_name=object_name,
                value=values[arm],
            )
    metrics_path = (
        arm_run_dir(factorial_root, baseline_root, "dvt_only", "can")
        / "metrics.json"
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["can"]["seg_F1_raw"] = 0.40
    metrics["can"]["seg_F1"] = 0.40
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    result = aggregate(
        factorial_root,
        baseline_root,
        support_path,
        expected_support_sha256=support_sha256,
    )

    assert result["effects"]["dvt_average_main_effect"]["collapsed_objects"]["seg_F1_raw"] == [
        "can",
    ]
    assert not result["effects"]["dvt_average_main_effect"]["supported_metrics"]
    assert result["factor_credit"]["dvt"]["credited"] is False


def test_attribution_verdict_names_single_main_effect(tmp_path: Path) -> None:
    support = _support()
    support_path = tmp_path / "support.json"
    support_path.write_text(json.dumps(support), encoding="utf-8")
    support_sha256 = hashlib.sha256(support_path.read_bytes()).hexdigest()
    factorial_root = tmp_path / "factorial"
    baseline_root = tmp_path / "baseline"
    values = {
        "a6": 0.50,
        "dvt_only": 0.53,
        "density_only": 0.50,
        "basic": 0.53,
    }
    for arm in ARMS:
        for object_name in OBJECTS:
            _write_fixture(
                factorial_root,
                baseline_root,
                support,
                arm=arm,
                object_name=object_name,
                value=values[arm],
            )

    result = aggregate(
        factorial_root,
        baseline_root,
        support_path,
        expected_support_sha256=support_sha256,
    )

    assert result["factor_credit"]["dvt"]["credited"] is True
    assert result["factor_credit"]["density"]["credited"] is False
    assert result["factor_credit"]["interaction"]["supported"] is False
    assert result["attribution_verdict"] == "DVT_MAIN_EFFECT"
