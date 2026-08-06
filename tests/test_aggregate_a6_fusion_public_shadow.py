from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from scripts.aggregate_a6_fusion_public_shadow import (
    ARM_CONTRACTS,
    ARMS,
    aggregate,
    arm_run_dir,
)
from scripts.aggregate_same16_baselines import OBJECTS

if TYPE_CHECKING:
    from pathlib import Path


def _write_fixture(
    root: Path,
    support: dict[str, list[str]],
    *,
    arm: str,
    object_name: str,
    value: float,
) -> None:
    run_dir = arm_run_dir(root, arm, object_name)
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
        **ARM_CONTRACTS[arm],
        "shots": 16,
        "seed": 0,
        "feature_fusion": "layer_norm_mean",
        "support_transforms": ["identity"],
        "expansion_budget": 1.0,
        "loo_standardization": "off",
        "rgb_guide": "none",
        "binary_postprocess": "none",
        "cleanup_maps": False,
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


def test_aggregate_a6_fusion_contract_and_gates(tmp_path: Path) -> None:
    support = {
        object_name: [f"/data/{object_name}/train/good/{index:03d}.png" for index in range(16)]
        for object_name in OBJECTS
    }
    support_path = tmp_path / "support.json"
    support_path.write_text(json.dumps(support), encoding="utf-8")
    support_sha256 = hashlib.sha256(support_path.read_bytes()).hexdigest()
    offsets = {
        "basic": 0.00,
        "a6": 0.02,
        "a7": 0.00,
        "a6_l17": 0.04,
        "l17": 0.01,
    }
    for arm in ARMS:
        for object_name in OBJECTS:
            _write_fixture(
                tmp_path,
                support,
                arm=arm,
                object_name=object_name,
                value=0.6 + offsets[arm],
            )

    result = aggregate(
        tmp_path,
        support_path,
        expected_support_sha256=support_sha256,
    )

    assert result["means"]["a6"]["seg_AUROC_0.05"] == pytest.approx(0.62)
    assert result["gates"] == {
        "a6_simplification_safe": True,
        "flow_without_dvt_supported": True,
        "dominant_l17_repair_supported": True,
    }
    assert result["comparisons"]["a6_minus_a7"]["positive_object_count"]["seg_F1_raw"] == 8


def test_aggregate_a6_fusion_rejects_contract_drift(tmp_path: Path) -> None:
    support = {
        object_name: [f"/data/{object_name}/train/good/{index:03d}.png" for index in range(16)]
        for object_name in OBJECTS
    }
    support_path = tmp_path / "support.json"
    support_path.write_text(json.dumps(support), encoding="utf-8")
    support_sha256 = hashlib.sha256(support_path.read_bytes()).hexdigest()
    for arm in ARMS:
        for object_name in OBJECTS:
            _write_fixture(
                tmp_path,
                support,
                arm=arm,
                object_name=object_name,
                value=0.6,
            )
    manifest_path = arm_run_dir(tmp_path, "a6", "can") / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["density_weight"] = 0.25
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="density_weight"):
        aggregate(
            tmp_path,
            support_path,
            expected_support_sha256=support_sha256,
        )


def test_flow_gate_rejects_collapse_on_another_primary_metric(tmp_path: Path) -> None:
    support = {
        object_name: [f"/data/{object_name}/train/good/{index:03d}.png" for index in range(16)]
        for object_name in OBJECTS
    }
    support_path = tmp_path / "support.json"
    support_path.write_text(json.dumps(support), encoding="utf-8")
    support_sha256 = hashlib.sha256(support_path.read_bytes()).hexdigest()
    offsets = {
        "basic": 0.00,
        "a6": 0.02,
        "a7": 0.00,
        "a6_l17": 0.04,
        "l17": 0.01,
    }
    for arm in ARMS:
        for object_name in OBJECTS:
            _write_fixture(
                tmp_path,
                support,
                arm=arm,
                object_name=object_name,
                value=0.6 + offsets[arm],
            )
    metrics_path = arm_run_dir(tmp_path, "a6", "can") / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["can"]["seg_AUROC_0.05"] = 0.50
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    result = aggregate(
        tmp_path,
        support_path,
        expected_support_sha256=support_sha256,
    )

    assert result["comparisons"]["a6_minus_a7"]["supported_metrics"]
    assert result["gates"]["flow_without_dvt_supported"] is False


def test_l17_gate_rejects_collapse_on_another_primary_metric(tmp_path: Path) -> None:
    support = {
        object_name: [f"/data/{object_name}/train/good/{index:03d}.png" for index in range(16)]
        for object_name in OBJECTS
    }
    support_path = tmp_path / "support.json"
    support_path.write_text(json.dumps(support), encoding="utf-8")
    support_sha256 = hashlib.sha256(support_path.read_bytes()).hexdigest()
    offsets = {
        "basic": 0.00,
        "a6": 0.02,
        "a7": 0.00,
        "a6_l17": 0.04,
        "l17": 0.01,
    }
    for arm in ARMS:
        for object_name in OBJECTS:
            _write_fixture(
                tmp_path,
                support,
                arm=arm,
                object_name=object_name,
                value=0.6 + offsets[arm],
            )
    metrics_path = arm_run_dir(tmp_path, "a6_l17", "can") / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["can"]["seg_AUROC_0.05"] = 0.50
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    result = aggregate(
        tmp_path,
        support_path,
        expected_support_sha256=support_sha256,
    )

    assert result["comparisons"]["a6_l17_minus_a6"]["supported_metrics"]
    assert result["gates"]["dominant_l17_repair_supported"] is False
