import csv
import json
import shutil
from pathlib import Path

from scripts.collect_psma_mr_dsba1 import (
    CANONICAL_FIELDS,
    append_canonical_ledger,
    condition_checks,
    parse_yaml_mapping,
)
from scripts.psma_mr_queue import validate_plan, v2_artifact_identity


EXPECTED = {
    "revision": "v2",
    "method": "PSMA_MR",
    "ledger_method": "PSMA_MR_v2",
    "dataset": "mvtec_ad2",
    "backbone": "dinov2_vitl14",
    "seed": 0,
    "shots": 16,
    "preprocess": "no_mask_no_rotation",
    "query_split": "test_public",
    "metric_contract": "mvtec_ad2_float16_streaming_v1",
    "feature_layers": [5, 11, 17, 23],
}


def _config(output_dir: str, preprocess: str = "no_mask_no_rotation") -> str:
    return f"""backbone: dinov2_vitl14
dataset: mvtec_ad2
dataset_config:
  objects:
  - can
  preprocess: {preprocess}
evaluation:
  metric_contract: mvtec_ad2_float16_streaming_v1
method: PSMA_MR
method_config:
  revision: v2
  resolutions:
  - 448
  - 672
output:
  save_anomaly_maps: false
  save_weights: false
output_dir: {output_dir}
backbone_config:
  feature_layers: [5, 11, 17, 23]
protocol:
  shots: 16
seed: 0
"""


def _manifest(rotation: bool = False) -> dict:
    return {
        "run_id": "run-v2-test",
        "config_hash": "config-v2-test",
        "code_sha256": "code-v2-test",
        "objects": {
            "can": {
                "method": "PSMA_MR",
                "revision": "v2",
                "feature_layers": [5, 11, 17, 23],
                "backbone": {"name": "dinov2_vitl14"},
                "protocol": {
                    "support_count": 16,
                    "support_paths": [f"/data/can/train/good/{i}.png" for i in range(16)],
                    "query_count": 80,
                    "query_identity_sha256": "query-identity",
                    "query_content_sha256": "query-content",
                    "ground_truth_sha256": "ground-truth",
                    "context": {"masking": False, "rotation": rotation},
                },
                "training": {"schedule_full_cartesian": True},
                "calibration": {
                    "clean_normal_only": True,
                    "query_used": False,
                    "ground_truth_used": False,
                },
            }
        },
    }


def test_collector_parses_multiline_yaml_and_current_manifest_schema() -> None:
    output_dir = "/workspace/results_remote/psma_mr_v2_dsba1_20260909/gpu0_can"
    config = parse_yaml_mapping(_config(output_dir))
    assert config is not None
    checks = condition_checks(
        config,
        _manifest(),
        {
            "metric_contract": "mvtec_ad2_float16_streaming_v1",
            "seg_AUPRO_0.05": 0.4,
            "seg_AUROC_0.05": 0.7,
        },
        "can",
        EXPECTED,
        output_dir,
    )
    assert all(checks.values()), checks
    assert "query_paths" not in _manifest()["objects"]["can"]["protocol"]


def test_collector_rejects_v1_preprocess_or_rotation_identity() -> None:
    output_dir = "/workspace/results_remote/psma_mr_v2_dsba1_20260909/gpu0_can"
    config = parse_yaml_mapping(_config(output_dir, preprocess="agnostic"))
    assert config is not None
    checks = condition_checks(
        config,
        _manifest(rotation=True),
        {
            "metric_contract": "mvtec_ad2_float16_streaming_v1",
            "seg_AUPRO_0.05": 0.4,
            "seg_AUROC_0.05": 0.7,
        },
        "can",
        EXPECTED,
        output_dir,
    )
    assert not checks["preprocess"]
    assert not checks["context_no_mask_no_rotation"]
    assert not checks["v2_identity"]
    assert not v2_artifact_identity(_config(output_dir, "agnostic"), _manifest(True), "can")
    assert v2_artifact_identity(_config(output_dir), _manifest(False), "can")


def test_canonical_append_preserves_history_and_is_idempotent(tmp_path: Path) -> None:
    source_csv = Path(
        "results/comparisons/psma_mr_vs_history_20260909/psma_mr_vs_history_20260909.csv"
    )
    source_md = source_csv.with_suffix(".md")
    canonical_csv = tmp_path / "canonical.csv"
    canonical_md = tmp_path / "canonical.md"
    shutil.copy2(source_csv, canonical_csv)
    shutil.copy2(source_md, canonical_md)
    original_rows = list(csv.DictReader(source_csv.open(newline="", encoding="utf-8")))
    assert len(original_rows) == 26

    classes = ["can", "sheet_metal", "fabric", "vial", "fruit_jelly", "wallplugs", "rice", "walnuts"]
    rows = []
    for index, name in enumerate(classes):
        rows.append(
            {
                "class": name,
                "result_status": "succeeded",
                "condition_status": "verified",
                "seg_AUPRO_0.05": 0.2 + index / 100.0,
                "seg_AUROC_0.05": 0.6 + index / 100.0,
                "config_hash": f"config-v2-{name}",
                "run_id": f"run-v2-{name}",
                "support_identity_sha256": f"support-{name}",
                "metrics_path": str(tmp_path / name / "metrics.json"),
            }
        )
    plan = {
        "local_root": str(tmp_path / "collector"),
        "canonical_csv": str(canonical_csv),
        "canonical_md": str(canonical_md),
        "expected": EXPECTED,
    }
    snapshot = {"terminal": True, "timestamp": "now", "rows": rows}
    result = append_canonical_ledger(plan, snapshot)
    assert result["status"] == "appended"
    assert result["appended"] == 9
    appended_rows = list(csv.DictReader(canonical_csv.open(newline="", encoding="utf-8")))
    assert appended_rows[:26] == original_rows
    assert len(appended_rows) == 35
    assert all(row["method"] == "PSMA_MR_v2" for row in appended_rows[26:])
    assert "## PSMA_MR_v2 dsba1" in canonical_md.read_text(encoding="utf-8")

    result_again = append_canonical_ledger(plan, snapshot)
    assert result_again["status"] == "already_present"
    assert result_again["appended"] == 0
    assert len(list(csv.DictReader(canonical_csv.open(newline="", encoding="utf-8")))) == 35


def test_canonical_append_is_incremental_without_partial_macro(tmp_path: Path) -> None:
    source_csv = Path(
        "results/comparisons/psma_mr_vs_history_20260909/psma_mr_vs_history_20260909.csv"
    )
    canonical_csv = tmp_path / "canonical.csv"
    canonical_md = tmp_path / "canonical.md"
    shutil.copy2(source_csv, canonical_csv)
    shutil.copy2(source_csv.with_suffix(".md"), canonical_md)
    row = {
        "class": "can",
        "result_status": "succeeded",
        "condition_status": "verified",
        "seg_AUPRO_0.05": 0.41,
        "seg_AUROC_0.05": 0.71,
        "config_hash": "config-v2-incremental",
        "run_id": "run-v2-incremental",
        "support_identity_sha256": "support-can",
        "metrics_path": str(tmp_path / "can" / "metrics.json"),
    }
    plan = {
        "local_root": str(tmp_path / "collector"),
        "canonical_csv": str(canonical_csv),
        "canonical_md": str(canonical_md),
        "expected": EXPECTED,
    }
    result = append_canonical_ledger(
        plan, {"terminal": False, "timestamp": "now", "rows": [row]}
    )
    assert result["scope"] == "dsba1_v2_partial"
    assert result["appended"] == 1
    rows = list(csv.DictReader(canonical_csv.open(newline="", encoding="utf-8")))
    assert len(rows) == 27
    assert rows[-1]["row_type"] == "class"
    assert not any(item["row_type"] == "macro" and item["method"] == "PSMA_MR_v2" for item in rows)


def test_queue_plan_requires_active_identity_before_watch() -> None:
    plan = json.loads(
        Path(
            "results/remote_runs/psma_mr_v2_dsba1_20260909/plans/gpu0.json"
        ).read_text(encoding="utf-8")
    )
    assert validate_plan(plan, require_active_identity=False) is None
    assert "active process pid is not populated" in (validate_plan(plan) or "")
