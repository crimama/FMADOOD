from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_flowtte_semantic_atlas_metric_remote.sh"


def source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_shell_syntax_is_valid() -> None:
    subprocess.run(["/usr/bin/bash", "-n", str(SCRIPT)], check=True)  # noqa: S603


def test_remote_controller_is_support_only_and_run_name_guarded() -> None:
    text = source()

    assert 'FIXED_CONTAINER="hun_fsad_tta_012"' in text
    assert 'HOST_GPU_ID="${HOST_GPU_ID:-3}"' in text
    assert "flowtte_semantic_atlas_" in text
    assert "^flowtte_semantic_atlas_" in text
    assert "run_public" not in text
    assert "all8_support_authorized.txt" in text
    assert "public_access_authorized" not in text
    assert "scripts/run_flowtte_semantic_atlas_metric.py" in text
    assert "scripts/aggregate_flowtte_semantic_atlas_metric.py" in text


def test_snapshot_preflight_and_hash_contract_are_frozen() -> None:
    text = source()

    assert "--exclude=.git" in text
    assert "--exclude=.venv" in text
    assert "--exclude=results" in text
    assert "--exclude=configs/remote" in text
    assert "9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5" in text
    assert "FMAD_DINOV3_OFFLINE=1" in text
    assert "torch.cuda.device_count()" in text
    assert "flow_tte.semantic_atlas_metric" in text
    assert "aggregate_flowtte_semantic_atlas_metric" in text


def test_status_surfaces_progress_and_gpu_guard() -> None:
    text = source()

    assert "progress.jsonl" in text
    assert "tail -n 12" in text
    assert "GPU_BINDING_OK" in text
    assert ".flowtte_semantic_atlas_gpu3.lock" in text
    assert "--query-compute-apps=pid" in text
    assert "kill -0" in text
    assert "kill -9" not in text
    assert "pkill" not in text


def test_pull_and_cleanup_are_scoped() -> None:
    text = source()

    assert '[[ "${snapshot}" == /workspace/isolated_runs/flowtte_semantic_atlas_* ]]' in text
    assert "! -path './compact_artifact_sha256.txt'" in text
    assert "grep -vE '  compact_artifact_sha256\\.txt$'" in text
    assert "sha256sum -c <(" in text
    assert "dense_feature_cache" in text
    assert "patch_score_cache" in text
    assert "start|status|pull|cleanup|invalidate" in text
