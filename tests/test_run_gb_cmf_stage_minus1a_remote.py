from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_gb_cmf_stage_minus1a_remote.sh"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_shell_syntax_is_valid() -> None:
    subprocess.run(["/usr/bin/bash", "-n", str(SCRIPT)], check=True)  # noqa: S603


def test_controller_freezes_container_case_gpu_and_protocol() -> None:
    text = _source()
    assert 'FIXED_CONTAINER="hun_fsad_tta_012"' in text
    assert "run_case 0 rice 0" in text
    assert "run_case 1 can 1" in text
    assert "run_case 2 rice 1" in text
    assert "run_case 3 can 0" in text
    assert 'CUDA_VISIBLE_DEVICES="${gpu}"' in text
    assert "FMAD_DINOV3_OFFLINE=1" in text
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in text
    assert "7,15,23,31" in text
    assert "docker run" not in text
    assert "docker rm" not in text


def test_controller_has_safe_gpu_lock_pull_and_cleanup() -> None:
    text = _source()
    assert "--query-compute-apps=pid" in text
    assert "flock -n 9" in text
    assert "sha256sum -c compact_artifact_sha256.txt" in text
    assert '[[ "${snapshot}" == /workspace/isolated_runs/gb_cmf_stage_minus1a_* ]]' in text
    assert "kill -9" not in text
    assert "pkill" not in text
    for dense_name in (
        "anomaly_maps",
        "score_maps",
        "dense_feature_cache",
        "transformed_images",
        "patch_score_cache",
    ):
        assert dense_name in text


def test_controller_never_launches_next_stage() -> None:
    text = _source()
    assert "stage_minus1b" not in text.lower()
    assert "start|status|pull|cleanup" in text
