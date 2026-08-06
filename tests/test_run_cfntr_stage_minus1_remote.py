from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_cfntr_stage_minus1_remote.sh"


def source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_shell_syntax_is_valid() -> None:
    subprocess.run(["/usr/bin/bash", "-n", str(SCRIPT)], check=True)  # noqa: S603


def test_controller_uses_only_the_frozen_fixed_container_and_safe_auth() -> None:
    text = source()
    assert 'FIXED_CONTAINER="hun_fsad_tta_012"' in text
    assert 'source "${PRESET}"' in text
    assert "SSH_ASKPASS_REQUIRE=force" in text
    assert "docker run" not in text
    assert "docker rm" not in text
    assert "nvidia-smi" in text
    assert "--query-compute-apps=pid" in text
    assert "kill -0" in text
    assert "kill -9" not in text
    assert "pkill" not in text


def test_snapshot_and_protocol_inputs_are_frozen() -> None:
    text = source()
    assert "REMOTE_SNAPSHOT_ROOT" in text
    assert "--exclude=.git" in text
    assert "--exclude=.venv" in text
    assert "--exclude=results" in text
    assert "--exclude=configs/remote" in text
    assert "MIN_FREE_GIB" in text
    assert "7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3" in text
    assert "FMAD_DINOV2_OFFLINE=1" in text
    assert "command -v flock" in text
    assert 'python3 -c \\"import cv2, numpy, sklearn, torch\\"' in text
    assert "torch.cuda.device_count()" in text
    assert "test_public" not in text


def test_gpu_groups_and_runner_slot_contract() -> None:
    text = source()
    assert 'OBJECT_GROUP_0="can fruit_jelly"' in text
    assert 'OBJECT_GROUP_1="fabric sheet_metal"' in text
    assert 'OBJECT_GROUP_2="rice wallplugs"' in text
    assert 'OBJECT_GROUP_3="vial walnuts"' in text
    assert 'CUDA_VISIBLE_DEVICES="${gpu}"' in text
    assert "--container-cuda-slot 0" in text
    assert "flock -n 9" in text
    aggregate_at = text.rindex("aggregate_cfntr_stage_minus1.py")
    all_groups_at = text.index("for gpu in 0 1 2 3; do", aggregate_at - 500)
    assert all_groups_at < aggregate_at


def test_compact_pull_and_cleanup_are_scoped() -> None:
    text = source()
    for dense_name in (
        "anomaly_maps",
        "score_maps",
        "dense_feature_cache",
        "transformed_images",
        "patch_score_cache",
    ):
        assert dense_name in text
    assert '[[ "${snapshot}" == /workspace/isolated_runs/cfntr_stage_minus1_* ]]' in text
    assert "docker exec -i '${FIXED_CONTAINER}' bash -s --" in text
    assert 'ps -o stat= -p "${controller_pid}"' in text
    assert '"${controller_state}" != Z*' in text
    assert 'test -f "${result_root}/complete.txt"' in text
    assert "summary.json per_object.tsv bootstrap.tsv verdict.txt" in text
    assert "sha256sum -c compact_artifact_sha256.txt" in text
    assert "start|status|pull|cleanup" in text
