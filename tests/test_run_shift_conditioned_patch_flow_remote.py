from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_shift_conditioned_patch_flow_remote.sh"


def _text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _function_body(name: str, next_name: str) -> str:
    text = _text()
    match = re.search(
        rf"^{re.escape(name)}\(\) \{{(?P<body>.*?)^\}}\n\n{re.escape(next_name)}\(\)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None
    return match.group("body")


def test_shell_has_valid_bash_syntax() -> None:
    result = subprocess.run(  # noqa: S603 - fixed local script syntax check.
        ["/usr/bin/bash", "-n", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_fixed_dsba5_resource_and_snapshot_custody() -> None:
    text = _text()
    assert 'PRESET="${ROOT_DIR}/configs/remote/dsba5.env"' in text
    assert 'FIXED_CONTAINER="hun_fsad_tta_012"' in text
    assert 'HOST_GPU_ID="3"' in text
    assert 'CONTAINER_CUDA_SLOT="0"' in text
    assert "torch.cuda.device_count()" in text
    assert " -eq 1" in text
    assert 'test ! -e \\"${snapshot}\\"' in text
    assert "snapshot_hash.txt" in text
    assert "host_uuid" in text
    assert "container_uuid" in text


def test_frozen_support_and_hplus_cache_are_preflighted() -> None:
    text = _text()
    assert "9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5" in text
    assert 'SMOKE_OBJECTS="fruit_jelly,walnuts,vial,fabric"' in text
    assert "preflight_dinov3_cache.py" in text
    assert "--model dinov3_vith16plus" in text
    assert "--feature-layers 7,15,23,31" in text
    assert "tests/test_shift_conditioned_flow.py" in text
    assert "--phase smoke" in text


def test_preflight_does_not_touch_public_or_private_data() -> None:
    body = _function_body("preflight", "run_mechanism")
    assert "test_public" not in body
    assert "test_private" not in body
    assert "ground_truth" not in body
    assert "/train/good" in body


def test_public_execution_is_strictly_after_machine_readable_sentinel() -> None:
    body = _function_body("internal_controller", "start_remote")
    sentinel_check = body.index('if [[ ! -f "${root}/mechanism/aggregate/mechanism_passed.txt" ]]')
    public_call = body.index("run_public")
    assert sentinel_check < public_call
    run_public = _function_body("run_public", "aggregate_public")
    assert '[[ -f "${sentinel}" ]] || return 86' in run_public
    assert '--mechanism-pass-sentinel "${sentinel}"' in run_public
    assert "--phase public" in run_public


def test_controller_has_guarded_lifecycle_commands() -> None:
    text = _text()
    assert (
        "{start|status|pull|cleanup|cleanup-preflight-failure|invalidate|bootstrap-runtime}"
        in text
    )
    assert "^shift_conditioned_patch_flow_[A-Za-z0-9._-]+$" in text
    cleanup = _function_body("cleanup_remote", "cleanup_preflight_failure_remote")
    assert "/workspace/isolated_runs/shift_conditioned_patch_flow_*" in cleanup
    assert "complete.txt" in cleanup
    assert "invalidated.txt" in cleanup
    assert "rm -rf -- '${snapshot}'" in cleanup
    assert "${REMOTE_RESULTS_ROOT}/${RUN_NAME}" not in cleanup.split("rm -rf --", maxsplit=1)[1]
    preflight_cleanup = _function_body(
        "cleanup_preflight_failure_remote",
        "invalidate_remote",
    )
    assert "started.txt" in preflight_cleanup
    assert "mechanism" in preflight_cleanup
    assert "public" in preflight_cleanup


def test_compact_results_are_checksummed_and_pull_is_verified() -> None:
    text = _text()
    assert "compact_artifact_sha256.txt" in text
    assert "sha256sum -c compact_artifact_sha256.txt" in text
    assert "mechanism/aggregate/verdict.json" in text
    assert "public/aggregate/verdict.json" in text
    assert "mechanism/${object_name}/checkpoint.pt" in text
