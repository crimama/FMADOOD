from __future__ import annotations

import os
import runpy
from pathlib import Path

import pytest


def test_gate1_runner_configures_deterministic_cublas_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the detached container does not provide a CuBLAS workspace setting.
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    runner = Path(__file__).resolve().parents[1] / "scripts" / "run_flow_tte_darc_gate1.py"

    # When: the production runner is loaded through its real entry module.
    runpy.run_path(str(runner), run_name="darc_gate1_runner_import_test")

    # Then: strict deterministic CUDA operations have the required workspace.
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_gate1_aggregate_rejects_partial_object_seed_matrix(tmp_path: Path) -> None:
    # Given
    runner = Path(__file__).resolve().parents[1] / "scripts" / "run_flow_tte_darc_gate1.py"
    namespace = runpy.run_path(str(runner), run_name="darc_gate1_partial_aggregate_test")

    # When / Then
    with pytest.raises(RuntimeError, match="exactly 15 objects x 3 seeds"):
        namespace["_aggregate"](tmp_path)


def test_remote_controller_reads_nested_completion_method_hash() -> None:
    # Given
    controller = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_flow_tte_darc_gate1_remote.sh"
    )

    # When
    source = controller.read_text(encoding="utf-8")

    # Then
    assert 'provenance.get("method_sha256")' in source
    assert 'payload.get("method_hash")' not in source
