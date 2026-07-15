from __future__ import annotations

import os
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_flow_tte_darc_gate2.py"
CONTROLLER = ROOT / "scripts" / "run_flow_tte_darc_gate2_remote.sh"


def test_gate2_runner_configures_deterministic_cublas_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the detached container does not provide a CuBLAS workspace setting.
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    # When: the production runner is loaded through its real entry module.
    runpy.run_path(str(RUNNER), run_name="darc_gate2_runner_import_test")

    # Then: strict deterministic CUDA operations have the required workspace.
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_gate2_runner_uses_registered_feature_layers_without_low_branch() -> None:
    # Given: the production runner source is the executable configuration boundary.
    source = RUNNER.read_text(encoding="utf-8")

    # When: its feature-stream configuration is inspected.
    # Then: Gate2 uses layer 7 micro, layer 23 coarse, and skips the unused low branch.
    assert "DINOv3EarlyExitAdapter(model, (7,)" in source
    assert "DINOv3EarlyExitAdapter(model, (23,)" in source
    assert "FeatureStreamConfig(device=device, include_low=False)" in source


def test_gate2_aggregate_rejects_partial_object_seed_matrix(tmp_path: Path) -> None:
    # Given: an empty output root cannot contain the frozen 45-cell population.
    namespace = runpy.run_path(str(RUNNER), run_name="darc_gate2_partial_aggregate_test")

    # When / Then: aggregation rejects the partial population.
    with pytest.raises(ValueError, match="exactly 45 object-seed completions"):
        namespace["_aggregate"](tmp_path)


def test_remote_controller_prechecks_gate1_and_never_runs_paper_baselines() -> None:
    # Given: the remote controller is the only multi-GPU launch surface.
    source = CONTROLLER.read_text(encoding="utf-8")

    # When / Then: it gates only on the frozen Gate1 population and schedules Gate2.
    assert (
        "/workspace/results_remote/darc_v11_g1_ad1synthetic15_p16r_20260710_v1/gate_decision.json"
    ) in source
    assert 'decision.get("passed") is not True' in source
    assert 'decision.get("source_count") != 720' in source
    assert "darc_v11_g2_ad1synthetic15_p16r_20260711_g4_v1" in source
    assert "run_flow_tte_darc_gate2.py" in source
    assert "superadd" not in source.lower()
    assert "superad" not in source.lower()


def test_remote_controller_validates_exact_nested_completion_population() -> None:
    # Given: Gate2 requires exactly 15 objects by three seeds.
    source = CONTROLLER.read_text(encoding="utf-8")

    # When / Then: the controller rejects incomplete or mixed-method populations.
    assert "expected 45 completion files" in source
    assert 'provenance.get("method_sha256")' in source
    assert 'payload.get("source_count") != 16' in source
    assert 'payload.get("smoke") is not False' in source


def test_remote_controller_launches_all_objects_concurrently_across_four_gpus() -> None:
    # Given: CPU-heavy correspondence can safely overlap independent object jobs.
    source = CONTROLLER.read_text(encoding="utf-8")
    assignments = {
        0: ("bottle", "carpet", "leather", "screw"),
        1: ("transistor", "cable", "grid", "metal_nut"),
        2: ("tile", "wood", "capsule", "hazelnut"),
        3: ("pill", "toothbrush", "zipper"),
    }

    # When / Then: every AD1 object has one independent job and all are awaited.
    for gpu, object_names in assignments.items():
        for object_name in object_names:
            assert f"launch_object {gpu} {object_name}" in source
    assert source.count("\nlaunch_object ") == 15
    assert 'for index in "${!pids[@]}"' in source


def test_remote_controller_forwards_termination_to_python_jobs() -> None:
    # Given: a detached controller may receive TERM during resource reconfiguration.
    source = CONTROLLER.read_text(encoding="utf-8")

    # When / Then: tracked jobs exec Python and the trap is armed before launch.
    assert "exec env CUDA_VISIBLE_DEVICES=" in source
    assert source.index("trap '") < source.index("launch_object 0 bottle")
