from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_same16_dvt_density_factorial_remote.sh"


def source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_shell_syntax_is_valid() -> None:
    subprocess.run(["/usr/bin/bash", "-n", str(SCRIPT)], check=True)  # noqa: S603


def test_only_missing_factorial_cells_are_run() -> None:
    text = source()
    assert 'for arm in dvt_only density_only; do' in text
    assert 'dvt="position_mean"; density="0.0"' in text
    assert 'dvt="none"; density="0.25"' in text
    assert "run_basic_object" not in text
    assert "run_knn_object" not in text


def test_snapshot_and_raw_map_contract_are_explicit() -> None:
    text = source()
    assert '--project-root "${snapshot}" --fsad-root "${snapshot}"' in text
    assert "--project-root /workspace" not in text
    assert "--rgb-guide none --threshold-calibration-mode none --binary-postprocess none" in text
    assert "--feature-layers 5,11,17,23" in text
    assert "--normality-mode fused" in text


def test_completed_run_cleans_dense_maps() -> None:
    text = source()
    assert "FAILED dense_maps_remain" in text
    assert "dense_anomaly_and_calibration_map_directories=0" in text
