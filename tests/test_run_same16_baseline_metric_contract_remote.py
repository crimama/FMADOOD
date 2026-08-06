from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_same16_baseline_metric_contract_remote.sh"


def source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_shell_syntax_is_valid() -> None:
    subprocess.run(["/usr/bin/bash", "-n", str(SCRIPT)], check=True)  # noqa: S603


def test_isolated_flow_tte_runs_import_from_the_snapshot() -> None:
    text = source()
    assert text.count('--project-root "${snapshot}" --fsad-root "${snapshot}"') == 2
    assert "--project-root /workspace" not in text


def test_postcompute_failures_write_a_failed_marker() -> None:
    text = source()
    assert "FAILED aggregate_same16_baselines" in text
    assert "FAILED aggregate_a6_fusion_public_shadow" in text
    assert "FAILED adjudicate_same16_a6" in text
