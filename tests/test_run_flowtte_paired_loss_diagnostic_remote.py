from __future__ import annotations

from pathlib import Path

SCRIPT = Path("scripts/run_flowtte_paired_loss_diagnostic_remote.sh")


def test_remote_controller_is_support_only_and_run_name_guarded() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "flowtte_paired_loss_" in text
    assert "^flowtte_paired_loss_" in text
    assert "run_public" not in text
    assert "test_public" not in text
    assert "all8_support_authorized.txt" in text
    assert "scripts/run_flowtte_paired_loss_diagnostic.py" in text
    assert "scripts/aggregate_flowtte_paired_loss_diagnostic.py" in text


def test_remote_status_surfaces_progress_rows() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "progress.jsonl" in text
    assert "tail -n 12" in text
    assert "GPU_BINDING_OK" in text
    assert ".flowtte_paired_loss_gpu3.lock" in text
