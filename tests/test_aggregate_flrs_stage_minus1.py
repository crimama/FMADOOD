from __future__ import annotations

# ruff: noqa: ANN202, PT018, SLF001
import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_flrs_stage_minus1.py"
    spec = importlib.util.spec_from_file_location("aggregate_flrs_stage_minus1", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows():
    rows = []
    for index in range(8):
        row = {
            "object": f"o{index}",
            "flrs_shift_inflation": 0.4,
            "raw_shift_inflation": 1.0,
            "shift_reduction": 0.6,
            "coverage_ok": True,
            "shift_pair_ok": True,
            "access_ok": True,
            "support_ok": True,
        }
        for population in ("clean", "shifted"):
            for metric in ("pAUROC_0.05", "F1"):
                prefix = f"{population}_{metric}"
                row[f"{prefix}_flrs"] = 0.90
                row[f"{prefix}_raw"] = 0.88
                row[f"{prefix}_raw_delta"] = 0.02
                row[f"{prefix}_null_envelope"] = 0.89
                row[f"{prefix}_null_margin"] = 0.01
        row["worst_null_margin_pAUROC_0.05"] = 0.01
        row["worst_null_margin_F1"] = 0.01
        rows.append(row)
    return rows


def test_strict_gate_passes_frozen_positive_fixture() -> None:
    module = _module()
    rows = _rows()
    macro = module._macro(rows)
    gates = module._gates(
        rows,
        macro,
        {
            "shift_reduction": 0.5,
            "null_margin_pAUROC_0.05": 0.005,
            "null_margin_F1": 0.005,
        },
    )
    assert gates["strict"] is True
    assert gates["soft"] is True
    json.dumps(gates)


def test_one_object_null_failure_blocks_strict() -> None:
    module = _module()
    rows = _rows()
    rows[0]["shifted_F1_null_margin"] = 0.0
    macro = module._macro(rows)
    gates = module._gates(
        rows,
        macro,
        {
            "shift_reduction": 0.5,
            "null_margin_pAUROC_0.05": 0.005,
            "null_margin_F1": 0.005,
        },
    )
    assert gates["G3_hard_null_superiority"] is False
    assert gates["strict"] is False


def test_zero_raw_shift_denominator_blocks_reduction_gate() -> None:
    module = _module()
    rows = _rows()
    rows[0]["raw_shift_inflation"] = 0.0
    macro = module._macro(rows)
    gates = module._gates(
        rows,
        macro,
        {
            "shift_reduction": 0.5,
            "null_margin_pAUROC_0.05": 0.005,
            "null_margin_F1": 0.005,
        },
    )
    assert gates["G1_shift_reduction"] is False
