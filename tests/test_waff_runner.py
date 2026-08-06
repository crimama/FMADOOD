from __future__ import annotations

import ast
from pathlib import Path


def _runner_tree() -> ast.Module:
    runner = Path(__file__).parents[1] / "scripts" / "run_waff_hplus_stage_minus1.py"
    return ast.parse(runner.read_text(encoding="utf-8"))


def test_runner_has_no_dataset_or_test_enumeration_call() -> None:
    tree = _runner_tree()
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert "get_test_images" not in called_attributes
    assert "stream_test_images" not in called_attributes
    assert "MVTecAD2Dataset" not in imported_names
    assert "fmad.datasets.mvtec_ad2" not in imported_names


def test_runner_freezes_only_low_frequency_defect_families() -> None:
    assignments = {
        target.id: ast.literal_eval(node.value)
        for node in _runner_tree().body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id == "DEFECT_FAMILIES"
    }

    assert assignments["DEFECT_FAMILIES"] == (
        "cutpaste",
        "chromatic_blotch",
        "gaussian_blur",
    )


def test_runner_records_zero_test_access_and_no_public_metrics() -> None:
    tree = _runner_tree()
    constants = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)}

    assert "test_access_attempts" in constants
    assert "public_metrics_computed" in constants
    assert "claim_authorized" in constants
