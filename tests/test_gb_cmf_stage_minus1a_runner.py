from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "scripts" / "run_gb_cmf_stage_minus1a.py"


def _tree() -> ast.Module:
    return ast.parse(RUNNER.read_text(encoding="utf-8"))


def test_runner_has_no_dataset_or_test_enumeration() -> None:
    tree = _tree()
    attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert "get_test_images" not in attributes
    assert "stream_test_images" not in attributes
    assert "MVTecAD2Dataset" not in imported
    assert "fmad.datasets.mvtec_ad2" not in imported


def test_case_assignment_and_five_arms_are_frozen() -> None:
    assignments = {
        target.id: ast.literal_eval(node.value)
        for node in _tree().body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id in {"ARMS", "ASSIGNMENTS", "EXPECTED_HELD_FILES"}
    }
    arms = assignments["ARMS"]
    cases = assignments["ASSIGNMENTS"]
    held_files = assignments["EXPECTED_HELD_FILES"]
    assert arms == (
        "default",
        "nonfactorial",
        "gb_cmf",
        "affine_null",
        "nonlinear_head_null",
    )
    assert set(cases) == {("rice", 0), ("can", 1), ("rice", 1), ("can", 0)}
    assert {value[3] for value in cases.values()} == {0, 1, 2, 3}
    assert all(len(value) == 4 for value in held_files.values())


def test_parser_enforces_frozen_steps_case_and_gpu() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'parser.add_argument("--steps", type=int, default=300)' in source
    assert "if key not in ASSIGNMENTS:" in source
    assert "if args.steps != 300:" in source
    assert "if int(args.host_gpu_id) != ASSIGNMENTS[key][3]:" in source


def test_runner_records_support_only_access_invariants() -> None:
    constants = {node.value for node in ast.walk(_tree()) if isinstance(node, ast.Constant)}
    assert "query_or_test_access" in constants
    assert "test_access_attempts" in constants
    assert "public_metrics_computed" in constants
    assert "stage_minus1b_authorized" in constants
