from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts.aggregate_waff_hplus_stage_minus1 import EXPECTED_CASES, aggregate

if TYPE_CHECKING:
    from pathlib import Path


def _write_case(root: Path, object_name: str, seed: int, *, noncollapse: bool = True) -> None:
    chunk = root / "chunks" / f"{object_name}_seed{seed}"
    chunk.mkdir(parents=True)
    payload = {
        "query_or_test_access": False,
        "public_metrics_computed": False,
        "seed": seed,
        "results": [
            {
                "object": object_name,
                "noncollapse": noncollapse,
                "static_memory_unchanged": True,
                "integration_max_abs_error": 0.0,
                "active_score_std": 0.2,
                "default_worst_axis_margin_drop": 0.4,
                "waff_worst_axis_margin_drop": 0.3,
                "waff_scorer": {"weight": [0.8, 0.45]},
                "simplex_feasibility": {
                    "joint_feasible": True,
                    "min_joint_violation": -0.1,
                },
            },
        ],
    }
    (chunk / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_aggregator_passes_four_static_noncollapse_cases(tmp_path: Path) -> None:
    for object_name, seed in EXPECTED_CASES:
        _write_case(tmp_path, object_name, seed)

    result = aggregate(tmp_path)

    assert result["gate_pass"] is True
    assert result["selection_credit"] is False
    assert result["l2_ready"] is True
    assert result["public_access_authorized"] is False
    assert result["case_count"] == 4


def test_aggregator_rejects_public_access_or_incomplete_cases(tmp_path: Path) -> None:
    _write_case(tmp_path, "can", 0)
    summary = tmp_path / "chunks" / "can_seed0" / "summary.json"
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["query_or_test_access"] = True
    summary.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="support-only"):
        aggregate(tmp_path)
