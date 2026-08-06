from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from aggregate_dscr_density_condition_audit import aggregate  # noqa: E402


def _write_object(
    root: Path,
    object_name: str,
    *,
    regular_p: float = 0.01,
    regular_f1: float = 0.01,
    interaction_p: float = -0.04,
    interaction_f1: float = -0.02,
) -> Path:
    metrics = ("seg_AUROC_0.05", "seg_F1", "pixel_AP")
    regular = {
        "seg_AUROC_0.05": regular_p,
        "seg_F1": regular_f1,
        "pixel_AP": 0.0,
    }
    interaction = {
        metric: {
            "regular_delta": regular[metric],
            "mean_shifted_delta": regular[metric]
            + {
                "seg_AUROC_0.05": interaction_p,
                "seg_F1": interaction_f1,
                "pixel_AP": -0.01,
            }[metric],
            "value": {
                "seg_AUROC_0.05": interaction_p,
                "seg_F1": interaction_f1,
                "pixel_AP": -0.01,
            }[metric],
        }
        for metric in metrics
    }
    payload = {
        "schema": "dscr-density-condition-audit-v1",
        "object": object_name,
        "conditions": {"regular": {"delta_vs_g0": {"float32": regular}}},
        "interaction": {"float32": interaction},
        "query_coverage": {"complete": True},
        "validity": {
            "finite": True,
            "nonconstant": True,
            "map_gt_shape_match": True,
            "float32_tiff_complete": True,
        },
        "precisions_evaluated": ["float16", "float32"],
    }
    path = root / f"{object_name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_aggregate_passes_only_preregistered_promising_pattern(tmp_path: Path) -> None:
    paths = [_write_object(tmp_path, name) for name in ("fabric", "rice", "vial")]

    result = aggregate(paths)

    assert result["gate_pass"] is True
    assert result["claim_verdict"] == "KILL_FOR_CLAIM"
    assert result["diagnostic_verdict"] == "PROMISING_DIAGNOSTIC"
    assert result["negative_pauroc_interaction_object_count"] == 3
    assert result["macro"]["interaction"]["seg_AUROC_0.05"] == pytest.approx(-0.04)


def test_aggregate_fails_when_regular_object_is_harmed(tmp_path: Path) -> None:
    paths = [
        _write_object(tmp_path, "fabric", regular_p=-0.04),
        _write_object(tmp_path, "rice"),
        _write_object(tmp_path, "vial"),
    ]

    result = aggregate(paths)

    assert result["gate_checks"]["no_regular_object_harm_beyond_floor"] is False
    assert result["gate_pass"] is False
    assert result["diagnostic_verdict"] == "NO_CONTINUE"


def test_aggregate_rejects_wrong_object_set(tmp_path: Path) -> None:
    paths = [_write_object(tmp_path, name) for name in ("fabric", "rice")]

    with pytest.raises(ValueError, match="expected objects"):
        aggregate(paths)
