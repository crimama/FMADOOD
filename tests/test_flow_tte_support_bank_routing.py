from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/run_flow_tte_support_bank_routing.py"
SPEC = importlib.util.spec_from_file_location("support_bank_routing", PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_route_bank_falls_back_below_frozen_margin() -> None:
    prototypes = np.eye(3, dtype=np.float32)
    decision = MODULE.route_bank(
        np.array([1.0, 0.99, 0.0], dtype=np.float32),
        prototypes,
        ("identity", "exposure_pos", "gamma_pos"),
        confidence_threshold=0.1,
    )
    assert decision.nearest_bank == "identity"
    assert decision.identity_fallback
    assert decision.selected_bank == "identity"


def test_route_bank_keeps_confident_nonidentity_choice() -> None:
    prototypes = np.eye(3, dtype=np.float32)
    decision = MODULE.route_bank(
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        prototypes,
        ("identity", "exposure_pos", "gamma_pos"),
        confidence_threshold=0.1,
    )
    assert decision.nearest_bank == "exposure_pos"
    assert not decision.identity_fallback
    assert decision.selected_bank == "exposure_pos"


def test_leave_one_view_threshold_uses_only_correct_routes() -> None:
    views = {
        "identity": [np.array([1.0, 0.01 * index]) for index in range(4)],
        "shift": [np.array([0.01 * index, 1.0]) for index in range(4)],
    }
    threshold, summary = MODULE.leave_one_view_margin_threshold(
        views,
        ("identity", "shift"),
    )
    assert summary["accuracy"] == 1.0
    assert summary["correct_margin_count"] == 8
    assert threshold > 0.0


def test_leave_one_view_threshold_fails_closed_when_no_route_is_correct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = iter(
        [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ],
    )
    monkeypatch.setattr(MODULE, "cosine_distances", lambda *_: next(calls))
    views = {
        "identity": [np.array([1.0, 0.0]), np.array([1.0, 0.0])],
        "shift": [np.array([0.0, 1.0]), np.array([0.0, 1.0])],
    }

    threshold, summary = MODULE.leave_one_view_margin_threshold(
        views,
        ("identity", "shift"),
    )

    assert threshold == np.finfo(np.float64).max
    assert summary["correct"] == 0
    assert summary["correct_margin_count"] == 0
    assert summary["correct_margin_mean"] is None
    assert summary["force_identity_fallback"] is True
