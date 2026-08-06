from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
PATH = SCRIPTS / "evaluate_flow_tte_guided_fallback.py"
SPEC = importlib.util.spec_from_file_location("guided_fallback", PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def record(stem: str, score: np.ndarray) -> dict[str, object]:
    return {
        "split": "good",
        "stem": stem,
        "condition": "regular",
        "score": score.astype(np.float32),
        "gt": np.zeros(score.shape, dtype=np.bool_),
    }


def test_fallback_selects_guided_only_when_all_gate_terms_pass() -> None:
    score = np.arange(100, dtype=np.float32).reshape(10, 10)
    raw = [record("accepted", score), record("rejected", score)]
    guided = [record("accepted", score), record("rejected", np.flip(score, axis=1))]
    selected, agreements = MODULE.fallback_records(raw, guided)
    assert agreements[0]["frozen_gate_accept"] is True
    assert agreements[1]["frozen_gate_accept"] is False
    assert np.array_equal(selected[0]["score"], guided[0]["score"])
    assert np.array_equal(selected[1]["score"], raw[1]["score"])


def test_object_harm_uses_either_frozen_limit() -> None:
    assert MODULE.is_object_harm(-0.021, 0.0)
    assert MODULE.is_object_harm(0.0, -0.031)
    assert not MODULE.is_object_harm(-0.02, -0.03)


def test_fallback_reuses_frozen_stage1_agreements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = [record("accepted", np.array([[1.0, 2.0]], dtype=np.float32))]
    guided = [record("accepted", np.array([[9.0, 8.0]], dtype=np.float32))]
    frozen = [{
        "split": "good",
        "stem": "accepted",
        "frozen_gate_accept": True,
    }]

    def fail_recompute(_raw: object, _guided: object) -> NoReturn:
        raise AssertionError("Stage 3 must reuse the exact Stage 1 gate inputs")

    monkeypatch.setattr(MODULE, "guide_agreement", fail_recompute)
    selected, agreements = MODULE.fallback_records(raw, guided, frozen)

    np.testing.assert_array_equal(selected[0]["score"], guided[0]["score"])
    assert agreements == frozen
