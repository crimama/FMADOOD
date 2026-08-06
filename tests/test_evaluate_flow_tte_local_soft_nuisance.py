from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/evaluate_flow_tte_local_soft_nuisance.py"
SPEC = importlib.util.spec_from_file_location("local_soft_eval", PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_stage4_harm_boundaries_are_strict() -> None:
    assert not MODULE.object_harm(-0.02, -0.03)
    assert MODULE.object_harm(-0.0201, 0.0)
    assert MODULE.object_harm(0.0, -0.0301)
