from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/evaluate_flow_tte_support_bank_routing.py"
SPEC = importlib.util.spec_from_file_location("support_bank_eval", PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def metrics(
    auroc: float,
    f1: float,
    inflation: float,
    separation: float,
) -> dict[str, float]:
    return {
        "seg_AUROC_0.05": auroc,
        "seg_F1": f1,
        "shifted_good_p95_inflation": inflation,
        "bad_good_p95_separation": separation,
    }


def test_continuation_requires_all_safety_and_shift_gates() -> None:
    identity = metrics(0.8, 0.5, 1.0, 0.4)
    routed = metrics(0.806, 0.5, 0.8, 0.4)
    verdict = MODULE.continuation_gate(
        identity,
        routed,
        has_object_harm=False,
        valid_object_count=8,
        expected_object_count=8,
    )
    assert verdict["continuation"]

    harmed = MODULE.continuation_gate(
        identity,
        routed,
        has_object_harm=True,
        valid_object_count=8,
        expected_object_count=8,
    )
    assert not harmed["continuation"]
    invalid = MODULE.continuation_gate(
        identity,
        routed,
        has_object_harm=False,
        valid_object_count=7,
        expected_object_count=8,
    )
    assert not invalid["continuation"]


def test_exact_harm_boundaries_are_not_harm() -> None:
    assert not MODULE.object_harm(-0.02, -0.03)
    assert MODULE.object_harm(-0.0201, 0.0)
    assert MODULE.object_harm(0.0, -0.0301)


def object_result(offset: float) -> dict[str, object]:
    arms = {}
    for index, arm in enumerate(MODULE.ARMS):
        value = offset + index * 0.01
        arms[arm] = {
            "seg_AUROC_0.05": 0.8 + value,
            "seg_F1": 0.5 + value,
            "coverage": {"pass": True},
            "conditions": {
                "regular": {"bad_good_p95_separation": 0.4 + value},
                "shift1": {
                    "good_p95_inflation_vs_regular": 0.2 - value,
                    "bad_good_p95_separation": 0.4 + value,
                },
            },
        }
    return {
        "arms": arms,
        "routed_vs_identity": {
            "seg_AUROC_0.05": 0.03,
            "seg_F1": 0.03,
            "object_harm": False,
        },
        "routing": {},
    }


def test_aggregate_object_summaries_matches_direct_summary(tmp_path: Path) -> None:
    objects = ("can", "fabric")
    per_object = {
        "can": object_result(0.0),
        "fabric": object_result(0.01),
    }
    for object_name in objects:
        root = tmp_path / "objects" / object_name
        root.mkdir(parents=True)
        (root / "stage2_metrics.json").write_text(
            json.dumps({"per_object": {object_name: per_object[object_name]}}),
            encoding="utf-8",
        )

    aggregate = MODULE.aggregate_object_summaries(tmp_path, objects)

    assert aggregate == MODULE.summarize(per_object, objects)
