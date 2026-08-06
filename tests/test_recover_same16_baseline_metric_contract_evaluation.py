from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "recover_same16_baseline_metric_contract_evaluation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("same16_recovery", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_target_contract_covers_only_stale_evaluator_branches() -> None:
    module = load_module()
    assert tuple(module.TARGETS) == ("basic", "a6", "a7", "a6_l17", "l17")
    assert "superad" not in module.TARGETS
    assert "rn_fmlk" not in module.TARGETS
    assert set(module.TARGETS) == {"basic", "a6", "a7", "a6_l17", "l17"}


def test_validate_payload_accepts_complete_raw_metric_contract() -> None:
    module = load_module()
    module.validate_payload(
        {
            "can": {
                "seg_AUROC_0.05": 0.75,
                "seg_AUPRO_0.05": 0.62,
                "seg_F1_raw": 0.4,
                "seg_F1": 0.4,
                "coverage": {
                    "prediction_map_count": 10,
                    "ground_truth_map_count": 10,
                    "shape_match_count": 10,
                    "non_finite_prediction_count": 0,
                },
            },
        },
        "can",
    )
