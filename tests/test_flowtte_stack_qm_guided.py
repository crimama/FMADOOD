from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np

from src.flow_tte_phase1_normalization import condition_group_quantile_match_to_regular
from src.flow_tte_phase2_refinement import transform_score


def _load_script_module():
    path = Path(__file__).parents[1] / "scripts" / "analyze_flowtte_stack_qm_guided.py"
    spec = importlib.util.spec_from_file_location("analyze_flowtte_stack_qm_guided", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stack_composes_quantile_match_before_guided_filter() -> None:
    module = _load_script_module()
    yy, xx = np.mgrid[:12, :14].astype(np.float32)
    scores = [
        xx + 0.2 * yy,
        4.0 + 1.7 * xx + 0.1 * yy,
        5.0 + 1.3 * xx + 0.3 * yy,
    ]
    stems = ["tiny_regular", "tiny_overexposed", "tiny_underexposed"]
    full_guidance = np.asarray((xx + yy) / float(xx.max() + yy.max()), dtype=np.float32)
    guidance_half = cv2.resize(full_guidance, (7, 6), interpolation=cv2.INTER_AREA)
    guidances = [np.asarray(guidance_half, dtype=np.float32) for _ in scores]

    actual = module.compose_quantile_match_then_guided(scores, stems, guidances)
    matched, _ = condition_group_quantile_match_to_regular(scores, stems)
    expected = [
        transform_score(score, guidance, "guided_r8_eps1e-2")
        for score, guidance in zip(matched, guidances)
    ]
    reverse_guided = [
        transform_score(score, guidance, "guided_r8_eps1e-2")
        for score, guidance in zip(scores, guidances)
    ]
    reversed_order, _ = condition_group_quantile_match_to_regular(reverse_guided, stems)

    for observed, wanted in zip(actual, expected):
        np.testing.assert_array_equal(observed, wanted)
    assert any(
        not np.allclose(observed, reversed_value, rtol=0.0, atol=1e-6)
        for observed, reversed_value in zip(actual, reversed_order)
    )
