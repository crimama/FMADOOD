from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_dscr_density_compact import crossfit_fixed_metrics
from flow_tte_phase3_scorer_suite import NativeMapRecord


def _record(split: str, stem: str, score: np.ndarray, gt: np.ndarray) -> NativeMapRecord:
    return NativeMapRecord(split=split, stem=stem, score=score.astype(np.float16), gt=gt)


def test_crossfit_fixed_metrics_are_image_disjoint_and_finite() -> None:
    good = tuple(
        _record(
            "good",
            f"g{index}",
            np.full((4, 4), index, dtype=np.float32),
            np.zeros((4, 4), dtype=np.bool_),
        )
        for index in range(4)
    )
    bad_score = np.zeros((4, 4), dtype=np.float32)
    bad_score[:2, :2] = 10.0
    bad_gt = np.zeros((4, 4), dtype=np.bool_)
    bad_gt[:2, :2] = True
    metrics = crossfit_fixed_metrics((*good, _record("bad", "b0", bad_score, bad_gt)))
    assert metrics["fixed_thresholds"] == [3.0, 3.0, 3.0, 2.0]
    assert metrics["fixed_raw_f1"] == 1.0 / 3.0
    assert metrics["fixed_normal_fpr"] == 0.25
    assert 0.0 <= metrics["fixed_raw_f1"] <= 1.0
    assert 0.0 <= metrics["fixed_normal_fpr"] <= 1.0
    assert all(np.isfinite(metrics["fixed_thresholds"]))
