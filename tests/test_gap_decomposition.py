from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import precision_recall_curve

from src.flow_tte_gap_decomposition import (
    boundary_tolerant_f1,
    component_recall,
    oracle_f1,
    parse_condition,
)


@pytest.mark.parametrize("cast_float16", [False, True])
def test_oracle_f1_matches_sklearn_pr_curve(cast_float16: bool) -> None:
    labels = np.array([0, 0, 1, 1, 0, 1], dtype=np.uint8)
    scores = np.array([0.10001, 0.20002, 0.80008, 0.70007, 0.60006, 0.90009], dtype=np.float32)
    evaluated = scores.astype(np.float16 if cast_float16 else np.float32)
    precision, recall, thresholds = precision_recall_curve(labels, evaluated)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * precision * recall / (precision + recall)
    finite = np.isfinite(f1)
    expected_f1 = float(np.max(f1[finite]))
    expected_threshold = float(thresholds[np.argmax(f1[finite])])

    actual = oracle_f1(labels, scores, cast_float16)

    assert actual["f1"] == pytest.approx(expected_f1)
    assert actual["threshold"] == pytest.approx(expected_threshold)


def test_float16_oracle_path_quantizes_threshold() -> None:
    labels = np.array([0, 1, 0, 1], dtype=np.uint8)
    scores = np.array([0.10001, 0.10002, 0.10003, 0.10004], dtype=np.float32)
    result = oracle_f1(labels, scores, cast_float16=True)
    assert result["threshold"] == float(np.float16(result["threshold"]))


def test_boundary_tolerance_on_shifted_known_square() -> None:
    gt = np.zeros((15, 15), dtype=bool)
    gt[4:9, 4:9] = True
    prediction = np.zeros_like(gt)
    prediction[4:9, 6:11] = True

    assert boundary_tolerant_f1(prediction, gt, 0) == pytest.approx(15 / 25)
    assert boundary_tolerant_f1(prediction, gt, 2) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("001_regular", "regular"),
        ("021_overexposed", "overexposed"),
        ("010_underexposed", "underexposed"),
        ("100_shift_3", "shift_3"),
    ],
)
def test_condition_parsing(stem: str, expected: str) -> None:
    assert parse_condition(stem) == expected


def test_condition_parsing_rejects_unlabeled_stem() -> None:
    with pytest.raises(ValueError, match="unrecognized lighting condition"):
        parse_condition("001")


def test_component_recall_counts_gt_components_hit_once() -> None:
    gt = np.zeros((8, 8), dtype=bool)
    gt[1:3, 1:3] = True
    gt[5:7, 5:7] = True
    prediction = np.zeros_like(gt)
    prediction[2, 2] = True

    result = component_recall(prediction, gt)

    assert result == {"hit": 1, "total": 2, "recall": 0.5}
