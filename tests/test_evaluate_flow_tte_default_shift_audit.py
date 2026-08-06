from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/evaluate_flow_tte_default_shift_audit.py"
SPEC = importlib.util.spec_from_file_location("default_shift_audit", PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def record(split: str, stem: str, score: np.ndarray, gt: np.ndarray) -> dict[str, object]:
    return {
        "split": split,
        "stem": stem,
        "condition": stem.split("_", 1)[1],
        "score": score.astype(np.float32),
        "gt": gt.astype(np.bool_),
    }


def test_continuous_condition_and_morphology_metrics_are_finite() -> None:
    zero = np.zeros((8, 8), dtype=np.bool_)
    defect = zero.copy()
    defect[2:5, 2:5] = True
    records = [
        record("good", "000_regular", np.linspace(0, 0.2, 64).reshape(8, 8), zero),
        record("bad", "001_regular", defect.astype(np.float32), defect),
        record("good", "000_shift_1", np.linspace(0.1, 0.3, 64).reshape(8, 8), zero),
        record("bad", "001_shift_1", defect.astype(np.float32) + 0.1, defect),
    ]
    metrics = MODULE.continuous_metrics(records)
    assert 0.5 <= metrics["seg_AUROC_0.05"] <= 1.0
    assert 0.0 <= metrics["seg_F1"] <= 1.0
    conditions = MODULE.condition_rows(records)
    assert conditions["shift_1"]["good_p95_inflation_vs_regular"] > 0.0
    morph = MODULE.morphology_metrics(records, metrics["best_threshold"])
    assert 0.0 <= morph["seg_F1"] <= 1.0
    morph_conditions = MODULE.morphology_condition_rows(records, metrics["best_threshold"])
    assert "morphology_precision" in morph_conditions["regular"]


def test_guide_agreement_applies_the_frozen_gate() -> None:
    score = np.arange(64, dtype=np.float32).reshape(8, 8)
    zero = np.zeros((8, 8), dtype=np.bool_)
    raw = [record("good", "000_regular", score, zero)]
    accepted = MODULE.guide_agreement(raw, [record("good", "000_regular", score, zero)])
    assert accepted[0]["frozen_gate_accept"] is True

    flipped = np.flip(score, axis=1).copy()
    rejected = MODULE.guide_agreement(raw, [record("good", "000_regular", flipped, zero)])
    assert rejected[0]["frozen_gate_accept"] is False


def test_average_ranks_handles_exact_ties() -> None:
    values = np.array([4.0, 1.0, 1.0, 9.0, 4.0], dtype=np.float32)
    np.testing.assert_array_equal(MODULE.average_ranks(values), [2.5, 0.5, 0.5, 4.0, 2.5])


def test_histogram_metrics_match_sklearn_on_float16_scores() -> None:
    generator = np.random.default_rng(12)
    labels = generator.integers(0, 2, size=20_000, dtype=np.uint8)
    scores = generator.normal(size=labels.size).astype(np.float16)
    observed = MODULE.binary_float16_metrics(labels, scores)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2.0 * precision * recall / (precision + recall)
    best = int(np.nanargmax(f1[:-1]))
    assert observed["seg_AUROC_0.05"] == pytest.approx(
        roc_auc_score(labels, scores, max_fpr=0.05),
        abs=1e-12,
    )
    assert observed["seg_F1"] == pytest.approx(f1[best], abs=1e-12)
    assert observed["best_threshold"] == float(thresholds[best])


def test_parity_rejects_mismatched_image_identities() -> None:
    score = np.arange(4, dtype=np.float32).reshape(2, 2)
    zero = np.zeros((2, 2), dtype=np.bool_)
    first = [record("good", "000_regular", score, zero)]
    second = [record("good", "001_regular", score, zero)]
    with pytest.raises(ValueError, match="identity mismatch"):
        MODULE.verify_parity({"a": first, "b": second})
