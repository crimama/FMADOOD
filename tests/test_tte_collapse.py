from __future__ import annotations

import numpy as np
import pytest
import torch

from flow_tte.config import ScoreConfig
from flow_tte.scoring import ScoreCalibration
from flow_tte.tte_collapse import (
    ArmSpec,
    ImageRecord,
    TteCollapseError,
    candidate_mask,
    run_arm,
    stream_order,
)
from flow_tte.tte_collapse_metrics import best_f1, compute_arm_metrics, partial_auroc


def _record(
    name: str,
    latents: np.ndarray,
    nll: np.ndarray,
    defect: np.ndarray,
    condition: str = "regular",
) -> ImageRecord:
    return ImageRecord(
        relative_path=name,
        condition=condition,
        is_good=not bool(defect.any()),
        latents=torch.as_tensor(latents, dtype=torch.float32),
        nll=torch.as_tensor(nll, dtype=torch.float32),
        defect=defect,
    )


def _fixture(n_images: int = 6, patches: int = 8, dim: int = 4):
    rng = np.random.default_rng(0)
    support = torch.as_tensor(rng.normal(size=(32, dim)).astype(np.float32))
    records = []
    for index in range(n_images):
        latents = rng.normal(size=(patches, dim)).astype(np.float32)
        nll = rng.normal(size=patches).astype(np.float32)
        defect = np.zeros(patches, dtype=bool)
        condition = "regular" if index % 2 == 0 else "overexposed"
        if index % 3 == 0:
            defect[:2] = True
            latents[:2] += 6.0
        records.append(_record(f"img{index}", latents, nll, defect, condition))
    return support, tuple(records)


def test_arm_spec_rejects_static_expansion() -> None:
    with pytest.raises(TteCollapseError):
        ArmSpec("bad", 1.25, protocol="static")


def test_arm_spec_rejects_sub_unit_budget() -> None:
    with pytest.raises(TteCollapseError):
        ArmSpec("bad", 0.5)


def test_image_record_rejects_labelled_good_image() -> None:
    with pytest.raises(TteCollapseError):
        ImageRecord(
            relative_path="x",
            condition="regular",
            is_good=True,
            latents=torch.zeros(3, 2),
            nll=torch.zeros(3),
            defect=np.array([True, False, False]),
        )


def test_stream_order_seed_zero_is_frozen_order() -> None:
    assert stream_order(5, 0) == (0, 1, 2, 3, 4)
    assert sorted(stream_order(5, 101)) == [0, 1, 2, 3, 4]
    assert stream_order(5, 101) != stream_order(5, 202)


def test_candidate_policies_narrow_the_gate() -> None:
    latents = np.zeros((4, 2), dtype=np.float32)
    nll = np.array([0.0, 0.0, 9.0, 0.0], dtype=np.float32)
    defect = np.array([True, False, False, False])
    record = _record("img", latents, nll, defect)
    assert int(candidate_mask(record, 1.0, "all").sum()) == 3
    assert int(candidate_mask(record, 1.0, "gt_normal").sum()) == 2
    assert int(candidate_mask(record, 1.0, "good_images").sum()) == 0


def test_static_arm_absorbs_nothing() -> None:
    support, records = _fixture()
    config = ScoreConfig(density_weight=0.0, query_chunk_size=64)
    calibration = ScoreCalibration.fit(support, config, None)
    result = run_arm(ArmSpec("A0", 1.0, protocol="static"), support, records, 10.0, calibration, config)
    assert result.absorbed_rows == 0
    assert result.contaminated_rows == 0
    assert result.self_hit_rate == 0.0
    assert len(result.scores) == len(records)


def test_oracle_clean_arm_removes_contamination() -> None:
    # A single streamed image fills every open slot in arrival order, so the
    # contamination count is deterministic rather than reservoir-sampled.
    support, records = _fixture()
    single = records[:1]
    assert bool(single[0].defect.any())
    config = ScoreConfig(density_weight=0.0, query_chunk_size=64)
    calibration = ScoreCalibration.fit(support, config, None)
    dirty = run_arm(ArmSpec("A1", 1.25), support, single, 10.0, calibration, config)
    clean = run_arm(
        ArmSpec("A2", 1.25, candidate_policy="gt_normal"), support, single, 10.0, calibration, config,
    )
    assert dirty.absorbed_rows == int(single[0].latents.shape[0])
    assert dirty.contaminated_rows == int(single[0].defect.sum())
    assert clean.contaminated_rows == 0


def test_two_pass_scores_every_image_against_the_same_bank() -> None:
    support, records = _fixture()
    config = ScoreConfig(density_weight=0.0, query_chunk_size=64)
    calibration = ScoreCalibration.fit(support, config, None)
    online = run_arm(ArmSpec("A1", 1.25), support, records, 10.0, calibration, config)
    two_pass = run_arm(ArmSpec("A4", 1.25, protocol="two_pass"), support, records, 10.0, calibration, config)
    assert online.absorbed_rows == two_pass.absorbed_rows
    # The first streamed image sees an empty buffer online but the full bank in two-pass.
    assert not np.allclose(online.scores[0], two_pass.scores[0])


def test_recalibration_changes_the_score_scale() -> None:
    support, records = _fixture()
    config = ScoreConfig(density_weight=0.0, query_chunk_size=64)
    calibration = ScoreCalibration.fit(support, config, None)
    plain = run_arm(ArmSpec("A4", 1.25, protocol="two_pass"), support, records, 10.0, calibration, config)
    recal = run_arm(
        ArmSpec("A5", 1.25, protocol="two_pass", recalibrate=True), support, records, 10.0, calibration, config,
    )
    assert plain.calibration_std != recal.calibration_std


def test_partial_auroc_matches_sklearn_contract() -> None:
    labels = np.array([True, False, True, False, False, False])
    scores = np.array([9.0, 0.1, 8.0, 0.2, 0.3, 0.4], dtype=np.float32)
    assert 0.5 <= partial_auroc(labels, scores) <= 1.0


def test_partial_auroc_agrees_with_sklearn_when_available() -> None:
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(7)
    labels = rng.random(4000) < 0.03
    scores = rng.normal(size=4000).astype(np.float32) + labels * 1.5
    expected = float(sklearn_metrics.roc_auc_score(labels, scores, max_fpr=0.05))
    assert partial_auroc(labels, scores) == pytest.approx(expected, abs=1e-9)


def test_best_f1_agrees_with_sklearn_when_available() -> None:
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(11)
    labels = rng.random(3000) < 0.05
    scores = rng.normal(size=3000).astype(np.float32) + labels * 1.2
    precision, recall, _ = sklearn_metrics.precision_recall_curve(labels, scores)
    denominator = precision + recall
    valid = denominator > 0
    expected = float((2.0 * precision[valid] * recall[valid] / denominator[valid]).max())
    assert best_f1(labels, scores) == pytest.approx(expected, abs=1e-9)


def test_arm_metrics_report_condition_and_drift_terms() -> None:
    support, records = _fixture()
    config = ScoreConfig(density_weight=0.0, query_chunk_size=64)
    calibration = ScoreCalibration.fit(support, config, None)
    result = run_arm(ArmSpec("A0", 1.0, protocol="static"), support, records, 10.0, calibration, config)
    metrics = compute_arm_metrics(
        result.scores,
        [record.defect for record in records],
        [record.condition for record in records],
        [record.is_good for record in records],
        stream_order(len(records), 0),
    )
    assert np.isfinite(metrics.patch_pauroc_005)
    assert np.isfinite(metrics.bad_good_separation)
    assert metrics.defect_score_mean > metrics.normal_score_mean
