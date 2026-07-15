"""Focused synthetic tests for FlowTTE Phase 1 normalization."""

from __future__ import annotations

import numpy as np
import pytest

from src.flow_tte_gap_decomposition import oracle_f1
from src.flow_tte_phase1_normalization import (
    OBJECTS,
    SUPPLEMENTARY_VARIANTS,
    analyze_supplementary_run,
    apply_sigma_floor,
    condition_affine_to_regular,
    condition_group_quantile_match_to_regular,
    condition_group_normalize,
    condition_tail_affine_to_regular,
    cross_fit_fixed_f1,
    evaluate_variant,
    large_defect_f1,
    normalize_per_image,
    regular_condition_identity_check,
    transform_variant,
)


def _record(stem: str, score: np.ndarray, gt: np.ndarray, split: str) -> dict[str, object]:
    return {"stem": stem, "score": score.astype(np.float32), "gt": gt.astype(bool), "split": split}


def test_identity_reproduces_plain_float16_oracle() -> None:
    records = [
        _record("a_regular", np.array([[0.0, 0.2], [0.1, 0.3]]), np.zeros((2, 2)), "good"),
        _record("b_regular", np.array([[0.2, 0.8], [0.3, 0.9]]), np.array([[0, 1], [0, 1]]), "bad"),
    ]
    scores, clamps = transform_variant(records, "identity")
    expected = oracle_f1(
        np.concatenate([record["gt"].ravel() for record in records]),
        np.concatenate([record["score"].ravel() for record in records]),
        cast_float16=True,
    )
    observed = oracle_f1(
        np.concatenate([record["gt"].ravel() for record in records]),
        np.concatenate([score.ravel() for score in scores]),
        cast_float16=True,
    )
    assert observed == expected
    assert clamps["count"] == 0


def test_per_image_normalization_lifts_pooled_f1_under_image_drift() -> None:
    scores = [
        np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
        np.array([[100.0, 100.0, 101.0, 101.0]], dtype=np.float32),
    ]
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.uint8)
    identity = oracle_f1(labels, np.concatenate(scores, axis=None), cast_float16=True)["f1"]
    normalized, _ = normalize_per_image(scores, "mean_std")
    shifted = oracle_f1(labels, np.concatenate(normalized, axis=None), cast_float16=True)["f1"]
    assert shifted == 1.0
    assert shifted > identity


def test_condition_grouping_uses_filename_condition_and_pooled_group_stats() -> None:
    scores = [
        np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32),
        np.array([[10.0, 11.0, 12.0, 13.0]], dtype=np.float32),
        np.array([[100.0, 101.0, 102.0, 103.0]], dtype=np.float32),
        np.array([[110.0, 111.0, 112.0, 113.0]], dtype=np.float32),
    ]
    stems = ["x_regular", "y_regular", "z_shift_1", "w_shift_1"]
    transformed, clamps = condition_group_normalize(scores, stems, quantile=0.7)
    assert np.allclose(transformed[1] - transformed[0], 10.0 / 1.4826, atol=1e-3)
    assert np.allclose(transformed[2], transformed[0], atol=1e-6)
    assert np.allclose(transformed[3], transformed[1], atol=1e-6)
    assert clamps["total"] == 2


def test_condition_affine_leaves_regular_untouched_and_matches_shift() -> None:
    scores = [
        np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32),
        np.array([[100.0, 101.0, 102.0, 103.0]], dtype=np.float32),
    ]
    transformed, _ = condition_affine_to_regular(scores, ["x_regular", "y_shift_1"])
    assert np.array_equal(transformed[0], scores[0])
    assert np.allclose(transformed[1], scores[0], atol=1e-6)


def test_sigma_floor_clamps_small_image_scale() -> None:
    scales, metadata = apply_sigma_floor(np.array([0.01, 1.0, 1.0, 1.0]))
    assert metadata["population_median_sigma"] == 1.0
    assert metadata["sigma_floor"] == 0.25
    assert metadata["count"] == 1
    assert scales[0] == 0.25


def test_evaluate_variant_has_expected_metric_shape() -> None:
    records = []
    for index in range(4):
        records.append(
            _record(
                f"good_{index}_regular",
                np.array([[0.0, 0.1], [0.0, 0.1]]),
                np.zeros((2, 2)),
                "good",
            ),
        )
    records.append(
        _record("bad_0_regular", np.array([[0.0, 1.0], [0.0, 1.0]]), np.array([[0, 1], [0, 1]]), "bad"),
    )
    metrics = evaluate_variant(records, [record["score"] for record in records])
    assert metrics["pooled_oracle_f1"] == 1.0
    assert metrics["cross_fit_fixed_threshold"]["label"] == "transductive_diagnostic"


def test_cross_fit_thresholds_exclude_each_good_fold() -> None:
    records = [
        _record(f"g{index}_regular", np.full((1, 2), index), np.zeros((1, 2)), "good")
        for index in range(4)
    ]
    records.append(
        _record("b0_regular", np.array([[0.0, 4.0]]), np.array([[0, 1]]), "bad"),
    )
    result = cross_fit_fixed_f1(records, [record["score"] for record in records])
    assert np.allclose(result["thresholds"], [2.998, 2.998, 2.999, 1.999], atol=2e-3)
    assert result["f1"] > 0.0


def test_large_defect_subset_uses_gt_area_p90_and_pooled_threshold() -> None:
    records, scores = [], []
    for index, area in enumerate(range(1, 11)):
        gt = np.zeros((1, 10), dtype=bool)
        gt[0, :area] = True
        score = gt.astype(np.float32)
        records.append(_record(f"b{index}_regular", score, gt, "bad"))
        scores.append(score)
    result = large_defect_f1(records, scores, threshold=0.5)
    assert result["n_images"] == 1
    assert result["stems"] == ["b9_regular"]
    assert result["f1"] == 1.0


def test_tail_affine_recovers_pooled_f1_when_median_affine_cannot() -> None:
    bulk = np.linspace(0.0, 1.0, 9800, dtype=np.float32)
    normal_tail = np.linspace(2.0, 3.0, 190, dtype=np.float32)
    anomaly_tail = np.linspace(4.0, 5.0, 10, dtype=np.float32)
    regular = np.concatenate((bulk, normal_tail, anomaly_tail))
    shifted = regular.copy()
    shifted[-200:] = 2.0 * shifted[-200:] + 10.0
    labels = np.zeros(regular.size, dtype=np.uint8)
    labels[-10:] = 1
    scores = [regular, shifted]
    stems = ["sample_regular", "sample_shift_1"]

    assert np.median(regular) == np.median(shifted)
    assert np.all(np.quantile(shifted, (0.99, 0.999)) > np.quantile(regular, (0.99, 0.999)))
    within_condition = oracle_f1(labels, regular, cast_float16=True)["f1"]
    identity_pooled = oracle_f1(
        np.tile(labels, 2), np.concatenate(scores), cast_float16=True
    )["f1"]

    tail_matched, diagnostics = condition_tail_affine_to_regular(scores, stems)
    tail_f1 = oracle_f1(
        np.tile(labels, 2), np.concatenate(tail_matched), cast_float16=True
    )["f1"]
    median_matched, _ = condition_affine_to_regular(scores, stems)
    median_f1 = oracle_f1(
        np.tile(labels, 2), np.concatenate(median_matched), cast_float16=True
    )["f1"]

    assert within_condition == 1.0
    assert identity_pooled < within_condition
    assert tail_f1 == within_condition
    assert median_f1 < within_condition
    assert diagnostics["count"] == 0
    assert diagnostics["affine"]["shift_1"]["slope"] == pytest.approx(0.5, rel=1e-5)


def test_condition_quantile_match_is_monotone_and_leaves_regular_exact() -> None:
    regular = np.linspace(-1.0, 3.0, 6000, dtype=np.float32)
    shifted = np.concatenate(
        (
            np.linspace(-1.0, 1.0, 5900, dtype=np.float32),
            np.linspace(10.0, 30.0, 100, dtype=np.float32),
        )
    )
    transformed, diagnostics = condition_group_quantile_match_to_regular(
        [regular, shifted], ["sample_regular", "sample_overexposed"]
    )
    order = np.argsort(shifted, kind="stable")

    assert np.array_equal(transformed[0], regular)
    assert np.all(np.diff(transformed[1][order]) >= 0.0)
    assert diagnostics["requested_knots"] == 4096
    assert all(diagnostics["monotone_by_condition"].values())


def test_tail_affine_degenerate_group_falls_back_to_identity() -> None:
    regular = np.linspace(0.0, 1.0, 1000, dtype=np.float32)
    degenerate = np.zeros(1000, dtype=np.float32)
    transformed, diagnostics = condition_tail_affine_to_regular(
        [regular, degenerate], ["sample_regular", "sample_underexposed"]
    )

    assert np.array_equal(transformed[1], degenerate)
    assert diagnostics["count"] == 1
    assert diagnostics["degenerate_conditions"] == ["underexposed"]


def test_supplementary_runner_refuses_output_collisions_before_loading(tmp_path) -> None:
    collision = tmp_path / f"{SUPPLEMENTARY_VARIANTS[0]}.json"
    collision.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        analyze_supplementary_run(
            tmp_path / "unused-anchor",
            tmp_path / "unused-data",
            OBJECTS,
            tmp_path,
        )
    assert collision.read_text(encoding="utf-8") == "existing\n"


def test_regular_condition_float16_oracle_check_is_bitwise_identical() -> None:
    records = [
        _record("a_regular", np.array([[0.0, 0.5, 1.0]]), np.array([[0, 0, 1]]), "bad"),
        _record("b_shift_2", np.array([[10.0, 20.0, 30.0]]), np.array([[0, 0, 1]]), "bad"),
    ]
    transformed, _ = transform_variant(
        records, "condition_group_quantile_match_to_regular_q4096"
    )
    result = regular_condition_identity_check(records, transformed)

    assert result["pass"] is True
    assert result["float16_score_arrays_equal"] is True
    assert result["identity_f1_float64_bits"] == result["variant_f1_float64_bits"]
