import numpy as np
import pytest

from flow_tte.cfntr import (
    PHYSICAL_AXES,
    VALIDATION_AMPLITUDE_SCALE,
    apply_rgb_transform,
    apply_training_axis_interventions,
    central_difference_observation,
    deterministic_random_basis,
    fit_image_level_basis,
    fold_basis_stability,
    image_level_delta_observation,
    make_axis_transform,
    make_compound_transform,
    minimum_subspace_cosine,
    normalize_rows,
    quadrant_fold_ids,
    score_crossfit_residuals,
    score_patchwise_residuals,
    score_raw_residuals,
    subspace_energy_coverage,
    subspace_overlap_stability,
    summarize_map_coverage,
    support_image_fold_ids,
)


def test_frozen_axis_interventions_are_exact_and_deterministic() -> None:
    image = np.array([[[0, 64, 255], [128, 192, 32]]], dtype=np.uint8)
    first = apply_training_axis_interventions(image)
    second = apply_training_axis_interventions(image.astype(np.float32) / 255.0)

    expected_names = tuple(
        make_axis_transform(axis, sign=sign).name for axis in PHYSICAL_AXES for sign in (-1, 1)
    )
    assert tuple(first) == expected_names
    assert len(first) == 8
    for name in first:
        assert first[name].dtype == np.uint8
        np.testing.assert_array_equal(first[name], second[name])
    positive_exposure = apply_rgb_transform(image, make_axis_transform("exposure", sign=1))
    expected = np.rint(np.clip(image / 255.0 * (2.0**0.25), 0.0, 1.0) * 255.0).astype(np.uint8)
    np.testing.assert_array_equal(positive_exposure, expected)


def test_validation_transform_specs_expose_double_amplitude_and_compounds() -> None:
    exposure = make_axis_transform(
        "exposure",
        sign=-1,
        amplitude_scale=VALIDATION_AMPLITUDE_SCALE,
    )
    compound = make_compound_transform(
        "gamma_contrast",
        signs=(1, -1),
    )

    assert exposure.exposure_ev == pytest.approx(-0.50)
    assert compound.gamma_log_step == pytest.approx(0.20)
    assert compound.contrast_log_step == pytest.approx(-0.20)
    assert compound.exposure_ev == 0.0
    assert compound.white_balance_log_step == 0.0


def test_gamma_contrast_and_reciprocal_white_balance_use_frozen_log_coordinates() -> None:
    image = np.array(
        [
            [[51, 102, 153], [76, 127, 178]],
            [[102, 153, 204], [127, 178, 229]],
        ],
        dtype=np.uint8,
    )
    rgb = image.astype(np.float64) / 255.0

    gamma_spec = make_axis_transform("gamma", sign=1)
    gamma_expected = np.rint(np.power(rgb, np.exp(0.10)) * 255.0).astype(np.uint8)
    np.testing.assert_array_equal(apply_rgb_transform(image, gamma_spec), gamma_expected)

    contrast_spec = make_axis_transform("contrast", sign=-1)
    mean = rgb.mean(axis=(0, 1), keepdims=True)
    contrast_expected = np.rint(
        (mean + np.exp(-0.10) * (rgb - mean)) * 255.0,
    ).astype(np.uint8)
    np.testing.assert_array_equal(apply_rgb_transform(image, contrast_spec), contrast_expected)

    balance_spec = make_axis_transform("white_balance", sign=1)
    gains = np.array([np.exp(0.08), 1.0, np.exp(-0.08)]).reshape(1, 1, 3)
    balance_expected = np.rint(np.clip(rgb * gains, 0.0, 1.0) * 255.0).astype(np.uint8)
    np.testing.assert_array_equal(apply_rgb_transform(image, balance_spec), balance_expected)
    assert float(gains[0, 0, 0] * gains[0, 0, 2]) == pytest.approx(1.0)


def test_rgb_intervention_rejects_unregistered_or_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="axis must"):
        make_axis_transform("brightness", sign=1)
    with pytest.raises(ValueError, match="sign must"):
        make_axis_transform("gamma", sign=0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        apply_rgb_transform(
            np.full((2, 2, 3), 1.1, dtype=np.float32),
            make_axis_transform("gamma", sign=1),
        )
    with pytest.raises(ValueError, match="RGB shape"):
        apply_rgb_transform(
            np.zeros((2, 2), dtype=np.uint8),
            make_axis_transform("gamma", sign=1),
        )


def test_row_normalization_and_image_delta_observation() -> None:
    original = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    transformed = np.array([[0.0, 5.0], [2.0, 0.0]], dtype=np.float32)
    normalized = normalize_rows(original)

    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.0, atol=1e-6)
    expected = (normalize_rows(transformed) - normalized).mean(axis=0)
    np.testing.assert_allclose(
        image_level_delta_observation(original, transformed),
        expected,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        central_difference_observation(original, transformed),
        expected,
        atol=1e-7,
    )
    with pytest.raises(ValueError, match="undefined"):
        normalize_rows(np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32))


def test_basis_fit_recovers_image_level_subspace_and_is_deterministic() -> None:
    rng = np.random.default_rng(4)
    truth, _ = np.linalg.qr(rng.normal(size=(8, 4)))
    coefficients = rng.normal(size=(80, 4)) * np.array([4.0, 3.0, 2.0, 1.0])
    observations = coefficients @ truth.T

    first = fit_image_level_basis(observations, rank=4, center=False)
    second = fit_image_level_basis(observations, rank=4, center=False)

    np.testing.assert_allclose(first.T @ first, np.eye(4), atol=2e-6)
    np.testing.assert_array_equal(first, second)
    assert minimum_subspace_cosine(first, truth.astype(np.float32)) > 0.99999


def test_centered_basis_removes_only_image_level_mean() -> None:
    observations = np.array(
        [[10.0, -2.0, 0.0], [11.0, -2.0, 0.0], [9.0, -2.0, 0.0]],
        dtype=np.float32,
    )
    basis = fit_image_level_basis(observations, rank=1, center=True)

    assert abs(float(basis[:, 0] @ np.array([1.0, 0.0, 0.0]))) > 0.9999
    with pytest.raises(ValueError, match="numerical rank"):
        fit_image_level_basis(observations, rank=2, center=True)


def test_random_basis_is_deterministic_orthonormal_and_seeded() -> None:
    first = deterministic_random_basis(feature_dim=12, rank=3, seed=17)
    second = deterministic_random_basis(feature_dim=12, rank=3, seed=17)
    other = deterministic_random_basis(feature_dim=12, rank=3, seed=18)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.T @ first, np.eye(3), atol=2e-6)
    assert not np.allclose(first, other)


def test_subspace_stability_uses_preregistered_average_squared_alignment() -> None:
    full = np.eye(4, 2, dtype=np.float32)
    same = full.copy()
    angle = np.deg2rad(60.0)
    rotated = np.stack(
        [np.array([np.cos(angle), 0.0, np.sin(angle), 0.0]), full[:, 1]],
        axis=1,
    ).astype(np.float32)
    stability = fold_basis_stability(full, [same, rotated])

    assert minimum_subspace_cosine(full, rotated) == pytest.approx(0.5, abs=1e-6)
    assert stability.per_fold == pytest.approx((1.0, 0.625), abs=1e-6)
    assert stability.median == pytest.approx(0.8125, abs=1e-6)
    # The separately exposed worst-angle audit remains distinguishable from G2.
    assert subspace_overlap_stability(full, rotated) == pytest.approx(0.625, abs=1e-6)


def test_preregistered_fold_assignments_are_deterministic() -> None:
    np.testing.assert_array_equal(support_image_fold_ids(8), [0, 1, 2, 3, 0, 1, 2, 3])
    np.testing.assert_array_equal(
        quadrant_fold_ids((5, 7)).reshape(5, 7),
        np.array(
            [
                [0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 1],
                [2, 2, 2, 3, 3, 3, 3],
                [2, 2, 2, 3, 3, 3, 3],
                [2, 2, 2, 3, 3, 3, 3],
            ],
        ),
    )


def test_crossfit_uses_one_training_median_per_held_out_quadrant() -> None:
    folds = quadrant_fold_ids((2, 4))
    basis = np.array([[1.0], [0.0]], dtype=np.float32)
    nuisance = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0], dtype=np.float32)
    residuals = np.stack([nuisance, np.full(8, 2.0, dtype=np.float32)], axis=1)
    result = score_crossfit_residuals(residuals, basis, folds)

    # Held-out fold 0 sees nuisance coordinates {2, 3, 4}: median 3.
    assert result.coefficients[0, 0] == pytest.approx(3.0)
    np.testing.assert_allclose(result.scores[:2], 0.5 * ((1.0 - 3.0) ** 2 + 2.0**2))
    assert result.scores.shape == (8,)


def test_raw_crossfit_and_patchwise_scores_have_distinct_frozen_semantics() -> None:
    folds = quadrant_fold_ids((2, 2))
    basis = np.array([[1.0], [0.0]], dtype=np.float32)
    residuals = np.array(
        [[2.0, 3.0], [2.0, 4.0], [2.0, 5.0], [2.0, 6.0]],
        dtype=np.float32,
    )
    raw = score_raw_residuals(residuals)
    crossfit = score_crossfit_residuals(residuals, basis, folds).scores
    patchwise = score_patchwise_residuals(residuals, basis)

    np.testing.assert_allclose(raw, 0.5 * np.sum(residuals**2, axis=1))
    np.testing.assert_allclose(crossfit, 0.5 * residuals[:, 1] ** 2)
    np.testing.assert_allclose(patchwise, 0.5 * residuals[:, 1] ** 2)


def test_energy_coverage_and_map_coverage_are_auditable() -> None:
    basis = np.array([[1.0], [0.0]], dtype=np.float32)
    assert subspace_energy_coverage(np.array([3.0, 4.0]), basis) == pytest.approx(9.0 / 25.0)

    coverage = summarize_map_coverage(
        [
            np.array([[0.0, 1.0]], dtype=np.float32),
            np.array([[2.0, 2.0]], dtype=np.float32),
            np.array([[0.0, np.nan]], dtype=np.float32),
        ],
    )
    assert coverage.total_maps == 3
    assert coverage.finite_maps == 2
    assert coverage.nonconstant_maps == 1
    assert coverage.valid_maps == 1
    assert coverage.finite_value_fraction == pytest.approx(5.0 / 6.0)
    assert coverage.complete is False


def test_scorers_reject_malformed_or_nonorthonormal_inputs() -> None:
    residuals = np.ones((4, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="orthonormal"):
        score_patchwise_residuals(residuals, np.ones((2, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="each quadrant"):
        score_crossfit_residuals(
            residuals,
            np.array([[1.0], [0.0]], dtype=np.float32),
            np.zeros(4, dtype=int),
        )
    with pytest.raises(ValueError, match="finite"):
        score_raw_residuals(np.array([[1.0, np.nan]], dtype=np.float32))
