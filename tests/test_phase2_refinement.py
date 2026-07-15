from __future__ import annotations

import cv2
import numpy as np
import pytest
from scipy import ndimage

from src.flow_tte_gap_decomposition import oracle_f1
from src.flow_tte_phase2_refinement import (
    box_mean,
    evaluate_variant,
    fast_guided_filter,
    transform_score,
)


def test_guided_step_edge_sharpens_soft_square_and_raises_t0_f1() -> None:
    gt = np.zeros((33, 33), dtype=np.bool_)
    gt[8:24, 8:24] = True
    guidance = gt.astype(np.float32)
    soft = ndimage.gaussian_filter(guidance, sigma=3.0, mode="reflect").astype(np.float32)

    refined = fast_guided_filter(guidance, soft, radius=4, eps=1e-2)
    raw_f1 = oracle_f1(gt, soft, cast_float16=True)["f1"]
    refined_f1 = oracle_f1(gt, refined, cast_float16=True)["f1"]

    assert refined_f1 > raw_f1 + 0.015
    assert refined_f1 == pytest.approx(1.0)


def test_identity_transform_preserves_float16_evaluator_parity() -> None:
    scores = [
        np.array([[0.10001, 0.10002], [0.6, 0.7]], dtype=np.float32),
        np.array([[0.10003, 0.10004], [0.8, 0.9]], dtype=np.float32),
    ]
    labels = [
        np.zeros((2, 2), dtype=np.bool_),
        np.array([[False, True], [False, True]]),
    ]
    transformed = [transform_score(score, None, "identity") for score in scores]

    assert all(np.array_equal(before, after) for before, after in zip(scores, transformed))
    expected = oracle_f1(
        np.concatenate([x.ravel() for x in labels]),
        np.concatenate(scores),
        cast_float16=True,
    )
    actual = oracle_f1(
        np.concatenate([x.ravel() for x in labels]),
        np.concatenate(transformed),
        cast_float16=True,
    )
    assert actual == expected

    records = [
        {"split": "good", "gt": labels[0]},
        {"split": "bad", "gt": labels[1]},
    ]
    metrics = evaluate_variant(records, transformed)
    assert metrics["pooled_oracle_f1"] == expected["f1"]
    assert metrics["pooled_oracle_threshold"] == expected["threshold"]
    assert metrics["boundary_tolerant_f1_native_px"]["0"]["f1"] == expected["f1"]


def test_constant_guidance_reduces_to_double_box_smoothing() -> None:
    rng = np.random.default_rng(2)
    source = rng.normal(size=(31, 29)).astype(np.float32)
    guidance = np.full_like(source, 0.3)
    radius = 3

    actual = fast_guided_filter(guidance, source, radius=radius, eps=1e-2)
    expected = box_mean(box_mean(source, radius), radius)
    crop = np.s_[2 * radius : -2 * radius, 2 * radius : -2 * radius]

    assert actual.dtype == np.float32
    assert np.all(np.isfinite(actual))
    assert actual[crop] == pytest.approx(expected[crop], abs=2e-5)


def test_guidance_aligned_edge_is_preserved() -> None:
    guidance = np.zeros((65, 65), dtype=np.float32)
    guidance[:, 32:] = 1.0
    source = guidance.copy()
    constant = np.full_like(guidance, 0.5)

    aligned = fast_guided_filter(guidance, source, radius=4, eps=1e-2)
    control = fast_guided_filter(constant, source, radius=4, eps=1e-2)
    aligned_jump = float(np.mean(aligned[:, 32]) - np.mean(aligned[:, 31]))
    control_jump = float(np.mean(control[:, 32]) - np.mean(control[:, 31]))

    assert aligned_jump > 0.8
    assert aligned_jump > 4.0 * control_jump
    assert float(np.mean(aligned[:, 31])) < 0.05
    assert float(np.mean(aligned[:, 32])) > 0.95


def test_gaussian_half_resolution_transform_returns_native_float32() -> None:
    score = np.arange(35, dtype=np.float32).reshape(5, 7)
    guidance = cv2.resize(score, (4, 2), interpolation=cv2.INTER_AREA)

    output = transform_score(score, guidance, "gaussian_blur_sigma8")

    assert output.shape == score.shape
    assert output.dtype == np.float32
