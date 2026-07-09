from __future__ import annotations

# pyright: reportMissingImports=false
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_ROOT = _REPO_ROOT / "scripts"
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

from flow_tte_score_field import (  # noqa: E402
    ScoreFieldConfig,
    apply_score_field_transform,
    fit_score_field_stats,
)
from flow_tte_score_priors import rgb_foreground_proxy  # noqa: E402

from scripts.flow_tte_components import summarize_components  # noqa: E402


def test_support_position_center_removes_repeating_position_bias() -> None:
    support_scores = (
        np.asarray([[3.0, 0.0], [3.0, 0.0]], dtype=np.float32),
        np.asarray([[5.0, 1.0], [5.0, 1.0]], dtype=np.float32),
    )
    support_features = (
        np.ones((2, 2, 1), dtype=np.float32),
        np.ones((2, 2, 1), dtype=np.float32),
    )
    config = ScoreFieldConfig(
        calibration_mode="support_position_center",
        calibration_alpha=1.0,
        foreground_mode="none",
    )

    stats = fit_score_field_stats(support_scores, support_features, config)
    transformed = apply_score_field_transform(
        np.asarray([[4.0, 9.0], [4.0, 9.0]], dtype=np.float32),
        stats,
        config,
    )

    assert np.allclose(transformed[:, 0], 0.0)
    assert np.all(transformed[:, 1] > 8.0)


def test_support_feature_energy_prior_suppresses_background_scores() -> None:
    support_scores = (np.zeros((2, 2), dtype=np.float32),)
    support_features = (
        np.asarray([[[0.0], [0.0]], [[4.0], [4.0]]], dtype=np.float32),
    )
    config = ScoreFieldConfig(
        calibration_mode="none",
        foreground_mode="support_feature_energy",
        foreground_quantile=0.5,
        background_multiplier=0.25,
        foreground_smooth_kernel=1,
    )

    stats = fit_score_field_stats(support_scores, support_features, config)
    transformed = apply_score_field_transform(np.ones((2, 2), dtype=np.float32), stats, config)

    assert np.allclose(transformed[0], 0.25)
    assert np.allclose(transformed[1], 1.0)


def test_support_rgb_contrast_prior_suppresses_border_like_scores() -> None:
    support_scores = (np.zeros((3, 3), dtype=np.float32),)
    support_features = (np.ones((3, 3, 1), dtype=np.float32),)
    support_object = (
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    config = ScoreFieldConfig(
        calibration_mode="none",
        foreground_mode="support_rgb_contrast",
        foreground_quantile=0.9,
        background_multiplier=0.2,
        foreground_smooth_kernel=1,
    )

    stats = fit_score_field_stats(
        support_scores,
        support_features,
        config,
        support_object_fields=support_object,
    )
    transformed = apply_score_field_transform(np.ones((3, 3), dtype=np.float32), stats, config)

    assert transformed[1, 1] == 1.0
    assert np.allclose(transformed[0], 0.2)


def test_support_score_reliability_downweights_normal_high_positions() -> None:
    support_scores = (
        np.asarray([[5.0, 0.0], [5.0, 0.0]], dtype=np.float32),
        np.asarray([[6.0, 0.0], [6.0, 0.0]], dtype=np.float32),
    )
    support_features = (
        np.ones((2, 2, 1), dtype=np.float32),
        np.ones((2, 2, 1), dtype=np.float32),
    )
    config = ScoreFieldConfig(
        calibration_mode="support_score_reliability",
        calibration_alpha=0.5,
        foreground_mode="none",
        support_score_quantile=0.5,
    )

    stats = fit_score_field_stats(support_scores, support_features, config)
    transformed = apply_score_field_transform(np.ones((2, 2), dtype=np.float32), stats, config)

    assert np.allclose(transformed[:, 0], 0.5)
    assert np.allclose(transformed[:, 1], 1.0)


def test_rgb_foreground_proxy_uses_border_background_contrast() -> None:
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    image[2, 2] = np.asarray([255, 255, 255], dtype=np.uint8)

    prior = rgb_foreground_proxy(image, (5, 5))

    assert prior[2, 2] == 1.0
    assert np.allclose(prior[0], 0.0)


def test_component_summary_counts_fragmented_high_score_regions() -> None:
    mask = np.asarray(
        [
            [True, False, True],
            [False, False, False],
            [True, True, False],
        ],
        dtype=np.bool_,
    )

    summary = summarize_components(mask)

    assert summary.component_count == 3
    assert summary.positive_area == 4 / 9
    assert summary.largest_component_share == 0.5
