# /// script
# requires-python = ">=3.8"
# dependencies = ["numpy", "torch", "typing-extensions"]
# ///
# pyright: reportMissingImports=false
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence, Tuple

import numpy as np
import numpy.typing as npt

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (_REPO_ROOT, _REPO_ROOT / "src"):
    _path_text = str(_path)
    if _path_text not in sys.path:
        sys.path.insert(0, _path_text)

from flow_tte_score_priors import (  # noqa: E402
    ScoreFieldConfigError,
    box_filter,
    feature_priors,
    mean_prior,
    object_priors,
    support_score_reliability,
    thresholded_prior,
)

from flow_tte.tensors import to_numpy  # noqa: E402

if TYPE_CHECKING:
    from flow_tte.pipeline import FlowTTE

FloatArray = npt.NDArray[np.float32]
CalibrationMode = Literal[
    "none",
    "support_position_center",
    "support_position_zscore",
    "support_score_reliability",
]
ForegroundMode = Literal[
    "none",
    "support_feature_energy",
    "support_rgb_contrast",
    "support_rgb_feature_product",
]


@dataclass(frozen=True)
class ScoreFieldConfig:
    calibration_mode: CalibrationMode = "none"
    calibration_alpha: float = 1.0
    position_std_floor: float = 0.25
    foreground_mode: ForegroundMode = "none"
    foreground_quantile: float = 0.20
    background_multiplier: float = 0.50
    foreground_smooth_kernel: int = 5
    support_score_quantile: float = 0.90

    def __post_init__(self) -> None:
        if self.calibration_alpha < 0.0:
            raise ScoreFieldConfigError("calibration_alpha", "must be non-negative")
        if self.position_std_floor <= 0.0:
            raise ScoreFieldConfigError("position_std_floor", "must be positive")
        if not 0.0 <= self.foreground_quantile <= 1.0:
            raise ScoreFieldConfigError("foreground_quantile", "must be in [0, 1]")
        if not 0.0 <= self.background_multiplier <= 1.0:
            raise ScoreFieldConfigError("background_multiplier", "must be in [0, 1]")
        if self.foreground_smooth_kernel <= 0:
            raise ScoreFieldConfigError("foreground_smooth_kernel", "must be positive")
        if not 0.0 <= self.support_score_quantile <= 1.0:
            raise ScoreFieldConfigError("support_score_quantile", "must be in [0, 1]")

    @property
    def enabled(self) -> bool:
        return self.calibration_mode != "none" or self.foreground_mode != "none"


@dataclass(frozen=True)
class ScoreFieldStats:
    position_mean: FloatArray
    position_std: FloatArray
    position_high: FloatArray
    foreground_prior: FloatArray


def fit_score_field_stats(
    support_score_fields: Sequence[FloatArray],
    support_feature_fields: Sequence[FloatArray],
    config: ScoreFieldConfig,
    support_object_fields: Sequence[FloatArray] = (),
) -> ScoreFieldStats:
    if not support_score_fields:
        raise ScoreFieldConfigError("support_score_fields", "must not be empty")
    target_shape = _field_shape(support_score_fields[0])
    score_stack = np.stack(
        [_resize_field(_as_float32(field), target_shape) for field in support_score_fields],
        axis=0,
    )
    return ScoreFieldStats(
        position_mean=np.mean(score_stack, axis=0).astype(np.float32, copy=False),
        position_std=np.maximum(
            np.std(score_stack, axis=0).astype(np.float32, copy=False),
            np.float32(config.position_std_floor),
        ),
        position_high=np.quantile(
            score_stack,
            config.support_score_quantile,
            axis=0,
        ).astype(np.float32, copy=False),
        foreground_prior=fit_foreground_prior(
            support_feature_fields,
            support_object_fields,
            target_shape,
            config,
        ),
    )


def fit_foreground_prior(
    support_feature_fields: Sequence[FloatArray],
    support_object_fields: Sequence[FloatArray],
    target_shape: Tuple[int, int],
    config: ScoreFieldConfig,
) -> FloatArray:
    if config.foreground_mode == "none":
        return np.ones(target_shape, dtype=np.float32)
    priors = foreground_prior_fields(
        support_feature_fields,
        support_object_fields,
        target_shape,
        config,
    )
    if not priors:
        return np.ones(target_shape, dtype=np.float32)
    prior = np.mean(np.stack(priors, axis=0), axis=0).astype(np.float32, copy=False)
    if config.foreground_smooth_kernel > 1:
        prior = box_filter(prior, config.foreground_smooth_kernel)
        max_value = float(np.max(prior))
        if max_value > 0.0:
            prior = prior / max_value
    return np.clip(prior, 0.0, 1.0).astype(np.float32, copy=False)


def foreground_prior_fields(
    support_feature_fields: Sequence[FloatArray],
    support_object_fields: Sequence[FloatArray],
    target_shape: Tuple[int, int],
    config: ScoreFieldConfig,
) -> Tuple[FloatArray, ...]:
    if config.foreground_mode == "none":
        return ()
    if config.foreground_mode == "support_feature_energy":
        priors = feature_priors(support_feature_fields, target_shape, _resize_field)
        return thresholded_prior(priors, config.foreground_quantile)
    if config.foreground_mode == "support_rgb_contrast":
        priors = object_priors(support_object_fields, target_shape, _resize_field)
        return thresholded_prior(priors, config.foreground_quantile)
    feature = mean_prior(feature_priors(support_feature_fields, target_shape, _resize_field))
    rgb = mean_prior(object_priors(support_object_fields, target_shape, _resize_field))
    if feature is None or rgb is None:
        return ()
    return thresholded_prior((feature * rgb,), config.foreground_quantile)


def apply_score_field_transform(
    score_field: FloatArray,
    stats: ScoreFieldStats,
    config: ScoreFieldConfig,
) -> FloatArray:
    output = _as_float32(score_field).copy()
    output_shape = _field_shape(output)
    position_mean = _resize_field(stats.position_mean, output_shape)
    position_std = _resize_field(stats.position_std, output_shape)
    if config.calibration_mode == "support_position_center":
        output = output - np.float32(config.calibration_alpha) * position_mean
    elif config.calibration_mode == "support_position_zscore":
        output = (
            output - np.float32(config.calibration_alpha) * position_mean
        ) / np.maximum(position_std, np.float32(config.position_std_floor))
    elif config.calibration_mode == "support_score_reliability":
        reliability = support_score_reliability(
            stats.position_high,
            output_shape,
            config.calibration_alpha,
            _resize_field,
        )
        output = output * reliability
    foreground_prior = _resize_field(stats.foreground_prior, output_shape)
    if config.foreground_mode != "none":
        multiplier = np.float32(config.background_multiplier) + (
            np.float32(1.0 - config.background_multiplier) * foreground_prior
        )
        output = output * multiplier
    return output.astype(np.float32, copy=False)


def support_leave_one_out_patch_scores(
    pipeline: FlowTTE,
    feature_field: FloatArray,
) -> FloatArray:
    has_flow_context = pipeline.config.flow.condition_mode != "none"
    has_score_context = pipeline.config.score.context_mode != "none"
    if has_flow_context or has_score_context:
        raise ScoreFieldConfigError(
            "pipeline",
            "support score-field calibration currently expects no flow/context conditioning",
        )
    if pipeline.estimator is None or pipeline.memory is None or pipeline.score_calibration is None:
        raise ScoreFieldConfigError("pipeline", "must be fitted before scoring support fields")
    evaluation = pipeline.estimator.evaluate(feature_field[np.newaxis, ...])
    score_config = pipeline.config.score
    if score_config.score_mode == "nf_nll":
        patch_scores = evaluation.nll
    else:
        query = pipeline.memory.bank.query(
            evaluation.z,
            k=2,
            chunk_size=score_config.query_chunk_size,
            squared=score_config.use_squared_distance,
        )
        distance_index = 1 if query.distances.shape[1] > 1 else 0
        distances = query.distances[:, distance_index]
        distance_scores = pipeline.score_calibration.normalize_distance(distances)
        density_penalty = pipeline.estimator.density_penalty(evaluation.nll)
        patch_scores = (
            score_config.distance_weight * distance_scores
            + score_config.density_weight * density_penalty
        )
    restored = to_numpy(evaluation.batch.restore(patch_scores))[0]
    return np.asarray(restored, dtype=np.float32)


def _as_float32(values: FloatArray) -> FloatArray:
    return np.asarray(values, dtype=np.float32)


def _field_shape(values: FloatArray) -> Tuple[int, int]:
    if values.ndim != 2:
        raise ScoreFieldConfigError("score_field", "must be HxW")
    return (int(values.shape[0]), int(values.shape[1]))


def _resize_field(field: FloatArray, target_shape: Tuple[int, int]) -> FloatArray:
    if tuple(field.shape) == target_shape:
        return field.astype(np.float32, copy=False)
    import cv2  # noqa: PLC0415

    return np.asarray(
        cv2.resize(
            field,
            (int(target_shape[1]), int(target_shape[0])),
            interpolation=cv2.INTER_LINEAR,
        ),
        dtype=np.float32,
    )
