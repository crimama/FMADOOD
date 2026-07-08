# /// script
# requires-python = ">=3.8"
# dependencies = ["numpy", "torch", "typing-extensions"]
# ///
# pyright: reportMissingImports=false
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Sequence, Tuple

import numpy as np
import numpy.typing as npt

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (_REPO_ROOT, _REPO_ROOT / "src"):
    _path_text = str(_path)
    if _path_text not in sys.path:
        sys.path.insert(0, _path_text)

from flow_tte.tensors import to_numpy  # noqa: E402

if TYPE_CHECKING:
    from flow_tte.pipeline import FlowTTE

FloatArray = npt.NDArray[np.float32]
CalibrationMode = Literal["none", "support_position_center", "support_position_zscore"]
ForegroundMode = Literal["none", "support_feature_energy"]

_MIN_STD: Final = 1e-6


@dataclass(frozen=True)
class ScoreFieldConfigError(ValueError):
    field: str
    reason: str

    def __str__(self) -> str:
        return f"Invalid {self.field}: {self.reason}"


@dataclass(frozen=True)
class ScoreFieldConfig:
    calibration_mode: CalibrationMode = "none"
    calibration_alpha: float = 1.0
    position_std_floor: float = 0.25
    foreground_mode: ForegroundMode = "none"
    foreground_quantile: float = 0.20
    background_multiplier: float = 0.50
    foreground_smooth_kernel: int = 5

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

    @property
    def enabled(self) -> bool:
        return self.calibration_mode != "none" or self.foreground_mode != "none"


@dataclass(frozen=True)
class ScoreFieldStats:
    position_mean: FloatArray
    position_std: FloatArray
    foreground_prior: FloatArray


def fit_score_field_stats(
    support_score_fields: Sequence[FloatArray],
    support_feature_fields: Sequence[FloatArray],
    config: ScoreFieldConfig,
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
        foreground_prior=fit_foreground_prior(support_feature_fields, target_shape, config),
    )


def fit_foreground_prior(
    support_feature_fields: Sequence[FloatArray],
    target_shape: Tuple[int, int],
    config: ScoreFieldConfig,
) -> FloatArray:
    if config.foreground_mode == "none" or not support_feature_fields:
        return np.ones(target_shape, dtype=np.float32)
    energy_fields = [
        _resize_field(feature_energy(_as_float32(field)), target_shape)
        for field in support_feature_fields
    ]
    energy = np.mean(np.stack(energy_fields, axis=0), axis=0)
    if float(np.max(energy) - np.min(energy)) <= _MIN_STD:
        return np.ones(target_shape, dtype=np.float32)
    threshold = float(np.quantile(energy.reshape(-1), config.foreground_quantile))
    prior = (energy >= threshold).astype(np.float32, copy=False)
    if config.foreground_smooth_kernel > 1:
        prior = box_filter(prior, config.foreground_smooth_kernel)
        max_value = float(np.max(prior))
        if max_value > 0.0:
            prior = prior / max_value
    return np.clip(prior, 0.0, 1.0).astype(np.float32, copy=False)


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
    foreground_prior = _resize_field(stats.foreground_prior, output_shape)
    if config.foreground_mode == "support_feature_energy":
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


def feature_energy(feature_field: FloatArray) -> FloatArray:
    if feature_field.ndim != 3:
        raise ScoreFieldConfigError("feature_field", "must be HxWxC")
    return np.linalg.norm(feature_field, axis=-1).astype(np.float32, copy=False)


def box_filter(values: FloatArray, kernel_size: int) -> FloatArray:
    radius = kernel_size // 2
    padded = np.pad(values, ((radius, radius), (radius, radius)), mode="edge")
    output = np.zeros(values.shape, dtype=np.float32)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            window = padded[y : y + kernel_size, x : x + kernel_size]
            output[y, x] = np.float32(np.mean(window))
    return output


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
