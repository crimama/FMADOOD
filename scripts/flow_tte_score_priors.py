from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Final, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
MIN_STD: Final = 1e-6


@dataclass(frozen=True)
class ScoreFieldConfigError(ValueError):
    field: str
    reason: str

    def __str__(self) -> str:
        return f"Invalid {self.field}: {self.reason}"


def feature_energy(feature_field: FloatArray) -> FloatArray:
    if feature_field.ndim != 3:
        raise ScoreFieldConfigError("feature_field", "must be HxWxC")
    return np.linalg.norm(feature_field, axis=-1).astype(np.float32, copy=False)


def rgb_foreground_proxy(
    image: npt.NDArray[np.uint8],
    grid_shape: Tuple[int, int],
) -> FloatArray:
    height, width = grid_shape
    import cv2  # noqa: PLC0415

    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    red = resized[:, :, 0].astype(np.float32)
    green = resized[:, :, 1].astype(np.float32)
    blue = resized[:, :, 2].astype(np.float32)
    gray = (0.299 * red + 0.587 * green + 0.114 * blue).astype(np.float32, copy=False)
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]], axis=0)
    background = float(np.median(border))
    return normalize_minmax(np.abs(gray - background).astype(np.float32, copy=False))


def support_score_reliability(
    position_high: FloatArray,
    output_shape: Tuple[int, int],
    calibration_alpha: float,
    resize_field: Callable[[FloatArray, Tuple[int, int]], FloatArray],
) -> FloatArray:
    high = normalize_minmax(resize_field(position_high, output_shape))
    reliability = np.float32(1.0) - np.float32(calibration_alpha) * high
    return np.clip(reliability, 0.0, 1.0).astype(np.float32, copy=False)


def feature_priors(
    support_feature_fields: Sequence[FloatArray],
    target_shape: Tuple[int, int],
    resize_field: Callable[[FloatArray, Tuple[int, int]], FloatArray],
) -> Tuple[FloatArray, ...]:
    return tuple(
        normalize_minmax(
            resize_field(feature_energy(np.asarray(field, dtype=np.float32)), target_shape),
        )
        for field in support_feature_fields
    )


def object_priors(
    support_object_fields: Sequence[FloatArray],
    target_shape: Tuple[int, int],
    resize_field: Callable[[FloatArray, Tuple[int, int]], FloatArray],
) -> Tuple[FloatArray, ...]:
    return tuple(
        normalize_minmax(resize_field(np.asarray(field, dtype=np.float32), target_shape))
        for field in support_object_fields
    )


def mean_prior(priors: Sequence[FloatArray]) -> Optional[FloatArray]:
    if not priors:
        return None
    return np.mean(np.stack(priors, axis=0), axis=0).astype(np.float32, copy=False)


def thresholded_prior(
    priors: Sequence[FloatArray],
    foreground_quantile: float,
) -> Tuple[FloatArray, ...]:
    thresholded = []
    for prior in priors:
        if float(np.max(prior) - np.min(prior)) <= MIN_STD:
            continue
        threshold = float(np.quantile(prior.reshape(-1), foreground_quantile))
        thresholded.append((prior >= threshold).astype(np.float32, copy=False))
    return tuple(thresholded)


def normalize_minmax(values: FloatArray) -> FloatArray:
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    if value_max - value_min <= MIN_STD:
        return np.ones(values.shape, dtype=np.float32)
    return ((values - value_min) / np.float32(value_max - value_min)).astype(
        np.float32,
        copy=False,
    )


def box_filter(values: FloatArray, kernel_size: int) -> FloatArray:
    radius = kernel_size // 2
    padded = np.pad(values, ((radius, radius), (radius, radius)), mode="edge")
    output = np.zeros(values.shape, dtype=np.float32)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            window = padded[y : y + kernel_size, x : x + kernel_size]
            output[y, x] = np.float32(np.mean(window))
    return output
