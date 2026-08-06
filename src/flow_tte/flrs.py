from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]

RADIUS1_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
RADIUS2_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (-2, -2),
    (-2, 0),
    (-2, 2),
    (0, -2),
    (0, 2),
    (2, -2),
    (2, 0),
    (2, 2),
)
RANDOM_NEIGHBOR_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (-5, -3),
    (-4, 5),
    (-3, 4),
    (3, -5),
    (4, 3),
    (5, -4),
    (5, 5),
    (-5, -5),
)


@dataclass(frozen=True)
class RobustEnvelope:
    center: FloatArray
    scale: FloatArray


def _finite_float(name: str, value: npt.ArrayLike, *, ndim: int) -> FloatArray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != ndim or any(size <= 0 for size in result.shape):
        msg = f"{name} must be a non-empty {ndim}D array"
        raise ValueError(msg)
    if not bool(np.all(np.isfinite(result))):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    return result


def normalize_feature_map(features: npt.ArrayLike) -> FloatArray:
    values = _finite_float("features", features, ndim=3)
    norms = np.linalg.norm(values.astype(np.float64), axis=-1, keepdims=True)
    if bool(np.any(norms <= 1e-12)):
        raise ValueError("features contain an undefined zero direction")
    return np.asarray(values.astype(np.float64) / norms, dtype=np.float32)


def _offset_cosines(
    normalized: FloatArray,
    offsets: Sequence[Tuple[int, int]],
) -> FloatArray:
    if not offsets:
        raise ValueError("offsets must not be empty")
    max_y = max(abs(offset[0]) for offset in offsets)
    max_x = max(abs(offset[1]) for offset in offsets)
    padded = np.pad(normalized, ((max_y, max_y), (max_x, max_x), (0, 0)), mode="edge")
    height, width = normalized.shape[:2]
    parts: list[FloatArray] = []
    for y_offset, x_offset in offsets:
        shifted = padded[
            max_y + y_offset : max_y + y_offset + height,
            max_x + x_offset : max_x + x_offset + width,
        ]
        cosine = np.einsum(
            "hwc,hwc->hw",
            normalized.astype(np.float64),
            shifted.astype(np.float64),
        )
        parts.append(np.asarray(np.clip(cosine, -1.0, 1.0), dtype=np.float32))
    return np.stack(parts, axis=-1).astype(np.float32, copy=False)


def local_relation_signature(
    features: npt.ArrayLike,
    *,
    radius1_offsets: Sequence[Tuple[int, int]] = RADIUS1_OFFSETS,
    radius2_offsets: Sequence[Tuple[int, int]] = RADIUS2_OFFSETS,
) -> FloatArray:
    normalized = normalize_feature_map(features)
    radius1 = _offset_cosines(normalized, radius1_offsets)
    radius2 = _offset_cosines(normalized, radius2_offsets)
    horizontal = _offset_cosines(normalized, ((0, -1), (0, 1))).mean(axis=-1)
    vertical = _offset_cosines(normalized, ((-1, 0), (1, 0))).mean(axis=-1)
    signature = np.stack(
        (
            radius1.mean(axis=-1),
            radius1.std(axis=-1),
            radius2.mean(axis=-1),
            np.abs(horizontal - vertical),
        ),
        axis=-1,
    )
    return np.asarray(signature, dtype=np.float32)


def radius1_relation_signature(features: npt.ArrayLike) -> FloatArray:
    normalized = normalize_feature_map(features)
    radius1 = _offset_cosines(normalized, RADIUS1_OFFSETS)
    horizontal = _offset_cosines(normalized, ((0, -1), (0, 1))).mean(axis=-1)
    vertical = _offset_cosines(normalized, ((-1, 0), (1, 0))).mean(axis=-1)
    signature = np.stack(
        (
            radius1.mean(axis=-1),
            radius1.std(axis=-1),
            np.abs(horizontal - vertical),
        ),
        axis=-1,
    )
    return np.asarray(signature, dtype=np.float32)


def random_neighbor_signature(features: npt.ArrayLike) -> FloatArray:
    normalized = normalize_feature_map(features)
    distant = _offset_cosines(normalized, RANDOM_NEIGHBOR_OFFSETS)
    first = distant[..., :4]
    second = distant[..., 4:]
    signature = np.stack(
        (
            first.mean(axis=-1),
            first.std(axis=-1),
            second.mean(axis=-1),
            np.abs(first.mean(axis=-1) - second.mean(axis=-1)),
        ),
        axis=-1,
    )
    return np.asarray(signature, dtype=np.float32)


def fit_robust_envelope(
    values: npt.ArrayLike,
    *,
    scale_floor: float = 1e-3,
) -> RobustEnvelope:
    matrix = _finite_float("values", values, ndim=2)
    if not np.isfinite(scale_floor) or scale_floor <= 0.0:
        raise ValueError("scale_floor must be finite and positive")
    center = np.median(matrix.astype(np.float64), axis=0)
    mad = np.median(np.abs(matrix.astype(np.float64) - center), axis=0)
    scale = np.maximum(1.4826 * mad, float(scale_floor))
    return RobustEnvelope(
        center=np.asarray(center, dtype=np.float32),
        scale=np.asarray(scale, dtype=np.float32),
    )


def score_robust_energy(values: npt.ArrayLike, envelope: RobustEnvelope) -> FloatArray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim < 2 or array.shape[-1] != envelope.center.shape[0]:
        raise ValueError("values and envelope feature dimensions must match")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError("values must contain only finite values")
    z = (array.astype(np.float64) - envelope.center.astype(np.float64)) / envelope.scale.astype(
        np.float64,
    )
    scores = np.mean(np.square(z), axis=-1)
    if not bool(np.all(np.isfinite(scores))):
        raise RuntimeError("robust energy produced non-finite scores")
    return np.asarray(scores, dtype=np.float32)


def fit_signature_envelope(signatures: Sequence[npt.ArrayLike]) -> RobustEnvelope:
    if not signatures:
        raise ValueError("signatures must not be empty")
    matrices = [
        _finite_float("signature", value, ndim=3).reshape(-1, np.asarray(value).shape[-1])
        for value in signatures
    ]
    dimensions = {matrix.shape[1] for matrix in matrices}
    if len(dimensions) != 1:
        raise ValueError("all signatures must have the same feature dimension")
    return fit_robust_envelope(np.concatenate(matrices, axis=0))


def spatial_permutation_indices(height: int, width: int, *, seed: int) -> Tuple[IntArray, IntArray]:
    if height <= 0 or width <= 0 or seed < 0:
        raise ValueError("height and width must be positive and seed non-negative")
    permutation = np.random.default_rng(seed).permutation(height * width).astype(np.int64)
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(permutation.size, dtype=np.int64)
    return permutation, inverse


def permute_feature_grid(features: npt.ArrayLike, *, seed: int) -> Tuple[FloatArray, IntArray]:
    values = _finite_float("features", features, ndim=3)
    height, width, channels = values.shape
    permutation, inverse = spatial_permutation_indices(height, width, seed=seed)
    permuted = values.reshape(-1, channels)[permutation].reshape(height, width, channels)
    return np.asarray(permuted, dtype=np.float32), inverse


def invert_permuted_map(score_map: npt.ArrayLike, inverse: npt.ArrayLike) -> FloatArray:
    scores = _finite_float("score_map", score_map, ndim=2)
    indices = np.asarray(inverse)
    if (
        indices.ndim != 1
        or indices.size != scores.size
        or not np.issubdtype(
            indices.dtype,
            np.integer,
        )
    ):
        raise ValueError("inverse must be an integer permutation for score_map")
    restored = scores.reshape(-1)[indices].reshape(scores.shape)
    return np.asarray(restored, dtype=np.float32)


def combine_layer_scores(score_maps: Sequence[npt.ArrayLike]) -> FloatArray:
    if not score_maps:
        raise ValueError("score_maps must not be empty")
    arrays = [_finite_float("score_map", value, ndim=2) for value in score_maps]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("all layer score maps must have the same shape")
    return np.asarray(np.mean(np.stack(arrays, axis=0), axis=0), dtype=np.float32)
