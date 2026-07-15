from __future__ import annotations

import math
from typing import Final, NamedTuple, Optional

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
Float64Array = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IndexArray = npt.NDArray[np.intp]

_EPSILON: Final = 1e-12
_CONFIDENCE_ERROR_SCALE: Final = 32.0


class ImageSize(NamedTuple):
    height: int
    width: int


class Point2D(NamedTuple):
    x: float
    y: float


class RealizedResize(NamedTuple):
    native: ImageSize
    realized: ImageSize
    retained: ImageSize

    @property
    def scale_x(self) -> float:
        return self.realized.width / self.native.width

    @property
    def scale_y(self) -> float:
        return self.realized.height / self.native.height

    def native_to_resized(self, point: Point2D) -> Point2D:
        return Point2D(
            x=self.scale_x * (point.x + 0.5) - 0.5,
            y=self.scale_y * (point.y + 0.5) - 0.5,
        )

    def resized_to_native(self, point: Point2D) -> Point2D:
        return Point2D(
            x=(point.x + 0.5) / self.scale_x - 0.5,
            y=(point.y + 0.5) / self.scale_y - 0.5,
        )

    def contains_retained_native(self, point: Point2D) -> bool:
        resized = self.native_to_resized(point)
        return (
            -0.5 <= resized.x < self.retained.width - 0.5
            and -0.5 <= resized.y < self.retained.height - 0.5
        )


class FeatureGrid(NamedTuple):
    features: FloatArray
    centers: FloatArray


class PointPairs(NamedTuple):
    support_points: FloatArray
    query_points: FloatArray


class SimilarityRansacConfig(NamedTuple):
    reprojection_threshold: float = 32.0
    max_iterations: int = 2000
    confidence: float = 0.99
    refinement_iterations: int = 10
    min_inliers: int = 12
    min_inlier_ratio: float = 0.25
    min_scale: float = 0.8
    max_scale: float = 1.25
    max_median_error: float = 24.0


_DEFAULT_RANSAC_CONFIG: Final = SimilarityRansacConfig()


class SimilarityAlignment(NamedTuple):
    matrix: FloatArray
    inlier_mask: BoolArray
    inlier_count: int
    inlier_ratio: float
    scale: float
    median_error: float
    confidence: float
    accepted: bool


def token_centers(shape: ImageSize, patch_size: int = 16) -> FloatArray:
    xs = np.arange(shape.width, dtype=np.float32) * patch_size + patch_size / 2.0 - 0.5
    ys = np.arange(shape.height, dtype=np.float32) * patch_size + patch_size / 2.0 - 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack((grid_x.reshape(-1), grid_y.reshape(-1)), axis=1).astype(
        np.float32,
        copy=False,
    )


def _normalized_rows(features: FloatArray) -> FloatArray:
    products: FloatArray = features * features
    squared: FloatArray = np.asarray(np.sum(products, axis=1, keepdims=True), dtype=np.float32)
    return np.asarray(features / np.sqrt(np.maximum(squared, _EPSILON)), dtype=np.float32)


def mutual_nearest_pairs(support: FeatureGrid, query: FeatureGrid) -> PointPairs:
    normalized_support = _normalized_rows(support.features)
    normalized_query = _normalized_rows(query.features)
    similarities: FloatArray = np.asarray(normalized_support @ normalized_query.T, dtype=np.float32)
    support_to_query: IndexArray = np.asarray(np.argmax(similarities, axis=1), dtype=np.intp)
    query_to_support: IndexArray = np.asarray(np.argmax(similarities, axis=0), dtype=np.intp)
    support_count = len(support.features)
    support_indices: IndexArray = np.arange(support_count, dtype=np.intp)
    mutual: BoolArray = np.asarray(
        query_to_support[support_to_query] == support_indices,
        dtype=np.bool_,
    )
    return PointPairs(
        support_points=support.centers[mutual].astype(np.float32, copy=False),
        query_points=query.centers[support_to_query[mutual]].astype(np.float32, copy=False),
    )


def _fit_similarity(support: FloatArray, query: FloatArray) -> Optional[FloatArray]:
    if len(support) < 2:
        return None
    support64: Float64Array = support.astype(np.float64, copy=False)
    query64: Float64Array = query.astype(np.float64, copy=False)
    support_center: Float64Array = np.asarray(np.mean(support64, axis=0), dtype=np.float64)
    query_center: Float64Array = np.asarray(np.mean(query64, axis=0), dtype=np.float64)
    centered_support: Float64Array = support64 - support_center
    centered_query: Float64Array = query64 - query_center
    denominator = float(np.sum(centered_support * centered_support))
    if denominator <= _EPSILON:
        return None
    a = float(np.sum(centered_support * centered_query)) / denominator
    cross: Float64Array = (
        centered_support[:, 0] * centered_query[:, 1]
        - centered_support[:, 1] * centered_query[:, 0]
    )
    b = float(np.sum(cross)) / denominator
    support_x, support_y = (float(np.sum(support_center[index : index + 1])) for index in range(2))
    query_x, query_y = (float(np.sum(query_center[index : index + 1])) for index in range(2))
    tx = query_x - a * support_x + b * support_y
    ty = query_y - b * support_x - a * support_y
    return np.asarray([[a, -b, tx], [b, a, ty]], dtype=np.float32)


def _errors(matrix: FloatArray, pairs: PointPairs) -> FloatArray:
    projected: FloatArray = np.asarray(
        pairs.support_points @ matrix[:, :2].T + matrix[:, 2],
        dtype=np.float32,
    )
    delta: FloatArray = np.asarray(projected - pairs.query_points, dtype=np.float32)
    squared: FloatArray = np.asarray(np.sum(delta * delta, axis=1), dtype=np.float32)
    return np.sqrt(squared).astype(np.float32, copy=False)


def _required_iterations(inlier_ratio: float, confidence: float, maximum: int) -> int:
    success = inlier_ratio * inlier_ratio
    if success >= 1.0:
        return 1
    denominator = math.log(max(_EPSILON, 1.0 - success))
    return min(maximum, max(1, math.ceil(math.log(1.0 - confidence) / denominator)))


def fit_similarity_ransac(
    pairs: PointPairs,
    config: SimilarityRansacConfig = _DEFAULT_RANSAC_CONFIG,
    seed: int = 0,
) -> SimilarityAlignment:
    count = len(pairs.support_points)
    best_matrix: FloatArray = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    best_mask: BoolArray = np.zeros(count, dtype=np.bool_)
    rng = np.random.default_rng(seed)
    limit = config.max_iterations
    iteration = 0
    while count >= 2 and iteration < limit:
        sample: IndexArray = np.asarray(rng.choice(count, size=2, replace=False), dtype=np.intp)
        candidate = _fit_similarity(pairs.support_points[sample], pairs.query_points[sample])
        iteration += 1
        if candidate is None:
            continue
        errors = _errors(candidate, pairs)
        mask = errors <= config.reprojection_threshold
        if np.count_nonzero(mask) > np.count_nonzero(best_mask):
            best_matrix = candidate
            best_mask = mask
            ratio = int(np.count_nonzero(mask)) / count
            limit = min(limit, _required_iterations(ratio, config.confidence, limit))
    for _ in range(config.refinement_iterations):
        refined = _fit_similarity(pairs.support_points[best_mask], pairs.query_points[best_mask])
        if refined is None:
            break
        best_matrix = refined
        best_mask = _errors(best_matrix, pairs) <= config.reprojection_threshold
    errors = _errors(best_matrix, pairs)
    inlier_count = int(np.count_nonzero(best_mask))
    ratio = inlier_count / count if count > 0 else 0.0
    median_error = float(np.median(errors[best_mask])) if inlier_count > 0 else math.inf
    scale = math.sqrt(float(np.sum(best_matrix[:, 0] * best_matrix[:, 0])))
    determinant = scale * scale
    accepted = (
        determinant > 0.0
        and inlier_count >= config.min_inliers
        and ratio >= config.min_inlier_ratio
        and config.min_scale <= scale <= config.max_scale
        and median_error <= config.max_median_error
    )
    raw_confidence = ratio * math.exp(-median_error / _CONFIDENCE_ERROR_SCALE)
    alignment_confidence = min(1.0, max(0.0, raw_confidence))
    return SimilarityAlignment(
        matrix=best_matrix,
        inlier_mask=best_mask,
        inlier_count=inlier_count,
        inlier_ratio=ratio,
        scale=scale,
        median_error=median_error,
        confidence=alignment_confidence,
        accepted=accepted,
    )
