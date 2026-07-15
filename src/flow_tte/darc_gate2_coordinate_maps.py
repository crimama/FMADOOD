from __future__ import annotations

import math
from hashlib import sha256
from typing import Set, Tuple

import numpy as np

from flow_tte.darc_feature_stream import ImageFeatures
from flow_tte.darc_gate2_correspondence_types import (
    DEFAULT_CORRESPONDENCE_CONFIG,
    MATRIX_EPSILON,
    CandidateRequest,
    CoarseGeometry,
    CorrespondenceConfig,
    CorrespondenceError,
    FeatureArray,
    Float64Array,
    FloatArray,
    IntArray,
    PreparedCorrespondence,
)
from flow_tte.darc_geometry import FeatureGrid, ImageSize, Point2D, RealizedResize, token_centers
from flow_tte.darc_tiling import TokenGrid


def coarse_geometry(
    image: ImageFeatures,
    config: CorrespondenceConfig = DEFAULT_CORRESPONDENCE_CONFIG,
) -> CoarseGeometry:
    if config.short_edge <= 0 or config.patch_size <= 0:
        raise CorrespondenceError("short edge and patch size must be positive")
    native = image.native_size
    if min(native.height, native.width) <= 0:
        raise CorrespondenceError("native image dimensions must be positive")
    scale = config.short_edge / min(native.height, native.width)
    realized = ImageSize(height=int(native.height * scale), width=int(native.width * scale))
    retained = ImageSize(
        height=realized.height - realized.height % config.patch_size,
        width=realized.width - realized.width % config.patch_size,
    )
    coarse: FeatureArray = np.asarray(image.coarse, dtype=np.float16)
    expected = (retained.height // config.patch_size, retained.width // config.patch_size)
    if coarse.ndim != 3 or coarse.shape[:2] != expected or coarse.shape[2] == 0:
        raise CorrespondenceError("coarse grid does not match realized aligned resize")
    if not np.all(np.isfinite(coarse)):
        raise CorrespondenceError("coarse grid must be finite")
    shape = ImageSize(height=expected[0], width=expected[1])
    feature_dimension = int(np.size(coarse, axis=2))
    return CoarseGeometry(
        resize=RealizedResize(native=native, realized=realized, retained=retained),
        grid=FeatureGrid(
            features=np.asarray(coarse.reshape(-1, feature_dimension), dtype=np.float32),
            centers=token_centers(shape, config.patch_size),
        ),
    )


def invert_similarity(matrix: FloatArray) -> FloatArray:
    values: Float64Array = np.asarray(matrix, dtype=np.float64)
    if values.shape != (2, 3) or not np.all(np.isfinite(values)):
        raise CorrespondenceError("similarity matrix must be finite 2x3")
    a, b, c, d = (
        float(np.sum(values[row : row + 1, column : column + 1]))
        for row in range(2)
        for column in range(2)
    )
    determinant = a * d - b * c
    if abs(determinant) <= MATRIX_EPSILON:
        raise CorrespondenceError("similarity matrix must be invertible")
    inverse_linear: Float64Array = np.asarray(
        np.asarray([[d, -b], [-c, a]], dtype=np.float64) / determinant,
        dtype=np.float64,
    )
    inverse_translation: Float64Array = np.asarray(
        -(inverse_linear @ values[:, 2]),
        dtype=np.float64,
    )
    return np.asarray(np.column_stack((inverse_linear, inverse_translation)), dtype=np.float32)


def derive_ransac_seed(query_id: str, support_id: str) -> int:
    digest = sha256(query_id.encode("utf-8") + b"\0" + support_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def map_identity_native(point: Point2D, query: ImageSize, support: ImageSize) -> Point2D:
    if min(query.height, query.width, support.height, support.width) <= 0:
        raise CorrespondenceError("identity mapping sizes must be positive")
    return Point2D(
        x=(point.x + 0.5) * support.width / query.width - 0.5,
        y=(point.y + 0.5) * support.height / query.height - 0.5,
    )


def query_token_points(
    prepared: PreparedCorrespondence,
    request: CandidateRequest,
) -> FloatArray:
    if request.crop_index < 0 or request.crop_index >= len(prepared.query.high):
        raise CorrespondenceError("candidate request crop is outside the prepared query")
    grid = prepared.query.high[request.crop_index]
    crop = prepared.query.crops[request.crop_index]
    token_grid = TokenGrid(
        shape=ImageSize(
            height=int(np.size(grid, axis=0)),
            width=int(np.size(grid, axis=1)),
        ),
        native_spacing=crop.width / int(np.size(grid, axis=1)),
    )
    token_count = int(np.size(grid, axis=0)) * int(np.size(grid, axis=1))
    if request.start < 0 or request.stop <= request.start or request.stop > token_count:
        raise CorrespondenceError("candidate request exceeds the query token grid")
    flat: IntArray = np.arange(request.start, request.stop, dtype=np.int64)
    rows: IntArray = np.asarray(flat // token_grid.shape.width, dtype=np.int64)
    columns: IntArray = np.asarray(flat % token_grid.shape.width, dtype=np.int64)
    xs: FloatArray = np.asarray(
        crop.x0 + (columns.astype(np.float32) + 0.5) * token_grid.native_spacing - 0.5,
        dtype=np.float32,
    )
    ys: FloatArray = np.asarray(
        crop.y0 + (rows.astype(np.float32) + 0.5) * token_grid.native_spacing - 0.5,
        dtype=np.float32,
    )
    return np.asarray(np.stack((xs, ys), axis=1), dtype=np.float32)


def high_feature_dimension(image: ImageFeatures) -> int:
    if not image.crops or len(image.crops) != len(image.high):
        raise CorrespondenceError("crops and high grids must have equal non-zero length")
    dimensions: Set[int] = set()
    for crop_index in range(len(image.crops)):
        grid, _ = high_feature_grid(image, crop_index)
        dimensions.add(int(np.size(grid, axis=2)))
    if len(dimensions) != 1 or 0 in dimensions:
        raise CorrespondenceError("all high grids must share one positive feature dimension")
    return next(iter(dimensions))


def high_feature_grid(image: ImageFeatures, crop_index: int) -> Tuple[FeatureArray, TokenGrid]:
    if crop_index < 0 or crop_index >= len(image.crops) or crop_index >= len(image.high):
        raise CorrespondenceError("crop index is outside the high feature inventory")
    grid: FeatureArray = np.asarray(image.high[crop_index], dtype=np.float16)
    if grid.ndim != 3 or min(grid.shape) <= 0:
        raise CorrespondenceError("high feature grid must be a non-empty HxWxD array")
    crop = image.crops[crop_index]
    spacing_x = crop.width / int(np.size(grid, axis=1))
    spacing_y = crop.height / int(np.size(grid, axis=0))
    if not math.isclose(spacing_x, spacing_y, rel_tol=1e-9, abs_tol=1e-9):
        raise CorrespondenceError("high feature grid must have one native token spacing")
    token_grid = TokenGrid(
        shape=ImageSize(
            height=int(np.size(grid, axis=0)),
            width=int(np.size(grid, axis=1)),
        ),
        native_spacing=spacing_x,
    )
    return np.asarray(grid, dtype=np.float16), token_grid


def inverse_or_identity(matrix: FloatArray) -> Tuple[FloatArray, bool]:
    try:
        return invert_similarity(matrix), True
    except CorrespondenceError:
        identity: FloatArray = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        return identity, False
