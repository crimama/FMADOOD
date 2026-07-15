from __future__ import annotations

import math
from typing import NamedTuple, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from flow_tte.darc_geometry import ImageSize, Point2D

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]


class TilingSpec(NamedTuple):
    crop_size: int = 512
    stride: int = 384
    interior_margin: int = 32
    hann_floor: float = 0.1


class NativeCrop(NamedTuple):
    y0: int
    x0: int
    height: int
    width: int

    def contains(self, point: Point2D) -> bool:
        return (
            self.x0 <= point.x < self.x0 + self.width
            and self.y0 <= point.y < self.y0 + self.height
        )

    def contains_interior(self, point: Point2D, margin: int) -> bool:
        return (
            self.x0 + margin <= point.x < self.x0 + self.width - margin
            and self.y0 + margin <= point.y < self.y0 + self.height - margin
        )

    def center_distance_squared(self, point: Point2D) -> float:
        dx = point.x - (self.x0 + (self.width - 1) / 2.0)
        dy = point.y - (self.y0 + (self.height - 1) / 2.0)
        return dx * dx + dy * dy


class TokenGrid(NamedTuple):
    shape: ImageSize
    native_spacing: float


class CropScores(NamedTuple):
    crop: NativeCrop
    scores: FloatArray


def _axis_starts(length: int, spec: TilingSpec) -> Tuple[int, ...]:
    final_start = max(0, length - spec.crop_size)
    starts = list(range(0, final_start + 1, spec.stride))
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(starts)


def native_crop_grid(shape: ImageSize, spec: TilingSpec) -> Tuple[NativeCrop, ...]:
    return tuple(
        NativeCrop(
            y0=y0,
            x0=x0,
            height=min(spec.crop_size, shape.height - y0),
            width=min(spec.crop_size, shape.width - x0),
        )
        for y0 in _axis_starts(shape.height, spec)
        for x0 in _axis_starts(shape.width, spec)
    )


def owner_crop(
    point: Point2D,
    crops: Sequence[NativeCrop],
    interior_margin: int = 32,
) -> Optional[NativeCrop]:
    containing = tuple(crop for crop in crops if crop.contains(point))
    if not containing:
        return None
    interior = tuple(crop for crop in containing if crop.contains_interior(point, interior_margin))
    candidates = interior or containing
    return min(
        candidates,
        key=lambda crop: (crop.center_distance_squared(point), crop.y0, crop.x0),
    )


def clipped_token_candidates(point: Point2D, crop: NativeCrop, grid: TokenGrid) -> IntArray:
    column_position = (point.x - crop.x0 + 0.5) / grid.native_spacing - 0.5
    row_position = (point.y - crop.y0 + 0.5) / grid.native_spacing - 0.5
    center_column = math.floor(column_position + 0.5)
    center_row = math.floor(row_position + 0.5)
    row_start = max(0, center_row - 1)
    row_stop = min(grid.shape.height, center_row + 2)
    column_start = max(0, center_column - 1)
    column_stop = min(grid.shape.width, center_column + 2)
    return np.asarray(
        [
            (row, column)
            for row in range(row_start, row_stop)
            for column in range(column_start, column_stop)
        ],
        dtype=np.int64,
    )


def hann_window(shape: ImageSize, floor: float = 0.1) -> FloatArray:
    vertical = np.hanning(shape.height) if shape.height > 1 else np.ones(1)
    horizontal = np.hanning(shape.width) if shape.width > 1 else np.ones(1)
    return np.maximum(np.outer(vertical, horizontal), floor).astype(np.float32)


def blend_crop_scores(
    shape: ImageSize,
    crop_scores: Sequence[CropScores],
    floor: float = 0.1,
) -> FloatArray:
    weighted_sum = np.zeros((shape.height, shape.width), dtype=np.float32)
    weight_sum = np.zeros_like(weighted_sum)
    for item in crop_scores:
        crop = item.crop
        rows = slice(crop.y0, crop.y0 + crop.height)
        columns = slice(crop.x0, crop.x0 + crop.width)
        weight = hann_window(ImageSize(height=crop.height, width=crop.width), floor)
        weighted_sum[rows, columns] += item.scores * weight
        weight_sum[rows, columns] += weight
    return np.divide(
        weighted_sum,
        weight_sum,
        out=np.zeros_like(weighted_sum),
        where=weight_sum > 0.0,
    )
