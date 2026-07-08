from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from typing_extensions import final

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float32]
Float64Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
@final
class Region:
    image_index: int
    mask: BoolArray
    size: int


def aupro_score(
    masks: BoolArray,
    scores: FloatArray,
    max_fpr: float,
    threshold_count: int,
) -> float:
    regions = _all_regions(masks)
    negative_mask = ~masks
    negative_count = _true_count(negative_mask)
    if not regions or negative_count == 0:
        return float("nan")

    score_min = _array_min(scores)
    score_max = _array_max(scores)
    thresholds = _linspace(score_max + 1e-6, score_min - 1e-6, threshold_count)
    fprs: List[float] = []
    pros: List[float] = []
    for threshold in thresholds:
        prediction: BoolArray
        prediction = scores >= threshold
        false_positive_count = _and_count(prediction, negative_mask)
        false_positive_rate = false_positive_count / negative_count
        region_overlaps: List[float] = []
        for region in regions:
            prediction_slice = _bool_slice(prediction, region.image_index)
            region_overlaps.append(_and_count(prediction_slice, region.mask) / region.size)
        fprs.append(false_positive_rate)
        pros.append(sum(region_overlaps) / len(region_overlaps))
    return _integrate_pro(fprs, pros, max_fpr)


def _all_regions(masks: BoolArray) -> List[Region]:
    regions: List[Region] = []
    for image_index in range(len(masks)):
        image_mask = _bool_slice(masks, image_index)
        regions.extend(
            Region(
                image_index=image_index,
                mask=region_mask,
                size=_true_count(region_mask),
            )
            for region_mask in _connected_components(image_mask)
        )
    return regions


def _connected_components(mask: BoolArray) -> List[BoolArray]:
    height = len(mask)
    width = len(_bool_slice(mask, 0))
    visited: BoolArray = np.zeros((height, width), dtype=np.bool_)
    regions: List[BoolArray] = []
    for row in range(height):
        for col in range(width):
            if visited[row, col] or not mask[row, col]:
                continue
            region: BoolArray = np.zeros((height, width), dtype=np.bool_)
            stack: List[Tuple[int, int]] = [(row, col)]
            visited[row, col] = True
            while stack:
                cur_row, cur_col = stack.pop()
                region[cur_row, cur_col] = True
                for next_row, next_col in _neighbors(cur_row, cur_col, height, width):
                    if visited[next_row, next_col] or not mask[next_row, next_col]:
                        continue
                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))
            regions.append(region)
    return regions


def _neighbors(row: int, col: int, height: int, width: int) -> Tuple[Tuple[int, int], ...]:
    candidates = ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
    return tuple(
        (next_row, next_col)
        for next_row, next_col in candidates
        if 0 <= next_row < height and 0 <= next_col < width
    )


def _integrate_pro(fprs: List[float], pros: List[float], max_fpr: float) -> float:
    pairs = sorted(zip(fprs, pros), key=lambda pair: pair[0])
    unique_fprs, unique_pros = _unique_max_curve(pairs)
    end_pro = _interpolate(max_fpr, unique_fprs, unique_pros)
    curve_fprs = [0.0]
    curve_pros = [0.0]
    for fpr, pro in zip(unique_fprs, unique_pros):
        if fpr <= max_fpr:
            curve_fprs.append(fpr)
            curve_pros.append(pro)
    curve_fprs.append(max_fpr)
    curve_pros.append(end_pro)
    return _trapezoid(curve_fprs, curve_pros) / max_fpr


def _unique_max_curve(pairs: List[Tuple[float, float]]) -> Tuple[List[float], List[float]]:
    unique_fprs: List[float] = []
    unique_pros: List[float] = []
    for fpr, pro in pairs:
        if unique_fprs and fpr == unique_fprs[-1]:
            unique_pros[-1] = max(unique_pros[-1], pro)
            continue
        unique_fprs.append(fpr)
        unique_pros.append(pro)
    return unique_fprs, unique_pros


def _interpolate(x: float, xs: List[float], ys: List[float]) -> float:
    if x <= xs[0]:
        return ys[0]
    for index in range(1, len(xs)):
        left_x = xs[index - 1]
        right_x = xs[index]
        if x <= right_x:
            ratio = (x - left_x) / max(right_x - left_x, 1e-12)
            return ys[index - 1] + ratio * (ys[index] - ys[index - 1])
    return ys[-1]


def _trapezoid(xs: List[float], ys: List[float]) -> float:
    total = 0.0
    for index in range(1, len(xs)):
        total += (xs[index] - xs[index - 1]) * (ys[index] + ys[index - 1]) / 2.0
    return total


def _linspace(start: float, stop: float, count: int) -> List[float]:
    step = (stop - start) / (count - 1)
    return [start + step * index for index in range(count)]


def _array_min(values: FloatArray) -> float:
    return float(np.min(values))


def _array_max(values: FloatArray) -> float:
    return float(np.max(values))


def _true_count(values: BoolArray) -> int:
    return int(np.count_nonzero(values))


def _and_count(left: BoolArray, right: BoolArray) -> int:
    combined: BoolArray = np.logical_and(left, right)
    return _true_count(combined)


def _bool_slice(values: BoolArray, index: int) -> BoolArray:
    array: BoolArray = np.asarray(values[index], dtype=np.bool_)
    return array
