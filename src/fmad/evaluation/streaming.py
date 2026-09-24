"""Common bounded-memory segmentation metrics for native-resolution score maps.

The accumulator consumes one mini-batch at a time.  Scores are quantized to
IEEE float16 order codes so pixel AUROC, oracle F1, and AUPRO can be recovered
from fixed-size histograms without retaining dense prediction maps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


_CODE_COUNT = 1 << 16


def _ordered_float16_codes(values: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float16)
    if not np.isfinite(scores).all():
        raise ValueError("streaming metrics require finite score maps")
    bits = scores.view(np.uint16).copy()
    bits[(bits & np.uint16(0x7FFF)) == 0] = 0
    negative = (bits & np.uint16(0x8000)) != 0
    return np.where(negative, np.bitwise_not(bits), bits ^ np.uint16(0x8000)).astype(
        np.uint16,
    )


def _code_to_float16(code: int) -> float:
    value = np.uint16(code)
    bits = value ^ np.uint16(0x8000) if value & np.uint16(0x8000) else ~value
    return float(np.asarray(bits, dtype=np.uint16).view(np.float16))


def _label8(mask: np.ndarray) -> tuple[np.ndarray, int]:
    binary = np.asarray(mask, dtype=bool)
    try:
        from scipy import ndimage

        labels, count = ndimage.label(
            binary,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        return labels.astype(np.int32, copy=False), int(count)
    except ImportError:
        labels = np.zeros(binary.shape, dtype=np.int32)
        count = 0
        for start in zip(*np.nonzero(binary), strict=True):
            if labels[start]:
                continue
            count += 1
            labels[start] = count
            stack = [start]
            while stack:
                y, x = stack.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if not (dy or dx):
                            continue
                        yy, xx = y + dy, x + dx
                        if (
                            0 <= yy < binary.shape[0]
                            and 0 <= xx < binary.shape[1]
                            and binary[yy, xx]
                            and not labels[yy, xx]
                        ):
                            labels[yy, xx] = count
                            stack.append((yy, xx))
        return labels, count


def _bounded_trapezoid(x: np.ndarray, y: np.ndarray, limit: float) -> float:
    if x[0] != 0.0:
        x = np.insert(x, 0, 0.0)
        y = np.insert(y, 0, 0.0)
    stop = int(np.searchsorted(x, limit, side="right"))
    clipped_x = x[:stop]
    clipped_y = y[:stop]
    if clipped_x[-1] < limit:
        right = min(stop, len(x) - 1)
        left = max(0, right - 1)
        if x[right] == x[left]:
            interpolated = y[right]
        else:
            fraction = (limit - x[left]) / (x[right] - x[left])
            interpolated = y[left] + fraction * (y[right] - y[left])
        clipped_x = np.append(clipped_x, limit)
        clipped_y = np.append(clipped_y, interpolated)
    return float(
        np.sum(
            0.5 * (clipped_y[1:] + clipped_y[:-1]) * np.diff(clipped_x),
        )
    )


@dataclass(frozen=True)
class StreamingSegmentationResult:
    seg_auroc_0_05: float
    seg_f1: float
    best_threshold: float
    seg_aupro_0_05: float
    prediction_map_count: int
    shape_match_count: int
    nonconstant_prediction_map_count: int
    non_finite_prediction_count: int
    pixel_score_quantization: str = "float16_order_histogram"
    seg_auroc_1_00: float = 0.0
    seg_aupro_0_30: float = 0.0


class StreamingSegmentationAccumulator:
    """Accumulate dataset-level metrics while releasing every dense batch."""

    def __init__(self, *, max_fpr: float = 0.05) -> None:
        if not 0.0 < max_fpr <= 1.0:
            raise ValueError("max_fpr must be in (0, 1]")
        self.max_fpr = float(max_fpr)
        self.positive = np.zeros(_CODE_COUNT, dtype=np.int64)
        self.negative = np.zeros(_CODE_COUNT, dtype=np.int64)
        self.pro_weight = np.zeros(_CODE_COUNT, dtype=np.float64)
        self.region_count = 0
        self.prediction_map_count = 0
        self.shape_match_count = 0
        self.nonconstant_prediction_map_count = 0
        self.non_finite_prediction_count = 0

    @property
    def storage_bytes(self) -> int:
        return int(self.positive.nbytes + self.negative.nbytes + self.pro_weight.nbytes)

    def update(self, score_maps: Sequence[np.ndarray], masks: Sequence[np.ndarray]) -> None:
        if len(score_maps) != len(masks):
            raise ValueError("score maps and masks must align")
        for score, mask in zip(score_maps, masks, strict=True):
            values = np.asarray(score, dtype=np.float32)
            truth = np.asarray(mask, dtype=bool)
            self.prediction_map_count += 1
            if values.ndim != 2 or truth.shape != values.shape:
                raise ValueError("each 2D score map must match its GT mask")
            self.shape_match_count += 1
            if not np.isfinite(values).all():
                self.non_finite_prediction_count += 1
                raise ValueError("streaming metrics require finite score maps")
            if np.ptp(values) > 0:
                self.nonconstant_prediction_map_count += 1
            codes = _ordered_float16_codes(values).reshape(-1)
            flat_truth = truth.reshape(-1)
            self.positive += np.bincount(codes[flat_truth], minlength=_CODE_COUNT)
            self.negative += np.bincount(codes[~flat_truth], minlength=_CODE_COUNT)

            labels, count = _label8(truth)
            self.region_count += count
            if count:
                flat_labels = labels.reshape(-1)
                sizes = np.bincount(flat_labels)
                foreground = flat_labels > 0
                weights = 1.0 / sizes[flat_labels[foreground]]
                self.pro_weight += np.bincount(
                    codes[foreground],
                    weights=weights,
                    minlength=_CODE_COUNT,
                )

    def compute(self) -> StreamingSegmentationResult:
        positive_total = int(self.positive.sum())
        negative_total = int(self.negative.sum())
        if positive_total <= 0 or negative_total <= 0 or self.region_count <= 0:
            raise ValueError("dataset metrics require positive, negative, and region mass")
        active = np.flatnonzero((self.positive + self.negative) > 0)[::-1]
        tp = np.cumsum(self.positive[active], dtype=np.float64)
        fp = np.cumsum(self.negative[active], dtype=np.float64)
        fn = positive_total - tp
        denominator = 2.0 * tp + fp + fn
        f1 = np.divide(2.0 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
        best = int(np.argmax(f1))

        fpr = np.concatenate(([0.0], fp / negative_total))
        tpr = np.concatenate(([0.0], tp / positive_total))
        raw_auc = _bounded_trapezoid(fpr, tpr, self.max_fpr)
        minimum = 0.5 * self.max_fpr**2
        standardized_auc = 0.5 * (1.0 + (raw_auc - minimum) / (self.max_fpr - minimum))

        raw_auc_1_00 = _bounded_trapezoid(fpr, tpr, 1.0)

        pro = np.concatenate(([0.0], np.cumsum(self.pro_weight[active]) / self.region_count))
        aupro = _bounded_trapezoid(fpr, np.clip(pro, 0.0, 1.0), self.max_fpr) / self.max_fpr
        aupro_0_30 = _bounded_trapezoid(fpr, np.clip(pro, 0.0, 1.0), 0.30) / 0.30
        return StreamingSegmentationResult(
            seg_auroc_0_05=float(standardized_auc),
            seg_f1=float(f1[best]),
            best_threshold=_code_to_float16(int(active[best])),
            seg_aupro_0_05=float(aupro),
            prediction_map_count=self.prediction_map_count,
            shape_match_count=self.shape_match_count,
            nonconstant_prediction_map_count=self.nonconstant_prediction_map_count,
            non_finite_prediction_count=self.non_finite_prediction_count,
            pixel_score_quantization="float16_order_histogram",
            seg_auroc_1_00=float(raw_auc_1_00),
            seg_aupro_0_30=float(aupro_0_30),
        )
