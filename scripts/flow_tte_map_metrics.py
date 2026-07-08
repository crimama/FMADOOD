from __future__ import annotations

# pyright: reportMissingImports=false
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Final, List, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

from flow_tte.metrics import average_precision, binary_auroc

_HISTOGRAM_BINS: Final = 65536
_DEFAULT_IMAGE_TOP_FRACTION: Final = 0.01
_PIXEL_PRO_MAX_FPR: Final = 0.30

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
Float64Array = npt.NDArray[np.float64]
JsonScalar = Union[str, int, float, bool, None]
JsonValue = Union[JsonScalar, Dict[str, "JsonValue"], List["JsonValue"]]


@dataclass(frozen=True)
class MapMetricInputError(ValueError):
    reason: str

    def __str__(self) -> str:
        return f"Invalid map metric inputs: {self.reason}"


@dataclass(frozen=True)
class MapMetricSet:
    image_auroc: float
    pixel_auroc: float
    image_ap: float
    pixel_ap: float
    pixel_pro: float
    image_score_aggregation: str
    pixel_score_quantization: str
    pixel_pro_max_fpr: float

    def as_dict(self) -> Dict[str, JsonValue]:
        return {
            "image_AUROC": self.image_auroc,
            "pixel_AUROC": self.pixel_auroc,
            "image_AP": self.image_ap,
            "pixel_AP": self.pixel_ap,
            "pixel_PRO": self.pixel_pro,
            "image_score_aggregation": self.image_score_aggregation,
            "pixel_score_quantization": self.pixel_score_quantization,
            "pixel_PRO_max_fpr": self.pixel_pro_max_fpr,
        }


def compute_map_metric_set(
    gt_filenames: Sequence[Optional[str]],
    prediction_filenames: Sequence[str],
    image_top_fraction: float = _DEFAULT_IMAGE_TOP_FRACTION,
) -> MapMetricSet:
    if len(gt_filenames) != len(prediction_filenames):
        raise MapMetricInputError("GT and prediction lists must have the same length")
    if not 0.0 < image_top_fraction <= 1.0:
        raise MapMetricInputError("image_top_fraction must be in (0, 1]")
    image_scores: List[float] = []
    image_labels: List[bool] = []
    pro_predictions: List[FloatArray] = []
    pro_masks: List[BoolArray] = []
    total_counts = np.zeros(_HISTOGRAM_BINS, dtype=np.float64)
    positive_counts = np.zeros(_HISTOGRAM_BINS, dtype=np.float64)
    for index, prediction_name in enumerate(prediction_filenames):
        prediction = _read_tiff_without_ext(prediction_name)
        gt_filename = gt_filenames[index]
        gt_mask = _read_gt_mask(gt_filename, prediction.shape)
        image_scores.append(_mean_top_fraction(prediction, image_top_fraction))
        image_labels.append(bool(np.any(gt_mask)))
        pro_predictions.append(prediction)
        pro_masks.append(gt_mask)
        codes = np.asarray(prediction, dtype=np.float16).ravel().view(np.uint16)
        positives = gt_mask.reshape(-1).astype(np.float64, copy=False)
        total_counts += np.bincount(codes, minlength=_HISTOGRAM_BINS)
        positive_counts += np.bincount(codes, weights=positives, minlength=_HISTOGRAM_BINS)
    histogram = histogram_binary_metrics(total_counts, positive_counts)
    return MapMetricSet(
        image_auroc=binary_auroc(
            np.asarray(image_labels, dtype=np.bool_),
            np.asarray(image_scores, dtype=np.float32),
        ),
        image_ap=average_precision(
            np.asarray(image_labels, dtype=np.bool_),
            np.asarray(image_scores, dtype=np.float32),
        ),
        pixel_auroc=histogram.pixel_auroc,
        pixel_ap=histogram.pixel_ap,
        pixel_pro=pixel_pro_score(pro_predictions, pro_masks, _PIXEL_PRO_MAX_FPR),
        image_score_aggregation=_image_score_aggregation_name(image_top_fraction),
        pixel_score_quantization="float16_histogram",
        pixel_pro_max_fpr=_PIXEL_PRO_MAX_FPR,
    )


def _read_tiff_without_ext(prediction_name: str) -> FloatArray:
    import tifffile as tiff  # noqa: PLC0415

    path = Path(f"{prediction_name}.tiff")
    if not path.is_file():
        message = f"Prediction TIFF not found: {path}"
        raise MapMetricInputError(message)
    return np.asarray(tiff.imread(path), dtype=np.float32)


def _read_gt_mask(gt_filename: Optional[str], shape: Sequence[int]) -> BoolArray:
    from PIL import Image  # noqa: PLC0415

    if gt_filename is None:
        return np.zeros(tuple(shape), dtype=np.bool_)
    mask = np.asarray(Image.open(gt_filename)) > 0
    if tuple(mask.shape) != tuple(shape):
        message = f"GT shape {mask.shape} does not match prediction shape {tuple(shape)}"
        raise MapMetricInputError(message)
    return np.asarray(mask, dtype=np.bool_)


def _mean_top_fraction(values: FloatArray, fraction: float) -> float:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    top_count = max(1, math.ceil(flat.size * fraction))
    threshold_index = flat.size - top_count
    return float(np.partition(flat, threshold_index)[threshold_index:].mean())


def _image_score_aggregation_name(fraction: float) -> str:
    percent_text = f"{fraction * 100:g}".replace(".", "p")
    return f"mean_top_{percent_text}_percent_full_resolution_map"


def pixel_pro_score(
    predictions: Sequence[FloatArray],
    masks: Sequence[BoolArray],
    max_fpr: float,
) -> float:
    from src.post_eval import compute_pro, trapezoid  # noqa: PLC0415

    fprs, pros = compute_pro(
        anomaly_maps=[np.asarray(prediction, dtype=np.float32) for prediction in predictions],
        ground_truth_maps=[np.asarray(mask, dtype=np.bool_) for mask in masks],
    )
    return float(trapezoid(fprs, pros, x_max=max_fpr) / max_fpr)


@dataclass(frozen=True)
class HistogramBinaryMetrics:
    pixel_auroc: float
    pixel_ap: float


def histogram_binary_metrics(
    total_counts: Float64Array,
    positive_counts: Float64Array,
) -> HistogramBinaryMetrics:
    positives = float(np.sum(positive_counts))
    total = float(np.sum(total_counts))
    negatives = total - positives
    if positives <= 0.0 or negatives <= 0.0:
        return HistogramBinaryMetrics(pixel_auroc=float("nan"), pixel_ap=float("nan"))
    present = total_counts > 0
    codes = np.flatnonzero(present).astype(np.uint16, copy=False)
    scores = codes.view(np.float16).astype(np.float32, copy=False)
    ascending_order = np.argsort(scores, kind="mergesort")
    total_present = total_counts[present][ascending_order]
    positive_present = positive_counts[present][ascending_order]
    cumulative_before = np.r_[0.0, np.cumsum(total_present[:-1])]
    average_ranks = cumulative_before + (total_present + 1.0) / 2.0
    positive_rank_sum = float(np.sum(positive_present * average_ranks))
    auroc = (positive_rank_sum - positives * (positives + 1.0) / 2.0) / (
        positives * negatives
    )
    descending_positive = positive_present[::-1]
    descending_total = total_present[::-1]
    cumulative_tp = np.cumsum(descending_positive)
    cumulative_total = np.cumsum(descending_total)
    precision = np.divide(
        cumulative_tp,
        cumulative_total,
        out=np.zeros_like(cumulative_tp),
        where=cumulative_total > 0,
    )
    ap = float(np.sum((descending_positive / positives) * precision))
    return HistogramBinaryMetrics(pixel_auroc=float(auroc), pixel_ap=ap)
