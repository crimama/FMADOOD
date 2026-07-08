# /// script
# requires-python = ">=3.8"
# dependencies = []
# ///
from __future__ import annotations

from typing import Dict, Final, Sequence, Tuple, cast

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
_EPS: Final = 1e-12


def normalize_minmax(values: FloatArray) -> FloatArray:
    array = values.astype(np.float32, copy=False)
    low = float(np.min(array))
    high = float(np.max(array))
    if high - low <= _EPS:
        return np.zeros_like(array, dtype=np.float32)
    return cast("FloatArray", ((array - low) / (high - low)).astype(np.float32, copy=False))


def safe_corrcoef(left: FloatArray, right: FloatArray) -> float:
    left_flat = left.astype(np.float32, copy=False).reshape(-1)
    right_flat = right.astype(np.float32, copy=False).reshape(-1)
    if left_flat.size != right_flat.size:
        raise RuntimeError("safe_corrcoef expects equal-sized arrays")
    if left_flat.size < 2:
        return float("nan")
    left_std = float(np.std(left_flat))
    right_std = float(np.std(right_flat))
    if left_std <= _EPS or right_std <= _EPS:
        return float("nan")
    return float(np.corrcoef(left_flat, right_flat)[0, 1])


def high_mask(values: FloatArray, percentile: float) -> npt.NDArray[np.bool_]:
    if percentile <= 0.0 or percentile >= 100.0:
        raise RuntimeError("percentile must be in (0, 100)")
    threshold = float(np.percentile(values.astype(np.float32, copy=False), percentile))
    return cast("npt.NDArray[np.bool_]", values >= threshold)


def high_region_share(values: FloatArray, reference: FloatArray, percentile: float) -> float:
    mask = high_mask(values, percentile)
    reference_mask = high_mask(reference, percentile)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.mean(reference_mask[mask]))


def low_rank_energy_summary(values: FloatArray, ranks: Sequence[int]) -> Dict[str, float]:
    matrix = values.reshape(-1, values.shape[-1]).astype(np.float32, copy=False)
    if matrix.shape[0] == 0:
        raise RuntimeError("low_rank_energy_summary expects non-empty values")
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = np.square(singular_values.astype(np.float64, copy=False))
    total = float(np.sum(energy))
    result: Dict[str, float] = {
        "rank_count": float(singular_values.size),
        "energy_total": total,
    }
    safe_total = max(total, _EPS)
    for rank in ranks:
        capped = min(max(1, int(rank)), int(energy.size))
        result[f"top{rank}_energy_share"] = float(np.sum(energy[:capped]) / safe_total)
    weights = energy / safe_total
    entropy = -float(np.sum(weights * np.log(np.maximum(weights, _EPS))))
    result["effective_rank"] = float(np.exp(entropy))
    return result


def top_percent_mean(values: FloatArray, top_percent: float) -> float:
    if top_percent <= 0.0 or top_percent > 1.0:
        raise RuntimeError("top_percent must be in (0, 1]")
    flat = values.reshape(-1).astype(np.float32, copy=False)
    if flat.size == 0:
        return float("nan")
    count = max(1, int(flat.size * top_percent))
    partition = np.partition(flat, flat.size - count)
    return float(np.mean(partition[-count:]))


def summarize_values(values: Sequence[float]) -> Tuple[int, float, float, float, float]:
    array = np.asarray(tuple(values), dtype=np.float32)
    if array.size == 0:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")
    return (
        int(array.size),
        float(np.mean(array)),
        float(np.std(array)),
        float(np.percentile(array, 50)),
        float(np.percentile(array, 95)),
    )
