"""Label-free score-map transforms shared by predictors and offline audits."""
from __future__ import annotations

import numpy as np


def component_quantile(
    score_map: np.ndarray,
    percentile: float = 98.0,
    q: float = 0.9,
    dilate_size: int = 0,
) -> np.ndarray:
    """Fill each high-score component with its own selected quantile.

    The threshold is computed independently for each query map. Pixels below
    it retain their original value, so the transform needs neither support nor
    evaluation labels.
    """
    import cv2

    scores = np.asarray(score_map, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError(f"component_quantile expects a 2D map, got {scores.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("component_quantile requires finite scores")
    if not 0.0 <= float(percentile) <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    if not 0.0 <= float(q) <= 1.0:
        raise ValueError("q must be in [0, 1]")
    if isinstance(dilate_size, bool) or int(dilate_size) != dilate_size or dilate_size < 0:
        raise ValueError("dilate_size must be a non-negative integer")

    threshold = float(np.percentile(scores, percentile))
    mask = (scores >= threshold).astype(np.uint8)
    if int(dilate_size) > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (int(dilate_size), int(dilate_size))
        )
        mask = cv2.dilate(mask, kernel)
    count, labels = cv2.connectedComponents(mask, connectivity=8)
    result = scores.copy()
    if count <= 1:
        return result
    flat_labels = labels.reshape(-1)
    flat_scores = scores.reshape(-1)
    order = np.argsort(flat_labels, kind="stable")
    sorted_labels = flat_labels[order]
    bounds = np.searchsorted(sorted_labels, np.arange(count + 1))
    flat_result = result.reshape(-1)
    for component in range(1, count):
        members = order[bounds[component]:bounds[component + 1]]
        if members.size:
            flat_result[members] = np.quantile(flat_scores[members], float(q))
    return flat_result.reshape(scores.shape).astype(np.float32)
