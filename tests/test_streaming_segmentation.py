from __future__ import annotations

import numpy as np
import pytest

from fmad.evaluation.streaming import StreamingSegmentationAccumulator


def _fixture() -> tuple[list[np.ndarray], list[np.ndarray]]:
    scores = [
        np.array([[0.0, 0.1, 0.2], [0.1, 0.5, 0.8]], dtype=np.float32),
        np.array([[0.05, 0.1, 0.3], [0.2, 0.7, 0.9]], dtype=np.float32),
        np.array([[0.0, 0.2, 0.1], [0.3, 0.4, 0.2]], dtype=np.float32),
    ]
    masks = [
        np.array([[0, 0, 0], [0, 1, 1]], dtype=bool),
        np.array([[0, 0, 0], [0, 1, 1]], dtype=bool),
        np.zeros((2, 3), dtype=bool),
    ]
    return scores, masks


def test_minibatch_updates_equal_single_batch_dataset_metrics() -> None:
    scores, masks = _fixture()
    one = StreamingSegmentationAccumulator()
    one.update(scores, masks)
    streamed = StreamingSegmentationAccumulator()
    streamed.update(scores[:2], masks[:2])
    streamed.update(scores[2:], masks[2:])
    assert streamed.compute() == one.compute()
    assert streamed.storage_bytes == 3 * 65536 * 8


def test_float16_histogram_matches_reference_metrics() -> None:
    metrics = pytest.importorskip("sklearn.metrics")
    from scipy.ndimage import label

    scores, masks = _fixture()
    accumulator = StreamingSegmentationAccumulator(max_fpr=0.05)
    accumulator.update(scores, masks)
    observed = accumulator.compute()

    flat_scores = np.concatenate([x.astype(np.float16).astype(np.float32).ravel() for x in scores])
    flat_masks = np.concatenate([x.ravel() for x in masks])
    expected_auc = metrics.roc_auc_score(flat_masks, flat_scores, max_fpr=0.05)
    precision, recall, _ = metrics.precision_recall_curve(flat_masks, flat_scores)
    expected_f1 = np.max(2.0 * precision * recall / np.maximum(precision + recall, 1e-12))
    # Independent dense, threshold-by-threshold region-overlap reference.
    quantized = [x.astype(np.float16).astype(np.float32) for x in scores]
    regions = [label(mask, structure=np.ones((3, 3)))[0] for mask in masks]
    fpr, pro = [0.0], [0.0]
    for threshold in sorted(set(flat_scores.tolist()), reverse=True):
        predicted = [x >= threshold for x in quantized]
        fpr.append(
            sum(np.count_nonzero(p & ~m) for p, m in zip(predicted, masks))
            / np.count_nonzero(~flat_masks)
        )
        overlaps = [
            np.mean(p[r == component])
            for p, r in zip(predicted, regions)
            for component in range(1, int(r.max()) + 1)
        ]
        pro.append(float(np.mean(overlaps)))
    # Include vertical segments at repeated FPR before interpolating the boundary.
    area = 0.0
    for left, right, low, high in zip(fpr[:-1], fpr[1:], pro[:-1], pro[1:]):
        if left >= 0.05:
            break
        endpoint = min(right, 0.05)
        end_height = (
            high if right == left else low + (high - low) * (endpoint - left) / (right - left)
        )
        area += (endpoint - left) * (low + end_height) / 2
    expected_aupro = area / 0.05
    assert observed.seg_auroc_0_05 == pytest.approx(expected_auc)
    assert observed.seg_f1 == pytest.approx(expected_f1)
    assert observed.seg_aupro_0_05 == pytest.approx(expected_aupro)


def test_dense_maps_are_not_retained_by_accumulator() -> None:
    scores, masks = _fixture()
    accumulator = StreamingSegmentationAccumulator()
    accumulator.update(scores, masks)
    assert not hasattr(accumulator, "score_maps")
    assert not hasattr(accumulator, "masks")
