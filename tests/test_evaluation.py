from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt
import pytest

from flow_tte.config import ExpansionConfig, FlowConfig, FlowTTEConfig, ScoreConfig
from flow_tte.evaluation import (
    EvaluationBatch,
    EvaluationConfig,
    EvaluationInputError,
    evaluate_flow_tte,
)
from flow_tte.metrics import MetricConfig, MetricInputs, compute_ad_metrics

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


def normal_array(
    rng: np.random.Generator,
    loc: float,
    scale: float,
    size: Tuple[int, ...],
) -> FloatArray:
    return rng.normal(loc=loc, scale=scale, size=size).astype(np.float32)


def test_compute_ad_metrics_returns_tte_metric_set() -> None:
    image_scores: FloatArray = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32)
    image_labels: BoolArray = np.array([0, 1, 0, 1], dtype=np.bool_)
    pixel_scores: FloatArray = np.array(
        [
            [[0.1, 0.2], [0.0, 0.1]],
            [[0.1, 0.9], [0.2, 0.8]],
            [[0.2, 0.1], [0.0, 0.1]],
            [[0.9, 0.8], [0.1, 0.2]],
        ],
        dtype=np.float32,
    )
    pixel_masks: BoolArray = np.array(
        [
            [[0, 0], [0, 0]],
            [[0, 1], [0, 1]],
            [[0, 0], [0, 0]],
            [[1, 1], [0, 0]],
        ],
        dtype=np.bool_,
    )

    metrics = compute_ad_metrics(
        inputs=MetricInputs(
            image_scores=image_scores,
            image_labels=image_labels,
            pixel_scores=pixel_scores,
            pixel_masks=pixel_masks,
        ),
        config=MetricConfig(aupro_thresholds=32),
    )

    assert metrics.i_auroc == 1.0
    assert metrics.i_ap == 1.0
    assert metrics.i_f1_max == 1.0
    assert metrics.p_auroc == 1.0
    assert metrics.p_ap == 1.0
    assert metrics.p_f1_max == 1.0
    assert metrics.aupro > 0.99


def test_evaluate_flow_tte_scores_batches_and_computes_metrics() -> None:
    rng = np.random.default_rng(31)
    support = normal_array(rng, loc=0.0, scale=0.25, size=(2, 4, 4, 4))
    test_features = normal_array(rng, loc=0.0, scale=0.25, size=(2, 4, 4, 4))
    test_features[1, 1:3, 1:3, :] = normal_array(rng, loc=4.0, scale=0.1, size=(2, 2, 4))
    image_labels: BoolArray = np.array([0, 1], dtype=np.bool_)
    masks: BoolArray = np.zeros((2, 4, 4), dtype=np.bool_)
    masks[1, 1:3, 1:3] = True
    config = FlowTTEConfig(
        flow=FlowConfig(n_coupling_layers=2, n_epochs=2, batch_size=16, seed=11),
        expansion=ExpansionConfig(budget=1.25, density_quantile=0.95, random_seed=5),
        score=ScoreConfig(density_weight=0.2, top_percent=0.25, query_chunk_size=8),
        device="cpu",
    )

    result = evaluate_flow_tte(
        support_features=support,
        batches=[
            EvaluationBatch(
                features=test_features,
                image_labels=image_labels,
                pixel_masks=masks,
            ),
        ],
        config=EvaluationConfig(pipeline_config=config, expand=False),
    )

    assert result.pixel_scores is not None
    assert result.pixel_scores.shape == (2, 4, 4)
    assert result.image_scores.shape == (2,)
    assert result.image_scores[1] > result.image_scores[0]
    assert result.metrics.i_auroc == 1.0
    assert result.metrics.p_auroc > 0.95
    assert result.memory_sizes_before == (32,)


def test_evaluate_flow_tte_rejects_partial_pixel_masks() -> None:
    rng = np.random.default_rng(9)
    support = normal_array(rng, loc=0.0, scale=0.1, size=(1, 2, 2, 4))
    labels: BoolArray = np.array([0], dtype=np.bool_)
    partial_masks: BoolArray = np.zeros((1, 2, 2), dtype=np.bool_)

    with pytest.raises(EvaluationInputError):
        evaluate_flow_tte(
            support_features=support,
            batches=(
                EvaluationBatch(
                    features=support,
                    image_labels=labels,
                    pixel_masks=partial_masks,
                ),
                EvaluationBatch(features=support, image_labels=labels),
            ),
            config=EvaluationConfig(
                pipeline_config=FlowTTEConfig.for_quick_probe(),
                expand=False,
            ),
        )
