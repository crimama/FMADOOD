from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from typing_extensions import final, override

from flow_tte.config import FlowTTEConfig
from flow_tte.metrics import (
    BoolArray,
    FloatArray,
    MetricConfig,
    MetricInputs,
    MetricScores,
    compute_ad_metrics,
)
from flow_tte.pipeline import BatchResult, FlowTTE
from flow_tte.tensors import FeatureArray


@dataclass(frozen=True)
class EvaluationInputError(ValueError):
    reason: str

    @override
    def __str__(self) -> str:
        return f"Invalid evaluation inputs: {self.reason}"


@dataclass(frozen=True)
@final
class EvaluationBatch:
    features: FeatureArray
    image_labels: BoolArray
    pixel_masks: Optional[BoolArray] = None


@dataclass(frozen=True)
@final
class EvaluationConfig:
    pipeline_config: FlowTTEConfig = field(default_factory=FlowTTEConfig)
    metric_config: MetricConfig = field(default_factory=MetricConfig)
    expand: bool = True


@dataclass(frozen=True)
@final
class EvaluationResult:
    metrics: MetricScores
    image_scores: FloatArray
    image_labels: BoolArray
    pixel_scores: Optional[FloatArray]
    pixel_masks: Optional[BoolArray]
    batch_results: Tuple[BatchResult, ...]
    memory_sizes_before: Tuple[int, ...]
    memory_sizes_after: Tuple[int, ...]


def evaluate_flow_tte(
    support_features: FeatureArray,
    batches: Sequence[EvaluationBatch],
    config: Optional[EvaluationConfig] = None,
) -> EvaluationResult:
    active_config = config if config is not None else EvaluationConfig()
    if not batches:
        raise EvaluationInputError("batches must contain at least one test batch")
    has_masks = [batch.pixel_masks is not None for batch in batches]
    if any(has_masks) and not all(has_masks):
        raise EvaluationInputError("either all batches or no batches must provide pixel_masks")

    pipeline = FlowTTE(active_config.pipeline_config)
    _ = pipeline.fit(support_features)

    batch_results: List[BatchResult] = []
    image_scores: List[FloatArray] = []
    image_labels: List[BoolArray] = []
    pixel_scores: List[FloatArray] = []
    pixel_masks: List[BoolArray] = []
    memory_sizes_before: List[int] = []
    memory_sizes_after: List[int] = []

    for batch in batches:
        if active_config.expand:
            result = pipeline.score_then_expand(batch.features)
        else:
            result = pipeline.score_static(batch.features)
        labels = _image_labels(batch.image_labels, len(result.image_scores))
        mask = _pixel_masks(batch.pixel_masks, result.patch_scores.shape)
        batch_results.append(result)
        image_scores.append(result.image_scores)
        image_labels.append(labels)
        pixel_scores.append(result.patch_scores)
        if mask is not None:
            pixel_masks.append(mask)
        memory_sizes_before.append(result.memory_size_before)
        memory_sizes_after.append(result.memory_size_after)

    all_image_scores = _concat_float(image_scores)
    all_image_labels = _concat_bool(image_labels)
    all_pixel_scores = _concat_float(pixel_scores) if pixel_scores else None
    all_pixel_masks = _concat_bool(pixel_masks) if pixel_masks else None
    metrics = compute_ad_metrics(
        inputs=MetricInputs(
            image_scores=all_image_scores,
            image_labels=all_image_labels,
            pixel_scores=all_pixel_scores,
            pixel_masks=all_pixel_masks,
        ),
        config=active_config.metric_config,
    )
    return EvaluationResult(
        metrics=metrics,
        image_scores=all_image_scores,
        image_labels=all_image_labels,
        pixel_scores=all_pixel_scores,
        pixel_masks=all_pixel_masks,
        batch_results=tuple(batch_results),
        memory_sizes_before=tuple(memory_sizes_before),
        memory_sizes_after=tuple(memory_sizes_after),
    )


def _image_labels(labels: BoolArray, expected: int) -> BoolArray:
    raw: BoolArray = np.asarray(labels, dtype=np.bool_)
    array: BoolArray = raw.reshape(-1)
    if len(array) != expected:
        raise EvaluationInputError("image_labels length must match FlowTTE image_scores length")
    return array


def _pixel_masks(
    masks: Optional[BoolArray],
    expected_shape: Tuple[int, ...],
) -> Optional[BoolArray]:
    if masks is None:
        return None
    array: BoolArray = np.asarray(masks, dtype=np.bool_)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.shape != expected_shape:
        raise EvaluationInputError("pixel_masks shape must match FlowTTE patch_scores shape")
    return array


def _concat_float(arrays: Sequence[FloatArray]) -> FloatArray:
    result: FloatArray = np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
    return result


def _concat_bool(arrays: Sequence[BoolArray]) -> BoolArray:
    result: BoolArray = np.concatenate(arrays, axis=0).astype(np.bool_, copy=False)
    return result
