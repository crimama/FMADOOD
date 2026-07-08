# /// script
# requires-python = ">=3.8"
# dependencies = []
# ///
# pyright: reportMissingImports=false
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple

import numpy as np
import torch
from dinov3_backbone import DINOv3Backbone
from flow_tte_mvtec_ad2_core import DatasetLike
from flow_tte_register_analysis_extract import build_support, extract_bundle, stream_test_images
from flow_tte_register_analysis_rows import build_context_rows, build_nf_rows, build_retrieval_rows
from flow_tte_register_analysis_types import (
    CONTEXT_SOURCES,
    AnalysisConfig,
    ContextSource,
    FloatArray,
    SplitName,
    context_values,
    cosine_distance_matrix,
)

from flow_tte.config import ExpansionConfig, FlowConfig, FlowTTEConfig, ScoreConfig
from flow_tte.pipeline import FlowTTE
from flow_tte.tensors import to_numpy
from flow_tte.trainer import FlowDensityEstimator

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class RetrievalInputs:
    config: AnalysisConfig
    support_contexts: FloatArray
    query_context: FloatArray
    source: ContextSource
    split: SplitName
    support_group_array: np.ndarray
    distances: torch.Tensor
    all_values: torch.Tensor
    all_groups: np.ndarray


def make_pipeline(config: AnalysisConfig, condition_mode: str) -> FlowTTE:
    return FlowTTE(
        FlowTTEConfig(
            flow=FlowConfig(
                n_coupling_layers=2,
                hidden_multiplier=1,
                condition_mode=condition_mode,
                n_epochs=3,
                lr=2e-4,
                clamp=1.9,
                tail_weight=0.3,
                tail_top_k_ratio=0.05,
                lambda_logdet=1e-3,
                batch_size=512,
                seed=config.seed,
            ),
            expansion=ExpansionConfig(budget=1.0, density_quantile=0.90, random_seed=config.seed),
            score=ScoreConfig(density_weight=0.25, query_chunk_size=512),
            device=config.device,
        ),
    )


def require_estimator(pipeline: FlowTTE) -> FlowDensityEstimator:
    estimator = pipeline.estimator
    if estimator is None:
        message = "FlowTTE estimator is unavailable before fit"
        raise RuntimeError(message)
    return estimator


def tensor_from_array(values: FloatArray, device: str) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device=torch.device(device))


def analyze_object(
    config: AnalysisConfig,
    dataset: DatasetLike,
    backbone: DINOv3Backbone,
    support_paths: Sequence[Path],
    object_name: str,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    info = dataset.get_object_info(object_name)
    backbone.set_resolution(info.resolution)
    support = build_support(backbone, support_paths)
    noctx = make_pipeline(config, "none")
    _ = noctx.fit(support.features)
    cond = make_pipeline(config, "context")
    _ = cond.fit(support.features, support_contexts=support.register_patch_contexts)
    support_z = noctx.transform_features(support.features).astype(np.float32, copy=False)
    support_z_cond = cond.transform_features(
        support.features,
        contexts=support.register_patch_contexts,
    ).astype(np.float32, copy=False)
    support_z_tensor = tensor_from_array(support_z, config.device)
    rng = np.random.default_rng(config.seed + sum(ord(char) for char in object_name))
    context_min: Dict[Tuple[ContextSource, SplitName], List[float]] = {}
    context_mean: Dict[Tuple[ContextSource, SplitName], List[float]] = {}
    retrieval: Dict[Tuple[ContextSource, SplitName], Dict[str, List[float]]] = {}
    nll_values: Dict[Tuple[str, SplitName], List[float]] = {}

    for item in stream_test_images(dataset, object_name):
        bundle = extract_bundle(backbone, item.path)
        sample_count = min(config.patch_samples_per_image, int(bundle.features.shape[0]))
        sample_indices = rng.choice(bundle.features.shape[0], size=sample_count, replace=False)
        sample_features = bundle.features[sample_indices]
        sample_register = np.broadcast_to(
            bundle.contexts.register,
            (sample_count, bundle.contexts.register.size),
        ).copy()
        noctx_eval = require_estimator(noctx).evaluate(sample_features)
        cond_eval = require_estimator(cond).evaluate(sample_features, contexts=sample_register)
        append_nll(nll_values, "unconditional", item.split, to_numpy(noctx_eval.nll))
        append_nll(nll_values, "register_conditional", item.split, to_numpy(cond_eval.nll))
        distances = torch.cdist(noctx_eval.z, support_z_tensor, p=2.0)
        all_values, all_indices = torch.min(distances, dim=1)
        all_groups = support.group_ids[to_numpy(all_indices).astype(np.int64, copy=False)]

        for source in CONTEXT_SOURCES:
            update_context_and_retrieval(
                inputs=RetrievalInputs(
                    config=config,
                    support_contexts=context_values(support.contexts, source),
                    query_context=context_values(bundle.contexts, source).reshape(1, -1),
                    source=source,
                    split=item.split,
                    support_group_array=support.group_ids,
                    distances=distances,
                    all_values=all_values,
                    all_groups=all_groups,
                ),
                context_min=context_min,
                context_mean=context_mean,
                retrieval=retrieval,
            )

    return (
        build_context_rows(object_name, context_min, context_mean),
        build_retrieval_rows(object_name, config.context_top_m, retrieval),
        build_nf_rows(object_name, support_z, support_z_cond, nll_values),
    )


def append_nll(
    nll_values: Dict[Tuple[str, SplitName], List[float]],
    model_name: str,
    split: SplitName,
    values: FloatArray,
) -> None:
    nll_values.setdefault((model_name, split), []).extend(float(value) for value in values)


def update_context_and_retrieval(
    inputs: RetrievalInputs,
    context_min: Dict[Tuple[ContextSource, SplitName], List[float]],
    context_mean: Dict[Tuple[ContextSource, SplitName], List[float]],
    retrieval: Dict[Tuple[ContextSource, SplitName], Dict[str, List[float]]],
) -> None:
    context_distances = cosine_distance_matrix(inputs.query_context, inputs.support_contexts)[0]
    context_min.setdefault((inputs.source, inputs.split), []).append(
        float(np.min(context_distances)),
    )
    context_mean.setdefault((inputs.source, inputs.split), []).append(
        float(np.mean(context_distances)),
    )
    top_groups = np.argsort(context_distances)[: inputs.config.context_top_m].astype(np.int64)
    allowed = torch.as_tensor(
        np.isin(inputs.support_group_array, top_groups),
        dtype=torch.bool,
        device=inputs.config.device,
    )
    routed = inputs.distances.masked_fill(~allowed.reshape(1, -1), torch.inf)
    routed_values, _ = torch.min(routed, dim=1)
    bucket = retrieval.setdefault(
        (inputs.source, inputs.split),
        {"all": [], "routed": [], "inflation": [], "retained": []},
    )
    all_np = to_numpy(inputs.all_values)
    routed_np = to_numpy(routed_values)
    retained = np.isin(inputs.all_groups, top_groups).astype(np.float32, copy=False)
    bucket["all"].extend(float(value) for value in all_np)
    bucket["routed"].extend(float(value) for value in routed_np)
    bucket["inflation"].extend(float(value) for value in routed_np - all_np)
    bucket["retained"].extend(float(value) for value in retained)
