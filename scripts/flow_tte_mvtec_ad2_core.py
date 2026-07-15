from __future__ import annotations

# allow: SIZE_OK — remote experiment core keeps protocol args and artifact writing together
# until the MVTec AD2 runner interface stabilizes.
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Protocol, Sequence, Tuple, Union, cast

import cv2
import numpy as np
import tifffile as tiff
import torch

from flow_tte.config import ExpansionConfig, FlowConfig, FlowTTEConfig, ScoreConfig
from flow_tte.conv2d_flow import Conv2DFlowDensityEstimator
from flow_tte.denoising import PositionMeanArtifactDenoiser, fit_feature_denoiser
from flow_tte.memory import TorchMemoryBank
from flow_tte.pipeline import FlowTTE
from flow_tte.shift_projection import fit_shift_projection
from flow_tte.scoring import ScoreCalibration, ScoreInputs, score_flow_memory
from flow_tte.superadd_parity import subsample_knn_score_rank
from flow_tte.trainer import FlowDensityEstimator
from flow_tte.transformer_flow import TransformerFlowDensityEstimator

if __package__:
    from .flow_tte_raw_nn import (
        ForegroundSplitConfig,
        ForegroundSplitRawNNState,
        RawNNResult,
        RawNNState,
        fit_foreground_split_raw_nn,
        fit_raw_nn,
        score_foreground_split_raw_nn,
        score_raw_nn,
    )
    from .flow_tte_score_field import (
        ScoreFieldConfig,
        ScoreFieldStats,
        apply_score_field_transform,
        fit_score_field_stats,
        support_leave_one_out_patch_scores,
    )
    from .flow_tte_score_priors import rgb_foreground_proxy
    from .flow_tte_superadd_preprocess import (
        BrightnessRange,
        TilingConfig,
        apply_brightness,
        resize_rgb,
        tile_starts,
    )
    from .flow_tte_support import (
        merge_layer_features,
        normalize_layer_features,
        read_rgb,
        select_support_paths_for_backbone,
        select_superadd_threshold_paths,
        transform_rgb,
    )
else:
    from flow_tte_raw_nn import (
        ForegroundSplitConfig,
        ForegroundSplitRawNNState,
        RawNNResult,
        RawNNState,
        fit_foreground_split_raw_nn,
        fit_raw_nn,
        score_foreground_split_raw_nn,
        score_raw_nn,
    )
    from flow_tte_score_field import (
        ScoreFieldConfig,
        ScoreFieldStats,
        apply_score_field_transform,
        fit_score_field_stats,
        support_leave_one_out_patch_scores,
    )
    from flow_tte_score_priors import rgb_foreground_proxy
    from flow_tte_superadd_preprocess import (
        BrightnessRange,
        TilingConfig,
        apply_brightness,
        resize_rgb,
        tile_starts,
    )
    from flow_tte_support import (
        merge_layer_features,
        normalize_layer_features,
        read_rgb,
        select_support_paths_for_backbone,
        select_superadd_threshold_paths,
        transform_rgb,
    )


class ObjectInfoLike(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def resolution(self) -> int: ...


class DatasetLike(Protocol):
    @property
    def data_root(self) -> str: ...

    def get_object_info(self, object_name: str) -> ObjectInfoLike: ...

    def get_train_images(self, object_name: str) -> List[str]: ...

    def get_test_images(
        self,
        object_name: str,
        split: str = "test_public",
    ) -> Dict[str, List[str]]: ...


class BackboneLike(Protocol):
    def set_resolution(self, smaller_edge_size: int) -> None: ...

    def prepare_image(self, img: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]: ...

    def extract_features(self, image_tensor: torch.Tensor) -> List[np.ndarray]: ...

    def extract_cls_features(self, image_tensor: torch.Tensor) -> torch.Tensor: ...

    def extract_context_features(
        self,
        image_tensor: torch.Tensor,
        context_source: str,
    ) -> torch.Tensor: ...

    def extract_context_token_features(
        self,
        image_tensor: torch.Tensor,
        context_source: str,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class RunConfig:
    data_root: Path
    output_root: Path
    project_root: Path
    fsad_root: Path
    objects: Tuple[str, ...]
    shots: int
    seed: int
    device: str
    flow_epochs: int
    coupling_layers: int
    hidden_multiplier: int
    flow_lr: float
    flow_clamp: float
    tail_weight: float
    tail_top_k_ratio: float
    lambda_logdet: float
    density_quantile: float
    expansion_budget: float
    distance_weight: float
    density_weight: float
    top_percent: float
    query_chunk_size: int
    calibration_sample_size: int
    loo_standardize: bool
    pro_integration_limit: float
    cleanup_maps: bool
    dataset_kind: str = "mvtec_ad2"
    latent_bank_subsample: str = "none"
    latent_bank_target_count: int = 100_000
    rgb_guide: str = "guided_r8"
    threshold_calibration_mode: str = "none"
    threshold_fraction: int = 8
    threshold_percentile: float = 95.0
    threshold_factor: float = 1.421
    binary_postprocess: str = "closefill_erode"
    morphology_line_length: int = 17
    morphology_angle_count: int = 16
    flow_transform_mode: str = "flow"
    score_mode: str = "latent_distance"
    use_squared_distance: bool = False
    flow_condition_mode: str = "none"
    context_source: str = "none"
    flow_context_source: str = "auto"
    memory_context_source: str = "auto"
    context_mode: str = "auto"
    context_weight: float = 0.0
    context_top_m: int = 1
    transformer_context_mode: Literal[
        "none",
        "cls",
        "register",
        "cls_register",
        "random_dummy",
        "learned_dummy",
    ] = "none"
    score_field_calibration_mode: Literal[
        "none",
        "local_contrast",
        "support_position_center",
        "support_position_zscore",
        "support_score_reliability",
    ] = "none"
    score_field_calibration_alpha: float = 1.0
    score_field_position_std_floor: float = 0.25
    score_field_foreground_mode: Literal[
        "none",
        "support_feature_energy",
        "support_rgb_contrast",
        "support_rgb_feature_product",
    ] = "none"
    score_field_foreground_quantile: float = 0.20
    score_field_background_multiplier: float = 0.50
    score_field_foreground_smooth_kernel: int = 5
    score_field_support_score_quantile: float = 0.90
    backbone_model: str = "dinov2_vitl14"
    backbone_resolution: Optional[int] = None
    preprocess_recipe: str = "fmad_shorter_edge"
    image_size: int = 448
    crop_size: int = 448
    feature_layers: Tuple[int, ...] = (5, 11, 17, 23)
    tile_patch_size: int = 0
    tile_overlap: int = 0
    image_resize_factor: float = 1.0
    feature_fusion: str = "layer_norm_mean"
    support_selection: str = "first"
    support_selection_seed: int = 0
    support_transforms: Tuple[str, ...] = ("identity",)
    support_brightness_range: BrightnessRange = field(default_factory=BrightnessRange)
    dvt_denoise_mode: str = "none"
    dvt_denoise_alpha: float = 1.0
    normality_mode: Literal[
        "fused",
        "layer_wise",
        "raw_nn",
        "raw_layer_wise",
        "raw_nn_nf_residual",
        "foreground_raw_nn",
        "foreground_flow_mixture",
        "conv2d_flow",
        "spatial_context_flow",
        "transformer_flow",
    ] = "fused"
    residual_weight: float = 0.25
    shift_projection_rank: int = 0
    shift_projection_trim: float = 0.20
    shift_projection_max_samples: int = 32768
    shift_projection_strength: float = 1.0


@dataclass(frozen=True)
class FeatureExtractionConfig:
    feature_fusion: str
    context_source: str = "none"
    tiling: TilingConfig = field(default_factory=TilingConfig)
    transformer_context_mode: str = "none"


@dataclass(frozen=True)
class SupportExtractionConfig:
    transform_names: Tuple[str, ...]
    feature: FeatureExtractionConfig
    brightness_range: BrightnessRange
    brightness_seed: int


@dataclass(frozen=True)
class ImageItem:
    anomaly_type: str
    path: Path


@dataclass(frozen=True)
class FeatureMap:
    values: np.ndarray
    image_shape: Tuple[int, int]
    contexts: Optional[np.ndarray] = None
    transformer_context_tokens: Optional[np.ndarray] = None


@dataclass(frozen=True)
class LayeredFeatureMap:
    layers: Tuple[FeatureMap, ...]


@dataclass(frozen=True)
class SupportFeatures:
    values: np.ndarray
    contexts: Optional[np.ndarray]


@dataclass(frozen=True)
class LayerWiseState:
    pipeline: FlowTTE
    feature_denoiser: Optional[PositionMeanArtifactDenoiser]
    score_field_stats: Optional[ScoreFieldStats]
    train_nll_mean: float
    train_nll_std: float
    density_threshold: float


@dataclass(frozen=True)
class RawFusedState:
    scorer: Union[RawNNState, ForegroundSplitRawNNState]  # noqa: UP007
    feature_denoiser: Optional[PositionMeanArtifactDenoiser]
    residual_pipeline: Optional[FlowTTE]
    residual_train_nll_mean: float
    residual_train_nll_std: float
    residual_density_threshold: float


@dataclass(frozen=True)
class RawFusedRuntime:
    backbone: BackboneLike
    flow_context_source: str
    memory_context_source: str
    config: RunConfig


@dataclass(frozen=True)
class RawLayerState:
    scorer: RawNNState
    feature_denoiser: Optional[PositionMeanArtifactDenoiser]


@dataclass(frozen=True)
class FlowMixtureState:
    foreground: FlowTTE
    background: Optional[FlowTTE]
    feature_denoiser: Optional[PositionMeanArtifactDenoiser]
    train_nll_mean: float
    train_nll_std: float
    density_threshold: float
    memory_size: int


@dataclass(frozen=True)
class LayerWiseItemScore:
    patch_scores: np.ndarray
    image_shape: Tuple[int, int]
    selected_count: int
    memory_size_before: int
    memory_size_after: int


@dataclass(frozen=True)
class LayerWiseRuntime:
    backbone: BackboneLike
    flow_context_source: str
    memory_context_source: str
    config: RunConfig


@dataclass(frozen=True)
class ObjectDiagnostics:
    object_name: str
    resolution: int
    train_good_count: int
    test_good_count: int
    test_bad_count: int
    selected_support_count: int
    selected_support_paths: Tuple[str, ...]
    processed_test_count: int
    mean_selected_patch_count: float
    initial_memory_size: int
    final_memory_size: int
    train_nll_mean: float
    train_nll_std: float
    density_threshold: float
    dvt_denoise_mode: str
    dvt_denoise_alpha: float
    dvt_artifact_l2_mean: float
    score_field_calibration_mode: str
    score_field_foreground_mode: str
    normality_mode: str
    elapsed_seconds: float


def log_object_stage(object_name: str, stage: str, started: float) -> None:
    elapsed = time.time() - started
    print(f"[flowtte][{object_name}] {stage} elapsed={elapsed:.1f}s", flush=True)


def run_object(
    dataset: DatasetLike,
    backbone: BackboneLike,
    object_name: str,
    config: RunConfig,
) -> ObjectDiagnostics:
    if config.normality_mode == "layer_wise":
        return run_object_layer_wise(dataset, backbone, object_name, config)
    if config.normality_mode == "raw_layer_wise":
        return run_object_raw_layer_wise(dataset, backbone, object_name, config)
    if config.normality_mode in ("raw_nn", "raw_nn_nf_residual", "foreground_raw_nn"):
        return run_object_raw_fused(dataset, backbone, object_name, config)
    if config.normality_mode == "foreground_flow_mixture":
        return run_object_flow_mixture(dataset, backbone, object_name, config)
    if config.normality_mode in ("conv2d_flow", "spatial_context_flow", "transformer_flow"):
        return run_object_map_flow(dataset, backbone, object_name, config)
    return run_object_fused(dataset, backbone, object_name, config)


def score_fused_path(
    backbone: BackboneLike,
    pipeline: FlowTTE,
    path: Path,
    config: RunConfig,
    feature_denoiser: Optional[PositionMeanArtifactDenoiser],
    flow_context_source: str,
    memory_context_source: str,
    score_field_stats: Optional[ScoreFieldStats],
    score_field_config: ScoreFieldConfig,
) -> Tuple[np.ndarray, int, int, int]:
    image = read_rgb(path)
    raw_feature_map = extract_feature_map_from_rgb(
        backbone,
        image,
        FeatureExtractionConfig(
            feature_fusion=config.feature_fusion,
            context_source=flow_context_source,
            tiling=tiling_config(config),
        ),
    )
    feature_map = apply_feature_denoiser(raw_feature_map, feature_denoiser)
    batch_contexts = None
    if feature_map.contexts is not None:
        batch_contexts = feature_map.contexts[np.newaxis, ...]
    memory_contexts = batch_contexts
    if memory_context_source != flow_context_source:
        memory_contexts = batch_feature_contexts_from_map(
            raw_feature_map,
            memory_context_source,
        )
        if memory_contexts is None and memory_context_source != "none":
            memory_map = extract_feature_map_from_rgb(
                backbone,
                image,
                FeatureExtractionConfig(
                    feature_fusion=config.feature_fusion,
                    context_source=memory_context_source,
                    tiling=tiling_config(config),
                ),
            )
            if memory_map.contexts is not None:
                memory_contexts = memory_map.contexts[np.newaxis, ...]
    result = pipeline.score_then_expand(
        feature_map.values[np.newaxis, ...],
        batch_contexts=batch_contexts,
        memory_contexts=memory_contexts,
    )
    patch_scores = result.patch_scores[0]
    if score_field_stats is not None:
        patch_scores = apply_score_field_transform(
            patch_scores,
            score_field_stats,
            score_field_config,
        )
    score_map = cv2.resize(
        patch_scores,
        (feature_map.image_shape[1], feature_map.image_shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return (
        score_map,
        result.memory_size_before,
        result.memory_size_after,
        result.selected_count,
    )


def extract_fused_query_map(
    backbone: BackboneLike,
    path: Path,
    config: RunConfig,
    feature_denoiser: Optional[PositionMeanArtifactDenoiser],
    flow_context_source: str,
) -> FeatureMap:
    image = read_rgb(path)
    raw_feature_map = extract_feature_map_from_rgb(
        backbone,
        image,
        FeatureExtractionConfig(
            feature_fusion=config.feature_fusion,
            context_source=flow_context_source,
            tiling=tiling_config(config),
        ),
    )
    values = apply_feature_denoiser(raw_feature_map, feature_denoiser).values
    return FeatureMap(
        values=values,
        image_shape=raw_feature_map.image_shape,
        contexts=raw_feature_map.contexts,
        transformer_context_tokens=raw_feature_map.transformer_context_tokens,
    )


def score_shift_projected_map(
    pipeline: FlowTTE,
    feature_map: FeatureMap,
    config: RunConfig,
    shift_basis: np.ndarray,
    score_field_stats: Optional[ScoreFieldStats],
    score_field_config: ScoreFieldConfig,
) -> Tuple[np.ndarray, int, int, int]:
    contexts = None if feature_map.contexts is None else feature_map.contexts[np.newaxis, ...]
    result = pipeline.score_static_shift_projected(
        feature_map.values[np.newaxis, ...],
        shift_basis=shift_basis,
        strength=config.shift_projection_strength,
        batch_contexts=contexts,
        memory_contexts=contexts,
    )
    patch_scores = result.patch_scores[0]
    if score_field_stats is not None:
        patch_scores = apply_score_field_transform(
            patch_scores,
            score_field_stats,
            score_field_config,
        )
    score_map = cv2.resize(
        patch_scores,
        (feature_map.image_shape[1], feature_map.image_shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return score_map, result.memory_size_before, result.memory_size_after, result.selected_count


def run_object_fused(  # noqa: PLR0915
    dataset: DatasetLike,
    backbone: BackboneLike,
    object_name: str,
    config: RunConfig,
) -> ObjectDiagnostics:
    started = time.time()
    log_object_stage(object_name, "start", started)
    info = dataset.get_object_info(object_name)
    backbone.set_resolution(config.backbone_resolution or info.resolution)
    train_paths = [Path(path) for path in dataset.get_train_images(object_name)]
    selected_paths = select_support_paths_for_backbone(
        backbone,
        train_paths,
        shots=config.shots,
        policy=config.support_selection,
        seed=config.support_selection_seed,
    )
    if len(selected_paths) < config.shots:
        message = f"{object_name}: train/good has fewer than {config.shots} images"
        raise SystemExit(message)
    log_object_stage(object_name, "support_selected", started)

    flow_context_source = resolve_flow_context_source(config)
    memory_context_source = resolve_memory_context_source(config)
    pipeline = build_pipeline(config)
    support_feature_maps = collect_support_feature_maps(
        backbone,
        selected_paths,
        support_extraction_config(config, flow_context_source),
    )
    log_object_stage(object_name, "support_features", started)
    feature_denoiser = fit_feature_denoiser(
        config.dvt_denoise_mode,
        [feature_map.values for feature_map in support_feature_maps],
        alpha=config.dvt_denoise_alpha,
    )
    log_object_stage(object_name, "feature_denoiser", started)
    support_features = flatten_support_feature_maps(
        support_feature_maps,
        require_contexts=flow_context_source != "none",
        feature_denoiser=feature_denoiser,
    )
    support_memory_contexts = support_features.contexts
    if memory_context_source != flow_context_source:
        support_memory_contexts = flatten_feature_contexts_from_feature_maps(
            support_feature_maps,
            memory_context_source,
        )
        if support_memory_contexts is None and memory_context_source != "none":
            support_memory_features = collect_support_features(
                backbone,
                selected_paths,
                support_extraction_config(config, memory_context_source),
            )
            support_memory_contexts = support_memory_features.contexts
    memory_selector = None
    if config.latent_bank_subsample == "superadd_knn_score":
        memory_selector = lambda latents: subsample_knn_score_rank(
            latents,
            target_count=config.latent_bank_target_count,
            knn_neighbors=100,
            query_chunk_size=min(config.query_chunk_size, 256),
        )
    stats = pipeline.fit(
        support_features.values,
        support_contexts=support_features.contexts,
        memory_contexts=support_memory_contexts,
        memory_selector=memory_selector,
    )
    if config.latent_bank_subsample == "superadd_knn_score":
        memory_size = pipeline.memory.bank.size() if pipeline.memory is not None else 0
        if memory_size > config.latent_bank_target_count:
            raise RuntimeError(f"{object_name}: latent bank selection exceeded target")
        log_object_stage(object_name, f"latent_bank_subsampled_{memory_size}", started)
    log_object_stage(object_name, "pipeline_fit", started)
    score_field_config = build_score_field_config(config)
    score_field_stats = fit_support_score_field_stats(
        pipeline,
        support_feature_maps,
        feature_denoiser,
        score_field_config,
        selected_paths,
    )
    log_object_stage(object_name, "score_field_fit", started)
    calibration_paths: Tuple[Path, ...] = ()
    if config.threshold_calibration_mode == "superadd_train95":
        calibration_paths = select_superadd_threshold_paths(train_paths)
        overlap = set(selected_paths).intersection(calibration_paths)
        if overlap or len(selected_paths) + len(calibration_paths) != len(train_paths):
            raise RuntimeError(f"{object_name}: invalid SuperADD 7/8 + 1/8 train split")
        for path in calibration_paths:
            score_map, _, _, _ = score_fused_path(
                backbone,
                pipeline,
                path,
                config,
                feature_denoiser,
                flow_context_source,
                memory_context_source,
                score_field_stats,
                score_field_config,
            )
            save_calibration_prediction(config.output_root, object_name, path, score_map)
        write_threshold_split_manifest(
            config.output_root,
            object_name,
            train_paths,
            selected_paths,
            calibration_paths,
        )
        log_object_stage(object_name, "threshold_calibration_scored", started)
    test_images = dataset.get_test_images(object_name, split="test_public")
    items = stream_test_images(test_images)
    selected_patch_counts: List[int] = []
    first_memory_size = 0
    last_memory_size = 0
    if config.shift_projection_rank > 0:
        cached_queries: List[Tuple[ImageItem, FeatureMap]] = []
        residual_batches: List[np.ndarray] = []
        for item in items:
            feature_map = extract_fused_query_map(
                backbone,
                item.path,
                config,
                feature_denoiser,
                flow_context_source,
            )
            cached_queries.append((item, feature_map))
            residual_batches.append(
                pipeline.latent_residuals(feature_map.values[np.newaxis, ...]),
            )
        projection = fit_shift_projection(
            residual_batches,
            rank=config.shift_projection_rank,
            trim_fraction=config.shift_projection_trim,
            max_samples=config.shift_projection_max_samples,
            seed=config.seed,
        )
        projection_dir = config.output_root / "shift_projection"
        projection_dir.mkdir(parents=True, exist_ok=True)
        np.save(projection_dir / f"{object_name}_basis.npy", projection.basis)
        (projection_dir / f"{object_name}.json").write_text(
            json.dumps({
                "rank": config.shift_projection_rank,
                "trim_fraction": config.shift_projection_trim,
                "strength": config.shift_projection_strength,
                "sampled_count": projection.sampled_count,
                "retained_count": projection.retained_count,
                "retained_energy_ratio": projection.retained_energy_ratio,
                "test_batch_size": len(items),
                "ground_truth_used": False,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        log_object_stage(object_name, "shift_projection_fit", started)
        for item, feature_map in cached_queries:
            score_map, memory_before, memory_after, selected_count = score_shift_projected_map(
                pipeline,
                feature_map,
                config,
                projection.basis,
                score_field_stats,
                score_field_config,
            )
            first_memory_size = first_memory_size or memory_before
            last_memory_size = memory_after
            selected_patch_counts.append(selected_count)
            save_prediction(config.output_root, object_name, item, score_map)
    else:
        for item in items:
            score_map, memory_before, memory_after, selected_count = score_fused_path(
                backbone,
                pipeline,
                item.path,
                config,
                feature_denoiser,
                flow_context_source,
                memory_context_source,
                score_field_stats,
                score_field_config,
            )
            first_memory_size = first_memory_size or memory_before
            last_memory_size = memory_after
            selected_patch_counts.append(selected_count)
            save_prediction(config.output_root, object_name, item, score_map)
    log_object_stage(object_name, "test_scored", started)
    return ObjectDiagnostics(
        object_name=object_name,
        resolution=info.resolution,
        train_good_count=len(train_paths),
        test_good_count=len(test_images.get("good", [])),
        test_bad_count=sum(
            len(paths) for anomaly_type, paths in test_images.items() if anomaly_type != "good"
        ),
        selected_support_count=len(selected_paths),
        selected_support_paths=tuple(str(path) for path in selected_paths),
        processed_test_count=len(items),
        mean_selected_patch_count=float(np.mean(selected_patch_counts)),
        initial_memory_size=first_memory_size,
        final_memory_size=last_memory_size,
        train_nll_mean=stats.train_nll_mean,
        train_nll_std=stats.train_nll_std,
        density_threshold=stats.density_threshold,
        dvt_denoise_mode=config.dvt_denoise_mode,
        dvt_denoise_alpha=config.dvt_denoise_alpha,
        dvt_artifact_l2_mean=artifact_l2_mean(feature_denoiser),
        score_field_calibration_mode=config.score_field_calibration_mode,
        score_field_foreground_mode=config.score_field_foreground_mode,
        normality_mode=config.normality_mode,
        elapsed_seconds=time.time() - started,
    )


def run_object_map_flow(
    dataset: DatasetLike,
    backbone: BackboneLike,
    object_name: str,
    config: RunConfig,
) -> ObjectDiagnostics:
    started = time.time()
    log_object_stage(object_name, "start", started)
    if (
        resolve_flow_context_source(config) != "none"
        or resolve_memory_context_source(config) != "none"
    ):
        message = f"{config.normality_mode} currently expects no context conditioning"
        raise RuntimeError(message)
    info = dataset.get_object_info(object_name)
    backbone.set_resolution(config.backbone_resolution or info.resolution)
    train_paths = [Path(path) for path in dataset.get_train_images(object_name)]
    selected_paths = select_support_paths_for_backbone(
        backbone,
        train_paths,
        shots=config.shots,
        policy=config.support_selection,
        seed=config.support_selection_seed,
    )
    if len(selected_paths) < config.shots:
        message = f"{object_name}: train/good has fewer than {config.shots} images"
        raise SystemExit(message)
    log_object_stage(object_name, "support_selected", started)

    support_feature_maps = collect_support_feature_maps(
        backbone,
        selected_paths,
        support_extraction_config(config, "none"),
    )
    log_object_stage(object_name, "support_features", started)
    feature_denoiser = fit_feature_denoiser(
        config.dvt_denoise_mode,
        [feature_map.values for feature_map in support_feature_maps],
        alpha=config.dvt_denoise_alpha,
    )
    support_maps = [
        apply_feature_denoiser(feature_map, feature_denoiser).values
        for feature_map in support_feature_maps
    ]
    support_context_tokens = transformer_context_tokens_for_feature_maps(
        support_feature_maps,
        config.transformer_context_mode,
    )
    log_object_stage(object_name, "feature_denoiser", started)
    estimator = build_map_flow_estimator(
        config,
        int(support_maps[0].shape[-1]),
        transformer_context_dim(support_context_tokens),
    )
    if isinstance(estimator, TransformerFlowDensityEstimator):
        stats = estimator.fit(
            support_maps,
            config.density_quantile,
            context_tokens=support_context_tokens,
        )
        support_eval = estimator.evaluate_many(
            support_maps,
            context_tokens=support_context_tokens,
        )
    elif isinstance(estimator, FlowDensityEstimator):
        support_input = np.stack(support_maps, axis=0)
        stats = estimator.fit(support_input, config.density_quantile)
        support_eval = estimator.evaluate_many(support_input)
    else:
        stats = estimator.fit(support_maps, config.density_quantile)
        support_eval = estimator.evaluate_many(support_maps)
    score_config = build_score_config(config)
    bank = TorchMemoryBank()
    bank.fit(support_eval.z)
    calibration = ScoreCalibration.fit(support_eval.z, score_config)
    log_object_stage(object_name, "pipeline_fit", started)
    test_images = dataset.get_test_images(object_name, split="test_public")
    items = stream_test_images(test_images)
    for item in items:
        image = read_rgb(item.path)
        feature_map = apply_feature_denoiser(
            extract_feature_map_from_rgb(
                backbone,
                image,
                FeatureExtractionConfig(
                    feature_fusion=config.feature_fusion,
                    context_source="none",
                    tiling=tiling_config(config),
                    transformer_context_mode=config.transformer_context_mode,
                ),
            ),
            feature_denoiser,
        )
        if isinstance(estimator, TransformerFlowDensityEstimator):
            evaluation = estimator.evaluate(
                feature_map.values,
                context_tokens=transformer_context_token_for_feature_map(
                    feature_map,
                    config.transformer_context_mode,
                ),
            )
        else:
            evaluation_input = (
                np.expand_dims(feature_map.values, axis=0)
                if isinstance(estimator, FlowDensityEstimator)
                else feature_map.values
            )
            evaluation = estimator.evaluate(evaluation_input)
        image_indices = torch.zeros(
            evaluation.z.shape[0],
            device=evaluation.z.device,
            dtype=torch.long,
        )
        result = score_flow_memory(
            inputs=ScoreInputs(
                query_z=evaluation.z,
                nll=evaluation.nll,
                nll_penalty=estimator.density_penalty(evaluation.nll),
                image_indices=image_indices,
                n_images=1,
            ),
            bank=bank,
            config=score_config,
            calibration=calibration,
        )
        patch_scores = result.patch_scores.reshape(evaluation.spatial_shape).detach().cpu().numpy()
        score_map = cv2.resize(
            patch_scores.astype(np.float32, copy=False),
            (feature_map.image_shape[1], feature_map.image_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        save_prediction(config.output_root, object_name, item, score_map)
    log_object_stage(object_name, "test_scored", started)
    return ObjectDiagnostics(
        object_name=object_name,
        resolution=info.resolution,
        train_good_count=len(train_paths),
        test_good_count=len(test_images.get("good", [])),
        test_bad_count=sum(
            len(paths) for anomaly_type, paths in test_images.items() if anomaly_type != "good"
        ),
        selected_support_count=len(selected_paths),
        selected_support_paths=tuple(str(path) for path in selected_paths),
        processed_test_count=len(items),
        mean_selected_patch_count=0.0,
        initial_memory_size=bank.size(),
        final_memory_size=bank.size(),
        train_nll_mean=stats.train_nll_mean,
        train_nll_std=stats.train_nll_std,
        density_threshold=stats.density_threshold,
        dvt_denoise_mode=config.dvt_denoise_mode,
        dvt_denoise_alpha=config.dvt_denoise_alpha,
        dvt_artifact_l2_mean=artifact_l2_mean(feature_denoiser),
        score_field_calibration_mode=config.score_field_calibration_mode,
        score_field_foreground_mode=config.score_field_foreground_mode,
        normality_mode=config.normality_mode,
        elapsed_seconds=time.time() - started,
    )


def run_object_flow_mixture(
    dataset: DatasetLike,
    backbone: BackboneLike,
    object_name: str,
    config: RunConfig,
) -> ObjectDiagnostics:
    started = time.time()
    info = dataset.get_object_info(object_name)
    backbone.set_resolution(config.backbone_resolution or info.resolution)
    train_paths = [Path(path) for path in dataset.get_train_images(object_name)]
    selected_paths = select_support_paths_for_backbone(
        backbone,
        train_paths,
        shots=config.shots,
        policy=config.support_selection,
        seed=config.support_selection_seed,
    )
    if len(selected_paths) < config.shots:
        message = f"{object_name}: train/good has fewer than {config.shots} images"
        raise SystemExit(message)
    if (
        resolve_flow_context_source(config) != "none"
        or resolve_memory_context_source(config) != "none"
    ):
        raise RuntimeError("foreground_flow_mixture currently expects no context conditioning")

    support_feature_maps = collect_support_feature_maps(
        backbone,
        selected_paths,
        support_extraction_config(config, "none"),
    )
    feature_denoiser = fit_feature_denoiser(
        config.dvt_denoise_mode,
        [feature_map.values for feature_map in support_feature_maps],
        alpha=config.dvt_denoise_alpha,
    )
    support_features = flatten_support_feature_maps(
        support_feature_maps,
        require_contexts=False,
        feature_denoiser=feature_denoiser,
    )
    denoised_maps = [
        apply_feature_denoiser(feature_map, feature_denoiser).values
        for feature_map in support_feature_maps
    ]
    split_mask = feature_energy_split_mask(
        denoised_maps,
        config.score_field_foreground_quantile,
    )
    state = fit_flow_mixture_state(support_features.values, split_mask, feature_denoiser, config)

    test_images = dataset.get_test_images(object_name, split="test_public")
    items = stream_test_images(test_images)
    for item in items:
        image = read_rgb(item.path)
        feature_map = apply_feature_denoiser(
            extract_feature_map_from_rgb(
                backbone,
                image,
                FeatureExtractionConfig(
                    feature_fusion=config.feature_fusion,
                    context_source="none",
                    tiling=tiling_config(config),
                ),
            ),
            state.feature_denoiser,
        )
        foreground = state.foreground.score_static(feature_map.values[np.newaxis, ...])
        patch_scores = foreground.patch_scores[0]
        background = state.background
        if background is not None:
            background_scores = background.score_static(
                feature_map.values[np.newaxis, ...],
            ).patch_scores[0]
            patch_scores = np.minimum(patch_scores, background_scores).astype(
                np.float32,
                copy=False,
            )
        score_map = cv2.resize(
            patch_scores,
            (feature_map.image_shape[1], feature_map.image_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        save_prediction(config.output_root, object_name, item, score_map)

    return ObjectDiagnostics(
        object_name=object_name,
        resolution=info.resolution,
        train_good_count=len(train_paths),
        test_good_count=len(test_images.get("good", [])),
        test_bad_count=sum(
            len(paths) for anomaly_type, paths in test_images.items() if anomaly_type != "good"
        ),
        selected_support_count=len(selected_paths),
        selected_support_paths=tuple(str(path) for path in selected_paths),
        processed_test_count=len(items),
        mean_selected_patch_count=0.0,
        initial_memory_size=state.memory_size,
        final_memory_size=state.memory_size,
        train_nll_mean=state.train_nll_mean,
        train_nll_std=state.train_nll_std,
        density_threshold=state.density_threshold,
        dvt_denoise_mode=config.dvt_denoise_mode,
        dvt_denoise_alpha=config.dvt_denoise_alpha,
        dvt_artifact_l2_mean=artifact_l2_mean(feature_denoiser),
        score_field_calibration_mode=config.score_field_calibration_mode,
        score_field_foreground_mode=config.score_field_foreground_mode,
        normality_mode=config.normality_mode,
        elapsed_seconds=time.time() - started,
    )


def fit_flow_mixture_state(
    support_features: np.ndarray,
    foreground_mask: np.ndarray,
    feature_denoiser: Optional[PositionMeanArtifactDenoiser],
    config: RunConfig,
) -> FlowMixtureState:
    if support_features.shape[0] != foreground_mask.shape[0]:
        raise RuntimeError("foreground_flow_mixture split mask must match support patches")
    foreground_values = support_features[foreground_mask]
    if foreground_values.shape[0] == 0:
        foreground_values = support_features
    foreground_pipeline = build_pipeline(config)
    foreground_stats = foreground_pipeline.fit(foreground_values)
    background_pipeline = None
    train_nll_mean = foreground_stats.train_nll_mean
    train_nll_std = foreground_stats.train_nll_std
    density_threshold = foreground_stats.density_threshold
    background_values = support_features[~foreground_mask]
    if background_values.shape[0] > 1:
        background_pipeline = build_pipeline(config)
        background_stats = background_pipeline.fit(background_values)
        train_nll_mean = float(
            np.mean([foreground_stats.train_nll_mean, background_stats.train_nll_mean]),
        )
        train_nll_std = float(
            np.mean([foreground_stats.train_nll_std, background_stats.train_nll_std]),
        )
        density_threshold = float(
            np.mean([foreground_stats.density_threshold, background_stats.density_threshold]),
        )
    memory_size = foreground_pipeline.memory.bank.size() if foreground_pipeline.memory else 0
    if background_pipeline is not None and background_pipeline.memory is not None:
        memory_size += background_pipeline.memory.bank.size()
    return FlowMixtureState(
        foreground=foreground_pipeline,
        background=background_pipeline,
        feature_denoiser=feature_denoiser,
        train_nll_mean=train_nll_mean,
        train_nll_std=train_nll_std,
        density_threshold=density_threshold,
        memory_size=memory_size,
    )


def run_object_raw_fused(
    dataset: DatasetLike,
    backbone: BackboneLike,
    object_name: str,
    config: RunConfig,
) -> ObjectDiagnostics:
    started = time.time()
    info = dataset.get_object_info(object_name)
    backbone.set_resolution(config.backbone_resolution or info.resolution)
    train_paths = [Path(path) for path in dataset.get_train_images(object_name)]
    selected_paths = select_support_paths_for_backbone(
        backbone,
        train_paths,
        shots=config.shots,
        policy=config.support_selection,
        seed=config.support_selection_seed,
    )
    if len(selected_paths) < config.shots:
        message = f"{object_name}: train/good has fewer than {config.shots} images"
        raise SystemExit(message)

    memory_context_source = resolve_memory_context_source(config)
    flow_context_source = resolve_flow_context_source(config)
    support_feature_maps = collect_support_feature_maps(
        backbone,
        selected_paths,
        support_extraction_config(config, memory_context_source),
    )
    feature_denoiser = fit_feature_denoiser(
        config.dvt_denoise_mode,
        [feature_map.values for feature_map in support_feature_maps],
        alpha=config.dvt_denoise_alpha,
    )
    support_features = flatten_support_feature_maps(
        support_feature_maps,
        require_contexts=memory_context_source != "none",
        feature_denoiser=feature_denoiser,
    )
    raw_state = fit_raw_fused_state(
        support_feature_maps,
        support_features,
        feature_denoiser,
        config,
    )
    if config.normality_mode == "raw_nn_nf_residual":
        flow_support_features = support_features
        if flow_context_source != memory_context_source:
            flow_maps = collect_support_feature_maps(
                backbone,
                selected_paths,
                support_extraction_config(config, flow_context_source),
            )
            flow_support_features = flatten_support_feature_maps(
                flow_maps,
                require_contexts=flow_context_source != "none",
                feature_denoiser=feature_denoiser,
            )
        residual_pipeline = build_pipeline(config)
        residual_stats = residual_pipeline.fit(
            flow_support_features.values,
            support_contexts=flow_support_features.contexts,
            memory_contexts=support_features.contexts,
        )
        raw_state = RawFusedState(
            scorer=raw_state.scorer,
            feature_denoiser=raw_state.feature_denoiser,
            residual_pipeline=residual_pipeline,
            residual_train_nll_mean=residual_stats.train_nll_mean,
            residual_train_nll_std=residual_stats.train_nll_std,
            residual_density_threshold=residual_stats.density_threshold,
        )

    test_images = dataset.get_test_images(object_name, split="test_public")
    items = stream_test_images(test_images)
    runtime = RawFusedRuntime(
        backbone=backbone,
        flow_context_source=flow_context_source,
        memory_context_source=memory_context_source,
        config=config,
    )
    for item in items:
        item_score = score_raw_fused_item(
            runtime,
            item,
            raw_state,
        )
        score_map = cv2.resize(
            item_score.patch_scores,
            (item_score.image_shape[1], item_score.image_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        save_prediction(config.output_root, object_name, item, score_map)
    return ObjectDiagnostics(
        object_name=object_name,
        resolution=info.resolution,
        train_good_count=len(train_paths),
        test_good_count=len(test_images.get("good", [])),
        test_bad_count=sum(
            len(paths) for anomaly_type, paths in test_images.items() if anomaly_type != "good"
        ),
        selected_support_count=len(selected_paths),
        selected_support_paths=tuple(str(path) for path in selected_paths),
        processed_test_count=len(items),
        mean_selected_patch_count=0.0,
        initial_memory_size=raw_state.scorer.memory_size,
        final_memory_size=raw_state.scorer.memory_size,
        train_nll_mean=raw_state.residual_train_nll_mean,
        train_nll_std=raw_state.residual_train_nll_std,
        density_threshold=raw_state.residual_density_threshold,
        dvt_denoise_mode=config.dvt_denoise_mode,
        dvt_denoise_alpha=config.dvt_denoise_alpha,
        dvt_artifact_l2_mean=artifact_l2_mean(feature_denoiser),
        score_field_calibration_mode=config.score_field_calibration_mode,
        score_field_foreground_mode=config.score_field_foreground_mode,
        normality_mode=config.normality_mode,
        elapsed_seconds=time.time() - started,
    )


def fit_raw_fused_state(
    support_feature_maps: Sequence[FeatureMap],
    support_features: SupportFeatures,
    feature_denoiser: Optional[PositionMeanArtifactDenoiser],
    config: RunConfig,
) -> RawFusedState:
    raw_score_config = build_raw_score_config(config)
    if config.normality_mode == "foreground_raw_nn":
        denoised_maps = [
            apply_feature_denoiser(feature_map, feature_denoiser).values
            for feature_map in support_feature_maps
        ]
        scorer: Union[RawNNState, ForegroundSplitRawNNState] = (  # noqa: UP007
            fit_foreground_split_raw_nn(
                support_feature_maps=denoised_maps,
                support_contexts=support_features.contexts,
                score_config=raw_score_config,
                split_config=ForegroundSplitConfig(
                    foreground_quantile=config.score_field_foreground_quantile,
                    background_multiplier=config.score_field_background_multiplier,
                ),
                device=config.device,
            )
        )
    else:
        scorer = fit_raw_nn(
            support_features=support_features.values,
            memory_contexts=support_features.contexts,
            score_config=raw_score_config,
            device=config.device,
        )
    return RawFusedState(
        scorer=scorer,
        feature_denoiser=feature_denoiser,
        residual_pipeline=None,
        residual_train_nll_mean=0.0,
        residual_train_nll_std=0.0,
        residual_density_threshold=0.0,
    )


def score_raw_fused_item(
    runtime: RawFusedRuntime,
    item: ImageItem,
    raw_state: RawFusedState,
) -> LayerWiseItemScore:
    image = read_rgb(item.path)
    feature_map = apply_feature_denoiser(
        extract_feature_map_from_rgb(
            runtime.backbone,
            image,
            FeatureExtractionConfig(
                feature_fusion=runtime.config.feature_fusion,
                context_source=runtime.memory_context_source,
                tiling=tiling_config(runtime.config),
            ),
        ),
        raw_state.feature_denoiser,
    )
    raw_result = score_raw_scorer(
        raw_state.scorer,
        feature_map.values[np.newaxis, ...],
        _batch_contexts(feature_map),
    )
    patch_scores = raw_result.patch_scores[0]
    residual_pipeline = raw_state.residual_pipeline
    if residual_pipeline is not None:
        residual_map = feature_map
        if runtime.flow_context_source != runtime.memory_context_source:
            residual_map = apply_feature_denoiser(
                extract_feature_map_from_rgb(
                    runtime.backbone,
                    image,
                    FeatureExtractionConfig(
                        feature_fusion=runtime.config.feature_fusion,
                        context_source=runtime.flow_context_source,
                        tiling=tiling_config(runtime.config),
                    ),
                ),
                raw_state.feature_denoiser,
            )
        residual_batch_contexts = (
            _batch_contexts(residual_map) if runtime.flow_context_source != "none" else None
        )
        residual_memory_contexts = (
            _batch_contexts(feature_map) if runtime.memory_context_source != "none" else None
        )
        residual = residual_pipeline.score_static(
            residual_map.values[np.newaxis, ...],
            batch_contexts=residual_batch_contexts,
            memory_contexts=residual_memory_contexts,
        )
        patch_scores = (
            patch_scores + np.float32(runtime.config.residual_weight) * residual.patch_scores[0]
        ).astype(np.float32, copy=False)
    return LayerWiseItemScore(
        patch_scores=patch_scores,
        image_shape=feature_map.image_shape,
        selected_count=0,
        memory_size_before=raw_result.memory_size_before,
        memory_size_after=raw_result.memory_size_after,
    )


def score_raw_scorer(
    scorer: Union[RawNNState, ForegroundSplitRawNNState],  # noqa: UP007
    query_features: np.ndarray,
    query_contexts: Optional[np.ndarray],
) -> RawNNResult:
    if isinstance(scorer, ForegroundSplitRawNNState):
        return score_foreground_split_raw_nn(scorer, query_features, query_contexts)
    return score_raw_nn(scorer, query_features, query_contexts)


def run_object_raw_layer_wise(
    dataset: DatasetLike,
    backbone: BackboneLike,
    object_name: str,
    config: RunConfig,
) -> ObjectDiagnostics:
    started = time.time()
    info = dataset.get_object_info(object_name)
    backbone.set_resolution(config.backbone_resolution or info.resolution)
    train_paths = [Path(path) for path in dataset.get_train_images(object_name)]
    selected_paths = select_support_paths_for_backbone(
        backbone,
        train_paths,
        shots=config.shots,
        policy=config.support_selection,
        seed=config.support_selection_seed,
    )
    if len(selected_paths) < config.shots:
        message = f"{object_name}: train/good has fewer than {config.shots} images"
        raise SystemExit(message)

    memory_context_source = resolve_memory_context_source(config)
    support_layered_maps = collect_support_layered_feature_maps(
        backbone,
        selected_paths,
        support_extraction_config(config, memory_context_source),
    )
    layer_states = fit_raw_layer_states(
        support_layered_maps,
        require_memory_contexts=memory_context_source != "none",
        config=config,
    )
    test_images = dataset.get_test_images(object_name, split="test_public")
    items = stream_test_images(test_images)
    initial_memory_size = sum(state.scorer.memory_size for state in layer_states)
    for item in items:
        item_score = score_raw_layer_wise_item(
            backbone,
            item,
            layer_states,
            config,
            memory_context_source,
        )
        score_map = cv2.resize(
            item_score.patch_scores,
            (item_score.image_shape[1], item_score.image_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        save_prediction(config.output_root, object_name, item, score_map)
    return ObjectDiagnostics(
        object_name=object_name,
        resolution=info.resolution,
        train_good_count=len(train_paths),
        test_good_count=len(test_images.get("good", [])),
        test_bad_count=sum(
            len(paths) for anomaly_type, paths in test_images.items() if anomaly_type != "good"
        ),
        selected_support_count=len(selected_paths),
        selected_support_paths=tuple(str(path) for path in selected_paths),
        processed_test_count=len(items),
        mean_selected_patch_count=0.0,
        initial_memory_size=initial_memory_size,
        final_memory_size=initial_memory_size,
        train_nll_mean=0.0,
        train_nll_std=0.0,
        density_threshold=0.0,
        dvt_denoise_mode=config.dvt_denoise_mode,
        dvt_denoise_alpha=config.dvt_denoise_alpha,
        dvt_artifact_l2_mean=float(
            np.mean([artifact_l2_mean(state.feature_denoiser) for state in layer_states]),
        ),
        score_field_calibration_mode=config.score_field_calibration_mode,
        score_field_foreground_mode=config.score_field_foreground_mode,
        normality_mode=config.normality_mode,
        elapsed_seconds=time.time() - started,
    )


def fit_raw_layer_states(
    support_layered_maps: Sequence[LayeredFeatureMap],
    require_memory_contexts: bool,
    config: RunConfig,
) -> Tuple[RawLayerState, ...]:
    if not support_layered_maps:
        raise RuntimeError("raw layer-wise NN requires support feature maps")
    layer_count = len(support_layered_maps[0].layers)
    states: List[RawLayerState] = []
    for layer_index in range(layer_count):
        support_maps = tuple(layered.layers[layer_index] for layered in support_layered_maps)
        feature_denoiser = fit_feature_denoiser(
            config.dvt_denoise_mode,
            [feature_map.values for feature_map in support_maps],
            alpha=config.dvt_denoise_alpha,
        )
        support_features = flatten_support_feature_maps(
            support_maps,
            require_contexts=require_memory_contexts,
            feature_denoiser=feature_denoiser,
        )
        states.append(
            RawLayerState(
                scorer=fit_raw_nn(
                    support_features=support_features.values,
                    memory_contexts=support_features.contexts,
                    score_config=build_raw_score_config(config),
                    device=config.device,
                ),
                feature_denoiser=feature_denoiser,
            ),
        )
    return tuple(states)


def score_raw_layer_wise_item(
    backbone: BackboneLike,
    item: ImageItem,
    layer_states: Sequence[RawLayerState],
    config: RunConfig,
    memory_context_source: str,
) -> LayerWiseItemScore:
    layered_map = extract_layer_feature_maps_from_rgb(
        backbone,
        read_rgb(item.path),
        FeatureExtractionConfig(
            feature_fusion=config.feature_fusion,
            context_source=memory_context_source,
            tiling=tiling_config(config),
        ),
    )
    score_parts: List[np.ndarray] = []
    memory_before = 0
    memory_after = 0
    for layer_index, state in enumerate(layer_states):
        feature_map = apply_feature_denoiser(
            layered_map.layers[layer_index],
            state.feature_denoiser,
        )
        result = score_raw_nn(
            state.scorer,
            feature_map.values[np.newaxis, ...],
            _batch_contexts(feature_map),
        )
        score_parts.append(result.patch_scores[0])
        memory_before += result.memory_size_before
        memory_after += result.memory_size_after
    patch_scores = np.mean(np.stack(score_parts, axis=0), axis=0).astype(
        np.float32,
        copy=False,
    )
    return LayerWiseItemScore(
        patch_scores=patch_scores,
        image_shape=layered_map.layers[0].image_shape,
        selected_count=0,
        memory_size_before=memory_before,
        memory_size_after=memory_after,
    )


def build_pipeline(config: RunConfig) -> FlowTTE:
    return FlowTTE(
        FlowTTEConfig(
            flow=build_flow_config(config),
            expansion=ExpansionConfig(
                budget=config.expansion_budget,
                density_quantile=config.density_quantile,
                random_seed=config.seed,
            ),
            score=build_score_config(config),
            device=config.device,
        ),
    )


def build_flow_config(config: RunConfig) -> FlowConfig:
    return FlowConfig(
        n_coupling_layers=config.coupling_layers,
        hidden_multiplier=config.hidden_multiplier,
        transform_mode=config.flow_transform_mode,
        condition_mode=config.flow_condition_mode,
        n_epochs=config.flow_epochs,
        lr=config.flow_lr,
        clamp=config.flow_clamp,
        tail_weight=config.tail_weight,
        tail_top_k_ratio=config.tail_top_k_ratio,
        lambda_logdet=config.lambda_logdet,
        batch_size=512,
        seed=config.seed,
        spatial_context=config.normality_mode == "spatial_context_flow",
    )


def build_score_config(config: RunConfig) -> ScoreConfig:
    return ScoreConfig(
        distance_weight=config.distance_weight,
        density_weight=config.density_weight,
        score_mode=config.score_mode,
        context_mode=resolve_score_context_mode(config),
        context_weight=config.context_weight,
        context_top_m=config.context_top_m,
        top_percent=config.top_percent,
        query_chunk_size=config.query_chunk_size,
        calibration_sample_size=config.calibration_sample_size,
        loo_standardize=config.loo_standardize,
        use_squared_distance=config.use_squared_distance,
    )


def build_map_flow_estimator(
    config: RunConfig,
    dim: int,
    transformer_context_dim_value: Optional[int] = None,
) -> Union[  # noqa: UP007
    FlowDensityEstimator,
    Conv2DFlowDensityEstimator,
    TransformerFlowDensityEstimator,
]:
    if config.normality_mode == "spatial_context_flow":
        return FlowDensityEstimator(
            dim=dim,
            config=build_flow_config(config),
            device=config.device,
        )
    if config.normality_mode == "conv2d_flow":
        return Conv2DFlowDensityEstimator(
            dim=dim,
            config=build_flow_config(config),
            device=config.device,
        )
    if config.normality_mode == "transformer_flow":
        return TransformerFlowDensityEstimator(
            dim=dim,
            config=build_flow_config(config),
            device=config.device,
            context_dim=transformer_context_dim_value,
            dummy_token_count=transformer_dummy_token_count(config.transformer_context_mode),
            dummy_trainable=config.transformer_context_mode == "learned_dummy",
        )
    message = f"Unsupported map flow mode: {config.normality_mode}"
    raise RuntimeError(message)


def transformer_uses_backbone_context(transformer_context_mode: str) -> bool:
    return transformer_context_mode in ("cls", "register", "cls_register")


def transformer_dummy_token_count(transformer_context_mode: str) -> int:
    if transformer_context_mode in ("random_dummy", "learned_dummy"):
        return 4
    return 0


def transformer_context_dim(
    context_tokens: Optional[Sequence[np.ndarray]],
) -> Optional[int]:
    if context_tokens is None:
        return None
    if not context_tokens:
        raise RuntimeError("Transformer context mode requires at least one context token set")
    return int(context_tokens[0].shape[-1])


def transformer_context_tokens_for_feature_maps(
    feature_maps: Sequence[FeatureMap],
    transformer_context_mode: str,
) -> Optional[Tuple[np.ndarray, ...]]:
    if not transformer_uses_backbone_context(transformer_context_mode):
        return None
    contexts: List[np.ndarray] = []
    for feature_map in feature_maps:
        if feature_map.transformer_context_tokens is None:
            message = (
                "Transformer context mode requires backbone context tokens, "
                f"but none were extracted for mode {transformer_context_mode!r}"
            )
            raise RuntimeError(message)
        contexts.append(feature_map.transformer_context_tokens)
    return tuple(contexts)


def transformer_context_token_for_feature_map(
    feature_map: FeatureMap,
    transformer_context_mode: str,
) -> Optional[np.ndarray]:
    if not transformer_uses_backbone_context(transformer_context_mode):
        return None
    if feature_map.transformer_context_tokens is None:
        message = (
            "Transformer context mode requires backbone context tokens, "
            f"but none were extracted for mode {transformer_context_mode!r}"
        )
        raise RuntimeError(message)
    return feature_map.transformer_context_tokens


def build_raw_score_config(config: RunConfig) -> ScoreConfig:
    return ScoreConfig(
        distance_weight=config.distance_weight,
        density_weight=0.0,
        score_mode="latent_distance",
        context_mode=resolve_score_context_mode(config),
        context_weight=config.context_weight,
        context_top_m=config.context_top_m,
        top_percent=config.top_percent,
        query_chunk_size=config.query_chunk_size,
        calibration_sample_size=config.calibration_sample_size,
        use_squared_distance=config.use_squared_distance,
    )


def run_object_layer_wise(
    dataset: DatasetLike,
    backbone: BackboneLike,
    object_name: str,
    config: RunConfig,
) -> ObjectDiagnostics:
    started = time.time()
    info = dataset.get_object_info(object_name)
    backbone.set_resolution(config.backbone_resolution or info.resolution)
    train_paths = [Path(path) for path in dataset.get_train_images(object_name)]
    selected_paths = select_support_paths_for_backbone(
        backbone,
        train_paths,
        shots=config.shots,
        policy=config.support_selection,
        seed=config.support_selection_seed,
    )
    if len(selected_paths) < config.shots:
        message = f"{object_name}: train/good has fewer than {config.shots} images"
        raise SystemExit(message)

    flow_context_source = resolve_flow_context_source(config)
    memory_context_source = resolve_memory_context_source(config)
    support_layered_maps = collect_support_layered_feature_maps(
        backbone,
        selected_paths,
        support_extraction_config(config, flow_context_source),
    )
    support_memory_layered_maps = support_layered_maps
    if memory_context_source != flow_context_source:
        if memory_context_source == "none":
            support_memory_layered_maps = []
        else:
            support_memory_layered_maps = collect_support_layered_feature_maps(
                backbone,
                selected_paths,
                support_extraction_config(config, memory_context_source),
            )
    layer_states = fit_layer_wise_states(
        support_layered_maps,
        support_memory_layered_maps,
        require_flow_contexts=flow_context_source != "none",
        require_memory_contexts=memory_context_source != "none",
        config=config,
    )
    test_images = dataset.get_test_images(object_name, split="test_public")
    items = stream_test_images(test_images)
    selected_patch_counts: List[int] = []
    first_memory_size = 0
    last_memory_size = 0
    runtime = LayerWiseRuntime(
        backbone=backbone,
        flow_context_source=flow_context_source,
        memory_context_source=memory_context_source,
        config=config,
    )
    for item in items:
        item_score = score_layer_wise_item(
            runtime,
            item,
            layer_states,
        )
        first_memory_size = first_memory_size or item_score.memory_size_before
        last_memory_size = item_score.memory_size_after
        selected_patch_counts.append(item_score.selected_count)
        score_map = cv2.resize(
            item_score.patch_scores,
            (item_score.image_shape[1], item_score.image_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        save_prediction(config.output_root, object_name, item, score_map)
    return ObjectDiagnostics(
        object_name=object_name,
        resolution=info.resolution,
        train_good_count=len(train_paths),
        test_good_count=len(test_images.get("good", [])),
        test_bad_count=sum(
            len(paths) for anomaly_type, paths in test_images.items() if anomaly_type != "good"
        ),
        selected_support_count=len(selected_paths),
        selected_support_paths=tuple(str(path) for path in selected_paths),
        processed_test_count=len(items),
        mean_selected_patch_count=float(np.mean(selected_patch_counts)),
        initial_memory_size=first_memory_size,
        final_memory_size=last_memory_size,
        train_nll_mean=float(np.mean([state.train_nll_mean for state in layer_states])),
        train_nll_std=float(np.mean([state.train_nll_std for state in layer_states])),
        density_threshold=float(np.mean([state.density_threshold for state in layer_states])),
        dvt_denoise_mode=config.dvt_denoise_mode,
        dvt_denoise_alpha=config.dvt_denoise_alpha,
        dvt_artifact_l2_mean=float(
            np.mean([artifact_l2_mean(state.feature_denoiser) for state in layer_states]),
        ),
        score_field_calibration_mode=config.score_field_calibration_mode,
        score_field_foreground_mode=config.score_field_foreground_mode,
        normality_mode=config.normality_mode,
        elapsed_seconds=time.time() - started,
    )


def score_layer_wise_item(
    runtime: LayerWiseRuntime,
    item: ImageItem,
    layer_states: Sequence[LayerWiseState],
) -> LayerWiseItemScore:
    image = read_rgb(item.path)
    layered_map = extract_layer_feature_maps_from_rgb(
        runtime.backbone,
        image,
        FeatureExtractionConfig(
            feature_fusion=runtime.config.feature_fusion,
            context_source=runtime.flow_context_source,
            tiling=tiling_config(runtime.config),
        ),
    )
    memory_layered_map = resolve_memory_layered_map(
        runtime,
        image,
        layered_map,
    )
    score_field_config = build_score_field_config(runtime.config)
    score_parts: List[np.ndarray] = []
    selected_count = 0
    memory_before = 0
    memory_after = 0
    for layer_index, state in enumerate(layer_states):
        feature_map = apply_feature_denoiser(
            layered_map.layers[layer_index],
            state.feature_denoiser,
        )
        batch_contexts = _batch_contexts(feature_map)
        memory_contexts = layer_memory_contexts(
            memory_layered_map,
            layer_index,
            batch_contexts,
            runtime.memory_context_source != runtime.flow_context_source,
        )
        result = state.pipeline.score_then_expand(
            feature_map.values[np.newaxis, ...],
            batch_contexts=batch_contexts,
            memory_contexts=memory_contexts,
        )
        layer_score = result.patch_scores[0]
        if state.score_field_stats is not None:
            layer_score = apply_score_field_transform(
                layer_score,
                state.score_field_stats,
                score_field_config,
            )
        score_parts.append(layer_score)
        selected_count += result.selected_count
        memory_before += result.memory_size_before
        memory_after += result.memory_size_after
    patch_scores = np.mean(np.stack(score_parts, axis=0), axis=0).astype(
        np.float32,
        copy=False,
    )
    return LayerWiseItemScore(
        patch_scores=patch_scores,
        image_shape=layered_map.layers[0].image_shape,
        selected_count=selected_count,
        memory_size_before=memory_before,
        memory_size_after=memory_after,
    )


def resolve_memory_layered_map(
    runtime: LayerWiseRuntime,
    image: np.ndarray,
    layered_map: LayeredFeatureMap,
) -> Optional[LayeredFeatureMap]:
    if runtime.memory_context_source == runtime.flow_context_source:
        return layered_map
    if runtime.memory_context_source == "none":
        return None
    return extract_layer_feature_maps_from_rgb(
        runtime.backbone,
        image,
        FeatureExtractionConfig(
            feature_fusion=runtime.config.feature_fusion,
            context_source=runtime.memory_context_source,
            tiling=tiling_config(runtime.config),
        ),
    )


def layer_memory_contexts(
    memory_layered_map: Optional[LayeredFeatureMap],
    layer_index: int,
    batch_contexts: Optional[np.ndarray],
    split_context_source: bool,
) -> Optional[np.ndarray]:
    if memory_layered_map is None:
        return None
    if split_context_source:
        return _batch_contexts(memory_layered_map.layers[layer_index])
    return batch_contexts


def fit_layer_wise_states(
    support_layered_maps: Sequence[LayeredFeatureMap],
    support_memory_layered_maps: Sequence[LayeredFeatureMap],
    require_flow_contexts: bool,
    require_memory_contexts: bool,
    config: RunConfig,
) -> Tuple[LayerWiseState, ...]:
    if not support_layered_maps:
        raise RuntimeError("layer-wise FlowTTE requires support feature maps")
    layer_count = len(support_layered_maps[0].layers)
    score_field_config = build_score_field_config(config)
    states: List[LayerWiseState] = []
    for layer_index in range(layer_count):
        support_maps = tuple(layered.layers[layer_index] for layered in support_layered_maps)
        feature_denoiser = fit_feature_denoiser(
            config.dvt_denoise_mode,
            [feature_map.values for feature_map in support_maps],
            alpha=config.dvt_denoise_alpha,
        )
        support_features = flatten_support_feature_maps(
            support_maps,
            require_contexts=require_flow_contexts,
            feature_denoiser=feature_denoiser,
        )
        memory_contexts = support_features.contexts
        if require_memory_contexts and support_memory_layered_maps is not support_layered_maps:
            memory_maps = tuple(
                layered.layers[layer_index] for layered in support_memory_layered_maps
            )
            memory_contexts = flatten_support_feature_maps(
                memory_maps,
                require_contexts=True,
                feature_denoiser=None,
            ).contexts
        elif not require_memory_contexts:
            memory_contexts = None
        pipeline = build_pipeline(config)
        stats = pipeline.fit(
            support_features.values,
            support_contexts=support_features.contexts,
            memory_contexts=memory_contexts,
        )
        score_field_stats = fit_support_score_field_stats(
            pipeline,
            support_maps,
            feature_denoiser,
            score_field_config,
            (),
        )
        states.append(
            LayerWiseState(
                pipeline=pipeline,
                feature_denoiser=feature_denoiser,
                score_field_stats=score_field_stats,
                train_nll_mean=stats.train_nll_mean,
                train_nll_std=stats.train_nll_std,
                density_threshold=stats.density_threshold,
            ),
        )
    return tuple(states)


def build_score_field_config(config: RunConfig) -> ScoreFieldConfig:
    return ScoreFieldConfig(
        calibration_mode=config.score_field_calibration_mode,
        calibration_alpha=config.score_field_calibration_alpha,
        position_std_floor=config.score_field_position_std_floor,
        foreground_mode=config.score_field_foreground_mode,
        foreground_quantile=config.score_field_foreground_quantile,
        background_multiplier=config.score_field_background_multiplier,
        foreground_smooth_kernel=config.score_field_foreground_smooth_kernel,
        support_score_quantile=config.score_field_support_score_quantile,
    )


def fit_support_score_field_stats(
    pipeline: FlowTTE,
    support_feature_maps: Sequence[FeatureMap],
    feature_denoiser: Optional[PositionMeanArtifactDenoiser],
    score_field_config: ScoreFieldConfig,
    support_paths: Sequence[Path],
) -> Optional[ScoreFieldStats]:
    if not score_field_config.enabled:
        return None
    denoised_maps = [
        apply_feature_denoiser(feature_map, feature_denoiser)
        for feature_map in support_feature_maps
    ]
    support_scores = [
        support_leave_one_out_patch_scores(pipeline, feature_map.values)
        for feature_map in denoised_maps
    ]
    support_object_fields = support_rgb_prior_fields(support_paths, denoised_maps)
    return fit_score_field_stats(
        support_scores,
        [feature_map.values for feature_map in denoised_maps],
        score_field_config,
        support_object_fields=support_object_fields,
    )


def support_rgb_prior_fields(
    support_paths: Sequence[Path],
    support_feature_maps: Sequence[FeatureMap],
) -> Tuple[np.ndarray, ...]:
    if len(support_paths) != len(support_feature_maps):
        return ()
    return tuple(
        rgb_foreground_proxy(
            read_rgb(path),
            (int(feature_map.values.shape[0]), int(feature_map.values.shape[1])),
        )
        for path, feature_map in zip(support_paths, support_feature_maps)
    )


def collect_support_features(
    backbone: BackboneLike,
    paths: Sequence[Path],
    extraction_config: SupportExtractionConfig,
) -> SupportFeatures:
    feature_maps = collect_support_feature_maps(
        backbone,
        paths,
        extraction_config,
    )
    return flatten_support_feature_maps(
        feature_maps,
        require_contexts=extraction_config.feature.context_source != "none",
        feature_denoiser=None,
    )


def collect_support_feature_maps(
    backbone: BackboneLike,
    paths: Sequence[Path],
    extraction_config: SupportExtractionConfig,
) -> List[FeatureMap]:
    feature_maps: List[FeatureMap] = []
    for path in paths:
        image = read_rgb(path)
        for transform_name in extraction_config.transform_names:
            transformed = transform_rgb(image, transform_name)
            brightness_factor = extraction_config.brightness_range.factor_for(
                len(feature_maps),
                extraction_config.brightness_seed,
            )
            transformed = apply_brightness(transformed, brightness_factor)
            feature_maps.append(
                extract_feature_map_from_rgb(
                    backbone,
                    transformed,
                    extraction_config.feature,
                ),
            )
    return feature_maps


def collect_support_layered_feature_maps(
    backbone: BackboneLike,
    paths: Sequence[Path],
    extraction_config: SupportExtractionConfig,
) -> List[LayeredFeatureMap]:
    feature_maps: List[LayeredFeatureMap] = []
    for path in paths:
        image = read_rgb(path)
        for transform_name in extraction_config.transform_names:
            transformed = transform_rgb(image, transform_name)
            brightness_factor = extraction_config.brightness_range.factor_for(
                len(feature_maps),
                extraction_config.brightness_seed,
            )
            transformed = apply_brightness(transformed, brightness_factor)
            feature_maps.append(
                extract_layer_feature_maps_from_rgb(
                    backbone,
                    transformed,
                    extraction_config.feature,
                ),
            )
    return feature_maps


def flatten_support_feature_maps(
    feature_maps: Sequence[FeatureMap],
    require_contexts: bool,
    feature_denoiser: Optional[PositionMeanArtifactDenoiser],
) -> SupportFeatures:
    features = []
    contexts = []
    for raw_feature_map in feature_maps:
        fmap = apply_feature_denoiser(raw_feature_map, feature_denoiser)
        features.append(fmap.values.reshape(-1, fmap.values.shape[-1]))
        if fmap.contexts is not None:
            contexts.append(fmap.contexts.reshape(-1, fmap.contexts.shape[-1]))
    context_values = None
    if require_contexts:
        if len(contexts) != len(features):
            raise RuntimeError("context_source requested, but the backbone returned no contexts")
        context_values = np.concatenate(contexts, axis=0).astype(np.float32, copy=False)
    return SupportFeatures(
        values=np.concatenate(features, axis=0).astype(np.float32, copy=False),
        contexts=context_values,
    )


def flatten_feature_contexts_from_feature_maps(
    feature_maps: Sequence[FeatureMap],
    context_source: str,
) -> Optional[np.ndarray]:
    contexts = []
    for feature_map in feature_maps:
        feature_contexts = build_feature_contexts(feature_map.values, context_source)
        if feature_contexts is None:
            return None
        contexts.append(feature_contexts.reshape(-1, feature_contexts.shape[-1]))
    if not contexts:
        return None
    return np.concatenate(contexts, axis=0).astype(np.float32, copy=False)


def batch_feature_contexts_from_map(
    feature_map: FeatureMap,
    context_source: str,
) -> Optional[np.ndarray]:
    contexts = build_feature_contexts(feature_map.values, context_source)
    if contexts is None:
        return None
    return contexts[np.newaxis, ...].astype(np.float32, copy=False)


def feature_energy_split_mask(
    feature_maps: Sequence[np.ndarray],
    foreground_quantile: float,
) -> np.ndarray:
    if not feature_maps:
        raise RuntimeError("foreground_flow_mixture requires support feature maps")
    energies = np.concatenate(
        [
            np.linalg.norm(np.asarray(feature_map, dtype=np.float32), axis=-1).reshape(-1)
            for feature_map in feature_maps
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    threshold = np.float32(np.quantile(energies, foreground_quantile))
    mask = energies >= threshold
    if bool(np.any(mask)):
        return mask.astype(np.bool_, copy=False)
    return np.ones(energies.shape, dtype=np.bool_)


def apply_feature_denoiser(
    feature_map: FeatureMap,
    feature_denoiser: Optional[PositionMeanArtifactDenoiser],
) -> FeatureMap:
    if feature_denoiser is None:
        return feature_map
    return FeatureMap(
        values=feature_denoiser.transform(feature_map.values),
        image_shape=feature_map.image_shape,
        contexts=feature_map.contexts,
        transformer_context_tokens=feature_map.transformer_context_tokens,
    )


def artifact_l2_mean(
    feature_denoiser: Optional[PositionMeanArtifactDenoiser],
) -> float:
    if feature_denoiser is None:
        return 0.0
    artifact = feature_denoiser.artifact.reshape(-1, feature_denoiser.artifact.shape[-1])
    return float(np.linalg.norm(artifact, axis=1).mean())


def extract_feature_map_from_rgb(
    backbone: BackboneLike,
    image: np.ndarray,
    extraction_config: FeatureExtractionConfig,
) -> FeatureMap:
    if extraction_config.tiling.enabled:
        return extract_tiled_feature_map_from_rgb(backbone, image, extraction_config)
    return extract_single_feature_map_from_rgb(backbone, image, extraction_config)


def extract_layer_feature_maps_from_rgb(
    backbone: BackboneLike,
    image: np.ndarray,
    extraction_config: FeatureExtractionConfig,
) -> LayeredFeatureMap:
    if extraction_config.tiling.enabled:
        return extract_tiled_layer_feature_maps_from_rgb(backbone, image, extraction_config)
    return extract_single_layer_feature_maps_from_rgb(backbone, image, extraction_config)


def extract_single_layer_feature_maps_from_rgb(
    backbone: BackboneLike,
    image: np.ndarray,
    extraction_config: FeatureExtractionConfig,
) -> LayeredFeatureMap:
    image_tensor, grid_size = backbone.prepare_image(image)
    layer_features = backbone.extract_features(image_tensor)
    if not layer_features:
        message = "Backbone returned no layer features"
        raise RuntimeError(message)
    height, width = grid_size
    return LayeredFeatureMap(
        layers=tuple(
            FeatureMap(
                values=(
                    values := normalize_layer_features(layer).reshape(
                        height,
                        width,
                        layer.shape[-1],
                    )
                ),
                image_shape=(int(image.shape[0]), int(image.shape[1])),
                contexts=build_contexts_for_feature_values(
                    backbone,
                    image_tensor,
                    context_source=extraction_config.context_source,
                    values=values,
                ),
            )
            for layer in layer_features
        ),
    )


def extract_tiled_layer_feature_maps_from_rgb(
    backbone: BackboneLike,
    image: np.ndarray,
    extraction_config: FeatureExtractionConfig,
) -> LayeredFeatureMap:
    resized = resize_rgb(image, extraction_config.tiling.resize_factor)
    tile_size = extraction_config.tiling.patch_size
    if resized.shape[0] <= tile_size and resized.shape[1] <= tile_size:
        layered = extract_single_layer_feature_maps_from_rgb(
            backbone,
            resized,
            extraction_config,
        )
        return LayeredFeatureMap(
            layers=tuple(
                FeatureMap(
                    values=layer.values,
                    image_shape=(int(image.shape[0]), int(image.shape[1])),
                    contexts=layer.contexts,
                )
                for layer in layered.layers
            ),
        )
    y_starts = tile_starts(resized.shape[0], tile_size, extraction_config.tiling.overlap)
    x_starts = tile_starts(resized.shape[1], tile_size, extraction_config.tiling.overlap)
    accumulators: Optional[List[np.ndarray]] = None
    counts: Optional[np.ndarray] = None
    token_stride = 16
    for y_start in y_starts:
        for x_start in x_starts:
            tile = resized[y_start : y_start + tile_size, x_start : x_start + tile_size]
            tile_layers = extract_single_layer_feature_maps_from_rgb(
                backbone,
                tile,
                FeatureExtractionConfig(
                    feature_fusion=extraction_config.feature_fusion,
                    context_source="none",
                ),
            )
            tile_stride = max(1, tile.shape[0] // tile_layers.layers[0].values.shape[0])
            token_stride = min(token_stride, tile_stride)
            y_token = y_start // token_stride
            x_token = x_start // token_stride
            y_stop = y_token + tile_layers.layers[0].values.shape[0]
            x_stop = x_token + tile_layers.layers[0].values.shape[1]
            if accumulators is None:
                height = max(1, resized.shape[0] // token_stride)
                width = max(1, resized.shape[1] // token_stride)
                accumulators = [
                    np.zeros((height, width, layer.values.shape[-1]), dtype=np.float32)
                    for layer in tile_layers.layers
                ]
                counts = np.zeros((height, width, 1), dtype=np.float32)
            if counts is None:
                raise RuntimeError("Tiled layer counts unavailable")
            for layer_index, layer in enumerate(tile_layers.layers):
                accumulators[layer_index][y_token:y_stop, x_token:x_stop] += layer.values
            counts[y_token:y_stop, x_token:x_stop] += 1.0
    if accumulators is None or counts is None:
        raise RuntimeError("Tiled layer extraction produced no feature tiles")
    image_shape = (int(image.shape[0]), int(image.shape[1]))
    return LayeredFeatureMap(
        layers=tuple(
            FeatureMap(
                values=(
                    layer_values := (values / np.maximum(counts, 1.0)).astype(
                        np.float32,
                        copy=False,
                    )
                ),
                image_shape=image_shape,
                contexts=build_image_contexts_for_feature_values(
                    backbone,
                    resized,
                    extraction_config.context_source,
                    values=layer_values,
                ),
            )
            for values in accumulators
        ),
    )


def extract_single_feature_map_from_rgb(
    backbone: BackboneLike,
    image: np.ndarray,
    extraction_config: FeatureExtractionConfig,
) -> FeatureMap:
    image_tensor, grid_size = backbone.prepare_image(image)
    layer_features, transformer_context_tokens = extract_layer_features_with_transformer_context(
        backbone,
        image_tensor,
        extraction_config.transformer_context_mode,
    )
    if not layer_features:
        message = "Backbone returned no features"
        raise RuntimeError(message)
    merged = merge_layer_features(layer_features, extraction_config.feature_fusion)
    height, width = grid_size
    values = merged.reshape(height, width, merged.shape[-1])
    contexts = build_contexts_for_feature_values(
        backbone,
        image_tensor,
        context_source=extraction_config.context_source,
        values=values,
    )
    return FeatureMap(
        values=values,
        image_shape=(int(image.shape[0]), int(image.shape[1])),
        contexts=contexts,
        transformer_context_tokens=transformer_context_tokens,
    )


def extract_layer_features_with_transformer_context(
    backbone: BackboneLike,
    image_tensor: torch.Tensor,
    transformer_context_mode: str,
) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    if not transformer_uses_backbone_context(transformer_context_mode):
        return backbone.extract_features(image_tensor), None
    combined_extractor = getattr(backbone, "extract_features_with_context_tokens", None)
    if callable(combined_extractor):
        typed_extractor = cast(
            "Callable[[torch.Tensor, str], Tuple[List[np.ndarray], np.ndarray]]",
            combined_extractor,
        )
        layer_features, context_tokens = typed_extractor(
            image_tensor,
            transformer_context_mode,
        )
        return layer_features, validate_transformer_context_tokens(context_tokens)
    layer_features = backbone.extract_features(image_tensor)
    context_tokens = extract_transformer_context_tokens_from_tensor(
        backbone,
        image_tensor,
        transformer_context_mode,
    )
    return layer_features, context_tokens


def extract_transformer_context_tokens_from_tensor(
    backbone: BackboneLike,
    image_tensor: torch.Tensor,
    transformer_context_mode: str,
) -> Optional[np.ndarray]:
    if not transformer_uses_backbone_context(transformer_context_mode):
        return None
    token_extractor = getattr(backbone, "extract_context_token_features", None)
    if not callable(token_extractor):
        message = f"Backbone does not support transformer context mode {transformer_context_mode!r}"
        raise TypeError(message)
    typed_extractor = cast("Callable[[torch.Tensor, str], torch.Tensor]", token_extractor)
    tokens = typed_extractor(image_tensor, transformer_context_mode)
    return validate_transformer_context_tokens(tokens.detach().cpu().numpy())


def validate_transformer_context_tokens(tokens: np.ndarray) -> np.ndarray:
    context_tokens = np.asarray(tokens, dtype=np.float32)
    if context_tokens.ndim != 2:
        raise RuntimeError("Transformer context tokens must be shaped TxC")
    return context_tokens.astype(np.float32, copy=False)


def build_contexts_for_feature_values(
    backbone: BackboneLike,
    image_tensor: torch.Tensor,
    context_source: str,
    values: np.ndarray,
) -> Optional[np.ndarray]:
    feature_contexts = build_feature_contexts(values, context_source)
    if feature_contexts is not None:
        return feature_contexts
    return build_broadcast_contexts(
        backbone,
        image_tensor,
        context_source=context_source,
        height=int(values.shape[0]),
        width=int(values.shape[1]),
    )


def build_image_contexts_for_feature_values(
    backbone: BackboneLike,
    image: np.ndarray,
    context_source: str,
    values: np.ndarray,
) -> Optional[np.ndarray]:
    feature_contexts = build_feature_contexts(values, context_source)
    if feature_contexts is not None:
        return feature_contexts
    return build_image_contexts(
        backbone,
        image,
        context_source,
        height=int(values.shape[0]),
        width=int(values.shape[1]),
    )


def build_feature_contexts(values: np.ndarray, context_source: str) -> Optional[np.ndarray]:
    source = context_source.lower()
    include_xy = False
    if source.endswith("_xy"):
        source = source[: -len("_xy")]
        include_xy = True
    if source not in (
        "feature_avg3",
        "feature_avg3_ch16",
        "feature_avg3_residual",
        "image_feature_mean",
        "image_feature_mean_ch16",
    ):
        return None

    values = values.astype(np.float32, copy=False)
    if source == "feature_avg3":
        parts = [average_pool_3x3(values)]
    elif source == "feature_avg3_ch16":
        parts = [channel_group_mean(average_pool_3x3(values), n_groups=16)]
    elif source == "feature_avg3_residual":
        local_average = average_pool_3x3(values)
        parts = [local_average, values - local_average]
    else:
        context_values = (
            channel_group_mean(values, n_groups=16)
            if source == "image_feature_mean_ch16"
            else values
        )
        image_mean = context_values.mean(axis=(0, 1), dtype=np.float32)
        parts = [
            np.broadcast_to(
                image_mean.reshape(1, 1, -1),
                (values.shape[0], values.shape[1], image_mean.shape[0]),
            ).copy(),
        ]

    if include_xy:
        parts.append(patch_xy_contexts(int(values.shape[0]), int(values.shape[1])))
    return np.concatenate(parts, axis=-1).astype(np.float32, copy=False)


def channel_group_mean(values: np.ndarray, n_groups: int) -> np.ndarray:
    group_count = min(max(1, n_groups), int(values.shape[-1]))
    chunks = np.array_split(values, group_count, axis=-1)
    return np.concatenate(
        [chunk.mean(axis=-1, dtype=np.float32, keepdims=True) for chunk in chunks],
        axis=-1,
    ).astype(np.float32, copy=False)


def average_pool_3x3(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values, ((1, 1), (1, 1), (0, 0)), mode="edge")
    pooled = np.zeros_like(values, dtype=np.float32)
    height, width = values.shape[:2]
    for y_offset in range(3):
        for x_offset in range(3):
            pooled += padded[y_offset : y_offset + height, x_offset : x_offset + width]
    pooled /= 9.0
    return pooled.astype(np.float32, copy=False)


def build_broadcast_contexts(
    backbone: BackboneLike,
    image_tensor: torch.Tensor,
    context_source: str,
    height: int,
    width: int,
) -> Optional[np.ndarray]:
    backbone_source, include_xy = split_context_source(context_source)
    context_parts: List[np.ndarray] = []
    if backbone_source != "none":
        context_vector = backbone.extract_context_features(
            image_tensor,
            backbone_source,
        )
        context_array = context_vector.detach().cpu().numpy().astype(np.float32, copy=False)
        context_parts.append(
            np.broadcast_to(
                context_array.reshape(1, 1, -1),
                (height, width, context_array.shape[0]),
            ).copy(),
        )
    if include_xy:
        context_parts.append(patch_xy_contexts(height, width))
    if not context_parts:
        return None
    return np.concatenate(context_parts, axis=-1).astype(np.float32, copy=False)


def build_image_contexts(
    backbone: BackboneLike,
    image: np.ndarray,
    context_source: str,
    height: int,
    width: int,
) -> Optional[np.ndarray]:
    if context_source == "none":
        return None
    image_tensor, _ = backbone.prepare_image(image)
    return build_broadcast_contexts(
        backbone,
        image_tensor,
        context_source,
        height,
        width,
    )


def split_context_source(context_source: str) -> Tuple[str, bool]:
    source = context_source.lower()
    if source == "xy":
        return "none", True
    if source.endswith("_xy"):
        base_source = source[: -len("_xy")]
        if base_source in (
            "cls",
            "register",
            "cls_register",
            "feature_avg3",
            "feature_avg3_ch16",
            "feature_avg3_residual",
            "image_feature_mean",
            "image_feature_mean_ch16",
        ):
            return base_source, True
    if source in (
        "none",
        "cls",
        "register",
        "cls_register",
        "feature_avg3",
        "feature_avg3_ch16",
        "feature_avg3_residual",
        "image_feature_mean",
        "image_feature_mean_ch16",
    ):
        return source, False
    message = f"Unsupported context source: {context_source}"
    raise ValueError(message)


def patch_xy_contexts(height: int, width: int) -> np.ndarray:
    y_values = np.linspace(-1.0, 1.0, num=max(1, height), dtype=np.float32)
    x_values = np.linspace(-1.0, 1.0, num=max(1, width), dtype=np.float32)
    y_grid, x_grid = np.meshgrid(y_values, x_values, indexing="ij")
    return np.stack((x_grid, y_grid), axis=-1).astype(np.float32, copy=False)


def _batch_contexts(feature_map: FeatureMap) -> Optional[np.ndarray]:
    if feature_map.contexts is None:
        return None
    return feature_map.contexts[np.newaxis, ...]


def extract_tiled_feature_map_from_rgb(
    backbone: BackboneLike,
    image: np.ndarray,
    extraction_config: FeatureExtractionConfig,
) -> FeatureMap:
    resized = resize_rgb(image, extraction_config.tiling.resize_factor)
    tile_size = extraction_config.tiling.patch_size
    if resized.shape[0] <= tile_size and resized.shape[1] <= tile_size:
        feature_map = extract_single_feature_map_from_rgb(
            backbone,
            resized,
            extraction_config,
        )
        image_shape = (int(image.shape[0]), int(image.shape[1]))
        return FeatureMap(
            values=feature_map.values,
            image_shape=image_shape,
            contexts=feature_map.contexts,
            transformer_context_tokens=feature_map.transformer_context_tokens,
        )
    y_starts = tile_starts(resized.shape[0], tile_size, extraction_config.tiling.overlap)
    x_starts = tile_starts(resized.shape[1], tile_size, extraction_config.tiling.overlap)
    accumulator: Optional[np.ndarray] = None
    counts: Optional[np.ndarray] = None
    token_stride = 16
    for y_start in y_starts:
        for x_start in x_starts:
            tile = resized[y_start : y_start + tile_size, x_start : x_start + tile_size]
            tile_map = extract_single_feature_map_from_rgb(
                backbone,
                tile,
                FeatureExtractionConfig(
                    feature_fusion=extraction_config.feature_fusion,
                    context_source="none",
                ),
            )
            tile_stride = max(1, tile.shape[0] // tile_map.values.shape[0])
            token_stride = min(token_stride, tile_stride)
            y_token = y_start // token_stride
            x_token = x_start // token_stride
            y_stop = y_token + tile_map.values.shape[0]
            x_stop = x_token + tile_map.values.shape[1]
            if accumulator is None:
                height = max(1, resized.shape[0] // token_stride)
                width = max(1, resized.shape[1] // token_stride)
                accumulator = np.zeros((height, width, tile_map.values.shape[-1]), dtype=np.float32)
                counts = np.zeros((height, width, 1), dtype=np.float32)
            accumulator[y_token:y_stop, x_token:x_stop] += tile_map.values
            if counts is None:
                raise RuntimeError("Tiled counts unavailable")
            counts[y_token:y_stop, x_token:x_stop] += 1.0
    if accumulator is None or counts is None:
        raise RuntimeError("Tiled extraction produced no feature tiles")
    values = accumulator / np.maximum(counts, 1.0)
    image_shape = (int(image.shape[0]), int(image.shape[1]))
    contexts = build_image_contexts_for_feature_values(
        backbone,
        resized,
        extraction_config.context_source,
        values=values,
    )
    image_tensor, _ = backbone.prepare_image(resized)
    transformer_context_tokens = extract_transformer_context_tokens_from_tensor(
        backbone,
        image_tensor,
        extraction_config.transformer_context_mode,
    )
    return FeatureMap(
        values=values.astype(np.float32, copy=False),
        image_shape=image_shape,
        contexts=contexts,
        transformer_context_tokens=transformer_context_tokens,
    )


def tiling_config(config: RunConfig) -> TilingConfig:
    return TilingConfig(
        patch_size=config.tile_patch_size,
        overlap=config.tile_overlap,
        resize_factor=config.image_resize_factor,
    )


def support_extraction_config(config: RunConfig, context_source: str) -> SupportExtractionConfig:
    return SupportExtractionConfig(
        transform_names=config.support_transforms,
        feature=FeatureExtractionConfig(
            feature_fusion=config.feature_fusion,
            context_source=context_source,
            tiling=tiling_config(config),
            transformer_context_mode=config.transformer_context_mode,
        ),
        brightness_range=config.support_brightness_range,
        brightness_seed=config.seed,
    )


def resolve_score_context_mode(config: RunConfig) -> str:
    if config.context_mode == "auto":
        return "soft_penalty" if resolve_memory_context_source(config) != "none" else "none"
    return config.context_mode


def resolve_flow_context_source(config: RunConfig) -> str:
    if config.flow_condition_mode != "context":
        return "none"
    if config.flow_context_source == "auto":
        return config.context_source
    return config.flow_context_source


def resolve_memory_context_source(config: RunConfig) -> str:
    if config.context_mode == "none":
        return "none"
    source = config.memory_context_source
    if source == "auto":
        source = config.context_source
    if config.context_mode == "auto" and source == "none":
        return "none"
    return source


def stream_test_images(test_images: Dict[str, List[str]]) -> List[ImageItem]:
    items = [
        ImageItem(anomaly_type=anomaly_type, path=Path(path))
        for anomaly_type, paths in test_images.items()
        for path in paths
    ]
    return sorted(items, key=lambda item: (item.path.name, item.anomaly_type))


def save_prediction(
    output_root: Path,
    object_name: str,
    item: ImageItem,
    score_map: np.ndarray,
) -> None:
    output_dir = output_root / "anomaly_maps" / object_name / "test" / item.anomaly_type
    output_dir.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(output_dir / f"{item.path.stem}.tiff"), score_map.astype(np.float32))


def save_calibration_prediction(
    output_root: Path,
    object_name: str,
    image_path: Path,
    score_map: np.ndarray,
) -> None:
    output_dir = output_root / "calibration_maps" / object_name / "good"
    output_dir.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(output_dir / f"{image_path.stem}.tiff"), score_map.astype(np.float32))


def write_threshold_split_manifest(
    output_root: Path,
    object_name: str,
    train_paths: Sequence[Path],
    prototype_paths: Sequence[Path],
    threshold_paths: Sequence[Path],
) -> None:
    output_dir = output_root / "threshold_splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "object": object_name,
        "ordered_train_count": len(train_paths),
        "prototype_count": len(prototype_paths),
        "threshold_count": len(threshold_paths),
        "split_rule": "sorted train/good index modulo 8; index%8==0 threshold, else prototype",
        "ordered_train_paths": [str(path) for path in train_paths],
        "prototype_paths": [str(path) for path in prototype_paths],
        "threshold_paths": [str(path) for path in threshold_paths],
    }
    (output_dir / f"{object_name}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
