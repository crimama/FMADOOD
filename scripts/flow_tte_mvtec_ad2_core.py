from __future__ import annotations

# allow: SIZE_OK — remote experiment core keeps protocol args and artifact writing together
# until the MVTec AD2 runner interface stabilizes.
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np
import tifffile as tiff
import torch

from flow_tte.config import ExpansionConfig, FlowConfig, FlowTTEConfig, ScoreConfig
from flow_tte.denoising import PositionMeanArtifactDenoiser, fit_feature_denoiser
from flow_tte.pipeline import FlowTTE

if __package__:
    from .flow_tte_superadd_preprocess import (
        BrightnessRange,
        TilingConfig,
        apply_brightness,
        resize_rgb,
        tile_starts,
    )
    from .flow_tte_support import (
        merge_layer_features,
        read_rgb,
        select_support_paths_for_backbone,
        transform_rgb,
    )
else:
    from flow_tte_superadd_preprocess import (
        BrightnessRange,
        TilingConfig,
        apply_brightness,
        resize_rgb,
        tile_starts,
    )
    from flow_tte_support import (
        merge_layer_features,
        read_rgb,
        select_support_paths_for_backbone,
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
    pro_integration_limit: float
    cleanup_maps: bool
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


@dataclass(frozen=True)
class FeatureExtractionConfig:
    feature_fusion: str
    context_source: str = "none"
    tiling: TilingConfig = field(default_factory=TilingConfig)


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


@dataclass(frozen=True)
class SupportFeatures:
    values: np.ndarray
    contexts: Optional[np.ndarray]


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
    elapsed_seconds: float


def run_object(
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
    pipeline = FlowTTE(
        FlowTTEConfig(
            flow=FlowConfig(
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
            ),
            expansion=ExpansionConfig(
                budget=config.expansion_budget,
                density_quantile=config.density_quantile,
                random_seed=config.seed,
            ),
            score=ScoreConfig(
                distance_weight=config.distance_weight,
                density_weight=config.density_weight,
                score_mode=config.score_mode,
                context_mode=resolve_score_context_mode(config),
                context_weight=config.context_weight,
                context_top_m=config.context_top_m,
                top_percent=config.top_percent,
                query_chunk_size=config.query_chunk_size,
                use_squared_distance=config.use_squared_distance,
            ),
            device=config.device,
        ),
    )
    support_feature_maps = collect_support_feature_maps(
        backbone,
        selected_paths,
        support_extraction_config(config, flow_context_source),
    )
    feature_denoiser = fit_feature_denoiser(
        config.dvt_denoise_mode,
        [feature_map.values for feature_map in support_feature_maps],
        alpha=config.dvt_denoise_alpha,
    )
    support_features = flatten_support_feature_maps(
        support_feature_maps,
        require_contexts=flow_context_source != "none",
        feature_denoiser=feature_denoiser,
    )
    support_memory_contexts = support_features.contexts
    if memory_context_source != flow_context_source:
        support_memory_contexts = None
        if memory_context_source != "none":
            support_memory_features = collect_support_features(
                backbone,
                selected_paths,
                support_extraction_config(config, memory_context_source),
            )
            support_memory_contexts = support_memory_features.contexts
    stats = pipeline.fit(
        support_features.values,
        support_contexts=support_features.contexts,
        memory_contexts=support_memory_contexts,
    )
    test_images = dataset.get_test_images(object_name, split="test_public")
    items = stream_test_images(test_images)
    selected_patch_counts: List[int] = []
    first_memory_size = 0
    last_memory_size = 0
    for item in items:
        feature_map = apply_feature_denoiser(
            extract_feature_map_from_rgb(
                backbone,
                read_rgb(item.path),
                FeatureExtractionConfig(
                    feature_fusion=config.feature_fusion,
                    context_source=flow_context_source,
                    tiling=tiling_config(config),
                ),
            ),
            feature_denoiser,
        )
        batch_contexts = None
        if feature_map.contexts is not None:
            batch_contexts = feature_map.contexts[np.newaxis, ...]
        memory_contexts = batch_contexts
        if memory_context_source != flow_context_source:
            memory_contexts = None
            if memory_context_source != "none":
                memory_map = extract_feature_map_from_rgb(
                    backbone,
                    read_rgb(item.path),
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
        first_memory_size = first_memory_size or result.memory_size_before
        last_memory_size = result.memory_size_after
        selected_patch_counts.append(result.selected_count)
        score_map = cv2.resize(
            result.patch_scores[0],
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
        mean_selected_patch_count=float(np.mean(selected_patch_counts)),
        initial_memory_size=first_memory_size,
        final_memory_size=last_memory_size,
        train_nll_mean=stats.train_nll_mean,
        train_nll_std=stats.train_nll_std,
        density_threshold=stats.density_threshold,
        dvt_denoise_mode=config.dvt_denoise_mode,
        dvt_denoise_alpha=config.dvt_denoise_alpha,
        dvt_artifact_l2_mean=artifact_l2_mean(feature_denoiser),
        elapsed_seconds=time.time() - started,
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


def extract_single_feature_map_from_rgb(
    backbone: BackboneLike,
    image: np.ndarray,
    extraction_config: FeatureExtractionConfig,
) -> FeatureMap:
    image_tensor, grid_size = backbone.prepare_image(image)
    layer_features = backbone.extract_features(image_tensor)
    if not layer_features:
        message = "Backbone returned no features"
        raise RuntimeError(message)
    merged = merge_layer_features(layer_features, extraction_config.feature_fusion)
    height, width = grid_size
    contexts = None
    if extraction_config.context_source != "none":
        context_vector = backbone.extract_context_features(
            image_tensor,
            extraction_config.context_source,
        )
        context_array = context_vector.detach().cpu().numpy().astype(np.float32, copy=False)
        contexts = np.broadcast_to(
            context_array.reshape(1, 1, -1),
            (height, width, context_array.shape[0]),
        ).copy()
    return FeatureMap(
        values=merged.reshape(height, width, merged.shape[-1]),
        image_shape=(int(image.shape[0]), int(image.shape[1])),
        contexts=contexts,
    )


def extract_tiled_feature_map_from_rgb(
    backbone: BackboneLike,
    image: np.ndarray,
    extraction_config: FeatureExtractionConfig,
) -> FeatureMap:
    if extraction_config.context_source != "none":
        message = "Tiled extraction currently supports patch features only, not context tokens"
        raise RuntimeError(message)
    resized = resize_rgb(image, extraction_config.tiling.resize_factor)
    tile_size = extraction_config.tiling.patch_size
    if resized.shape[0] <= tile_size and resized.shape[1] <= tile_size:
        feature_map = extract_single_feature_map_from_rgb(
            backbone,
            resized,
            extraction_config,
        )
        image_shape = (int(image.shape[0]), int(image.shape[1]))
        return FeatureMap(values=feature_map.values, image_shape=image_shape, contexts=None)
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
    return FeatureMap(values=values.astype(np.float32, copy=False), image_shape=image_shape)


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
