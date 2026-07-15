"""Scoring, coreset, and threshold functions for the official SuperADD baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Sequence, Tuple, TypedDict

import numpy as np
import torch
from torch.nn import functional

if TYPE_CHECKING:
    from pathlib import Path

OFFICIAL_COMMIT: Final = "44cf25144442fbbc1334ea59d1632327a4376d1a"
OFFICIAL_SOURCE_SHA256: Final = "6696fb0d4454850390624f461e12c838117362925f4b2f502084869877f9ae0c"


class SuperADDParityError(ValueError):
    """Raised when parity inputs violate the frozen official protocol."""


@dataclass(frozen=True)
class TrainPartition:
    prototypes: Tuple[Path, ...]
    threshold: Tuple[Path, ...]


@dataclass(frozen=True)
class CoresetConfig:
    target_count: int = 100_000
    iterations: int = 100
    knn_neighbors: int = 100


_DEFAULT_CORESET = CoresetConfig()


@dataclass(frozen=True)
class ModelProvenance:
    model_id: str
    revision: str
    model_class: str
    patch_size: int
    depth: int
    register_count: int
    config_sha256: str
    resolved_config_sha256: str
    weight_sha256: str
    transformers_version: str


@dataclass(frozen=True)
class ManifestContext:
    category: str
    resource_protocol: str
    support_paths: Tuple[Path, ...]
    partition: TrainPartition
    model: ModelProvenance
    implementation_sha256: str
    used_early_exit: bool


class ModelManifest(TypedDict):
    model_id: str
    revision: str
    model_class: str
    patch_size: int
    depth: int
    register_count: int
    config_sha256: str
    resolved_config_sha256: str
    weight_sha256: str
    transformers_version: str


class ParityManifest(TypedDict):
    category: str
    resource_protocol: str
    official_commit: str
    source_sha256: str
    method_identity: str
    model: ModelManifest
    support_paths: Tuple[str, ...]
    prototype_paths: Tuple[str, ...]
    threshold_paths: Tuple[str, ...]
    adaptations: Tuple[str, ...]
    algorithm_contract_matched: bool
    official_runtime_comparable: bool
    resource_comparable: bool
    implementation_sha256: str
    used_early_exit: bool


def partition_training_paths(
    paths: Sequence[Path],
    threshold_fraction: int = 8,
) -> TrainPartition:
    """Hold out indices divisible by threshold_fraction, preserving input order."""
    if threshold_fraction < 2:
        raise SuperADDParityError("threshold_fraction must be at least two")
    ordered = tuple(paths)
    if not ordered:
        raise SuperADDParityError("training paths must be non-empty")
    return TrainPartition(
        prototypes=tuple(path for index, path in enumerate(ordered) if index % threshold_fraction),
        threshold=tuple(
            path for index, path in enumerate(ordered) if not index % threshold_fraction
        ),
    )


def nearest_distance_map(
    query_grid: torch.Tensor,
    prototypes: torch.Tensor,
    query_chunk_size: int = 1024,
) -> torch.Tensor:
    """Return official unsquared, unnormalized Euclidean 1-NN distance per channel."""
    if query_grid.ndim == 4 and int(query_grid.shape[0]) == 1:
        query_grid = query_grid[0]
    if query_grid.ndim != 3 or prototypes.ndim != 2:
        raise SuperADDParityError("query grid and prototypes must be [H,W,C] and [N,C]")
    height, width, channels = (int(value) for value in query_grid.shape)
    if int(prototypes.shape[1]) != channels or query_chunk_size < 1:
        raise SuperADDParityError("feature dimensions must agree and chunk size must be positive")
    flat = query_grid.reshape(-1, channels).to(device=prototypes.device, dtype=torch.float32)
    keys = prototypes.to(dtype=torch.float32)
    minima = [
        torch.cdist(flat[start : start + query_chunk_size], keys).min(dim=1).values
        for start in range(0, int(flat.shape[0]), query_chunk_size)
    ]
    return torch.cat(minima).reshape(height, width) / channels


def layerwise_anomaly_map(
    layer_maps: Sequence[torch.Tensor],
    native_size: Tuple[int, int],
    evaluation_downscale: int = 4,
) -> torch.Tensor:
    """Bilinearly resize each layer map, then apply official arithmetic-mean fusion."""
    if not layer_maps or evaluation_downscale < 1:
        raise SuperADDParityError("layer maps must be non-empty and downscale positive")
    output = (
        native_size[0] // evaluation_downscale,
        native_size[1] // evaluation_downscale,
    )
    if min(output) < 1:
        raise SuperADDParityError("native image is too small for evaluation_downscale")
    stacked = torch.stack(tuple(layer_maps), dim=0).to(dtype=torch.float32)
    resized = functional.interpolate(
        stacked.unsqueeze(0),
        size=output,
        mode="bilinear",
        align_corners=False,
    )[0]
    return resized.mean(dim=0).cpu()


def fixed_threshold(
    calibration_maps: Sequence[np.ndarray],
    percentile: float = 95.0,
    factor: float = 1.421,
) -> float:
    """Compute official percentile-times-factor threshold over calibration pixels."""
    if not calibration_maps or not 0.0 <= percentile <= 100.0 or factor <= 0:
        raise SuperADDParityError("invalid calibration maps, percentile, or factor")
    flat = np.concatenate(tuple(np.asarray(score).reshape(-1) for score in calibration_maps))
    return float(np.percentile(flat, percentile) * factor)


def subsample_distance_based(
    features: np.ndarray,
    device: torch.device,
    rng: np.random.RandomState,
    config: CoresetConfig = _DEFAULT_CORESET,
) -> np.ndarray:
    """Reproduce official iterative distance-based coreset sampling."""
    if (
        features.ndim != 2
        or config.target_count < 1
        or config.iterations < 1
        or config.knn_neighbors < 1
    ):
        raise SuperADDParityError("invalid coreset shape or sampling configuration")
    if len(features) <= config.target_count:
        return features
    kept = np.zeros(len(features), dtype=np.bool_)
    subset_size = int(len(features) / config.iterations)
    subset_target = config.target_count // config.iterations
    if subset_size < 1 or subset_target < 1:
        raise SuperADDParityError("official iterative split requires positive subset sizes")
    for _ in range(config.iterations):
        candidates = np.flatnonzero(~kept)
        indices = rng.choice(candidates, size=min(subset_size, len(candidates)), replace=False)
        keep_subset = _subsample_subset(
            features[indices],
            subset_target,
            device,
            rng,
            config.knn_neighbors,
        )
        kept[indices] = keep_subset
    difference = config.target_count - int(kept.sum())
    if difference > 0:
        additions = np.flatnonzero(~kept)
        rng.shuffle(additions)
        kept[additions[:difference]] = True
    return features[kept]


def subsample_knn_score_rank(
    features: torch.Tensor,
    target_count: int = 100_000,
    knn_neighbors: int = 100,
    query_chunk_size: int = 256,
) -> torch.Tensor:
    """Apply the paper's exact global k-NN score-ranking selection."""
    if (
        features.ndim != 2
        or target_count < 1
        or knn_neighbors < 1
        or query_chunk_size < 1
    ):
        raise SuperADDParityError("invalid k-NN score coreset configuration")
    if len(features) <= target_count:
        return features
    neighbor_count = min(knn_neighbors, len(features))
    knn_distances = torch.empty(
        (len(features), neighbor_count),
        dtype=torch.float32,
        device="cpu",
    )
    distance_sum = 0.0
    for start in range(0, len(features), query_chunk_size):
        end = min(start + query_chunk_size, len(features))
        distances = (
            torch.cdist(
                features[start:end],
                features,
                compute_mode="use_mm_for_euclid_dist",
            )
            .topk(k=neighbor_count, dim=-1, largest=False, sorted=False)
            .values.to(device="cpu", dtype=torch.float32)
        )
        knn_distances[start:end] = distances
        distance_sum += float(distances.to(dtype=torch.float64).sum().item())
    threshold = distance_sum / knn_distances.numel()
    scores = np.sum(
        knn_distances.numpy() < threshold,
        axis=-1,
        dtype=np.int32,
    )
    selected = np.argsort(scores, kind="stable")[:target_count]
    selected_indices = torch.from_numpy(selected.copy()).to(features.device)
    return features[selected_indices]


def _subsample_subset(
    features: np.ndarray,
    target_count: int,
    device: torch.device,
    rng: np.random.RandomState,
    knn_neighbors: int,
) -> np.ndarray:
    values = torch.from_numpy(features).to(device=device, dtype=torch.float32)
    neighbor_count = min(knn_neighbors, len(features))
    distances = (
        torch.cdist(values, values, compute_mode="use_mm_for_euclid_dist")
        .topk(
            k=neighbor_count,
            dim=-1,
            largest=False,
            sorted=False,
        )
        .values.cpu()
        .numpy()
    )
    target_distance = float(np.mean(distances, dtype=np.float64) / 10.0)
    random_values = rng.rand(len(features))
    sample_count = target_count + 1
    keep = np.ones(len(features), dtype=np.bool_)
    while sample_count > target_count:
        factors = np.sum(distances < target_distance, axis=-1) + 1
        keep = random_values < (1.0 / factors)
        sample_count = int(keep.sum())
        target_distance *= 1.1
    return keep


def build_parity_manifest(
    context: ManifestContext,
) -> ParityManifest:
    """Record source identity and the limits of the HF execution adaptation."""
    if context.resource_protocol not in {"Pfull", "P16-official-native-split"}:
        raise SuperADDParityError(
            "resource protocol must be Pfull or P16-official-native-split",
        )
    is_official_full = context.resource_protocol == "Pfull"
    adaptations = ["hf_backbone_adapter", "chunked_exact_query_cdist"]
    adaptations.append("skip_coreset_below_cap_no_score_effect")
    if not is_official_full:
        adaptations.append("official_index_modulo_8_split")
    return {
        "category": context.category,
        "resource_protocol": context.resource_protocol,
        "official_commit": OFFICIAL_COMMIT,
        "source_sha256": OFFICIAL_SOURCE_SHA256,
        "method_identity": "superadd_hf_adaptation",
        "model": {
            "model_id": context.model.model_id,
            "revision": context.model.revision,
            "model_class": context.model.model_class,
            "patch_size": context.model.patch_size,
            "depth": context.model.depth,
            "register_count": context.model.register_count,
            "config_sha256": context.model.config_sha256,
            "resolved_config_sha256": context.model.resolved_config_sha256,
            "weight_sha256": context.model.weight_sha256,
            "transformers_version": context.model.transformers_version,
        },
        "support_paths": tuple(str(path) for path in context.support_paths),
        "prototype_paths": tuple(str(path) for path in context.partition.prototypes),
        "threshold_paths": tuple(str(path) for path in context.partition.threshold),
        "adaptations": tuple(adaptations),
        "algorithm_contract_matched": True,
        "official_runtime_comparable": False,
        "resource_comparable": is_official_full,
        "implementation_sha256": context.implementation_sha256,
        "used_early_exit": context.used_early_exit,
    }
