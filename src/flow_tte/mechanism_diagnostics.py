from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt
import torch
from scipy.stats import spearmanr

from flow_tte.trainer import FlowDensityEstimator


FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class NeighborDiagnostics:
    identity_retention: float
    clean_margin_mean: float
    shifted_margin_mean: float
    margin_delta: float
    source_entropy: float
    source_gini: float
    normalized_position_displacement: float
    tie_rate: float


@dataclass(frozen=True)
class GeometryDiagnostics:
    pair_distance_spearman: float
    raw_effective_rank: float
    latent_effective_rank: float
    latent_rms: float
    logdet_q05: float
    logdet_q50: float
    logdet_q95: float
    sampled_rows: int
    sample_sha256: str


def _validate_knn(indices: npt.ArrayLike, distances: npt.ArrayLike) -> tuple[IntArray, FloatArray]:
    index_array = np.asarray(indices, dtype=np.int64)
    distance_array = np.asarray(distances, dtype=np.float32)
    if index_array.ndim != 2 or distance_array.shape != index_array.shape:
        raise ValueError("indices and distances must share shape (patches, k)")
    if index_array.shape[1] < 2:
        raise ValueError("neighbor diagnostics require k >= 2")
    if not bool(np.all(np.isfinite(distance_array))):
        raise ValueError("neighbor distances must be finite")
    return index_array, distance_array


def _normalized_entropy(counts: npt.ArrayLike) -> float:
    values = np.asarray(counts, dtype=np.float64)
    positive = values[values > 0]
    if positive.size <= 1:
        return 0.0
    probabilities = positive / positive.sum()
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(values.size))


def _gini(counts: npt.ArrayLike) -> float:
    values = np.asarray(counts, dtype=np.float64)
    if values.size == 0 or float(values.sum()) == 0.0:
        return 0.0
    ordered = np.sort(values)
    ranks = np.arange(1, ordered.size + 1, dtype=np.float64)
    return float(
        (2.0 * np.sum(ranks * ordered) / (ordered.size * ordered.sum()))
        - (ordered.size + 1.0) / ordered.size
    )


def neighbor_diagnostics(
    clean_indices: npt.ArrayLike,
    clean_distances: npt.ArrayLike,
    shifted_indices: npt.ArrayLike,
    shifted_distances: npt.ArrayLike,
    *,
    patches_per_source: int,
    grid: tuple[int, int],
    source_count: int,
    tie_tolerance: float = 1e-7,
) -> NeighborDiagnostics:
    """Summarize paired clean/shift nearest-neighbor stability.

    Memory rows are assumed to be source-major and row-major within each
    source, matching ``flatten_support_feature_maps``.
    """

    clean_i, clean_d = _validate_knn(clean_indices, clean_distances)
    shifted_i, shifted_d = _validate_knn(shifted_indices, shifted_distances)
    if clean_i.shape != shifted_i.shape:
        raise ValueError("clean and shifted neighbor arrays must match")
    height, width = grid
    if patches_per_source != height * width or patches_per_source <= 0:
        raise ValueError("patches_per_source must equal grid height times width")
    if source_count <= 0:
        raise ValueError("source_count must be positive")
    if int(max(clean_i.max(initial=0), shifted_i.max(initial=0))) >= (
        source_count * patches_per_source
    ):
        raise ValueError("neighbor index exceeds the declared support bank")

    clean_first = clean_i[:, 0]
    shifted_first = shifted_i[:, 0]
    identity_retention = float(np.mean(clean_first == shifted_first))
    clean_margin = clean_d[:, 1] - clean_d[:, 0]
    shifted_margin = shifted_d[:, 1] - shifted_d[:, 0]
    shifted_sources = shifted_first // patches_per_source
    counts = np.bincount(shifted_sources, minlength=source_count)

    query_positions = np.arange(shifted_first.size, dtype=np.int64) % patches_per_source
    bank_positions = shifted_first % patches_per_source
    query_y, query_x = np.divmod(query_positions, width)
    bank_y, bank_x = np.divmod(bank_positions, width)
    diagonal = max(math.hypot(max(height - 1, 1), max(width - 1, 1)), 1.0)
    displacement = np.hypot(query_y - bank_y, query_x - bank_x) / diagonal
    tie_rate = float(np.mean(np.abs(shifted_margin) <= tie_tolerance))
    return NeighborDiagnostics(
        identity_retention=identity_retention,
        clean_margin_mean=float(clean_margin.mean()),
        shifted_margin_mean=float(shifted_margin.mean()),
        margin_delta=float(shifted_margin.mean() - clean_margin.mean()),
        source_entropy=_normalized_entropy(counts),
        source_gini=_gini(counts),
        normalized_position_displacement=float(displacement.mean()),
        tie_rate=tie_rate,
    )


def effective_rank(values: npt.ArrayLike) -> float:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError("effective_rank expects a non-empty matrix")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = np.square(singular)
    if float(energy.sum()) <= 0.0:
        return 0.0
    probabilities = energy / energy.sum()
    positive = probabilities[probabilities > 0]
    return float(np.exp(-(positive * np.log(positive)).sum()))


def _pair_distances(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        return np.empty(0, dtype=np.float64)
    delta = values[:, None, :] - values[None, :, :]
    upper = np.triu_indices(values.shape[0], k=1)
    return np.linalg.norm(delta[upper], axis=1)


def derive_logdet(z: npt.ArrayLike, nll: npt.ArrayLike) -> FloatArray:
    latent = np.asarray(z, dtype=np.float32)
    negative_log_likelihood = np.asarray(nll, dtype=np.float32).reshape(-1)
    if latent.ndim != 2 or latent.shape[0] != negative_log_likelihood.size:
        raise ValueError("z and nll row counts must match")
    base_nll = 0.5 * np.sum(np.square(latent), axis=1)
    base_nll += 0.5 * latent.shape[1] * math.log(2.0 * math.pi)
    return (base_nll - negative_log_likelihood).astype(np.float32, copy=False)


def geometry_diagnostics(
    estimator: FlowDensityEstimator,
    features: npt.ArrayLike,
    *,
    sample_rows: int,
    seed: int,
) -> GeometryDiagnostics:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("features must be a non-empty 2D matrix")
    if sample_rows <= 1:
        raise ValueError("sample_rows must exceed one")
    rng = np.random.default_rng(seed)
    count = min(sample_rows, matrix.shape[0])
    indices = np.sort(rng.choice(matrix.shape[0], size=count, replace=False))
    sampled = matrix[indices]
    evaluation = estimator.evaluate(sampled)
    latent = evaluation.z.detach().cpu().numpy().astype(np.float32, copy=False)
    nll = evaluation.nll.detach().cpu().numpy().astype(np.float32, copy=False)
    raw_distances = _pair_distances(sampled)
    latent_distances = _pair_distances(latent)
    correlation = spearmanr(raw_distances, latent_distances).statistic
    if not np.isfinite(correlation):
        correlation = 1.0 if np.array_equal(raw_distances, latent_distances) else 0.0
    logdet = derive_logdet(latent, nll)
    import hashlib

    sample_sha256 = hashlib.sha256(indices.astype(np.int64).tobytes()).hexdigest()
    return GeometryDiagnostics(
        pair_distance_spearman=float(correlation),
        raw_effective_rank=effective_rank(sampled),
        latent_effective_rank=effective_rank(latent),
        latent_rms=float(np.sqrt(np.mean(np.square(latent)))),
        logdet_q05=float(np.quantile(logdet, 0.05)),
        logdet_q50=float(np.quantile(logdet, 0.50)),
        logdet_q95=float(np.quantile(logdet, 0.95)),
        sampled_rows=count,
        sample_sha256=sample_sha256,
    )


def cosine_displacement(clean: npt.ArrayLike, shifted: npt.ArrayLike) -> float:
    left = np.asarray(clean, dtype=np.float32)
    right = np.asarray(shifted, dtype=np.float32)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("clean and shifted features must share shape (patches, dim)")
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denominator = np.maximum(left_norm * right_norm, 1e-12)
    cosine = np.sum(left * right, axis=1) / denominator
    return float(np.mean(1.0 - np.clip(cosine, -1.0, 1.0)))


def dvt_variance_removed(
    raw_maps: Sequence[npt.ArrayLike],
    denoised_maps: Sequence[npt.ArrayLike],
) -> float:
    if not raw_maps or len(raw_maps) != len(denoised_maps):
        raise ValueError("raw and denoised map sequences must be non-empty and matched")
    raw = np.stack([np.asarray(value, dtype=np.float32) for value in raw_maps])
    denoised = np.stack([np.asarray(value, dtype=np.float32) for value in denoised_maps])
    if raw.shape != denoised.shape:
        raise ValueError("raw and denoised maps must share shape")
    raw_variance = float(np.var(raw, axis=0).mean())
    denoised_variance = float(np.var(denoised, axis=0).mean())
    if raw_variance <= 1e-12:
        return 0.0
    return float((raw_variance - denoised_variance) / raw_variance)


def assert_static_memory(before: torch.Tensor, after: torch.Tensor) -> None:
    if before.shape != after.shape or not torch.equal(before, after):
        raise RuntimeError("static support memory mutated during mechanism scoring")
