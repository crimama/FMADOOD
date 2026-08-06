# ruff: noqa: COM812, E501, EM102, PLC0415, PLR0913, RUF046, TC006

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence, cast

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]

_EPS = 1e-12


@dataclass(frozen=True)
class SemanticAtlasConfig:
    n_clusters: int = 8
    beta: float = 0.25
    semantic_tangent_rank: int = 4
    min_cluster_sources: int = 6
    min_cluster_rows: int = 512
    orthogonal_residual_min_ratio: float = 0.25
    neighbor_k: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        if not 0.0 <= self.beta < 1.0:
            raise ValueError("beta must be in [0, 1)")
        if self.semantic_tangent_rank < 0:
            raise ValueError("semantic_tangent_rank must be non-negative")
        if self.min_cluster_sources <= 0:
            raise ValueError("min_cluster_sources must be positive")
        if self.min_cluster_rows <= 0:
            raise ValueError("min_cluster_rows must be positive")
        if not 0.0 <= self.orthogonal_residual_min_ratio <= 1.0:
            raise ValueError("orthogonal_residual_min_ratio must be in [0, 1]")
        if self.neighbor_k <= 0:
            raise ValueError("neighbor_k must be positive")


@dataclass(frozen=True)
class ClusterBasis:
    cluster: int
    center: FloatArray
    shift_basis: FloatArray
    tangent_basis: FloatArray
    valid: bool
    reason: str
    row_count: int
    source_count: int
    shift_energy: float
    residual_norm_ratio: float

    @property
    def direction(self) -> FloatArray:
        if self.shift_basis.shape[1] == 0:
            return np.zeros((self.center.shape[0],), dtype=np.float32)
        return np.asarray(self.shift_basis[:, 0], dtype=np.float32)


@dataclass(frozen=True)
class SemanticAtlasMetric:
    config: SemanticAtlasConfig
    centers: FloatArray
    bases: tuple[ClusterBasis, ...]

    @property
    def dim(self) -> int:
        return int(self.centers.shape[1])

    @property
    def valid_cluster_count(self) -> int:
        return sum(int(item.valid) for item in self.bases)


def fit_semantic_atlas(
    clean: npt.ArrayLike,
    shifted: npt.ArrayLike,
    sources: Sequence[object],
    config: SemanticAtlasConfig | None = None,
    *,
    cluster_reference: npt.ArrayLike | None = None,
    centers: npt.ArrayLike | None = None,
) -> SemanticAtlasMetric:
    """Fit the support-only semantic atlas on aligned clean/shifted latents."""
    resolved = config or SemanticAtlasConfig()
    clean_matrix = _finite_matrix("clean", clean)
    shifted_matrix = _finite_matrix("shifted", shifted)
    if clean_matrix.shape != shifted_matrix.shape:
        raise ValueError("clean and shifted must share shape")
    source_ids = _source_ids(sources, clean_matrix.shape[0])
    if centers is not None and cluster_reference is not None:
        raise ValueError("provide either centers or cluster_reference, not both")
    if centers is not None:
        center_matrix = _finite_matrix("centers", centers)
        if center_matrix.shape != (resolved.n_clusters, clean_matrix.shape[1]):
            raise ValueError("centers must have shape [n_clusters, feature_dim]")
    else:
        reference = clean_matrix if cluster_reference is None else _finite_matrix("cluster_reference", cluster_reference)
        if reference.shape[1] != clean_matrix.shape[1]:
            raise ValueError("cluster_reference feature dimension must match clean")
        _, center_matrix = deterministic_kmeans(reference, resolved.n_clusters, seed=resolved.seed)
    labels = assign_clusters(clean_matrix, center_matrix)
    residuals = shifted_matrix - clean_matrix
    bases: list[ClusterBasis] = []
    for cluster in range(resolved.n_clusters):
        mask = labels == cluster
        bases.append(
            _fit_cluster_basis(
                cluster=cluster,
                center=center_matrix[cluster],
                clean=clean_matrix[mask],
                residuals=residuals[mask],
                sources=source_ids[mask],
                config=resolved,
            )
        )
    return SemanticAtlasMetric(config=resolved, centers=center_matrix, bases=tuple(bases))


def fit_unorthogonalized_atlas(
    clean: npt.ArrayLike,
    shifted: npt.ArrayLike,
    sources: Sequence[object],
    config: SemanticAtlasConfig | None = None,
    *,
    cluster_reference: npt.ArrayLike | None = None,
    centers: npt.ArrayLike | None = None,
) -> SemanticAtlasMetric:
    resolved = config or SemanticAtlasConfig()
    clean_matrix = _finite_matrix("clean", clean)
    shifted_matrix = _finite_matrix("shifted", shifted)
    atlas = fit_semantic_atlas(
        clean_matrix,
        shifted_matrix,
        sources,
        resolved,
        cluster_reference=cluster_reference,
        centers=centers,
    )
    labels = assign_clusters(clean_matrix, atlas.centers)
    bases = []
    for item in atlas.bases:
        if item.reason == "orthogonal_residual_too_small":
            raw = _top_pca_basis(shifted_matrix[labels == item.cluster] - clean_matrix[labels == item.cluster], rank=1, center=False)
            valid = raw.shape[1] == 1
            bases.append(
                ClusterBasis(
                    cluster=item.cluster,
                    center=item.center,
                    shift_basis=raw,
                    tangent_basis=np.zeros((item.center.shape[0], 0), dtype=np.float32),
                    valid=valid,
                    reason="valid" if valid else "no_shift_basis",
                    row_count=item.row_count,
                    source_count=item.source_count,
                    shift_energy=item.shift_energy,
                    residual_norm_ratio=1.0 if valid else 0.0,
                )
            )
        else:
            bases.append(
                ClusterBasis(
                    cluster=item.cluster,
                    center=item.center,
                    shift_basis=_raw_shift_basis_for_cluster(clean_matrix, shifted_matrix, atlas.centers, item.cluster),
                    tangent_basis=np.zeros((item.center.shape[0], 0), dtype=np.float32),
                    valid=item.reason in {"valid", "orthogonal_residual_too_small"},
                    reason="valid" if item.reason in {"valid", "orthogonal_residual_too_small"} else item.reason,
                    row_count=item.row_count,
                    source_count=item.source_count,
                    shift_energy=item.shift_energy,
                    residual_norm_ratio=1.0 if item.reason in {"valid", "orthogonal_residual_too_small"} else 0.0,
                )
            )
    return SemanticAtlasMetric(config=resolved, centers=atlas.centers, bases=tuple(bases))


def fit_global_rank_metric(
    clean: npt.ArrayLike,
    shifted: npt.ArrayLike,
    *,
    rank: int = 4,
    beta: float = 0.25,
    seed: int = 0,
) -> SemanticAtlasMetric:
    """Fit a beta-matched global low-rank metric control."""
    clean_matrix = _finite_matrix("clean", clean)
    shifted_matrix = _finite_matrix("shifted", shifted)
    if clean_matrix.shape != shifted_matrix.shape:
        raise ValueError("clean and shifted must share shape")
    if rank <= 0:
        raise ValueError("rank must be positive")
    basis = _top_pca_basis(shifted_matrix - clean_matrix, rank=rank, center=False)
    config = SemanticAtlasConfig(
        n_clusters=1,
        beta=beta,
        semantic_tangent_rank=0,
        min_cluster_sources=1,
        min_cluster_rows=1,
        seed=seed,
    )
    center = clean_matrix.mean(axis=0).astype(np.float32)
    valid = basis.shape[1] > 0
    basis_obj = ClusterBasis(
        cluster=0,
        center=center,
        shift_basis=basis,
        tangent_basis=np.zeros((clean_matrix.shape[1], 0), dtype=np.float32),
        valid=valid,
        reason="valid" if valid else "no_shift_basis",
        row_count=int(clean_matrix.shape[0]),
        source_count=1,
        shift_energy=projected_energy_ratio(shifted_matrix - clean_matrix, basis),
        residual_norm_ratio=1.0 if valid else 0.0,
    )
    return SemanticAtlasMetric(
        config=config,
        centers=center.reshape(1, -1).astype(np.float32),
        bases=(basis_obj,),
    )


def shuffled_atlas(atlas: SemanticAtlasMetric, *, seed: int) -> SemanticAtlasMetric:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(atlas.bases))
    if len(order) > 1 and np.all(order == np.arange(len(order))):
        order = np.roll(order, 1)
    bases = []
    for cluster, src_index in enumerate(order):
        src = atlas.bases[int(src_index)]
        bases.append(
            ClusterBasis(
                cluster=cluster,
                center=atlas.centers[cluster],
                shift_basis=src.shift_basis.copy(),
                tangent_basis=src.tangent_basis.copy(),
                valid=src.valid,
                reason=src.reason,
                row_count=src.row_count,
                source_count=src.source_count,
                shift_energy=src.shift_energy,
                residual_norm_ratio=src.residual_norm_ratio,
            )
        )
    return SemanticAtlasMetric(config=atlas.config, centers=atlas.centers.copy(), bases=tuple(bases))


def random_atlas_like(atlas: SemanticAtlasMetric, *, seed: int) -> SemanticAtlasMetric:
    bases = []
    for item in atlas.bases:
        basis = deterministic_random_basis(item.center.shape[0], 1, seed=_stable_seed(seed, item.cluster))
        bases.append(
            ClusterBasis(
                cluster=item.cluster,
                center=item.center.copy(),
                shift_basis=basis,
                tangent_basis=np.zeros((item.center.shape[0], 0), dtype=np.float32),
                valid=item.valid,
                reason="valid" if item.valid else item.reason,
                row_count=item.row_count,
                source_count=item.source_count,
                shift_energy=0.0,
                residual_norm_ratio=1.0 if item.valid else 0.0,
            )
        )
    return SemanticAtlasMetric(config=atlas.config, centers=atlas.centers.copy(), bases=tuple(bases))


def assign_clusters(values: npt.ArrayLike, centers: npt.ArrayLike) -> IntArray:
    matrix = _finite_matrix("values", values)
    center_matrix = _finite_matrix("centers", centers)
    if matrix.shape[1] != center_matrix.shape[1]:
        raise ValueError("values and centers feature dimensions must match")
    distances = _squared_distances(matrix, center_matrix)
    return np.asarray(np.argmin(distances, axis=1), dtype=np.int64)


def semantic_atlas_1nn(
    queries: npt.ArrayLike,
    memory: npt.ArrayLike,
    atlas: SemanticAtlasMetric | None,
    *,
    memory_labels: npt.ArrayLike | None = None,
    chunk_size: int = 512,
) -> tuple[FloatArray, IntArray]:
    """Return memory-conditioned 1-NN distances and indices."""
    query_matrix = _finite_matrix("queries", queries)
    memory_matrix = _finite_matrix("memory", memory)
    if query_matrix.shape[1] != memory_matrix.shape[1]:
        raise ValueError("queries and memory feature dimensions must match")
    if atlas is None:
        return _chunked_1nn(query_matrix, memory_matrix, None, None, chunk_size=chunk_size)
    labels = (
        assign_clusters(memory_matrix, atlas.centers)
        if memory_labels is None
        else np.asarray(memory_labels, dtype=np.int64)
    )
    if labels.shape != (memory_matrix.shape[0],):
        raise ValueError("memory_labels must have one label per memory row")
    return _chunked_1nn(query_matrix, memory_matrix, atlas, labels, chunk_size=chunk_size)


def loo_distances(
    memory: npt.ArrayLike,
    atlas: SemanticAtlasMetric | None,
    *,
    memory_labels: npt.ArrayLike | None = None,
    chunk_size: int = 512,
) -> FloatArray:
    matrix = _finite_matrix("memory", memory)
    labels = None if atlas is None else (
        assign_clusters(matrix, atlas.centers)
        if memory_labels is None
        else np.asarray(memory_labels, dtype=np.int64)
    )
    values = np.empty((matrix.shape[0],), dtype=np.float32)
    for start in range(0, matrix.shape[0], chunk_size):
        end = min(start + chunk_size, matrix.shape[0])
        distances = _distance_matrix_for_chunk(matrix[start:end], matrix, atlas, labels, chunk_size=chunk_size)
        row = np.arange(end - start)
        distances[row, np.arange(start, end)] = np.inf
        values[start:end] = np.min(distances, axis=1)
    return values


def projected_energy_ratio(residuals: npt.ArrayLike, basis: npt.ArrayLike) -> float:
    residual_matrix = _finite_matrix("residuals", residuals)
    basis_matrix = np.asarray(basis, dtype=np.float32)
    if basis_matrix.size == 0:
        return 0.0
    if basis_matrix.ndim == 1:
        basis_matrix = basis_matrix.reshape(-1, 1)
    if basis_matrix.shape[0] != residual_matrix.shape[1]:
        raise ValueError("basis feature dimension mismatch")
    total = float(np.sum(np.square(residual_matrix, dtype=np.float64)))
    if total <= _EPS:
        return 0.0
    projected = residual_matrix @ basis_matrix
    return float(np.sum(np.square(projected, dtype=np.float64)) / total)


def hard_neighbor_retention(
    reference: npt.ArrayLike,
    candidate: npt.ArrayLike,
    *,
    k: int = 10,
    max_samples: int = 2048,
) -> float:
    ref = _finite_matrix("reference", reference)
    cand = _finite_matrix("candidate", candidate)
    if ref.shape != cand.shape:
        raise ValueError("reference and candidate must share shape")
    if ref.shape[0] <= 1:
        return 1.0
    sample = _sample_indices(ref.shape[0], max_samples)
    k_eff = min(k, ref.shape[0] - 1)
    ref_dist = _euclidean_distance_matrix(ref[sample], ref)
    cand_dist = _euclidean_distance_matrix(cand[sample], cand)
    for row, index in enumerate(sample):
        ref_dist[row, index] = np.inf
        cand_dist[row, index] = np.inf
    ref_nn = np.argpartition(ref_dist, kth=k_eff - 1, axis=1)[:, :k_eff]
    cand_nn = np.argpartition(cand_dist, kth=k_eff - 1, axis=1)[:, :k_eff]
    overlaps = [
        len(set(ref_nn[row].tolist()) & set(cand_nn[row].tolist())) / float(k_eff)
        for row in range(sample.shape[0])
    ]
    return float(np.mean(overlaps))


def pair_distance_spearman(
    reference: npt.ArrayLike,
    candidate: npt.ArrayLike,
    *,
    max_pairs: int = 4096,
    seed: int = 0,
) -> float:
    ref = _finite_matrix("reference", reference)
    cand = _finite_matrix("candidate", candidate)
    if ref.shape != cand.shape:
        raise ValueError("reference and candidate must share shape")
    if ref.shape[0] <= 1:
        return 1.0
    rng = np.random.default_rng(seed)
    left = rng.integers(0, ref.shape[0], size=max_pairs)
    right = rng.integers(0, ref.shape[0] - 1, size=max_pairs)
    right = right + (right >= left)
    ref_dist = np.linalg.norm(ref[left] - ref[right], axis=1)
    cand_dist = np.linalg.norm(cand[left] - cand[right], axis=1)
    return _spearman(ref_dist, cand_dist)


def basis_diagnostics(atlas: SemanticAtlasMetric, *, row_labels: npt.ArrayLike | None = None) -> dict[str, float | int | bool]:
    valid = [item for item in atlas.bases if item.valid]
    max_orth_error = 0.0
    min_eigen = 1.0
    for item in atlas.bases:
        basis = item.shift_basis
        if basis.size:
            gram = basis.T @ basis
            max_orth_error = max(max_orth_error, float(np.max(np.abs(gram - np.eye(gram.shape[0])))))
            if item.valid:
                min_eigen = min(min_eigen, 1.0 - atlas.config.beta)
    disabled_patch_mass = 0.0
    if row_labels is not None:
        labels = np.asarray(row_labels, dtype=np.int64)
        if labels.ndim != 1:
            raise ValueError("row_labels must be a 1D array")
        invalid = {item.cluster for item in atlas.bases if not item.valid}
        disabled_patch_mass = float(np.mean(np.isin(labels, list(invalid)))) if labels.size else 0.0
    return {
        "valid_cluster_count": int(len(valid)),
        "cluster_count": int(len(atlas.bases)),
        "max_orthonormal_error": max_orth_error,
        "min_metric_eigenvalue": min_eigen,
        "disabled_patch_mass": disabled_patch_mass,
        "finite": bool(
            np.all(np.isfinite(atlas.centers))
            and all(np.all(np.isfinite(item.shift_basis)) for item in atlas.bases)
        ),
    }


def deterministic_kmeans(values: npt.ArrayLike, n_clusters: int, *, seed: int) -> tuple[IntArray, FloatArray]:
    matrix = _finite_matrix("values", values)
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    if matrix.shape[0] < n_clusters:
        raise ValueError("n_clusters cannot exceed row count")
    try:
        from sklearn.cluster import KMeans
    except ImportError as error:  # pragma: no cover - sklearn is part of the experiment env
        del error
        return _numpy_kmeans(matrix, n_clusters, seed=seed)
    model = KMeans(n_clusters=n_clusters, random_state=int(seed), n_init=10, algorithm="lloyd")
    labels = model.fit_predict(matrix)
    centers = np.asarray(model.cluster_centers_, dtype=np.float32)
    return np.asarray(labels, dtype=np.int64), centers


def _numpy_kmeans(
    matrix: FloatArray,
    n_clusters: int,
    *,
    seed: int,
    max_iter: int = 100,
) -> tuple[IntArray, FloatArray]:
    rng = np.random.default_rng(seed)
    centers = _kmeans_plus_plus_init(matrix, n_clusters, rng)
    labels = np.zeros((matrix.shape[0],), dtype=np.int64)
    for _ in range(max_iter):
        next_labels = np.argmin(_squared_distances(matrix, centers), axis=1).astype(np.int64)
        next_centers = centers.copy()
        for cluster in range(n_clusters):
            mask = next_labels == cluster
            if np.any(mask):
                next_centers[cluster] = matrix[mask].mean(axis=0)
            else:
                farthest = int(np.argmax(np.min(_squared_distances(matrix, next_centers), axis=1)))
                next_centers[cluster] = matrix[farthest]
        if np.array_equal(next_labels, labels) and np.allclose(next_centers, centers):
            centers = next_centers.astype(np.float32)
            labels = next_labels
            break
        centers = next_centers.astype(np.float32)
        labels = next_labels
    return labels, centers.astype(np.float32, copy=False)


def _kmeans_plus_plus_init(matrix: FloatArray, n_clusters: int, rng: np.random.Generator) -> FloatArray:
    centers = np.empty((n_clusters, matrix.shape[1]), dtype=np.float32)
    first = int(rng.integers(0, matrix.shape[0]))
    centers[0] = matrix[first]
    closest = _squared_distances(matrix, centers[:1]).reshape(-1)
    for cluster in range(1, n_clusters):
        total = float(closest.sum())
        if total <= _EPS:
            candidate = int(cluster % matrix.shape[0])
        else:
            probabilities = closest.astype(np.float64) / total
            probabilities = probabilities / probabilities.sum()
            candidate = int(rng.choice(matrix.shape[0], p=probabilities))
        centers[cluster] = matrix[candidate]
        closest = np.minimum(closest, _squared_distances(matrix, centers[cluster : cluster + 1]).reshape(-1))
    return centers


def deterministic_random_basis(dim: int, rank: int, *, seed: int) -> FloatArray:
    if dim <= 0 or rank <= 0:
        raise ValueError("dim and rank must be positive")
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dim, rank)).astype(np.float64)
    q, _ = np.linalg.qr(matrix, mode="reduced")
    return np.asarray(q[:, :rank], dtype=np.float32)


def _fit_cluster_basis(
    *,
    cluster: int,
    center: FloatArray,
    clean: FloatArray,
    residuals: FloatArray,
    sources: IntArray,
    config: SemanticAtlasConfig,
) -> ClusterBasis:
    dim = int(center.shape[0])
    row_count = int(clean.shape[0])
    source_count = int(np.unique(sources).size)
    empty = np.zeros((dim, 0), dtype=np.float32)
    if row_count < config.min_cluster_rows:
        return _invalid_cluster(cluster, center, "too_few_rows", row_count, source_count)
    if source_count < config.min_cluster_sources:
        return _invalid_cluster(cluster, center, "too_few_sources", row_count, source_count)
    raw = _top_pca_basis(residuals, rank=1, center=False)
    if raw.shape[1] == 0:
        return _invalid_cluster(cluster, center, "no_shift_basis", row_count, source_count)
    tangent = _clean_tangent_basis(clean, config)
    projected = raw[:, 0]
    if tangent.shape[1] > 0:
        projected = projected - tangent @ (tangent.T @ projected)
    raw_norm = max(float(np.linalg.norm(raw[:, 0])), _EPS)
    projected_norm = float(np.linalg.norm(projected))
    ratio = projected_norm / raw_norm
    if ratio < config.orthogonal_residual_min_ratio:
        return ClusterBasis(
            cluster=cluster,
            center=center.astype(np.float32, copy=True),
            shift_basis=empty,
            tangent_basis=tangent,
            valid=False,
            reason="orthogonal_residual_too_small",
            row_count=row_count,
            source_count=source_count,
            shift_energy=projected_energy_ratio(residuals, raw),
            residual_norm_ratio=ratio,
        )
    direction = (projected / max(projected_norm, _EPS)).reshape(-1, 1).astype(np.float32)
    return ClusterBasis(
        cluster=cluster,
        center=center.astype(np.float32, copy=True),
        shift_basis=direction,
        tangent_basis=tangent,
        valid=True,
        reason="valid",
        row_count=row_count,
        source_count=source_count,
        shift_energy=projected_energy_ratio(residuals, direction),
        residual_norm_ratio=ratio,
    )


def _invalid_cluster(
    cluster: int,
    center: FloatArray,
    reason: str,
    row_count: int,
    source_count: int,
) -> ClusterBasis:
    dim = int(center.shape[0])
    return ClusterBasis(
        cluster=cluster,
        center=center.astype(np.float32, copy=True),
        shift_basis=np.zeros((dim, 0), dtype=np.float32),
        tangent_basis=np.zeros((dim, 0), dtype=np.float32),
        valid=False,
        reason=reason,
        row_count=row_count,
        source_count=source_count,
        shift_energy=0.0,
        residual_norm_ratio=0.0,
    )


def _raw_shift_basis_for_cluster(
    clean: npt.ArrayLike,
    shifted: npt.ArrayLike,
    centers: npt.ArrayLike,
    cluster: int,
) -> FloatArray:
    clean_matrix = _finite_matrix("clean", clean)
    shifted_matrix = _finite_matrix("shifted", shifted)
    labels = assign_clusters(clean_matrix, centers)
    return _top_pca_basis(shifted_matrix[labels == cluster] - clean_matrix[labels == cluster], rank=1, center=False)


def _clean_tangent_basis(clean: FloatArray, config: SemanticAtlasConfig) -> FloatArray:
    if config.semantic_tangent_rank == 0 or clean.shape[0] <= 1:
        return np.zeros((clean.shape[1], 0), dtype=np.float32)
    distances = _squared_distances(clean, clean)
    np.fill_diagonal(distances, np.inf)
    k_eff = min(config.neighbor_k, clean.shape[0] - 1)
    neighbors = np.argpartition(distances, kth=k_eff - 1, axis=1)[:, :k_eff]
    row = np.repeat(np.arange(clean.shape[0]), k_eff)
    col = neighbors.reshape(-1)
    diffs = clean[row] - clean[col]
    return _top_pca_basis(diffs, rank=config.semantic_tangent_rank, center=True)


def _top_pca_basis(values: npt.ArrayLike, *, rank: int, center: bool) -> FloatArray:
    matrix = _finite_matrix("values", values)
    if rank <= 0 or matrix.shape[0] < 2:
        return np.zeros((matrix.shape[1], 0), dtype=np.float32)
    work = matrix.astype(np.float64, copy=False)
    if center:
        work = work - work.mean(axis=0, keepdims=True)
    covariance = work.T @ work / max(float(work.shape[0]), 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    keep = [idx for idx in order if float(eigenvalues[idx]) > _EPS][:rank]
    if not keep:
        return np.zeros((matrix.shape[1], 0), dtype=np.float32)
    basis = eigenvectors[:, keep]
    q, _ = np.linalg.qr(basis, mode="reduced")
    return np.asarray(q[:, : len(keep)], dtype=np.float32)


def _chunked_1nn(
    queries: FloatArray,
    memory: FloatArray,
    atlas: SemanticAtlasMetric | None,
    labels: npt.NDArray[np.integer] | None,
    *,
    chunk_size: int,
) -> tuple[FloatArray, IntArray]:
    best = np.empty((queries.shape[0],), dtype=np.float32)
    best_idx = np.empty((queries.shape[0],), dtype=np.int64)
    for start in range(0, queries.shape[0], chunk_size):
        end = min(start + chunk_size, queries.shape[0])
        distances = _distance_matrix_for_chunk(queries[start:end], memory, atlas, labels, chunk_size=chunk_size)
        indices = np.argmin(distances, axis=1)
        best[start:end] = distances[np.arange(end - start), indices]
        best_idx[start:end] = indices.astype(np.int64, copy=False)
    return best, best_idx


def _distance_matrix_for_chunk(
    queries: FloatArray,
    memory: FloatArray,
    atlas: SemanticAtlasMetric | None,
    labels: npt.NDArray[np.integer] | None,
    *,
    chunk_size: int,
) -> FloatArray:
    if atlas is None:
        return _euclidean_distance_matrix(queries, memory, chunk_size=chunk_size)
    resolved_labels = assign_clusters(memory, atlas.centers) if labels is None else np.asarray(labels, dtype=np.int64)
    output = np.empty((queries.shape[0], memory.shape[0]), dtype=np.float32)
    for cluster, basis in enumerate(atlas.bases):
        columns = np.flatnonzero(resolved_labels == cluster)
        if columns.size == 0:
            continue
        output[:, columns] = _local_distance_matrix(
            queries,
            memory[columns],
            basis,
            beta=atlas.config.beta,
            chunk_size=chunk_size,
        )
    return output


def _local_distance_matrix(
    queries: FloatArray,
    memory: FloatArray,
    basis: ClusterBasis,
    *,
    beta: float,
    chunk_size: int,
) -> FloatArray:
    if not basis.valid or basis.shift_basis.size == 0:
        return _euclidean_distance_matrix(queries, memory, chunk_size=chunk_size)
    output = np.empty((queries.shape[0], memory.shape[0]), dtype=np.float32)
    u = basis.shift_basis.astype(np.float32, copy=False)
    for start in range(0, queries.shape[0], chunk_size):
        end = min(start + chunk_size, queries.shape[0])
        raw_d2 = _squared_distances(queries[start:end], memory)
        query_proj = queries[start:end] @ u
        memory_proj = memory @ u
        proj_d2 = _squared_distances(query_proj.astype(np.float32), memory_proj.astype(np.float32))
        shrink = np.float32(2.0 * beta - beta * beta)
        output[start:end] = np.sqrt(np.maximum(raw_d2 - shrink * proj_d2, np.float32(0.0))).astype(np.float32)
    return output


def _euclidean_distance_matrix(
    left: FloatArray,
    right: FloatArray,
    *,
    chunk_size: int = 512,
) -> FloatArray:
    output = np.empty((left.shape[0], right.shape[0]), dtype=np.float32)
    for start in range(0, left.shape[0], chunk_size):
        end = min(start + chunk_size, left.shape[0])
        output[start:end] = np.sqrt(_squared_distances(left[start:end], right)).astype(np.float32)
    return output


def _squared_distances(left: FloatArray, right: FloatArray) -> FloatArray:
    left_norm = np.sum(left * left, axis=1, keepdims=True)
    right_norm = np.sum(right * right, axis=1, keepdims=True).T
    distances = left_norm + right_norm - np.float32(2.0) * (left @ right.T)
    return np.maximum(distances, np.float32(0.0)).astype(np.float32)


def _finite_matrix(name: str, value: npt.ArrayLike) -> FloatArray:
    matrix = cast(FloatArray, np.asarray(value, dtype=np.float32))
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _source_ids(sources: Sequence[object], row_count: int) -> IntArray:
    if len(sources) != row_count:
        raise ValueError("sources must have one entry per row")
    ids = {value: index for index, value in enumerate(dict.fromkeys(sources))}
    return np.asarray([ids[value] for value in sources], dtype=np.int64)


def _sample_indices(row_count: int, max_samples: int) -> IntArray:
    if row_count <= max_samples:
        return np.arange(row_count, dtype=np.int64)
    return np.linspace(0, row_count - 1, num=max_samples).round().astype(np.int64)


def _spearman(left: npt.ArrayLike, right: npt.ArrayLike) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("spearman inputs must be same-shape vectors")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    if denom <= _EPS:
        return 1.0
    return float(np.dot(rx, ry) / denom)


def _rankdata(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def summarize_energy_capture(
    residuals_by_name: Mapping[str, npt.ArrayLike],
    atlas: SemanticAtlasMetric,
    labels: npt.ArrayLike,
) -> dict[str, float]:
    resolved_labels = np.asarray(labels, dtype=np.int64)
    output: dict[str, float] = {}
    for name, residuals in residuals_by_name.items():
        matrix = _finite_matrix(name, residuals)
        if matrix.shape[0] != resolved_labels.shape[0]:
            raise ValueError(f"{name}: row count must match labels")
        weighted = []
        weights = []
        for basis in atlas.bases:
            mask = resolved_labels == basis.cluster
            if not np.any(mask):
                continue
            weighted.append(projected_energy_ratio(matrix[mask], basis.shift_basis))
            weights.append(int(np.sum(mask)))
        output[name] = float(np.average(weighted, weights=weights)) if weighted else 0.0
    return output
