from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from flow_tte.metrics import MetricInputError as ScoringInputError

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]

_NORM_EPS = 1e-6
@dataclass(frozen=True)  # noqa: SLOTS_OK -- Python 3.8 has no dataclass slots.
class LocalCandidateSet:
    values: FloatArray
    valid: BoolArray


@dataclass(frozen=True)  # noqa: SLOTS_OK -- Python 3.8 has no dataclass slots.
class MicroScoreBatch:
    scores: FloatArray
    valid: BoolArray


@dataclass(frozen=True)  # noqa: SLOTS_OK -- Python 3.8 has no dataclass slots.
class FusionInputs:
    coarse_evidence: FloatArray
    micro_evidence: FloatArray
    confidence: FloatArray
    micro_valid: BoolArray


def upper_tail_evidence(reference_scores: FloatArray, values: FloatArray) -> FloatArray:
    reference: FloatArray = _finite_float(reference_scores, "reference_scores").reshape(-1)
    query: FloatArray = _finite_float(values, "values")
    if reference.size == 0:
        raise ScoringInputError("reference_scores must be non-empty")
    ordered: FloatArray = np.sort(reference)
    lower_indices: npt.NDArray[np.intp] = np.searchsorted(ordered, query, side="left")
    upper_counts: npt.NDArray[np.intp] = reference.size - lower_indices
    probabilities: FloatArray = np.asarray(
        (1.0 + upper_counts) / float(reference.size + 1), dtype=np.float32,
    )
    return np.asarray(-np.log(probabilities), dtype=np.float32)


def g0_cosine_scores(query: FloatArray, memory: FloatArray, chunk_size: int) -> FloatArray:
    query_matrix = _normalized_matrix(query, "query")
    memory_matrix = _normalized_matrix(memory, "memory")
    if query_matrix.shape[1] != memory_matrix.shape[1]:
        raise ScoringInputError("query and memory feature dimensions must match")
    if chunk_size < 1:
        raise ScoringInputError("chunk_size must be positive")
    scores: FloatArray = np.empty(len(query_matrix), dtype=np.float32)
    for start in range(0, len(query_matrix), chunk_size):
        end = min(start + chunk_size, len(query_matrix))
        similarities: FloatArray = np.asarray(
            query_matrix[start:end] @ memory_matrix.T,
            dtype=np.float32,
        )
        best: FloatArray = np.asarray(np.max(similarities, axis=1), dtype=np.float32)
        scores[start:end] = 1.0 - np.clip(best, -1.0, 1.0)
    return scores


def leave_one_image_out_g0(
    feature_images: Sequence[FloatArray],
    chunk_size: int,
) -> Tuple[FloatArray, ...]:
    matrices = tuple(_matrix(values, "feature image") for values in feature_images)
    if len(matrices) < 2:
        raise ScoringInputError("leave-one-image-out scoring requires at least two images")
    dimensions = {matrix.shape[1] for matrix in matrices}
    if len(dimensions) != 1:
        raise ScoringInputError("all feature images must share one feature dimension")
    results: List[FloatArray] = []
    for index, query in enumerate(matrices):
        memory = np.concatenate(matrices[:index] + matrices[index + 1 :], axis=0)
        results.append(g0_cosine_scores(query, memory, chunk_size))
    return tuple(results)


def local_min_cosine_scores(
    query: FloatArray,
    candidates: LocalCandidateSet,
) -> MicroScoreBatch:
    query_matrix, values, candidate_valid = _local_inputs(query, candidates)
    query_norms: FloatArray = np.asarray(np.linalg.norm(query_matrix, axis=1), dtype=np.float32)
    candidate_norms: FloatArray = np.asarray(np.linalg.norm(values, axis=3), dtype=np.float32)
    finite_candidates: BoolArray = np.asarray(
        np.all(np.isfinite(values), axis=3),
        dtype=np.bool_,
    )
    usable: BoolArray = candidate_valid & finite_candidates & (candidate_norms >= _NORM_EPS)
    support_valid: BoolArray = np.asarray(np.any(usable, axis=2), dtype=np.bool_)
    output_valid: BoolArray = np.asarray(
        (np.sum(support_valid, axis=1) >= 3) & (query_norms >= _NORM_EPS),
        dtype=np.bool_,
    )
    safe_queries: FloatArray = np.asarray(
        query_matrix / np.maximum(query_norms[:, None], _NORM_EPS),
        dtype=np.float32,
    )
    safe_candidates: FloatArray = np.asarray(
        values / np.maximum(candidate_norms[..., None], _NORM_EPS),
        dtype=np.float32,
    )
    similarities: FloatArray = np.asarray(
        np.sum(safe_candidates * safe_queries[:, None, None, :], axis=3),
        dtype=np.float32,
    )
    similarities[~usable] = -np.inf
    best: FloatArray = np.asarray(
        np.max(similarities.reshape(len(query_matrix), -1), axis=1),
        dtype=np.float32,
    )
    scores: FloatArray = np.zeros(len(query_matrix), dtype=np.float32)
    scores[output_valid] = 1.0 - np.clip(best[output_valid], -1.0, 1.0)
    return MicroScoreBatch(scores=scores, valid=np.asarray(output_valid, dtype=np.bool_))


def r1_cosine_scores(
    query: FloatArray,
    candidates: LocalCandidateSet,
) -> MicroScoreBatch:
    query_matrix, values, candidate_valid = _local_inputs(query, candidates)
    scores: FloatArray = np.zeros(len(query_matrix), dtype=np.float32)
    output_valid: BoolArray = np.zeros(len(query_matrix), dtype=np.bool_)
    for query_index in range(len(query_matrix)):
        query_vector: FloatArray = np.asarray(query_matrix[query_index], dtype=np.float32)
        query_candidates: FloatArray = np.asarray(values[query_index], dtype=np.float32)
        query_valid: BoolArray = np.asarray(candidate_valid[query_index], dtype=np.bool_)
        prototypes: List[FloatArray] = []
        for support_index in range(len(query_candidates)):
            usable: BoolArray = np.asarray(query_valid[support_index], dtype=np.bool_)
            support_values: FloatArray = query_candidates[support_index, usable]
            support_values = support_values[np.all(np.isfinite(support_values), axis=1)]
            if support_values.shape[0] == 0:
                continue
            prototype: FloatArray = np.asarray(np.median(support_values, axis=0), dtype=np.float32)
            norm = float(np.linalg.norm(prototype))
            if norm >= _NORM_EPS:
                prototypes.append(np.asarray(prototype / norm, dtype=np.float32))
        if len(prototypes) < 3:
            continue
        stacked: FloatArray = np.asarray(np.stack(prototypes), dtype=np.float32)
        prototype, converged = _geometric_median(stacked)
        query_norm = float(np.linalg.norm(query_vector))
        prototype_norm = float(np.linalg.norm(prototype))
        if not converged or query_norm < _NORM_EPS or prototype_norm < _NORM_EPS:
            continue
        product: FloatArray = np.asarray(
            (query_vector / query_norm) * (prototype / prototype_norm), dtype=np.float32,
        )
        cosine = float(np.sum(product))
        scores[query_index] = 1.0 - max(-1.0, min(1.0, cosine))
        output_valid[query_index] = True
    return MicroScoreBatch(scores=scores, valid=output_valid)


def fuse_confidence(inputs: FusionInputs) -> FloatArray:
    coarse = _finite_float(inputs.coarse_evidence, "coarse_evidence")
    micro = _finite_float(inputs.micro_evidence, "micro_evidence")
    confidence = _finite_float(inputs.confidence, "confidence")
    valid: BoolArray = np.asarray(inputs.micro_valid, dtype=np.bool_)
    shapes_match = coarse.shape == micro.shape == confidence.shape == valid.shape
    if not shapes_match:
        raise ScoringInputError("fusion arrays must have identical shapes")
    if np.any(confidence < 0.0) or np.any(confidence > 1.0):
        raise ScoringInputError("confidence must be in [0, 1]")
    fused = np.maximum(coarse, confidence * micro)
    return np.asarray(np.where(valid, fused, coarse), dtype=np.float32)


def _geometric_median(points: FloatArray) -> Tuple[FloatArray, bool]:
    current: FloatArray = np.asarray(np.mean(points, axis=0), dtype=np.float32)
    for _ in range(50):
        dists: FloatArray = np.asarray(np.linalg.norm(points - current, axis=1), dtype=np.float32)
        weights: FloatArray = np.asarray(
            1.0 / np.maximum(dists, _NORM_EPS), dtype=np.float32,
        )
        updated: FloatArray = np.asarray(
            np.sum(points * weights[:, None], axis=0) / np.sum(weights),
            dtype=np.float32,
        )
        if float(np.linalg.norm(updated - current)) <= 1e-5:
            return updated, True
        current = updated
    return current, False


def _local_inputs(
    query: FloatArray,
    candidates: LocalCandidateSet,
) -> Tuple[FloatArray, FloatArray, BoolArray]:
    query_matrix = _matrix(query, "query")
    values: FloatArray = np.asarray(candidates.values, dtype=np.float32)
    valid: BoolArray = np.asarray(candidates.valid, dtype=np.bool_)
    if values.ndim != 4 or valid.shape != values.shape[:3]:
        raise ScoringInputError("candidates require values (N,S,C,D) and valid (N,S,C)")
    if values.shape[0] != query_matrix.shape[0] or values.shape[3] != query_matrix.shape[1]:
        raise ScoringInputError("candidate query count and feature dimension must match query")
    return query_matrix, values, valid


def _matrix(values: FloatArray, name: str) -> FloatArray:
    matrix: FloatArray = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        reason = f"{name} must be a non-empty 2D array"
        raise ScoringInputError(reason)
    return matrix


def _normalized_matrix(values: FloatArray, name: str) -> FloatArray:
    matrix = _matrix(values, name)
    norms: FloatArray = np.asarray(np.linalg.norm(matrix, axis=1), dtype=np.float32)
    if not np.all(np.isfinite(matrix)) or np.any(norms < _NORM_EPS):
        reason = f"{name} must contain finite non-zero features"
        raise ScoringInputError(reason)
    return np.asarray(matrix / norms[:, None], dtype=np.float32)


def _finite_float(values: FloatArray, name: str) -> FloatArray:
    array: FloatArray = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        reason = f"{name} must contain only finite values"
        raise ScoringInputError(reason)
    return array
