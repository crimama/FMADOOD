from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from typing import Final, NamedTuple, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch

from flow_tte.darc_gate2_calibration import (
    calibrate_rung_evidence as calibrate_rung_evidence,  # noqa: PLC0414 -- re-export
)
from flow_tte.darc_gate2_scoring_types import (
    RungNormalReferences as RungNormalReferences,  # noqa: PLC0414 -- compatibility re-export
)
from flow_tte.darc_gate2_scoring_types import (
    RungScores,
    RungScoringConfig,
    RungScoringInput,
    SupportValidityAudit,
)
from flow_tte.darc_scoring import (
    LocalCandidateSet,
    MicroScoreBatch,
    g0_cosine_scores,
    local_min_cosine_scores,
)
from flow_tte.metrics import MetricInputError as ScoringInputError

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]

_NORM_EPS: Final = 1e-6


class _TorchLocalRungs(NamedTuple):
    l0: MicroScoreBatch
    l1: MicroScoreBatch
    l0_support: BoolArray
    l1_support: BoolArray


def score_rungs(inputs: RungScoringInput, config: RungScoringConfig) -> RungScores:
    """Score one shared token/support population through the frozen Gate 2 ladder."""
    if config.g0_chunk_size < 1 or config.r1_chunk_size < 1:
        raise ScoringInputError("rung scoring chunk sizes must be positive")
    query, l0_values, l0_valid = _candidate_arrays(inputs.query, inputs.identity_candidates)
    _, l1_values, l1_valid = _candidate_arrays(query, inputs.aligned_candidates)
    if l0_values.shape != l1_values.shape:
        raise ScoringInputError("L0 and L1 candidates must have identical ordered populations")

    torch_local: Optional[_TorchLocalRungs] = None
    if config.device == "cpu":
        l0_support = _local_support_validity(l0_values, l0_valid)
        l1_support = _local_support_validity(l1_values, l1_valid)
    else:
        torch_local = _local_rungs_torch(
            query,
            l0_values,
            l0_valid,
            l1_values,
            l1_valid,
            config.device,
        )
        l0_support = torch_local.l0_support
        l1_support = torch_local.l1_support
    shared_support: BoolArray = np.asarray(l0_support & l1_support, dtype=np.bool_)
    shared_mask = shared_support[:, :, None]
    l0_shared = LocalCandidateSet(l0_values, np.asarray(l0_valid & shared_mask, dtype=np.bool_))
    l1_shared = LocalCandidateSet(l1_values, np.asarray(l1_valid & shared_mask, dtype=np.bool_))

    if torch_local is None:
        l0_batch = local_min_cosine_scores(query, l0_shared)
        l1_batch = local_min_cosine_scores(query, l1_shared)
    else:
        l0_batch = torch_local.l0
        l1_batch = torch_local.l1
    r1_batch, r1_support = _r1_with_audit(
        query,
        l1_shared,
        config.r1_chunk_size,
        config.device,
    )
    common_valid: BoolArray = np.asarray(
        l0_batch.valid & l1_batch.valid & r1_batch.valid,
        dtype=np.bool_,
    )
    fallback: BoolArray = np.asarray(~common_valid, dtype=np.bool_)
    g0: FloatArray
    g0_valid: BoolArray
    if inputs.precomputed_g0 is not None:
        precomputed: FloatArray = np.asarray(inputs.precomputed_g0, dtype=np.float32)
        if precomputed.shape != (len(query),) or not np.all(np.isfinite(precomputed)):
            raise ScoringInputError("precomputed G0 must be a finite vector matching query tokens")
        g0 = precomputed
        g0_valid = np.ones(len(query), dtype=np.bool_)
    elif config.complete_g0:
        g0 = g0_cosine_scores(query, inputs.global_memory, config.g0_chunk_size)
        g0_valid = np.ones(len(query), dtype=np.bool_)
    else:
        g0 = np.zeros(len(query), dtype=np.float32)
        g0_valid = np.asarray(fallback, dtype=np.bool_)
        if np.any(fallback):
            g0[fallback] = g0_cosine_scores(
                query[fallback],
                inputs.global_memory,
                config.g0_chunk_size,
            )
    l0: FloatArray = np.asarray(np.where(fallback, g0, l0_batch.scores), dtype=np.float32)
    l1: FloatArray = np.asarray(np.where(fallback, g0, l1_batch.scores), dtype=np.float32)
    r1: FloatArray = np.asarray(np.where(fallback, g0, r1_batch.scores), dtype=np.float32)
    if not all(np.all(np.isfinite(values)) for values in (g0, l0, l1, r1)):
        raise ScoringInputError("rung scoring produced non-finite residuals")
    return RungScores(
        g0=np.asarray(g0, dtype=np.float32),
        g0_valid=g0_valid,
        l0=l0,
        l1=l1,
        r1=r1,
        common_fallback=fallback,
        support_validity=SupportValidityAudit(l0_support, l1_support, shared_support, r1_support),
    )


def r1_cosine_scores_chunked(
    query: FloatArray,
    candidates: LocalCandidateSet,
    chunk_size: int,
    device: str = "cpu",
) -> MicroScoreBatch:
    """Evaluate frozen scalar R1 semantics in deterministic query chunks."""
    result, _ = _r1_with_audit(query, candidates, chunk_size, device)
    return result


def _r1_with_audit(
    query: FloatArray,
    candidates: LocalCandidateSet,
    chunk_size: int,
    device: str,
) -> Tuple[MicroScoreBatch, BoolArray]:
    if chunk_size < 1:
        raise ScoringInputError("r1 chunk_size must be positive")
    query_matrix, values, valid = _candidate_arrays(query, candidates)
    scores: FloatArray = np.zeros(len(query_matrix), dtype=np.float32)
    output_valid: BoolArray = np.zeros(len(query_matrix), dtype=np.bool_)
    support_valid: BoolArray = np.zeros(values.shape[:2], dtype=np.bool_)
    for start in range(0, len(query_matrix), chunk_size):
        stop = min(start + chunk_size, len(query_matrix))
        chunk_inputs = (
            query_matrix[start:stop],
            values[start:stop],
            valid[start:stop],
        )
        chunk_scores, chunk_valid, chunk_support = (
            _r1_chunk(*chunk_inputs) if device == "cpu" else _r1_chunk_torch(*chunk_inputs, device)
        )
        scores[start:stop] = chunk_scores
        output_valid[start:stop] = chunk_valid
        support_valid[start:stop] = chunk_support
    return MicroScoreBatch(scores=scores, valid=output_valid), support_valid


def _r1_chunk(
    query: FloatArray,
    values: FloatArray,
    valid: BoolArray,
) -> Tuple[FloatArray, BoolArray, BoolArray]:
    prototypes, support_valid = _component_median_prototypes(values, valid)
    centers, converged = _geometric_medians(prototypes, support_valid)
    query_norms: FloatArray = np.asarray(np.linalg.norm(query, axis=1), dtype=np.float32)
    center_norms: FloatArray = np.asarray(np.linalg.norm(centers, axis=1), dtype=np.float32)
    output_valid: BoolArray = np.asarray(
        converged & (query_norms >= _NORM_EPS) & (center_norms >= _NORM_EPS),
        dtype=np.bool_,
    )
    safe_query: FloatArray = np.asarray(
        query / np.maximum(query_norms[:, None], _NORM_EPS),
        dtype=np.float32,
    )
    safe_center: FloatArray = np.asarray(
        centers / np.maximum(center_norms[:, None], _NORM_EPS),
        dtype=np.float32,
    )
    cosine: FloatArray = np.asarray(np.sum(safe_query * safe_center, axis=1), dtype=np.float32)
    scores: FloatArray = np.zeros(len(query), dtype=np.float32)
    scores[output_valid] = 1.0 - np.clip(cosine[output_valid], -1.0, 1.0)
    return scores, output_valid, support_valid


def _r1_chunk_torch(
    query: FloatArray,
    values: FloatArray,
    valid: BoolArray,
    device: str,
) -> Tuple[FloatArray, BoolArray, BoolArray]:
    active = torch.device(device)
    query_tensor = torch.as_tensor(query, dtype=torch.float32, device=active)
    value_tensor = torch.as_tensor(values, dtype=torch.float32, device=active)
    valid_tensor = torch.as_tensor(valid, dtype=torch.bool, device=active)
    usable = valid_tensor & torch.all(torch.isfinite(value_tensor), dim=3)
    counts = torch.sum(usable, dim=2)
    ordered = torch.sort(
        torch.where(
            usable.unsqueeze(-1),
            value_tensor,
            torch.full_like(value_tensor, torch.inf),
        ),
        dim=2,
    ).values
    dimension = value_tensor.shape[3]
    lower_indices = torch.clamp((counts - 1) // 2, min=0)
    upper_indices = counts // 2
    lower = torch.gather(
        ordered,
        2,
        lower_indices[..., None, None].expand(-1, -1, 1, dimension),
    ).squeeze(2)
    upper = torch.gather(
        ordered,
        2,
        upper_indices[..., None, None].expand(-1, -1, 1, dimension),
    ).squeeze(2)
    has_values = counts > 0
    medians = torch.where(
        has_values.unsqueeze(-1),
        (lower + upper) * 0.5,
        torch.zeros_like(lower),
    )
    norms = torch.linalg.vector_norm(medians, dim=2)
    support_valid = has_values & (norms >= _NORM_EPS)
    prototypes = medians / torch.clamp(norms.unsqueeze(-1), min=_NORM_EPS)
    prototypes = torch.where(support_valid.unsqueeze(-1), prototypes, 0.0)
    centers, converged = _geometric_medians_torch(prototypes, support_valid)
    query_norms = torch.linalg.vector_norm(query_tensor, dim=1)
    center_norms = torch.linalg.vector_norm(centers, dim=1)
    output_valid = converged & (query_norms >= _NORM_EPS) & (center_norms >= _NORM_EPS)
    safe_query = query_tensor / torch.clamp(query_norms.unsqueeze(-1), min=_NORM_EPS)
    safe_center = centers / torch.clamp(center_norms.unsqueeze(-1), min=_NORM_EPS)
    cosine = torch.sum(safe_query * safe_center, dim=1)
    scores = torch.zeros(len(query_tensor), dtype=torch.float32, device=active)
    scores[output_valid] = 1.0 - torch.clamp(cosine[output_valid], -1.0, 1.0)
    return (
        np.asarray(scores.detach().cpu().numpy(), dtype=np.float32),
        np.asarray(output_valid.detach().cpu().numpy(), dtype=np.bool_),
        np.asarray(support_valid.detach().cpu().numpy(), dtype=np.bool_),
    )


def _geometric_medians_torch(
    points: torch.Tensor,
    valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    counts = torch.sum(valid, dim=1)
    current = torch.sum(points, dim=1) / torch.clamp(counts.unsqueeze(-1), min=1)
    active = counts >= 3
    converged = torch.zeros(len(points), dtype=torch.bool, device=points.device)
    for _ in range(50):
        distances = torch.linalg.vector_norm(points - current.unsqueeze(1), dim=2)
        weights = torch.where(
            valid,
            1.0 / torch.clamp(distances, min=_NORM_EPS),
            0.0,
        )
        weight_sums = torch.sum(weights, dim=1)
        updated = torch.sum(points * weights.unsqueeze(-1), dim=1) / torch.clamp(
            weight_sums.unsqueeze(-1),
            min=_NORM_EPS,
        )
        deltas = torch.linalg.vector_norm(updated - current, dim=1)
        current = torch.where(active.unsqueeze(-1), updated, current)
        newly_converged = active & (deltas <= 1e-5)
        converged |= newly_converged
        active &= ~newly_converged
        if not bool(torch.any(active).item()):
            break
    return current, converged


def _component_median_prototypes(
    values: FloatArray,
    valid: BoolArray,
) -> Tuple[FloatArray, BoolArray]:
    usable: BoolArray = np.asarray(valid & np.all(np.isfinite(values), axis=3), dtype=np.bool_)
    counts: npt.NDArray[np.int64] = np.asarray(np.sum(usable, axis=2), dtype=np.int64)
    ordered = np.sort(np.where(usable[..., None], values, np.float32(np.inf)), axis=2)
    dimension: int = np.size(values, axis=3)
    lower_indices: npt.NDArray[np.int64] = np.maximum((counts - 1) // 2, 0)
    upper_indices: npt.NDArray[np.int64] = counts // 2
    lower: FloatArray = np.take_along_axis(
        ordered,
        np.broadcast_to(lower_indices[..., None, None], (*counts.shape, 1, dimension)),
        axis=2,
    )[:, :, 0, :]
    upper: FloatArray = np.take_along_axis(
        ordered,
        np.broadcast_to(upper_indices[..., None, None], (*counts.shape, 1, dimension)),
        axis=2,
    )[:, :, 0, :]
    has_values: BoolArray = np.asarray(counts > 0, dtype=np.bool_)
    lower = np.where(has_values[..., None], lower, np.float32(0.0))
    upper = np.where(has_values[..., None], upper, np.float32(0.0))
    medians: FloatArray = np.asarray((lower + upper) * np.float32(0.5), dtype=np.float32)
    norms: FloatArray = np.asarray(np.linalg.norm(medians, axis=2), dtype=np.float32)
    support_valid: BoolArray = np.asarray(has_values & (norms >= _NORM_EPS), dtype=np.bool_)
    prototypes: FloatArray = np.asarray(
        medians / np.maximum(norms[..., None], _NORM_EPS),
        dtype=np.float32,
    )
    prototypes[~support_valid] = 0.0
    return prototypes, support_valid


def _geometric_medians(points: FloatArray, valid: BoolArray) -> Tuple[FloatArray, BoolArray]:
    counts: npt.NDArray[np.int64] = np.asarray(np.sum(valid, axis=1), dtype=np.int64)
    current: FloatArray = np.asarray(
        np.sum(points, axis=1) / np.maximum(counts[:, None], 1),
        dtype=np.float32,
    )
    active: BoolArray = np.asarray(counts >= 3, dtype=np.bool_)
    converged: BoolArray = np.zeros(len(points), dtype=np.bool_)
    for _ in range(50):
        distances: FloatArray = np.asarray(
            np.linalg.norm(points - current[:, None, :], axis=2),
            dtype=np.float32,
        )
        weights: FloatArray = np.asarray(
            np.where(valid, 1.0 / np.maximum(distances, _NORM_EPS), 0.0),
            dtype=np.float32,
        )
        weight_sums: FloatArray = np.asarray(np.sum(weights, axis=1), dtype=np.float32)
        denominators: FloatArray = np.maximum(weight_sums[:, None], _NORM_EPS)
        updated: FloatArray = np.asarray(
            np.sum(points * weights[..., None], axis=1) / denominators,
            dtype=np.float32,
        )
        deltas: FloatArray = np.asarray(np.linalg.norm(updated - current, axis=1), dtype=np.float32)
        current = np.asarray(np.where(active[:, None], updated, current), dtype=np.float32)
        newly_converged: BoolArray = np.asarray(
            active & (deltas <= 1e-5),
            dtype=np.bool_,
        )
        converged |= newly_converged
        active &= ~newly_converged
        if not np.any(active):
            break
    return current, converged


def _candidate_arrays(
    query: FloatArray,
    candidates: LocalCandidateSet,
) -> Tuple[FloatArray, FloatArray, BoolArray]:
    query_matrix: FloatArray = np.asarray(query, dtype=np.float32)
    values: FloatArray = np.asarray(candidates.values, dtype=np.float32)
    valid: BoolArray = np.asarray(candidates.valid, dtype=np.bool_)
    query_shape_valid = query_matrix.ndim == 2 and min(query_matrix.shape) > 0
    candidate_shape_valid = values.ndim == 4 and min(values.shape) > 0
    if not query_shape_valid or not candidate_shape_valid or valid.shape != values.shape[:3]:
        raise ScoringInputError("query and candidates require non-empty (N,D)/(N,S,C,D) arrays")
    if values.shape[0] != query_matrix.shape[0] or values.shape[3] != query_matrix.shape[1]:
        raise ScoringInputError("candidate query count and feature dimension must match query")
    return query_matrix, values, valid


def _local_support_validity(values: FloatArray, valid: BoolArray) -> BoolArray:
    norms: FloatArray = np.asarray(np.linalg.norm(values, axis=3), dtype=np.float32)
    usable: BoolArray = np.asarray(
        valid & np.all(np.isfinite(values), axis=3) & (norms >= _NORM_EPS),
        dtype=np.bool_,
    )
    return np.asarray(np.any(usable, axis=2), dtype=np.bool_)


def _local_rungs_torch(  # noqa: PLR0913 -- explicit paired-rung population
    query: FloatArray,
    l0_values: FloatArray,
    l0_valid: BoolArray,
    l1_values: FloatArray,
    l1_valid: BoolArray,
    device: str,
) -> _TorchLocalRungs:
    active = torch.device(device)
    query_tensor = torch.as_tensor(query, dtype=torch.float32, device=active)
    l0_tensor = torch.as_tensor(l0_values, dtype=torch.float32, device=active)
    l1_tensor = torch.as_tensor(l1_values, dtype=torch.float32, device=active)
    l0_valid_tensor = torch.as_tensor(l0_valid, dtype=torch.bool, device=active)
    l1_valid_tensor = torch.as_tensor(l1_valid, dtype=torch.bool, device=active)
    l0_norms, l0_usable, l0_support = _torch_candidate_state(l0_tensor, l0_valid_tensor)
    l1_norms, l1_usable, l1_support = _torch_candidate_state(l1_tensor, l1_valid_tensor)
    shared_support = l0_support & l1_support
    l0_batch = _torch_local_min(
        query_tensor,
        l0_tensor,
        l0_norms,
        l0_usable & shared_support.unsqueeze(2),
    )
    l1_batch = _torch_local_min(
        query_tensor,
        l1_tensor,
        l1_norms,
        l1_usable & shared_support.unsqueeze(2),
    )
    return _TorchLocalRungs(
        l0=l0_batch,
        l1=l1_batch,
        l0_support=np.asarray(l0_support.detach().cpu().numpy(), dtype=np.bool_),
        l1_support=np.asarray(l1_support.detach().cpu().numpy(), dtype=np.bool_),
    )


def _torch_candidate_state(
    values: torch.Tensor,
    valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    norms = torch.linalg.vector_norm(values, dim=3)
    usable = valid & torch.all(torch.isfinite(values), dim=3) & (norms >= _NORM_EPS)
    return norms, usable, torch.any(usable, dim=2)


def _torch_local_min(
    query: torch.Tensor,
    values: torch.Tensor,
    candidate_norms: torch.Tensor,
    usable: torch.Tensor,
) -> MicroScoreBatch:
    query_norms = torch.linalg.vector_norm(query, dim=1)
    support_valid = torch.any(usable, dim=2)
    output_valid = (torch.sum(support_valid, dim=1) >= 3) & (query_norms >= _NORM_EPS)
    safe_queries = query / torch.clamp(query_norms.unsqueeze(1), min=_NORM_EPS)
    safe_candidates = values / torch.clamp(candidate_norms.unsqueeze(3), min=_NORM_EPS)
    similarities = torch.sum(safe_candidates * safe_queries[:, None, None, :], dim=3)
    similarities = torch.where(usable, similarities, -torch.inf)
    best = torch.max(similarities.reshape(len(query), -1), dim=1).values
    scores = torch.zeros(len(query), dtype=torch.float32, device=query.device)
    scores[output_valid] = 1.0 - torch.clamp(best[output_valid], -1.0, 1.0)
    return MicroScoreBatch(
        scores=np.asarray(scores.detach().cpu().numpy(), dtype=np.float32),
        valid=np.asarray(output_valid.detach().cpu().numpy(), dtype=np.bool_),
    )
