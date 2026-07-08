from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as torch_functional


@dataclass(frozen=True)
class ContextQueryState:
    bank_contexts: torch.Tensor
    norm_group_contexts: Optional[torch.Tensor]
    group_ids: Optional[torch.Tensor]


@dataclass(frozen=True)
class ContextAdjustment:
    query_contexts: Optional[torch.Tensor]
    start: int
    chunk_size: int
    context_state: Optional[ContextQueryState]
    context_weight: float
    context_top_m: Optional[int]


def normalize_contexts(contexts: torch.Tensor) -> torch.Tensor:
    return torch_functional.normalize(contexts, dim=1, eps=1e-12)


def group_contexts(contexts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    unique_contexts: list[torch.Tensor] = []
    group_ids = torch.empty(contexts.shape[0], dtype=torch.long, device=contexts.device)
    for row_idx in range(contexts.shape[0]):
        row = contexts[row_idx]
        group_id = _matching_context_group(row, unique_contexts)
        if group_id is None:
            group_id = len(unique_contexts)
            unique_contexts.append(row.detach().clone())
        group_ids[row_idx] = group_id
    return torch.stack(unique_contexts, dim=0), group_ids


def apply_context_adjustment(
    distances: torch.Tensor,
    adjustment: ContextAdjustment,
) -> torch.Tensor:
    if adjustment.context_state is None:
        return distances
    if adjustment.query_contexts is None:
        raise RuntimeError("TorchMemoryBank context state is inconsistent")
    query_context_chunk = normalize_contexts(
        adjustment.query_contexts[adjustment.start : adjustment.start + adjustment.chunk_size],
    )
    adjusted = distances
    if adjustment.context_weight > 0.0:
        context_distances = 1.0 - query_context_chunk @ adjustment.context_state.bank_contexts.T
        adjusted = adjusted + float(adjustment.context_weight) * context_distances.to(
            adjusted.dtype,
        )
    if adjustment.context_top_m is None or adjustment.context_state.norm_group_contexts is None:
        return adjusted
    return _mask_to_top_m_context_groups(
        distances=adjusted,
        query_contexts=query_context_chunk,
        norm_group_contexts=adjustment.context_state.norm_group_contexts,
        group_ids=adjustment.context_state.group_ids,
        context_top_m=adjustment.context_top_m,
    )


def _matching_context_group(
    row: torch.Tensor,
    unique_contexts: list[torch.Tensor],
) -> Optional[int]:
    for group_id, candidate in enumerate(unique_contexts):
        if bool(torch.equal(row, candidate)):
            return group_id
    return None


def _mask_to_top_m_context_groups(
    distances: torch.Tensor,
    query_contexts: torch.Tensor,
    norm_group_contexts: torch.Tensor,
    group_ids: Optional[torch.Tensor],
    context_top_m: int,
) -> torch.Tensor:
    if group_ids is None:
        raise RuntimeError("TorchMemoryBank context group state is inconsistent")
    top_m_safe = min(context_top_m, norm_group_contexts.shape[0])
    if top_m_safe >= norm_group_contexts.shape[0]:
        return distances
    group_distances = 1.0 - query_contexts @ norm_group_contexts.T
    selected_groups = torch.topk(group_distances, k=top_m_safe, largest=False, dim=1).indices
    allowed = (selected_groups.unsqueeze(-1) == group_ids.reshape(1, 1, group_ids.shape[0])).any(
        dim=1,
    )
    return distances.masked_fill(~allowed, torch.inf)
