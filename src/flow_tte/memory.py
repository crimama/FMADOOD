from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from typing_extensions import final

from flow_tte.context_query import (
    ContextAdjustment,
    ContextQueryState,
    apply_context_adjustment,
    group_contexts,
    normalize_contexts,
)


@dataclass(frozen=True)
class QueryResult:
    distances: torch.Tensor
    indices: torch.Tensor


@final
class TorchMemoryBank:
    def __init__(self) -> None:
        self.features: torch.Tensor = torch.empty(0, 0)
        self.contexts: Optional[torch.Tensor] = None
        self.context_group_contexts: Optional[torch.Tensor] = None
        self.context_group_ids: Optional[torch.Tensor] = None

    def fit(self, features: torch.Tensor, contexts: Optional[torch.Tensor] = None) -> None:
        if features.ndim != 2:
            raise RuntimeError("TorchMemoryBank expects 2D features")
        if contexts is not None:
            _validate_contexts(contexts, features.shape[0], "contexts")
        self.features = features.detach().clone()
        self.contexts = None if contexts is None else contexts.detach().clone()
        self.context_group_contexts = None
        self.context_group_ids = None

    def query(  # noqa: PLR0913
        self,
        features: torch.Tensor,
        k: int = 1,
        chunk_size: int = 8192,
        squared: bool = False,
        query_contexts: Optional[torch.Tensor] = None,
        context_weight: float = 0.0,
        context_top_m: Optional[int] = None,
    ) -> QueryResult:
        if self.features.numel() == 0:
            raise RuntimeError("TorchMemoryBank is empty")
        if features.ndim != 2:
            raise RuntimeError("TorchMemoryBank query expects 2D features")
        if chunk_size <= 0:
            raise RuntimeError("TorchMemoryBank query chunk_size must be positive")
        if context_weight < 0.0:
            raise RuntimeError("TorchMemoryBank query context_weight must be non-negative")
        if context_top_m is not None and context_top_m <= 0:
            raise RuntimeError("TorchMemoryBank query context_top_m must be positive")
        context_state = self._prepare_context_state(
            query_contexts=query_contexts,
            query_rows=int(features.shape[0]),
            context_weight=context_weight,
            context_top_m=context_top_m,
        )
        k_safe = min(k, self.features.shape[0])
        values: list[torch.Tensor] = []
        indices: list[torch.Tensor] = []
        for start in range(0, features.shape[0], chunk_size):
            query_chunk = features[start : start + chunk_size]
            distances = torch.cdist(query_chunk, self.features, p=2.0)
            if squared:
                distances = distances.square()
            distances = apply_context_adjustment(
                distances=distances,
                adjustment=ContextAdjustment(
                    query_contexts=query_contexts,
                    start=start,
                    chunk_size=chunk_size,
                    context_state=context_state,
                    context_weight=context_weight,
                    context_top_m=context_top_m,
                ),
            )
            top = torch.topk(distances, k=k_safe, largest=False, dim=1)
            values.append(top.values)
            indices.append(top.indices)
        return QueryResult(distances=torch.cat(values, dim=0), indices=torch.cat(indices, dim=0))

    def size(self) -> int:
        return int(self.features.shape[0])

    def _prepare_context_state(
        self,
        query_contexts: Optional[torch.Tensor],
        query_rows: int,
        context_weight: float,
        context_top_m: Optional[int],
    ) -> Optional[ContextQueryState]:
        if context_weight == 0.0 and context_top_m is None:
            return None
        if self.contexts is None:
            raise RuntimeError("TorchMemoryBank query requires fitted contexts")
        if query_contexts is None:
            raise RuntimeError("TorchMemoryBank query requires query contexts")
        _validate_contexts(query_contexts, query_rows, "query_contexts")
        if query_contexts.shape[1] != self.contexts.shape[1]:
            raise RuntimeError("TorchMemoryBank query context dimensions must match")
        group_contexts = self.context_group_contexts
        group_ids = self.context_group_ids
        if context_top_m is not None and self.contexts is not None and group_contexts is None:
            group_contexts, group_ids = self._ensure_context_groups()
        if context_top_m is not None and (group_contexts is None or group_ids is None):
            raise RuntimeError("TorchMemoryBank context groups are not fitted")
        norm_group_contexts = None
        if group_contexts is not None:
            norm_group_contexts = normalize_contexts(group_contexts)
        return ContextQueryState(
            bank_contexts=normalize_contexts(self.contexts),
            norm_group_contexts=norm_group_contexts,
            group_ids=group_ids,
        )

    def _ensure_context_groups(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.contexts is None:
            raise RuntimeError("TorchMemoryBank contexts are not fitted")
        if self.context_group_contexts is None or self.context_group_ids is None:
            self.context_group_contexts, self.context_group_ids = group_contexts(self.contexts)
        return self.context_group_contexts, self.context_group_ids


@final
class ReservoirMemory:
    def __init__(
        self,
        m0_features: torch.Tensor,
        budget: float,
        random_seed: int,
        m0_contexts: Optional[torch.Tensor] = None,
    ) -> None:
        if m0_contexts is not None:
            _validate_contexts(m0_contexts, m0_features.shape[0], "m0_contexts")
        self.m0_features: torch.Tensor = m0_features.detach().clone()
        self.m0_contexts: Optional[torch.Tensor] = (
            None if m0_contexts is None else m0_contexts.detach().clone()
        )
        self.capacity: int = int(max(0, round((budget - 1.0) * self.m0_features.shape[0])))
        self.buffer: torch.Tensor = torch.empty(
            0,
            self.m0_features.shape[1],
            device=self.m0_features.device,
            dtype=self.m0_features.dtype,
        )
        self.buffer_contexts: Optional[torch.Tensor] = None
        if self.m0_contexts is not None:
            self.buffer_contexts = torch.empty(
                0,
                self.m0_contexts.shape[1],
                device=self.m0_contexts.device,
                dtype=self.m0_contexts.dtype,
            )
        self.generator: torch.Generator = torch.Generator(device=self.m0_features.device)
        _ = self.generator.manual_seed(random_seed)
        self.n_seen: int = 0
        self.bank: TorchMemoryBank = TorchMemoryBank()
        self._rebuild()

    def absorb(  # noqa: C901
        self,
        candidates: torch.Tensor,
        candidate_contexts: Optional[torch.Tensor] = None,
    ) -> None:
        if self.capacity == 0 or candidates.shape[0] == 0:
            return
        candidates = candidates.detach()
        if self.m0_contexts is not None:
            if candidate_contexts is None:
                raise RuntimeError("ReservoirMemory absorb requires candidate contexts")
            _validate_contexts(candidate_contexts, candidates.shape[0], "candidate_contexts")
            candidate_contexts = candidate_contexts.detach()
        elif candidate_contexts is not None:
            raise RuntimeError("ReservoirMemory was fitted without contexts")
        open_slots = self.capacity - self.buffer.shape[0]
        if open_slots > 0:
            fill = min(open_slots, candidates.shape[0])
            self.buffer = torch.cat([self.buffer, candidates[:fill]], dim=0)
            if self.buffer_contexts is not None and candidate_contexts is not None:
                self.buffer_contexts = torch.cat(
                    [self.buffer_contexts, candidate_contexts[:fill]],
                    dim=0,
                )
            self.n_seen += fill
            candidates = candidates[fill:]
            if candidate_contexts is not None:
                candidate_contexts = candidate_contexts[fill:]

        if candidates.shape[0] > 0:
            stream_ids = self.n_seen + torch.arange(
                1,
                candidates.shape[0] + 1,
                device=candidates.device,
            )
            draws = torch.rand(
                candidates.shape[0],
                generator=self.generator,
                device=candidates.device,
            )
            replace_at = torch.floor(draws * stream_ids).to(torch.long)
            accepted = replace_at < self.capacity
            if bool(accepted.any()):
                self.buffer[replace_at[accepted]] = candidates[accepted]
                if self.buffer_contexts is not None and candidate_contexts is not None:
                    self.buffer_contexts[replace_at[accepted]] = candidate_contexts[accepted]
            self.n_seen += int(candidates.shape[0])
        self._rebuild()

    def _rebuild(self) -> None:
        if self.buffer.shape[0] == 0:
            self.bank.fit(self.m0_features, contexts=self.m0_contexts)
            return
        contexts = None
        if self.m0_contexts is not None and self.buffer_contexts is not None:
            contexts = torch.cat([self.m0_contexts, self.buffer_contexts], dim=0)
        self.bank.fit(torch.cat([self.m0_features, self.buffer], dim=0), contexts=contexts)


def _validate_contexts(contexts: torch.Tensor, n_rows: int, name: str) -> None:
    if contexts.ndim != 2:
        message = f"{name} must be a 2D tensor"
        raise RuntimeError(message)
    if contexts.shape[0] != n_rows:
        message = f"{name} row count must match features"
        raise RuntimeError(message)
