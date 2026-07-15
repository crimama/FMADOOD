from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import torch
from typing_extensions import override

FloatArray = npt.NDArray[np.floating]
Float32Array = npt.NDArray[np.float32]
IndexArray = npt.NDArray[np.int64]


class MemoryChunkObserver(Protocol):
    def memory_chunk_transferred(self, start: int, stop: int) -> None: ...


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class DarcKnnError(ValueError):
    reason: str

    @override
    def __str__(self) -> str:
        return f"Invalid DARC k-NN input: {self.reason}"


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class ChunkedKnnConfig:
    device: str
    query_chunk_size: int = 256
    memory_chunk_size: int = 16384
    top_k: int = 5
    epsilon: float = 1e-12
    transfer_observer: Optional[MemoryChunkObserver] = None

    def __post_init__(self) -> None:
        positive = (
            self.query_chunk_size > 0
            and self.memory_chunk_size > 0
            and self.top_k > 0
            and self.epsilon > 0.0
        )
        if not positive:
            raise DarcKnnError("chunk sizes, top_k, and epsilon must be positive")


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class SupportTokens:
    support_id: str
    tokens: FloatArray


def cosine_1nn_scores(
    query: FloatArray,
    memory: FloatArray,
    config: Optional[ChunkedKnnConfig] = None,
) -> Float32Array:
    """Return exact cosine 1-NN distances without a full pairwise matrix."""
    active = config if config is not None else ChunkedKnnConfig(device="cpu")
    similarities, _ = _nearest_indices(query, memory, active)
    return np.asarray(1.0 - np.clip(similarities, -1.0, 1.0), dtype=np.float32)


def mutual_nn_similarity(
    support: FloatArray,
    query: FloatArray,
    config: Optional[ChunkedKnnConfig] = None,
) -> float:
    """Return the median cosine over reciprocal nearest-neighbor pairs."""
    active = config if config is not None else ChunkedKnnConfig(device="cpu")
    support_values, support_to_query = _nearest_indices(support, query, active)
    _, query_to_support = _nearest_indices(query, support, active)
    support_indices = np.arange(support_to_query.size, dtype=np.int64)
    mutual = query_to_support[support_to_query] == support_indices
    if not np.any(mutual):
        raise DarcKnnError("finite feature sets produced no mutual nearest neighbours")
    return float(np.median(support_values[mutual]))


def rank_supports(
    query: FloatArray,
    supports: Sequence[SupportTokens],
    config: Optional[ChunkedKnnConfig] = None,
) -> Tuple[str, ...]:
    """Rank supports by coarse similarity with support-ID tie breaking."""
    active = config if config is not None else ChunkedKnnConfig(device="cpu")
    if not supports:
        raise DarcKnnError("supports must be non-empty")
    unique_ids = {support.support_id for support in supports}
    if len(unique_ids) != len(supports):
        raise DarcKnnError("support IDs must be unique")
    ranked = sorted(
        (
            (mutual_nn_similarity(support.tokens, query, active), support.support_id)
            for support in supports
        ),
        key=lambda item: (-item[0], item[1]),
    )
    return tuple(support_id for _, support_id in ranked[: min(active.top_k, len(ranked))])


def _matrix(values: FloatArray, name: str) -> FloatArray:
    matrix = np.asarray(values)
    valid = matrix.ndim == 2 and matrix.shape[0] > 0 and matrix.shape[1] > 0
    if not valid or not np.all(np.isfinite(matrix)):
        reason = f"{name} must be a finite non-empty matrix"
        raise DarcKnnError(reason)
    return matrix


def _nearest_indices(
    query: FloatArray,
    memory: FloatArray,
    config: ChunkedKnnConfig,
) -> Tuple[Float32Array, IndexArray]:
    query_matrix = _matrix(query, "query")
    memory_matrix = _matrix(memory, "memory")
    if query_matrix.shape[1] != memory_matrix.shape[1]:
        raise DarcKnnError("query and memory dimensions differ")
    best = np.full(query_matrix.shape[0], -np.inf, dtype=np.float32)
    best_indices = np.full(query_matrix.shape[0], -1, dtype=np.int64)
    device = torch.device(config.device)
    for memory_start in range(0, memory_matrix.shape[0], config.memory_chunk_size):
        memory_stop = min(memory_start + config.memory_chunk_size, memory_matrix.shape[0])
        memory_tensor = _normalized_tensor(memory_matrix[memory_start:memory_stop], device, config)
        if config.transfer_observer is not None:
            config.transfer_observer.memory_chunk_transferred(memory_start, memory_stop)
        for query_start in range(0, query_matrix.shape[0], config.query_chunk_size):
            query_stop = min(query_start + config.query_chunk_size, query_matrix.shape[0])
            query_tensor = _normalized_tensor(query_matrix[query_start:query_stop], device, config)
            values, local_indices = torch.max(query_tensor @ memory_tensor.T, dim=1)
            candidate_values = values.cpu().numpy()
            candidate_indices = local_indices.cpu().numpy().astype(np.int64) + memory_start
            current_values = best[query_start:query_stop]
            current_indices = best_indices[query_start:query_stop]
            update = (candidate_values > current_values) | (
                (candidate_values == current_values) & (candidate_indices < current_indices)
            )
            current_values[update] = candidate_values[update]
            current_indices[update] = candidate_indices[update]
    return best, best_indices


def _normalized_tensor(
    values: FloatArray,
    device: torch.device,
    config: ChunkedKnnConfig,
) -> torch.Tensor:
    tensor = torch.tensor(np.asarray(values, dtype=np.float32), device=device)
    norms = torch.linalg.vector_norm(tensor, dim=1, keepdim=True)
    if bool(torch.any(norms < config.epsilon)):
        raise DarcKnnError("feature rows must be non-zero")
    return tensor.div_(norms)
