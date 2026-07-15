from __future__ import annotations

import numpy as np
import pytest

from flow_tte.darc_knn import (
    ChunkedKnnConfig,
    MemoryChunkObserver,
    SupportTokens,
    cosine_1nn_scores,
    mutual_nn_similarity,
    rank_supports,
)


class _TransferRecorder(MemoryChunkObserver):
    def __init__(self) -> None:
        self.spans: list[tuple[int, int]] = []

    def memory_chunk_transferred(self, start: int, stop: int) -> None:
        self.spans.append((start, stop))


def test_cosine_1nn_scans_both_dimensions_when_chunks_are_one() -> None:
    # Given
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    memory = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32)
    config = ChunkedKnnConfig(device="cpu", query_chunk_size=1, memory_chunk_size=1)

    # When
    scores = cosine_1nn_scores(query, memory, config)

    # Then
    np.testing.assert_allclose(scores, np.asarray([0.0, 1.0], dtype=np.float32))


def test_rank_supports_is_deterministic_for_equal_similarity() -> None:
    # Given
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    supports = (
        SupportTokens("z", np.asarray([[1.0, 0.0]], dtype=np.float32)),
        SupportTokens("a", np.asarray([[1.0, 0.0]], dtype=np.float32)),
        SupportTokens("far", np.asarray([[0.0, 1.0]], dtype=np.float32)),
    )
    config = ChunkedKnnConfig(
        device="cpu",
        query_chunk_size=1,
        memory_chunk_size=1,
        top_k=2,
    )

    # When
    selected = rank_supports(query, supports, config)

    # Then
    assert selected == ("a", "z")


def test_cosine_1nn_transfers_each_memory_chunk_once_per_call() -> None:
    # Given
    recorder = _TransferRecorder()
    query = np.eye(5, dtype=np.float32)
    memory = np.concatenate((np.eye(5, dtype=np.float32), np.eye(2, 5, dtype=np.float32)))
    config = ChunkedKnnConfig(
        device="cpu",
        query_chunk_size=2,
        memory_chunk_size=3,
        transfer_observer=recorder,
    )

    # When
    cosine_1nn_scores(query, memory, config)

    # Then
    assert recorder.spans == [(0, 3), (3, 6), (6, 7)]


def test_cosine_1nn_matches_full_matrix_across_uneven_chunks() -> None:
    # Given
    generator = np.random.default_rng(19)
    query = generator.normal(size=(7, 4)).astype(np.float32)
    memory = generator.normal(size=(11, 4)).astype(np.float32)
    config = ChunkedKnnConfig(device="cpu", query_chunk_size=3, memory_chunk_size=4)
    normalized_query = query / np.linalg.norm(query, axis=1, keepdims=True)
    normalized_memory = memory / np.linalg.norm(memory, axis=1, keepdims=True)
    expected = 1.0 - np.max(normalized_query @ normalized_memory.T, axis=1)

    # When
    scores = cosine_1nn_scores(query, memory, config)

    # Then
    np.testing.assert_allclose(scores, expected, rtol=1e-6, atol=1e-6)


def test_mutual_nn_similarity_discards_nonreciprocal_neighbours() -> None:
    # Given
    query = _angle_tokens((0, 90, 180, 270))
    support = _angle_tokens((0, 15, 30, 45, 60, 75))

    # When
    similarity = mutual_nn_similarity(support, query, ChunkedKnnConfig(device="cpu"))

    # Then
    expected = float(np.median((1.0, np.cos(np.deg2rad(15.0)))))
    assert similarity == pytest.approx(expected)


def test_rank_supports_uses_median_mutual_nn_not_directed_medians() -> None:
    # Given
    query = _angle_tokens((0, 90, 180, 270))
    supports = (
        SupportTokens("mnn-best", _angle_tokens((0, 15, 30, 45, 60, 75))),
        SupportTokens("directed-best", _angle_tokens((0, 15, 30, 45, 60, 150))),
    )
    config = ChunkedKnnConfig(device="cpu", top_k=1)

    # When
    selected = rank_supports(query, supports, config)

    # Then
    assert selected == ("mnn-best",)


def _angle_tokens(angles: tuple[int, ...]) -> np.ndarray:
    radians = np.deg2rad(np.asarray(angles, dtype=np.float32))
    return np.stack((np.cos(radians), np.sin(radians)), axis=1).astype(np.float32)
