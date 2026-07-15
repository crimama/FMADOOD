from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pytest

import flow_tte.darc_gate2_pipeline as pipeline_module
from flow_tte.darc_feature_stream import ImageFeatures
from flow_tte.darc_gate2_correspondence import CorrespondenceConfig
from flow_tte.darc_gate2_pipeline import (
    QueryLadderInput,
    QueryPipelineConfig,
    calibrate_query_ladder,
    score_query_ladder,
)
from flow_tte.darc_gate2_scoring import RungNormalReferences
from flow_tte.darc_geometry import ImageSize
from flow_tte.darc_knn import ChunkedKnnConfig
from flow_tte.darc_tiling import NativeCrop

_NATIVE = ImageSize(height=512, width=896)
_CROPS = (
    NativeCrop(y0=0, x0=0, height=512, width=512),
    NativeCrop(y0=0, x0=384, height=512, width=512),
)


def _coarse_identity() -> np.ndarray:
    return np.eye(28, dtype=np.float16).reshape(4, 7, 28)


def _high_grid(offset: float) -> np.ndarray:
    rows, columns = np.meshgrid(
        np.arange(8, dtype=np.float32),
        np.arange(8, dtype=np.float32),
        indexing="ij",
    )
    return np.stack(
        (rows + 1.0, columns + 1.0, rows + columns + 1.0, np.full_like(rows, offset)),
        axis=2,
    ).astype(np.float16)


def _image() -> ImageFeatures:
    high = (_high_grid(1.0), _high_grid(2.0))
    return ImageFeatures(
        native_size=_NATIVE,
        crops=_CROPS,
        coarse=_coarse_identity(),
        low=high,
        high=high,
    )


def _inputs() -> QueryLadderInput:
    candidate_ids = ("zeta", "beta", "epsilon", "alpha", "delta", "gamma")
    candidates: Dict[str, ImageFeatures] = {support_id: _image() for support_id in candidate_ids}
    return QueryLadderInput(
        query_id="query-clean",
        query=_image(),
        candidates=candidates,
        knn_config=ChunkedKnnConfig(
            device="cpu",
            query_chunk_size=17,
            memory_chunk_size=64,
            top_k=2,
        ),
    )


def _config(complete_g0: bool) -> QueryPipelineConfig:
    return QueryPipelineConfig(
        correspondence=CorrespondenceConfig(short_edge=64),
        complete_g0=complete_g0,
    )


def test_pipeline_ranks_exact_top_five_and_preserves_registration_order() -> None:
    # Given: six equal coarse candidates in deliberately non-sorted mapping order.
    inputs = _inputs()

    # When: the query pipeline performs frozen coarse ranking and registration.
    result = score_query_ladder(inputs, _config(complete_g0=True))

    # Then: ID tie-breaking selects exactly five and every audit retains that order.
    expected = ("alpha", "beta", "delta", "epsilon", "gamma")
    assert result.selected_support_ids == expected
    assert tuple(item.support_id for item in result.registration_audit) == expected
    assert all(item.accepted and item.pair_count == 28 for item in result.registration_audit)
    assert tuple(item.crop_index for item in result.crops) == (0, 1)
    assert result.crop_shapes == (ImageSize(8, 8), ImageSize(8, 8))


def test_pipeline_identity_rungs_keep_every_crop_token_and_stable_audits() -> None:
    # Given: query and selected supports with identical coarse/high features.
    inputs = _inputs()

    # When: the complete-reference ladder is evaluated twice.
    first = score_query_ladder(inputs, _config(complete_g0=True))
    second = score_query_ladder(inputs, _config(complete_g0=True))

    # Then: L0/L1 agree, overlap tokens remain, and all audit digests are deterministic.
    assert np.array_equal(first.concatenate_l0_residuals(), first.concatenate_l1_residuals())
    assert len(first.concatenate_l0_residuals()) == 2 * 8 * 8
    assert len(first.concatenate_fallback_mask()) == 2 * 8 * 8
    assert first.audit == second.audit
    assert all(len(value) == 64 for value in first.audit)


def test_pipeline_complete_and_sparse_g0_modes_preserve_rung_outputs() -> None:
    # Given: the same query/support population in reference and query modes.
    inputs = _inputs()

    # When: complete G0 and fallback-only G0 executions are compared.
    complete = score_query_ladder(inputs, _config(complete_g0=True))
    sparse = score_query_ladder(inputs, _config(complete_g0=False))

    # Then: every scientific rung/fallback result and audit digest is identical.
    for complete_crop, sparse_crop in zip(complete.crops, sparse.crops):
        assert np.array_equal(complete_crop.scores.l0, sparse_crop.scores.l0)
        assert np.array_equal(complete_crop.scores.l1, sparse_crop.scores.l1)
        assert np.array_equal(complete_crop.scores.r1, sparse_crop.scores.r1)
        assert np.array_equal(
            complete_crop.scores.common_fallback,
            sparse_crop.scores.common_fallback,
        )
        assert np.all(complete_crop.scores.g0_valid)
        assert np.all(sparse_crop.scores.g0_valid)
    assert complete.audit == sparse.audit


def test_pipeline_operational_chunk_sizes_preserve_exact_outputs() -> None:
    # Given: identical populations evaluated with small and production-sized chunks.
    inputs = _inputs()
    large_chunks = inputs._replace(
        knn_config=ChunkedKnnConfig(
            device="cpu",
            query_chunk_size=1024,
            memory_chunk_size=262144,
            top_k=5,
        ),
    )

    # When: both executions use the same frozen scientific ladder.
    small = score_query_ladder(inputs, _config(complete_g0=True))
    large = score_query_ladder(large_chunks, _config(complete_g0=True))

    # Then: chunking changes only scheduling, never selections, rungs, or audits.
    assert small.selected_support_ids == large.selected_support_ids
    assert small.audit == large.audit
    for small_crop, large_crop in zip(small.crops, large.crops):
        assert np.array_equal(small_crop.scores.g0, large_crop.scores.g0)
        assert np.array_equal(small_crop.scores.l0, large_crop.scores.l0)
        assert np.array_equal(small_crop.scores.l1, large_crop.scores.l1)
        assert np.array_equal(small_crop.scores.r1, large_crop.scores.r1)
        assert np.array_equal(
            small_crop.scores.common_fallback,
            large_crop.scores.common_fallback,
        )


def test_pipeline_batches_all_crop_tokens_into_one_gpu_capable_g0_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: two query crops and five selected supports with two memory crops each.
    inputs = _inputs()
    calls: list[Tuple[int, int]] = []
    original = pipeline_module.cosine_1nn_scores

    def observed(
        query: np.ndarray,
        memory: np.ndarray,
        config: ChunkedKnnConfig,
    ) -> np.ndarray:
        calls.append((len(query), len(memory)))
        return original(query, memory, config)

    monkeypatch.setattr(pipeline_module, "cosine_1nn_scores", observed)

    # When: the full per-query ladder is scored.
    score_query_ladder(inputs, _config(complete_g0=True))

    # Then: G0 normalizes the selected memory once for the full ordered image population.
    assert calls == [(2 * 8 * 8, 5 * 2 * 8 * 8)]


def test_pipeline_calibration_stitches_finite_native_micro_evidence_maps() -> None:
    # Given: a complete query ladder and distinct frozen normal-only rung tails.
    result = score_query_ladder(_inputs(), _config(complete_g0=True))
    references = RungNormalReferences(
        g0=np.asarray([0.0, 0.5, 1.0, 2.0], dtype=np.float32),
        l0=np.asarray([0.0, 0.25, 0.5], dtype=np.float32),
        l1=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        r1=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
    )

    # When: crop token evidence is reshaped and stitched into native coordinates.
    evidence = calibrate_query_ladder(result, references)

    # Then: every rung covers the full native image with finite evidence.
    maps: Tuple[np.ndarray, ...] = (evidence.l0, evidence.l1, evidence.r1)
    assert all(values.shape == (_NATIVE.height, _NATIVE.width) for values in maps)
    assert all(np.all(np.isfinite(values)) for values in maps)
