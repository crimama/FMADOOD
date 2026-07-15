from __future__ import annotations

# pyright: reportMissingImports=false
import json
import logging
import os
from dataclasses import replace
from time import perf_counter
from typing import Dict, Final, List, NamedTuple, Tuple

import numpy as np

from flow_tte.darc_feature_stream import FeatureArray, ImageFeatures, stitch_score_grids
from flow_tte.darc_gate2_coordinate_maps import high_feature_grid
from flow_tte.darc_gate2_correspondence import (
    build_l0_candidate_chunk,
    build_l1_candidate_chunk,
    iter_query_token_chunks,
    prepare_correspondence,
)
from flow_tte.darc_gate2_correspondence_types import (
    CorrespondenceQuery,
    PreparedCorrespondence,
    SelectedSupport,
)
from flow_tte.darc_gate2_pipeline_audit import query_ladder_audit
from flow_tte.darc_gate2_pipeline_types import (
    BoolArray,
    CropLadderResult,
    FloatArray,
    Gate2PipelineError,
    QueryEvidenceMaps,
    QueryLadderInput,
    QueryLadderResult,
    QueryPipelineConfig,
)
from flow_tte.darc_gate2_scoring import calibrate_rung_evidence, score_rungs
from flow_tte.darc_gate2_scoring_types import (
    RungNormalReferences,
    RungScores,
    RungScoringConfig,
    RungScoringInput,
    SupportValidityAudit,
)
from flow_tte.darc_geometry import ImageSize
from flow_tte.darc_knn import ChunkedKnnConfig, SupportTokens, cosine_1nn_scores, rank_supports

_DEFAULT_PIPELINE_CONFIG: Final = QueryPipelineConfig()
_PROFILE_ENV: Final = "FMAD_DARC_GATE2_PROFILE"
_LOGGER = logging.getLogger(__name__)


class _ScoreContext(NamedTuple):
    global_memory: FloatArray
    knn_config: ChunkedKnnConfig
    complete_g0: bool


def score_query_ladder(
    inputs: QueryLadderInput,
    config: QueryPipelineConfig = _DEFAULT_PIPELINE_CONFIG,
) -> QueryLadderResult:
    """Run one query through the deterministic exact-five Gate 2 ladder."""
    query_started = perf_counter()
    stage_started = query_started
    selected_ids = _rank_exact_five(inputs)
    _profile_event(
        "rank",
        inputs.query_id,
        stage_started,
        {"candidate_count": len(inputs.candidates)},
    )
    stage_started = perf_counter()
    selected = tuple(
        SelectedSupport(support_id, inputs.candidates[support_id]) for support_id in selected_ids
    )
    prepared = prepare_correspondence(
        CorrespondenceQuery(inputs.query_id, inputs.query),
        selected,
        config.correspondence,
    )
    _profile_event("registration", inputs.query_id, stage_started, {"selected_count": 5})
    stage_started = perf_counter()
    query_crops = tuple(
        _high_matrix(inputs.query, index) for index in range(len(inputs.query.high))
    )
    global_memory = _global_memory(selected)
    all_query: FloatArray = np.asarray(
        np.concatenate(query_crops, axis=0),
        dtype=np.float32,
    )
    _profile_event(
        "g0_prepare",
        inputs.query_id,
        stage_started,
        {"query_tokens": len(all_query), "memory_tokens": len(global_memory)},
    )
    stage_started = perf_counter()
    full_g0 = cosine_1nn_scores(all_query, global_memory, inputs.knn_config)
    _profile_event("g0_exact", inputs.query_id, stage_started, {})
    stage_started = perf_counter()
    score_context = _ScoreContext(global_memory, inputs.knn_config, config.complete_g0)

    crop_results: List[CropLadderResult] = []
    g0_offset = 0
    for crop_index, query_crop in enumerate(query_crops):
        token_shape = _token_shape(inputs.query, crop_index)
        scores = _score_crop(
            prepared,
            crop_index,
            query_crop,
            np.asarray(full_g0[g0_offset : g0_offset + len(query_crop)], dtype=np.float32),
            score_context,
        )
        crop_results.append(
            CropLadderResult(crop_index, inputs.query.crops[crop_index], token_shape, scores),
        )
        g0_offset += len(query_crop)
    if g0_offset != len(full_g0):
        raise Gate2PipelineError("G0 population did not match the ordered crop population")
    _profile_event(
        "local_ladder",
        inputs.query_id,
        stage_started,
        {"crop_count": len(query_crops)},
    )
    _profile_event("query_total", inputs.query_id, query_started, {})
    crops = tuple(crop_results)
    return QueryLadderResult(
        query_id=inputs.query_id,
        native_size=inputs.query.native_size,
        selected_support_ids=selected_ids,
        crops=crops,
        registration_audit=prepared.registrations,
        audit=query_ladder_audit(inputs.query_id, selected_ids, crops),
    )


def calibrate_query_ladder(
    result: QueryLadderResult,
    references: RungNormalReferences,
) -> QueryEvidenceMaps:
    """Calibrate crop rungs and stitch them into native-resolution evidence maps."""
    l0_grids: List[FloatArray] = []
    l1_grids: List[FloatArray] = []
    r1_grids: List[FloatArray] = []
    for crop in result.crops:
        expected = crop.token_shape.height * crop.token_shape.width
        if len(crop.scores.l0) != expected:
            raise Gate2PipelineError("crop score population does not match its token grid")
        evidence = calibrate_rung_evidence(crop.scores, references)
        shape = (crop.token_shape.height, crop.token_shape.width)
        l0_grids.append(np.asarray(evidence.l0.reshape(shape), dtype=np.float32))
        l1_grids.append(np.asarray(evidence.l1.reshape(shape), dtype=np.float32))
        r1_grids.append(np.asarray(evidence.r1.reshape(shape), dtype=np.float32))
    native_crops = tuple(crop.crop for crop in result.crops)
    return QueryEvidenceMaps(
        l0=stitch_score_grids(result.native_size, native_crops, tuple(l0_grids)),
        l1=stitch_score_grids(result.native_size, native_crops, tuple(l1_grids)),
        r1=stitch_score_grids(result.native_size, native_crops, tuple(r1_grids)),
    )


def _rank_exact_five(inputs: QueryLadderInput) -> Tuple[str, ...]:
    if not inputs.query_id:
        raise Gate2PipelineError("query identity must be non-empty")
    if len(inputs.candidates) < 5:
        raise Gate2PipelineError("at least five candidate supports are required")
    ordered_ids = tuple(sorted(inputs.candidates))
    if any(not support_id for support_id in ordered_ids):
        raise Gate2PipelineError("candidate support identities must be non-empty")
    query_tokens = _flatten_grid(inputs.query.coarse, "query coarse")
    supports = tuple(
        SupportTokens(
            support_id=support_id,
            tokens=_flatten_grid(inputs.candidates[support_id].coarse, "support coarse"),
        )
        for support_id in ordered_ids
    )
    ranking_config = replace(inputs.knn_config, top_k=5)
    selected = rank_supports(query_tokens, supports, ranking_config)
    if len(selected) != 5:
        raise Gate2PipelineError("coarse ranking must select exactly five supports")
    return selected


def _score_crop(
    prepared: PreparedCorrespondence,
    crop_index: int,
    query: FloatArray,
    g0: FloatArray,
    context: _ScoreContext,
) -> RungScores:
    chunks: List[RungScores] = []
    expected_start = 0
    for token_chunk in iter_query_token_chunks(
        prepared,
        crop_index,
        chunk_size=context.knn_config.query_chunk_size,
    ):
        request = token_chunk.request
        if request.start != expected_start:
            raise Gate2PipelineError("query token chunks are not contiguous")
        chunk_details: Dict[str, object] = {
            "crop_index": crop_index,
            "start": request.start,
            "stop": request.stop,
        }
        stage_started = perf_counter()
        l0 = build_l0_candidate_chunk(prepared, request)
        _profile_event("candidate_l0", prepared.query_id, stage_started, chunk_details)
        stage_started = perf_counter()
        l1 = build_l1_candidate_chunk(prepared, request)
        _profile_event("candidate_l1", prepared.query_id, stage_started, chunk_details)
        selected_ids = tuple(item.support_id for item in prepared.supports)
        support_order_changed = (
            l0.audit.ordered_support_ids != selected_ids
            or l1.audit.ordered_support_ids != selected_ids
        )
        if support_order_changed:
            raise Gate2PipelineError("candidate support order changed during correspondence")
        stage_started = perf_counter()
        chunks.append(
            score_rungs(
                RungScoringInput(
                    query=np.asarray(query[request.start : request.stop], dtype=np.float32),
                    global_memory=context.global_memory,
                    identity_candidates=l0.local,
                    aligned_candidates=l1.local,
                    precomputed_g0=np.asarray(g0[request.start : request.stop], dtype=np.float32),
                ),
                RungScoringConfig(
                    g0_chunk_size=context.knn_config.query_chunk_size,
                    r1_chunk_size=context.knn_config.query_chunk_size,
                    complete_g0=context.complete_g0,
                    device=context.knn_config.device,
                ),
            ),
        )
        _profile_event("score_rungs", prepared.query_id, stage_started, chunk_details)
        expected_start = request.stop
    if expected_start != len(query):
        raise Gate2PipelineError("query token chunks did not cover the complete crop")
    return _concatenate_scores(tuple(chunks), len(query))


def _concatenate_scores(chunks: Tuple[RungScores, ...], expected: int) -> RungScores:
    if not chunks:
        raise Gate2PipelineError("a scored crop must contain at least one token chunk")

    result = RungScores(
        g0=_concat_float(tuple(chunk.g0 for chunk in chunks)),
        g0_valid=_concat_bool(tuple(chunk.g0_valid for chunk in chunks)),
        l0=_concat_float(tuple(chunk.l0 for chunk in chunks)),
        l1=_concat_float(tuple(chunk.l1 for chunk in chunks)),
        r1=_concat_float(tuple(chunk.r1 for chunk in chunks)),
        common_fallback=_concat_bool(tuple(chunk.common_fallback for chunk in chunks)),
        support_validity=SupportValidityAudit(
            l0=_concat_bool(tuple(chunk.support_validity.l0 for chunk in chunks)),
            l1=_concat_bool(tuple(chunk.support_validity.l1 for chunk in chunks)),
            shared=_concat_bool(tuple(chunk.support_validity.shared for chunk in chunks)),
            r1=_concat_bool(tuple(chunk.support_validity.r1 for chunk in chunks)),
        ),
    )
    if len(result.g0) != expected:
        raise Gate2PipelineError("concatenated score population does not match the crop")
    return result


def _concat_float(values: Tuple[FloatArray, ...]) -> FloatArray:
    return np.asarray(np.concatenate(values, axis=0), dtype=np.float32)


def _concat_bool(values: Tuple[BoolArray, ...]) -> BoolArray:
    return np.asarray(np.concatenate(values, axis=0), dtype=np.bool_)


def _high_matrix(image: ImageFeatures, crop_index: int) -> FloatArray:
    grid, _ = high_feature_grid(image, crop_index)
    return np.asarray(grid.reshape(-1, int(np.size(grid, axis=2))), dtype=np.float32)


def _token_shape(image: ImageFeatures, crop_index: int) -> ImageSize:
    _, token_grid = high_feature_grid(image, crop_index)
    return token_grid.shape


def _global_memory(selected: Tuple[SelectedSupport, ...]) -> FloatArray:
    matrices = tuple(
        _high_matrix(support.features, crop_index)
        for support in selected
        for crop_index in range(len(support.features.high))
    )
    if not matrices:
        raise Gate2PipelineError("selected supports have no high-resolution memory")
    return np.asarray(np.concatenate(matrices, axis=0), dtype=np.float32)


def _flatten_grid(values: FeatureArray, name: str) -> FloatArray:
    grid: FloatArray = np.asarray(values, dtype=np.float32)
    if grid.ndim != 3 or min(grid.shape) <= 0 or not np.all(np.isfinite(grid)):
        reason = f"{name} must be a finite non-empty HxWxD grid"
        raise Gate2PipelineError(reason)
    return np.asarray(grid.reshape(-1, int(np.size(grid, axis=2))), dtype=np.float32)


def _profile_event(
    event: str,
    query_id: str,
    started: float,
    details: Dict[str, object],
) -> None:
    if os.environ.get(_PROFILE_ENV) != "1":
        return
    payload: Dict[str, object] = {
        "event": event,
        "query_id": query_id,
        "elapsed_s": round(perf_counter() - started, 6),
    }
    payload.update(details)
    _LOGGER.warning("DARC_GATE2_PROFILE %s", json.dumps(payload, sort_keys=True))
