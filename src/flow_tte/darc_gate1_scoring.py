from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

import numpy as np
import numpy.typing as npt

from flow_tte.darc_feature_stream import ImageFeatures, resize_score_grid, stitch_score_grids
from flow_tte.darc_geometry import ImageSize
from flow_tte.darc_knn import ChunkedKnnConfig, SupportTokens, cosine_1nn_scores
from flow_tte.darc_knn import rank_supports as coarse_rank_supports

FloatArray = npt.NDArray[np.float32]
FeatureArray = npt.NDArray[np.float16]


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class CandidateFeatures:
    features: Mapping[str, ImageFeatures]
    config: ChunkedKnnConfig


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class ScoredMaps:
    low: FloatArray
    bilinear_null: FloatArray
    high: FloatArray
    selected_support_ids: Tuple[str, ...]


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class ReferenceScores:
    low: FloatArray
    bilinear_null: FloatArray
    high: FloatArray


def score_query(query: ImageFeatures, candidates: CandidateFeatures) -> ScoredMaps:
    """Score one query against coarse-ranked supports at paired resolutions."""
    coarse_supports = tuple(
        SupportTokens(support_id, feature.coarse.reshape(-1, feature.coarse.shape[-1]))
        for support_id, feature in candidates.features.items()
    )
    selected_ids = coarse_rank_supports(
        query.coarse.reshape(-1, query.coarse.shape[-1]),
        coarse_supports,
        candidates.config,
    )
    selected = tuple(candidates.features[support_id] for support_id in selected_ids)
    low_grids = _score_grids(query.low, selected, candidates.config, high=False)
    high_grids = _score_grids(query.high, selected, candidates.config, high=True)
    high_grid_size = ImageSize(
        height=int(query.high[0].shape[0]),
        width=int(query.high[0].shape[1]),
    )
    null_grids = tuple(resize_score_grid(grid, high_grid_size) for grid in low_grids)
    return ScoredMaps(
        low=stitch_score_grids(query.native_size, query.crops, low_grids),
        bilinear_null=stitch_score_grids(query.native_size, query.crops, null_grids),
        high=stitch_score_grids(query.native_size, query.crops, high_grids),
        selected_support_ids=selected_ids,
    )


def leave_one_out_references(memory: CandidateFeatures) -> ReferenceScores:
    """Build normal score tails with each image excluded from its own bank."""
    if len(memory.features) < 2:
        message = "leave-one-out references require at least two images"
        raise ValueError(message)
    low = []
    null = []
    high = []
    for query_id, query in memory.features.items():
        candidates = CandidateFeatures(
            features={
                support_id: features
                for support_id, features in memory.features.items()
                if support_id != query_id
            },
            config=memory.config,
        )
        scored = score_query(query, candidates)
        low.append(scored.low.reshape(-1))
        null.append(scored.bilinear_null.reshape(-1))
        high.append(scored.high.reshape(-1))
    return ReferenceScores(
        low=np.asarray(np.concatenate(low), dtype=np.float32),
        bilinear_null=np.asarray(np.concatenate(null), dtype=np.float32),
        high=np.asarray(np.concatenate(high), dtype=np.float32),
    )


def _score_grids(
    query_grids: Tuple[FeatureArray, ...],
    supports: Tuple[ImageFeatures, ...],
    config: ChunkedKnnConfig,
    *,
    high: bool,
) -> Tuple[FloatArray, ...]:
    support_grids = tuple(
        grid
        for support in supports
        for grid in (support.high if high else support.low)
    )
    dimension = int(query_grids[0].shape[-1])
    query_tokens = np.concatenate(
        tuple(grid.reshape(-1, dimension) for grid in query_grids),
        axis=0,
    )
    memory_tokens = np.concatenate(
        tuple(grid.reshape(-1, dimension) for grid in support_grids),
        axis=0,
    )
    scores = cosine_1nn_scores(query_tokens, memory_tokens, config)
    output = []
    offset = 0
    for grid in query_grids:
        token_count = int(grid.shape[0] * grid.shape[1])
        output.append(scores[offset : offset + token_count].reshape(grid.shape[:2]))
        offset += token_count
    return tuple(output)
