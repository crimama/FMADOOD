from __future__ import annotations

from typing import Final, Iterator, List, NamedTuple, Optional, Tuple

import numpy as np

from flow_tte.darc_feature_stream import ImageFeatures
from flow_tte.darc_gate2_coordinate_maps import (
    coarse_geometry,
    derive_ransac_seed,
    high_feature_dimension,
    high_feature_grid,
    inverse_or_identity,
    query_token_points,
)
from flow_tte.darc_gate2_correspondence_types import (
    DEFAULT_CORRESPONDENCE_CONFIG,
    MAX_LOCAL_CANDIDATES,
    BoolArray,
    CandidateAudit,
    CandidateChunk,
    CandidateLocation,
    CandidateRequest,
    CorrespondenceConfig,
    CorrespondenceError,
    CorrespondenceQuery,
    FeatureArray,
    Float64Array,
    FloatArray,
    IntArray,
    PreparedCorrespondence,
    SelectedSupport,
    SupportRegistration,
    TokenPointChunk,
)
from flow_tte.darc_geometry import Point2D, fit_similarity_ransac, mutual_nearest_pairs
from flow_tte.darc_scoring import LocalCandidateSet
from flow_tte.darc_tiling import clipped_token_candidates, owner_crop

_ROW_OFFSETS: Final[IntArray] = np.asarray(
    (-1, -1, -1, 0, 0, 0, 1, 1, 1),
    dtype=np.int64,
)
_COLUMN_OFFSETS: Final[IntArray] = np.asarray(
    (-1, 0, 1, -1, 0, 1, -1, 0, 1),
    dtype=np.int64,
)


class _SupportCandidateOutputs(NamedTuple):
    values: FloatArray
    valid: BoolArray
    mapped_points: FloatArray
    counts: IntArray


def prepare_correspondence(
    query: CorrespondenceQuery,
    supports: Tuple[SelectedSupport, ...],
    config: CorrespondenceConfig = DEFAULT_CORRESPONDENCE_CONFIG,
) -> PreparedCorrespondence:
    if not query.query_id:
        raise CorrespondenceError("query identity must be non-empty")
    if not supports or len({item.support_id for item in supports}) != len(supports):
        raise CorrespondenceError("ordered supports must be non-empty with unique IDs")
    query_dimension = high_feature_dimension(query.features)
    query_coarse = coarse_geometry(query.features, config)
    registrations: List[SupportRegistration] = []
    for support in supports:
        if not support.support_id or high_feature_dimension(support.features) != query_dimension:
            raise CorrespondenceError("support IDs/features must share the query dimension")
        support_coarse = coarse_geometry(support.features, config)
        if support_coarse.grid.features.shape[1] != query_coarse.grid.features.shape[1]:
            raise CorrespondenceError("coarse support/query feature dimensions must match")
        pairs = mutual_nearest_pairs(support_coarse.grid, query_coarse.grid)
        ransac_seed = derive_ransac_seed(query.query_id, support.support_id)
        alignment = fit_similarity_ransac(pairs, config.ransac, seed=ransac_seed)
        inverse, invertible = inverse_or_identity(alignment.matrix)
        registrations.append(
            SupportRegistration(
                support_id=support.support_id,
                query_resize=query_coarse.resize,
                support_resize=support_coarse.resize,
                alignment=alignment,
                query_to_support=inverse,
                pair_count=len(pairs.support_points),
                ransac_seed=ransac_seed,
                invertible=invertible,
            ),
        )
    return PreparedCorrespondence(
        query.query_id,
        query.features,
        supports,
        tuple(registrations),
        query_dimension,
    )


def locate_support_candidates(
    support: ImageFeatures,
    point: Point2D,
) -> Optional[CandidateLocation]:
    crop = owner_crop(point, support.crops)
    if crop is None:
        return None
    crop_index = support.crops.index(crop)
    _, token_grid = high_feature_grid(support, crop_index)
    return CandidateLocation(
        crop_index=crop_index,
        token_indices=clipped_token_candidates(point, crop, token_grid),
    )


def iter_query_token_chunks(
    prepared: PreparedCorrespondence,
    crop_index: int,
    chunk_size: int = 256,
) -> Iterator[TokenPointChunk]:
    if chunk_size <= 0:
        raise CorrespondenceError("token chunk size must be positive")
    if crop_index < 0 or crop_index >= len(prepared.query.high):
        raise CorrespondenceError("query crop index is outside the prepared inventory")
    grid = prepared.query.high[crop_index]
    token_count = int(np.size(grid, axis=0)) * int(np.size(grid, axis=1))
    for start in range(0, token_count, chunk_size):
        request = CandidateRequest(crop_index, start, min(start + chunk_size, token_count))
        yield TokenPointChunk(request=request, points=query_token_points(prepared, request))


def build_l0_candidate_chunk(
    prepared: PreparedCorrespondence,
    request: CandidateRequest,
) -> CandidateChunk:
    return _build_candidate_chunk(prepared, request, aligned=False)


def build_l1_candidate_chunk(
    prepared: PreparedCorrespondence,
    request: CandidateRequest,
) -> CandidateChunk:
    return _build_candidate_chunk(prepared, request, aligned=True)


def _build_candidate_chunk(
    prepared: PreparedCorrespondence,
    request: CandidateRequest,
    aligned: bool,
) -> CandidateChunk:
    points = query_token_points(prepared, request)
    shape = (len(points), len(prepared.supports))
    values: FloatArray = np.zeros(
        (*shape, MAX_LOCAL_CANDIDATES, prepared.feature_dimension),
        dtype=np.float32,
    )
    valid: BoolArray = np.zeros((*shape, MAX_LOCAL_CANDIDATES), dtype=np.bool_)
    mapped_points: FloatArray = np.zeros((*shape, 2), dtype=np.float32)
    counts: IntArray = np.zeros(shape, dtype=np.int64)
    for support_index, (support, registration) in enumerate(
        zip(prepared.supports, prepared.registrations),
    ):
        if aligned and not registration.accepted:
            continue
        _fill_support_candidates(
            prepared,
            support,
            registration,
            points,
            aligned,
            _SupportCandidateOutputs(
                values=values[:, support_index],
                valid=valid[:, support_index],
                mapped_points=mapped_points[:, support_index],
                counts=counts[:, support_index],
            ),
        )
    support_valid: BoolArray = np.asarray(counts > 0, dtype=np.bool_)
    audit = CandidateAudit(
        request=request,
        ordered_support_ids=tuple(item.support_id for item in prepared.supports),
        alignment_accepted=np.asarray(
            [registration.accepted for registration in prepared.registrations],
            dtype=np.bool_,
        ),
        mapped_support_points=mapped_points,
        candidate_counts=counts,
        support_valid=support_valid,
    )
    return CandidateChunk(local=LocalCandidateSet(values=values, valid=valid), audit=audit)


def _fill_support_candidates(  # noqa: PLR0913 -- explicit vectorized output views
    prepared: PreparedCorrespondence,
    support: SelectedSupport,
    registration: SupportRegistration,
    points: FloatArray,
    aligned: bool,
    outputs: _SupportCandidateOutputs,
) -> None:
    support_features = support.features
    mapped = _mapped_support_points(
        prepared,
        support_features,
        registration,
        points,
        aligned,
    )
    outputs.mapped_points[:] = mapped
    owners = _owner_crop_indices(support_features, mapped)
    safe_owners: IntArray = np.asarray(np.maximum(owners, 0), dtype=np.int64)
    crop_x0: Float64Array = np.asarray(
        [crop.x0 for crop in support_features.crops],
        dtype=np.float64,
    )
    crop_y0: Float64Array = np.asarray(
        [crop.y0 for crop in support_features.crops],
        dtype=np.float64,
    )
    spacings: Float64Array = np.asarray(
        [
            crop.width / int(np.size(grid, axis=1))
            for crop, grid in zip(support_features.crops, support_features.high)
        ],
        dtype=np.float64,
    )
    column_positions: Float64Array = np.asarray(
        (mapped[:, 0] - crop_x0[safe_owners] + 0.5) / spacings[safe_owners] - 0.5,
        dtype=np.float64,
    )
    row_positions: Float64Array = np.asarray(
        (mapped[:, 1] - crop_y0[safe_owners] + 0.5) / spacings[safe_owners] - 0.5,
        dtype=np.float64,
    )
    center_columns: IntArray = np.asarray(
        np.floor(column_positions + 0.5),
        dtype=np.int64,
    )
    center_rows: IntArray = np.asarray(
        np.floor(row_positions + 0.5),
        dtype=np.int64,
    )

    for crop_index, grid in enumerate(support_features.high):
        token_indices: IntArray = np.asarray(
            np.arange(len(points), dtype=np.int64)[owners == crop_index],
            dtype=np.int64,
        )
        if not len(token_indices):
            continue
        height = int(np.size(grid, axis=0))
        width = int(np.size(grid, axis=1))
        rows: IntArray = np.asarray(
            center_rows[token_indices, None] + _ROW_OFFSETS[None, :],
            dtype=np.int64,
        )
        columns: IntArray = np.asarray(
            center_columns[token_indices, None] + _COLUMN_OFFSETS[None, :],
            dtype=np.int64,
        )
        inside: BoolArray = np.asarray(
            (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width),
            dtype=np.bool_,
        )
        order: IntArray = np.asarray(
            np.argsort(~inside, axis=1, kind="stable"),
            dtype=np.int64,
        )
        ordered_inside: BoolArray = np.asarray(
            np.take_along_axis(inside, order, axis=1),
            dtype=np.bool_,
        )
        ordered_rows: IntArray = np.asarray(
            np.take_along_axis(rows, order, axis=1),
            dtype=np.int64,
        )
        ordered_columns: IntArray = np.asarray(
            np.take_along_axis(columns, order, axis=1),
            dtype=np.int64,
        )
        safe_rows: IntArray = np.asarray(np.clip(ordered_rows, 0, height - 1), dtype=np.int64)
        safe_columns: IntArray = np.asarray(
            np.clip(ordered_columns, 0, width - 1),
            dtype=np.int64,
        )
        gathered: FeatureArray = np.asarray(
            grid[safe_rows, safe_columns],
            dtype=np.float16,
        )
        gathered[~ordered_inside] = np.float16(0.0)
        outputs.values[token_indices] = gathered
        outputs.valid[token_indices] = ordered_inside
        outputs.counts[token_indices] = np.asarray(
            np.sum(ordered_inside, axis=1),
            dtype=np.int64,
        )


def _mapped_support_points(
    prepared: PreparedCorrespondence,
    support: ImageFeatures,
    registration: SupportRegistration,
    points: FloatArray,
    aligned: bool,
) -> Float64Array:
    if not aligned:
        values: Float64Array = np.asarray(points, dtype=np.float64)
        return np.asarray(
            np.column_stack(
                (
                    (values[:, 0] + 0.5)
                    * support.native_size.width
                    / prepared.query.native_size.width
                    - 0.5,
                    (values[:, 1] + 0.5)
                    * support.native_size.height
                    / prepared.query.native_size.height
                    - 0.5,
                ),
            ),
            dtype=np.float64,
        )
    mapped: Float64Array = np.zeros((len(points), 2), dtype=np.float64)
    for token_index in range(len(points)):
        point = Point2D(
            x=float(np.sum(points[token_index : token_index + 1, 0:1])),
            y=float(np.sum(points[token_index : token_index + 1, 1:2])),
        )
        value = registration.map_query_native(point)
        mapped[token_index] = value
    return mapped


def _owner_crop_indices(support: ImageFeatures, mapped: Float64Array) -> IntArray:
    ordered_python = sorted(
        range(len(support.crops)),
        key=lambda index: (support.crops[index].y0, support.crops[index].x0),
    )
    ordered_indices: IntArray = np.asarray(
        ordered_python,
        dtype=np.int64,
    )
    crops = tuple(support.crops[index] for index in ordered_python)
    x0: Float64Array = np.asarray([crop.x0 for crop in crops], dtype=np.float64)
    y0: Float64Array = np.asarray([crop.y0 for crop in crops], dtype=np.float64)
    widths: Float64Array = np.asarray([crop.width for crop in crops], dtype=np.float64)
    heights: Float64Array = np.asarray([crop.height for crop in crops], dtype=np.float64)
    xs = mapped[:, 0:1]
    ys = mapped[:, 1:2]
    containing: BoolArray = np.asarray(
        (xs >= x0) & (xs < x0 + widths) & (ys >= y0) & (ys < y0 + heights),
        dtype=np.bool_,
    )
    margin = 32.0
    interior: BoolArray = np.asarray(
        containing
        & (xs >= x0 + margin)
        & (xs < x0 + widths - margin)
        & (ys >= y0 + margin)
        & (ys < y0 + heights - margin),
        dtype=np.bool_,
    )
    use_interior: BoolArray = np.asarray(np.any(interior, axis=1), dtype=np.bool_)
    candidates: BoolArray = np.asarray(
        np.where(use_interior[:, None], interior, containing),
        dtype=np.bool_,
    )
    center_x: Float64Array = np.asarray(x0 + (widths - 1.0) / 2.0, dtype=np.float64)
    center_y: Float64Array = np.asarray(y0 + (heights - 1.0) / 2.0, dtype=np.float64)
    distances: Float64Array = np.asarray(
        (xs - center_x) * (xs - center_x) + (ys - center_y) * (ys - center_y),
        dtype=np.float64,
    )
    masked: Float64Array = np.asarray(
        np.where(candidates, distances, np.inf),
        dtype=np.float64,
    )
    positions: IntArray = np.asarray(np.argmin(masked, axis=1), dtype=np.int64)
    owners: IntArray = np.asarray(ordered_indices[positions], dtype=np.int64)
    owners[~np.any(containing, axis=1)] = -1
    return owners
