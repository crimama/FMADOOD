from __future__ import annotations

import math
from hashlib import sha256

import numpy as np
import pytest

from flow_tte import darc_gate2_correspondence as correspondence_module
from flow_tte.darc_feature_stream import ImageFeatures
from flow_tte.darc_gate2_coordinate_maps import (
    invert_similarity,
    map_identity_native,
    query_token_points,
)
from flow_tte.darc_gate2_correspondence import (
    CandidateRequest,
    CorrespondenceConfig,
    CorrespondenceQuery,
    SelectedSupport,
    build_l0_candidate_chunk,
    build_l1_candidate_chunk,
    coarse_geometry,
    iter_query_token_chunks,
    locate_support_candidates,
    prepare_correspondence,
)
from flow_tte.darc_geometry import ImageSize, Point2D
from flow_tte.darc_tiling import NativeCrop

_DEFAULT_NATIVE = ImageSize(height=512, width=512)


def _coarse_identity() -> np.ndarray:
    return np.eye(16, dtype=np.float16).reshape(4, 4, 16)


def _high_grid(value: float = 0.0) -> np.ndarray:
    rows, columns = np.meshgrid(
        np.arange(64, dtype=np.float16),
        np.arange(64, dtype=np.float16),
        indexing="ij",
    )
    return np.stack((rows + value, columns + value), axis=2).astype(np.float16)


def _image(
    coarse: np.ndarray,
    crops: tuple[NativeCrop, ...] | None = None,
    high: tuple[np.ndarray, ...] | None = None,
    native: ImageSize = _DEFAULT_NATIVE,
) -> ImageFeatures:
    active_crops = crops or (NativeCrop(y0=0, x0=0, height=512, width=512),)
    active_high = high or tuple(_high_grid(float(index)) for index in range(len(active_crops)))
    return ImageFeatures(
        native_size=native,
        crops=active_crops,
        coarse=np.asarray(coarse, dtype=np.float16),
        low=active_high,
        high=active_high,
    )


def test_coarse_geometry_round_trips_odd_aspect_realized_resize() -> None:
    # Given: an odd-aspect native image and its realized aligned coarse grid.
    native = ImageSize(height=513, width=777)
    scale = 672 / native.height
    realized = ImageSize(height=int(native.height * scale), width=int(native.width * scale))
    retained = ImageSize(
        height=realized.height - realized.height % 16,
        width=realized.width - realized.width % 16,
    )
    coarse = np.zeros((retained.height // 16, retained.width // 16, 2), dtype=np.float16)
    image = _image(coarse, native=native)
    point = Point2D(x=731.25, y=487.75)

    # When: the frozen 672-short-edge geometry maps the point out and back.
    geometry = coarse_geometry(image)
    restored = geometry.resize.resized_to_native(geometry.resize.native_to_resized(point))

    # Then: integer realization, patch alignment, and half-pixel mapping are exact.
    assert geometry.resize.realized == realized
    assert geometry.resize.retained == retained
    assert math.isclose(restored.x, point.x, abs_tol=1e-9)
    assert math.isclose(restored.y, point.y, abs_tol=1e-9)


def test_prepared_identity_alignment_maps_query_back_to_support() -> None:
    # Given: equal-size images with unique, identical coarse descriptors.
    query = _image(_coarse_identity())
    support = SelectedSupport(support_id="support-a", features=_image(_coarse_identity()))
    config = CorrespondenceConfig(short_edge=64)
    point = Point2D(x=203.5, y=347.5)

    # When: MNN and RANSAC estimate support-to-query geometry.
    prepared = prepare_correspondence(CorrespondenceQuery("query-clean", query), (support,), config)
    mapped = prepared.registrations[0].map_query_native(point)

    # Then: the accepted inverse transform returns the same native center.
    assert prepared.registrations[0].accepted
    assert prepared.registrations[0].pair_count == 16
    expected_seed = int.from_bytes(
        sha256(b"query-clean\0support-a").digest()[:8],
        byteorder="big",
        signed=False,
    )
    assert prepared.registrations[0].ransac_seed == expected_seed
    assert math.isclose(mapped.x, point.x, abs_tol=1e-5)
    assert math.isclose(mapped.y, point.y, abs_tol=1e-5)


def test_identity_mapping_uses_native_size_half_pixel_affine() -> None:
    # Given: a query and support with different native sizes.
    point = Point2D(x=99.5, y=49.5)

    # When: L0 maps their normalized native positions without registration.
    mapped = map_identity_native(
        point,
        ImageSize(height=100, width=200),
        ImageSize(height=200, width=300),
    )

    # Then: the deterministic half-pixel resize convention is used.
    assert math.isclose(mapped.x, 149.5)
    assert math.isclose(mapped.y, 99.5)


def test_inverse_similarity_round_trips_nontrivial_transform() -> None:
    # Given: a scaled rotation plus translation and a support-space point.
    angle = math.radians(17.0)
    scale = 1.13
    matrix = np.asarray(
        [
            [scale * math.cos(angle), -scale * math.sin(angle), 21.0],
            [scale * math.sin(angle), scale * math.cos(angle), -13.0],
        ],
        dtype=np.float32,
    )
    support_point = np.asarray([37.5, 81.5], dtype=np.float32)
    query_point = matrix[:, :2] @ support_point + matrix[:, 2]

    # When: the accepted support-to-query transform is inverted.
    inverse = invert_similarity(matrix)
    restored = inverse[:, :2] @ query_point + inverse[:, 2]

    # Then: query coordinates return to the original support coordinates.
    assert np.allclose(restored, support_point, atol=1e-5)


def test_candidate_location_resolves_overlap_and_clips_edge_neighbourhood() -> None:
    # Given: overlapping support crops and a frozen 64x64 high-token grid.
    crops = (
        NativeCrop(y0=0, x0=0, height=512, width=512),
        NativeCrop(y0=0, x0=384, height=512, width=512),
        NativeCrop(y0=384, x0=0, height=512, width=512),
        NativeCrop(y0=384, x0=384, height=512, width=512),
    )
    image = _image(
        _coarse_identity(),
        crops=crops,
        high=tuple(_high_grid(float(index)) for index in range(4)),
        native=ImageSize(height=896, width=896),
    )

    # When: an overlap tie and a top-left edge point are localized.
    overlap = locate_support_candidates(image, Point2D(x=447.5, y=447.5))
    edge = locate_support_candidates(image, Point2D(x=3.5, y=3.5))

    # Then: ownership is deterministic and the 3x3 edge window clips to four tokens.
    assert overlap is not None
    assert overlap.crop_index == 0
    assert edge is not None
    assert edge.crop_index == 0
    assert np.array_equal(
        edge.token_indices,
        np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64),
    )


def test_chunked_l1_equals_l0_for_accepted_identity_alignment() -> None:
    # Given: one accepted identity support and a non-full-map token slice.
    query = _image(_coarse_identity())
    support = SelectedSupport(support_id="support-a", features=_image(_coarse_identity()))
    prepared = prepare_correspondence(
        CorrespondenceQuery("query-clean", query),
        (support,),
        CorrespondenceConfig(short_edge=64),
    )
    request = CandidateRequest(crop_index=0, start=56 * 64 + 56, stop=56 * 64 + 58)

    # When: L0 and L1 candidates are built for only that chunk.
    l0 = build_l0_candidate_chunk(prepared, request)
    l1 = build_l1_candidate_chunk(prepared, request)

    # Then: identity registration preserves exact candidates, shape, and validity.
    assert l0.local.values.shape == (2, 1, 9, 2)
    assert np.array_equal(l1.local.values, l0.local.values)
    assert np.array_equal(l1.local.valid, l0.local.valid)
    assert np.array_equal(l1.audit.candidate_counts, np.asarray([[9], [9]]))


def test_candidate_chunk_reuses_prevalidated_high_grids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: prepare_correspondence has already validated every high feature grid.
    query = _image(_coarse_identity())
    support = SelectedSupport(support_id="support-a", features=_image(_coarse_identity()))
    prepared = prepare_correspondence(
        CorrespondenceQuery("query-clean", query),
        (support,),
        CorrespondenceConfig(short_edge=64),
    )

    def unexpected_revalidation(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("candidate assembly revalidated a prepared high grid")

    monkeypatch.setattr(correspondence_module, "high_feature_grid", unexpected_revalidation)

    # When: local candidates are gathered from that prepared correspondence.
    chunk = build_l0_candidate_chunk(prepared, CandidateRequest(0, 0, 2))

    # Then: assembly indexes the validated float16 grid without rescanning it.
    assert chunk.local.values.shape == (2, 1, 9, 2)
    assert np.array_equal(chunk.audit.candidate_counts, np.asarray([[4], [6]]))


def test_vectorized_candidate_gather_matches_scalar_overlap_reference() -> None:
    # Given: overlapping crops and a contiguous token band crossing their interior boundary.
    crops = (
        NativeCrop(y0=0, x0=0, height=512, width=512),
        NativeCrop(y0=0, x0=384, height=512, width=512),
        NativeCrop(y0=384, x0=0, height=512, width=512),
        NativeCrop(y0=384, x0=384, height=512, width=512),
    )
    image = _image(
        _coarse_identity(),
        crops=crops,
        high=tuple(_high_grid(float(index)) for index in range(4)),
        native=ImageSize(height=896, width=896),
    )
    support = SelectedSupport(support_id="support-a", features=image)
    prepared = prepare_correspondence(
        CorrespondenceQuery("query-overlap", image),
        (support,),
        CorrespondenceConfig(short_edge=64),
    )
    request = CandidateRequest(crop_index=0, start=50 * 64, stop=58 * 64)

    # When: the vectorized gather and the scalar public locator assemble candidates.
    chunk = build_l0_candidate_chunk(prepared, request)
    reference_values = np.zeros_like(chunk.local.values)
    reference_valid = np.zeros_like(chunk.local.valid)
    reference_counts = np.zeros_like(chunk.audit.candidate_counts)
    for token_index, mapped_values in enumerate(chunk.audit.mapped_support_points[:, 0]):
        point = Point2D(x=float(mapped_values[0]), y=float(mapped_values[1]))
        location = locate_support_candidates(image, point)
        assert location is not None
        indices = location.token_indices
        count = len(indices)
        reference_values[token_index, 0, :count] = np.asarray(
            image.high[location.crop_index][indices[:, 0], indices[:, 1]],
            dtype=np.float32,
        )
        reference_valid[token_index, 0, :count] = True
        reference_counts[token_index, 0] = count

    # Then: crop ownership, compact row-major ordering, values, and masks are exact.
    assert np.array_equal(chunk.local.values, reference_values)
    assert np.array_equal(chunk.local.valid, reference_valid)
    assert np.array_equal(chunk.audit.candidate_counts, reference_counts)


def test_vectorized_registered_mapping_matches_scalar_float32_path() -> None:
    # Given: an accepted registration with a nontrivial query-to-support transform.
    image = _image(_coarse_identity())
    support = SelectedSupport(support_id="support-a", features=image)
    prepared = prepare_correspondence(
        CorrespondenceQuery("query-shifted", image),
        (support,),
        CorrespondenceConfig(short_edge=64),
    )
    registration = prepared.registrations[0]._replace(
        query_to_support=np.asarray(
            [[0.997, -0.021, 1.25], [0.021, 0.997, -0.75]],
            dtype=np.float32,
        ),
    )
    transformed = prepared._replace(registrations=(registration,))
    request = CandidateRequest(crop_index=0, start=100, stop=140)

    # When: the vectorized L1 path and scalar registration map the same token points.
    chunk = build_l1_candidate_chunk(transformed, request)
    points = query_token_points(transformed, request)
    scalar = np.asarray(
        [
            registration.map_query_native(
                Point2D(x=float(point[0]), y=float(point[1])),
            )
            for point in points
        ],
        dtype=np.float32,
    )

    # Then: the persisted mapped coordinates retain the exact scalar float32 result.
    assert np.array_equal(chunk.audit.mapped_support_points[:, 0], scalar)


def test_rejected_alignment_keeps_ordered_support_slot_invalid() -> None:
    # Given: one alignable support followed by one support with only one MNN pair.
    query = _image(_coarse_identity())
    accepted = SelectedSupport(support_id="accepted", features=_image(_coarse_identity()))
    rejected_coarse = np.ones((4, 4, 16), dtype=np.float16)
    rejected = SelectedSupport(support_id="rejected", features=_image(rejected_coarse))
    prepared = prepare_correspondence(
        CorrespondenceQuery("query-clean", query),
        (accepted, rejected),
        CorrespondenceConfig(short_edge=64),
    )
    request = CandidateRequest(crop_index=0, start=0, stop=1)

    # When: the aligned candidate chunk is built.
    result = build_l1_candidate_chunk(prepared, request)

    # Then: K/order stay fixed and the rejected support is represented as invalid.
    assert result.local.values.shape == (1, 2, 9, 2)
    assert result.audit.ordered_support_ids == ("accepted", "rejected")
    assert np.array_equal(result.audit.alignment_accepted, [True, False])
    assert np.all(result.local.valid[0, 0, :4])
    assert not np.any(result.local.valid[:, 1])
    assert np.array_equal(result.audit.candidate_counts, np.asarray([[4, 0]]))


def test_query_token_iterator_partitions_without_materializing_candidates() -> None:
    # Given: a 64x64 query grid and a 256-token runtime chunk target.
    query = _image(_coarse_identity())
    support = SelectedSupport(support_id="support-a", features=_image(_coarse_identity()))
    prepared = prepare_correspondence(
        CorrespondenceQuery("query-clean", query),
        (support,),
        CorrespondenceConfig(short_edge=64),
    )

    # When: deterministic query token chunks are enumerated.
    chunks = tuple(iter_query_token_chunks(prepared, crop_index=0, chunk_size=256))

    # Then: all 4096 row-major tokens are covered exactly once in bounded chunks.
    assert len(chunks) == 16
    assert chunks[0].request == CandidateRequest(crop_index=0, start=0, stop=256)
    assert chunks[-1].request == CandidateRequest(crop_index=0, start=3840, stop=4096)
    assert np.array_equal(chunks[0].points[0], np.asarray([3.5, 3.5], dtype=np.float32))
