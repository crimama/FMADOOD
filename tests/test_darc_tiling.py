from __future__ import annotations

import numpy as np

from flow_tte.darc_geometry import ImageSize, Point2D
from flow_tte.darc_tiling import (
    CropScores,
    NativeCrop,
    TilingSpec,
    TokenGrid,
    blend_crop_scores,
    clipped_token_candidates,
    hann_window,
    native_crop_grid,
    owner_crop,
)


def test_native_crop_grid_covers_right_bottom_edges() -> None:
    # Given: an image whose dimensions are not stride-aligned.
    shape = ImageSize(height=1000, width=900)

    # When: 512-pixel crops are laid out at stride 384.
    crops = native_crop_grid(shape, TilingSpec())

    # Then: explicit edge starts cover the complete image.
    origins = {(crop.y0, crop.x0) for crop in crops}
    assert origins == {
        (0, 0),
        (0, 384),
        (0, 388),
        (384, 0),
        (384, 384),
        (384, 388),
        (488, 0),
        (488, 384),
        (488, 388),
    }


def test_owner_crop_prefers_interior_then_y_x_tie_break() -> None:
    # Given: four overlapping crops equidistant from one point.
    crops = (
        NativeCrop(y0=0, x0=0, height=512, width=512),
        NativeCrop(y0=0, x0=384, height=512, width=512),
        NativeCrop(y0=384, x0=0, height=512, width=512),
        NativeCrop(y0=384, x0=384, height=512, width=512),
    )
    point = Point2D(x=447.5, y=447.5)

    # When: ownership is resolved with the frozen 32-pixel preference.
    owner = owner_crop(point, crops)

    # Then: the deterministic (y0, x0) tie-break selects the first crop.
    assert owner == crops[0]


def test_owner_crop_uses_edge_crop_without_interior_candidate() -> None:
    # Given: an image-edge point contained by a single crop without 32-pixel margin.
    crop = NativeCrop(y0=0, x0=0, height=512, width=512)

    # When: ownership is resolved.
    owner = owner_crop(Point2D(x=4.0, y=7.0), (crop,))

    # Then: the containing edge crop remains usable.
    assert owner == crop


def test_clipped_candidates_use_half_pixel_token_location() -> None:
    # Given: the native center of the top-left 8-pixel-spacing token.
    crop = NativeCrop(y0=100, x0=200, height=512, width=512)
    grid = TokenGrid(shape=ImageSize(height=64, width=64), native_spacing=8.0)
    point = Point2D(x=203.5, y=103.5)

    # When: its 3x3 neighbourhood is clipped to the token grid.
    candidates = clipped_token_candidates(point, crop, grid)

    # Then: only the four valid corner neighbours remain in row-major order.
    expected = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    assert np.array_equal(candidates, expected)


def test_hann_window_has_frozen_floor_and_unit_center() -> None:
    # Given: an odd two-dimensional crop shape.
    shape = ImageSize(height=5, width=5)

    # When: the separable Hann blend weight is constructed.
    weight = hann_window(shape)

    # Then: its edge is floored at 0.1 and its center stays one.
    assert np.min(weight) == np.float32(0.1)
    assert weight[2, 2] == np.float32(1.0)


def test_blend_crop_scores_normalizes_overlap_weights() -> None:
    # Given: two one-row crops with one overlapping pixel.
    left = CropScores(
        crop=NativeCrop(y0=0, x0=0, height=1, width=2),
        scores=np.asarray([[1.0, 1.0]], dtype=np.float32),
    )
    right = CropScores(
        crop=NativeCrop(y0=0, x0=1, height=1, width=2),
        scores=np.asarray([[3.0, 3.0]], dtype=np.float32),
    )

    # When: Hann-weighted crop maps are accumulated and normalized.
    blended = blend_crop_scores(ImageSize(height=1, width=3), (left, right))

    # Then: the overlap is the normalized weighted mean.
    assert np.array_equal(blended, np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))
