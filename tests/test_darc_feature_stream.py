from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from flow_tte.darc_feature_stream import (
    DarcFeatureStream,
    FeatureStreamConfig,
    stitch_score_grids,
)
from flow_tte.darc_geometry import ImageSize
from flow_tte.darc_tiling import TilingSpec, native_crop_grid


class _RecordingExtractor:
    def __init__(self, spacing: int) -> None:
        self.spacing = spacing
        self.calls: List[Tuple[int, int]] = []

    def __call__(self, pixels: torch.Tensor) -> torch.Tensor:
        height, width = int(pixels.shape[-2]), int(pixels.shape[-1])
        self.calls.append((height, width))
        return torch.ones((height // self.spacing, width // self.spacing, 2))


def test_feature_stream_keeps_native_crops_and_paired_token_spacings() -> None:
    # Given
    micro = _RecordingExtractor(spacing=16)
    coarse = _RecordingExtractor(spacing=16)
    stream = DarcFeatureStream(
        micro_extractor=micro,
        coarse_extractor=coarse,
        config=FeatureStreamConfig(device="cpu"),
    )
    image = np.zeros((700, 700, 3), dtype=np.uint8)

    # When
    features = stream.extract(image)

    # Then
    assert len(features.crops) == 4
    assert features.low[0].shape == (32, 32, 2)
    assert features.high[0].shape == (64, 64, 2)
    assert coarse.calls == [(672, 672)]
    assert micro.calls.count((512, 512)) == 4
    assert micro.calls.count((1024, 1024)) == 4


def test_stitch_score_grids_preserves_constant_overlaps() -> None:
    # Given
    shape = ImageSize(height=700, width=700)
    crops = native_crop_grid(shape, TilingSpec())
    grids = tuple(np.full((32, 32), 0.25, dtype=np.float32) for _ in crops)

    # When
    score_map = stitch_score_grids(shape, crops, grids)

    # Then
    np.testing.assert_allclose(score_map, 0.25, atol=1e-6)


def test_feature_stream_can_skip_unused_low_resolution_branch() -> None:
    # Given
    micro = _RecordingExtractor(spacing=16)
    coarse = _RecordingExtractor(spacing=16)
    stream = DarcFeatureStream(
        micro_extractor=micro,
        coarse_extractor=coarse,
        config=FeatureStreamConfig(device="cpu", include_low=False),
    )

    # When
    features = stream.extract(np.zeros((512, 512, 3), dtype=np.uint8))

    # Then
    assert features.low == ()
    assert features.high[0].shape == (64, 64, 2)
    assert micro.calls == [(1024, 1024)]
    assert coarse.calls == [(672, 672)]
