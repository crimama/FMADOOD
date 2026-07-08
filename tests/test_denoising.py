from __future__ import annotations

from typing import List, Tuple, cast

import numpy as np
import numpy.typing as npt
import pytest

from flow_tte.denoising import FeatureDenoisingError, PositionMeanArtifactDenoiser


def test_position_mean_artifact_denoiser_removes_support_position_bias() -> None:
    content = cast("npt.NDArray[np.float32]", np.array([10.0, -3.0], dtype=np.float32))
    artifact = cast(
        "npt.NDArray[np.float32]",
        np.array(
            [
                [[1.0, -2.0], [0.5, -1.0]],
                [[-0.25, 0.5], [2.0, -4.0]],
            ],
            dtype=np.float32,
        ),
    )
    support: List[npt.NDArray[np.float32]] = [
        cast("npt.NDArray[np.float32]", content.reshape(1, 1, 2) + artifact + shift)
        for shift in (
            cast("npt.NDArray[np.float32]", np.array([0.0, 0.0], dtype=np.float32)),
            cast("npt.NDArray[np.float32]", np.array([2.0, 1.0], dtype=np.float32)),
        )
    ]

    denoiser = PositionMeanArtifactDenoiser.fit(support, alpha=1.0)
    cleaned = np.stack([denoiser.transform(feature_map) for feature_map in support], axis=0)
    cleaned_shape = cast("Tuple[int, int, int, int]", tuple(cleaned.shape))
    position_mean = cast("npt.NDArray[np.float32]", cleaned.mean(axis=0))
    global_mean = cast(
        "npt.NDArray[np.float32]",
        cleaned.reshape(-1, cleaned_shape[-1]).mean(axis=0),
    )

    assert cleaned.dtype == np.float32
    assert np.allclose(position_mean, global_mean.reshape(1, 1, 2), atol=1e-6)


def test_position_mean_artifact_denoiser_rejects_shape_mismatch() -> None:
    feature_map = np.ones((2, 2, 3), dtype=np.float32)
    denoiser = PositionMeanArtifactDenoiser.fit([feature_map], alpha=0.5)

    with pytest.raises(FeatureDenoisingError, match="expected"):
        denoiser.transform(np.ones((2, 3, 3), dtype=np.float32))
