from __future__ import annotations

# pyright: reportMissingImports=false
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import numpy.typing as npt
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_ROOT = _REPO_ROOT / "scripts"
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

from flow_tte_mvtec_ad2_core import (  # noqa: E402
    FeatureExtractionConfig,
    extract_layer_feature_maps_from_rgb,
)
from flow_tte_raw_nn import (  # noqa: E402
    ForegroundSplitConfig,
    fit_foreground_split_raw_nn,
    fit_raw_nn,
    score_foreground_split_raw_nn,
    score_raw_nn,
)

from flow_tte.config import ScoreConfig  # noqa: E402


def test_raw_nn_scores_exact_support_patch_lower_than_far_patch() -> None:
    support = np.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=np.float32,
    )
    query = np.asarray([[[0.0, 0.0], [4.0, 0.0]]], dtype=np.float32)
    state = fit_raw_nn(
        support_features=support,
        memory_contexts=None,
        score_config=ScoreConfig(query_chunk_size=2),
        device="cpu",
    )

    result = score_raw_nn(state, query_features=query, query_contexts=None)

    assert result.patch_scores.shape == (1, 2)
    assert result.patch_scores[0, 0] < result.patch_scores[0, 1]
    assert result.memory_size_before == 3
    assert result.memory_size_after == 3


def test_foreground_split_raw_nn_suppresses_background_like_patch() -> None:
    support_fields = (
        np.asarray(
            [
                [[0.1, 0.0], [0.2, 0.0]],
                [[3.0, 0.0], [4.0, 0.0]],
            ],
            dtype=np.float32,
        ),
    )
    state = fit_foreground_split_raw_nn(
        support_feature_maps=support_fields,
        support_contexts=None,
        score_config=ScoreConfig(query_chunk_size=2),
        split_config=ForegroundSplitConfig(
            foreground_quantile=0.5,
            background_multiplier=0.25,
        ),
        device="cpu",
    )

    result = score_foreground_split_raw_nn(
        state,
        query_features=support_fields[0],
        query_contexts=None,
    )
    foreground_only = score_raw_nn(
        state.foreground,
        query_features=support_fields[0],
        query_contexts=None,
    )

    assert np.isclose(
        result.patch_scores[0, 0],
        foreground_only.patch_scores[0, 0] * 0.25,
    )
    assert np.isclose(result.patch_scores[1, 0], foreground_only.patch_scores[1, 0])
    assert result.memory_size_before == 4


def test_tiled_layer_wise_extraction_preserves_each_layer() -> None:
    class FakeBackbone:
        def set_resolution(self, smaller_edge_size: int) -> None:
            _ = smaller_edge_size

        def prepare_image(
            self,
            img: npt.NDArray[np.uint8],
        ) -> Tuple[torch.Tensor, Tuple[int, int]]:
            height = int(img.shape[0])
            width = int(img.shape[1])
            return torch.zeros((3, height, width), dtype=torch.float32), (
                max(1, height // 2),
                max(1, width // 2),
            )

        def extract_features(self, image_tensor: torch.Tensor) -> list[npt.NDArray[np.float32]]:
            height = int(image_tensor.shape[1])
            width = int(image_tensor.shape[2])
            grid_h = max(1, height // 2)
            grid_w = max(1, width // 2)
            base = np.ones((grid_h * grid_w, 2), dtype=np.float32)
            return [
                base * np.asarray([float(height), float(width)], dtype=np.float32),
                base * np.asarray([float(height + width), float(height - width)], dtype=np.float32),
            ]

        def extract_cls_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
            _ = image_tensor
            return torch.tensor([1.0, 0.0], dtype=torch.float32)

        def extract_context_features(
            self,
            image_tensor: torch.Tensor,
            context_source: str,
        ) -> torch.Tensor:
            _ = image_tensor, context_source
            return torch.tensor([1.0, 0.0], dtype=torch.float32)

    layered = extract_layer_feature_maps_from_rgb(
        FakeBackbone(),
        np.zeros((6, 6, 3), dtype=np.uint8),
        FeatureExtractionConfig(
            feature_fusion="layer_norm_mean",
            context_source="none",
            tiling=__import__("flow_tte_superadd_preprocess").TilingConfig(
                patch_size=4,
                overlap=2,
            ),
        ),
    )

    assert len(layered.layers) == 2
    assert layered.layers[0].values.shape == (3, 3, 2)
    assert layered.layers[1].values.shape == (3, 3, 2)
