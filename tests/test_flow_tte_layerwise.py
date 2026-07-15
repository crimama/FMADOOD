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
    build_feature_contexts,
    extract_layer_feature_maps_from_rgb,
    feature_energy_split_mask,
    patch_xy_contexts,
)
from flow_tte_support import normalize_layer_features  # noqa: E402


def test_layer_wise_extraction_keeps_layers_separate_with_context() -> None:
    class FakeBackbone:
        def set_resolution(self, smaller_edge_size: int) -> None:
            _ = smaller_edge_size

        def prepare_image(
            self,
            img: npt.NDArray[np.uint8],
        ) -> Tuple[torch.Tensor, Tuple[int, int]]:
            _ = img
            return torch.zeros((3, 16, 32), dtype=torch.float32), (1, 2)

        def extract_features(self, image_tensor: torch.Tensor) -> list[npt.NDArray[np.float32]]:
            _ = image_tensor
            return [
                np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32),
                np.array([[5.0, 0.0], [6.0, 8.0]], dtype=np.float32),
            ]

        def extract_cls_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
            _ = image_tensor
            return torch.tensor([0.25, 0.75], dtype=torch.float32)

        def extract_context_features(
            self,
            image_tensor: torch.Tensor,
            context_source: str,
        ) -> torch.Tensor:
            _ = image_tensor, context_source
            return torch.tensor([0.25, 0.75], dtype=torch.float32)

    layered = extract_layer_feature_maps_from_rgb(
        FakeBackbone(),
        np.zeros((20, 40, 3), dtype=np.uint8),
        FeatureExtractionConfig(feature_fusion="layer_norm_mean", context_source="cls"),
    )

    assert len(layered.layers) == 2
    assert layered.layers[0].values.shape == (1, 2, 2)
    assert np.allclose(
        layered.layers[0].values.reshape(2, 2),
        normalize_layer_features(
            np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32),
        ),
    )
    layer_contexts = layered.layers[0].contexts
    assert layer_contexts is not None
    assert layer_contexts.shape == (1, 2, 2)
    assert np.allclose(layer_contexts[0, 1], np.array([0.25, 0.75], dtype=np.float32))


def test_xy_contexts_are_patch_grid_coordinates() -> None:
    contexts = patch_xy_contexts(2, 3)

    assert contexts.shape == (2, 3, 2)
    assert np.allclose(contexts[0, 0], np.array([-1.0, -1.0], dtype=np.float32))
    assert np.allclose(contexts[-1, -1], np.array([1.0, 1.0], dtype=np.float32))


def test_feature_avg3_context_uses_local_neighborhood() -> None:
    values = np.arange(9, dtype=np.float32).reshape(3, 3, 1)

    contexts = build_feature_contexts(values, "feature_avg3")

    assert contexts is not None
    assert contexts.shape == (3, 3, 1)
    assert np.isclose(contexts[1, 1, 0], 4.0)
    assert np.isclose(contexts[0, 0, 0], (0.0 + 0.0 + 1.0 + 0.0 + 0.0 + 1.0 + 3.0 + 3.0 + 4.0) / 9.0)


def test_feature_avg3_ch16_context_compresses_channel_groups() -> None:
    values = np.arange(2 * 2 * 32, dtype=np.float32).reshape(2, 2, 32)

    contexts = build_feature_contexts(values, "feature_avg3_ch16")

    assert contexts is not None
    assert contexts.shape == (2, 2, 16)


def test_image_feature_mean_context_is_image_level_subbank_key() -> None:
    values = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=np.float32,
    )

    contexts = build_feature_contexts(values, "image_feature_mean")

    assert contexts is not None
    assert contexts.shape == (2, 2, 2)
    assert np.allclose(contexts[0, 0], np.array([4.0, 5.0], dtype=np.float32))
    assert np.allclose(contexts[-1, -1], np.array([4.0, 5.0], dtype=np.float32))


def test_feature_energy_split_mask_separates_high_energy_patches() -> None:
    feature_maps = (
        np.asarray([[[0.0], [1.0]], [[3.0], [4.0]]], dtype=np.float32),
    )

    mask = feature_energy_split_mask(feature_maps, foreground_quantile=0.5)

    assert mask.tolist() == [False, False, True, True]
