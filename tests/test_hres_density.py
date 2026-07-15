from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from flow_tte.config import FlowConfig
from flow_tte.darc_ad2_density_runtime import DensityRuntimeConfig, DensityRuntimeError
from flow_tte.darc_feature_stream import ImageFeatures
from flow_tte.darc_geometry import ImageSize
from flow_tte.darc_knn import ChunkedKnnConfig
from flow_tte.darc_tiling import NativeCrop
from flow_tte.hres_density import (
    HresDensityConfig,
    HresDensityError,
    g0_native_map,
    high_token_matrix,
    train_density_head,
)

_DIM = 8
_GRID = 4
_NATIVE = 16


def _single_crop_features(grid: np.ndarray) -> ImageFeatures:
    height, width = int(grid.shape[0]), int(grid.shape[1])
    return ImageFeatures(
        native_size=ImageSize(height=_NATIVE, width=_NATIVE),
        crops=(NativeCrop(y0=0, x0=0, height=_NATIVE, width=_NATIVE),),
        coarse=np.zeros((height, width, _DIM), dtype=np.float16),
        low=(),
        high=(np.asarray(grid, dtype=np.float16),),
    )


def _normal_grid(rng: np.random.Generator, mean: np.ndarray) -> np.ndarray:
    tokens = mean[None, None, :] + 0.15 * rng.standard_normal((_GRID, _GRID, _DIM))
    return np.asarray(tokens, dtype=np.float32)


def _offmanifold_grid(mean: np.ndarray) -> np.ndarray:
    tokens = np.broadcast_to(mean + 3.0, (_GRID, _GRID, _DIM))
    return np.asarray(tokens, dtype=np.float32)


def _train_config() -> HresDensityConfig:
    flow = FlowConfig(n_coupling_layers=2, n_epochs=3, batch_size=64, seed=0)
    return HresDensityConfig(flow=flow, train_sample_cap=4096)


def _trained_head_and_mean() -> Tuple[object, np.ndarray]:
    rng = np.random.default_rng(0)
    mean = rng.standard_normal(_DIM).astype(np.float32)
    memory = tuple(_single_crop_features(_normal_grid(rng, mean)) for _ in range(4))
    heldout = tuple(_single_crop_features(_normal_grid(rng, mean)) for _ in range(2))
    head = train_density_head(memory, heldout, _train_config(), device="cpu")
    return head, mean


def test_high_token_matrix_concatenates_all_crops() -> None:
    grid = np.zeros((_GRID, _GRID, _DIM), dtype=np.float16)
    features = ImageFeatures(
        native_size=ImageSize(height=_NATIVE, width=_NATIVE),
        crops=(
            NativeCrop(0, 0, _NATIVE, _NATIVE),
            NativeCrop(0, 0, _NATIVE, _NATIVE),
        ),
        coarse=grid,
        low=(),
        high=(grid, grid),
    )
    matrix = high_token_matrix(features)
    assert matrix.shape == (2 * _GRID * _GRID, _DIM)
    assert matrix.dtype == np.float32


def test_density_head_scores_offmanifold_above_normal() -> None:
    head, mean = _trained_head_and_mean()
    rng = np.random.default_rng(99)
    normal_nll = head.token_nll_grids(_single_crop_features(_normal_grid(rng, mean)))[0]
    offmanifold_nll = head.token_nll_grids(_single_crop_features(_offmanifold_grid(mean)))[0]
    assert float(offmanifold_nll.mean()) > float(normal_nll.mean())


def test_density_native_map_has_native_shape_and_is_finite() -> None:
    head, mean = _trained_head_and_mean()
    rng = np.random.default_rng(7)
    native = head.density_native_map(_single_crop_features(_normal_grid(rng, mean)))
    assert native.shape == (_NATIVE, _NATIVE)
    assert np.all(np.isfinite(native))


def test_calibrated_map_is_nonnegative_and_ranks_offmanifold_higher() -> None:
    head, mean = _trained_head_and_mean()
    rng = np.random.default_rng(11)
    normal = head.calibrated_native_map(_single_crop_features(_normal_grid(rng, mean)))
    offmanifold = head.calibrated_native_map(_single_crop_features(_offmanifold_grid(mean)))
    assert np.all(normal >= 0.0)
    assert np.all(offmanifold >= 0.0)
    assert float(offmanifold.mean()) > float(normal.mean())


def test_g0_native_map_is_near_zero_on_identical_memory() -> None:
    rng = np.random.default_rng(3)
    mean = rng.standard_normal(_DIM).astype(np.float32)
    query = _single_crop_features(_normal_grid(rng, mean))
    residual = g0_native_map(query, (query,), ChunkedKnnConfig(device="cpu"))
    assert residual.shape == (_NATIVE, _NATIVE)
    assert float(residual.max()) < 1e-3


def test_g0_native_map_flags_offmanifold_query() -> None:
    rng = np.random.default_rng(5)
    mean = rng.standard_normal(_DIM).astype(np.float32)
    memory = _single_crop_features(_normal_grid(rng, mean))
    offmanifold = _single_crop_features(_offmanifold_grid(mean))
    normal_residual = g0_native_map(memory, (memory,), ChunkedKnnConfig(device="cpu"))
    off_residual = g0_native_map(offmanifold, (memory,), ChunkedKnnConfig(device="cpu"))
    assert float(off_residual.mean()) > float(normal_residual.mean())


def test_high_token_matrix_rejects_empty_high() -> None:
    features = ImageFeatures(
        native_size=ImageSize(height=_NATIVE, width=_NATIVE),
        crops=(),
        coarse=np.zeros((1, 1, _DIM), dtype=np.float16),
        low=(),
        high=(),
    )
    with pytest.raises(HresDensityError):
        _ = high_token_matrix(features)


def test_density_config_rejects_bad_quantile() -> None:
    with pytest.raises(HresDensityError):
        _ = HresDensityConfig(density_quantile=1.0)


def test_runtime_config_rejects_bad_folds_and_shard() -> None:
    base = {
        "data_root": Path("/data"),
        "output_root": Path("/out"),
        "object_name": "can",
        "device": "cpu",
        "seed": 0,
    }
    with pytest.raises(DensityRuntimeError):
        _ = DensityRuntimeConfig(fold_indices=(0, 0), **base)
    with pytest.raises(DensityRuntimeError):
        _ = DensityRuntimeConfig(fold_indices=(0,), shard_index=2, shard_count=2, **base)
