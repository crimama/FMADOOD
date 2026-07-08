# /// script
# requires-python = ">=3.8"
# dependencies = ["numpy", "torch", "typing-extensions"]
# ///
# pyright: reportMissingImports=false
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt
import torch
from typing_extensions import override

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from flow_tte.config import ScoreConfig  # noqa: E402
from flow_tte.memory import TorchMemoryBank  # noqa: E402
from flow_tte.scoring import ScoreCalibration  # noqa: E402
from flow_tte.tensors import (  # noqa: E402
    FeatureArray,
    as_2d_float_tensor,
    as_patch_batch,
    resolve_device,
    to_numpy,
)

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class RawNNError(ValueError):
    field: str
    reason: str

    @override
    def __str__(self) -> str:
        return f"Invalid {self.field}: {self.reason}"


@dataclass(frozen=True)
class RawNNResult:
    patch_scores: FloatArray
    distances: FloatArray
    selected_count: int
    memory_size_before: int
    memory_size_after: int


@dataclass(frozen=True)
class RawNNState:
    bank: TorchMemoryBank
    calibration: ScoreCalibration
    score_config: ScoreConfig
    device: str

    @property
    def memory_size(self) -> int:
        return self.bank.size()


@dataclass(frozen=True)
class ForegroundSplitConfig:
    foreground_quantile: float = 0.5
    background_multiplier: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.foreground_quantile <= 1.0:
            raise RawNNError("foreground_quantile", "must be in [0, 1]")
        if not 0.0 <= self.background_multiplier <= 1.0:
            raise RawNNError("background_multiplier", "must be in [0, 1]")


@dataclass(frozen=True)
class ForegroundSplitRawNNState:
    foreground: RawNNState
    background: Optional[RawNNState]
    split_config: ForegroundSplitConfig

    @property
    def memory_size(self) -> int:
        if self.background is None:
            return self.foreground.memory_size
        return self.foreground.memory_size + self.background.memory_size


def fit_raw_nn(
    support_features: FeatureArray,
    memory_contexts: Optional[FeatureArray],
    score_config: ScoreConfig,
    device: str,
) -> RawNNState:
    device_object = resolve_device(device)
    support = as_2d_float_tensor(support_features, device_object)
    contexts = _optional_contexts(
        memory_contexts,
        device_object,
        n_rows=int(support.shape[0]),
        name="memory_contexts",
    )
    bank = TorchMemoryBank()
    bank.fit(support, contexts=contexts)
    calibration = ScoreCalibration.fit(support, score_config, contexts)
    return RawNNState(
        bank=bank,
        calibration=calibration,
        score_config=score_config,
        device=device,
    )


def score_raw_nn(
    state: RawNNState,
    query_features: FeatureArray,
    query_contexts: Optional[FeatureArray],
) -> RawNNResult:
    device_object = resolve_device(state.device)
    batch = as_patch_batch(query_features, device_object)
    contexts = _optional_contexts(
        query_contexts,
        device_object,
        n_rows=int(batch.flat_features.shape[0]),
        name="query_contexts",
    )
    context_weight = (
        state.score_config.context_weight
        if state.score_config.context_mode == "soft_penalty"
        else 0.0
    )
    context_top_m = (
        state.score_config.context_top_m
        if state.score_config.context_mode == "top_m"
        else None
    )
    query = state.bank.query(
        batch.flat_features,
        k=1,
        chunk_size=state.score_config.query_chunk_size,
        squared=state.score_config.use_squared_distance,
        query_contexts=contexts,
        context_weight=context_weight,
        context_top_m=context_top_m,
    )
    distances = query.distances[:, 0]
    patch_scores = state.score_config.distance_weight * state.calibration.normalize_distance(
        distances,
    )
    return RawNNResult(
        patch_scores=to_numpy(batch.restore(patch_scores)),
        distances=to_numpy(batch.restore(distances)),
        selected_count=0,
        memory_size_before=state.memory_size,
        memory_size_after=state.memory_size,
    )


def fit_foreground_split_raw_nn(
    support_feature_maps: Sequence[FloatArray],
    support_contexts: Optional[FeatureArray],
    score_config: ScoreConfig,
    split_config: ForegroundSplitConfig,
    device: str,
) -> ForegroundSplitRawNNState:
    if not support_feature_maps:
        raise RawNNError("support_feature_maps", "must not be empty")
    features = _flatten_feature_maps(support_feature_maps)
    mask = _foreground_mask(support_feature_maps, split_config.foreground_quantile)
    contexts = None
    if support_contexts is not None:
        contexts = np.asarray(support_contexts, dtype=np.float32).reshape(features.shape[0], -1)
    foreground_features = features[mask]
    foreground_contexts = None if contexts is None else contexts[mask]
    background_features = features[~mask]
    background_contexts = None if contexts is None else contexts[~mask]
    foreground_state = fit_raw_nn(
        support_features=foreground_features,
        memory_contexts=foreground_contexts,
        score_config=score_config,
        device=device,
    )
    background_state = None
    if background_features.shape[0] > 0:
        background_state = fit_raw_nn(
            support_features=background_features,
            memory_contexts=background_contexts,
            score_config=score_config,
            device=device,
        )
    return ForegroundSplitRawNNState(
        foreground=foreground_state,
        background=background_state,
        split_config=split_config,
    )


def score_foreground_split_raw_nn(
    state: ForegroundSplitRawNNState,
    query_features: FeatureArray,
    query_contexts: Optional[FeatureArray],
) -> RawNNResult:
    foreground = score_raw_nn(state.foreground, query_features, query_contexts)
    if state.background is None:
        return foreground
    background = score_raw_nn(state.background, query_features, query_contexts)
    background_like = background.distances < foreground.distances
    patch_scores = np.where(
        background_like,
        foreground.patch_scores * np.float32(state.split_config.background_multiplier),
        foreground.patch_scores,
    ).astype(np.float32, copy=False)
    return RawNNResult(
        patch_scores=patch_scores,
        distances=foreground.distances,
        selected_count=0,
        memory_size_before=state.memory_size,
        memory_size_after=state.memory_size,
    )


def _optional_contexts(
    contexts: Optional[FeatureArray],
    device: torch.device,
    n_rows: int,
    name: str,
) -> Optional[torch.Tensor]:
    if contexts is None:
        return None
    tensor = as_2d_float_tensor(contexts, device)
    if tensor.shape[0] != n_rows:
        raise RawNNError(name, "row count must match features")
    return tensor


def _flatten_feature_maps(feature_maps: Sequence[FloatArray]) -> FloatArray:
    return np.concatenate(
        [
            np.asarray(feature_map, dtype=np.float32).reshape(-1, feature_map.shape[-1])
            for feature_map in feature_maps
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def _foreground_mask(
    feature_maps: Sequence[FloatArray],
    foreground_quantile: float,
) -> BoolArray:
    energies = np.concatenate(
        [
            np.linalg.norm(np.asarray(feature_map, dtype=np.float32), axis=-1).reshape(-1)
            for feature_map in feature_maps
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    threshold = np.float32(np.quantile(energies, foreground_quantile))
    mask = energies >= threshold
    if bool(np.any(mask)):
        return mask.astype(np.bool_, copy=False)
    return np.ones(energies.shape, dtype=np.bool_)
