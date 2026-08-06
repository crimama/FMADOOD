"""Pure feature-context construction helpers for FlowTTE experiment runners."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

_FEATURE_CONTEXT_SOURCES = frozenset(
    {
        "feature_avg3",
        "feature_avg3_ch16",
        "feature_avg3_residual",
        "image_feature_mean",
        "image_feature_mean_ch16",
    },
)
_BACKBONE_CONTEXT_SOURCES = frozenset({"cls", "register", "cls_register"})
_CONTEXT_SOURCES = _FEATURE_CONTEXT_SOURCES | _BACKBONE_CONTEXT_SOURCES | {"none"}
_XY_SUFFIX = "_xy"


def validate_transformer_context_tokens(tokens: np.ndarray) -> np.ndarray:
    context_tokens = np.asarray(tokens, dtype=np.float32)
    if context_tokens.ndim != 2:
        raise RuntimeError("Transformer context tokens must be shaped TxC")
    return context_tokens.astype(np.float32, copy=False)


def build_feature_contexts(values: np.ndarray, context_source: str) -> Optional[np.ndarray]:
    source, include_xy = _split_xy_suffix(context_source)
    if source not in _FEATURE_CONTEXT_SOURCES:
        return None

    values = values.astype(np.float32, copy=False)
    if source == "feature_avg3":
        parts = [average_pool_3x3(values)]
    elif source == "feature_avg3_ch16":
        parts = [channel_group_mean(average_pool_3x3(values), n_groups=16)]
    elif source == "feature_avg3_residual":
        local_average = average_pool_3x3(values)
        parts = [local_average, values - local_average]
    else:
        context_values = (
            channel_group_mean(values, n_groups=16)
            if source == "image_feature_mean_ch16"
            else values
        )
        image_mean = context_values.mean(axis=(0, 1), dtype=np.float32)
        parts = [
            np.broadcast_to(
                image_mean.reshape(1, 1, -1),
                (values.shape[0], values.shape[1], image_mean.shape[0]),
            ).copy(),
        ]

    if include_xy:
        parts.append(patch_xy_contexts(int(values.shape[0]), int(values.shape[1])))
    return np.concatenate(parts, axis=-1).astype(np.float32, copy=False)


def channel_group_mean(values: np.ndarray, n_groups: int) -> np.ndarray:
    group_count = min(max(1, n_groups), int(values.shape[-1]))
    chunks = np.array_split(values, group_count, axis=-1)
    return np.concatenate(
        [chunk.mean(axis=-1, dtype=np.float32, keepdims=True) for chunk in chunks],
        axis=-1,
    ).astype(np.float32, copy=False)


def average_pool_3x3(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values, ((1, 1), (1, 1), (0, 0)), mode="edge")
    pooled = np.zeros_like(values, dtype=np.float32)
    height, width = values.shape[:2]
    for y_offset in range(3):
        for x_offset in range(3):
            pooled += padded[y_offset : y_offset + height, x_offset : x_offset + width]
    pooled /= 9.0
    return pooled.astype(np.float32, copy=False)


def split_context_source(context_source: str) -> Tuple[str, bool]:
    source = context_source.lower()
    if source == "xy":
        return "none", True
    if source.endswith(_XY_SUFFIX):
        base_source = source[: -len(_XY_SUFFIX)]
        if base_source in _FEATURE_CONTEXT_SOURCES | _BACKBONE_CONTEXT_SOURCES:
            return base_source, True
    if source in _CONTEXT_SOURCES:
        return source, False
    message = f"Unsupported context source: {context_source}"
    raise ValueError(message)


def patch_xy_contexts(height: int, width: int) -> np.ndarray:
    y_values = np.linspace(-1.0, 1.0, num=max(1, height), dtype=np.float32)
    x_values = np.linspace(-1.0, 1.0, num=max(1, width), dtype=np.float32)
    y_grid, x_grid = np.meshgrid(y_values, x_values, indexing="ij")
    return np.stack((x_grid, y_grid), axis=-1).astype(np.float32, copy=False)


def _split_xy_suffix(context_source: str) -> Tuple[str, bool]:
    source = context_source.lower()
    if source == "xy":
        return "none", True
    if source.endswith(_XY_SUFFIX):
        return source[: -len(_XY_SUFFIX)], True
    return source, False
