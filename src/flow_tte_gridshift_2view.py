"""Pure geometry and score-combination helpers for two-view grid shifting."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as torch_functional


FloatArray = npt.NDArray[np.float32]


def _shape_yx(shape: Sequence[int], name: str) -> tuple[int, int]:
    if len(shape) < 2:
        raise ValueError(f"{name} must contain height and width")
    height, width = int(shape[-2]), int(shape[-1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} dimensions must be positive: {(height, width)}")
    return height, width


def _offset_yx(offset_yx: Sequence[int]) -> tuple[int, int]:
    if len(offset_yx) != 2:
        raise ValueError("offset_yx must contain exactly (y, x)")
    offset_y, offset_x = int(offset_yx[0]), int(offset_yx[1])
    if offset_y < 0 or offset_x < 0:
        raise ValueError(f"offsets must be non-negative: {(offset_y, offset_x)}")
    return offset_y, offset_x


def shift_resized_tensor_right_down(
    image: torch.Tensor,
    offset_yx: Sequence[int] = (8, 8),
) -> torch.Tensor:
    """Shift a CHW or NCHW resized image right/down with edge replication.

    Padding is added at the top and left and the bottom/right are cropped so
    the output shape is unchanged. The operation preserves dtype and device.
    """
    if image.ndim not in (3, 4):
        raise ValueError(f"image must be CHW or NCHW, got shape {tuple(image.shape)}")
    offset_y, offset_x = _offset_yx(offset_yx)
    if offset_y == 0 and offset_x == 0:
        return image.clone()
    height, width = _shape_yx(image.shape, "image shape")
    padded = torch_functional.pad(
        image,
        (offset_x, 0, offset_y, 0),
        mode="replicate",
    )
    return padded[..., :height, :width]


def native_offset_yx(
    native_shape: Sequence[int],
    resized_shape: Sequence[int],
    resized_offset_yx: Sequence[int] = (8, 8),
) -> tuple[float, float]:
    """Convert a resized-image offset to native pixels using realized scales."""
    native_height, native_width = _shape_yx(native_shape, "native_shape")
    resized_height, resized_width = _shape_yx(resized_shape, "resized_shape")
    offset_y, offset_x = _offset_yx(resized_offset_yx)
    scale_y = resized_height / native_height
    scale_x = resized_width / native_width
    return offset_y / scale_y, offset_x / scale_x


def align_shifted_native_map(
    shifted_score: npt.ArrayLike,
    resized_shape: Sequence[int],
    resized_offset_yx: Sequence[int] = (8, 8),
) -> FloatArray:
    """Translate a shifted-view native score map back into view-0 coordinates."""
    score = np.asarray(shifted_score, dtype=np.float32)
    if score.ndim != 2:
        raise ValueError(f"shifted_score must be 2D, got shape {score.shape}")
    native_height, native_width = _shape_yx(score.shape, "shifted_score shape")
    native_y, native_x = native_offset_yx(
        score.shape,
        resized_shape,
        resized_offset_yx,
    )
    translation = np.array(
        [[1.0, 0.0, -native_x], [0.0, 1.0, -native_y]],
        dtype=np.float32,
    )
    aligned = cv2.warpAffine(
        score,
        translation,
        (native_width, native_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return np.asarray(aligned, dtype=np.float32)


def _stack_maps(maps: Sequence[npt.ArrayLike]) -> FloatArray:
    if not maps:
        raise ValueError("at least one aligned map is required")
    arrays = [np.asarray(score, dtype=np.float32) for score in maps]
    reference_shape = arrays[0].shape
    if any(score.shape != reference_shape for score in arrays):
        raise ValueError("all aligned maps must have the same shape")
    return np.stack(arrays, axis=0).astype(np.float32, copy=False)


def mean_aligned_maps(maps: Sequence[npt.ArrayLike]) -> FloatArray:
    """Compute the elementwise arithmetic mean of aligned score maps."""
    return np.mean(_stack_maps(maps), axis=0, dtype=np.float32)


def max_aligned_maps(maps: Sequence[npt.ArrayLike]) -> FloatArray:
    """Compute the elementwise maximum of aligned score maps."""
    return np.max(_stack_maps(maps), axis=0)


def combine_aligned_maps(
    maps: Sequence[npt.ArrayLike],
    method: str,
) -> FloatArray:
    """Combine aligned maps with one of the preregistered smoke-test rules."""
    if method == "mean":
        return mean_aligned_maps(maps)
    if method == "max":
        return max_aligned_maps(maps)
    raise ValueError(f"unknown aligned-map combiner: {method}")
