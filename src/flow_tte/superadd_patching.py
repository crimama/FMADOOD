"""Official SuperADD preprocessing and token-ROI patch stitching."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import ceil, floor
from typing import Callable, Sequence, Tuple

import torch
from torch.nn import functional

Roi = Tuple[int, int]
GridExtractor = Callable[[torch.Tensor], Sequence[torch.Tensor]]


class SuperADDPatchError(ValueError):
    """Raised when patch geometry cannot satisfy the official ownership rule."""


@dataclass(frozen=True)  # noqa: SLOTS_OK -- the project supports Python 3.8.
class PatchConfig:
    patch_size: int = 640
    overlap: int = 128
    model_patch_size: int = 16

    def __post_init__(self) -> None:
        if self.overlap <= 0 or self.patch_size <= 2 * self.overlap:
            raise SuperADDPatchError("patch_size must exceed twice a positive overlap")
        if self.patch_size % self.model_patch_size:
            raise SuperADDPatchError("patch_size must align to the model patch size")
        if self.overlap % self.model_patch_size:
            raise SuperADDPatchError("overlap must align to the model patch size")


@dataclass(frozen=True)  # noqa: SLOTS_OK -- the project supports Python 3.8.
class PreprocessConfig:
    resize_factor: float = 640.0 / 1024.0
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        if self.resize_factor <= 0:
            raise SuperADDPatchError("resize_factor must be positive")


@dataclass(frozen=True)  # noqa: SLOTS_OK -- the project supports Python 3.8.
class AxisSplit:
    input_rois: Tuple[Roi, ...]
    prediction_rois: Tuple[Roi, ...]
    result_rois: Tuple[Roi, ...]


def preprocess_image(
    image: torch.Tensor,
    config: PreprocessConfig,
    brightness_factor: float = 1.0,
) -> torch.Tensor:
    """Apply tensor brightness, integer resize, and ImageNet normalization."""
    if image.ndim != 3 or int(image.shape[0]) != 3:
        raise SuperADDPatchError("image must have shape [3,H,W]")
    height = int(int(image.shape[-2]) * config.resize_factor)
    width = int(int(image.shape[-1]) * config.resize_factor)
    if height < 1 or width < 1:
        raise SuperADDPatchError("resize factor produced an empty image")
    batch = torch.clip(image.unsqueeze(0) * brightness_factor, 0.0, 1.0)
    resized = functional.interpolate(
        batch,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    mean = resized.new_tensor(config.mean).reshape(1, 3, 1, 1)
    std = resized.new_tensor(config.std).reshape(1, 3, 1, 1)
    return (resized - mean) / std


def axis_patch_split(dim_size: int, config: PatchConfig) -> AxisSplit:
    """Reproduce SuperADD's token-space input and prediction ownership ROIs."""
    if dim_size < config.patch_size:
        raise SuperADDPatchError("resized image dimension is smaller than patch_size")
    dim_tokens = dim_size // config.model_patch_size
    overlap_tokens = config.overlap // config.model_patch_size
    patch_tokens = config.patch_size // config.model_patch_size
    step = patch_tokens - 2 * overlap_tokens
    patch_count = ceil((dim_tokens - patch_tokens) / step) + 1
    span = dim_tokens - patch_tokens
    divisor = max(1, patch_count - 1)
    inputs_tokens = tuple(
        (index * span // divisor, index * span // divisor + patch_tokens)
        for index in range(patch_count)
    )
    predictions = tuple(
        (
            0 if index == 0 else ceil((inputs_tokens[index - 1][1] - inputs_tokens[index][0]) / 2),
            patch_tokens
            if index == patch_count - 1
            else patch_tokens - floor((inputs_tokens[index][1] - inputs_tokens[index + 1][0]) / 2),
        )
        for index in range(patch_count)
    )
    results = []
    cursor = 0
    for start, end in predictions:
        width = end - start
        results.append((cursor, cursor + width))
        cursor += width
    inputs = tuple(
        (start * config.model_patch_size, end * config.model_patch_size)
        for start, end in inputs_tokens
    )
    return AxisSplit(inputs, predictions, tuple(results))


def extract_patched_features(
    image_batch: torch.Tensor,
    extract_grids: GridExtractor,
    config: PatchConfig,
) -> Tuple[torch.Tensor, ...]:
    """Extract overlapping patches in one batch and stitch owned token ROIs."""
    if image_batch.ndim != 4:
        raise SuperADDPatchError("image_batch must have shape [B,C,H,W]")
    batch, channels, height, width = (int(value) for value in image_batch.shape)
    y_split = axis_patch_split(height, config)
    x_split = axis_patch_split(width, config)
    patches = [
        image_batch[:, :, y0:y1, x0:x1]
        for (y0, y1), (x0, x1) in product(y_split.input_rois, x_split.input_rois)
    ]
    patch_batch = torch.stack(patches, dim=1).reshape(
        -1,
        channels,
        config.patch_size,
        config.patch_size,
    )
    predictions = tuple(extract_grids(patch_batch))
    patch_count = len(patches)
    result_shape = (
        batch,
        height // config.model_patch_size,
        width // config.model_patch_size,
    )
    stitched = tuple(
        _stitch_layer(layer, patch_count, result_shape, y_split, x_split) for layer in predictions
    )
    return tuple(grid.detach().cpu() for grid in stitched)


def _stitch_layer(
    layer: torch.Tensor,
    patch_count: int,
    result_shape: Tuple[int, int, int],
    y_split: AxisSplit,
    x_split: AxisSplit,
) -> torch.Tensor:
    batch, height, width = result_shape
    if layer.ndim != 4 or int(layer.shape[0]) != batch * patch_count:
        raise SuperADDPatchError("extractor returned an incompatible patch grid")
    tokens_y, tokens_x, channels = (int(value) for value in layer.shape[1:])
    prediction = layer.reshape(batch, patch_count, tokens_y, tokens_x, channels)
    result = layer.new_zeros((batch, height, width, channels))
    pred_rois = product(y_split.prediction_rois, x_split.prediction_rois)
    result_rois = product(y_split.result_rois, x_split.result_rois)
    for index, ((py, px), (ry, rx)) in enumerate(zip(pred_rois, result_rois)):
        result[:, ry[0] : ry[1], rx[0] : rx[1]] = prediction[
            :,
            index,
            py[0] : py[1],
            px[0] : px[1],
        ]
    return result
