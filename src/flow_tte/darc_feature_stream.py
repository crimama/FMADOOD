from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Tuple

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as functional
from PIL import Image
from typing_extensions import override

from flow_tte.darc_geometry import ImageSize
from flow_tte.darc_tiling import CropScores, NativeCrop, TilingSpec, blend_crop_scores
from flow_tte.darc_tiling import native_crop_grid as build_native_crop_grid

FloatArray = npt.NDArray[np.float32]
FeatureArray = npt.NDArray[np.float16]
RgbArray = npt.NDArray[np.uint8]

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class GridExtractor(Protocol):
    def __call__(self, pixels: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class DarcFeatureStreamError(ValueError):
    reason: str

    @override
    def __str__(self) -> str:
        return f"Invalid DARC feature-stream input: {self.reason}"


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class FeatureStreamConfig:
    device: str
    crop_spec: TilingSpec = field(default_factory=TilingSpec)
    low_input_size: int = 512
    high_input_size: int = 1024
    coarse_short_edge: int = 672
    patch_size: int = 16
    include_low: bool = True

    def __post_init__(self) -> None:
        sizes = (
            self.low_input_size,
            self.high_input_size,
            self.coarse_short_edge,
            self.patch_size,
        )
        if any(size <= 0 for size in sizes):
            raise DarcFeatureStreamError("input and patch sizes must be positive")
        if self.low_input_size % self.patch_size or self.high_input_size % self.patch_size:
            raise DarcFeatureStreamError("micro input sizes must be divisible by patch size")


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class ImageFeatures:
    native_size: ImageSize
    crops: Tuple[NativeCrop, ...]
    coarse: FeatureArray
    low: Tuple[FeatureArray, ...]
    high: Tuple[FeatureArray, ...]


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class DarcFeatureStream:
    micro_extractor: GridExtractor
    coarse_extractor: GridExtractor
    config: FeatureStreamConfig

    def extract(self, image: RgbArray) -> ImageFeatures:
        rgb = _rgb(image)
        native_size = ImageSize(height=int(rgb.shape[0]), width=int(rgb.shape[1]))
        crops = build_native_crop_grid(native_size, self.config.crop_spec)
        if any(
            crop.height != self.config.crop_spec.crop_size
            or crop.width != self.config.crop_spec.crop_size
            for crop in crops
        ):
            raise DarcFeatureStreamError("native image is smaller than the frozen crop size")
        coarse_pixels = _prepare_short_edge(rgb, self.config.coarse_short_edge, self.config)
        coarse = _feature_grid(self.coarse_extractor(coarse_pixels), "coarse")
        low = []
        high = []
        for crop in crops:
            crop_rgb = rgb[crop.y0 : crop.y0 + crop.height, crop.x0 : crop.x0 + crop.width]
            high_pixels = _prepare_square(crop_rgb, self.config.high_input_size, self.config)
            if self.config.include_low:
                low_pixels = _prepare_square(crop_rgb, self.config.low_input_size, self.config)
                low.append(_feature_grid(self.micro_extractor(low_pixels), "low"))
            high.append(_feature_grid(self.micro_extractor(high_pixels), "high"))
        return ImageFeatures(
            native_size=native_size,
            crops=crops,
            coarse=coarse,
            low=tuple(low),
            high=tuple(high),
        )


def resize_score_grid(scores: FloatArray, output_size: ImageSize) -> FloatArray:
    grid = np.asarray(scores, dtype=np.float32)
    if grid.ndim != 2 or grid.size == 0 or not np.all(np.isfinite(grid)):
        raise DarcFeatureStreamError("score grid must be a finite non-empty matrix")
    tensor = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0)
    resized = functional.interpolate(
        tensor,
        size=(output_size.height, output_size.width),
        mode="bilinear",
        align_corners=False,
    )
    return np.asarray(resized.squeeze(0).squeeze(0).numpy(), dtype=np.float32)


def stitch_score_grids(
    shape: ImageSize,
    crops: Tuple[NativeCrop, ...],
    score_grids: Tuple[FloatArray, ...],
) -> FloatArray:
    if len(crops) != len(score_grids) or not crops:
        raise DarcFeatureStreamError("crops and score grids must have equal non-zero length")
    expanded = tuple(
        CropScores(
            crop=crop,
            scores=resize_score_grid(
                grid,
                ImageSize(height=crop.height, width=crop.width),
            ),
        )
        for crop, grid in zip(crops, score_grids)
    )
    return blend_crop_scores(shape, expanded)


def _prepare_square(rgb: RgbArray, size: int, config: FeatureStreamConfig) -> torch.Tensor:
    resized = Image.fromarray(rgb).resize((size, size), resample=Image.Resampling.BICUBIC)
    return _normalized_pixels(np.asarray(resized), config)


def _prepare_short_edge(
    rgb: RgbArray,
    short_edge: int,
    config: FeatureStreamConfig,
) -> torch.Tensor:
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    scale = short_edge / min(height, width)
    realized_height = int(height * scale)
    realized_width = int(width * scale)
    resized = Image.fromarray(rgb).resize(
        (realized_width, realized_height),
        resample=Image.Resampling.BICUBIC,
    )
    aligned_height = realized_height - realized_height % config.patch_size
    aligned_width = realized_width - realized_width % config.patch_size
    aligned = np.asarray(resized)[:aligned_height, :aligned_width]
    return _normalized_pixels(aligned, config)


def _normalized_pixels(rgb: RgbArray, config: FeatureStreamConfig) -> torch.Tensor:
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).unsqueeze(0)
    mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    return ((tensor - mean) / std).to(config.device)


def _feature_grid(tensor: torch.Tensor, name: str) -> FeatureArray:
    if tensor.ndim != 3 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
        reason = f"{name} extractor must return an HxWxD grid"
        raise DarcFeatureStreamError(reason)
    values = tensor.detach().to(dtype=torch.float16, device="cpu").numpy()
    if not np.all(np.isfinite(values)):
        reason = f"{name} extractor returned non-finite features"
        raise DarcFeatureStreamError(reason)
    return np.asarray(values, dtype=np.float16)


def _rgb(image: RgbArray) -> RgbArray:
    rgb = np.asarray(image)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise DarcFeatureStreamError("image must be an HxWx3 uint8 RGB array")
    return rgb
