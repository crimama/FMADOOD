# /// script
# requires-python = ">=3.8"
# dependencies = []
# ///
# pyright: reportMissingImports=false
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Protocol, Tuple, cast

import numpy as np
import numpy.typing as npt
from typing_extensions import override


@dataclass(frozen=True)
class SuperADDPreprocessError(ValueError):
    field: str
    reason: str

    @override
    def __str__(self) -> str:
        return f"Invalid {self.field}: {self.reason}"


@dataclass(frozen=True)
class BrightnessRange:
    min_factor: float = 1.0
    max_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.min_factor <= 0.0 or self.max_factor <= 0.0:
            raise SuperADDPreprocessError("brightness_range", "factors must be positive")
        if self.min_factor > self.max_factor:
            raise SuperADDPreprocessError("brightness_range", "min must be <= max")

    @property
    def enabled(self) -> bool:
        return self.min_factor != 1.0 or self.max_factor != 1.0

    def factor_for(self, index: int, seed: int) -> float:
        if not self.enabled:
            return 1.0
        rng = np.random.default_rng(seed + index)
        return float(rng.uniform(self.min_factor, self.max_factor))


@dataclass(frozen=True)
class TilingConfig:
    patch_size: int = 0
    overlap: int = 0
    resize_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.patch_size < 0:
            raise SuperADDPreprocessError("patch_size", "must be non-negative")
        if self.overlap < 0:
            raise SuperADDPreprocessError("overlap", "must be non-negative")
        if self.patch_size > 0 and self.overlap >= self.patch_size:
            raise SuperADDPreprocessError("overlap", "must be smaller than patch_size")
        if self.resize_factor <= 0.0:
            raise SuperADDPreprocessError("resize_factor", "must be positive")

    @property
    def enabled(self) -> bool:
        return self.patch_size > 0


class Cv2ResizeModule(Protocol):
    INTER_CUBIC: int

    def resize(
        self,
        src: npt.NDArray[np.uint8],
        dsize: Tuple[int, int],
        interpolation: int,
    ) -> npt.NDArray[np.uint8]: ...


def parse_feature_layers(raw: str) -> Tuple[int, ...]:
    values = tuple(int(part) for part in raw.replace(",", " ").split() if part)
    if not values:
        raise SuperADDPreprocessError("feature_layers", "must contain at least one layer")
    if any(layer < 0 for layer in values):
        raise SuperADDPreprocessError("feature_layers", "layers must be non-negative")
    return values


def parse_brightness_range(raw: str) -> BrightnessRange:
    values = tuple(float(part) for part in raw.replace(",", " ").split() if part)
    if len(values) != 2:
        raise SuperADDPreprocessError("brightness_range", "expected two factors")
    return BrightnessRange(min_factor=values[0], max_factor=values[1])


def apply_brightness(
    image: npt.NDArray[np.uint8],
    factor: float,
) -> npt.NDArray[np.uint8]:
    if factor == 1.0:
        return image
    adjusted = image.astype(np.float32, copy=False) * factor
    return cast(
        "npt.NDArray[np.uint8]",
        np.clip(adjusted, 0.0, 255.0).astype(np.uint8, copy=False),
    )


def resize_rgb(
    image: npt.NDArray[np.uint8],
    resize_factor: float,
) -> npt.NDArray[np.uint8]:
    if resize_factor == 1.0:
        return image
    cv2_module = load_cv2_resize_module()
    height = cast("int", image.shape[0])
    width = cast("int", image.shape[1])
    resized_size = (
        max(1, round(width * resize_factor)),
        max(1, round(height * resize_factor)),
    )
    return cv2_module.resize(image, resized_size, interpolation=cv2_module.INTER_CUBIC)


def load_cv2_resize_module() -> Cv2ResizeModule:
    try:
        cv2_module = import_module("cv2")
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is required for SuperADD-style resized tiling") from exc
    return cast("Cv2ResizeModule", cast("object", cv2_module))


def tile_starts(length: int, patch_size: int, overlap: int) -> Tuple[int, ...]:
    if length <= 0:
        raise SuperADDPreprocessError("length", "must be positive")
    if patch_size <= 0:
        raise SuperADDPreprocessError("patch_size", "must be positive")
    if overlap < 0 or overlap >= patch_size:
        raise SuperADDPreprocessError("overlap", "must be in [0, patch_size)")
    if length <= patch_size:
        return (0,)
    stride = patch_size - overlap
    starts = list(range(0, max(1, length - patch_size + 1), stride))
    final_start = length - patch_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(starts)
