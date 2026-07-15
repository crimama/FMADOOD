"""Exact OpenCV morphology copied from official SuperADD post-processing."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


class SuperADDMorphologyError(ValueError):
    """Raised when the frozen morphology configuration is invalid."""


@dataclass(frozen=True)  # noqa: SLOTS_OK -- the project supports Python 3.8.
class MorphologyConfig:
    radius: int = 26
    angles: int = 16
    lower_factor: float = 0.8
    erosion: int = 1

    def __post_init__(self) -> None:
        if self.radius < 0 or self.angles < 1 or self.lower_factor <= 0 or self.erosion < 0:
            raise SuperADDMorphologyError("invalid radius, angles, lower factor, or erosion")


_DEFAULT_MORPHOLOGY = MorphologyConfig()


def postprocess_binary(
    score_map: np.ndarray,
    threshold: float,
    config: MorphologyConfig = _DEFAULT_MORPHOLOGY,
) -> np.ndarray:
    """Apply official multi-oriented close, fill, and ellipse erosion."""
    thresholded = np.where(score_map > threshold, 255, 0).astype(np.uint8)
    padding = config.radius + 1
    padded = np.pad(thresholded, padding, mode="constant", constant_values=0)
    versions = []
    thick = np.ones((2, 2), dtype=np.uint8)
    for index in range(config.angles):
        kernel = _line_kernel(config.radius, 180.0 * index / config.angles)
        dilated_kernel = cv2.morphologyEx(kernel, cv2.MORPH_DILATE, thick)
        dilated = cv2.morphologyEx(padded, cv2.MORPH_DILATE, dilated_kernel)
        versions.append(cv2.morphologyEx(dilated, cv2.MORPH_ERODE, kernel))
    closed = np.maximum.reduce((padded, *versions))[padding:-padding, padding:-padding]
    lower = np.asarray(score_map) > threshold * config.lower_factor
    closed = np.where(lower, closed, 0).astype(np.uint8)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(closed)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    erosion = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * config.erosion + 1, 2 * config.erosion + 1),
    )
    return cv2.morphologyEx(filled, cv2.MORPH_ERODE, erosion)


def _line_kernel(radius: int, angle_degrees: float) -> np.ndarray:
    size = 2 * radius + 1
    kernel = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    radians = np.deg2rad(angle_degrees)
    delta_x, delta_y = radius * np.cos(radians), radius * np.sin(radians)
    cv2.line(
        kernel,
        (round(center - delta_x), round(center - delta_y)),
        (round(center + delta_x), round(center + delta_y)),
        color=1,
        thickness=2,
    )
    return kernel
