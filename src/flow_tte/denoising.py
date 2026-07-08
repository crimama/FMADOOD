from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, cast

import numpy as np
import numpy.typing as npt
from typing_extensions import override


@dataclass(frozen=True)
class FeatureDenoisingError(ValueError):
    field: str
    reason: str

    @override
    def __str__(self) -> str:
        return f"Invalid {self.field}: {self.reason}"


@dataclass(frozen=True)
class PositionMeanArtifactDenoiser:
    """Remove a support-fitted position-dependent feature bias field."""

    artifact: npt.NDArray[np.float32]
    alpha: float

    @classmethod
    def fit(
        cls,
        feature_maps: Sequence[npt.NDArray[np.float32]],
        alpha: float,
    ) -> "PositionMeanArtifactDenoiser":
        if alpha < 0.0:
            raise FeatureDenoisingError("alpha", "must be non-negative")
        shapes = _validate_feature_maps(feature_maps)
        maps = tuple(
            cast("npt.NDArray[np.float32]", feature_map.astype(np.float32, copy=False))
            for feature_map in feature_maps
        )
        stack = cast(
            "npt.NDArray[np.float32]",
            np.stack(maps, axis=0).astype(np.float32, copy=False),
        )
        stack_shape = cast("Tuple[int, int, int, int]", tuple(stack.shape))
        feature_dim = stack_shape[-1]
        position_mean = cast(
            "npt.NDArray[np.float64]",
            np.mean(stack, axis=0, dtype=np.float64),
        )
        global_mean = cast(
            "npt.NDArray[np.float64]",
            np.mean(stack.reshape(-1, feature_dim), axis=0, dtype=np.float64),
        )
        artifact = cast(
            "npt.NDArray[np.float32]",
            (position_mean - global_mean.reshape(1, 1, feature_dim)).astype(
                np.float32,
                copy=False,
            ),
        )
        if artifact.shape != shapes[0]:
            raise FeatureDenoisingError("artifact", "internal artifact shape mismatch")
        return cls(artifact=artifact, alpha=float(alpha))

    def transform(
        self,
        feature_map: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        if tuple(feature_map.shape) != tuple(self.artifact.shape):
            raise FeatureDenoisingError(
                "feature_map",
                f"expected {self.artifact.shape}, got {feature_map.shape}",
            )
        denoised = feature_map.astype(np.float32, copy=False) - (self.alpha * self.artifact)
        return cast("npt.NDArray[np.float32]", denoised.astype(np.float32, copy=False))


def fit_feature_denoiser(
    mode: str,
    feature_maps: Sequence[npt.NDArray[np.float32]],
    alpha: float,
) -> Optional[PositionMeanArtifactDenoiser]:
    if mode == "none":
        return None
    if mode == "position_mean":
        return PositionMeanArtifactDenoiser.fit(feature_maps, alpha=alpha)
    raise FeatureDenoisingError("mode", "must be 'none' or 'position_mean'")


def _validate_feature_maps(
    feature_maps: Sequence[npt.NDArray[np.float32]],
) -> Tuple[Tuple[int, int, int], ...]:
    if not feature_maps:
        raise FeatureDenoisingError("feature_maps", "must contain at least one map")
    shapes = tuple(_spatial_shape(feature_map) for feature_map in feature_maps)
    first_shape = shapes[0]
    if len(first_shape) != 3:
        raise FeatureDenoisingError("feature_maps", "expected maps shaped (height, width, dim)")
    if any(shape != first_shape for shape in shapes):
        raise FeatureDenoisingError("feature_maps", "all maps must share the same shape")
    if min(first_shape) <= 0:
        raise FeatureDenoisingError("feature_maps", "height, width, and dim must be positive")
    if first_shape[-1] < 2:
        raise FeatureDenoisingError("feature_maps", "feature dimension must be at least 2")
    return shapes


def _spatial_shape(feature_map: npt.NDArray[np.float32]) -> Tuple[int, int, int]:
    shape = cast("Tuple[int, ...]", tuple(feature_map.shape))
    if len(shape) != 3:
        raise FeatureDenoisingError("feature_maps", "expected maps shaped (height, width, dim)")
    return shape[0], shape[1], shape[2]
