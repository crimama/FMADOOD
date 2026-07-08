from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
from typing_extensions import override

FeatureArray = Union[npt.NDArray[np.float32], torch.Tensor]


@dataclass(frozen=True)
class FeatureShapeError(ValueError):
    shape: str
    reason: str

    @override
    def __str__(self) -> str:
        return f"Invalid feature shape {self.shape}: {self.reason}"


@dataclass(frozen=True)
class PatchBatch:
    flat_features: torch.Tensor
    image_indices: torch.Tensor
    n_images: int
    patch_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    flat_input: bool

    @property
    def patches_per_image(self) -> int:
        return int(self.flat_features.shape[0] // self.n_images)

    def image_tensor(self) -> torch.Tensor:
        return self.flat_features.reshape(
            self.n_images,
            self.patches_per_image,
            self.flat_features.shape[1],
        )

    def restore(self, values: torch.Tensor) -> torch.Tensor:
        return values.reshape(self.output_shape)

    def restore_features(self, values: torch.Tensor) -> torch.Tensor:
        return values.reshape((*self.output_shape, values.shape[-1]))


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _to_float_tensor(features: FeatureArray, device: torch.device) -> torch.Tensor:
    if isinstance(features, np.ndarray):
        return torch.as_tensor(features, dtype=torch.float32, device=device)
    return features.detach().to(device=device, dtype=torch.float32)


def _image_indices(n_images: int, patches_per_image: int, device: torch.device) -> torch.Tensor:
    return torch.arange(
        n_images,
        device=device,
        dtype=torch.long,
    ).repeat_interleave(patches_per_image)


def as_patch_batch(features: FeatureArray, device: torch.device) -> PatchBatch:
    tensor = _to_float_tensor(features, device)
    shape = tuple(tensor.shape)
    if tensor.ndim == 2:
        n_patches, dim = int(tensor.shape[0]), int(tensor.shape[1])
        if n_patches == 0:
            raise FeatureShapeError(str(shape), "expected at least one patch")
        if dim < 2:
            raise FeatureShapeError(str(shape), "expected feature dimension >= 2")
        return PatchBatch(
            flat_features=tensor,
            image_indices=torch.zeros(n_patches, device=device, dtype=torch.long),
            n_images=1,
            patch_shape=(n_patches,),
            output_shape=(n_patches,),
            flat_input=True,
        )
    if tensor.ndim == 3:
        n_images, patches_per_image, dim = (
            int(tensor.shape[0]),
            int(tensor.shape[1]),
            int(tensor.shape[2]),
        )
        if n_images == 0 or patches_per_image == 0:
            raise FeatureShapeError(str(shape), "expected at least one image and patch")
        if dim < 2:
            raise FeatureShapeError(str(shape), "expected feature dimension >= 2")
        return PatchBatch(
            flat_features=tensor.reshape(n_images * patches_per_image, dim),
            image_indices=_image_indices(n_images, patches_per_image, device),
            n_images=n_images,
            patch_shape=(patches_per_image,),
            output_shape=(n_images, patches_per_image),
            flat_input=False,
        )
    if tensor.ndim == 4:
        n_images, height, width, dim = (
            int(tensor.shape[0]),
            int(tensor.shape[1]),
            int(tensor.shape[2]),
            int(tensor.shape[3]),
        )
        if n_images == 0 or height == 0 or width == 0:
            raise FeatureShapeError(str(shape), "expected at least one image and spatial patch")
        if dim < 2:
            raise FeatureShapeError(str(shape), "expected feature dimension >= 2")
        patches_per_image = height * width
        return PatchBatch(
            flat_features=tensor.reshape(n_images * patches_per_image, dim),
            image_indices=_image_indices(n_images, patches_per_image, device),
            n_images=n_images,
            patch_shape=(height, width),
            output_shape=(n_images, height, width),
            flat_input=False,
        )
    raise FeatureShapeError(
        str(shape),
        "expected (n_patches, dim), (n_images, patches, dim), or (n_images, height, width, dim)",
    )


def as_2d_float_tensor(features: FeatureArray, device: torch.device) -> torch.Tensor:
    batch = as_patch_batch(features, device)
    if batch.flat_features.shape[1] < 2:
        raise FeatureShapeError(
            str(tuple(batch.flat_features.shape)),
            "expected feature dimension >= 2",
        )
    return batch.flat_features


def to_numpy(tensor: torch.Tensor) -> npt.NDArray[np.float32]:
    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)
