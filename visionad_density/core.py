from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass(frozen=True)
class DensityStats:
    mean: torch.Tensor
    std: torch.Tensor
    energy_mean: torch.Tensor
    energy_std: torch.Tensor

    @classmethod
    def fit(cls, support: torch.Tensor) -> "DensityStats":
        mean = support.mean(0, keepdim=True)
        std = support.std(0, keepdim=True, unbiased=False).clamp_min(1e-6)
        energy = 0.5 * ((support - mean) / std).square().sum(-1)
        return cls(mean, std, energy.mean(), energy.std(unbiased=False).clamp_min(1e-6))

    def penalty(self, query: torch.Tensor) -> torch.Tensor:
        energy = 0.5 * ((query - self.mean) / self.std).square().sum(-1)
        return torch.relu((energy - self.energy_mean) / self.energy_std)


def support_views(image: Image.Image) -> tuple[Image.Image, ...]:
    transpose = Image.Transpose
    return (
        image,
        image.transpose(transpose.ROTATE_90),
        image.transpose(transpose.ROTATE_180),
        image.transpose(transpose.ROTATE_270),
        image.transpose(transpose.FLIP_TOP_BOTTOM),
        image.transpose(transpose.FLIP_LEFT_RIGHT),
    )


def fuse_layers(features: Sequence[torch.Tensor]) -> torch.Tensor:
    if not features:
        raise ValueError("at least one feature layer is required")
    return torch.stack(tuple(features), dim=0).mean(0)


def cosine_nn_map(query: torch.Tensor, support: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    support_norm = F.normalize(support, dim=-1)
    rows = []
    for start in range(0, query.shape[0], chunk):
        q = F.normalize(query[start : start + chunk], dim=-1)
        rows.append(1.0 - (q @ support_norm.T).amax(-1))
    return torch.cat(rows)


def combine_score(
    query: torch.Tensor,
    support: torch.Tensor,
    density: DensityStats,
    density_weight: float,
) -> torch.Tensor:
    return cosine_nn_map(query, support) + density_weight * density.penalty(query)


def restore_query_map(values: torch.Tensor, grid: tuple[int, int], flipped: bool) -> torch.Tensor:
    score = values.reshape(1, 1, *grid)
    if flipped:
        score = torch.flip(score, dims=(-2,))
    return score


def upsample_map(score: torch.Tensor, size: int = 256) -> np.ndarray:
    return F.interpolate(score, size=(size, size), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
