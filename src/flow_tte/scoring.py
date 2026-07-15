from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from typing_extensions import final

from flow_tte.config import ScoreConfig
from flow_tte.memory import TorchMemoryBank


@dataclass(frozen=True)
@final
class ScoreCalibration:
    distance_mean: float
    distance_std: float

    @classmethod
    def fit(
        cls,
        m0_features: torch.Tensor,
        config: ScoreConfig,
        m0_contexts: Optional[torch.Tensor] = None,
    ) -> "ScoreCalibration":
        if not config.loo_standardize:
            return cls(distance_mean=0.0, distance_std=1.0)
        if m0_features.shape[0] <= 1:
            return cls(distance_mean=0.0, distance_std=1.0)
        calibration_features = m0_features
        calibration_contexts = m0_contexts
        if 1 < config.calibration_sample_size < m0_features.shape[0]:
            sample_indices = torch.linspace(
                0,
                m0_features.shape[0] - 1,
                steps=config.calibration_sample_size,
                device=m0_features.device,
            ).round().to(torch.long)
            calibration_features = m0_features[sample_indices]
            if m0_contexts is not None:
                calibration_contexts = m0_contexts[sample_indices]
        bank = TorchMemoryBank()
        bank.fit(calibration_features, contexts=calibration_contexts)
        context_weight = config.context_weight if config.context_mode == "soft_penalty" else 0.0
        context_top_m = config.context_top_m if config.context_mode == "top_m" else None
        query = bank.query(
            calibration_features,
            k=2,
            chunk_size=config.query_chunk_size,
            squared=config.use_squared_distance,
            query_contexts=calibration_contexts,
            context_weight=context_weight,
            context_top_m=context_top_m,
        )
        leave_one_out = query.distances[:, 1]
        distance_std = float(leave_one_out.std(unbiased=False).clamp_min(1e-6).detach().cpu())
        return cls(
            distance_mean=float(leave_one_out.mean().detach().cpu()),
            distance_std=distance_std,
        )

    def normalize_distance(self, distances: torch.Tensor) -> torch.Tensor:
        return (distances - self.distance_mean) / self.distance_std


@dataclass(frozen=True)
@final
class ScoreInputs:
    query_z: torch.Tensor
    nll: torch.Tensor
    nll_penalty: torch.Tensor
    image_indices: torch.Tensor
    n_images: int
    query_contexts: Optional[torch.Tensor] = None


@dataclass(frozen=True)
@final
class ScoreResult:
    patch_scores: torch.Tensor
    image_scores: torch.Tensor
    image_score: float
    distances: torch.Tensor
    distance_scores: torch.Tensor
    density_penalty: torch.Tensor


def top_percent_mean(scores: torch.Tensor, top_percent: float) -> float:
    k = max(1, int(scores.numel() * top_percent))
    return float(torch.topk(scores.reshape(-1), k=k).values.mean().detach().cpu())


def image_top_percent_mean(
    scores: torch.Tensor,
    image_indices: torch.Tensor,
    n_images: int,
    top_percent: float,
) -> torch.Tensor:
    image_scores: list[torch.Tensor] = []
    for image_idx in range(n_images):
        image_patch_scores = scores[image_indices == image_idx]
        k = max(1, int(image_patch_scores.numel() * top_percent))
        image_scores.append(torch.topk(image_patch_scores, k=k).values.mean())
    return torch.stack(image_scores)


def score_flow_memory(
    inputs: ScoreInputs,
    bank: TorchMemoryBank,
    config: ScoreConfig,
    calibration: ScoreCalibration,
) -> ScoreResult:
    if config.score_mode == "nf_nll":
        patch_scores = inputs.nll
        zero_scores = torch.zeros_like(inputs.nll)
        image_scores = image_top_percent_mean(
            patch_scores,
            inputs.image_indices,
            inputs.n_images,
            config.top_percent,
        )
        return ScoreResult(
            patch_scores=patch_scores,
            image_scores=image_scores,
            image_score=float(image_scores.max().detach().cpu()),
            distances=zero_scores,
            distance_scores=zero_scores,
            density_penalty=inputs.nll_penalty,
        )

    context_weight = config.context_weight if config.context_mode == "soft_penalty" else 0.0
    context_top_m = config.context_top_m if config.context_mode == "top_m" else None
    query = bank.query(
        inputs.query_z,
        k=1,
        chunk_size=config.query_chunk_size,
        squared=config.use_squared_distance,
        query_contexts=inputs.query_contexts,
        context_weight=context_weight,
        context_top_m=context_top_m,
    )
    distances = query.distances[:, 0]
    distance_scores = calibration.normalize_distance(distances)
    patch_scores = (
        config.distance_weight * distance_scores
        + config.density_weight * inputs.nll_penalty
    )
    image_scores = image_top_percent_mean(
        patch_scores,
        inputs.image_indices,
        inputs.n_images,
        config.top_percent,
    )
    return ScoreResult(
        patch_scores=patch_scores,
        image_scores=image_scores,
        image_score=float(image_scores.max().detach().cpu()),
        distances=distances,
        distance_scores=distance_scores,
        density_penalty=inputs.nll_penalty,
    )
