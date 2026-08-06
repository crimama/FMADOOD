from __future__ import annotations

import math

import torch
import torch.nn.functional as functional


def _as_flat_pair(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if left.shape != right.shape:
        message = f"paired tensors must share shape, got {left.shape} and {right.shape}"
        raise ValueError(message)
    if left.shape[-1] <= 0:
        raise ValueError("paired tensors must have a non-empty feature dimension")
    return left.reshape(-1, left.shape[-1]), right.reshape(-1, right.shape[-1])


def raw_paired_canonicalization_loss(
    latent: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Mean raw Euclidean pair distance, scaled by sqrt(feature_dim)."""
    left, right = _as_flat_pair(latent, target)
    scale = math.sqrt(float(left.shape[-1]))
    return torch.linalg.vector_norm(left - right, ord=2, dim=-1).mean() / scale


def raw_pair_distance_loss(latent: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return raw_paired_canonicalization_loss(latent, target)


def normalized_huber_pair_loss(
    latent: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    left, right = _as_flat_pair(latent, target)
    left = functional.normalize(left, dim=-1)
    right = functional.normalize(right, dim=-1)
    return functional.smooth_l1_loss(left, right, beta=beta)


def normalized_pair_huber_loss(
    latent: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    return normalized_huber_pair_loss(latent, target, beta=beta)


def identity_anchor_loss(latent: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return raw_paired_canonicalization_loss(latent, teacher.detach())


def clean_identity_loss(latent: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return identity_anchor_loss(latent, teacher)


def latent_norm_preservation_loss(latent: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    left, right = _as_flat_pair(latent, target)
    scale = math.sqrt(float(left.shape[-1]))
    return torch.abs(
        torch.linalg.vector_norm(left, ord=2, dim=-1)
        - torch.linalg.vector_norm(right.detach(), ord=2, dim=-1),
    ).mean() / scale


def effective_rank(values: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Entropy effective rank of the centered covariance spectrum."""
    if values.ndim != 2:
        values = values.reshape(-1, values.shape[-1])
    if values.shape[0] < 2:
        return values.new_tensor(1.0)
    centered = values - values.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered.float())
    weights = singular.square()
    total = weights.sum()
    if float(total.detach().cpu()) <= eps:
        return values.new_tensor(1.0)
    probs = (weights / total).clamp_min(eps)
    entropy = -(probs * probs.log()).sum()
    return entropy.exp().to(dtype=values.dtype, device=values.device)


def effective_rank_ratio(latent: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return effective_rank(latent) / effective_rank(teacher.detach()).clamp_min(1e-6)


def deterministic_neighbor_retention(
    latent: torch.Tensor,
    teacher: torch.Tensor,
    *,
    k: int = 10,
    max_samples: int = 256,
) -> torch.Tensor:
    """Fraction of sampled teacher kNN identities preserved in latent kNN sets."""
    left, right = _as_flat_pair(latent, teacher)
    n_rows = int(left.shape[0])
    if n_rows <= 1:
        return left.new_tensor(1.0)
    sample_count = min(max_samples, n_rows)
    indices = torch.linspace(
        0,
        n_rows - 1,
        steps=sample_count,
        device=left.device,
    ).round().long().unique()
    k_eff = min(k, n_rows - 1)
    student_distance = torch.cdist(left[indices].float(), left.float(), p=2.0)
    teacher_distance = torch.cdist(right[indices].float(), right.float(), p=2.0)
    row = torch.arange(indices.numel(), device=left.device)
    student_distance[row, indices] = float("inf")
    teacher_distance[row, indices] = float("inf")
    student_neighbors = student_distance.topk(k_eff, largest=False, dim=1).indices
    teacher_neighbors = teacher_distance.topk(k_eff, largest=False, dim=1).indices
    matches = student_neighbors.unsqueeze(-1).eq(teacher_neighbors.unsqueeze(1)).any(dim=-1)
    return matches.float().mean().to(dtype=left.dtype)


def neighbor_identity_retention(
    latent: torch.Tensor,
    teacher: torch.Tensor,
    *,
    k: int = 10,
    max_samples: int = 256,
) -> torch.Tensor:
    return deterministic_neighbor_retention(latent, teacher, k=k, max_samples=max_samples)
