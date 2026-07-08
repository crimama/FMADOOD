from __future__ import annotations

import math

import torch


def patch_log_prob(z: torch.Tensor, logdet: torch.Tensor) -> torch.Tensor:
    dim = z.shape[-1]
    log_pz = -0.5 * z.square().sum(dim=-1) - 0.5 * dim * math.log(2.0 * math.pi)
    return log_pz + logdet


def patch_nll(z: torch.Tensor, logdet: torch.Tensor) -> torch.Tensor:
    return -patch_log_prob(z, logdet)


def mean_nll(z: torch.Tensor, logdet: torch.Tensor) -> torch.Tensor:
    return patch_nll(z, logdet).mean()


def tail_aware_nll(
    z: torch.Tensor,
    logdet: torch.Tensor,
    tail_weight: float,
    tail_top_k_ratio: float,
) -> torch.Tensor:
    nll = patch_nll(z, logdet)
    flat = nll.reshape(1, -1) if nll.ndim == 1 else nll.reshape(nll.shape[0], -1)
    n_patches = flat.shape[1]
    k = max(1, int(n_patches * tail_top_k_ratio))
    tail = torch.topk(flat, k, dim=1).values.mean()
    mean = flat.mean()
    return (1.0 - tail_weight) * mean + tail_weight * tail
