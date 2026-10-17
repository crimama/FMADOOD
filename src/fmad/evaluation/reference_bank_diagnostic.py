"""Post-prediction reference-size diagnosis; never updates fitted state."""
from __future__ import annotations

import numpy as np
import torch


def paired_reference_distances(query, bank, owners, source_count):
    """Exact cosine 1NN against full and every source-excluded bank.

    Reuse one distance matrix. The excluded-source minimum changes only when
    that source owns the winning nearest neighbour. Inputs are unit features.
    """
    if source_count < 2 or query.ndim != 2 or bank.ndim != 2:
        raise ValueError('requires >=2 sources and feature matrices')
    owners = torch.as_tensor(owners, device=bank.device, dtype=torch.long)
    if len(owners) != len(bank) or set(owners.cpu().tolist()) != set(range(source_count)):
        raise ValueError('bank must contain every declared source exactly by identity')
    if not torch.isfinite(query).all() or not torch.isfinite(bank).all():
        raise ValueError('nonfinite reference features')
    distances = 1. - query @ bank.T
    by_source = torch.stack([distances[:, owners == s].amin(1)
                             for s in range(source_count)], dim=1)
    first_two, indices = by_source.topk(2, dim=1, largest=False)
    full = first_two[:, 0]
    excluded = full[:, None].expand(-1, source_count).clone()
    excluded.scatter_(1, indices[:, :1], first_two[:, 1:2])
    return full, excluded


def summarize_reference_shift(full_images, excluded_images, support_q95):
    """Equal probes per image; exclusions are repeated conditions, not samples."""
    full = np.asarray(full_images, dtype=np.float64)
    excluded = np.asarray(excluded_images, dtype=np.float64)
    if (full.ndim != 2 or excluded.ndim != 3 or excluded.shape[:2] != full.shape
            or not full.size or not np.isfinite(full).all()
            or not np.isfinite(excluded).all() or not np.isfinite(support_q95)
            or support_q95 <= 0):
        raise ValueError('invalid paired reference samples')
    if np.any(excluded < full[..., None] - 2e-6):
        raise ValueError('source removal cannot decrease exact 1NN distance')
    qfull = float(np.quantile(full, .95))
    qexcluded = float(np.quantile(excluded, .95))
    return dict(normal_images=len(full), probes_per_image=full.shape[1],
        excluded_sources=excluded.shape[2], support_loo_q95=float(support_q95),
        query_full_q95=qfull, query_excluded_q95=qexcluded,
        full_to_support_q95=qfull/support_q95,
        excluded_to_support_q95=qexcluded/support_q95,
        excluded_to_full_q95=qexcluded/qfull if qfull>1e-12 else None,
        full_quantile_ratio_defined=qfull>1e-12,
        full_exceed_support_q95=float(np.mean(full > support_q95)),
        excluded_exceed_support_q95=float(np.mean(excluded > support_q95)),
        per_image_full_q95=np.quantile(full,.95,axis=1).tolist(),
        per_image_excluded_q95=np.quantile(excluded,.95,axis=1).tolist(),
        per_image_full_exceed=(full > support_q95).mean(1).tolist(),
        per_image_excluded_exceed=(excluded > support_q95).mean(1).tolist(),
        minimum_paired_distance_increase=float((excluded-full[...,None]).min()),
        sampling='same deterministic 256 spatial probes per image, all source exclusions; descriptive only',
        independent_token_or_exclusion_claim=False)
