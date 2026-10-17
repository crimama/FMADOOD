"""Reference-matching ablations for frozen DINOv2 features.

R0 is the raw late-layer nearest-neighbour control.  R1 keeps one sampled
support-token identity across two layers but lets each layer select its own
nearest neighbour.  R2 uses a late-layer ring context key to select one
reference and scores that same reference in both layers.

The implementation deliberately keeps only capped support tokens and computes
nearest-neighbour distances in query chunks; dense anomaly maps are produced
only for the current query and are never accumulated in the method state.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

from fmad.methods.PSMA_MR import _feature_matrix, _read_rgb
from fmad.methods.base import BaseMethod, Prediction
from fmad.postprocess import component_quantile
from fmad.registry import METHOD_REGISTRY


def immerkaer_sigma(image: np.ndarray) -> float:
    """Estimate sensor noise variance using Immerkaer 1996 method.

    Input is an RGB array (uint8, 0-255). Returns sigma in 8-bit units.
    The method applies a 2-D Laplacian-like kernel via convolution on the
    grayscale image and estimates sigma from the convolution magnitude.
    """
    if image.dtype != np.uint8:
        raise ValueError("image must be uint8 (0-255 range)")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HxWx3 RGB")

    # Convert to grayscale using standard weights
    gray = (0.299 * image[..., 0] +
            0.587 * image[..., 1] +
            0.114 * image[..., 2]).astype(np.float64)

    # Immerkaer kernel
    M = np.array([[1, -2, 1],
                  [-2, 4, -2],
                  [1, -2, 1]], dtype=np.float64)

    # Convolution with 'valid' mode
    from scipy.signal import convolve2d
    conv = convolve2d(gray, M, mode='valid')

    # Estimate sigma
    h, w = gray.shape
    sigma = np.sqrt(np.pi / 2.0) / (6.0 * (h - 2) * (w - 2)) * np.sum(np.abs(conv))
    return float(sigma)


def _distance_chunks(query: torch.Tensor, bank: torch.Tensor, chunk: int, k: int = 1,
                     return_indices: bool = False):
    """Return each query's mean cosine distance to its k nearest bank tokens.

    When ``return_indices`` is set, also return the indices (into ``bank``'s
    first dimension) of those k nearest tokens per query, sorted nearest
    first -- ``torch.topk(..., largest=False)`` is sorted by default, and the
    ``k == 1`` path uses ``argmin`` as its single-column companion. Default
    behaviour and return type (a single tensor) are unchanged when
    ``return_indices`` is left False; other modules import this function.
    """
    k = min(k, bank.shape[0])
    values = []
    indices = [] if return_indices else None
    for start in range(0, len(query), chunk):
        distances = 1.0 - query[start:start + chunk] @ bank.T
        if k <= 1:
            values.append(distances.amin(dim=1))
            if return_indices:
                indices.append(distances.argmin(dim=1, keepdim=True))
        else:
            topk = distances.topk(k, dim=1, largest=False)
            values.append(topk.values.mean(dim=1))
            if return_indices:
                indices.append(topk.indices)
    values_cat = torch.cat(values) if values else query.new_empty((0,))
    if not return_indices:
        return values_cat
    indices_cat = (torch.cat(indices) if indices
                    else query.new_empty((0, max(k, 1)), dtype=torch.long))
    return values_cat, indices_cat


def local_scale(neighbour_index: torch.Tensor, loo: torch.Tensor, global_scale: torch.Tensor,
                k: int, shrinkage: float) -> torch.Tensor:
    """Shrink a per-query-token local LOO scale estimate toward the global scalar.

    ``neighbour_index`` holds, for each query token, the indices (nearest
    first) of its nearest bank tokens in one layer -- as returned by
    ``_distance_chunks(..., return_indices=True)``. ``loo`` is that layer's
    per-bank-token leave-source-out nearest-neighbour distance, aligned with
    the bank's token order (see ``_fit_scales_for_resolution``). Only the
    first ``k`` neighbour columns are used, averaged into a local estimate of
    how loose the normal model is around that query token, then shrunk toward
    ``global_scale`` with weight ``shrinkage`` (0 = pure local, large = pure
    global): ``(n_eff * local_mean + shrinkage * global_scale) / (n_eff +
    shrinkage)`` where ``n_eff`` is the number of neighbour columns actually
    used. The result is clamped to a minimum of 1e-4. Pure function: neither
    input tensor is mutated, and a new tensor is always returned.
    """
    neighbour_index = neighbour_index[:, :k]
    n_eff = neighbour_index.shape[1]
    local_mean = loo[neighbour_index].mean(dim=1)
    sigma_local = (n_eff * local_mean + shrinkage * global_scale) / (n_eff + shrinkage)
    return sigma_local.clamp_min(1e-4)


def _top_fraction_mean(values: np.ndarray, fraction: float = 0.01) -> float:
    """Mean of the top max(1, floor(N * fraction)) values (same rule as SuperAD)."""
    flat = np.asarray(values).reshape(-1)
    if flat.size == 0:
        raise ValueError("cannot aggregate an empty score map")
    k = max(1, int(np.floor(flat.size * fraction)))
    return float(np.mean(np.partition(flat, -k)[-k:]))


def rank_normalize(scores: torch.Tensor) -> torch.Tensor:
    """Rank-normalise a 1-D score vector to [0, 1].

    ``rank = argsort(argsort(scores))`` gives each element's ordinal
    position (0-indexed) under an ascending sort; dividing by ``n - 1``
    maps that into ``[0, 1]``. Ties are broken by the stable argsort's
    input order -- deterministic, but otherwise arbitrary among equal
    values. For ``n == 1`` the result is a single zero.
    """
    n = scores.shape[0]
    if n <= 1:
        return torch.zeros_like(scores)
    order = torch.argsort(scores, stable=True)
    ranks = torch.argsort(order, stable=True)
    return ranks.to(scores.dtype) / (n - 1)


def coreset_indices(features: torch.Tensor, count: int) -> torch.Tensor:
    """Deterministic greedy farthest-point (k-center) token selection."""
    n = features.shape[0]
    count = min(count, n)
    mean = F.normalize(features.mean(dim=0, keepdim=True), dim=-1)
    start = torch.argmax((features @ mean.T).squeeze(1)).item()
    selected = [start]
    min_distance = 1.0 - features @ features[start]
    for _ in range(1, count):
        candidate = torch.argmax(min_distance).item()
        if min_distance[candidate].item() <= 1e-6:
            break
        selected.append(candidate)
        new_distance = 1.0 - features @ features[candidate]
        min_distance = torch.minimum(min_distance, new_distance)
    return torch.as_tensor(selected, dtype=torch.long, device=features.device)


def trim_outlier_tokens(records: list[dict[str, Any]], chunk: int, fraction: float,
                        device=None) -> list[dict[str, Any]]:
    """Drop the globally most support-atypical bank tokens, per resolution.

    For every record's tokens, compute the leave-source-out nearest-neighbour
    cosine distance -- using the layer-concatenated, L2-normalised identity
    vector of each token -- against tokens of the OTHER support images, then
    remove the ``fraction`` of tokens with the largest such distance, never
    dropping a source below one surviving token. Returns NEW record dicts;
    the input records (and their tensors) are never mutated.
    """
    if not records:
        return []
    if not (0.0 <= fraction < 1.0):
        raise ValueError("bank_outlier_trim must be a finite number in [0, 1)")
    if fraction == 0.0:
        return [dict(record) for record in records]

    # Records are CPU-staged; the leave-source-out search runs on the bank device.
    identities = [F.normalize(torch.cat(record["layers"], dim=1), dim=-1).to(device) for record in records]
    total_tokens = sum(identity.shape[0] for identity in identities)
    max_drop = max(0, total_tokens - len(records))
    target_drop = min(round(fraction * total_tokens), max_drop)
    if target_drop <= 0:
        return [dict(record) for record in records]

    owners: list[int] = []
    distances_per_source = []
    for source_index, identity in enumerate(identities):
        others = torch.cat([identities[j] for j in range(len(identities)) if j != source_index])
        distances_per_source.append(_distance_chunks(identity, others, chunk, k=1))
        owners.extend([source_index] * identity.shape[0])
    distances = torch.cat(distances_per_source)

    order = torch.argsort(distances, descending=True)
    remaining = {index: identity.shape[0] for index, identity in enumerate(identities)}
    dropped = torch.zeros(total_tokens, dtype=torch.bool)
    removed = 0
    for position in order.tolist():
        if removed >= target_drop:
            break
        owner = owners[position]
        if remaining[owner] <= 1:
            continue
        dropped[position] = True
        remaining[owner] -= 1
        removed += 1

    new_records = []
    offset = 0
    for source_index, record in enumerate(records):
        count = identities[source_index].shape[0]
        keep = (~dropped[offset:offset + count]).nonzero(as_tuple=True)[0]
        offset += count
        new_record = dict(record)
        new_record["layers"] = [layer[keep] for layer in record["layers"]]
        new_record["indices"] = record["indices"][keep]
        if "context" in record:
            new_record["context"] = record["context"][keep]
        new_records.append(new_record)
    return new_records


def _ring_context(features: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    """Compute a normalized 3x3 ring descriptor without retaining maps."""
    image = features.T.reshape(1, -1, *grid)
    total = F.avg_pool2d(image, 3, stride=1, padding=1, count_include_pad=True) * 9 - image
    count = F.avg_pool2d(torch.ones_like(image[:, :1]), 3, stride=1,
                         padding=1, count_include_pad=True) * 9 - 1
    ring = (total / count.clamp_min(1)).flatten(2)[0].T
    return F.normalize(ring, dim=-1)


@METHOD_REGISTRY.register("reference_matching")
class ReferenceMatching(BaseMethod):
    """Configurable R0/R1/R2 reference correspondence scorer."""

    def __init__(self, backbone, config: dict):
        super().__init__(backbone, config)
        self.device = torch.device(backbone.device)
        self.variant = str(config.get("variant", config.get("stage", "R0"))).upper()
        if self.variant not in {"R0", "R1", "R2"}:
            raise ValueError("reference_matching variant must be R0, R1, or R2")
        raw_resolutions = config.get("resolutions", [448, 672])
        if not isinstance(raw_resolutions, (list, tuple)) or not raw_resolutions:
            raise ValueError("resolutions must be a non-empty sequence")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
               not math.isfinite(float(v)) or float(v) <= 0 or int(v) != float(v)
               for v in raw_resolutions):
            raise ValueError("resolutions must contain finite positive integers")
        self.resolutions = tuple(int(v) for v in raw_resolutions)
        # R0 stays pinned to the single late layer (it is the raw control).
        # R1/R2 default to [11, 23] but accept any declared stack so that the
        # feature-depth choice can be ablated; the backbone must return exactly
        # the declared layers (checked below).
        raw_layers = config.get("feature_layers", [23] if self.variant == "R0" else [11, 23])
        if not isinstance(raw_layers, (list, tuple)) or not raw_layers:
            raise ValueError("feature_layers must be a non-empty sequence")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in raw_layers):
            raise ValueError("feature_layers must contain non-negative integers")
        self.layers = tuple(int(v) for v in raw_layers)
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("feature_layers must not repeat a layer")
        if self.variant == "R0" and self.layers != (23,):
            raise ValueError("R0 uses feature layer [23]")
        # "mad" divides each component by its leave-source-out normal MAD, which
        # standardises the CENTRE of that component's normal distribution.
        # AUPRO@0.05 operates in the upper TAIL, so components with equal MAD but
        # unequal tails contribute unequally to the FPR budget. "tail" instead
        # maps each component onto -log(1 - F) using the support's own normal
        # CDF, so every component is expressed in the same FPR units before they
        # are averaged. Support-only; no test data is used to fit it.
        # "tailscale" keeps the divide-by-a-constant form (so the magnitude
        # information that scale_fusion='rank' destroyed is preserved) but uses
        # the component's own leave-source-out normal q95 instead of its MAD.
        # "tail" is the aggressive variant: -log(1 - F) with an exponential tail
        # above q95, which is rank-like inside the normal body and linear in the
        # excess above it. Both are fitted from supports only.
        # "none" means raw distance with divisor 1 (uncalibrated equal-weight baseline).
        self.score_normalization = str(config.get("score_normalization", "mad"))
        if self.score_normalization not in {"mad", "tailscale", "tail", "none"}:
            raise ValueError("score_normalization must be 'mad', 'tailscale', 'tail', or 'none'")
        tail_quantile = config.get("tail_quantile", 0.95)
        if (isinstance(tail_quantile, bool) or not isinstance(tail_quantile, (int, float))
                or not 0.5 <= float(tail_quantile) < 1.0):
            raise ValueError("tail_quantile must be a number in [0.5, 1)")
        self.tail_quantile = float(tail_quantile)
        rho = config.get("tailscale_center_rho", 0.0)
        if (isinstance(rho, bool) or not isinstance(rho, (int, float))
                or not 0.0 <= float(rho) <= 1.0):
            raise ValueError("tailscale_center_rho must be a number in [0, 1]")
        self.tailscale_center_rho = float(rho)
        if self.tailscale_center_rho and (self.score_normalization != "tailscale"
                or self.variant != "R1" or config.get("local_reliability", "off") != "off"):
            raise ValueError("centered tailscale requires R1 tailscale with local_reliability off")
        self.cap = int(config.get("bank_tokens_per_source", 256))
        self.chunk = int(config.get("query_chunk_size", 128))
        # Optional override for the bank token subsample only. Support image
        # selection still follows context.seed, so identical supports can be
        # scored with different token subsamples to measure sampling noise.
        bank_seed = config.get("bank_seed")
        if bank_seed is not None and (isinstance(bank_seed, bool) or not isinstance(bank_seed, int)
                                      or not 0 <= bank_seed < 2**32):
            raise ValueError("bank_seed must be null or an integer in [0, 2**32)")
        self.bank_seed = bank_seed
        self.context_lambda = float(config.get("context_lambda", 0.25))
        self.robust_scale = self.variant != "R0"
        self.bank_selection = str(config.get("bank_selection", "random"))
        if self.bank_selection not in {"random", "coreset"}:
            raise ValueError("bank_selection must be 'random' or 'coreset'")
        knn_k = config.get("knn_k", 1)
        if isinstance(knn_k, bool) or not isinstance(knn_k, int) or knn_k < 1:
            raise ValueError("knn_k must be an integer >= 1")
        self.knn_k = knn_k
        if self.cap <= 0 or self.chunk <= 0:
            raise ValueError("bank_tokens_per_source and query_chunk_size must be positive")
        if not math.isfinite(self.context_lambda) or self.context_lambda < 0:
            raise ValueError("context_lambda must be finite and non-negative")
        bank_outlier_trim = config.get("bank_outlier_trim", 0.0)
        if (isinstance(bank_outlier_trim, bool) or not isinstance(bank_outlier_trim, (int, float))
                or not math.isfinite(float(bank_outlier_trim))
                or not 0.0 <= float(bank_outlier_trim) < 1.0):
            raise ValueError("bank_outlier_trim must be a finite number in [0, 1)")
        self.bank_outlier_trim = float(bank_outlier_trim)
        bank_bags = config.get("bank_bags", 1)
        if isinstance(bank_bags, bool) or not isinstance(bank_bags, int) or bank_bags < 1:
            raise ValueError("bank_bags must be an integer >= 1")
        if bank_bags > 1 and self.bank_selection == "coreset":
            raise ValueError("bank_bags > 1 is not supported with bank_selection='coreset'")
        if bank_bags > 1 and self.variant != "R1":
            raise ValueError("bank_bags > 1 is only supported for variant R1")
        self.bank_bags = bank_bags
        self.scale_fusion = str(config.get("scale_fusion", "mean"))
        if self.scale_fusion not in {"mean", "max", "rank"}:
            raise ValueError("scale_fusion must be 'mean', 'max', or 'rank'")
        # Fusion across feature LAYERS (distinct from scale_fusion, which fuses
        # across RESOLUTIONS). Only meaningful for R1, which has multiple
        # layers; R0 has one layer and R2 fuses layers via shared reference
        # identity, so this option is rejected outside R1.
        self.layer_fusion = str(config.get("layer_fusion", "mean"))
        if self.layer_fusion not in {"mean", "max", "quantile", "residual_spike", "deep_plus_shallow"}:
            raise ValueError(
                "layer_fusion must be 'mean', 'max', 'quantile', 'residual_spike', or 'deep_plus_shallow'"
            )
        layer_fusion_q = config.get("layer_fusion_q", 0.9)
        if (isinstance(layer_fusion_q, bool) or not isinstance(layer_fusion_q, (int, float))
                or not math.isfinite(float(layer_fusion_q)) or not 0.0 < float(layer_fusion_q) <= 1.0):
            raise ValueError("layer_fusion_q must be a finite number in (0, 1]")
        self.layer_fusion_q = float(layer_fusion_q)
        layer_fusion_tau = config.get("layer_fusion_tau", 1.2)
        if (isinstance(layer_fusion_tau, bool) or not isinstance(layer_fusion_tau, (int, float))
                or not math.isfinite(float(layer_fusion_tau)) or float(layer_fusion_tau) < 0):
            raise ValueError("layer_fusion_tau must be a finite number >= 0")
        self.layer_fusion_tau = float(layer_fusion_tau)
        self.layer_fusion_tau_mode = str(config.get("layer_fusion_tau_mode", "fixed"))
        if self.layer_fusion_tau_mode not in {"fixed", "support_quantile"}:
            raise ValueError("layer_fusion_tau_mode must be 'fixed' or 'support_quantile'")
        layer_fusion_tau_q = config.get("layer_fusion_tau_q", 0.995)
        if (isinstance(layer_fusion_tau_q, bool) or not isinstance(layer_fusion_tau_q, (int, float))
                or not math.isfinite(float(layer_fusion_tau_q)) or not 0.0 < float(layer_fusion_tau_q) <= 1.0):
            raise ValueError("layer_fusion_tau_q must be a finite number in (0, 1]")
        self.layer_fusion_tau_q = float(layer_fusion_tau_q)
        layer_fusion_gamma = config.get("layer_fusion_gamma", 1.0)
        if (isinstance(layer_fusion_gamma, bool) or not isinstance(layer_fusion_gamma, (int, float))
                or not math.isfinite(float(layer_fusion_gamma)) or float(layer_fusion_gamma) < 0):
            raise ValueError("layer_fusion_gamma must be a finite number >= 0")
        self.layer_fusion_gamma = float(layer_fusion_gamma)
        layer_fusion_shallow_idx = config.get("layer_fusion_shallow_idx", 0)
        if (isinstance(layer_fusion_shallow_idx, bool) or not isinstance(layer_fusion_shallow_idx, int)
                or layer_fusion_shallow_idx < 0):
            raise ValueError("layer_fusion_shallow_idx must be an integer >= 0")
        self.layer_fusion_shallow_idx = layer_fusion_shallow_idx
        # Optional bounded negative branch below tau for residual_spike.
        # alpha=0 keeps the original relu(s - tau) behaviour exactly.
        layer_fusion_neg_alpha = config.get("layer_fusion_neg_alpha", 0.0)
        if (isinstance(layer_fusion_neg_alpha, bool) or not isinstance(layer_fusion_neg_alpha, (int, float))
                or not math.isfinite(float(layer_fusion_neg_alpha)) or float(layer_fusion_neg_alpha) < 0):
            raise ValueError("layer_fusion_neg_alpha must be a finite number >= 0")
        self.layer_fusion_neg_alpha = float(layer_fusion_neg_alpha)
        if self.layer_fusion_neg_alpha > 0 and self.layer_fusion != "residual_spike":
            raise ValueError("layer_fusion_neg_alpha requires layer_fusion='residual_spike'")
        # Image-level score options. Defaults (max of the final map, no separate
        # image gate) reproduce the original image score exactly.
        self.image_score_aggregation = str(config.get("image_score_aggregation", "max"))
        if self.image_score_aggregation not in {"max", "top1pct"}:
            raise ValueError("image_score_aggregation must be 'max' or 'top1pct'")
        self.image_score_source = str(config.get("image_score_source", "final"))
        if self.image_score_source not in {"final", "image_gate"}:
            raise ValueError("image_score_source must be 'final' or 'image_gate'")
        image_residual_tau_q = config.get("image_residual_tau_q")
        if image_residual_tau_q is not None:
            if (isinstance(image_residual_tau_q, bool) or not isinstance(image_residual_tau_q, (int, float))
                    or not math.isfinite(float(image_residual_tau_q))
                    or not 0.0 < float(image_residual_tau_q) <= 1.0):
                raise ValueError("image_residual_tau_q must be a finite number in (0, 1]")
            image_residual_tau_q = float(image_residual_tau_q)
            if (self.variant != "R1" or self.layer_fusion != "residual_spike"
                    or self.layer_fusion_tau_mode != "support_quantile"):
                raise ValueError("image_residual_tau_q requires R1 residual_spike support_quantile fusion")
        self.image_residual_tau_q = image_residual_tau_q
        if self.image_score_source == "image_gate" and self.image_residual_tau_q is None:
            raise ValueError("image_score_source='image_gate' requires image_residual_tau_q")
        self.image_score_candidates = config.get("image_score_candidates", False)
        if type(self.image_score_candidates) is not bool:
            raise ValueError("image_score_candidates must be a bool")
        prefix = "max" if self.image_score_aggregation == "max" else "top1pct_mean"
        self.image_score_key = f"{prefix}_{self.image_score_source}"
        self._image_token_records = None
        self._last_image_candidates = None
        if self.layer_fusion != "mean" and self.variant != "R1":
            raise ValueError("layer_fusion is only supported for variant R1")
        # local_reliability calibrates (and, in "weight" mode, fuses) layer
        # scores using a per-query-token LOCAL estimate of the bank's normal
        # looseness instead of one global scalar per (resolution, layer, bag).
        # It is purely a property of the support set: no test data or labels
        # are involved. R1-only, since R0 has one layer and R2 reuses one
        # shared reference identity across layers rather than scoring each
        # layer independently.
        self.local_reliability = str(config.get("local_reliability", "off"))
        if self.local_reliability not in {"off", "scale", "weight", "center"}:
            raise ValueError("local_reliability must be 'off', 'scale', 'weight', or 'center'")
        if self.local_reliability != "off" and self.variant != "R1":
            raise ValueError("local_reliability is only supported for variant R1")
        if self.score_normalization == "none" and self.local_reliability != "off":
            raise ValueError("score_normalization='none' is incompatible with local_reliability")
        local_center_alpha = config.get("local_center_alpha", 0.0)
        if (isinstance(local_center_alpha, bool)
                or not isinstance(local_center_alpha, (int, float))
                or not math.isfinite(float(local_center_alpha))
                or not 0.0 <= float(local_center_alpha) <= 1.0):
            raise ValueError("local_center_alpha must be a finite number in [0, 1]")
        self.local_center_alpha = float(local_center_alpha)
        if self.local_reliability == "center" and self.score_normalization != "tailscale":
            raise ValueError("local_reliability='center' requires tailscale normalization")
        if self.local_center_alpha and self.local_reliability != "center":
            raise ValueError("local_center_alpha requires local_reliability='center'")
        local_reliability_k = config.get("local_reliability_k", 8)
        if (isinstance(local_reliability_k, bool) or not isinstance(local_reliability_k, int)
                or local_reliability_k < 1):
            raise ValueError("local_reliability_k must be an integer >= 1")
        self.local_reliability_k = local_reliability_k
        local_reliability_shrinkage = config.get("local_reliability_shrinkage", 8.0)
        if (isinstance(local_reliability_shrinkage, bool)
                or not isinstance(local_reliability_shrinkage, (int, float))
                or not math.isfinite(float(local_reliability_shrinkage))
                or float(local_reliability_shrinkage) < 0):
            raise ValueError("local_reliability_shrinkage must be a finite number >= 0")
        self.local_reliability_shrinkage = float(local_reliability_shrinkage)
        # "weight" replaces the layer-fusion rule with a precision-weighted
        # mean, so an independently-chosen non-mean layer_fusion would be
        # ambiguous (which rule wins?); reject that combination outright.
        if self.local_reliability == "weight" and self.layer_fusion != "mean":
            raise ValueError("local_reliability='weight' requires layer_fusion='mean'")
        self.prediction_postprocess = str(config.get("prediction_postprocess", "none"))
        if self.prediction_postprocess not in {"none", "component_quantile"}:
            raise ValueError("prediction_postprocess must be 'none' or 'component_quantile'")
        component_percentile = config.get("component_percentile", 99.0)
        component_q = config.get("component_q", 1.0)
        component_dilate_size = config.get("component_dilate_size", 0)
        if (isinstance(component_percentile, bool)
                or not isinstance(component_percentile, (int, float))
                or not math.isfinite(float(component_percentile))
                or not 0.0 <= float(component_percentile) <= 100.0):
            raise ValueError("component_percentile must be a finite number in [0, 100]")
        if (isinstance(component_q, bool) or not isinstance(component_q, (int, float))
                or not math.isfinite(float(component_q)) or not 0.0 <= float(component_q) <= 1.0):
            raise ValueError("component_q must be a finite number in [0, 1]")
        if (isinstance(component_dilate_size, bool) or not isinstance(component_dilate_size, int)
                or component_dilate_size < 0):
            raise ValueError("component_dilate_size must be an integer >= 0")
        self.component_percentile = float(component_percentile)
        self.component_q = float(component_q)
        self.component_dilate_size = component_dilate_size
        # Noise guard: optional per-query noise detection and remediation
        noise_guard = config.get("noise_guard", False)
        if type(noise_guard) is not bool:
            raise ValueError("noise_guard must be a bool")
        self.noise_guard = noise_guard
        noise_guard_k = config.get("noise_guard_k", 9.0)
        if (isinstance(noise_guard_k, bool) or not isinstance(noise_guard_k, (int, float))
                or not math.isfinite(float(noise_guard_k)) or float(noise_guard_k) <= 1.0):
            raise ValueError("noise_guard_k must be a finite number > 1.0")
        self.noise_guard_k = float(noise_guard_k)
        noise_guard_action = config.get("noise_guard_action", "lowres")
        if noise_guard_action not in {"lowres", "deep_only", "lowres_deep_only", "denoise_nlm", "lowres_pool"}:
            raise ValueError("noise_guard_action must be one of: lowres, deep_only, lowres_deep_only, denoise_nlm, lowres_pool")
        self.noise_guard_action = noise_guard_action
        noise_guard_resolutions = config.get("noise_guard_resolutions", [448, 672])
        if not isinstance(noise_guard_resolutions, (list, tuple)) or not noise_guard_resolutions:
            raise ValueError("noise_guard_resolutions must be a non-empty sequence")
        self.noise_guard_resolutions = tuple(int(v) for v in noise_guard_resolutions)
        # Only the lowres actions route through this subset, so only they need it to exist.
        if (self.noise_guard and noise_guard_action in {"lowres", "lowres_deep_only", "lowres_pool"}
                and not set(self.noise_guard_resolutions).issubset(set(self.resolutions))):
            raise ValueError("noise_guard_resolutions must be a subset of resolutions")
        noise_guard_nlm_scale = config.get("noise_guard_nlm_scale", 1.0)
        if (isinstance(noise_guard_nlm_scale, bool) or not isinstance(noise_guard_nlm_scale, (int, float))
                or not math.isfinite(float(noise_guard_nlm_scale)) or float(noise_guard_nlm_scale) <= 0):
            raise ValueError("noise_guard_nlm_scale must be a finite number > 0")
        self.noise_guard_nlm_scale = float(noise_guard_nlm_scale)
        # Leave-One-Out (LOO) estimator mode: "source" uses leave-source-out
        # when >1 support, else within-image leave-token-out (default); "token"
        # uses within-image leave-token-out for EVERY record. Only "source" and
        # "token" are valid; "source" is the default.
        self.loo_estimator = str(config.get("loo_estimator", "source"))
        if self.loo_estimator not in {"source", "token"}:
            raise ValueError("loo_estimator must be 'source' or 'token'")
        # Feature pooling: optional spatial averaging of per-layer token features
        self.feature_pooling = str(config.get("feature_pooling", "none"))
        if self.feature_pooling not in {"none", "avg3"}:
            raise ValueError("feature_pooling must be 'none' or 'avg3'")
        declared = tuple(getattr(backbone, "feature_layers", ()))
        if declared and declared != self.layers:
            raise ValueError(f"backbone feature_layers must be {list(self.layers)} for {self.variant}")
        self.records: dict[int, list[dict[str, Any]]] = {}
        self.banks: dict[int, dict[str, Any]] = {}
        self.bag_banks: dict[int, list[dict[str, Any]]] = {}
        self.scales: dict[int, torch.Tensor] = {}
        self.bag_scales: dict[int, list[torch.Tensor]] = {}
        self.tails: dict[int, list[dict[str, Any]]] = {}
        self.bag_tails: dict[int, list[list[dict[str, Any]]]] = {}
        self._last_tails: list[dict[str, Any]] = []
        self._current_tails: Any = None
        # Per-bank-token leave-source-out distance, one 1-D tensor per layer,
        # aligned with the full (un-excluded) bank's token order. Only
        # populated when local_reliability != "off" (see
        # _fit_scales_for_resolution); the default path allocates nothing
        # extra.
        self.local: dict[int, list[torch.Tensor]] = {}
        self.bag_local: dict[int, list[list[torch.Tensor]]] = {}
        self._last_local: list[torch.Tensor] | None = None
        self._current_local: Any = None
        self.bank_trim_removed: dict[int, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.parameter_count = 0
        self.support_count = 0
        # Test seam: expose per-record LOO distances when loo_estimator=="token"
        self._test_token_mode_distances: dict[int, list[list[torch.Tensor]]] | None = None
        # Noise guard state
        self.noise_support_sigmas: list[float] = []
        self.noise_reference_sigma: float = 0.0
        self.noise_guard_log: list[dict[str, Any]] = []
        self._noise_suppress_shallow = False
        # Resolutions the last predict() actually scored; a fired lowres guard uses a subset.
        self.last_resolutions: tuple[int, ...] = self.resolutions
        # Pooled banks for lowres_pool noise guard action
        self.pooled_banks: dict[int, dict[str, Any]] = {}
        self.bag_pooled_banks: dict[int, list[dict[str, Any]]] = {}
        self.pooled_scales: dict[int, torch.Tensor] = {}
        self.bag_pooled_scales: dict[int, list[torch.Tensor]] = {}
        self.pooled_tails: dict[int, list[dict[str, Any]]] = {}
        self.bag_pooled_tails: dict[int, list[list[dict[str, Any]]]] = {}
        self.pooled_local: dict[int, list[torch.Tensor]] = {}
        self.bag_pooled_local: dict[int, list[list[torch.Tensor]]] = {}

    def _apply_feature_pooling(self, matrix: torch.Tensor, grid: tuple[int, int], enable: bool | None = None) -> torch.Tensor:
        """Apply optional spatial pooling to feature matrix.

        Input: [patches, dim] matrix where patches = H * W
        Grid: (H, W) spatial dimensions

        For avg3: reshape to spatial [1, dim, H, W], apply avg_pool2d(kernel=3, stride=1,
        padding=1, count_include_pad=False), reshape back to [patches, dim].

        If enable is None, uses self.feature_pooling. If enable is explicitly True/False,
        overrides and applies pooling accordingly.

        Returns: pooled matrix of same shape [patches, dim], or original if no pooling
        """
        # Determine pooling mode
        if enable is None:
            pooling_mode = self.feature_pooling
        else:
            pooling_mode = "avg3" if enable else "none"

        if pooling_mode == "none":
            return matrix
        if pooling_mode != "avg3":
            raise ValueError(f"Unknown feature_pooling: {pooling_mode}")

        # Reshape [patches, dim] to [1, dim, H, W]
        H, W = grid
        patches, dim = matrix.shape
        if patches != H * W:
            raise ValueError(f"Feature matrix patches ({patches}) != grid product ({H}*{W})")

        # Transpose for spatial layout: [1, dim, H, W]
        spatial = matrix.reshape(H, W, dim).permute(2, 0, 1).unsqueeze(0)  # [1, dim, H, W]

        # Apply 3x3 average pooling with stride 1, padding 1
        pooled = F.avg_pool2d(spatial, kernel_size=3, stride=1, padding=1, count_include_pad=False)

        # Reshape back to [patches, dim]
        result = pooled.squeeze(0).permute(1, 2, 0).reshape(patches, dim)  # [H, W, dim] -> [patches, dim]
        return result

    def _extract(self, image, resolution: int, pool=None):
        """Extract and optionally pool features before normalization.

        Args:
            image: Input image path
            resolution: Target resolution
            pool: If None, use self.feature_pooling. If True/False, override setting.
        """
        tensor, grid = self.backbone.prepare_image(image, resolution=resolution)
        with torch.no_grad():
            outputs = self.backbone.extract_features(tensor)
            declared = tuple(getattr(self.backbone, "feature_layers", self.layers))
            by_layer = {layer: output for layer, output in zip(declared, outputs)}
            if any(layer not in by_layer for layer in self.layers):
                raise ValueError(f"backbone did not return required layers {list(self.layers)}")
            layers = []
            for layer in self.layers:
                matrix, output_grid = _feature_matrix(by_layer[layer], grid)
                if tuple(output_grid) != tuple(grid):
                    raise ValueError("feature layer grids must match")
                # Apply feature pooling BEFORE normalization (respecting pool override)
                if pool is None:
                    matrix = self._apply_feature_pooling(matrix, output_grid)
                elif pool is True:
                    matrix = self._apply_feature_pooling(matrix, output_grid, enable=True)
                # If pool is False, do not apply pooling
                layers.append(F.normalize(matrix.to(self.device), dim=-1))
        return layers, tuple(grid)

    def _bank_for(self, resolution: int, excluded: int | None = None):
        records = [r for r in self.records[resolution] if r["source"] != excluded]
        if not records:
            raise ValueError("source-excluded fitting requires at least two supports")
        layers = [torch.cat([r["layers"][i] for r in records]).to(self.device)
                  for i in range(len(self.layers))]
        result = {"layers": layers, "owners": [r["source"] for r in records
                                                   for _ in range(len(r["indices"]))]}
        if self.variant == "R2":
            result["contexts"] = torch.cat([r["context"] for r in records]).to(self.device)
        return result

    def _fuse_layers(self, scores: list[torch.Tensor], tau: float | None = None) -> torch.Tensor:
        """Combine per-layer normalised score vectors within one bag (R1 only).

        This runs on whatever normalised per-layer scores score_normalization
        produced. max/quantile-over-layers is only meaningful once the layers
        are on a common scale: with "mad" each layer is divided by its own
        MAD, which standardises the CENTRE of its normal distribution but not
        its upper tail, so a layer with a fatter tail dominates the maximum;
        "tailscale" divides by the leave-source-out q95 instead, putting the
        layers in common FPR units so a max/quantile across them is fair.
        """
        if self.layer_fusion == "deep_plus_shallow":
            if len(scores) < 2:
                raise ValueError("deep_plus_shallow requires at least two feature layers")
            shallow_idx = min(self.layer_fusion_shallow_idx, len(scores) - 1)
            deep = [s for i, s in enumerate(scores) if i != shallow_idx]
            if self._noise_suppress_shallow:
                return torch.stack(deep).mean(0)
            return torch.stack(deep).mean(0) + self.layer_fusion_gamma * scores[shallow_idx]
        if self.layer_fusion == "residual_spike":
            if len(scores) < 2:
                return scores[0]
            shallow_idx = min(self.layer_fusion_shallow_idx, len(scores) - 1)
            shallow = scores[shallow_idx]
            deep = [s for i, s in enumerate(scores) if i != shallow_idx]
            base = torch.stack(deep).mean(0)
            # When noise_suppress_shallow is active (deep_only action), return base without spike
            if self._noise_suppress_shallow:
                return base
            threshold = tau if tau is not None else self.layer_fusion_tau
            delta = shallow - threshold
            spike = torch.relu(delta)
            if self.layer_fusion_neg_alpha > 0:
                # ELU-style bounded negative contribution in [-alpha, 0].
                spike = spike - self.layer_fusion_neg_alpha * (1.0 - torch.exp(torch.clamp(delta, max=0.0)))
            return base + self.layer_fusion_gamma * spike
        stacked = torch.stack(scores)
        if self.layer_fusion == "mean":
            return stacked.mean(0)
        if self.layer_fusion == "max":
            return stacked.max(0).values
        # "quantile": q=1.0 is defined to equal "max" (torch.quantile with
        # q=1.0 returns the maximum along the given dimension).
        return torch.quantile(stacked, self.layer_fusion_q, dim=0)

    def _weighted_fuse(self, scores: list[torch.Tensor], weights: list[torch.Tensor]) -> torch.Tensor:
        """Precision-weighted mean across layers, per query token (R1 "weight" mode).

        ``weights`` are the per-layer, per-token inverse-variance weights
        derived from ``local_scale``; this replaces (rather than composes
        with) ``_fuse_layers``, so it is only reachable when layer_fusion ==
        "mean" (enforced in the constructor).
        """
        stacked_scores = torch.stack(scores)
        stacked_weights = torch.stack(weights)
        return (stacked_scores * stacked_weights).sum(0) / stacked_weights.sum(0).clamp_min(1e-12)

    def _observe_layer_scores(self, distances, scores, *, tau):
        """Prediction-only observation hook; never receives evaluation labels."""

    def _image_gate_fuse(self, scores, bag_index):
        """Residual fusion with the image-level support threshold (no audit hooks)."""
        tail = self._active_tail(self.layer_fusion_shallow_idx, bag_index)
        return ReferenceMatching._fuse_layers(self, scores, tau=tail["image_spike_tau"])

    def _record_image_tokens(self, image_bag_scores, grid):
        if self._image_token_records is not None and image_bag_scores:
            self._image_token_records.append((torch.stack(image_bag_scores).mean(0), tuple(grid)))

    def _image_score(self, result, image) -> float:
        """Image score from the final map or the image-gate map; labels never enter."""
        records = self._image_token_records
        self._image_token_records = None
        candidates = {"max_final": float(np.max(result)),
                      "top1pct_mean_final": _top_fraction_mean(result)}
        if records is not None:
            if not records:
                raise RuntimeError("image gate produced no token scores")
            natives = []
            for token_score, grid in records:
                if self.scale_fusion == "rank":
                    token_score = rank_normalize(token_score)
                natives.append(F.interpolate(token_score.reshape(1, 1, *grid), size=result.shape[:2],
                                             mode="bilinear", align_corners=False)[0, 0])
            stacked = torch.stack(natives)
            total = stacked.max(0).values if self.scale_fusion == "max" else stacked.mean(0)
            image_map = ReferenceMatching._postprocess_prediction(self, total.detach().cpu().numpy(), image)
            if not np.isfinite(image_map).all():
                raise RuntimeError("nonfinite image-gate map")
            candidates["max_image_gate"] = float(np.max(image_map))
            candidates["top1pct_mean_image_gate"] = _top_fraction_mean(image_map)
        self._last_image_candidates = candidates
        return candidates[self.image_score_key]

    def predict_batch(self, test_image_paths):
        if not self.image_score_candidates:
            return super().predict_batch(test_image_paths)
        predictions = []
        for path in test_image_paths:
            started = time.perf_counter()
            score_map, image_score = self.predict(path)
            predictions.append(Prediction(np.asarray(score_map), float(image_score),
                                          time.perf_counter() - started,
                                          {"image_score_candidates": dict(self._last_image_candidates)}))
        return predictions

    def _score_tokens(self, layers, grid, bank, *, normalize=True):
        if self.variant == "R0":
            return _distance_chunks(layers[0], bank["layers"][0], self.chunk)
        if self.variant == "R1":
            # `bank` is a single bag dict in the common case, or a list of
            # per-bag bank dicts when bagging is active; averaging across
            # bags happens after each bag's own per-layer normalisation.
            bags = bank if isinstance(bank, list) else [bank]
            bag_scores = []
            image_bag_scores = []
            for bag_index, bag_bank in enumerate(bags):
                scores = []
                distances = []
                weights = [] if (normalize and self.local_reliability == "weight") else None
                for layer_index, (query, values) in enumerate(zip(layers, bag_bank["layers"])):
                    distance = _distance_chunks(query, values, self.chunk, k=self.knn_k)
                    distances.append(distance)
                    global_scale = self._active_scale(layer_index, values.device, bag_index)
                    sigma_local = None
                    if normalize and self.local_reliability != "off":
                        # local_reliability is purely a support-set property:
                        # `loo` came from the leave-source-out fit, no test
                        # data or labels are involved here.
                        loo = self._active_local(layer_index, bag_index).to(values.device)
                        neighbour_k = max(self.knn_k, self.local_reliability_k)
                        _, nn_indices = _distance_chunks(query, values, self.chunk,
                                                         k=neighbour_k, return_indices=True)
                        if self.local_reliability != "center":
                            sigma_local = local_scale(nn_indices, loo, global_scale,
                                                      self.local_reliability_k,
                                                      self.local_reliability_shrinkage)
                    if normalize and self.local_reliability == "center":
                        scores.append(self._local_center_score(
                            distance, nn_indices, loo, global_scale))
                    elif normalize and self.score_normalization == "tail":
                        # The tail CDF mapping already encodes the support's
                        # own distribution shape per token rank; local_scale
                        # calibrates a DIVISOR, which has no role here, so
                        # "scale" leaves this branch untouched. "weight" mode
                        # still uses sigma_local below, purely for fusion.
                        scores.append(self._tail_transform(
                            self._active_tail(layer_index, bag_index), distance))
                    elif normalize and sigma_local is not None:
                        scores.append(distance / sigma_local)
                    elif normalize and self.tailscale_center_rho:
                        scores.append(self._centered_tail_score(
                            distance, self._active_tail(layer_index, bag_index), global_scale))
                    else:
                        scores.append(distance / (global_scale if normalize else 1.0))
                    if weights is not None:
                        weights.append(1.0 / sigma_local.clamp_min(1e-4).pow(2))
                if weights is not None:
                    bag_scores.append(self._weighted_fuse(scores, weights))
                else:
                    tau = None
                    if self.layer_fusion == "residual_spike" and self.layer_fusion_tau_mode == "support_quantile":
                        active_tail_dict = self._active_tail(self.layer_fusion_shallow_idx, bag_index)
                        tau = active_tail_dict.get("spike_tau", self.layer_fusion_tau)
                    self._observe_layer_scores(distances, scores, tau=tau)
                    bag_scores.append(self._fuse_layers(scores, tau=tau))
                    if self._image_token_records is not None:
                        image_bag_scores.append(self._image_gate_fuse(scores, bag_index))
            self._record_image_tokens(image_bag_scores, grid)
            return torch.stack(bag_scores).mean(0)

        # R2: choose the reference with late appearance + ring context, then
        # reuse that identity for both feature-layer distances.
        query_context = _ring_context(layers[-1], grid)
        values_late = bank["layers"][-1]
        scores = []
        for start in range(0, len(layers[-1]), self.chunk):
            stop = start + self.chunk
            appearance = layers[-1][start:stop] @ values_late.T
            context = query_context[start:stop] @ bank["contexts"].T
            selected = (appearance + self.context_lambda * context).argmax(1)
            selected_scores = []
            for layer_index, query_layer in enumerate(layers):
                selected_values = bank["layers"][layer_index][selected]
                selected_scores.append(1.0 - (query_layer[start:stop] * selected_values).sum(1))
            chunk_score = torch.stack([
                value / (self._active_scale(i, value.device) if normalize else 1.0)
                for i, value in enumerate(selected_scores)
            ]).mean(0)
            scores.append(chunk_score)
        return torch.cat(scores)

    def _active_scale(self, layer_index: int, device, bag_index: int = 0) -> torch.Tensor:
        # _current_scales is set in predict and never contains query-derived
        # data. It is a single per-layer tensor unless bagging is active, in
        # which case it is a list of one tensor per bag.
        scales = self._current_scales
        if isinstance(scales, list):
            scales = scales[bag_index]
        return scales[layer_index].to(device)

    def _active_tail(self, layer_index: int, bag_index: int = 0) -> dict[str, Any]:
        tails = self._current_tails
        if tails and isinstance(tails[0], list):
            tails = tails[bag_index]
        return tails[layer_index]

    def _active_local(self, layer_index: int, bag_index: int = 0) -> torch.Tensor:
        # _current_local is a per-layer list of LOO distance tensors, or (when
        # bagging is active) a list of one such list per bag.
        local = self._current_local
        if local and isinstance(local[0], list):
            local = local[bag_index]
        return local[layer_index]

    def _local_center_score(self, distance, neighbour_indices, loo, global_scale):
        """Subtract support-LOO neighbourhood bias; retain the global q95 unit.

        The query selects neighbours only. Both location estimates are normal
        support statistics, and signed scores are retained without clipping.
        Alpha zero reproduces the original q95 score exactly.
        """
        if self.local_center_alpha == 0.0:
            return distance / global_scale
        sample = loo.float()
        local_mean = sample[neighbour_indices[:, :self.local_reliability_k]].mean(dim=1)
        correction = self.local_center_alpha * (local_mean - sample.mean())
        return (distance - correction) / global_scale

    def _fit_scales_for_resolution(self, resolution: int) -> torch.Tensor:
        """Leave-source-out MAD scale for the CURRENT self.records[resolution]."""
        values = [[] for _ in self.layers]
        num_sources = len(self.records[resolution])

        # Test seam: capture per-record distances for token mode validation
        per_record_distances = None
        if self.loo_estimator == "token":
            per_record_distances = [[] for _ in self.layers]

        if self.loo_estimator == "token":
            # Token mode: EVERY record uses within-image leave-token-out
            # Each record's tokens are compared against THAT RECORD'S OWN tokens only,
            # with the token's own diagonal entry masked (exactly as the 1-shot fallback).
            for record in self.records[resolution]:
                for index, layer in enumerate(record["layers"]):
                    # layer is CPU-staged; move to device for distance computation
                    q = layer.to(self.device)
                    # Distance matrix: each row is one token, columns are all tokens in this record
                    dist_matrix = 1.0 - q @ q.T
                    # Mask the diagonal (each token vs itself)
                    dist_matrix.fill_diagonal_(10.0)
                    # Nearest neighbor distance (excluding self)
                    distances = dist_matrix.amin(dim=1)
                    values[index].append(distances.detach().cpu())
                    # Store per-record distances for testing
                    if per_record_distances is not None:
                        per_record_distances[index].append(distances.detach().cpu())
        elif num_sources > 1:
            # Source mode with multiple supports: leave-source-out
            for record in self.records[resolution]:
                bank = self._bank_for(resolution, excluded=record["source"])
                for index, layer in enumerate(record["layers"]):
                    # Records are CPU-staged; transfer only this capped query
                    # to the bank device for leave-source-out calibration.
                    reference = bank["layers"][index]
                    distances = _distance_chunks(layer.to(reference.device), reference, self.chunk,
                                                 k=self.knn_k)
                    values[index].append(distances.detach().cpu())
        else:
            # 1-shot fallback: leave-token-out within the single support image
            record = self.records[resolution][0]
            bank = self._bank_for(resolution, excluded=None)
            for index, layer in enumerate(record["layers"]):
                reference = bank["layers"][index]
                q = layer.to(reference.device)
                dist_matrix = 1.0 - q @ reference.T
                dist_matrix.fill_diagonal_(10.0)
                distances = dist_matrix.amin(dim=1)
                values[index].append(distances.detach().cpu())
        scales = []
        tails = []
        local = [] if self.local_reliability != "off" else None
        for layer_values in values:
            # `sample` is, in the SAME order as `_bank_for(resolution)`
            # concatenates records, each record's leave-source-out nearest-
            # neighbour distance -- i.e. bank token i's LOO distance was
            # computed against a bank that EXCLUDES token i's own source
            # record, whereas the full bank (used at query time) INCLUDES
            # it. That mismatch is intentional: it is the honest, non-
            # self-matching estimate of how loose the normal model is
            # around that token, and is exactly what MAD/tail already use.
            sample = torch.cat(layer_values)
            median = sample.median()
            mad = (sample - median).abs().median()
            mad_scale = (1.4826 * mad).clamp_min(1e-4)
            tail_dict = self._tail_params(sample)
            self._set_tail_center(tail_dict)
            if self.layer_fusion == "residual_spike" and self.layer_fusion_tau_mode == "support_quantile":
                if self.score_normalization == "tailscale":
                    divisor = self._tailscale_divisor(tail_dict).item()
                elif self.score_normalization == "none":
                    divisor = 1.0
                else:
                    divisor = mad_scale.item()
                q_val = torch.quantile(sample.detach().float().cpu(), self.layer_fusion_tau_q).item()
                offset = float(tail_dict.get("center_offset", 0.0))
                tail_dict["spike_tau"] = float((q_val - offset) / max(divisor, 1e-4))
                if self.image_residual_tau_q is not None:
                    q_img = torch.quantile(sample.detach().float().cpu(), self.image_residual_tau_q).item()
                    tail_dict["image_spike_tau"] = float((q_img - offset) / max(divisor, 1e-4))
            tails.append(tail_dict)
            if self.score_normalization == "tailscale":
                # Normalise by the component's own normal tail instead of its
                # centre, so every component contributes the same share of the
                # FPR budget that AUPRO@0.05 actually operates in.
                scales.append(self._tailscale_divisor(tail_dict))
            elif self.score_normalization == "none":
                # "none" means raw distance with divisor 1 (uncalibrated)
                scales.append(torch.ones_like(mad_scale))
            else:
                scales.append(mad_scale)
            if local is not None:
                local.append(sample)
        self._last_tails = tails
        self._last_local = local
        # Store per-record distances for testing
        if per_record_distances is not None:
            if self._test_token_mode_distances is None:
                self._test_token_mode_distances = {}
            self._test_token_mode_distances[resolution] = per_record_distances
        return torch.stack(scales)

    def _set_tail_center(self, params):
        if self.tailscale_center_rho:
            params["center_offset"] = self.tailscale_center_rho * torch.quantile(params["sample"], 0.5)

    @staticmethod
    def _tailscale_divisor(params):
        return (params["u"] - params.get("center_offset", 0.0)).clamp_min(1e-4)

    @staticmethod
    def _centered_tail_score(distance, params, scale):
        # Preserve negative scores: clipping would introduce a separate gate.
        return (distance - params["center_offset"].to(distance.device)) / scale

    def _tail_params(self, sample: torch.Tensor) -> dict[str, Any]:
        """Support-only tail description of one component's normal distances.

        `u` is the tail_quantile of the leave-source-out normal sample and
        `beta` the mean excess above it, i.e. an exponential fit to the tail
        used to extrapolate beyond the largest observed normal distance.
        """
        sample = sample.detach().float().flatten().cpu()
        u = torch.quantile(sample, self.tail_quantile)
        excess = sample[sample > u] - u
        beta = excess.mean() if excess.numel() > 0 else sample.std()
        if not torch.isfinite(beta) or beta <= 0:
            beta = torch.tensor(1e-4)
        return {"sample": torch.sort(sample).values, "u": u,
                "beta": beta.clamp_min(1e-6), "p_u": torch.tensor(1.0 - self.tail_quantile)}

    def _tail_transform(self, params: dict[str, Any], distance: torch.Tensor) -> torch.Tensor:
        """Map distances onto -log(1 - F) using the support's own normal CDF.

        Below `u` the empirical CDF is used (floored so the log stays finite);
        above it the exponential tail fit extrapolates, which keeps the mapping
        monotone and linear in the excess so magnitude is not discarded.
        """
        device = distance.device
        sample = params["sample"].to(device)
        u, beta, p_u = (params[k].to(device) for k in ("u", "beta", "p_u"))
        n = sample.numel()
        rank = torch.searchsorted(sample, distance.contiguous(), right=True)
        survival = (1.0 - rank.to(distance.dtype) / n).clamp_min(1.0 / (n + 1))
        body = -torch.log(survival)
        tail = -torch.log(p_u) + (distance - u) / beta
        return torch.where(distance > u, tail, body)

    def fit(self, support_paths: list[str], context) -> None:
        if context.masking or context.rotation:
            raise ValueError("reference_matching requires no_mask_no_rotation")
        if len(support_paths) < 1 or len(set(support_paths)) != len(support_paths):
            raise ValueError("need at least one unique support")
        torch.manual_seed(context.seed)
        images = [_read_rgb(path) for path in support_paths]
        self.support_count = len(images)
        base_seed = context.seed if self.bank_seed is None else self.bank_seed
        # Compute noise sigmas for all support images if noise_guard is enabled
        if self.noise_guard:
            self.noise_support_sigmas = [immerkaer_sigma(img) for img in images]
            self.noise_reference_sigma = float(np.median(self.noise_support_sigmas))
        self.noise_guard_log = []

        self.banks = {}
        self.bag_banks = {}
        self.scales = {}
        self.bag_scales = {}
        self.local = {}
        self.bag_local = {}
        trim_removed: dict[int, Any] = {}
        # One RNG per bag, created once and reused across resolutions -- this
        # matches the pre-bagging behaviour where a single RNG's draws for
        # bank_selection == "random" advanced across the whole resolution
        # loop, so bag 0 stays byte-identical to the original single-bank path.
        rngs = [np.random.default_rng(base_seed + 1000003 * bag) for bag in range(self.bank_bags)]

        for resolution in self.resolutions:
            # Extract once per support per resolution; bags only resample the
            # token subset drawn from these already-extracted features.
            extracted = [self._extract(image, resolution) for image in images]
            bag_banks = []
            bag_scales = []
            bag_tails = []
            bag_local = []
            bag_removed = []
            for bag in range(self.bank_bags):
                rng = rngs[bag]
                records = []
                for source, (layers, grid) in enumerate(extracted):
                    if self.bank_selection == "coreset":
                        identity = F.normalize(torch.cat(layers, dim=1), dim=-1)
                        indices = coreset_indices(identity, min(self.cap, len(layers[0])))
                    else:
                        indices = torch.as_tensor(
                            rng.choice(len(layers[0]), min(self.cap, len(layers[0])), replace=False),
                            device=self.device)
                    record = {"source": source,
                             "layers": [x[indices].cpu() for x in layers],
                             "grid": grid, "indices": indices.cpu()}
                    if self.variant == "R2":
                        record["context"] = _ring_context(layers[-1], grid)[indices].cpu()
                    records.append(record)
                before_total = sum(len(record["indices"]) for record in records)
                if self.bank_outlier_trim > 0.0:
                    records = trim_outlier_tokens(records, self.chunk, self.bank_outlier_trim,
                                                  device=self.device)
                after_total = sum(len(record["indices"]) for record in records)
                bag_removed.append(before_total - after_total)

                self.records = {resolution: records}
                bag_banks.append(self._bank_for(resolution))
                if self.robust_scale:
                    bag_scales.append(self._fit_scales_for_resolution(resolution))
                    bag_tails.append(self._last_tails)
                    if self.local_reliability != "off":
                        bag_local.append(self._last_local)

            self.bag_banks[resolution] = bag_banks
            self.banks[resolution] = bag_banks[0]
            if self.robust_scale:
                self.bag_scales[resolution] = bag_scales
                self.scales[resolution] = bag_scales[0]
                self.bag_tails[resolution] = bag_tails
                self.tails[resolution] = bag_tails[0]
                if self.local_reliability != "off":
                    self.bag_local[resolution] = bag_local
                    self.local[resolution] = bag_local[0]
            trim_removed[resolution] = bag_removed if self.bank_bags > 1 else bag_removed[0]

        self.bank_trim_removed = trim_removed
        self.records = {}
        self.history = []

        # Create pooled banks when noise_guard is enabled with lowres_pool action
        if self.noise_guard and self.noise_guard_action == "lowres_pool":
            self.pooled_banks = {}
            self.bag_pooled_banks = {}
            self.pooled_scales = {}
            self.bag_pooled_scales = {}
            self.pooled_tails = {}
            self.bag_pooled_tails = {}
            self.pooled_local = {}
            self.bag_pooled_local = {}
            # Match a direct feature_pooling="avg3" fit: each bag's random
            # selection advances across all resolutions in order.
            pooled_rngs = [np.random.default_rng(base_seed + 1000003 * bag)
                           for bag in range(self.bank_bags)]

            for resolution in self.noise_guard_resolutions:
                if resolution not in self.resolutions:
                    continue  # Skip resolutions not in the main set

                # Re-extract with pool=True to apply pooling BEFORE normalization (global semantics)
                extracted_pooled = []
                for image in images:
                    # pool=True applies pooling inside _extract before normalization
                    layers_pooled, grid = self._extract(image, resolution, pool=True)
                    extracted_pooled.append((layers_pooled, grid))

                # Create pooled banks using the same per-bag RNG stream as a
                # directly configured pooled model.
                bag_pooled_banks = []
                bag_pooled_scales = []
                bag_pooled_tails = []
                bag_pooled_local = []

                for bag in range(self.bank_bags):
                    rng_pooled = pooled_rngs[bag]
                    records_pooled = []
                    for source, (layers_pooled, grid) in enumerate(extracted_pooled):
                        if self.bank_selection == "coreset":
                            identity = F.normalize(torch.cat(layers_pooled, dim=1), dim=-1)
                            indices = coreset_indices(identity, min(self.cap, len(layers_pooled[0])))
                        else:
                            indices = torch.as_tensor(
                                rng_pooled.choice(len(layers_pooled[0]), min(self.cap, len(layers_pooled[0])), replace=False),
                                device=self.device)
                        record_pooled = {"source": source,
                                        "layers": [x[indices].cpu() for x in layers_pooled],
                                        "grid": grid, "indices": indices.cpu()}
                        if self.variant == "R2":
                            record_pooled["context"] = _ring_context(layers_pooled[-1], grid)[indices].cpu()
                        records_pooled.append(record_pooled)

                    # Trim outliers from pooled records
                    before_total = sum(len(record["indices"]) for record in records_pooled)
                    if self.bank_outlier_trim > 0.0:
                        records_pooled = trim_outlier_tokens(records_pooled, self.chunk, self.bank_outlier_trim,
                                                           device=self.device)

                    self.records = {resolution: records_pooled}
                    bag_pooled_banks.append(self._bank_for(resolution))
                    if self.robust_scale:
                        bag_pooled_scales.append(self._fit_scales_for_resolution(resolution))
                        bag_pooled_tails.append(self._last_tails)
                        if self.local_reliability != "off":
                            bag_pooled_local.append(self._last_local)

                self.bag_pooled_banks[resolution] = bag_pooled_banks
                self.pooled_banks[resolution] = bag_pooled_banks[0]
                if self.robust_scale:
                    self.bag_pooled_scales[resolution] = bag_pooled_scales
                    self.pooled_scales[resolution] = bag_pooled_scales[0]
                    self.bag_pooled_tails[resolution] = bag_pooled_tails
                    self.pooled_tails[resolution] = bag_pooled_tails[0]
                    if self.local_reliability != "off":
                        self.bag_pooled_local[resolution] = bag_pooled_local
                        self.pooled_local[resolution] = bag_pooled_local[0]

            self.records = {}

    def predict(self, test_image_path: str) -> tuple[np.ndarray, float]:
        image = _read_rgb(test_image_path)
        self._image_token_records = [] if self.image_residual_tau_q is not None else None
        # Noise guard: check if query is too noisy
        sigma_q = None
        fired = False
        if self.noise_guard:
            sigma_q = immerkaer_sigma(image)
            # Use a small positive floor only when the support median is
            # exactly noiseless, keeping ratios finite for JSON diagnostics.
            # Two noiseless estimates are equal and leave the guard off.
            if self.noise_reference_sigma == 0.0:
                ratio = sigma_q / 1e-12 if sigma_q > 0.0 else 1.0
            else:
                ratio = sigma_q / self.noise_reference_sigma
            fired = ratio > self.noise_guard_k
            self.noise_guard_log.append({
                "image": test_image_path,
                "sigma": sigma_q,
                "ratio": ratio,
                "fired": fired
            })
        # Apply noise guard action if fired
        if fired:
            action = self.noise_guard_action
            denoised_image = image
            if action == "denoise_nlm":
                if cv2 is None:
                    raise RuntimeError("cv2 (OpenCV) is required for denoise_nlm action")
                # NLM filtering strength tracks the estimated noise level (8-bit units).
                h = float(self.noise_guard_nlm_scale * sigma_q)
                # OpenCV's colour NLM converts from BGR to Lab, so reorder around the call.
                bgr = np.ascontiguousarray(image[..., ::-1])
                denoised = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 7, 21)
                denoised_image = np.ascontiguousarray(denoised[..., ::-1])
            try:
                if action in {"deep_only", "lowres_deep_only"}:
                    self._noise_suppress_shallow = True
                # Determine which resolutions to use and whether to use pooled banks
                use_pooled = action == "lowres_pool"
                if action in {"lowres", "lowres_deep_only", "lowres_pool"}:
                    resolutions_to_use = [r for r in self.resolutions if r in self.noise_guard_resolutions]
                else:
                    resolutions_to_use = list(self.resolutions)
                self.last_resolutions = tuple(resolutions_to_use)
                total = None
                bagged = self.variant == "R1" and self.bank_bags > 1
                for resolution in resolutions_to_use:
                    # Extract with pooling when lowres_pool action (pool=True applies global semantics)
                    if use_pooled:
                        layers, grid = self._extract(denoised_image, resolution, pool=True)
                    else:
                        layers, grid = self._extract(denoised_image, resolution)

                    if use_pooled:
                        # Use pooled banks (no additional pooling needed, already applied in _extract)
                        if bagged:
                            self._current_scales = self.bag_pooled_scales.get(
                                resolution, [torch.ones(len(self.layers))] * self.bank_bags)
                            self._current_tails = self.bag_pooled_tails.get(resolution)
                            self._current_local = self.bag_pooled_local.get(resolution)
                            bank = self.bag_pooled_banks.get(resolution)
                        else:
                            self._current_scales = self.pooled_scales.get(resolution, torch.ones(len(self.layers)))
                            self._current_tails = self.pooled_tails.get(resolution)
                            self._current_local = self.pooled_local.get(resolution)
                            bank = self.pooled_banks.get(resolution)
                        if bank is None:
                            raise ValueError(f"pooled bank missing for resolution {resolution} with lowres_pool action")
                    else:
                        if bagged:
                            self._current_scales = self.bag_scales.get(
                                resolution, [torch.ones(len(self.layers))] * self.bank_bags)
                            self._current_tails = self.bag_tails.get(resolution)
                            self._current_local = self.bag_local.get(resolution)
                            bank = self.bag_banks[resolution]
                        else:
                            self._current_scales = self.scales.get(resolution, torch.ones(len(self.layers)))
                            self._current_tails = self.tails.get(resolution)
                            self._current_local = self.local.get(resolution)
                            bank = self.banks[resolution]
                    score = self._score_tokens(layers, grid, bank)
                    if self.scale_fusion == "rank":
                        score = rank_normalize(score)
                    native = F.interpolate(score.reshape(1, 1, *grid), size=denoised_image.shape[:2],
                                           mode="bilinear", align_corners=False)[0, 0]
                    if self.scale_fusion == "max":
                        total = native if total is None else torch.maximum(total, native)
                    else:
                        total = native if total is None else total + native
                # Average only over the resolutions we used
                if total is not None:
                    if self.scale_fusion == "max":
                        result = total.detach().cpu().numpy()
                    else:
                        result = (total / len(resolutions_to_use)).detach().cpu().numpy()
                else:
                    # Fallback if all resolutions were skipped
                    result = np.zeros(denoised_image.shape[:2])
            finally:
                self._noise_suppress_shallow = False
        else:
            # Standard prediction path when noise_guard is False or not fired
            self.last_resolutions = self.resolutions
            total = None
            bagged = self.variant == "R1" and self.bank_bags > 1
            for resolution in self.resolutions:
                layers, grid = self._extract(image, resolution)
                if bagged:
                    self._current_scales = self.bag_scales.get(
                        resolution, [torch.ones(len(self.layers))] * self.bank_bags)
                    self._current_tails = self.bag_tails.get(resolution)
                    self._current_local = self.bag_local.get(resolution)
                    bank = self.bag_banks[resolution]
                else:
                    self._current_scales = self.scales.get(resolution, torch.ones(len(self.layers)))
                    self._current_tails = self.tails.get(resolution)
                    self._current_local = self.local.get(resolution)
                    bank = self.banks[resolution]
                score = self._score_tokens(layers, grid, bank)
                if self.scale_fusion == "rank":
                    # Per-resolution scores are calibrated by different LOO-MAD
                    # scales; rank-normalising each resolution's token-grid
                    # score before averaging removes residual scale bias.
                    score = rank_normalize(score)
                native = F.interpolate(score.reshape(1, 1, *grid), size=image.shape[:2],
                                       mode="bilinear", align_corners=False)[0, 0]
                if self.scale_fusion == "max":
                    total = native if total is None else torch.maximum(total, native)
                else:
                    total = native if total is None else total + native
            if self.scale_fusion == "max":
                result = total.detach().cpu().numpy()
            else:
                result = (total / len(self.resolutions)).detach().cpu().numpy()
        if not np.isfinite(result).all():
            raise RuntimeError("nonfinite reference_matching prediction")
        result = self._postprocess_prediction(result, image)
        return result, self._image_score(result, image)

    def _postprocess_prediction(self, result, image):
        """Prediction-only hook; default preserves the original CQ operation."""
        if self.prediction_postprocess == "component_quantile":
            result = component_quantile(
                result,
                percentile=self.component_percentile,
                q=self.component_q,
                dilate_size=self.component_dilate_size,
            )
        return result

    def object_metadata(self) -> dict:
        if self.bank_selection == "coreset":
            seed_source = "deterministic_coreset_no_seed"
        elif self.bank_seed is None:
            seed_source = "context_seed"
        else:
            seed_source = "bank_seed_override_same_supports"
        bagged = self.bank_bags > 1
        if bagged:
            seed_source = "bagged_random_subsamples"
        score_definition = "late_layer_cosine_nn" if self.variant == "R0" else \
            "mean_layer_distance_with_support_loo_mad"
        if bagged:
            score_definition = "bag_mean_layer_distance_with_support_loo_mad"
        # Suffix order is fixed (scale first, then layer) so score_definition
        # stays deterministic when both scale_fusion and layer_fusion are
        # non-default at once.
        if self.scale_fusion != "mean":
            score_definition = f"{score_definition}_scale_fusion_{self.scale_fusion}"
        if self.layer_fusion != "mean":
            score_definition = f"{score_definition}_layer_fusion_{self.layer_fusion}"
        # Local reliability suffix comes LAST, after the scale/layer fusion
        # suffixes above, since it is an orthogonal calibration/fusion choice
        # layered on top of whatever those two produced.
        if self.local_reliability != "off":
            score_definition = f"{score_definition}_local_{self.local_reliability}"
        if self.prediction_postprocess != "none":
            score_definition = f"{score_definition}_postprocess_{self.prediction_postprocess}"
        # Feature pooling suffix comes before noise guard
        if self.feature_pooling != "none":
            score_definition = f"{score_definition}_pool_{self.feature_pooling}"
        # Noise guard suffix comes at the very end
        if self.noise_guard:
            score_definition = f"{score_definition}_noise_guard_{self.noise_guard_action}"
        # LOO estimator suffix comes after noise guard if present
        if self.loo_estimator == "token":
            score_definition = f"{score_definition}_loo_token"
        metadata = {
            "variant": self.variant,
            "feature_layers": list(self.layers),
            "resolutions": list(self.resolutions),
            "bank_tokens_per_source": self.cap,
            "bank_seed": self.bank_seed,
            "bank_sampling_seed_source": seed_source,
            "bank_selection": self.bank_selection,
            "knn_k": self.knn_k,
            "query_chunk_size": self.chunk,
            "context_lambda": self.context_lambda if self.variant == "R2" else None,
            "normalization": "positive_support_loo_mad" if self.robust_scale else "none",
            "normalization_reference": (
                "independent_layer_nn_leave_source_out_fixed_across_R1_R2"
                if self.robust_scale else "not_applicable_R0"
            ),
            "reference_identity_shared_across_layers": self.variant == "R2",
            "backbone_frozen": True,
            "trainable_parameters": 0,
            "support_count": self.support_count,
            "dense_maps_accumulated": False,
            "steps_completed": 0,
            "bank_sizes": {str(r): len(b["layers"][0]) for r, b in self.banks.items()},
            "robust_scales": {str(r): [float(v) for v in scale]
                              for r, scale in self.scales.items()},
            "score_definition": score_definition,
            "bank_outlier_trim": self.bank_outlier_trim,
            "bank_bags": self.bank_bags,
            "bank_trim_removed": {str(r): v for r, v in self.bank_trim_removed.items()},
            "scale_fusion": self.scale_fusion,
            "score_normalization": self.score_normalization,
            "tail_quantile": self.tail_quantile,
            "tailscale_center_rho": self.tailscale_center_rho,
            "layer_fusion": self.layer_fusion,
            "layer_fusion_q": self.layer_fusion_q,
            "layer_fusion_tau": self.layer_fusion_tau,
            "layer_fusion_gamma": self.layer_fusion_gamma,
            "layer_fusion_shallow_idx": self.layer_fusion_shallow_idx,
            "layer_fusion_neg_alpha": self.layer_fusion_neg_alpha,
            "image_score_aggregation": self.image_score_aggregation,
            "image_score_source": self.image_score_source,
            "image_residual_tau_q": self.image_residual_tau_q,
            "image_score_key": self.image_score_key,
            "local_reliability": self.local_reliability,
            "local_center_alpha": self.local_center_alpha,
            "local_reliability_k": self.local_reliability_k,
            "local_reliability_shrinkage": self.local_reliability_shrinkage,
            "prediction_postprocess": self.prediction_postprocess,
            "component_percentile": self.component_percentile,
            "component_q": self.component_q,
            "component_dilate_size": self.component_dilate_size,
            "feature_pooling": self.feature_pooling,
        }
        # Add loo_estimator metadata only when "token" mode
        if self.loo_estimator == "token":
            metadata["loo_estimator"] = "token"
        # Add noise guard metadata only when enabled
        if self.noise_guard:
            metadata["noise_guard"] = {
                "k": self.noise_guard_k,
                "action": self.noise_guard_action,
                "resolutions": list(self.noise_guard_resolutions),
                "nlm_scale": self.noise_guard_nlm_scale,
                "reference_sigma": self.noise_reference_sigma,
                "support_sigmas": self.noise_support_sigmas,
            }
            # Compute summary statistics from the log
            if self.noise_guard_log:
                ratios = [entry["ratio"] for entry in self.noise_guard_log]
                metadata["noise_guard_summary"] = {
                    "queries": len(self.noise_guard_log),
                    "fired": sum(1 for entry in self.noise_guard_log if entry["fired"]),
                    "ratio_min": float(np.min(ratios)),
                    "ratio_median": float(np.median(ratios)),
                    "ratio_max": float(np.max(ratios)),
                }
        return metadata
