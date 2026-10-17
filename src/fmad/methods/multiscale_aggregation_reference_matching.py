"""Statistical multi-scale aggregation on Q95-calibrated ScopeAD maps.

Tests whether signal dilution across resolutions limits AUPRO@0.05 when
individual observations are already Q95-normalized. All support banks,
sampled tokens, Q95 divisors, and backbone features remain identical to
StatisticalReferenceMatching(statistical_normalization='q95').
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F

from fmad.methods.reference_matching import _read_rgb, immerkaer_sigma
from fmad.methods.statistical_reference_matching import StatisticalReferenceMatching
from fmad.registry import METHOD_REGISTRY


AGGREGATION_MODES = (
    "mean",
    "scale_lse",
    "scale_top3",
    "spatial_contrast_top3",
    "spatial_contrast_order_linear",
    "scale_order_renyi",
    "spatial_contrast_order_renyi",
    "scale_order_linear",
    "scale_order_convex",
    "scale_power_mean_p15",
    "scale_power_mean_p20",
    "scale_power_top3_p15",
    "scale_power_top3_p20",
)

DEEP_LAYER_MODES = (
    "mean",
    "power_p15",
    "power_p20",
    "order_linear",
)


def combine_resolution_maps(
    maps: list[torch.Tensor],
    mode: str,
    lse_beta: float = 2.0,
) -> torch.Tensor:
    """Combine native-image-sized resolution maps [H, W] without labels."""
    if not maps:
        raise ValueError("maps must not be empty")
    if mode not in AGGREGATION_MODES:
        raise ValueError(f"unknown aggregation mode: {mode}")
    stacked = torch.stack(maps, dim=0)  # [R, H, W]
    if stacked.ndim != 3 or not torch.isfinite(stacked).all():
        raise ValueError("expected finite [R, H, W] resolution score maps")
    r_count = stacked.shape[0]
    if r_count == 1 or mode == "mean":
        return stacked.mean(dim=0)

    if mode == "scale_lse":
        if lse_beta <= 0 or not math.isfinite(lse_beta):
            raise ValueError("lse_beta must be positive and finite")
        return (torch.logsumexp(lse_beta * stacked, dim=0) - math.log(r_count)) / lse_beta

    if mode == "spatial_contrast_top3":
        # Align each resolution's spatial background median while preserving
        # the image's mean background level across resolutions.
        flat = stacked.reshape(r_count, -1)
        medians = flat.median(dim=1).values[:, None, None]
        mean_median = medians.mean(dim=0, keepdim=True)
        stacked = stacked - medians + mean_median

    if mode in {"spatial_contrast_order_linear", "spatial_contrast_order_renyi"}:
        flat = stacked.reshape(r_count, -1)
        medians = flat.median(dim=1).values[:, None, None]
        mean_median = medians.mean(dim=0, keepdim=True)
        stacked = stacked - medians + mean_median

    if mode in {"scale_order_linear", "scale_order_convex", "scale_order_renyi", "spatial_contrast_order_linear", "spatial_contrast_order_renyi"}:
        sorted_maps = torch.sort(stacked, dim=0, descending=False).values
        if mode in {"scale_order_linear", "spatial_contrast_order_linear"}:
            # Order weights [0.1, 0.2, 0.3, 0.4] for 4 resolutions
            weights = torch.linspace(0.1, 0.4, steps=r_count, device=stacked.device, dtype=stacked.dtype)
        elif mode in {"scale_order_renyi", "spatial_contrast_order_renyi"}:
            # Rényi spacing weights for standard exponential order statistics [1/4, 1/4+1/3, 1/4+1/3+1/2, 1/4+1/3+1/2+1]
            # For r_count=4: [0.25, 0.5833, 1.0833, 2.0833] -> normalized: [0.0625, 0.1458, 0.2708, 0.5208]
            renyi_spacings = [sum(1.0 / (r_count - j) for j in range(k)) for k in range(1, r_count + 1)]
            weights = torch.tensor(renyi_spacings, device=stacked.device, dtype=stacked.dtype)
        else:
            # Order weights [0.05, 0.25, 0.35, 0.35]
            weights = torch.tensor([0.05, 0.25, 0.35, 0.35], device=stacked.device, dtype=stacked.dtype)
        weights = (weights / weights.sum())[:, None, None]
        return (sorted_maps * weights).sum(dim=0)

    if mode in {"scale_power_mean_p15", "scale_power_mean_p20"}:
        p = 1.5 if mode == "scale_power_mean_p15" else 2.0
        pos_stacked = torch.clamp_min(stacked, 0.0)
        return torch.pow(torch.pow(pos_stacked, p).mean(dim=0), 1.0 / p)

    if mode in {"scale_power_top3_p15", "scale_power_top3_p20"}:
        p = 1.5 if mode == "scale_power_top3_p15" else 2.0
        pos_stacked = torch.clamp_min(stacked, 0.0)
        k = max(1, r_count - 1)
        top_k = torch.topk(pos_stacked, k=k, dim=0, largest=True).values
        return torch.pow(torch.pow(top_k, p).mean(dim=0), 1.0 / p)

    # Lower-trimmed mean (top-(R-1) order statistics across resolutions)
    k = max(1, r_count - 1)
    top_k = torch.topk(stacked, k=k, dim=0, largest=True).values
    return top_k.mean(dim=0)


@METHOD_REGISTRY.register("multiscale_aggregation_reference_matching")
class MultiscaleAggregationReferenceMatching(StatisticalReferenceMatching):
    def __init__(self, backbone, config: dict):
        self.multiscale_aggregation = str(config.get("multiscale_aggregation", "mean"))
        if self.multiscale_aggregation not in AGGREGATION_MODES:
            raise ValueError(f"multiscale_aggregation must be one of {AGGREGATION_MODES}")
        self.multiscale_lse_beta = float(config.get("multiscale_lse_beta", 2.0))
        if not math.isfinite(self.multiscale_lse_beta) or self.multiscale_lse_beta <= 0:
            raise ValueError("multiscale_lse_beta must be finite and > 0")
        self.deep_layer_fusion = str(config.get("deep_layer_fusion", "mean"))
        if self.deep_layer_fusion not in DEEP_LAYER_MODES:
            raise ValueError(f"deep_layer_fusion must be one of {DEEP_LAYER_MODES}")
        super().__init__(backbone, config)
        if (
            self.statistical_normalization not in {"q95", "none"}
            or self.scale_fusion != "mean"
            or self.calibration_audit
            or self.noise_guard_action == "lowres_pool"
            or self.image_residual_tau_q is not None
        ):
            raise ValueError(
                "multiscale aggregation requires Q95/none settings without pooled/image/audit overrides"
            )

    def _fuse_layers(self, scores, tau=None):
        if self.deep_layer_fusion == "mean":
            return super()._fuse_layers(scores, tau=tau)
        if len(scores) < 2:
            raise ValueError("shallow ablation requires at least two layers")
        index = min(self.layer_fusion_shallow_idx, len(scores) - 1)
        shallow = scores[index]
        deep_tensors = [s for i, s in enumerate(scores) if i != index]
        stacked_deep = torch.stack(deep_tensors, dim=0) # [2, N]
        if self.deep_layer_fusion in {"power_p15", "power_p20"}:
            p = 1.5 if self.deep_layer_fusion == "power_p15" else 2.0
            pos = torch.clamp_min(stacked_deep, 0.0)
            deep = torch.pow(torch.pow(pos, p).mean(dim=0), 1.0 / p)
        elif self.deep_layer_fusion == "order_linear":
            # Order weights [0.35, 0.65] for min and max deep layer
            s_min = torch.minimum(stacked_deep[0], stacked_deep[1])
            s_max = torch.maximum(stacked_deep[0], stacked_deep[1])
            deep = 0.35 * s_min + 0.65 * s_max
        else:
            deep = stacked_deep.mean(0)
        if self._noise_suppress_shallow or self.ablation_shallow_mode == "off":
            return deep
        contribution = shallow
        if self.ablation_shallow_mode == "hard":
            threshold = self.layer_fusion_tau if tau is None else tau
            contribution = shallow * (shallow > threshold)
        return deep + self.layer_fusion_gamma * contribution

    def _predict_resolutions(self, image: np.ndarray, resolutions_to_use: Sequence[int]) -> np.ndarray:
        bagged = self.variant == "R1" and self.bank_bags > 1
        maps: list[torch.Tensor] = []
        for resolution in resolutions_to_use:
            layers, grid = self._extract(image, resolution)
            if bagged:
                self._current_scales = self.bag_scales.get(
                    resolution, [torch.ones(len(self.layers))] * self.bank_bags
                )
                self._current_tails = self.bag_tails.get(resolution)
                self._current_local = self.bag_local.get(resolution)
                bank = self.bag_banks[resolution]
            else:
                self._current_scales = self.scales.get(resolution, torch.ones(len(self.layers)))
                self._current_tails = self.tails.get(resolution)
                self._current_local = self.local.get(resolution)
                bank = self.banks[resolution]
            score = self._score_tokens(layers, grid, bank)
            native = F.interpolate(
                score.reshape(1, 1, *grid),
                size=image.shape[:2],
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            maps.append(native)
        combined = self._combine_maps(maps, resolutions_to_use)
        return combined.detach().cpu().numpy()

    def _combine_maps(self, maps, resolutions_to_use):
        return combine_resolution_maps(
            maps,
            mode=self.multiscale_aggregation,
            lse_beta=self.multiscale_lse_beta,
        )

    def predict(self, test_image_path: str) -> tuple[np.ndarray, float]:
        if self.multiscale_aggregation == "mean":
            return super().predict(test_image_path)

        image = _read_rgb(test_image_path)
        self._image_token_records = None
        sigma_q = None
        fired = False
        if self.noise_guard:
            sigma_q = immerkaer_sigma(image)
            if self.noise_reference_sigma == 0.0:
                ratio = sigma_q / 1e-12 if sigma_q > 0.0 else 1.0
            else:
                ratio = sigma_q / self.noise_reference_sigma
            fired = ratio > self.noise_guard_k
            self.noise_guard_log.append(
                {"image": test_image_path, "sigma": sigma_q, "ratio": ratio, "fired": fired}
            )

        if fired:
            action = self.noise_guard_action
            denoised_image = image
            try:
                if action in {"deep_only", "lowres_deep_only"}:
                    self._noise_suppress_shallow = True
                if action in {"lowres", "lowres_deep_only"}:
                    resolutions_to_use = [
                        r for r in self.resolutions if r in self.noise_guard_resolutions
                    ]
                else:
                    resolutions_to_use = list(self.resolutions)
                self.last_resolutions = tuple(resolutions_to_use)
                if resolutions_to_use:
                    result = self._predict_resolutions(denoised_image, resolutions_to_use)
                else:
                    result = np.zeros(denoised_image.shape[:2])
            finally:
                self._noise_suppress_shallow = False
        else:
            self.last_resolutions = self.resolutions
            result = self._predict_resolutions(image, self.resolutions)

        if not np.isfinite(result).all():
            raise RuntimeError("nonfinite multiscale_aggregation_reference_matching prediction")
        result = self._postprocess_prediction(result, image)
        return result, self._image_score(result, image)

    def object_metadata(self) -> dict:
        metadata = super().object_metadata()
        metadata.update(
            multiscale_aggregation=self.multiscale_aggregation,
            multiscale_lse_beta=(
                self.multiscale_lse_beta if self.multiscale_aggregation == "scale_lse" else None
            ),
            deep_layer_fusion=self.deep_layer_fusion,
            score_definition=f"{metadata['score_definition']}_multiscale_{self.multiscale_aggregation}",
            query_used_for_fit=False,
        )
        return metadata
