"""L1: low-FPR operating-region ranking for synthetic defects.

The K1 synthetic selector, preservation controller, replay path and inference
are inherited unchanged.  Only the ranking objective changes: clean and
photometric-clean scores define differentiable low-FPR thresholds, while
synthetic positives are averaged per 8-connected mask region and then per
synthetic view.  Region labels are temporary and never cached.
"""
import math

import numpy as np
import torch
from torch.nn import functional as F

from fmad.evaluation.streaming import _label8
from fmad.methods.boundary_synthetic import BoundarySynthetic
from fmad.registry import METHOD_REGISTRY


def operating_region_thresholds(clean_scores, fpr_fractions):
    """Return live quantiles at ``1 - fpr`` for pooled clean score maps."""
    if clean_scores.ndim == 0 or not clean_scores.numel():
        raise ValueError("clean scores must be non-empty")
    values = clean_scores.reshape(-1)
    if not torch.isfinite(values).all():
        raise ValueError("clean scores must be finite")
    quantiles = torch.as_tensor([1. - float(fpr) for fpr in fpr_fractions],
                                device=values.device, dtype=values.dtype)
    return torch.quantile(values, quantiles)


def operating_region_loss(scores, mask, fpr_fractions, temperature, margin):
    """Compute equal-view, equal-region low-FPR soft hinge risk.

    ``temperature * softplus`` preserves the scale of a hinge-like loss while
    keeping the quantile and positive score paths differentiable.  No random
    sampling is performed, and the mask labels are a temporary CPU object.
    """
    if len(scores) != 4:
        raise ValueError("operating-region ranking expects four score views")
    binary = np.asarray(mask.detach().cpu().numpy(), dtype=bool)
    labels, count = _label8(binary)
    if count == 0:
        # Keep the common four-leaf replay contract even for an empty mask.
        return sum((score.sum() * 0. for score in scores), scores[0].sum() * 0.)
    clean = torch.cat((scores[0].reshape(-1), scores[1].reshape(-1)))
    thresholds = operating_region_thresholds(clean, fpr_fractions)
    losses = []
    for positive_view in scores[2:]:
        regions = []
        for region_id in range(1, count + 1):
            region = torch.as_tensor(labels == region_id, device=positive_view.device,
                                     dtype=torch.bool)
            positive = positive_view[region]
            # Every region contributes equally, independent of pixel area.
            per_threshold = []
            for threshold in thresholds.to(positive_view.device):
                per_threshold.append(
                    (temperature * F.softplus(
                        (threshold + margin - positive) / temperature)).mean())
            regions.append(torch.stack(per_threshold).mean())
        losses.append(torch.stack(regions).mean())
    return torch.stack(losses).mean()


@METHOD_REGISTRY.register("operating_region")
class OperatingRegion(BoundarySynthetic):
    """K1 plus a low-FPR, region-balanced synthetic ranking objective."""

    def __init__(self, backbone, config):
        super().__init__(backbone, config)
        self.operating_enabled = config.get("operating_region", True)
        if type(self.operating_enabled) is not bool:
            raise ValueError("operating_region must be a boolean")
        self.operating_fprs = tuple(float(x) for x in config.get(
            "operating_fprs", (.01, .025, .05)))
        if (not self.operating_fprs
                or any(not math.isfinite(x) or not 0 < x <= .05
                       for x in self.operating_fprs)
                or tuple(sorted(set(self.operating_fprs))) != self.operating_fprs):
            raise ValueError("operating_fprs must be strictly increasing in (0, .05]")
        self.operating_temperature = float(config.get("operating_temperature", .05))
        if not math.isfinite(self.operating_temperature) or self.operating_temperature <= 0:
            raise ValueError("operating_temperature must be finite and positive")
        self.operating_margin = float(config.get("margin", .05))
        if not math.isfinite(self.operating_margin) or self.operating_margin <= 0:
            raise ValueError("operating margin must be finite and positive")

    def ranking_loss(self, scores, mask, sampler):
        if not self.operating_enabled:
            return super().ranking_loss(scores, mask, sampler)
        loss = operating_region_loss(
            scores, mask, self.operating_fprs, self.operating_temperature,
            self.operating_margin)
        with torch.no_grad():
            clean = torch.cat((scores[0].reshape(-1), scores[1].reshape(-1)))
            thresholds = operating_region_thresholds(clean, self.operating_fprs)
            _, region_count = _label8(mask.detach().cpu().numpy())
            positive = torch.cat([view[mask] for view in scores[2:]])
        self._gradient_diagnostics.update(
            operating_region_loss=float(loss.detach()),
            operating_region_count=int(region_count),
            operating_region_positive_mean=float(positive.mean().detach()) if positive.numel() else 0.,
            **{f"operating_threshold_{i}": float(value)
               for i, value in enumerate(thresholds)})
        return loss

    def object_metadata(self):
        return {**super().object_metadata(), "operating_region": self.operating_enabled,
                "operating_fprs": list(self.operating_fprs),
                "operating_temperature": self.operating_temperature,
                "ranking_target": "low_fpr_region_balanced_synthetic"}
