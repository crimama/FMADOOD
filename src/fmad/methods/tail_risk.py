"""J1: concentrate H1 ranking on the hardest sampled normal/defect pairs."""
import math

import torch
from torch.nn import functional as F

from fmad.methods.relational_tail import RelationalTail
from fmad.methods.tail_contrast import tail_samples
from fmad.registry import METHOD_REGISTRY


def upper_tail_mean(losses, fraction):
    """Empirical upper-tail mean; detached selection, live selected gradients.

    The retained mass is ceil(fraction * N) / N. This is a sampled hinge-risk
    surrogate, not an exact pAUC or region-weighted AUPRO estimator.
    """
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("tail risk fraction must be in (0, 1]")
    flat = losses.reshape(-1)
    if not flat.numel():
        return flat.sum()
    count = math.ceil(fraction * flat.numel())
    ids = flat.detach().topk(count).indices
    return flat[ids].mean()


@METHOD_REGISTRY.register("tail_risk")
class TailRisk(RelationalTail):
    def __init__(self, backbone, config):
        super().__init__(backbone, config)
        self.enabled = config.get("tail_risk", True)
        if type(self.enabled) is not bool or not self.immediate_preservation:
            raise ValueError("tail_risk requires a boolean switch and H1 controller")
        if self.features["trusted_weak_synthetic"]:
            raise ValueError("tail_risk requires H1's unchanged synthetic generator")
        self.risk_fraction = float(config.get("tail_risk_fraction", .25))
        if not math.isfinite(self.risk_fraction) or not 0 < self.risk_fraction <= 1:
            raise ValueError("tail_risk_fraction must be in (0, 1]")
        margin = float(config.get("margin", .05))
        if not math.isfinite(margin) or margin <= 0:
            raise ValueError("tail risk margin must be finite and positive")

    def ranking_loss(self, scores, mask, sampler):
        if not self.enabled:
            return super().ranking_loss(scores, mask, sampler)
        # Exactly H1's pools and RNG consumption; only risk reduction changes.
        clean_negatives = torch.cat([
            tail_samples(view, self.tail_fraction, self.train_tokens, sampler)
            for view in scores[:2]
        ])
        terms, means, active = [], [], []
        for view in scores[2:]:
            positive = view[mask]
            ids = torch.randperm(len(positive), device=view.device,
                                 generator=sampler)[:self.train_tokens]
            positive = positive[ids]
            negative = torch.cat((clean_negatives, tail_samples(
                view[~mask], self.tail_fraction, self.train_tokens, sampler)))
            hinge = F.relu(float(self.config.get("margin", .05))
                           - positive[:, None] + negative[None])
            # At most train_tokens x (6 * train_tokens), not a dense image bank.
            terms.append(upper_tail_mean(hinge, self.risk_fraction))
            means.append(hinge.mean() if hinge.numel() else hinge.sum())
            active.append((hinge.detach() > 0).float().mean()
                          if hinge.numel() else hinge.detach().sum())
        risk, mean = torch.stack(terms).mean(), torch.stack(means).mean()
        self._gradient_diagnostics.update(
            sampled_pair_mean_loss=float(mean.detach()),
            sampled_pair_tail_loss=float(risk.detach()),
            sampled_pair_active_fraction=float(torch.stack(active).mean()),
            tail_risk_fraction=self.risk_fraction)
        return risk

    def object_metadata(self):
        return {**super().object_metadata(), "tail_risk": self.enabled,
                "tail_risk_fraction": self.risk_fraction,
                "ranking_target": "upper_tail_sampled_pair_hinge"
                if self.enabled else "absolute_tail"}
