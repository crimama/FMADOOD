"""I2: learn local defect uplift instead of cross-location absolute ranking."""
import torch
from torch.nn import functional as F

from fmad.methods.relational_tail import RelationalTail
from fmad.registry import METHOD_REGISTRY


@METHOD_REGISTRY.register("paired_margin")
class PairedMargin(RelationalTail):
    def __init__(self, backbone, config):
        super().__init__(backbone, config)
        self.enabled = config.get("paired_margin", True)
        if type(self.enabled) is not bool or not self.immediate_preservation:
            raise ValueError("paired_margin requires a boolean switch and H1 controller")
        margin = float(config.get("margin", .05))
        if not 0 < margin < float("inf"):
            raise ValueError("paired margin must be finite and positive")
        self._shift_displacement = None

    def training_step_context(self, scores, baseline_clean, baseline_shift):
        super().training_step_context(scores, baseline_clean, baseline_shift)
        if self.enabled:
            self._shift_displacement = (baseline_shift - baseline_clean).detach().abs()

    def ranking_loss(self, scores, mask, sampler):
        if not self.enabled:
            return super().ranking_loss(scores, mask, sampler)
        if self._shift_displacement is None:
            raise RuntimeError("paired margin requires current support-view context")
        displacement = self._shift_displacement[mask]
        self._shift_displacement = None  # Never retain a dense map across steps.
        count = min(self.train_tokens, displacement.numel())
        if not count:
            return sum(score.sum() * 0 for score in scores)
        ids = torch.randperm(displacement.numel(), device=mask.device,
                            generator=sampler)[:count]
        margin = float(self.config.get("margin", .05))
        target = margin + displacement[ids].clamp(max=margin)
        clean_uplift = (scores[2][mask] - scores[0][mask])[ids]
        shift_uplift = (scores[3][mask] - scores[1][mask])[ids]
        worst_uplift = torch.minimum(clean_uplift, shift_uplift)
        self._gradient_diagnostics.update(
            paired_margin_mean=float(target.mean()),
            paired_uplift_mean=float(worst_uplift.detach().mean()))
        return F.relu(target - worst_uplift).mean()

    def object_metadata(self):
        return {**super().object_metadata(), "paired_margin": self.enabled,
                "ranking_target": "worst_view_local_uplift" if self.enabled else "absolute_tail"}
