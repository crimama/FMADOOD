"""K1: choose synthetic strength near a frozen, support-only normal boundary.

Only the synthetic construction changes. H1's absolute normal-tail ranking,
preservation, four-view replay and prediction remain unchanged. The teacher
measures proxy difficulty, not synthetic correctness or real-defect realism.
"""
import math

import numpy as np
import torch

from fmad.methods.relational_tail import RelationalTail
from fmad.registry import METHOD_REGISTRY


def boundary_choice(positive_scores, target):
    """Closest feasible score above target; use the full-strength last fallback."""
    feasible = [i for i, score in enumerate(positive_scores) if score >= target]
    if not feasible:
        return len(positive_scores) - 1, False
    return min(feasible, key=lambda i: (positive_scores[i] - target, i)), True


@METHOD_REGISTRY.register("boundary_synthetic")
class BoundarySynthetic(RelationalTail):
    def __init__(self, backbone, config):
        super().__init__(backbone, config)
        self.enabled = config.get("boundary_synthetic", True)
        if type(self.enabled) is not bool or not self.immediate_preservation:
            raise ValueError("boundary_synthetic requires a boolean switch and H1 controller")
        if self.features["trusted_weak_synthetic"] or self.budget_aware_synthetic:
            raise ValueError("boundary_synthetic cannot combine F2/G2 synthetic weighting")
        self.boundary_strengths = tuple(float(x) for x in config.get(
            "boundary_strengths", [.25, .5, 1.]))
        if (not self.boundary_strengths or self.boundary_strengths[-1] != 1.
                or any(not math.isfinite(x) or not 0 < x <= 1
                       for x in self.boundary_strengths)
                or tuple(sorted(set(self.boundary_strengths))) != self.boundary_strengths):
            raise ValueError("boundary_strengths must increase in (0, 1] and end at 1")
        self.boundary_margin = float(config.get("margin", .05))
        if not math.isfinite(self.boundary_margin) or self.boundary_margin <= 0:
            raise ValueError("boundary margin must be finite and positive")
        # Only one scalar per support. Never retain teacher maps or features.
        self._normal_thresholds = {}

    def _prepare_fit(self, support_paths, context):
        self._normal_thresholds.clear()
        return super()._prepare_fit(support_paths, context)

    @torch.no_grad()
    def _teacher_quantile(self, image, banks, quantile, mask=None):
        branches = {r: self.extract(image, r) for r in self.resolutions}
        scores = self.fused(branches, banks, image.shape[:2], baseline=True)
        values = scores.flatten() if mask is None else scores[
            torch.as_tensor(mask, device=scores.device, dtype=torch.bool)]
        if not values.numel() or not torch.isfinite(values).all():
            raise RuntimeError("boundary teacher requires nonempty finite support scores")
        # Empirical inverse CDF; no interpolated quantile or retained dense cache.
        k = max(1, math.ceil(quantile * values.numel()))
        return float(values.kthvalue(k).values)

    @staticmethod
    def _blend(image, anomaly, strength):
        if strength == 1.:
            return anomaly
        mixed = image.astype(np.float32) + strength * (
            anomaly.astype(np.float32) - image.astype(np.float32))
        return np.ascontiguousarray(np.rint(mixed).clip(0, 255).astype(np.uint8))

    def training_anomaly(self, image, rng, family_index, step):
        anomaly, mask, family = super().training_anomaly(image, rng, family_index, step)
        if not self.enabled:
            return anomaly, mask, family
        source = step % self.support_count
        # Same leave-one-source-out population as H1 fitting, never self-matching.
        banks = {r: (bank[0][torch.tensor(bank[2], device=self.device) != source],
                     None, [owner for owner in bank[2] if owner != source])
                 for r, bank in self.banks.items()}
        if any(not bank[0].shape[0] or source in bank[2] for bank in banks.values()):
            raise RuntimeError("boundary teacher requires a nonempty source-excluded bank")
        if source not in self._normal_thresholds:
            self._normal_thresholds[source] = self._teacher_quantile(
                image, banks, 1. - self.tail_fraction)
        normal = self._normal_thresholds[source]
        target = normal + self.boundary_margin
        positive = []
        for strength in self.boundary_strengths:
            candidate = self._blend(image, anomaly, strength)
            positive.append(self._teacher_quantile(candidate, banks, .25, mask))
        selected, feasible = boundary_choice(positive, target)
        self.anomaly_strength = self.boundary_strengths[selected]
        self._gradient_diagnostics.update(
            boundary_strength=self.anomaly_strength, boundary_normal_q=normal,
            boundary_target=target, boundary_positive_q=positive[selected],
            boundary_excess=positive[selected] - target,
            boundary_feasible=int(feasible), boundary_candidates=len(positive))
        return self._blend(image, anomaly, self.anomaly_strength), mask, family

    def object_metadata(self):
        return {**super().object_metadata(), "boundary_synthetic": self.enabled,
                "boundary_strengths": list(self.boundary_strengths),
                "boundary_teacher": "frozen_source_excluded_normal_bank",
                "boundary_normal_quantile": 1. - self.tail_fraction,
                "boundary_positive_quantile": .25,
                "boundary_fallback": "original_full_strength"}
