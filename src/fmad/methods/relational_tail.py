"""Relationally guarded tail adaptation (RGTA).

RGTA keeps the D3 low-rank residual adapter and its exact, memory-safe
four-view replay.  F1 adds a frozen-teacher normal score relation (a compact
score-level surrogate for neighbourhood preservation) and actually optimizes
the clean keep term.  F2 gates synthetic-positive ranking by a detached,
teacher-free reliability signal.  F3 applies conflict projection in score
space before the existing exact VJP replay; this is intentionally exposed as
a diagnostic, not claimed to be parameter-space PCGrad.
"""
import math

import torch
import numpy as np
from torch.nn import functional as F

from fmad.methods.guarded_metric import hard_positive_samples
from fmad.methods.tail_contrast import TailContrast, tail_samples
from fmad.registry import METHOD_REGISTRY


def relational_score_loss(student, teacher, eps=1e-6):
    """Bounded score-level relational surrogate.

    Centering/scaling removes arbitrary calibration while preserving relative
    normal-score geometry.  The detached teacher is fitted only from support
    normal data; no query labels or maps are retained.
    """
    teacher = teacher.detach()
    s = (student - student.mean()) / student.std(unbiased=False).clamp_min(eps)
    t = (teacher - teacher.mean()) / teacher.std(unbiased=False).clamp_min(eps)
    return F.smooth_l1_loss(s, t)


@METHOD_REGISTRY.register("relational_tail")
class RelationalTail(TailContrast):
    """F1/F2/F3 RGTA ablation family, with D3 as the all-on base structure."""

    def __init__(self, backbone, config):
        required = dict(hard_normal_ranking=True, paired_contrast=True,
                        low_rank_adapter=True)
        for name, value in required.items():
            if name in config and config[name] is not value:
                raise ValueError(f"relational_tail requires {name}={value}")
        super().__init__(backbone, {**config, **required})
        names = ("relational_teacher", "trusted_weak_synthetic", "conflict_safe_gradient")
        self.features = {name: config.get(name, False) for name in names}
        if any(type(v) is not bool for v in self.features.values()):
            raise ValueError("RGTA switches must be booleans")
        if self.features["trusted_weak_synthetic"] and not self.features["relational_teacher"]:
            raise ValueError("trusted_weak_synthetic requires relational_teacher")
        if self.features["conflict_safe_gradient"] and not self.features["relational_teacher"]:
            raise ValueError("conflict_safe_gradient requires relational_teacher")
        self.reliability_temperature = float(config.get("reliability_temperature", .05))
        self.reliability_threshold = float(config.get("reliability_threshold", .02))
        self.strengths = tuple(float(x) for x in config.get("anomaly_strengths", [.35, .65, 1.]))
        if self.reliability_temperature <= 0:
            raise ValueError("reliability_temperature must be positive")
        if not self.strengths or any(not 0 < x <= 1 for x in self.strengths):
            raise ValueError("anomaly_strengths must be non-empty values in (0, 1]")
        self._gradient_diagnostics = {}
        self.anomaly_strength = 1.
        self.adaptive_preservation = bool(config.get("adaptive_preservation", False))
        self.budget_aware_synthetic = bool(config.get("budget_aware_synthetic", False))
        if self.budget_aware_synthetic and not self.features["trusted_weak_synthetic"]:
            raise ValueError("budget_aware_synthetic requires trusted_weak_synthetic")
        self.relational_beta = float(config.get("relational_beta", 0.25))
        self.budget_base = float(config.get("budget_base", 0.05))
        self.susceptibility_scale = float(config.get("susceptibility_scale", 1.0))
        self.lambda_lr = float(config.get("lambda_lr", 0.1))
        self.lambda_max = float(config.get("lambda_max", 2.0))
        self.violation_ema_decay = float(config.get("violation_ema_decay", .9))
        self.lambda_value = float(config.get("lambda_init", 0.1))
        if not 0 <= self.violation_ema_decay < 1 or self.lambda_lr <= 0:
            raise ValueError("invalid dual update parameters")
        self.immediate_preservation = config.get("immediate_preservation", False)
        if type(self.immediate_preservation) is not bool:
            raise ValueError("immediate_preservation must be a boolean")
        self._lambda_initial = self.lambda_value
        if self.immediate_preservation:
            if not self.adaptive_preservation or not self.features["relational_teacher"]:
                raise ValueError("immediate_preservation requires adaptive_preservation and relational_teacher")
            if self.features["conflict_safe_gradient"]:
                raise ValueError("immediate_preservation cannot use conflict_safe_gradient")
            if (not all(math.isfinite(x) for x in (self._lambda_initial, self.lambda_max,
                                                   self.budget_base, self.susceptibility_scale))
                    or not 0 < self._lambda_initial <= self.lambda_max
                    or self.budget_base <= 0 or self.susceptibility_scale < 0):
                raise ValueError("invalid immediate preservation bounds or budget")
        self._violation_ema = 0.
        self._budget = self.budget_base
        self._nu = 0.

    def training_anomaly(self, image, rng, family_index, step):
        anomaly, mask, family = super().training_anomaly(image, rng, family_index, step)
        self.anomaly_strength = 1.
        if self.features["trusted_weak_synthetic"]:
            # Cycle across source/step so one support image is not permanently
            # assigned a weak or strong synthetic family.
            self.anomaly_strength = self.strengths[step % len(self.strengths)]
            mixed = image.astype(np.float32) + self.anomaly_strength * (
                anomaly.astype(np.float32) - image.astype(np.float32))
            anomaly = np.ascontiguousarray(np.rint(mixed).clip(0, 255).astype(np.uint8))
        return anomaly, mask, family

    def ranking_loss(self, scores, mask, sampler):
        if not self.features["trusted_weak_synthetic"]:
            return super().ranking_loss(scores, mask, sampler)
        normal = torch.cat([tail_samples(s, self.tail_fraction, self.train_tokens, sampler)
                            for s in scores[:2]])
        terms, reliabilities = [], []
        for view in scores[2:]:
            positive = view[mask]
            if not len(positive):
                continue
            # Detached reliability: weak synthetic positives that do not move
            # beyond the clean score are downweighted, never treated as truth.
            clean_ref = scores[0].detach()[mask]
            reliability = torch.sigmoid((positive.detach() - clean_ref
                                          - self.reliability_threshold)
                                         / self.reliability_temperature)
            count = min(self.train_tokens, len(positive))
            hard_count = count // 2
            hard_pool = max(1, int(np.ceil(len(positive) *
                                           float(self.config.get("hard_positive_fraction", .25)))))
            hard_ids = positive.detach().topk(hard_pool, largest=False).indices
            hard_ids = hard_ids[torch.randperm(len(hard_ids), device=view.device,
                                                generator=sampler)[:hard_count]]
            random_ids = torch.randperm(len(positive), device=view.device,
                                        generator=sampler)[:count - hard_count]
            positive_ids = torch.cat((hard_ids, random_ids))
            # Index both tensors together; reliability cannot drift from its
            # selected positive after hard/random sampling.
            positive, reliability = positive[positive_ids], reliability[positive_ids]
            negative = torch.cat((normal,
                                  tail_samples(view[~mask], self.tail_fraction,
                                               self.train_tokens, sampler)))
            hinge = F.relu(float(self.config.get("margin", .05))
                           - positive[:, None] + negative[None])
            weight = reliability
            if self.budget_aware_synthetic:
                # Both factors are detached: this is sample selection, not a
                # second path for synthetic gradients to bypass preservation.
                consistency = torch.exp(-(scores[1].detach() - scores[0].detach()).abs()
                                         .mean() / self.reliability_temperature)
                slack = torch.sigmoid(torch.as_tensor(self._budget_slack,
                                                       device=view.device) /
                                      self.reliability_temperature)
                weight = (weight * consistency * slack).clamp(0., 1.)
                self._gradient_diagnostics["synthetic_weight_mean"] = float(weight.mean())
            terms.append((hinge * weight[:, None]).mean())
            reliabilities.append(weight.mean())
        if not terms:
            return scores[0].sum() * 0
        self._gradient_diagnostics["synthetic_reliability_mean"] = float(
            torch.stack(reliabilities).mean().detach())
        return torch.stack(terms).mean()

    def preservation_terms(self, clean, teacher, keep):
        """Method hook; the default retains the historical full-map objective."""
        return keep, relational_score_loss(clean, teacher)

    def compose_training_loss(self, scores, mask, baseline_clean, rank, keep,
                              shift, contrast):
        # Keep diagnostics emitted by ranking_loss (notably F2 reliability).
        for key in ("conflict_steps", "preserve_anomaly_dot", "conflict_projection"):
            self._gradient_diagnostics.pop(key, None)
        preserve = torch.zeros_like(rank)
        preservation_keep = keep
        if self.features["relational_teacher"]:
            preservation_keep, rel = self.preservation_terms(scores[0], baseline_clean, keep)
            preserve = (float(self.config.get("keep_weight", 1.)) * preservation_keep
                        + (self.relational_beta if self.adaptive_preservation
                           else float(self.config.get("relational_weight", .25))) * rel)
        anomaly = (rank + float(self.config.get("shift_weight", 1.)) * shift
                   + float(self.config.get("contrast_weight", 1.)) * contrast)
        self._preserve_loss, self._anomaly_loss = preserve, anomaly
        if self.immediate_preservation:
            # Current-step feedback, not accumulated dual state. Detach the gain
            # so replay differentiates the weighted loss, not its controller.
            relative = float(preserve.detach()) / self._budget
            self.lambda_value = max(self._lambda_initial, min(self.lambda_max, relative))
        coefficient = self.lambda_value if self.adaptive_preservation else 1.
        total = anomaly + coefficient * preserve
        violation = preserve.detach() - self._budget
        if self.adaptive_preservation and not self.immediate_preservation:
            self._update_dual(float(violation))
        return total, dict(relational_loss=float((preserve - float(
            self.config.get("keep_weight", 1.)) * preservation_keep).detach()),
            preservation_keep_loss=float(preservation_keep.detach()),
            keep_loss_optimized=True, preservation=float(preserve.detach()),
            preservation_budget=float(self._budget), preservation_violation=float(violation),
            dual_lambda=float(coefficient), P=float(preserve.detach()),
            budget=float(self._budget), violation=float(violation),
            lambda_value=float(self.lambda_value))

    def _update_dual(self, violation):
        """Update detached dual state from a clipped budget-relative EMA."""
        relative = max(-1., min(1., violation / max(abs(self._budget), 1e-8)))
        self._violation_ema = (self.violation_ema_decay * self._violation_ema
                               + (1. - self.violation_ema_decay) * relative)
        self.lambda_value = float(max(0., min(
            self.lambda_max, self.lambda_value + self.lambda_lr * self._violation_ema)))

    def training_step_context(self, scores, baseline_clean, baseline_shift):
        if not self.adaptive_preservation:
            return None
        # Teacher inputs are baseline scores of support-derived views only.
        clean = baseline_clean.detach()
        shifted = baseline_shift.detach()
        clean_scale = clean.abs().mean().clamp_min(1e-6)
        self._nu = float((shifted - clean).abs().mean() / clean_scale)
        self._budget = self.budget_base / (1. + self.susceptibility_scale * self._nu)
        keep = (scores[0].detach() - clean).square().mean()
        keep, rel = self.preservation_terms(scores[0].detach(), clean, keep)
        self._budget_slack = self._budget - float(keep + self.relational_beta * rel)
        self._gradient_diagnostics.update(shift_susceptibility=self._nu,
                                          preservation_budget=self._budget,
                                          budget_slack=self._budget_slack,
                                          nu=self._nu, budget=self._budget)

    def training_gradients(self, total, score_leaves):
        if not self.features["conflict_safe_gradient"]:
            return super().training_gradients(total, score_leaves)
        preserve = torch.autograd.grad(self._preserve_loss, score_leaves,
                                       retain_graph=True, allow_unused=True)
        anomaly = torch.autograd.grad(self._anomaly_loss, score_leaves,
                                      retain_graph=True, allow_unused=True)
        pairs = [(torch.zeros_like(ref) if p is None else p,
                  torch.zeros_like(ref) if a is None else a)
                 for ref, p, a in zip(score_leaves, preserve, anomaly)]
        pflat = torch.cat([p.reshape(-1) for p, _ in pairs])
        aflat = torch.cat([a.reshape(-1) for _, a in pairs])
        dot = torch.dot(pflat, aflat)
        pnorm = torch.dot(pflat, pflat).clamp_min(1e-12)
        conflict = bool(dot.detach() < 0)
        projected = list(anomaly)
        if conflict:
            coeff = dot / pnorm
            projected = [None if a is None else a - coeff *
                         (torch.zeros_like(a) if p is None else p)
                         for a, p in zip(anomaly, preserve)]
        self._gradient_diagnostics.update(conflict_steps=int(conflict),
                                          preserve_anomaly_dot=float(dot.detach()),
                                          conflict_projection=float((dot / pnorm).detach())
                                          if conflict else 0.)
        return tuple((torch.zeros_like(ref) if p is None else p) +
                     (torch.zeros_like(ref) if a is None else a)
                     for ref, p, a in zip(score_leaves, preserve, projected))

    def training_diagnostics(self):
        return dict(self._gradient_diagnostics, anomaly_strength=self.anomaly_strength)

    def object_metadata(self):
        metadata = super().object_metadata()
        metadata.update(rgta_components=self.features,
                        relational_surrogate="standardized_normal_score_smooth_l1",
                        anomaly_strengths=list(self.strengths),
                        conflict_projection="score_leaf_pcgrad" if self.features["conflict_safe_gradient"] else None)
        metadata.update(adaptive_preservation=self.adaptive_preservation,
                        immediate_preservation=self.immediate_preservation,
                        budget_aware_synthetic=self.budget_aware_synthetic,
                        budget_base=self.budget_base, susceptibility_scale=self.susceptibility_scale,
                        relational_beta=self.relational_beta, lambda_max=self.lambda_max)
        return metadata
