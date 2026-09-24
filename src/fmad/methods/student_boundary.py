"""K2: select synthetic strength against the current student's ranking pool.

Selection is detached and support-only. The paired transform and ranking RNG
are previewed without advancing either training stream. Only construction
changes; K1/H1 losses, preservation, exact replay and inference stay fixed.
"""
import copy
import math

import torch
from torch.nn import functional as F

from fmad.methods.boundary_synthetic import BoundarySynthetic, boundary_choice
from fmad.methods.PSMA_MR import _apply_photometric, _sample_photometric_transform
from fmad.methods.relational_tail import RelationalTail
from fmad.methods.tail_contrast import tail_ranking_pairs
from fmad.registry import METHOD_REGISTRY


def empirical_quantile(values, fraction):
    if not values.numel() or not torch.isfinite(values).all():
        raise RuntimeError("student boundary requires nonempty finite scores")
    return values.flatten().kthvalue(max(1, math.ceil(fraction * values.numel()))).values


@torch.no_grad()
def student_boundary_stats(scores, mask, fraction, count, margin, sampler):
    """Worst-view quartile gap and exact mean hinge on the same sampled pairs.

    A quantile gap is a selection proxy, not a guarantee for every pair or
    the query population. Never advance the sampler used by actual training.
    """
    preview = torch.Generator(device=sampler.device)
    preview.set_state(sampler.get_state())
    gaps, losses = [], []
    for positive, negative in tail_ranking_pairs(scores, mask, fraction, count, preview):
        gaps.append(empirical_quantile(positive, .25)
                    - empirical_quantile(negative, 1. - fraction))
        losses.append(F.relu(margin - positive[:, None] + negative[None]).mean())
    return float(torch.stack(gaps).min()), float(torch.stack(losses).mean())


@METHOD_REGISTRY.register("student_boundary")
class StudentBoundary(BoundarySynthetic):
    def __init__(self, backbone, config):
        super().__init__(backbone, config)
        self.student_boundary = config.get("student_boundary", True)
        if type(self.student_boundary) is not bool:
            raise ValueError("student_boundary must be a boolean")
        if self.student_boundary and not self.enabled:
            raise ValueError("student_boundary requires boundary_synthetic")
        self._ranking_sampler = None

    def _prepare_fit(self, support_paths, context):
        prepared = super()._prepare_fit(support_paths, context)
        self._ranking_sampler = prepared[3]
        return prepared

    @torch.no_grad()
    def _student_scores(self, image, banks):
        branches = {r: self.extract(image, r) for r in self.resolutions}
        return self.fused(branches, banks, image.shape[:2])

    @torch.no_grad()
    def training_anomaly(self, image, rng, family_index, step):
        if not self.student_boundary:
            return super().training_anomaly(image, rng, family_index, step)
        # Bypass K1's teacher selection, not the common synthetic generator.
        anomaly, mask, family = RelationalTail.training_anomaly(
            self, image, rng, family_index, step)
        transform = _sample_photometric_transform(copy.deepcopy(rng), 1.)
        source = step % self.support_count
        banks = {r: (b[0][torch.tensor(b[2], device=self.device) != source], None,
                     [owner for owner in b[2] if owner != source])
                 for r, b in self.banks.items()}
        if any(not b[0].shape[0] or source in b[2] for b in banks.values()):
            raise RuntimeError("student boundary requires source-excluded support banks")
        foreground = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        clean = [self._student_scores(view, banks) for view in
                 (image, _apply_photometric(image, transform))]
        gaps, losses = [], []
        for strength in self.boundary_strengths:
            candidate = self._blend(image, anomaly, strength)
            views = [self._student_scores(view, banks) for view in
                     (candidate, _apply_photometric(candidate, transform))]
            gap, loss = student_boundary_stats(
                clean + views, foreground, self.tail_fraction, self.train_tokens,
                self.boundary_margin, self._ranking_sampler)
            gaps.append(gap)
            losses.append(loss)
            del views
        selected, feasible = boundary_choice(gaps, self.boundary_margin)
        self.anomaly_strength = self.boundary_strengths[selected]
        # Only scalars survive selection; no candidate maps/features are cached.
        self._gradient_diagnostics.update(
            boundary_strength=self.anomaly_strength, boundary_feasible=int(feasible),
            boundary_candidates=len(gaps), student_boundary_gap=gaps[selected],
            student_boundary_excess=gaps[selected] - self.boundary_margin,
            student_boundary_target=self.boundary_margin,
            student_selected_rank_loss=losses[selected])
        for index, (gap, loss) in enumerate(zip(gaps, losses)):
            self._gradient_diagnostics[f"student_candidate_{index}_gap"] = gap
            self._gradient_diagnostics[f"student_candidate_{index}_rank_loss"] = loss
        return self._blend(image, anomaly, self.anomaly_strength), mask, family

    def ranking_loss(self, scores, mask, sampler):
        loss = super().ranking_loss(scores, mask, sampler)
        if self.student_boundary:
            predicted = self._gradient_diagnostics["student_selected_rank_loss"]
            self._gradient_diagnostics["student_selection_replay_error"] = abs(
                float(loss.detach()) - predicted)
        return loss

    def object_metadata(self):
        metadata = {**super().object_metadata(), "student_boundary": self.student_boundary}
        if self.student_boundary:
            metadata.update(boundary_teacher=None, boundary_normal_quantile=None,
                            boundary_scorer="current_student_source_excluded",
                            boundary_negative_quantile=1. - self.tail_fraction,
                            boundary_views="worst_of_paired_unshifted_and_shifted",
                            boundary_sampler="preview_exact_training_state",
                            boundary_selection="closest_feasible_quartile_gap")
        return metadata
