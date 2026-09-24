"""Keep MC1's full-query corrections; charge for changing QR1 correspondences.

CG0=QR1, CG1=MC1, CG2=mode-change cost, CG3=additional within-mode cost.
Costs use the same query and actual final candidate bank, not donor refits or
spatially excluded evidence. QR1 is a query-dependent anchor, not ground truth.
This can preserve an erroneous anchor and cannot protect anomalies absorbed by
their original reference. No normality, ranking, or novelty guarantee is made.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from fmad.methods.mode_reference import ModeReference, mode_translations
from fmad.methods.query_reference import fit_translation
from fmad.registry import METHOD_REGISTRY


def empty_guard_stats():
    return {key: 0 for key in (
        "tokens", "mc1_reference_switch_tokens", "mc1_mode_switch_tokens",
        "guarded_reference_switch_tokens", "guarded_mode_switch_tokens",
        "changed_choice_tokens", "raised_score_tokens", "score_raise_sum",
        "score_raise_max", "selected_cost_sum", "qr1_score_sum", "mc1_score_sum",
        "guarded_score_sum", "mc1_benefit_sum", "removed_mc1_benefit_sum",
        "lower_bound_violations", "anchor_bound_violations")}


def correspondence_cost(anchor_distances, labels, *, mode_guard, identity_guard, weight):
    """Nonnegative QR1 mode gap plus optional within-mode identity gap.

    mode_gap(q,k) = min_{j in k} d0(q,j) - min_j d0(q,j)
    identity_gap(q,j) = d0(q,j) - min_{l in mode(j)} d0(q,l)
    With both switches their sum is d0(q,j)-min_j d0(q,j).
    This is a soft cost, not a hard ban or a probability of a bad match.
    """
    if identity_guard and not mode_guard:
        raise ValueError("identity_guard requires mode_guard")
    if not mode_guard or weight == 0:
        return torch.zeros_like(anchor_distances)
    best = anchor_distances.amin(1, keepdim=True)
    if identity_guard:
        # Algebraically identical to summing both gaps, with one allocation.
        return weight * (anchor_distances - best).clamp_min(0)
    unique, inverse = labels.unique(sorted=True, return_inverse=True)
    mode_best = torch.stack([anchor_distances[:, labels == k].amin(1) for k in unique], dim=1)
    return weight * (mode_best[:, inverse] - best).clamp_min(0)


def guarded_distances(query, anchor, candidate, labels, *, chunk, mode_guard,
                      identity_guard, weight):
    """Chunk-by-bank matrices only; return one query's scores and scalar totals.

    min d1 <= min(d1+cost) <= d1 at the QR1-nearest reference, since that
    reference has zero cost. These are score bounds, NOT metric guarantees.
    """
    stats = empty_guard_stats()
    scores = []
    for start in range(0, len(query), chunk):
        values = query[start:start + chunk]
        d0 = 1.0 - values @ anchor.T
        d1 = 1.0 - values @ candidate.T
        base, original = d0.min(1)
        raw, unguarded = d1.min(1)
        cost = correspondence_cost(d0, labels, mode_guard=mode_guard,
                                   identity_guard=identity_guard, weight=weight)
        score, selected = (d1 + cost).min(1)
        rows = torch.arange(len(values), device=query.device)
        increase = (score - raw).clamp_min(0)
        benefit = (base - raw).clamp_min(0)
        stats["tokens"] += len(values)
        stats["mc1_reference_switch_tokens"] += int((unguarded != original).sum())
        stats["mc1_mode_switch_tokens"] += int((labels[unguarded] != labels[original]).sum())
        stats["guarded_reference_switch_tokens"] += int((selected != original).sum())
        stats["guarded_mode_switch_tokens"] += int((labels[selected] != labels[original]).sum())
        stats["changed_choice_tokens"] += int((selected != unguarded).sum())
        stats["raised_score_tokens"] += int((increase > 1e-7).sum())
        stats["score_raise_sum"] += float(increase.sum())
        stats["score_raise_max"] = max(stats["score_raise_max"], float(increase.max()))
        stats["selected_cost_sum"] += float(cost[rows, selected].sum())
        stats["qr1_score_sum"] += float(base.sum())
        stats["mc1_score_sum"] += float(raw.sum())
        stats["guarded_score_sum"] += float(score.sum())
        stats["mc1_benefit_sum"] += float(benefit.sum())
        stats["removed_mc1_benefit_sum"] += float(torch.minimum(increase, benefit).sum())
        stats["lower_bound_violations"] += int((score < raw - 1e-6).sum())
        stats["anchor_bound_violations"] += int((score > d1[rows, original] + 1e-6).sum())
        scores.append(score)
    result = torch.cat(scores) if scores else query.new_empty(0)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("nonfinite correspondence-guard score")
    return result, stats


@METHOD_REGISTRY.register("correspondence_guard")
class CorrespondenceGuard(ModeReference):
    def __init__(self, backbone, config):
        super().__init__(backbone, config)
        if self.normal_subspace or self.orthogonal_gate:
            raise ValueError("correspondence_guard keeps unrestricted MC1 corrections")
        for name in ("mode_guard", "identity_guard"):
            value = config.get(name, False)
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")
            setattr(self, name, value)
        if self.mode_guard and not self.mode_conditioning:
            raise ValueError("mode_guard requires mode_conditioning")
        if self.identity_guard and not self.mode_guard:
            raise ValueError("identity_guard requires mode_guard")
        self.guard_weight = self._number(config, "guard_weight", 1., 0., None, allow_zero=True)

    def _reset_diagnostics(self):
        super()._reset_diagnostics()
        self.guard_stats = empty_guard_stats()

    def fit(self, support_paths, context):
        super().fit(support_paths, context)
        self.log_training(step=None, event="correspondence_guard_ready",
                          mode_guard=self.mode_guard, identity_guard=self.identity_guard,
                          guard_weight=self.guard_weight, trainable_parameters=0)

    @torch.no_grad()
    def _score_tokens(self, layers, grid, bank, *, normalize=True):
        if not self.mode_guard:
            return super()._score_tokens(layers, grid, bank, normalize=normalize)
        layer_scores = []
        for index, (query, reference, geometry) in enumerate(zip(layers, bank["layers"], bank["mode_geometry"])):
            residuals, modes = [], []
            for start in range(0, len(query), self.chunk):
                values = query[start:start + self.chunk]
                selected = (values @ reference.T).argmax(1)
                residuals.append(values - reference[selected])
                modes.append(geometry["labels"][selected])
            residuals, modes = torch.cat(residuals), torch.cat(modes)
            # Same function, full-query inputs, clusters and parameters as MC1.
            deltas, stats, extra = mode_translations(
                residuals, modes, geometry, ridge=self.ridge, max_norm=self.max_norm,
                min_tokens=self.min_tokens, min_mode_tokens=self.min_mode_tokens,
                prior_tokens=self.prior_tokens, constrain=False, gate=False)
            baseline, _ = fit_translation(residuals, keep_fraction=1., ridge=self.ridge,
                                          max_norm=self.max_norm, min_tokens=self.min_tokens)
            anchor = reference if stats["fallback"] else F.normalize(reference + baseline, dim=-1)
            candidate = reference if stats["fallback"] else F.normalize(reference + deltas[geometry["labels"]], dim=-1)
            scores, evidence = guarded_distances(
                query, anchor, candidate, geometry["labels"], chunk=self.chunk,
                mode_guard=self.mode_guard, identity_guard=self.identity_guard,
                weight=self.guard_weight)
            for key, value in evidence.items():
                if key == "score_raise_max":
                    self.guard_stats[key] = max(self.guard_stats[key], value)
                else:
                    self.guard_stats[key] += value
            layer_scores.append(scores / (self._active_scale(index, query.device) if normalize else 1.))
            self.translation_stats["fits"] += 1
            self.translation_stats["fallback_fits"] += int(stats["fallback"])
            self.translation_stats["clipped_fits"] += int(stats["clipped"])
            self.translation_stats["available_tokens"] += stats["available_tokens"]
            self.translation_stats["retained_tokens"] += stats["retained_tokens"]
            self.translation_stats["norm_sum"] += stats["norm"]
            self.mode_stats["fits"] += 1
            for key, value in extra.items():
                self.mode_stats[key] += value
        return torch.stack(layer_scores).mean(0)

    def object_metadata(self):
        metadata = super().object_metadata()
        metadata.update(
            mode_guard=self.mode_guard, identity_guard=self.identity_guard,
            guard_weight=self.guard_weight,
            correspondence_guard_diagnostics_including_warmup=dict(self.guard_stats),
            guard_anchor="same_query_QR1_global_shifted_bank_not_ground_truth",
            guard_is_normality_label=False,
            full_query_MC1_corrections_unchanged=True,
            guard_cost="QR1_mode_gap_plus_optional_within_mode_identity_gap",
            guard_diagnostic_scope="all_query_tokens_layers_resolutions_including_warmup; raw_cosine_score_units",
            guard_limit="wrong_QR1_matches_can_be_preserved; original_reference_anomaly_absorption_is_unprotected",
            score_definition=("min_MC1_cosine_distance_plus_QR1_correspondence_cost_fixed_support_MAD"
                              if self.mode_guard else metadata["score_definition"]),
        )
        return metadata
