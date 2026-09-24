"""Residual-consensus weighting of QR1's ephemeral shared translation.

Disjoint normal-support groups provide independent reference *sets*, not
independent statistical evidence or normality labels. Large, globally shared
defects can also agree across these sets. No novelty or robustness guarantee
is claimed. The backbone, initial R1 correspondences and final scorer stay fixed.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from fmad.methods.query_reference import QueryReference, fit_translation
from fmad.methods.reference_matching import _distance_chunks
from fmad.registry import METHOD_REGISTRY


def support_groups(owners, count, device):
    """Round-robin whole supports into disjoint, deterministic reference sets."""
    sources = sorted(set(owners))
    if len(sources) < count:
        raise ValueError("consensus_groups cannot exceed the number of normal supports")
    assignment = {source: index % count for index, source in enumerate(sources)}
    labels = torch.tensor([assignment[source] for source in owners], device=device)
    return [(labels == group).nonzero(as_tuple=True)[0] for group in range(count)]


def consensus_residuals(query, reference, groups, chunk):
    """Return original R1 residuals/identities and mean pairwise direction cosine.

    One chunk-by-bank similarity matrix is reused for all groups. Only a running
    direction sum is kept, never a query-by-group-by-channel residual tensor.
    Zero residuals contribute zero directional evidence.
    """
    residuals, identities, agreements = [], [], []
    for start in range(0, len(query), chunk):
        values = query[start:start + chunk]
        similarity = values @ reference.T
        selected = similarity.argmax(1)
        direction_sum = torch.zeros_like(values)
        squared_sum = values.new_zeros(len(values))
        for indices in groups:
            match = indices[similarity.index_select(1, indices).argmax(1)]
            direction = F.normalize(values - reference[match], dim=-1)
            direction_sum += direction
            squared_sum += direction.square().sum(1)
        pairwise = (direction_sum.square().sum(1) - squared_sum) / (len(groups) * (len(groups) - 1))
        agreements.append(pairwise.clamp(0.0, 1.0))
        residuals.append(values - reference[selected])
        identities.append(selected)
    return torch.cat(residuals), torch.cat(identities), torch.cat(agreements)


def translation_weights(agreement, identities, *, floor, balance):
    weights = floor + (1.0 - floor) * agreement
    if balance:
        # NN Voronoi-cell occupancy is a redundancy proxy, NOT a semantic mode
        # label. Square-root balancing limits repeated-background influence
        # without giving a singleton the entire mass of a populated cell.
        _, inverse, counts = identities.unique(return_inverse=True, return_counts=True)
        weights = weights / counts[inverse].to(weights.dtype).sqrt()
    return weights


def weighted_translation(residuals, weights, *, ridge, max_norm, min_tokens, match_norm):
    baseline, stats = fit_translation(residuals, keep_fraction=1.0, ridge=ridge,
                                      max_norm=max_norm, min_tokens=min_tokens)
    total = weights.sum()
    extras = dict(effective_tokens=float(total.square() / weights.square().sum().clamp_min(1e-12)),
                  baseline_norm=stats["norm"], weighted_unclipped_norm=0.0,
                  direction_cosine=0.0, norm_match_fallback=0)
    if stats["fallback"]:
        return baseline, stats, extras
    delta = (residuals * weights[:, None]).sum(0) / total / (1.0 + ridge)
    norm = torch.linalg.vector_norm(delta)
    extras["weighted_unclipped_norm"] = float(norm)
    extras["direction_cosine"] = float(F.cosine_similarity(delta[None], baseline[None]))
    if match_norm:
        # Control correction strength per image/layer/resolution. If weighted
        # directions cancel, preserve QR1 instead of amplifying numerical noise.
        if bool(norm <= 1e-12):
            delta = baseline
            extras["norm_match_fallback"] = 1
        else:
            delta = delta * (torch.linalg.vector_norm(baseline) / norm)
        stats["clipped"] = False  # target norm already obeys QR1's cap
    else:
        stats["clipped"] = bool(norm > max_norm)
        delta = delta * (max_norm / norm.clamp_min(1e-12)).clamp_max(1.0)
    if not bool(torch.isfinite(delta).all()):
        raise RuntimeError("nonfinite consensus reference translation")
    stats["norm"] = float(torch.linalg.vector_norm(delta))
    return delta, stats, extras


@METHOD_REGISTRY.register("reference_consensus")
class ReferenceConsensus(QueryReference):
    """RC0=QR1; RC1=consensus; RC2=norm control; RC3=cell balancing."""

    def __init__(self, backbone, config):
        config = {"query_adaptation": True, **config}
        super().__init__(backbone, config)
        if not self.query_adaptation or self.robust_trim or self.spatial_crossfit:
            raise ValueError("reference_consensus starts from QR1, without trim/crossfit")
        for name in ("consensus_weighting", "match_translation_norm", "balance_reference_cells"):
            value = config.get(name, False)
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")
            setattr(self, name, value)
        if (self.match_translation_norm or self.balance_reference_cells) and not self.consensus_weighting:
            raise ValueError("norm matching and cell balancing require consensus_weighting")
        self.group_count = self._integer(config, "consensus_groups", 4, 2)
        self.weight_floor = self._number(config, "consensus_weight_floor", .1, 0.0, 1.0)

    def _reset_diagnostics(self):
        super()._reset_diagnostics()
        self.consensus_stats = dict(fits=0, tokens=0, agreement_sum=0.0,
                                   low_agreement_tokens=0, weight_sum=0.0,
                                   effective_tokens_sum=0.0, baseline_norm_sum=0.0,
                                   weighted_unclipped_norm_sum=0.0,
                                   direction_cosine_sum=0.0, norm_match_fallbacks=0)

    def fit(self, support_paths, context):
        super().fit(support_paths, context)
        if self.consensus_weighting:
            for bank in self.banks.values():
                bank["consensus_groups"] = support_groups(bank["owners"], self.group_count, self.device)
        self.log_training(step=None, event="reference_consensus_ready",
                          consensus_weighting=self.consensus_weighting,
                          match_translation_norm=self.match_translation_norm,
                          balance_reference_cells=self.balance_reference_cells,
                          consensus_groups=self.group_count, trainable_parameters=0)

    @torch.no_grad()
    def _score_tokens(self, layers, grid, bank, *, normalize=True):
        if not self.consensus_weighting:
            return super()._score_tokens(layers, grid, bank, normalize=normalize)
        layer_scores = []
        for index, (query, reference) in enumerate(zip(layers, bank["layers"])):
            residuals, identities, agreement = consensus_residuals(
                query, reference, bank["consensus_groups"], self.chunk)
            weights = translation_weights(agreement, identities, floor=self.weight_floor,
                                          balance=self.balance_reference_cells)
            delta, stats, extras = weighted_translation(
                residuals, weights, ridge=self.ridge, max_norm=self.max_norm,
                min_tokens=self.min_tokens, match_norm=self.match_translation_norm)
            shifted = reference if stats["fallback"] else F.normalize(reference + delta, dim=-1)
            scores = _distance_chunks(query, shifted, self.chunk)
            layer_scores.append(scores / (self._active_scale(index, query.device) if normalize else 1.0))
            self.translation_stats["fits"] += 1
            self.translation_stats["fallback_fits"] += int(stats["fallback"])
            self.translation_stats["clipped_fits"] += int(stats["clipped"])
            self.translation_stats["available_tokens"] += stats["available_tokens"]
            self.translation_stats["retained_tokens"] += stats["retained_tokens"]
            self.translation_stats["norm_sum"] += stats["norm"]
            self.consensus_stats["fits"] += 1
            self.consensus_stats["tokens"] += len(query)
            self.consensus_stats["agreement_sum"] += float(agreement.sum())
            self.consensus_stats["low_agreement_tokens"] += int((agreement < .25).sum())
            self.consensus_stats["weight_sum"] += float(weights.sum())
            for key in ("effective_tokens", "baseline_norm", "weighted_unclipped_norm", "direction_cosine"):
                self.consensus_stats[f"{key}_sum"] += extras[key]
            self.consensus_stats["norm_match_fallbacks"] += extras["norm_match_fallback"]
        return torch.stack(layer_scores).mean(0)

    def object_metadata(self):
        metadata = super().object_metadata()
        metadata.update(
            consensus_weighting=self.consensus_weighting,
            match_translation_norm=self.match_translation_norm,
            balance_reference_cells=self.balance_reference_cells,
            consensus_groups=self.group_count, consensus_weight_floor=self.weight_floor,
            reference_partition="whole_normal_support_round_robin_disjoint_groups",
            consensus_is_normality_label=False,
            cell_balance="inverse_sqrt_query_occupancy_of_initial_R1_bank_token",
            fit_objective=("consensus_weighted_squared_residual_plus_ridge"
                           if self.consensus_weighting else metadata["fit_objective"]),
            norm_control="QR1_capped_norm" if self.match_translation_norm else "none",
            consensus_diagnostics_including_warmup=dict(self.consensus_stats),
        )
        return metadata
