"""Spatially cross-validated mode innovations on the fixed QR1 reference.

VR0/VR1 delegate exactly to QR1/MC1. VR2 estimates each scored block's extra
mode correction outside that block and its halo. VR3 admits a correction only
when leave-another-block-out proposals improve cosine-NN distances in at least
two witness blocks. No source-PCA restriction or residual-magnitude normal label.

This protects only the added innovations: the common QR1 global translation
and DINO features still depend on the whole query. Spatially repeated defects
can pass validation; improvement of unlabelled NN fit is not proof of normality.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from fmad.methods.mode_reference import ModeReference
from fmad.methods.query_reference import fit_translation, spatial_folds
from fmad.methods.reference_matching import _distance_chunks
from fmad.registry import METHOD_REGISTRY


def context_innovations(residuals, modes, count, *, ridge, prior, minimum, min_mode):
    """Unclipped MC1 innovations, centered only on the supplied context."""
    innovations = residuals.new_zeros((count, residuals.shape[1]))
    valid = torch.zeros(count, dtype=torch.bool, device=residuals.device)
    if len(residuals) < minimum:
        return innovations, valid
    mean = residuals.mean(0)
    for k in range(count):
        values = residuals[modes == k]
        if len(values) >= min_mode:
            innovations[k] = (values - mean).sum(0) / (len(values) + prior) / (1.0 + ridge)
            valid[k] = True
    return innovations, valid


def bounded_deltas(baseline, innovations, max_norm):
    values = baseline + innovations
    norms = values.norm(dim=-1, keepdim=True)
    return values * (max_norm / norms.clamp_min(1e-12)).clamp_max(1.0)


def single_mode_gain(query, reference, labels, global_reference, candidate, mode, chunk):
    """True cosine-NN gain replacing ONE cell, with every other cell fixed.

    Search the whole bank, not only the original correspondence. Final scoring
    can move several admitted cells together, so this is not a joint-score or
    anomaly-preservation guarantee. Mean gain is relative to mean QR1 distance.
    """
    before = _distance_chunks(query, global_reference, chunk)
    selected = labels == mode
    changed = F.normalize(reference[selected] + candidate, dim=-1)
    after = _distance_chunks(query, changed, chunk)
    if bool((~selected).any()):
        after = torch.minimum(after, _distance_chunks(query, global_reference[~selected], chunk))
    gain = (before - after).mean() / before.mean().clamp_min(1e-6)
    if not bool(torch.isfinite(gain)):
        raise RuntimeError("nonfinite cross-region validation gain")
    return float(gain)


def transfer_admission(query, residuals, modes, reference, geometry, baseline,
                       context_mask, folds, eligible, *, ridge, prior, minimum,
                       min_mode, max_norm, chunk, min_witness, witness_cap,
                       min_blocks, min_gain):
    """Nested spatial exclusion: scored block outside both donors and witnesses.

    Validation donors also exclude the witness block plus halo. The full-query
    QR1 baseline remains fixed in every comparison; no independence claim about
    that baseline or the globally attentive feature extractor is made.
    """
    count = len(geometry["centers"])
    gains = [[] for _ in range(count)]
    global_reference = F.normalize(reference + baseline, dim=-1)
    witness_tokens = 0
    for witness, outside_witness in folds:
        validation = context_mask & witness
        if not bool(validation.any()):
            continue
        training = context_mask & outside_witness
        innovations, valid = context_innovations(
            residuals[training], modes[training], count, ridge=ridge, prior=prior,
            minimum=minimum, min_mode=min_mode)
        candidates = bounded_deltas(baseline, innovations, max_norm)
        for k in range(count):
            if not bool(eligible[k] & valid[k]):
                continue
            indices = torch.where(validation & (modes == k))[0]
            if len(indices) < min_witness:
                continue
            if len(indices) > witness_cap:
                # Bounded, spatially spread witnesses without random-state use.
                positions = torch.arange(witness_cap, device=query.device) * len(indices) // witness_cap
                indices = indices[positions]
            witness_tokens += len(indices)
            gains[k].append(single_mode_gain(
                query[indices], reference, geometry["labels"], global_reference,
                candidates[k], k, chunk))
    accepted = eligible.clone()
    stats = dict(validation_pairs=0, successful_pairs=0, insufficient_modes=0,
                 witness_tokens=witness_tokens, relative_gain_sum=0.0)
    for k, values in enumerate(gains):
        if not bool(eligible[k]):
            continue
        successes = sum(gain >= min_gain for gain in values)
        stats["validation_pairs"] += len(values)
        stats["successful_pairs"] += successes
        stats["relative_gain_sum"] += sum(values)
        if len(values) < min_blocks:
            stats["insufficient_modes"] += 1
        accepted[k] = (len(values) >= min_blocks and successes >= min_blocks
                       and sum(values) / max(len(values), 1) >= min_gain)
    return accepted, stats


@METHOD_REGISTRY.register("validated_reference")
class ValidatedReference(ModeReference):
    def __init__(self, backbone, config):
        super().__init__(backbone, config)
        if self.normal_subspace or self.orthogonal_gate:
            raise ValueError("validated_reference does not use source subspaces or orthogonal gates")
        for name in ("innovation_crossfit", "transfer_validation"):
            value = config.get(name, False)
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")
            setattr(self, name, value)
        if self.innovation_crossfit and not self.mode_conditioning:
            raise ValueError("innovation_crossfit requires mode_conditioning")
        if self.transfer_validation and not self.innovation_crossfit:
            raise ValueError("transfer_validation requires innovation_crossfit")
        self.min_witness = self._integer(config, "validation_min_tokens", 4, 1)
        self.witness_cap = self._integer(config, "validation_tokens_per_mode", 16, 1)
        self.min_blocks = self._integer(config, "validation_min_blocks", 2, 2)
        self.min_gain = self._number(config, "validation_min_gain", .01, 0., 1.)
        if self.witness_cap < self.min_witness or self.min_blocks >= self.blocks**2:
            raise ValueError("invalid witness capacity or validation block count")

    def _reset_diagnostics(self):
        super()._reset_diagnostics()
        self.validation_stats = {key: 0 for key in (
            "score_folds", "proposed_modes", "accepted_modes", "rejected_modes",
            "validation_pairs", "successful_pairs", "insufficient_modes",
            "witness_tokens", "relative_gain_sum")}

    def fit(self, support_paths, context):
        super().fit(support_paths, context)
        self.log_training(step=None, event="validated_reference_ready",
                          innovation_crossfit=self.innovation_crossfit,
                          transfer_validation=self.transfer_validation, trainable_parameters=0)

    @torch.no_grad()
    def _score_tokens(self, layers, grid, bank, *, normalize=True):
        if not self.innovation_crossfit:
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
            baseline, stats = fit_translation(residuals, keep_fraction=1., ridge=self.ridge,
                                              max_norm=self.max_norm, min_tokens=self.min_tokens)
            self.translation_stats["fits"] += 1
            self.translation_stats["fallback_fits"] += int(stats["fallback"])
            self.translation_stats["clipped_fits"] += int(stats["clipped"])
            self.translation_stats["available_tokens"] += stats["available_tokens"]
            self.translation_stats["retained_tokens"] += stats["retained_tokens"]
            self.translation_stats["norm_sum"] += stats["norm"]
            folds = list(spatial_folds(grid, self.blocks, self.halo, query.device))
            scores = query.new_empty(len(query))
            count = len(geometry["centers"])
            for scored, context in folds:
                innovations, valid = context_innovations(
                    residuals[context], modes[context], count, ridge=self.ridge,
                    prior=self.prior_tokens, minimum=self.min_tokens, min_mode=self.min_mode_tokens)
                if stats["fallback"]:
                    innovations.zero_()
                    valid.zero_()
                accepted = valid
                if self.transfer_validation:
                    accepted, evidence = transfer_admission(
                        query, residuals, modes, reference, geometry, baseline, context, folds, valid,
                        ridge=self.ridge, prior=self.prior_tokens, minimum=self.min_tokens,
                        min_mode=self.min_mode_tokens, max_norm=self.max_norm, chunk=self.chunk,
                        min_witness=self.min_witness, witness_cap=self.witness_cap,
                        min_blocks=self.min_blocks, min_gain=self.min_gain)
                    for key, value in evidence.items():
                        self.validation_stats[key] += value
                applied = innovations * accepted[:, None]
                deltas = bounded_deltas(baseline, applied, self.max_norm)
                shifted = reference if stats["fallback"] else F.normalize(reference + deltas[geometry["labels"]], dim=-1)
                scores[scored] = _distance_chunks(query[scored], shifted, self.chunk)
                self.validation_stats["score_folds"] += 1
                self.validation_stats["proposed_modes"] += int(valid.sum())
                self.validation_stats["accepted_modes"] += int(accepted.sum())
                self.validation_stats["rejected_modes"] += int((valid & ~accepted).sum())
                self.mode_stats["fits"] += 1
                self.mode_stats["mode_slots"] += count
                self.mode_stats["active_modes"] += int(valid.sum())
                self.mode_stats["fallback_modes"] += int((~valid).sum())
                self.mode_stats["query_tokens"] += int(context.sum())
                self.mode_stats["evidence_mass"] += int(context.sum())
                self.mode_stats["innovation_before_norm_sum"] += float(innovations.norm(dim=1).sum())
                self.mode_stats["innovation_after_norm_sum"] += float(applied.norm(dim=1).sum())
                self.mode_stats["mode_global_difference_norm_sum"] += float((deltas - baseline).norm(dim=1).sum())
                self.mode_stats["mode_translation_norm_sum"] += float(deltas.norm(dim=1).sum())
                self.mode_stats["clipped_modes"] += int(((baseline + applied).norm(dim=1) > self.max_norm).sum())
            if not bool(torch.isfinite(scores).all()):
                raise RuntimeError("nonfinite validated reference score")
            layer_scores.append(scores / (self._active_scale(index, query.device) if normalize else 1.0))
        return torch.stack(layer_scores).mean(0)

    def object_metadata(self):
        metadata = super().object_metadata()
        metadata.update(
            innovation_crossfit=self.innovation_crossfit, transfer_validation=self.transfer_validation,
            validation_min_tokens=self.min_witness, validation_tokens_per_mode=self.witness_cap,
            validation_min_blocks=self.min_blocks, validation_min_gain=self.min_gain,
            validation_diagnostics_including_warmup=dict(self.validation_stats),
            validation_is_normality_label=False, full_query_QR1_component_unchanged=True,
            validation_objective="relative_mean_cosine_NN_gain_one_mode_changed_others_QR1",
            validation_selection="at_least_two_positive_witness_blocks_and_mean_gain_at_least_margin",
            validation_limit="global_QR1_and_backbone_remain_query_dependent; repeated_defects_can_pass",
        )
        if self.innovation_crossfit:
            metadata.update(
                fit_objective="context_centered_ridge_mode_innovation_outside_scored_block_and_halo",
                adaptation_scope="QR1_global_plus_held_block_mode_innovations_optional_nested_validation",
                transform="bounded_QR1_global_plus_admitted_context_innovations_then_unit_normalization",
                score_definition="held_block_cosine_NN_to_mode_deformed_bank_fixed_support_MAD",
                mode_diagnostics_scope="overlapping_outer_context_proposals_before_gate; after_norm_and_delta_after_gate",
            )
        return metadata
