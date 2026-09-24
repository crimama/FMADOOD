"""Attenuate QR1 using agreement of spatially separated residual means.

Agreement is not a normality label or statistically independent evidence:
global-attention features and spatially repeated defects can agree as well.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from fmad.methods.query_reference import QueryReference, fit_translation, nearest_residuals
from fmad.methods.reference_matching import _distance_chunks
from fmad.registry import METHOD_REGISTRY


def spatial_confidence(residuals, grid, *, min_tokens, halo=1):
    """Minimum positive, magnitude-sensitive agreement across the two axes."""
    if len(grid) != 2 or grid[0] * grid[1] != len(residuals):
        raise ValueError("residual count does not match patch grid")
    if min_tokens < 1 or halo < 0:
        raise ValueError("invalid confidence context limits")
    if not bool(torch.isfinite(residuals).all()):
        raise ValueError("nonfinite confidence residuals")
    values = residuals.reshape(*grid, residuals.shape[-1])
    agreements = []
    insufficient = 0
    for axis, size in enumerate(grid):
        midpoint = size // 2
        left = values.narrow(axis, 0, max(0, midpoint - halo)).reshape(-1, residuals.shape[-1])
        start = min(size, midpoint + halo)
        right = values.narrow(axis, start, size - start).reshape(-1, residuals.shape[-1])
        if min(len(left), len(right)) < min_tokens:
            agreements.append(residuals.new_zeros(()))
            insufficient += 1
            continue
        a, b = left.mean(0), right.mean(0)
        value = 2 * (a * b).sum() / (a.square().sum() + b.square().sum()).clamp_min(1e-12)
        agreements.append(value.clamp(0., 1.))
    confidence = torch.stack(agreements).min()
    return confidence, dict(vertical=float(agreements[0]), horizontal=float(agreements[1]),
                           insufficient_axes=insufficient)


@METHOD_REGISTRY.register("spatial_confidence")
class SpatialConfidence(QueryReference):
    def __init__(self, backbone, config):
        super().__init__(backbone, config)
        self.confidence_enabled = config.get("confidence_enabled", True)
        if type(self.confidence_enabled) is not bool:
            raise ValueError("confidence_enabled must be boolean")
        if (not self.query_adaptation or self.robust_trim or self.spatial_crossfit
                or self.translation_control != "fitted"):
            raise ValueError("spatial_confidence requires unmodified QR1 translation")
        if (self.bank_bags != 1 or self.bank_selection != "random" or self.bank_outlier_trim != 0
                or self.score_normalization != "mad" or self.layer_fusion != "mean"
                or self.scale_fusion != "mean" or self.local_reliability != "off"
                or self.prediction_postprocess != "none"):
            raise ValueError("spatial_confidence requires original R1 pipeline")

    def _reset_diagnostics(self):
        super()._reset_diagnostics()
        self.confidence_records = []
        self._query_confidence = []

    def predict(self, test_image_path):
        self._query_confidence = []
        result = super().predict(test_image_path)
        if self.confidence_enabled:
            self.confidence_records.append(dict(query=str(test_image_path),
                                                estimates=self._query_confidence))
        self._query_confidence = []
        return result

    @torch.no_grad()
    def _score_tokens(self, layers, grid, bank, *, normalize=True):
        if not self.confidence_enabled:
            return super()._score_tokens(layers, grid, bank, normalize=normalize)
        scores = []
        for index, (query, reference) in enumerate(zip(layers, bank["layers"])):
            residuals = nearest_residuals(query, reference, self.chunk)
            delta, stats = fit_translation(residuals, keep_fraction=1., ridge=self.ridge,
                                          max_norm=self.max_norm, min_tokens=self.min_tokens)
            confidence, evidence = spatial_confidence(residuals, grid, min_tokens=self.min_tokens)
            applied = confidence * delta
            identity = stats["fallback"] or float(confidence) == 0.
            shifted = reference if identity else F.normalize(reference + applied, dim=-1)
            values = _distance_chunks(query, shifted, self.chunk)
            scores.append(values / (self._active_scale(index, query.device) if normalize else 1.))
            self.translation_stats["fits"] += 1
            self.translation_stats["fallback_fits"] += int(identity)
            self.translation_stats["clipped_fits"] += int(stats["clipped"])
            self.translation_stats["available_tokens"] += stats["available_tokens"]
            self.translation_stats["retained_tokens"] += stats["retained_tokens"]
            self.translation_stats["norm_sum"] += float(applied.norm())
            self._query_confidence.append(dict(layer=self.layers[index], grid=list(grid),
                confidence=float(confidence), qr1_norm=stats["norm"],
                applied_norm=float(applied.norm()), identity=identity, **evidence))
        return torch.stack(scores).mean(0)

    def object_metadata(self):
        metadata = super().object_metadata()
        metadata.update(confidence_enabled=self.confidence_enabled,
            confidence_rule="min_axes_clamp_2dot_over_sum_squared_norms",
            confidence_partition="top_bottom_and_left_right_midpoint_halo1",
            confidence_is_normality_label=False,
            confidence_records_including_warmup_and_calibration=self.confidence_records,
            transform="spatial_agreement_times_QR1_translation_then_bank_unit_normalization",
            confidence_scope="unlabelled_current_query_only_no_learned_threshold",
            confidence_limit="global_attention_and_repeated_defects_can_agree",
            score_definition=("mean_layer_cosine_nn_to_spatial_confidence_translated_bank_fixed_support_MAD"
                              if self.confidence_enabled else metadata["score_definition"]))
        return metadata
