"""Query-conditioned normal-reference translation on the fixed R1 pipeline.

QR0 is exactly R1. QR1 fits one bounded translation per layer/resolution to
initial R1 nearest-neighbour residuals. QR2 trims large residuals before this
fit. QR3 estimates each translation outside the scored spatial block + halo.

This is a one-step, fixed-correspondence estimator, not a synthetic classifier,
free patch reconstruction, or a reproduction of FastRef. Translations have
rank one across spatial positions but are full-channel vectors. They are
discarded after each query; neither the backbone nor the normal bank is updated.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from fmad.methods.reference_matching import ReferenceMatching, _distance_chunks
from fmad.registry import METHOD_REGISTRY


def nearest_residuals(query, bank, chunk):
    """R1's initial correspondences, independently for each layer."""
    residuals = []
    for start in range(0, len(query), chunk):
        values = query[start:start + chunk]
        selected = (values @ bank.T).argmax(1)
        residuals.append(values - bank[selected])
    return torch.cat(residuals)


def controlled_translation(delta, mode, seed):
    """Direction-only ablation; preserves pre-normalization translation norm.

The isolated CPU generator does not affect support/bank RNGs. A fixed random
direction per layer/grid is deliberately independent of query content; its
magnitude still comes from that query's fitted translation.
"""
    if mode == "fitted":
        return delta
    if mode == "opposite":
        return -delta
    if mode != "random":
        raise ValueError("invalid translation control")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    direction = torch.randn(delta.shape, generator=generator, dtype=torch.float32).to(delta)
    return direction / direction.norm().clamp_min(1e-12) * delta.norm()


def spatial_folds(grid, blocks, halo, device):
    """Yield score/fit masks; each token is scored exactly once.

    A held block and its halo are never used to estimate its own transform.
    This prevents direct self-fitting, not global-attention feature leakage.
    """
    height, width = grid
    rows = torch.arange(height, device=device)[:, None]
    cols = torch.arange(width, device=device)[None, :]
    for by in range(blocks):
        top, bottom = by * height // blocks, (by + 1) * height // blocks
        for bx in range(blocks):
            left, right = bx * width // blocks, (bx + 1) * width // blocks
            score = (rows >= top) & (rows < bottom) & (cols >= left) & (cols < right)
            if not bool(score.any()):
                continue
            excluded = ((rows >= top - halo) & (rows < bottom + halo)
                        & (cols >= left - halo) & (cols < right + halo))
            yield score.flatten(), (~excluded).flatten()


def fit_translation(residuals, *, keep_fraction, ridge, max_norm, min_tokens):
    """Solve a trimmed, ridge-shrunk mean followed by an L2-ball projection.

    Selection uses initial R1 residual length, not labels or a claim that the
    retained patches are correct normals. At least min_tokens are required;
    insufficient context returns the identity without relaxing the trim rule.

    delta = argmin ||R_selected - delta||^2 / n + ridge * ||delta||^2
    subject to ||delta|| <= max_norm.
    """
    count = math.floor(len(residuals) * keep_fraction)
    if count < min_tokens:
        return residuals.new_zeros(residuals.shape[1]), {
            "available_tokens": len(residuals), "retained_tokens": 0,
            "fallback": True, "clipped": False, "norm": 0.0,
        }
    selected = residuals
    if count < len(residuals):
        order = torch.argsort(residuals.square().sum(1), stable=True)
        selected = residuals[order[:count]]
    delta = selected.mean(0) / (1.0 + ridge)
    norm = torch.linalg.vector_norm(delta)
    clipped = bool(norm > max_norm)
    delta = delta * (max_norm / norm.clamp_min(1e-12)).clamp_max(1.0)
    if not bool(torch.isfinite(delta).all()):
        raise RuntimeError("nonfinite query reference translation")
    return delta, {
        "available_tokens": len(residuals), "retained_tokens": count,
        "fallback": False, "clipped": clipped,
        "norm": float(torch.linalg.vector_norm(delta)),
    }


@METHOD_REGISTRY.register("query_reference")
class QueryReference(ReferenceMatching):
    """Normal-only R1 fit; ephemeral, unlabelled single-query adaptation."""

    def __init__(self, backbone, config: dict):
        if config.get("variant", "R1") != "R1":
            raise ValueError("query_reference requires variant R1")
        super().__init__(backbone, {**config, "variant": "R1"})
        for name in ("query_adaptation", "robust_trim", "spatial_crossfit"):
            value = config.get(name, False)
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")
            setattr(self, name, value)
        if (self.robust_trim or self.spatial_crossfit) and not self.query_adaptation:
            raise ValueError("robust_trim/spatial_crossfit require query_adaptation")
        self.keep_fraction = self._number(config, "keep_fraction", 0.75, 0.0, 1.0)
        self.ridge = self._number(config, "ridge", 1.0, 0.0, None, allow_zero=True)
        self.max_norm = self._number(config, "max_shift_norm", 0.1, 0.0, 1.0)
        if self.max_norm >= 1:
            raise ValueError("max_shift_norm must be < 1 for unit-normalized references")
        self.blocks = self._integer(config, "crossfit_blocks", 2, 2)
        self.halo = self._integer(config, "crossfit_halo", 1, 0)
        self.min_tokens = self._integer(config, "min_fit_tokens", 32, 1)
        self.translation_control = config.get("translation_control", "fitted")
        if self.translation_control not in {"fitted", "random", "opposite"}:
            raise ValueError("invalid translation_control")
        if not self.query_adaptation and self.translation_control != "fitted":
            raise ValueError("direction controls require query_adaptation")
        self._reset_diagnostics()

    @staticmethod
    def _number(config, name, default, lower, upper, *, allow_zero=False):
        value = config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number")
        value = float(value)
        if (not math.isfinite(value) or value < lower or
                (not allow_zero and value == lower) or
                (upper is not None and value > upper)):
            raise ValueError(f"invalid {name}")
        return value

    @staticmethod
    def _integer(config, name, default, minimum):
        value = config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
        return value

    def _reset_diagnostics(self):
        self.translation_stats = dict(
            fits=0, fallback_fits=0, clipped_fits=0,
            available_tokens=0, retained_tokens=0, norm_sum=0.0,
        )

    def fit(self, support_paths, context):
        super().fit(support_paths, context)
        self._reset_diagnostics()
        self.log_training(
            step=None, event="query_reference_ready", query_adaptation=self.query_adaptation,
            robust_trim=self.robust_trim, spatial_crossfit=self.spatial_crossfit,
            synthetic_training=False, trainable_parameters=0,
        )

    @torch.no_grad()
    def _score_tokens(self, layers, grid, bank, *, normalize=True):
        if not self.query_adaptation:
            return super()._score_tokens(layers, grid, bank, normalize=normalize)
        layer_scores = []
        for index, (query, reference) in enumerate(zip(layers, bank["layers"])):
            residuals = nearest_residuals(query, reference, self.chunk)
            if self.spatial_crossfit:
                folds = spatial_folds(grid, self.blocks, self.halo, query.device)
            else:
                all_tokens = torch.ones(len(query), dtype=torch.bool, device=query.device)
                folds = [(all_tokens, all_tokens)]
            scores = query.new_empty(len(query))
            for score_mask, fit_mask in folds:
                delta, stats = fit_translation(
                    residuals[fit_mask],
                    keep_fraction=self.keep_fraction if self.robust_trim else 1.0,
                    ridge=self.ridge, max_norm=self.max_norm, min_tokens=self.min_tokens,
                )
                # A temporary bank is bounded in size and discarded each fold.
                # Never mutate reference, shared backbone, or support MAD scales.
                delta = controlled_translation(
                    delta, self.translation_control,
                    20260920 + index * 1000003 + grid[0] * 1009 + grid[1])
                shifted = reference if stats["fallback"] else F.normalize(reference + delta, dim=-1)
                scores[score_mask] = _distance_chunks(query[score_mask], shifted, self.chunk)
                self.translation_stats["fits"] += 1
                self.translation_stats["fallback_fits"] += int(stats["fallback"])
                self.translation_stats["clipped_fits"] += int(stats["clipped"])
                self.translation_stats["available_tokens"] += stats["available_tokens"]
                self.translation_stats["retained_tokens"] += stats["retained_tokens"]
                self.translation_stats["norm_sum"] += stats["norm"]
            layer_scores.append(scores / (self._active_scale(index, query.device)
                                          if normalize else 1.0))
        return torch.stack(layer_scores).mean(0)

    def object_metadata(self):
        metadata = super().object_metadata()
        stats = dict(self.translation_stats)
        stats["mean_translation_norm"] = stats.pop("norm_sum") / max(stats["fits"], 1)
        metadata.update(
            translation_control=self.translation_control,
            translation_control_norm="matched_before_bank_unit_normalization_not_equal_score_magnitude",
            query_adaptation=self.query_adaptation, robust_trim=self.robust_trim,
            spatial_crossfit=self.spatial_crossfit, keep_fraction=self.keep_fraction,
            ridge=self.ridge, max_shift_norm=self.max_norm,
            crossfit_blocks=self.blocks, crossfit_halo=self.halo,
            min_fit_tokens=self.min_tokens, synthetic_training=False,
            correspondence="initial_R1_independent_layer_nn_fixed_for_translation_fit",
            transform="bounded_shared_full_channel_translation_then_unit_normalization",
            fit_objective="trimmed_mean_squared_feature_residual_plus_ridge",
            fit_objective_is_cosine_score=False,
            adaptation_scope="single_query_layer_resolution_and_optional_held_block",
            state_reset="every_query_no_query_tokens_added_to_bank",
            inference_uses_query_context=self.query_adaptation,
            query_labels_used=False, persistent_adapted_state=False,
            translation_parameters_per_fit=(next(iter(self.banks.values()))["layers"][0].shape[1]
                                            if self.banks and self.query_adaptation else 0),
            translation_diagnostics_including_warmup=stats,
            score_definition=("mean_layer_cosine_nn_to_query_translated_bank_fixed_support_MAD"
                              if self.query_adaptation else metadata["score_definition"]),
        )
        return metadata
