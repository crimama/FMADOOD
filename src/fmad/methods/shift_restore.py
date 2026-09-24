"""L2/L2C shift-conditioned restoration over frozen patch features.

The K1 synthetic generator and source-excluded bank are retained only for
training-time synthetic selection.  When enabled, scoring is a restoration
residual, not the contextual nearest-neighbour metric.  Features are frozen
and held for one fitting step only; no dense maps or weights are persisted.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fmad.methods.boundary_synthetic import BoundarySynthetic
from fmad.methods.tail_contrast import ResidualAdapter
from fmad.methods.PSMA_MR import _apply_photometric, _sample_photometric_transform
from fmad.registry import METHOD_REGISTRY


def mask_coverage(mask, grid, device, return_fallback=False):
    """Return fractional foreground coverage on the feature-token grid.

    Area interpolation preserves partial coverage.  An explicit max-pool
    fallback is used only when a non-empty pixel mask would otherwise vanish
    on a very small grid; the fallback is recorded in metadata rather than
    silently changing the normal case.
    """
    value = torch.as_tensor(mask, device=device, dtype=torch.float32)[None, None]
    coverage = F.interpolate(value, size=grid, mode="area")[0, 0].clamp(0., 1.)
    fallback = float(coverage.sum()) == 0. and bool(np.asarray(mask).any())
    if fallback:
        coverage = F.adaptive_max_pool2d(value, grid)[0, 0]
    return (coverage, fallback) if return_fallback else coverage


def balanced_reconstruction(prediction, target, coverage):
    """Average feature MSE over fractional foreground and background regions."""
    error = (prediction - target).square().mean(dim=-1)
    inside = coverage
    outside = 1. - coverage
    inside_loss = (error * inside).sum() / inside.sum().clamp_min(1e-8)
    outside_loss = (error * outside).sum() / outside.sum().clamp_min(1e-8)
    # A missing region contributes zero, while the two available regions keep
    # equal weight.  Synthetic masks normally contain both regions.
    terms = []
    if float(inside.sum()) > 0:
        terms.append(inside_loss)
    if float(outside.sum()) > 0:
        terms.append(outside_loss)
    if not terms:
        return error.mean()
    return torch.stack(terms).mean()


@METHOD_REGISTRY.register("shift_restore")
class ShiftRestore(BoundarySynthetic):
    """Restore a normal feature under either the same shift or canonical target."""

    def __init__(self, backbone, config):
        super().__init__(backbone, config)
        self.restoration_enabled = config.get("shift_restore", True)
        if type(self.restoration_enabled) is not bool:
            raise ValueError("shift_restore must be a boolean")
        self.restoration_target = config.get("restoration_target", "same_shift")
        if self.restoration_target not in {"same_shift", "canonical"}:
            raise ValueError("restoration_target must be same_shift or canonical")
        self.restoration_weight = float(config.get("restoration_weight", 1.0))
        if not np.isfinite(self.restoration_weight) or self.restoration_weight <= 0:
            raise ValueError("restoration_weight must be finite and positive")
        self.restorer = None
        self._used_mask_fallback = False

    def _restore_loss(self, source, target, mask=None, grid=None):
        prediction = self.restorer(source)
        if mask is None:
            return (prediction - target).square().mean()
        coverage, fallback = mask_coverage(mask, grid, source.device, return_fallback=True)
        self._used_mask_fallback = self._used_mask_fallback or fallback
        return balanced_reconstruction(prediction, target, coverage.reshape(-1))

    def fit(self, support_paths, context):
        # Exact K1 delegation is intentional: this branch must preserve K1's
        # generator, optimizer, metric, history and prediction behavior.
        self.restorer = None
        if not self.restoration_enabled:
            return super().fit(support_paths, context)

        self.adapter = None
        images, rng, _, _ = self._prepare_fit(support_paths, context)
        self.records.clear()
        dimension = self.banks[self.resolutions[0]][0].shape[1]
        # BoundarySynthetic creates its contextual metric for K1 selection;
        # restoration must not retain or optimize that unused scoring module.
        self.metric = None
        self.restorer = ResidualAdapter(dimension, int(self.config.get("adapter_rank", 16))).to(self.device)
        parameters = list(self.restorer.parameters())
        self.parameter_count = sum(p.numel() for p in parameters)
        optimizer = torch.optim.AdamW(parameters,
                                      lr=float(self.config.get("learning_rate", .001)),
                                      weight_decay=0.)
        self.history = []
        self._used_mask_fallback = False
        for step in range(self.steps):
            source = step % len(images)
            family_index = (step // len(images)) % 2
            image = images[source]
            anomaly, mask, family = self.training_anomaly(image, rng, family_index, step)
            transform = _sample_photometric_transform(rng, 1.)
            optimizer.zero_grad(set_to_none=True)
            normal_values, synthetic_values = [], []
            resolution_count = len(self.resolutions)
            for resolution in self.resolutions:
                views = (image, _apply_photometric(image, transform), anomaly,
                         _apply_photometric(anomaly, transform))
                with torch.no_grad():
                    extracted = [self.extract(view, resolution) for view in views]
                clean, shifted, defect, shifted_defect = [item[0] for item in extracted]
                grid = extracted[0][1]
                canonical = clean.detach()
                shifted_target = shifted.detach()
                normal_identity = self._restore_loss(clean, canonical)
                normal_shift = self._restore_loss(
                    shifted, shifted_target if self.restoration_target == "same_shift" else canonical)
                defect_identity = self._restore_loss(defect, canonical, mask, grid)
                defect_shift = self._restore_loss(
                    shifted_defect,
                    shifted_target if self.restoration_target == "same_shift" else canonical,
                    mask, grid)
                scale = 1. / (2. * resolution_count)
                (normal_identity * scale).backward()
                (normal_shift * scale).backward()
                (defect_identity * scale * self.restoration_weight).backward()
                (defect_shift * scale * self.restoration_weight).backward()
                normal_values.extend((float(normal_identity.detach()), float(normal_shift.detach())))
                synthetic_values.extend((float(defect_identity.detach()), float(defect_shift.detach())))
                del extracted, clean, shifted, defect, shifted_defect
            normal_loss = sum(normal_values) / len(normal_values)
            synthetic_loss = sum(synthetic_values) / len(synthetic_values)
            total_value = normal_loss + self.restoration_weight * synthetic_loss
            if not np.isfinite(total_value):
                raise RuntimeError("nonfinite shift restoration loss")
            grad_norm = nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
            optimizer.step()
            row = dict(loss=float(total_value), normal_restore_loss=float(normal_loss),
                       synthetic_restore_loss=float(synthetic_loss),
                       normal_identity_error=float(np.mean(normal_values[0::2])),
                       normal_shift_error=float(np.mean(normal_values[1::2])),
                       defect_identity_error=float(np.mean(synthetic_values[0::2])),
                       defect_shift_error=float(np.mean(synthetic_values[1::2])),
                       gradient_norm=float(grad_norm), source=source,
                       synthetic_family=family, resolutions=list(self.resolutions),
                       restoration_target=self.restoration_target,
                       mask_fallback=bool(self._used_mask_fallback))
            row.update(self.training_diagnostics())
            self.history.append(row)
            self.log_training(step, **row)
        self.restorer.eval()
        self.records.clear()

    def predict(self, test_image_path):
        if not self.restoration_enabled:
            return super().predict(test_image_path)
        from fmad.methods.PSMA_MR import _read_rgb
        image = _read_rgb(test_image_path)
        total = None
        with torch.no_grad():
            for resolution in self.resolutions:
                features, grid = self.extract(image, resolution)
                score = (features - self.restorer(features)).square().sum(dim=-1)
                native = F.interpolate(score.reshape(1, 1, *grid), size=image.shape[:2],
                                        mode="bilinear", align_corners=False)[0, 0]
                total = native if total is None else total + native
            result = (total / len(self.resolutions)).cpu().numpy()
        if not np.isfinite(result).all():
            raise RuntimeError("nonfinite shift restoration prediction")
        return result, float(np.max(result))

    def object_metadata(self):
        metadata = super().object_metadata()
        metadata.update(shift_restore=self.restoration_enabled,
                        restoration_target=self.restoration_target,
                        restoration_module="low_rank_residual_on_normalized_patch_features",
                        restoration_parameters=self.parameter_count if self.restoration_enabled else 0,
                        trainable_parameters=self.parameter_count,
                        score_definition="mean_resolution_squared_l2_restoration_residual"
                        if self.restoration_enabled else metadata.get("score_definition"),
                        restoration_mask_weighting="area_fraction_foreground_background_equal",
                        tiny_mask_fallback="adaptive_max_pool_only_when_area_empty",
                        source_excluded_score_training=False if self.restoration_enabled else
                        metadata.get("source_excluded_score_training", True),
                        support_exclusion_for_selection="retained_k1_boundary_teacher_only",
                        query_pair_required=False)
        return metadata
