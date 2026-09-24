"""Normal-only closed-form contraction with stable local-contrast protection.

The protected subspace is preserved before, NOT after, final unit normalization.
No anomaly labels, synthetic defects, target samples or gradients are used.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from fmad.methods.PSMA_MR import _read_rgb
from fmad.methods.query_reference import QueryReference
from fmad.methods.reference_matching import ReferenceMatching
from fmad.registry import METHOD_REGISTRY


def even_indices(count, cap, device):
    if count <= cap:
        return torch.arange(count, device=device)
    return torch.linspace(0, count - 1, cap, device=device).long()


def photometric_views(image):
    for gain in ((0.7, 0.7, 0.7), (0.8, 1.0, 1.2)):
        yield np.ascontiguousarray(np.rint(image.astype(np.float32) *
                                          np.asarray(gain, dtype=np.float32)).clip(0, 255).astype(np.uint8))


def edge_indices(grid, device):
    ids = torch.arange(grid[0] * grid[1], device=device).reshape(grid)
    return (torch.cat((ids[:, :-1].flatten(), ids[:-1, :].flatten())),
            torch.cat((ids[:, 1:].flatten(), ids[1:, :].flatten())))


def stable_edges(clean, views, grid, cap, cosine_min):
    left, right = edge_indices(grid, clean.device)
    diff = clean[right] - clean[left]
    norm = diff.norm(dim=-1)
    keep = norm > 1e-5
    for view in views:
        other = view[right] - view[left]
        other_norm = other.norm(dim=-1)
        cosine = (diff * other).sum(-1) / (norm * other_norm).clamp_min(1e-12)
        ratio = other_norm / norm.clamp_min(1e-12)
        keep &= (other_norm > 1e-5) & (cosine >= cosine_min) & (ratio >= .5) & (ratio <= 2.)
    eligible = keep.nonzero().flatten()
    selected = eligible[even_indices(len(eligible), cap, clean.device)]
    return left[selected], right[selected]


def moment_basis(moment, rank):
    eigenvalues, vectors = torch.linalg.eigh((moment + moment.T) * .5)
    threshold = max(1e-10, float(eigenvalues[-1]) * 1e-6)
    ids = (eigenvalues > threshold).nonzero().flatten()[-rank:]
    return vectors[:, ids]


def contraction_factor(nuisance, protected=None):
    if protected is None:
        return nuisance
    # An absent protection estimate means insufficient evidence, not permission
    # to suppress every nuisance component in the protected arm.
    if protected.shape[1] == 0:
        return nuisance[:, :0]
    return nuisance - protected @ (protected.T @ nuisance)


def transform_features(features, factor, strength, *, normalize=True):
    if factor.shape[1] == 0 or strength == 0:
        return features
    transformed = features - strength * ((features @ factor) @ factor.T)
    return F.normalize(transformed, dim=-1) if normalize else transformed


def trace_matched_factor(nuisance, protected):
    """Keep nuisance directions but match the protected contraction trace."""
    target = contraction_factor(nuisance, protected)
    if nuisance.shape[1] == 0:
        return nuisance
    if target.shape[1] == 0:
        return target
    ratio = (target.square().sum() / nuisance.square().sum()).clamp(0., 1.)
    return nuisance * ratio.sqrt()


@METHOD_REGISTRY.register("protected_geometry")
class ProtectedGeometry(QueryReference):
    def __init__(self, backbone, config):
        super().__init__(backbone, config)
        self.geometry_enabled = config.get("geometry_enabled", False)
        self.protect_contrasts = config.get("protect_contrasts", False)
        self.match_protected_trace = config.get("match_protected_trace", False)
        if type(self.match_protected_trace) is not bool:
            raise ValueError("match_protected_trace must be boolean")
        if self.match_protected_trace and (not self.geometry_enabled or self.protect_contrasts):
            raise ValueError("trace matching requires unprotected enabled geometry")
        if any(type(v) is not bool for v in (self.geometry_enabled, self.protect_contrasts)):
            raise ValueError("geometry switches must be boolean")
        if self.protect_contrasts and not self.geometry_enabled:
            raise ValueError("protection requires geometry")
        if self.geometry_enabled and self.query_adaptation:
            raise ValueError("geometry candidate must isolate itself from QR adaptation")
        if (self.bank_bags != 1 or self.bank_selection != "random" or self.bank_outlier_trim != 0
                or self.score_normalization != "mad" or self.layer_fusion != "mean"
                or self.scale_fusion != "mean" or self.local_reliability != "off"
                or self.prediction_postprocess != "none"):
            raise ValueError("protected_geometry requires original R1 random-bank/MAD/mean conditions")
        self.geometry_strength = self._number(config, "geometry_strength", .5, 0., 1., allow_zero=True)
        if self.geometry_strength >= 1:
            raise ValueError("geometry_strength must be <1")
        self.nuisance_rank = self._integer(config, "nuisance_rank", 16, 1)
        self.protected_rank = self._integer(config, "protected_rank", 32, 1)
        self.evidence_cap = self._integer(config, "geometry_evidence_tokens", 256, 1)
        self.stability_cosine = self._number(config, "stability_cosine", .9, 0., 1.)
        self.geometry_factors = {}
        self.geometry_evidence = {}

    def _extract(self, image, resolution):
        layers, grid = super()._extract(image, resolution)
        if resolution in self.geometry_factors:
            layers = [transform_features(q, a, self.geometry_strength)
                      for q, a in zip(layers, self.geometry_factors[resolution])]
        return layers, grid

    @torch.no_grad()
    def fit(self, support_paths, context):
        # Fail before extracting anything, as in the common parent contract.
        if context.masking or context.rotation:
            raise ValueError("protected_geometry requires no_mask_no_rotation")
        if not support_paths or len(set(support_paths)) != len(support_paths):
            raise ValueError("need unique normal support paths")
        self.geometry_factors, self.geometry_evidence = {}, {}
        if self.geometry_enabled:
            self._fit_geometry(support_paths)
        # Parent resets bank sampling RNG. Only features and their fitted MAD
        # change; original sampled source/token identities remain unchanged.
        super().fit(support_paths, context)
        self.log_training(event="protected_geometry_ready", trainable_parameters=0,
                          geometry_enabled=self.geometry_enabled,
                          protect_contrasts=self.protect_contrasts,
                          geometry_evidence=self.geometry_evidence)

    @torch.no_grad()
    def _fit_geometry(self, support_paths):
        for resolution in self.resolutions:
            moments, contrasts, counts, samples = None, None, None, None
            for path in support_paths:
                image = _read_rgb(path)
                clean, grid = ReferenceMatching._extract(self, image, resolution)
                views = []
                for view in photometric_views(image):
                    layers, other_grid = ReferenceMatching._extract(self, view, resolution)
                    if other_grid != grid:
                        raise ValueError("photometric evidence changed spatial alignment")
                    views.append(layers)
                if moments is None:
                    moments = [q.new_zeros((q.shape[1], q.shape[1])) for q in clean]
                    contrasts = [q.new_zeros((q.shape[1], q.shape[1])) for q in clean]
                    counts = [dict(stable_edges=0, contributing_images=0, nuisance_tokens=0) for _ in clean]
                    samples = [[] for _ in clean]
                for index, q in enumerate(clean):
                    ids = even_indices(len(q), self.evidence_cap, q.device)
                    for layers in views:
                        delta = layers[index][ids] - q[ids]
                        moments[index].add_(delta.T @ delta / (len(ids) * len(views) * len(support_paths)))
                        counts[index]["nuisance_tokens"] += len(ids)
                    left, right = stable_edges(q, [v[index] for v in views], grid,
                                               self.evidence_cap, self.stability_cosine)
                    if len(left):
                        diff = F.normalize(q[right] - q[left], dim=-1)
                        contrasts[index].add_(diff.T @ diff / len(left))
                        counts[index]["stable_edges"] += len(left)
                        counts[index]["contributing_images"] += 1
                        sample = even_indices(len(left), 32, q.device)
                        samples[index].append((q[left[sample]].cpu(), q[right[sample]].cpu()))
            factors, evidence = [], []
            for index, moment in enumerate(moments):
                nuisance = moment_basis(moment, self.nuisance_rank)
                local = contrasts[index] / max(1, counts[index]["contributing_images"])
                protected = moment_basis(local, self.protected_rank)
                factor = contraction_factor(nuisance, protected if self.protect_contrasts else None)
                if self.match_protected_trace:
                    factor = trace_matched_factor(nuisance, protected)
                factors.append(factor)
                operator = torch.eye(moment.shape[0], device=moment.device) - self.geometry_strength * (factor @ factor.T)
                before = float(torch.trace(moment))
                after = float(torch.trace(operator @ moment @ operator.T))
                original_energy = changed_energy = 0.
                for left, right in samples[index]:
                    left, right = left.to(self.device), right.to(self.device)
                    # Squared Euclidean distance on unit vectors is twice
                    # cosine distance, and avoids catastrophic 1-dot cancellation.
                    original_energy += float((right - left).square().sum())
                    lt = transform_features(left, factor, self.geometry_strength)
                    rt = transform_features(right, factor, self.geometry_strength)
                    changed_energy += float((rt - lt).square().sum())
                ratio = changed_energy / original_energy if original_energy > 1e-12 else None
                entry = dict(layer=self.layers[index], **counts[index],
                             nuisance_rank=nuisance.shape[1], protected_rank=protected.shape[1],
                             fallback_identity=factor.shape[1] == 0,
                             nuisance_protected_overlap=float((protected.T @ nuisance).square().sum()),
                             effective_contraction_trace=float(self.geometry_strength * factor.square().sum()),
                             protected_linear_error=float((operator @ protected - protected).norm()),
                             nuisance_energy_before=before, nuisance_energy_after=after,
                             local_cosine_distance_ratio_after_normalization=ratio,
                             local_contrast_distortion_after_normalization=abs(ratio - 1) if ratio is not None else None)
                evidence.append(entry)
                self.log_training(event="geometry_layer_fitted", resolution=resolution, **entry)
            self.geometry_factors[resolution] = factors
            self.geometry_evidence[str(resolution)] = evidence

    def object_metadata(self):
        metadata = super().object_metadata()
        metadata.update(geometry_enabled=self.geometry_enabled, protect_contrasts=self.protect_contrasts,
                        match_protected_trace=self.match_protected_trace,
                        geometry_strength=self.geometry_strength, geometry_evidence=self.geometry_evidence,
                        geometry_operator="normalize((I-strength*A*A^T)z)",
                        protected_invariance="linear_pre_normalization_only_not_defect_preservation",
                        geometry_fit_scope="selected_source_normal_support_only",
                        geometry_perturbations=[[.7, .7, .7], [.8, 1., 1.2]],
                        geometry_evidence_tokens=self.evidence_cap, nuisance_rank=self.nuisance_rank,
                        protected_rank=self.protected_rank, stability_cosine=self.stability_cosine,
                        geometry_trainable_parameters=0,
                        geometry_normalization="refit_support_MAD_in_transformed_space",
                        geometry_query_dependent=False)
        return metadata
