"""Normal-mode-conditioned, ephemeral reference deformation on fixed QR1.

MC0 is exactly QR1. MC1 adds shrunk mode-specific innovations to its global
translation; MC2 projects only those innovations into normal-support variation
subspaces; MC3 gates their fitting by source-excluded normal orthogonal residual
scales. Modes are feature clusters, not semantic or normality labels. Neither
normal tangent directions nor orthogonal gates guarantee anomaly preservation.
"""
from __future__ import annotations

import hashlib

import torch
from torch.nn import functional as F

from fmad.methods.query_reference import QueryReference, fit_translation
from fmad.methods.reference_matching import _distance_chunks
from fmad.registry import METHOD_REGISTRY


def tensor_fingerprint(value):
    tensor = value.detach().cpu().contiguous()
    header = f"{tuple(tensor.shape)}:{tensor.dtype}:".encode()
    return hashlib.sha256(header + tensor.numpy().tobytes()).hexdigest()


def normal_modes(reference, count, iterations):
    """Deterministic farthest-first spherical k-means on the unchanged bank."""
    count = min(count, len(reference))
    first = int((reference @ F.normalize(reference.mean(0), dim=0)).argmax())
    chosen = [first]
    distance = 1.0 - reference @ reference[first]
    for _ in range(1, count):
        candidate = int(distance.argmax())
        if float(distance[candidate]) <= 1e-6:
            break  # Duplicate bank vectors must not fabricate extra modes.
        chosen.append(candidate)
        distance = torch.minimum(distance, 1.0 - reference @ reference[candidate])
    centers = reference[chosen].clone()
    for _ in range(iterations):
        labels = (reference @ centers.T).argmax(1)
        # Avoid CUDA index_add atomic accumulation: tiny order-dependent center
        # changes could otherwise alter assignments between ablation stages.
        updated = []
        for index, center in enumerate(centers):
            values = reference[labels == index]
            updated.append(F.normalize(values.mean(0), dim=0) if len(values) else center)
        centers = torch.stack(updated)
    labels = (reference @ centers.T).argmax(1)
    return labels, centers


def normal_basis(values, rank, seed):
    """Bounded deterministic randomized PCA; no global RNG state is changed.

    Rank+4 sketch, two covariance power iterations, and a thin SVD avoid a
    channel-by-channel covariance decomposition. Degenerate directions are
    discarded. Geometry uses only the original normal support bank.
    """
    if len(values) < 2:
        return values.new_empty((values.shape[1], 0))
    centered = values - values.mean(0)
    width = min(rank + 4, len(values) - 1, values.shape[1])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sketch = torch.randn(values.shape[1], width, generator=generator).to(values)
    for _ in range(2):
        sketch = torch.linalg.qr(centered.T @ (centered @ sketch), mode="reduced").Q
    _, singular, vh = torch.linalg.svd(centered @ sketch, full_matrices=False)
    usable = min(rank, int((singular > singular[0].clamp_min(1e-12) * 1e-5).sum()))
    return (sketch @ vh[:usable].T).contiguous()


def orthogonal_norm(residuals, basis):
    return torch.linalg.vector_norm(residuals - (residuals @ basis) @ basis.T, dim=-1)


def normal_gate_scales(reference, owners, labels, bases, chunk, quantile, minimum):
    """Normal-only leave-source-out NN calibration, with fixed all-support bases.

    Nearest-reference selection excludes the originating support image, but
    clustering/PCA are NOT refitted leave-source-out. No holdout claim is made.
    Calibration uses the matched reference's mode, exactly as query fitting.
    """
    owner_ids = torch.as_tensor(owners, device=reference.device)
    if owner_ids.unique().numel() < 2:
        raise ValueError("normal gate calibration needs two independent supports")
    distances = reference.new_empty(len(reference))
    matched_modes = torch.empty(len(reference), device=reference.device, dtype=torch.long)
    for start in range(0, len(reference), chunk):
        values = reference[start:start + chunk]
        similarity = values @ reference.T
        excluded = owner_ids[start:start + chunk, None] == owner_ids[None]
        selected = similarity.masked_fill(excluded, -torch.inf).argmax(1)
        modes = labels[selected]
        residuals = values - reference[selected]
        local_distances = values.new_empty(len(values))
        for index, basis in enumerate(bases):
            mask = modes == index
            local_distances[mask] = orthogonal_norm(residuals[mask], basis)
        distances[start:start + len(values)] = local_distances
        matched_modes[start:start + len(values)] = modes
    global_scale = torch.quantile(distances, quantile).clamp_min(1e-6)
    scales, counts = [], []
    for index in range(len(bases)):
        values = distances[matched_modes == index]
        counts.append(len(values))
        scales.append(torch.quantile(values, quantile).clamp_min(1e-6)
                      if len(values) >= minimum else global_scale)
    return torch.stack(scales), counts


def mode_translations(residuals, modes, geometry, *, ridge, max_norm, min_tokens,
                      min_mode_tokens, prior_tokens, constrain, gate):
    """Hierarchical innovations; sparse/untrusted modes retain the QR1 component.

    mu = mean(r); delta0 = capped(mu/(1+ridge));
    e_k = sum(w*(r-mu))/((sum(w)+prior_tokens)*(1+ridge));
    delta_k = cap(delta0 + [B_k B_k^T] e_k).
    Gates and projection affect only e_k, never change the QR1 comparator.
    """
    baseline, stats = fit_translation(residuals, keep_fraction=1.0, ridge=ridge,
                                      max_norm=max_norm, min_tokens=min_tokens)
    count = len(geometry["centers"])
    deltas = baseline.repeat(count, 1)
    extra = dict(mode_slots=count, active_modes=0, fallback_modes=0,
                 evidence_mass=0.0, query_tokens=len(residuals),
                 innovation_before_norm_sum=0.0, innovation_after_norm_sum=0.0,
                 mode_global_difference_norm_sum=0.0, mode_translation_norm_sum=0.0,
                 clipped_modes=0, low_gate_tokens=0)
    if stats["fallback"]:
        extra["fallback_modes"] = count
        return deltas, stats, extra
    mean = residuals.mean(0)
    for index in range(count):
        values = residuals[modes == index]
        weights = values.new_ones(len(values))
        if gate:
            orthogonal = orthogonal_norm(values, geometry["bases"][index])
            weights = 1.0 / (1.0 + (orthogonal / geometry["scales"][index]).square())
        mass = weights.sum()
        extra["evidence_mass"] += float(mass)
        extra["low_gate_tokens"] += int((weights < .25).sum())
        if len(values) < min_mode_tokens or float(mass) < min_mode_tokens:
            extra["fallback_modes"] += 1
            continue
        innovation = ((values - mean) * weights[:, None]).sum(0) / (mass + prior_tokens) / (1.0 + ridge)
        extra["innovation_before_norm_sum"] += float(innovation.norm())
        if constrain:
            basis = geometry["bases"][index]
            innovation = (innovation @ basis) @ basis.T
        extra["innovation_after_norm_sum"] += float(innovation.norm())
        delta = baseline + innovation
        norm = delta.norm()
        extra["clipped_modes"] += int(norm > max_norm)
        deltas[index] = delta * (max_norm / norm.clamp_min(1e-12)).clamp_max(1.0)
        extra["active_modes"] += 1
    if not bool(torch.isfinite(deltas).all()):
        raise RuntimeError("nonfinite normal-mode translation")
    extra["mode_global_difference_norm_sum"] = float((deltas - baseline).norm(dim=1).sum())
    extra["mode_translation_norm_sum"] = float(deltas.norm(dim=1).sum())
    return deltas, stats, extra


@METHOD_REGISTRY.register("mode_reference")
class ModeReference(QueryReference):
    def __init__(self, backbone, config):
        super().__init__(backbone, {"query_adaptation": True, **config})
        if not self.query_adaptation or self.robust_trim or self.spatial_crossfit:
            raise ValueError("mode_reference requires QR1 without trim/crossfit")
        for name in ("mode_conditioning", "normal_subspace", "orthogonal_gate"):
            value = config.get(name, False)
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")
            setattr(self, name, value)
        if self.normal_subspace and not self.mode_conditioning:
            raise ValueError("normal_subspace requires mode_conditioning")
        if self.orthogonal_gate and not self.normal_subspace:
            raise ValueError("orthogonal_gate requires normal_subspace")
        self.mode_count = self._integer(config, "mode_count", 8, 2)
        self.mode_iterations = self._integer(config, "mode_kmeans_iters", 5, 1)
        self.subspace_rank = self._integer(config, "mode_subspace_rank", 4, 1)
        self.min_mode_tokens = self._integer(config, "min_mode_tokens", 16, 1)
        self.prior_tokens = self._number(config, "mode_prior_tokens", 64., 0., None)
        self.gate_quantile = self._number(config, "normal_gate_quantile", .9, 0., 1.)

    def _reset_diagnostics(self):
        super()._reset_diagnostics()
        self.mode_stats = {key: 0 for key in (
            "fits", "mode_slots", "active_modes", "fallback_modes", "evidence_mass",
            "query_tokens", "innovation_before_norm_sum", "innovation_after_norm_sum",
            "mode_global_difference_norm_sum", "mode_translation_norm_sum",
            "clipped_modes", "low_gate_tokens")}

    @torch.no_grad()
    def fit(self, support_paths, context):
        super().fit(support_paths, context)
        if self.mode_conditioning:
            for resolution, bank in self.banks.items():
                geometries = []
                for layer_index, reference in enumerate(bank["layers"]):
                    labels, centers = normal_modes(reference, self.mode_count, self.mode_iterations)
                    geometry = dict(labels=labels, centers=centers)
                    if self.normal_subspace:
                        geometry["bases"] = [normal_basis(reference[labels == k], self.subspace_rank,
                                                         context.seed + resolution * 101 + layer_index * 17 + k)
                                             for k in range(len(centers))]
                    if self.orthogonal_gate:
                        geometry["scales"], geometry["calibration_counts"] = normal_gate_scales(
                            reference, bank["owners"], labels, geometry["bases"], self.chunk,
                            self.gate_quantile, self.min_mode_tokens)
                    geometries.append(geometry)
                bank["mode_geometry"] = geometries
        self.log_training(step=None, event="mode_reference_ready",
                          mode_conditioning=self.mode_conditioning, normal_subspace=self.normal_subspace,
                          orthogonal_gate=self.orthogonal_gate, trainable_parameters=0)

    @torch.no_grad()
    def _score_tokens(self, layers, grid, bank, *, normalize=True):
        if not self.mode_conditioning:
            return super()._score_tokens(layers, grid, bank, normalize=normalize)
        layer_scores = []
        for index, (query, reference, geometry) in enumerate(zip(layers, bank["layers"], bank["mode_geometry"])):
            residuals, modes = [], []
            for start in range(0, len(query), self.chunk):
                values = query[start:start + self.chunk]
                selected = (values @ reference.T).argmax(1)
                residuals.append(values - reference[selected])
                modes.append(geometry["labels"][selected])
            deltas, stats, extra = mode_translations(
                torch.cat(residuals), torch.cat(modes), geometry, ridge=self.ridge,
                max_norm=self.max_norm, min_tokens=self.min_tokens,
                min_mode_tokens=self.min_mode_tokens, prior_tokens=self.prior_tokens,
                constrain=self.normal_subspace, gate=self.orthogonal_gate)
            shifted = reference if stats["fallback"] else F.normalize(reference + deltas[geometry["labels"]], dim=-1)
            scores = _distance_chunks(query, shifted, self.chunk)
            layer_scores.append(scores / (self._active_scale(index, query.device) if normalize else 1.0))
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
        geometry_summary = {}
        for resolution, bank in self.banks.items():
            geometry_summary[str(resolution)] = [dict(
                mode_bank_counts=torch.bincount(g["labels"], minlength=len(g["centers"])).tolist(),
                labels_sha256=tensor_fingerprint(g["labels"]),
                centers_sha256=tensor_fingerprint(g["centers"]),
                normal_basis_sha256=[tensor_fingerprint(b) for b in g.get("bases", [])],
                normal_basis_ranks=[b.shape[1] for b in g.get("bases", [])],
                orthogonal_scales=g["scales"].tolist() if "scales" in g else [],
                gate_calibration_counts=g.get("calibration_counts", []),
            ) for g in bank.get("mode_geometry", [])]
        metadata.update(
            mode_conditioning=self.mode_conditioning, normal_subspace=self.normal_subspace,
            orthogonal_gate=self.orthogonal_gate, mode_count=self.mode_count,
            mode_kmeans_iters=self.mode_iterations, mode_subspace_rank=self.subspace_rank,
            mode_prior_tokens=self.prior_tokens, min_mode_tokens=self.min_mode_tokens,
            normal_gate_quantile=self.gate_quantile,
            normal_geometry=geometry_summary,
            basis_estimator="normal_only_seeded_rank_plus4_sketch_two_covariance_iterations_thin_svd",
            mode_assignments="initial_R1_reference_mode_independent_per_layer",
            gate_calibration="source_excluded_NN_residual_with_fixed_all_support_geometry",
            gate_is_normality_label=False,
            translation_diagnostics_scope="QR1_global_component_only",
            mode_diagnostics_including_warmup=dict(self.mode_stats),
            transform=("bounded_global_plus_mode_innovation_then_unit_normalization"
                       if self.mode_conditioning else metadata["transform"]),
            fit_objective=("ridge_shrunk_mode_residual_innovation_with_optional_normal_projection_and_gate"
                           if self.mode_conditioning else metadata["fit_objective"]),
            adaptation_scope="single_query_layer_resolution_no_persistent_query_state",
            score_definition=("mean_layer_cosine_nn_to_mode_deformed_bank_fixed_support_MAD"
                              if self.mode_conditioning else metadata["score_definition"]),
        )
        return metadata
