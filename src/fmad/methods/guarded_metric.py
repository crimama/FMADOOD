"""Original-space bounded low-rank metric with optional nuisance protection.

E1 replaces D3's normalized feature adapter with a bounded quadratic metric.
E2 restricts extra distance contraction to support-derived shift directions,
attenuating directions also excited by synthetic defects. E3 adds weak image
anomalies and hard-positive sampling. Shared TailContrast fitting/replay and
the public CLI/pipeline remain unchanged. No dense maps or weights are saved.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fmad.methods.PSMA_MR import (
    _apply_photometric, _sample_photometric_transform, _synthetic_anomalies,
)
from fmad.methods.tail_contrast import TailContrast, tail_samples
from fmad.registry import METHOD_REGISTRY


class GuardedLowRank(nn.Module):
    """M_extra = P diag(a+) P^T - N diag(a-) N^T; ||P||, ||N|| <= 1.

Without protection N=P, giving a signed low-rank metric. With protection,
N=Q diag(sqrt(g)) Q^T P. Q and g are frozen support-only evidence, not labels
or learned confidence. Each PSD term has norm <= radius, so the combined
metric with BCMA's diagonal has eigenvalues in [1-bound-radius,1+bound+radius].
"""
    def __init__(self, dimension, rank, radius):
        super().__init__()
        if not 1 <= rank <= dimension or not 0 < radius < 1:
            raise ValueError("invalid guarded metric rank/radius")
        self.radius = radius
        self.basis = nn.Parameter(torch.randn(dimension, rank) / math.sqrt(dimension))
        self.logits = nn.Parameter(torch.zeros(rank))
        self.register_buffer("nuisance_basis", torch.empty(dimension, 0))
        self.register_buffer("nuisance_gate", torch.empty(0))

    def factors(self):
        p = torch.linalg.qr(self.basis, mode="reduced").Q
        n = p
        if self.nuisance_gate.numel():
            q = self.nuisance_basis
            n = q @ (self.nuisance_gate.sqrt()[:, None] * (q.T @ p))
        signed = self.radius * self.logits.tanh()
        # Unlike ReLU at zero, these subgradients allow identity initialization
        # to learn. For N=P this is exactly P diag(signed) P^T, also in gradient.
        positive = (signed + signed.abs()) * .5
        negative = (signed.abs() - signed) * .5
        return p, n, positive, negative

    @torch.no_grad()
    def set_evidence(self, shift_covariance, defect_covariance):
        covariance = (shift_covariance + shift_covariance.T) * .5
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        rank = self.basis.shape[1]
        q = eigenvectors[:, -rank:]
        shift_energy = eigenvalues[-rank:].clamp_min(0)
        defect_energy = ((q.T @ defect_covariance) * q.T).sum(1).clamp_min(0)
        # Zero/unobserved shift gives zero permission to contract, not confidence.
        gate = shift_energy / (shift_energy + defect_energy + 1e-8)
        self.nuisance_basis = q.detach().clone()
        self.nuisance_gate = gate.clamp(0, 1).detach().clone()
        return dict(shift_energy=float(shift_energy.sum()),
                    defect_energy=float(defect_energy.sum()),
                    gate_min=float(gate.min()), gate_mean=float(gate.mean()),
                    gate_max=float(gate.max()),
                    explained_shift_fraction=float(shift_energy.sum() / eigenvalues.clamp_min(0).sum().clamp_min(1e-8)))


def hard_positive_samples(values, count, fraction, generator):
    """Fixed positive budget: half low-score positives, half uniform positives."""
    count = min(count, values.numel())
    hard_count = count // 2
    k = max(1, math.ceil(values.numel() * fraction), hard_count)
    hard = values.detach().topk(k, largest=False).indices
    ids = torch.randperm(k, device=values.device, generator=generator)[:hard_count]
    random = torch.randperm(values.numel(), device=values.device, generator=generator)[:count-hard_count]
    return torch.cat((values[hard[ids]], values[random]))


@METHOD_REGISTRY.register("guarded_metric")
class GuardedMetric(TailContrast):
    def __init__(self, backbone, config):
        required = dict(hard_normal_ranking=True, paired_contrast=True, low_rank_adapter=False)
        for name, value in required.items():
            if name in config and config[name] is not value:
                raise ValueError(f"guarded_metric requires {name}={value}")
        super().__init__(backbone, {**config, **required})
        self.nuisance_protection = config.get("nuisance_protection", False)
        self.weak_anomaly_training = config.get("weak_anomaly_training", False)
        if any(type(flag) is not bool for flag in (self.nuisance_protection, self.weak_anomaly_training)):
            raise ValueError("guarded metric switches must be booleans")
        if self.weak_anomaly_training and not self.nuisance_protection:
            raise ValueError("weak_anomaly_training requires nuisance_protection")
        self.radius = float(config.get("low_rank_bound", .3))
        self.rank = int(config.get("metric_rank", 16))
        self.evidence_tokens = int(config.get("evidence_tokens", 128))
        self.positive_fraction = float(config.get("hard_positive_fraction", .25))
        self.strengths = tuple(float(x) for x in config.get("anomaly_strengths", [.35, .65, 1.]))
        if not 0 < self.radius < 1-self.bound or self.rank < 1 or self.evidence_tokens < 1:
            raise ValueError("metric must be positive definite with positive token/rank budgets")
        if not 0 < self.positive_fraction <= 1 or not self.strengths or any(not 0 < s <= 1 for s in self.strengths):
            raise ValueError("invalid positive fraction or anomaly strengths")
        self.low_rank = None
        self.evidence = {}
        self.anomaly_strength = 1.

    def _prepare_fit(self, support_paths, context):
        self.low_rank, self.evidence = None, {}
        self.anomaly_strength = 1.
        prepared = super()._prepare_fit(support_paths, context)
        images, _, evidence_rng, _ = prepared
        dimension = self.banks[self.resolutions[0]][0].shape[1]
        self.low_rank = GuardedLowRank(dimension, self.rank, self.radius).to(self.device)
        if self.nuisance_protection:
            self._fit_evidence(images, evidence_rng)
        return prepared

    @torch.no_grad()
    def _fit_evidence(self, images, rng):
        """Stream two small D-by-D moments; only rank-r Q/g survive fitting.

Both anomaly families contribute for every source and every resolution.
Use the separate augmentation RNG: main training views/bank sampling stay
identical to E1 and D3. Weak training does not change E2/E3 evidence.
"""
        dimension = self.low_rank.basis.shape[0]
        shift_cov = torch.zeros(dimension, dimension, device=self.device)
        defect_cov = torch.zeros_like(shift_cov)
        shift_count = defect_count = 0
        for source, image in enumerate(images):
            shifted = _apply_photometric(image, _sample_photometric_transform(rng, 1.))
            anomalies = _synthetic_anomalies(image, rng, [])
            for resolution in self.resolutions:
                _, original, grid, _ = self.records[resolution][source]
                original = original.to(self.device)
                shifted_features, shifted_grid = self.extract(shifted, resolution)
                if grid != shifted_grid:
                    raise ValueError("photometric evidence changed spatial alignment")
                ids = torch.as_tensor(rng.choice(len(original), min(self.evidence_tokens, len(original)), replace=False), device=self.device)
                delta = shifted_features[ids] - original[ids]
                shift_cov.add_(delta.T @ delta)
                shift_count += len(ids)
                for anomaly, mask, _ in anomalies:
                    features, anomaly_grid = self.extract(anomaly, resolution)
                    if grid != anomaly_grid:
                        raise ValueError("synthetic evidence changed spatial alignment")
                    coverage = F.adaptive_avg_pool2d(torch.as_tensor(mask, device=self.device, dtype=torch.float32)[None, None], grid).flatten()
                    eligible = (coverage >= .25).nonzero().flatten()
                    if not len(eligible):
                        eligible = (coverage > 0).nonzero().flatten()
                    if not len(eligible):
                        raise ValueError("synthetic evidence has no visible defect tokens")
                    chosen = rng.choice(len(eligible), min(self.evidence_tokens, len(eligible)), replace=False)
                    ids = eligible[torch.as_tensor(chosen, device=self.device)]
                    delta = features[ids] - original[ids]
                    # Equal source/family/resolution weights, independent of mask area.
                    defect_cov.add_(delta.T @ delta / len(ids))
                    defect_count += 1
        self.evidence = self.low_rank.set_evidence(shift_cov / shift_count, defect_cov / defect_count)
        self.evidence.update(support_count=len(images), shift_tokens=shift_count,
                             defect_groups=defect_count, token_budget=self.evidence_tokens,
                             families=["rectangle_cutpaste", "scratch_noise"],
                             resolutions=list(self.resolutions), selection="support_only_seed_plus_100003")

    def extra_parameters(self):
        return self.low_rank.parameters()

    def score(self, features, grid, bank, indices=None, baseline=False):
        if baseline or self.low_rank is None:
            return super().score(features, grid, bank, indices, baseline)
        correction = self.metric(features, grid)
        if indices is not None:
            features, correction = features[indices], correction[indices]
        values = bank[0]
        p, n, positive, negative = self.low_rank.factors()
        # Search in chunks with Q-by-bank matrices, never Q-by-bank-by-D maps.
        with torch.no_grad():
            projected_bank = (values @ p, values @ n)
        output = []
        for start in range(0, len(features), self.chunk):
            query = features[start:start+self.chunk]
            change = correction[start:start+self.chunk]
            with torch.no_grad():
                distance = 1 - query @ values.T + .5 * (
                    (change * query.square()).sum(1, keepdim=True)
                    + change @ values.square().T - 2 * (change * query) @ values.T)
                for basis, reference, weight, sign in ((p, projected_bank[0], positive, 1), (n, projected_bank[1], negative, -1)):
                    projected = query @ basis
                    distance.add_(sign * .5 * ((projected.square() * weight).sum(1, keepdim=True)
                                  + (reference.square() * weight).sum(1)[None]
                                  - 2 * (projected * weight) @ reference.T))
                selected = distance.argmin(1)
            reference = values[selected]
            residual = query - reference
            distance = 1 - (query * reference).sum(1) + .5 * (change * residual.square()).sum(1)
            distance = distance + .5 * ((residual @ p).square() * positive).sum(1)
            distance = distance - .5 * ((residual @ n).square() * negative).sum(1)
            output.append(distance)
        return torch.cat(output)

    def training_anomaly(self, image, rng, family_index, step):
        anomaly, mask, family = super().training_anomaly(image, rng, family_index, step)
        self.anomaly_strength = 1.
        if self.weak_anomaly_training:
            # Rotate strengths over source AND cycle; no permanently weak source.
            index = (step % self.support_count + step // self.support_count) % len(self.strengths)
            self.anomaly_strength = self.strengths[index]
            mixed = image.astype(np.float32) + self.anomaly_strength * (anomaly.astype(np.float32) - image)
            anomaly = np.ascontiguousarray(np.rint(mixed).clip(0, 255).astype(np.uint8))
        return anomaly, mask, family

    def ranking_loss(self, scores, mask, sampler):
        if not self.weak_anomaly_training:
            return super().ranking_loss(scores, mask, sampler)
        normal = torch.cat([tail_samples(s, self.tail_fraction, self.train_tokens, sampler) for s in scores[:2]])
        terms = []
        for view in scores[2:]:
            positive = hard_positive_samples(view[mask], self.train_tokens, self.positive_fraction, sampler)
            negative = torch.cat((normal, tail_samples(view[~mask], self.tail_fraction, self.train_tokens, sampler)))
            terms.append(F.relu(float(self.config.get("margin", .05)) - positive[:, None] + negative[None]).mean())
        return torch.stack(terms).mean()

    def training_diagnostics(self):
        with torch.no_grad():
            coefficients = self.radius * self.low_rank.logits.tanh()
            return dict(anomaly_strength=self.anomaly_strength,
                        low_rank_coefficient_min=float(coefficients.min()),
                        low_rank_coefficient_max=float(coefficients.max()),
                        nuisance_gate_mean=self.evidence.get("gate_mean"),
                        hard_positive_fraction=self.positive_fraction if self.weak_anomaly_training else 0.)

    def object_metadata(self):
        metadata = super().object_metadata()
        metadata.pop("baseline_equivalence", None)
        metadata.update(components=dict(bounded_low_rank_metric=True,
                                        nuisance_protection=self.nuisance_protection,
                                        weak_anomaly_training=self.weak_anomaly_training),
                        score_definition="original_space_bounded_diagonal_plus_signed_low_rank_full_nn",
                        feature_adapter=False, metric_rank=self.rank, low_rank_bound=self.radius,
                        original_distance_ratio_bounds=[1-self.bound-self.radius, 1+self.bound+self.radius],
                        low_rank_parameters=sum(p.numel() for p in self.low_rank.parameters()),
                        evidence=self.evidence, anomaly_strengths=list(self.strengths) if self.weak_anomaly_training else [1.],
                        hard_positive_fraction=self.positive_fraction if self.weak_anomaly_training else 0.)
        return metadata
