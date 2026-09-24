"""R1-preserving residual ranking with source-separated reference evidence.

Only sampled scalar descriptors are cached on CPU during fitting. Neither
backbone graphs nor query maps are accumulated, and no query enters fitting.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fmad.methods.PSMA_MR import _read_rgb
from fmad.methods.reference_matching import ReferenceMatching
from fmad.methods.tail_aware_fusion import TailAwareFusion, _digest, sample_native
from fmad.registry import METHOD_REGISTRY


class BoundedResidual(nn.Module):
    def __init__(self, geometry: bool, bound: float):
        super().__init__()
        self.geometry = geometry
        self.bound = bound
        self.network = nn.Sequential(nn.Linear(6, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, values):
        inputs = torch.log1p(values.clamp_min(0))
        if not self.geometry:
            inputs = torch.cat((inputs[..., :2], torch.zeros_like(inputs[..., 2:])), -1)
        residual = self.network(inputs).squeeze(-1).tanh() * self.bound
        return values[..., :2].mean(-1) + residual


@torch.no_grad()
def source_distances(query, bank, owners, sources: int, chunk: int):
    """Exact independent NN and per-source minima; bound the pairwise temporary."""
    nearest, identities, distances = [], [], []
    groups = [torch.where(owners == source)[0] for source in range(sources)]
    for start in range(0, len(query), chunk):
        matrix = 1.0 - query[start:start + chunk] @ bank.T
        value, identity = matrix.min(1)
        per_source = matrix.new_full((len(matrix), sources), float("inf"))
        for source, indices in enumerate(groups):
            if len(indices):
                per_source[:, source] = matrix.index_select(1, indices).amin(1)
        nearest.append(value)
        identities.append(identity)
        distances.append(per_source)
    return torch.cat(nearest), torch.cat(identities), torch.cat(distances)


def reference_radii(table, owners, neighbors: int, excluded: int | None = None):
    """Exclude both the reference owner and the held-out episode source."""
    values = table.clone()
    values[torch.arange(len(values), device=values.device), owners] = float("inf")
    if excluded is not None:
        values[:, excluded] = float("inf")
    selected = values.topk(neighbors, largest=False, dim=1).values
    if not torch.isfinite(selected).all():
        raise ValueError("not enough distinct reference sources after exclusions")
    return selected.clamp_min(0).mean(1)


def weighted_tail(scores, weights, fraction: float):
    """Indices and probability mass of a stable, fractional upper-mass tail."""
    scores, weights = np.asarray(scores), np.asarray(weights, dtype=np.float64)
    if (scores.ndim != 1 or weights.shape != scores.shape or not len(scores)
            or not np.isfinite(scores).all() or not np.isfinite(weights).all()
            or np.any(weights <= 0) or not 0 < fraction <= 1):
        raise ValueError("invalid weighted-tail population")
    order = np.argsort(-scores, kind="stable")
    ordered = weights[order]
    target = weights.sum() * fraction
    cumulative = np.cumsum(ordered)
    count = min(len(order), int(np.searchsorted(cumulative, target, side="left")) + 1)
    mass = ordered[:count].copy()
    mass[-1] = min(mass[-1], target - (cumulative[count - 2] if count > 1 else 0.))
    keep = mass > 0
    return order[:count][keep], mass[keep] / mass[keep].sum()


class CompactPool:
    """Preallocated CPU arrays; active rows only are sampled and hashed."""
    def __init__(self, capacity: int, resolutions: int):
        self.values = np.empty((capacity, resolutions, 6), dtype=np.float32)
        self.weights = np.empty(capacity, dtype=np.float64)
        self.sources = np.empty(capacity, dtype=np.int32)
        self.episodes = np.empty(capacity, dtype=np.int32)
        self.size = 0
        self.ranges = {}

    @property
    def nbytes(self):
        return sum(array.nbytes for array in
                   (self.values, self.weights, self.sources, self.episodes))

    def append(self, values, weight: float, source: int, episode: int):
        stop = self.size + len(values)
        if stop > len(self.values) or not np.isfinite(values).all() or not weight > 0:
            raise RuntimeError("invalid descriptor pool append or capacity exceeded")
        section = slice(self.size, stop)
        self.values[section] = values
        self.weights[section] = weight
        self.sources[section] = source
        self.episodes[section] = episode
        first = self.ranges.get(episode, (self.size, stop))[0]
        self.ranges[episode] = (first, stop)
        self.size = stop

    def digest(self):
        digest = hashlib.sha256()
        for values in (self.values, self.weights, self.sources, self.episodes):
            digest.update(memoryview(values[:self.size]).cast("B"))
        return digest.hexdigest()


@METHOD_REGISTRY.register("cross_image_reference")
class CrossImageReference(ReferenceMatching):
    def __init__(self, backbone, config):
        if config.get("variant", "R1") != "R1":
            raise ValueError("cross_image_reference requires R1")
        super().__init__(backbone, {**config, "variant": "R1"})
        self.components = {key: config.get(key, False) for key in
                           ("learn_residual", "global_tail_pool", "reference_geometry")}
        if any(type(value) is not bool for value in self.components.values()):
            raise ValueError("CRR switches must be booleans")
        learn, global_pool, geometry = self.components.values()
        if (global_pool and not learn) or (geometry and not global_pool):
            raise ValueError("geometry requires global tail ranking, which requires learning")
        self.stage = "CR3" if geometry else "CR2" if global_pool else "CR1" if learn else "CR0"
        for key, default in (("steps", 64), ("train_tokens", 128), ("normal_candidates", 4096),
                             ("head_chunk_size", 4096), ("reference_sources", 3),
                             ("descriptor_pool_max_mib", 128)):
            value = config.get(key, default)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
            setattr(self, key, value)
        self.bound = float(config.get("residual_bound", .5))
        self.tail_fraction = float(config.get("normal_tail_fraction", .05))
        self.learning_rate = float(config.get("learning_rate", .01))
        self.margin = float(config.get("margin", .5))
        if (not all(math.isfinite(x) and x > 0 for x in
                    (self.bound, self.tail_fraction, self.learning_rate, self.margin))
                or self.tail_fraction > 1):
            raise ValueError("invalid CRR bound, tail fraction or optimizer setting")
        self.head = None
        self.tables = {}
        self.inference_radii = {}
        self.episodes = []
        self.pool_metadata = {}
        self.fit_backbone_calls = 0

    @torch.no_grad()
    def _fit_geometry(self):
        for resolution, bank in self.banks.items():
            owners = torch.tensor(bank["owners"], device=self.device)
            tables, radii = [], []
            for reference in bank["layers"]:
                _, _, table = source_distances(reference, reference, owners,
                                               self.support_count, self.chunk)
                tables.append(table)
                radii.append(reference_radii(table, owners, self.reference_sources))
            self.tables[resolution] = tables
            self.inference_radii[resolution] = radii

    def _geometry_bank(self, resolution, source=None):
        bank = self.banks[resolution]
        owners = torch.tensor(bank["owners"], device=self.device)
        if source is None:
            return {**bank, "owner_tensor": owners, "radii": self.inference_radii[resolution]}
        selected = owners != source
        radii = [reference_radii(table, owners, self.reference_sources, source)[selected]
                 for table in self.tables[resolution]]
        return {"layers": [layer[selected] for layer in bank["layers"]],
                "owner_tensor": owners[selected], "radii": radii}

    @torch.no_grad()
    def _descriptor_grid(self, image, resolution, bank):
        layers, grid = self._extract(image, resolution)
        distances, radii, agreements = [], [], []
        for index, (query, reference) in enumerate(zip(layers, bank["layers"])):
            distance, identity, by_source = source_distances(
                query, reference, bank["owner_tensor"], self.support_count, self.chunk)
            scale = self.scales[resolution][index].to(self.device)
            distances.append(distance / scale)
            radii.append(bank["radii"][index][identity] / scale)
            agreements.append(by_source.topk(self.reference_sources, largest=False, dim=1)
                              .values.clamp_min(0).mean(1) / scale)
        return torch.stack(distances + radii + agreements).reshape(6, *grid)

    def _cache_descriptors(self, support_paths, context):
        normal_capacity, positive_capacity = self.steps * 2 * self.normal_candidates, self.steps * 2 * self.train_tokens
        bytes_per_row = len(self.resolutions) * 6 * 4 + 8 + 4 + 4
        budget = (normal_capacity + positive_capacity) * bytes_per_row
        if budget > self.descriptor_pool_max_mib * 1024**2:
            raise ValueError("compact descriptor pool exceeds configured host-memory cap")
        normals = CompactPool(normal_capacity, len(self.resolutions))
        positives = CompactPool(positive_capacity, len(self.resolutions))
        for step in range(self.steps):
            source = step % len(support_paths)
            image = _read_rgb(support_paths[source])
            # Reuse the established deterministic deck without modifying TBF.
            views, coordinates, normal_count, episode = TailAwareFusion._episode(
                self, image, source, step, context.seed)
            positive_count = len(coordinates) - normal_count
            changed = (np.any(views[0] != views[2], axis=-1)
                       & np.any(views[1] != views[3], axis=-1))
            if _digest(changed) != episode["changed_mask_sha256"]:
                raise RuntimeError("synthetic population differs from the declared changed mask")
            changed_area = int(changed.sum())
            values = [[], [], [], []]
            for resolution in self.resolutions:
                bank = self._geometry_bank(resolution, source)
                for view_index, view in enumerate(views):
                    selected = coordinates[:normal_count] if view_index < 2 else coordinates[normal_count:]
                    grid = self._descriptor_grid(view, resolution, bank)
                    values[view_index].append(sample_native(grid, selected, image.shape[:2]).cpu().numpy())
                    self.fit_backbone_calls += 1
            for view_index, branches in enumerate(values):
                pool = normals if view_index < 2 else positives
                weight = image.shape[0] * image.shape[1] / normal_count if view_index < 2 else changed_area / positive_count
                pool.append(np.stack(branches, 1), weight, source, step)
            self.episodes.append(episode)
            self.log_training(None, phase="descriptor_cache", episodes_completed=step + 1,
                              episodes_total=self.steps, source=source,
                              episode_sha256=episode["episode_sha256"])
        self.pool_metadata = dict(normal_rows=normals.size, positive_rows=positives.size,
                                  allocated_bytes=normals.nbytes + positives.nbytes,
                                  normal_sha256=normals.digest(), positive_sha256=positives.digest())
        return normals, positives

    def _score_rows(self, pool, indices):
        values = torch.as_tensor(pool.values[indices], device=self.device)
        return self.head(values).mean(-1)

    @torch.no_grad()
    def _rescore(self, pool):
        scores = np.empty(pool.size, dtype=np.float32)
        for start in range(0, pool.size, self.head_chunk_size):
            stop = min(pool.size, start + self.head_chunk_size)
            scores[start:stop] = self._score_rows(pool, slice(start, stop)).cpu().numpy()
        if not np.isfinite(scores).all():
            raise RuntimeError("nonfinite current normal scores")
        return scores

    def _sample_pairs(self, normals, positives, scores, seed, step):
        positive_rng = np.random.default_rng(np.random.SeedSequence([seed, step, 811]))
        positive_weights = positives.weights[:positives.size]
        positive_ids = positive_rng.choice(positives.size, self.train_tokens, replace=True,
                                            p=positive_weights / positive_weights.sum())
        negative_rng = np.random.default_rng(np.random.SeedSequence([seed, step, 812]))
        uniforms = negative_rng.random((self.train_tokens, self.train_tokens))
        negative_ids = np.empty(uniforms.shape, dtype=np.int64)
        tails = {}
        for row, positive_id in enumerate(positive_ids):
            episode = int(positives.episodes[positive_id])
            key = -1 if self.components["global_tail_pool"] else episode
            if key not in tails:
                start, stop = (0, normals.size) if key == -1 else normals.ranges[episode]
                ids, mass = weighted_tail(scores[start:stop], normals.weights[start:stop], self.tail_fraction)
                cdf = np.cumsum(mass)
                cdf[-1] = 1.
                tails[key] = (ids + start, cdf)
            ids, cdf = tails[key]
            negative_ids[row] = ids[np.searchsorted(cdf, uniforms[row], side="right")]
        tail_sources = np.unique(np.concatenate([normals.sources[ids] for ids, _ in tails.values()]))
        return positive_ids, negative_ids, len(tail_sources)

    def fit(self, support_paths, context):
        self.head = None
        self.tables, self.inference_radii = {}, {}
        self.episodes, self.pool_metadata = [], {}
        self.parameter_count = self.fit_backbone_calls = 0
        if self.components["learn_residual"] and len(support_paths) < self.reference_sources + 2:
            raise ValueError("CRR learning needs at least reference_sources + 2 supports")
        super().fit(support_paths, context)
        self.fit_backbone_calls = len(support_paths) * len(self.resolutions)
        if not self.components["learn_residual"]:
            return
        self._fit_geometry()
        self.head = BoundedResidual(self.components["reference_geometry"], self.bound).to(self.device)
        self.parameter_count = sum(parameter.numel() for parameter in self.head.parameters())
        normals, positives = self._cache_descriptors(support_paths, context)
        optimizer = torch.optim.AdamW(self.head.parameters(), lr=self.learning_rate, weight_decay=0)
        for step in range(self.steps):
            current_scores = self._rescore(normals)
            positive_ids, negative_ids, coverage = self._sample_pairs(
                normals, positives, current_scores, context.seed, step)
            positive_scores = self._score_rows(positives, positive_ids)
            negative_scores = self._score_rows(normals, negative_ids.reshape(-1)).reshape(negative_ids.shape)
            loss = F.relu(self.margin - positive_scores[:, None] + negative_scores).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite CRR ranking loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = nn.utils.clip_grad_norm_(self.head.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            if not all(torch.isfinite(parameter).all() for parameter in self.head.parameters()):
                raise RuntimeError("nonfinite CRR parameters after optimizer update")
            row = dict(loss=float(loss.detach()), rank_loss=float(loss.detach()),
                       gradient_norm=float(gradient),
                       normal_selected_mean=float(negative_scores.detach().mean()),
                       anomaly_mean=float(positive_scores.detach().mean()),
                       cross_source_pair_fraction=float(np.mean(
                           normals.sources[negative_ids] != positives.sources[positive_ids, None])),
                       tail_source_coverage=coverage, pair_terms=int(negative_ids.size),
                       positive_ids_sha256=_digest(positive_ids), negative_ids_sha256=_digest(negative_ids))
            self.history.append(row)
            self.log_training(step, **row)
        self.head.eval()
        # Pools and learned-score samples are fit locals; none enter method state.

    @torch.no_grad()
    def predict(self, test_image_path):
        if not self.components["learn_residual"]:
            return super().predict(test_image_path)
        if self.head is None or len(self.history) != self.steps:
            raise RuntimeError("CRR fit must complete before prediction")
        image = _read_rgb(test_image_path)
        size = image.shape[0] * image.shape[1]
        total = torch.zeros(size, device=self.device)
        for resolution in self.resolutions:
            grid = self._descriptor_grid(image, resolution, self._geometry_bank(resolution))
            for start in range(0, size, self.head_chunk_size):
                stop = min(size, start + self.head_chunk_size)
                values = sample_native(grid, np.arange(start, stop), image.shape[:2])
                total[start:stop] += self.head(values)
        result = (total / len(self.resolutions)).reshape(image.shape[:2]).cpu().numpy()
        if not np.isfinite(result).all():
            raise RuntimeError("nonfinite CRR prediction")
        return result, float(result.max())

    def object_metadata(self):
        metadata = super().object_metadata()
        metadata.update(
            variant=self.stage, ablation=self.components, trainable_parameters=self.parameter_count,
            effective_geometry_inputs=self.components["reference_geometry"],
            residual_bound=self.bound, head_hidden_dim=16, head_shared_across_resolutions=True,
            descriptor_order=["d11", "d23", "rho11", "rho23", "m11", "m23"],
            descriptor_transform="native_bilinear_then_log1p_nonnegative_scaled_scalars",
            reference_sources=self.reference_sources,
            training_reference_policy="exclude_episode_source_from_matching_and_reference_geometry",
            reference_radius_policy="exclude_reference_owner_and_training_episode_source",
            inference_reference_policy="all_normal_supports",
            training_loss="pixel_mass_weighted_pairwise_hinge_no_auxiliary_loss",
            negative_sampling="global_class_weighted_tail" if self.components["global_tail_pool"] else "same_episode_weighted_tail",
            normal_tail_fraction=self.tail_fraction, normal_candidates=self.normal_candidates,
            positive_sampling="pixel_mass_weighted_changed_native_pixels_across_episode_deck",
            margin=self.margin, learning_rate=self.learning_rate, steps_completed=len(self.history),
            fit_backbone_calls=self.fit_backbone_calls, episode_manifest=self.episodes,
            compact_descriptor_pool=self.pool_metadata, descriptor_pool_retained_after_fit=False,
            training_quantiles_are_sampled=True, dense_maps_accumulated=False,
            score_definition="R1_exact" if self.head is None else "R1_plus_bounded_residual_then_mean_resolution",
        )
        return metadata
