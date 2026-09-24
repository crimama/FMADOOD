"""Frozen R1 references with bounded, support-trained layer fusion.

Only a scalar or a 65-parameter gate is learned. Training retains one episode
of sampled scalar distances, not dense maps or backbone computation graphs.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fmad.methods.PSMA_MR import (
    _apply_photometric, _read_rgb, _sample_photometric_transform, _synthetic_anomalies,
)
from fmad.methods.reference_matching import ReferenceMatching, _distance_chunks
from fmad.registry import METHOD_REGISTRY


class BoundedFusion(nn.Module):
    def __init__(self, local: bool, bound: float):
        super().__init__()
        self.bound = bound
        if local:
            self.network = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 1))
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)
        else:
            self.network = None
            self.logit = nn.Parameter(torch.zeros(()))

    def weights(self, distances):
        # Clipping affects gate inputs only, never the original R1 distances.
        logits = (self.network(torch.log1p(distances.clamp_min(0))).squeeze(-1)
                  if self.network is not None else self.logit.expand(distances.shape[:-1]))
        return (1 + self.bound * logits.tanh()) / 2

    def forward(self, distances):
        weight = self.weights(distances)
        return weight * distances[..., 0] + (1 - weight) * distances[..., 1]


def sample_native(grid: torch.Tensor, indices: np.ndarray, shape: tuple[int, int]):
    """Sample the native bilinear map without allocating that dense map."""
    height, width = shape
    ids = torch.as_tensor(indices, device=grid.device, dtype=torch.long)
    yy, xx = ids.div(width, rounding_mode="floor"), ids.remainder(width)
    coordinates = torch.stack((2 * (xx.float() + .5) / width - 1,
                               2 * (yy.float() + .5) / height - 1), -1)
    sampled = F.grid_sample(grid.unsqueeze(0), coordinates.reshape(1, 1, -1, 2),
                            mode="bilinear", padding_mode="border", align_corners=False)
    return sampled[0, :, 0].T.contiguous()


def select_negatives(scores, *, hard: bool, fraction: float, count: int, rng):
    candidates = (scores.detach().topk(max(1, math.ceil(scores.numel() * fraction))).indices
                  if hard else torch.arange(scores.numel(), device=scores.device))
    sampled = rng.choice(len(candidates), min(count, len(candidates)), replace=False)
    return candidates[torch.as_tensor(sampled, device=scores.device)]


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@METHOD_REGISTRY.register("tail_aware_fusion")
class TailAwareFusion(ReferenceMatching):
    def __init__(self, backbone, config):
        if config.get("variant", "R1") != "R1":
            raise ValueError("tail_aware_fusion requires the R1 reference scorer")
        super().__init__(backbone, {**config, "variant": "R1"})
        self.components = {key: config.get(key, False) for key in
                           ("learn_fusion", "hard_normal_ranking", "local_fusion")}
        if any(type(value) is not bool for value in self.components.values()):
            raise ValueError("fusion switches must be booleans")
        learn, hard, local = self.components.values()
        if (hard and not learn) or (local and not hard):
            raise ValueError("local fusion requires hard ranking, which requires learning")
        self.stage = "TF3" if local else "TF2" if hard else "TF1" if learn else "TF0"
        for key, default in (("steps", 64), ("train_tokens", 128),
                             ("normal_candidates", 4096), ("fusion_chunk_size", 4096)):
            value = config.get(key, default)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
            setattr(self, key, value)
        self.bound = float(config.get("fusion_bound", .5))
        self.tail_fraction = float(config.get("normal_tail_fraction", .05))
        self.learning_rate = float(config.get("learning_rate", .01))
        self.margin = float(config.get("margin", .5))
        self.pair_weight = float(config.get("pair_weight", .1))
        if not 0 < self.bound < 1 or not 0 < self.tail_fraction <= 1:
            raise ValueError("fusion bound and tail fraction are outside valid ranges")
        if (not all(math.isfinite(x) for x in
                    (self.learning_rate, self.margin, self.pair_weight)) or
                self.learning_rate <= 0 or self.margin <= 0 or self.pair_weight < 0):
            raise ValueError("invalid optimizer/loss settings")
        self.gate = None
        self.episodes = []
        self.fit_backbone_calls = 0

    def _source_excluded(self, source):
        banks = {}
        for resolution, bank in self.banks.items():
            selected = torch.tensor(bank["owners"], device=self.device) != source
            if not selected.any():
                raise ValueError("source exclusion leaves an empty reference bank")
            banks[resolution] = {
                "layers": [layer[selected] for layer in bank["layers"]],
                "owners": [owner for owner in bank["owners"] if owner != source],
            }
        return banks

    @torch.no_grad()
    def _distance_grid(self, image, resolution, bank):
        layers, grid = self._extract(image, resolution)
        scales = self.scales[resolution].to(self.device)
        distances = [_distance_chunks(query, reference, self.chunk) / scales[index]
                     for index, (query, reference) in enumerate(zip(layers, bank["layers"]))]
        return torch.stack(distances).reshape(2, *grid)

    @torch.no_grad()
    def _sample_views(self, views, banks, coordinates):
        sampled = []
        for view in views:
            branches = []
            for resolution in self.resolutions:
                grid = self._distance_grid(view, resolution, banks[resolution])
                branches.append(sample_native(grid, coordinates, view.shape[:2]))
                self.fit_backbone_calls += 1
            sampled.append(torch.stack(branches, 1))
        # [view, native sample, resolution, layer]; no dense maps retained.
        return torch.stack(sampled)

    def _episode(self, image, source, step, seed):
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, 809]))
        family_index = (step // self.support_count) % 2
        anomaly, mask, family = _synthetic_anomalies(image, rng, [])[family_index]
        transform = _sample_photometric_transform(rng, 1.)
        shifted = _apply_photometric(image, transform)
        shifted_anomaly = _apply_photometric(anomaly, transform)
        # Both paired views must actually differ; saturation is not a label.
        changed = mask & np.any(image != anomaly, axis=-1) & np.any(shifted != shifted_anomaly, axis=-1)
        positive = np.flatnonzero(changed)
        if not len(positive):
            raise RuntimeError("synthetic episode contains no changed pixels after shift")
        positive = rng.choice(positive, min(self.train_tokens, len(positive)), replace=False)
        normal = rng.choice(image.shape[0] * image.shape[1],
                            min(self.normal_candidates, image.shape[0] * image.shape[1]),
                            replace=False)
        coordinates = np.concatenate((normal, positive))
        views = (image, shifted, anomaly, shifted_anomaly)
        record = dict(step=step, source=source, synthetic_family=family,
                      excluded_sources=[source], transform=transform,
                      view_sha256=[_digest(view) for view in views],
                      changed_mask_sha256=_digest(changed), coordinates_sha256=_digest(coordinates),
                      normal_candidates_per_view=len(normal), positive_pixels=len(positive))
        record["episode_sha256"] = hashlib.sha256(
            json.dumps(record, sort_keys=True).encode()).hexdigest()
        return views, coordinates, len(normal), record

    def fit(self, support_paths, context):
        self.gate = None
        self.episodes = []
        self.fit_backbone_calls = 0
        super().fit(support_paths, context)
        self.fit_backbone_calls = len(support_paths) * len(self.resolutions)
        if not self.components["learn_fusion"]:
            return
        self.gate = BoundedFusion(self.components["local_fusion"], self.bound).to(self.device)
        self.parameter_count = sum(parameter.numel() for parameter in self.gate.parameters())
        optimizer = torch.optim.AdamW(self.gate.parameters(), lr=self.learning_rate, weight_decay=0)
        for step in range(self.steps):
            source = step % len(support_paths)
            image = _read_rgb(support_paths[source])
            views, coordinates, normal_count, episode = self._episode(image, source, step, context.seed)
            banks = self._source_excluded(source)
            values = self._sample_views(views, banks, coordinates)
            del views, image, banks
            scores = self.gate(values).mean(-1)
            negatives = scores[:2, :normal_count].reshape(-1)
            rng = np.random.default_rng(np.random.SeedSequence([context.seed, step, 810]))
            indices = select_negatives(negatives, hard=self.components["hard_normal_ranking"],
                                       fraction=self.tail_fraction, count=self.train_tokens, rng=rng)
            positives = scores[2:, normal_count:].reshape(-1)
            rank = F.relu(self.margin - positives[:, None] + negatives[indices][None]).mean()
            delta = scores[2, normal_count:] - scores[0, normal_count:]
            shifted_delta = scores[3, normal_count:] - scores[1, normal_count:]
            pair = (delta - shifted_delta).square().mean()
            loss = rank + self.pair_weight * pair
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite fusion training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = nn.utils.clip_grad_norm_(self.gate.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                weights = self.gate.weights(values)
                row = dict(loss=float(loss), rank_loss=float(rank), pair_loss=float(pair),
                           gradient_norm=float(gradient), source=source, synthetic_family=episode["synthetic_family"],
                           episode_sha256=episode["episode_sha256"],
                           normal_selected_mean=float(negatives[indices].mean()),
                           normal_p95=float(torch.quantile(negatives, .95)),
                           anomaly_mean=float(positives.mean()),
                           middle_weight_mean=float(weights.mean()), middle_weight_min=float(weights.min()),
                           middle_weight_max=float(weights.max()),
                           normal_selected_count=len(indices), positive_count=positives.numel())
            self.history.append(row)
            self.episodes.append(episode)
            self.log_training(step, **row)
            del values, scores, negatives, positives, loss, rank, pair, delta, shifted_delta, weights
        self.gate.eval()

    @torch.no_grad()
    def predict(self, test_image_path):
        if not self.components["learn_fusion"]:
            return super().predict(test_image_path)
        if self.gate is None:
            raise RuntimeError("fit must complete before prediction")
        image = _read_rgb(test_image_path)
        total = None
        for resolution in self.resolutions:
            grid = self._distance_grid(image, resolution, self.banks[resolution])
            native = F.interpolate(grid.unsqueeze(0), size=image.shape[:2],
                                   mode="bilinear", align_corners=False)[0]
            distances = native.flatten(1).T
            result = torch.empty(len(distances), device=self.device)
            for start in range(0, len(distances), self.fusion_chunk_size):
                stop = start + self.fusion_chunk_size
                result[start:stop] = self.gate(distances[start:stop])
            total = result if total is None else total + result
        result = (total / len(self.resolutions)).reshape(image.shape[:2]).cpu().numpy()
        if not np.isfinite(result).all():
            raise RuntimeError("nonfinite fusion prediction")
        return result, float(result.max())

    def object_metadata(self):
        metadata = super().object_metadata()
        metadata.update(
            variant=self.stage, ablation=self.components, fusion_bound=self.bound,
            gate_inputs="log1p_nonnegative_support_scaled_layer_distances",
            gate_shared_across_resolutions=True, gate_hidden_dim=16 if self.components["local_fusion"] else None,
            trainable_parameters=self.parameter_count, steps_completed=len(self.history),
            normal_tail_fraction=self.tail_fraction, normal_candidates=self.normal_candidates,
            negative_sampling="current_sampled_top_tail" if self.components["hard_normal_ranking"] else "uniform",
            positive_sampling="uniform_changed_native_pixels_within_episode",
            training_reference_policy="leave_source_out_same_image_synthesis_no_external_donor",
            inference_reference_policy="all_normal_supports",
            scale_reference="R1_frozen_independent_nn_leave_source_out",
            training_loss="pixel_pairwise_hinge_plus_shift_contrast_consistency",
            learning_rate=self.learning_rate, margin=self.margin, pair_weight=self.pair_weight,
            fit_backbone_calls=self.fit_backbone_calls, episode_manifest=self.episodes,
            score_definition="R1_exact" if self.gate is None else "bounded_layer_fusion_then_mean_resolution",
            training_quantiles_are_sampled=True, dense_maps_accumulated=False,
        )
        return metadata
