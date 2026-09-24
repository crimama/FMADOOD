"""Support-calibrated local star-graph discrepancy on a frozen RSG detector.

Sorted center-to-neighbor cosine distances describe an unordered local star,
not a global registration or a guarantee of SE(3) invariance. Candidate search
uses the unchanged appearance bank. All calibration uses normal supports.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F

from fmad.methods.reference_matching import ReferenceMatching
from fmad.registry import METHOD_REGISTRY


def star_descriptor(features, grid, *, shuffled=False):
    """Eight sorted edge lengths; replicate padding defines boundary behavior.

The null permutes feature locations before constructing edges, then restores
center identities. It preserves features and edge count, but destroys locality.
An isolated CPU generator does not consume the bank sampling RNG.
"""
    h, w = grid
    if h * w != len(features):
        raise ValueError("grid and token count disagree")
    order = None
    if shuffled:
        generator = torch.Generator(device="cpu").manual_seed(918273 + h * 1009 + w)
        order = torch.randperm(len(features), generator=generator).to(features.device)
        features = features[order]
    field = features.reshape(h, w, -1).permute(2, 0, 1)[None]
    padded = F.pad(field, (1, 1, 1, 1), mode="replicate")
    edges = []
    for dy, dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
        neighbour = padded[:, :, 1+dy:1+dy+h, 1+dx:1+dx+w]
        edges.append((1 - (field * neighbour).sum(1)).flatten().clamp_min(0))
    result = torch.stack(edges, 1).sort(1).values
    if order is not None:
        restored = torch.empty_like(result)
        restored[order] = result
        result = restored
    return result


def candidate_relations(query, reference, descriptor, reference_descriptor, k):
    distance = (1 - query @ reference.T).clamp_min(0)
    values, indices = distance.topk(min(k, len(reference)), largest=False, sorted=True)
    relation = (descriptor[:, None] - reference_descriptor[indices]).abs().mean(-1)
    return values, relation


@METHOD_REGISTRY.register("local_relation")
class LocalRelation(ReferenceMatching):
    def __init__(self, backbone, config):
        super().__init__(backbone, config)
        self.relation_weight = float(config.get("relation_weight", 0.25))
        self.relation_k = config.get("relation_k", 5)
        self.relation_layer = config.get("relation_layer", 11)
        self.relation_null = config.get("relation_null", False)
        if not math.isfinite(self.relation_weight) or self.relation_weight < 0:
            raise ValueError("relation_weight must be finite and nonnegative")
        if isinstance(self.relation_k, bool) or not isinstance(self.relation_k, int) or self.relation_k < 1:
            raise ValueError("relation_k must be a positive integer")
        if self.relation_layer not in self.layers or not isinstance(self.relation_null, bool):
            raise ValueError("invalid relation layer/null")
        if self.variant != "R1" or self.score_normalization != "tailscale" or self.bank_outlier_trim:
            raise ValueError("local_relation requires R1 tailscale without trimming")
        self.relation_index = self.layers.index(self.relation_layer)
        self._relation_sources = {}
        self._relation_fitting = False
        self.relation_stats = {}

    def _extract(self, image, resolution):
        layers, grid = super()._extract(image, resolution)
        if self._relation_fitting:
            descriptor = star_descriptor(layers[self.relation_index], grid, shuffled=self.relation_null)
            self._relation_sources.setdefault(resolution, []).append(descriptor.cpu())
        return layers, grid

    def _bank_for(self, resolution, excluded=None):
        bank = super()._bank_for(resolution, excluded)
        records = [r for r in self.records[resolution] if r["source"] != excluded]
        bank["relation"] = torch.cat([
            self._relation_sources[resolution][r["source"]][r["indices"]] for r in records
        ]).to(self.device)
        if excluded is None:
            # Estimate discrepancy of appearance-nearest normal matches,
            # excluding the entire source image (or the same token at 1-shot).
            samples = []
            owners = torch.tensor(bank["owners"], device=self.device)
            reference = bank["layers"][self.relation_index]
            for start in range(0, len(reference), self.chunk):
                stop = min(start + self.chunk, len(reference))
                distance = (1 - reference[start:stop] @ reference.T).clamp_min(0)
                if len(records) > 1:
                    distance.masked_fill_(owners[start:stop, None] == owners[None], float("inf"))
                else:
                    distance[torch.arange(stop-start, device=self.device),
                             torch.arange(start, stop, device=self.device)] = float("inf")
                nearest = distance.argmin(1)
                if not torch.isfinite(distance.min(1).values).all():
                    raise ValueError("normal calibration needs non-self reference tokens")
                samples.append((bank["relation"][start:stop] - bank["relation"][nearest]).abs().mean(1))
            bank["relation_tau"] = torch.quantile(torch.cat(samples).float(), 0.95).clamp_min(1e-4)
        return bank

    def fit(self, support_paths, context):
        self._relation_sources = {}
        self._relation_fitting = True
        try:
            super().fit(support_paths, context)
        finally:
            self._relation_sources = {}
            self._relation_fitting = False
        self.relation_stats = {"tokens": 0, "positive_penalty_tokens": 0, "penalty_sum": 0.0}
        self.log_training(step=None, event="local_relation_ready", **self._relation_metadata())

    def _score_tokens(self, layers, grid, bank, *, normalize=True):
        baseline = super()._score_tokens(layers, grid, bank, normalize=normalize)
        if self.relation_weight == 0:
            return baseline
        descriptor = star_descriptor(layers[self.relation_index], grid, shuffled=self.relation_null)
        bags = bank if isinstance(bank, list) else [bank]
        penalties = []
        for bag_index, bag in enumerate(bags):
            chunks = []
            scale = self._active_scale(self.relation_index, layers[0].device, bag_index)
            for start in range(0, len(descriptor), self.chunk):
                stop = start + self.chunk
                distance, relation = candidate_relations(
                    layers[self.relation_index][start:stop], bag["layers"][self.relation_index],
                    descriptor[start:stop], bag["relation"], self.relation_k)
                # Candidate-specific appearance opportunity cost plus excess
                # relation discrepancy. Min permits alternative normal matches.
                cost = (distance - distance[:, :1]) / scale
                excess = (relation / bag["relation_tau"] - 1).clamp_min(0)
                chunks.append((cost + excess).min(1).values)
            penalties.append(torch.cat(chunks))
        penalty = torch.stack(penalties).mean(0)
        self.relation_stats["tokens"] += len(penalty)
        self.relation_stats["positive_penalty_tokens"] += int((penalty > 1e-7).sum())
        self.relation_stats["penalty_sum"] += float(penalty.sum())
        return baseline + self.relation_weight * penalty

    def _relation_metadata(self):
        return {"relation_weight": self.relation_weight, "relation_k": self.relation_k,
                "relation_layer": self.relation_layer, "relation_null": self.relation_null,
                "relation_descriptor": "sorted_8_neighbor_cosine_star_edges",
                "relation_calibration": "normal_source_excluded_nearest_match_q95; 1shot_token_exclusion",
                "relation_thresholds": {str(r): [float(b["relation_tau"]) for b in bags]
                                        for r, bags in self.bag_banks.items()},
                "relation_diagnostics": dict(self.relation_stats)}

    def object_metadata(self):
        return {**super().object_metadata(), **self._relation_metadata()}
