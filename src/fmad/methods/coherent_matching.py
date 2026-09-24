"""Local shared-reference matching versus its independent-reference relaxation.

Frozen appearance candidates; no learned parameters or query-set adaptation.
This is translation-local window consistency, not rotation invariance.
"""
from __future__ import annotations

import math
import torch

from fmad.methods.reference_matching import ReferenceMatching
from fmad.registry import METHOD_REGISTRY


def window_indices(grid, device):
    h, w = grid
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    return torch.stack([((y + dy).clamp(0, h-1)*w + (x + dx).clamp(0, w-1)).flatten()
                        for dy, dx in ((0,0),(-1,0),(1,0),(0,-1),(0,1))], 1)


def coherence_gap(query_windows, reference_windows):
    """N x K x five-position costs; independent matching is a relaxation.

Each candidate window comes from one actual support-image location. The
relaxation can choose a different candidate for each position, the constrained
match must choose one candidate for all five positions.
"""
    costs = (1 - (query_windows[:, None].float() * reference_windows.float()).sum(-1)).clamp_min(0)
    return (costs.mean(-1).amin(1) - costs.amin(1).mean(-1)).clamp_min(0)


def quantile_transfer(values, source_sorted, target_sorted):
    """Empirical rank transport, saturating outside normal calibration range."""
    ranks = torch.searchsorted(source_sorted.contiguous(), values.contiguous())
    return target_sorted[ranks.clamp_max(len(target_sorted)-1)]


@METHOD_REGISTRY.register("coherent_matching")
class CoherentMatching(ReferenceMatching):
    def __init__(self, backbone, config):
        super().__init__(backbone, config)
        self.coherence_mode = config.get("coherence_mode", "joint")
        self.coherence_weight = float(config.get("coherence_weight", .25))
        self.coherence_k = config.get("coherence_k", 5)
        self.coherence_layer = config.get("coherence_layer", 11)
        if self.coherence_mode not in {"joint", "relaxed", "magnitude"}:
            raise ValueError("invalid coherence_mode")
        if not math.isfinite(self.coherence_weight) or self.coherence_weight < 0:
            raise ValueError("invalid coherence_weight")
        if isinstance(self.coherence_k, bool) or not isinstance(self.coherence_k, int) or self.coherence_k < 2:
            raise ValueError("coherence_k must be an integer >=2")
        if self.coherence_layer not in self.layers:
            raise ValueError("invalid coherence_layer")
        if self.variant != "R1" or self.score_normalization != "tailscale" or self.bank_outlier_trim:
            raise ValueError("coherent_matching requires untrimmed R1 tailscale")
        self.coherence_index = self.layers.index(self.coherence_layer)
        self._window_sources = {}
        self._window_fitting = False
        self.coherence_stats = {}

    def _extract(self, image, resolution):
        layers, grid = super()._extract(image, resolution)
        if self._window_fitting:
            self._window_sources.setdefault(resolution, []).append((layers[self.coherence_index].cpu(), grid))
        return layers, grid

    def _bank_for(self, resolution, excluded=None):
        bank = super()._bank_for(resolution, excluded)
        records = [r for r in self.records[resolution] if r["source"] != excluded]
        windows = []
        for record in records:
            features, grid = self._window_sources[resolution][record["source"]]
            indices = window_indices(grid, features.device)[record["indices"]]
            windows.append(features[indices].half())
        bank["windows"] = torch.cat(windows).to(self.device)
        if excluded is None:
            reference = bank["layers"][self.coherence_index]
            owners = torch.tensor(bank["owners"], device=self.device)
            gaps, distances = [], []
            for start in range(0, len(reference), self.chunk):
                stop = min(start+self.chunk, len(reference))
                distance = (1-reference[start:stop] @ reference.T).clamp_min(0)
                distance.masked_fill_(owners[start:stop,None] == owners[None], float("inf"))
                # Single support cannot estimate cross-image coherence.
                if len(records) < 2:
                    raise ValueError("coherent matching calibration needs >=2 normal supports")
                count = min(self.coherence_k, int(torch.isfinite(distance).sum(1).min()))
                if count < 2:
                    raise ValueError("coherence calibration needs >=2 other-source candidates")
                values, selected = distance.topk(count, largest=False, sorted=True)
                gaps.append(coherence_gap(bank["windows"][start:stop], bank["windows"][selected]))
                distances.append(values[:,0])
            gaps = torch.cat(gaps)
            bank["coherence_tau"] = torch.quantile(gaps, .95).clamp_min(1e-4)
            bank["normal_distance"] = torch.cat(distances).sort().values
            bank["normal_penalty"] = (gaps / bank["coherence_tau"] - 1).clamp_min(0).sort().values
        return bank

    def fit(self, support_paths, context):
        if len(support_paths) < 2:
            raise ValueError("coherent matching requires >=2 normal supports")
        self._window_sources = {}
        self._window_fitting = True
        try:
            super().fit(support_paths, context)
        finally:
            self._window_sources = {}
            self._window_fitting = False
        self.coherence_stats = {"tokens":0, "positive_tokens":0, "penalty_sum":0.0}
        self.log_training(step=None, event="coherent_matching_ready", **self._coherence_metadata())

    def _score_tokens(self, layers, grid, bank, *, normalize=True):
        baseline = super()._score_tokens(layers, grid, bank, normalize=normalize)
        if self.coherence_mode == "relaxed" or self.coherence_weight == 0:
            return baseline
        query = layers[self.coherence_index]
        indices = window_indices(grid, query.device)
        bags = bank if isinstance(bank, list) else [bank]
        penalties = []
        for bag in bags:
            chunks = []
            for start in range(0, len(query), self.chunk):
                stop = min(start+self.chunk, len(query))
                distance = (1-query[start:stop] @ bag["layers"][self.coherence_index].T).clamp_min(0)
                values, selected = distance.topk(min(self.coherence_k, distance.shape[1]), largest=False, sorted=True)
                if self.coherence_mode == "magnitude":
                    penalty = quantile_transfer(values[:,0], bag["normal_distance"], bag["normal_penalty"])
                else:
                    # Both operands use the same fp16 storage precision as calibration.
                    gap = coherence_gap(query[indices[start:stop]].half(), bag["windows"][selected])
                    penalty = (gap / bag["coherence_tau"] - 1).clamp_min(0)
                chunks.append(penalty)
            penalties.append(torch.cat(chunks))
        penalty = torch.stack(penalties).mean(0)
        self.coherence_stats["tokens"] += len(penalty)
        self.coherence_stats["positive_tokens"] += int((penalty>1e-7).sum())
        self.coherence_stats["penalty_sum"] += float(penalty.sum())
        return baseline + self.coherence_weight * penalty

    def _coherence_metadata(self):
        return {"coherence_mode":self.coherence_mode, "coherence_weight":self.coherence_weight,
                "coherence_k":self.coherence_k, "coherence_layer":self.coherence_layer,
                "coherence_window":"center_and_four_axis_neighbors_replicate_boundary_fp16_storage",
                "coherence_calibration":"whole_source_excluded_normal_q95_floor_1e-4",
                "coherence_thresholds":{str(r):[float(b["coherence_tau"]) for b in bags]
                                        for r,bags in self.bag_banks.items()},
                "coherence_magnitude_control":"normal_empirical_rank_transport_saturated_tails_not_query_matched",
                "coherence_diagnostics":dict(self.coherence_stats)}

    def object_metadata(self):
        return {**super().object_metadata(), **self._coherence_metadata()}
