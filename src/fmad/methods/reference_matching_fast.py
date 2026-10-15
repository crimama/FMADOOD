"""Opt-in exact compute reuse for reference matching.

This variant preserves ``ReferenceMatching`` scoring.  The optional source-LOO
path computes one full GPU bank and masks source-owned candidates per query
chunk; the optional bag path evaluates a shared, identity-deduplicated union
and gathers each bag's original members before nearest-neighbour reduction.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any

import torch
from torch.nn import functional as F

from fmad.methods.PSMA_MR import _infer_grid
from fmad.methods.base import Prediction
from fmad.methods.reference_matching import ReferenceMatching
from fmad.registry import METHOD_REGISTRY


def _device_feature_matrix(features: Any, grid_size: Any, device: torch.device):
    """Normalize common feature layouts without forcing a device tensor to CPU."""
    tensor = torch.as_tensor(features, dtype=torch.float32).detach().to(device)
    if tensor.ndim == 0:
        raise ValueError("backbone feature output must have at least one dimension")
    while tensor.ndim > 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError("backbone feature output must contain [patches, dimension]")

    requested = None
    if grid_size is not None:
        try:
            values = tuple(int(v) for v in grid_size)
            if len(values) == 2 and values[0] > 0 and values[1] > 0:
                requested = values
        except (TypeError, ValueError):
            requested = None
    if requested is not None and tensor.shape[0] != requested[0] * requested[1]:
        if tensor.shape[1] == requested[0] * requested[1]:
            tensor = tensor.T.contiguous()
    grid = _infer_grid(requested, int(tensor.shape[0]))
    return tensor.contiguous(), grid


def _masked_loo_distances(
    query: torch.Tensor,
    bank: torch.Tensor,
    owners: torch.Tensor,
    excluded_owner: int,
    chunk: int,
    k: int,
) -> torch.Tensor:
    """Compute exact source-LOO kNN distances against a reusable full bank.

    Each query chunk multiplies against the same full-bank tensor. Distances
    owned by ``excluded_owner`` are masked before top-k, with per-source k
    clamped to the number of eligible tokens just as ``_distance_chunks`` does
    for a physically source-excluded bank.
    """
    excluded = owners == int(excluded_owner)
    effective_k = min(int(k), int((~excluded).sum().item()))
    if effective_k <= 0:
        raise ValueError("source-excluded fitting requires at least two supports")
    values = []
    for start in range(0, len(query), chunk):
        distances = 1.0 - query[start:start + chunk] @ bank.T
        distances.masked_fill_(excluded.unsqueeze(0), 10.0)
        if effective_k <= 1:
            values.append(distances.amin(dim=1))
        else:
            values.append(distances.topk(effective_k, dim=1, largest=False).values.mean(dim=1))
    return torch.cat(values) if values else query.new_empty((0,))


def _build_shared_bag_union(
    bag_banks: list[torch.Tensor],
    keys_by_bag: list[list[tuple[int, int, int]]],
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Build a deduplicated token union and each bag's original membership map."""
    all_rows = torch.cat(bag_banks, dim=0)
    unique: dict[tuple[int, int, int], int] = {}
    first_positions: list[int] = []
    members = []
    offset = 0
    for bank, keys in zip(bag_banks, keys_by_bag):
        local = []
        for row, raw_key in enumerate(keys):
            key = tuple(raw_key)
            union_index = unique.get(key)
            if union_index is None:
                union_index = len(first_positions)
                unique[key] = union_index
                first_positions.append(offset + row)
            local.append(union_index)
        members.append(torch.as_tensor(local, dtype=torch.long, device=device))
        offset += len(bank)
    first = torch.as_tensor(first_positions, dtype=torch.long, device=device)
    return all_rows.index_select(0, first), members


def _shared_bag_distances(
    query: torch.Tensor,
    union: torch.Tensor,
    member_indices: list[torch.Tensor],
    chunk: int,
    k: int = 1,
    *,
    return_indices: bool = False,
    index_k: int | None = None,
):
    """Compute each bag's kNN distances from one cached union matmul per chunk."""
    if not member_indices:
        return ([], []) if return_indices else []
    if union.ndim != 2 or union.device != query.device or union.shape[1] != query.shape[1]:
        raise ValueError("query and shared bank union must be compatible matrices")
    value_parts = [[] for _ in member_indices]
    index_parts = [[] for _ in member_indices] if return_indices else None
    requested_index_k = int(index_k if index_k is not None else k)
    for start in range(0, len(query), chunk):
        query_chunk = query[start:start + chunk]
        union_distances = 1.0 - query_chunk @ union.T
        for bag_index, member in enumerate(member_indices):
            distances = union_distances.index_select(1, member)
            effective_k = min(int(k), distances.shape[1])
            if effective_k <= 0:
                raise ValueError("bag banks must contain at least one reference token")
            if effective_k <= 1:
                value_parts[bag_index].append(distances.amin(dim=1))
            else:
                value_parts[bag_index].append(
                    distances.topk(effective_k, dim=1, largest=False).values.mean(dim=1))
            if return_indices:
                effective_index_k = min(requested_index_k, distances.shape[1])
                if effective_index_k <= 1:
                    index_parts[bag_index].append(distances.argmin(dim=1, keepdim=True))
                else:
                    index_parts[bag_index].append(
                        distances.topk(effective_index_k, dim=1, largest=False).indices)
    values = [torch.cat(parts) if parts else query.new_empty((0,)) for parts in value_parts]
    if not return_indices:
        return values
    indices = [
        torch.cat(parts) if parts else query.new_empty((0, min(requested_index_k, len(member))), dtype=torch.long)
        for parts, member in zip(index_parts, member_indices)
    ]
    return values, indices


@METHOD_REGISTRY.register("reference_matching_fast")
class ReferenceMatchingFast(ReferenceMatching):
    """Reference matcher with optional source-bank and bag-distance reuse."""

    def __init__(self, backbone, config: dict):
        super().__init__(backbone, config)
        self.ablation_shallow_mode = config.get("ablation_shallow_mode", "gated")
        self.ablation_feature_matching = config.get("ablation_feature_matching", "layerwise")
        if self.ablation_feature_matching not in {"layerwise", "concat"}:
            raise ValueError("ablation_feature_matching must be layerwise or concat")
        # concat: unit-normalised layers are joined (each scaled 1/sqrt(L)) so one
        # bank and one cosine distance equal the mean of per-layer cosine distances
        # under a single shared reference token. The backbone still extracts every
        # declared layer; downstream scoring then sees a single feature layer.
        self._concat_source_layers = None
        if self.ablation_feature_matching == "concat":
            if self.variant != "R1" or len(self.layers) < 2:
                raise ValueError("concat matching requires R1 with at least two feature layers")
            if self.ablation_shallow_mode != "gated":
                raise ValueError("concat matching uses the single fused feature; keep ablation_shallow_mode=gated")
            self._concat_source_layers = tuple(self.layers)
            self.layers = (self.layers[-1],)
        self.ablation_bank_audit = config.get("ablation_bank_audit", False)
        if type(self.ablation_bank_audit) is not bool:
            raise ValueError("ablation_bank_audit must be a bool")
        if self.ablation_shallow_mode not in {"gated", "all", "hard", "off"}:
            raise ValueError("ablation_shallow_mode must be gated, all, hard, or off")
        if self.ablation_shallow_mode != "gated" and (
                self.variant != "R1" or self.layer_fusion not in {"residual_spike", "deep_plus_shallow"}
                or self.layer_fusion_neg_alpha != 0):
            raise ValueError("shallow ablation requires R1 residual or linear fusion without negative branch")
        self.fast_reuse_calibration_bank = config.get("fast_reuse_calibration_bank", False)
        self.fast_shared_bag_distances = config.get("fast_shared_bag_distances", False)
        if type(self.fast_reuse_calibration_bank) is not bool:
            raise ValueError("fast_reuse_calibration_bank must be a bool")
        if type(self.fast_shared_bag_distances) is not bool:
            raise ValueError("fast_shared_bag_distances must be a bool")
        self._fast_full_bank_cache: tuple[int, int, dict[str, Any]] | None = None
        self._shared_union_cache: dict[tuple[int, ...], tuple[torch.Tensor, list[torch.Tensor]]] = {}
        self.calibration_audit = config.get('calibration_audit', False)
        if type(self.calibration_audit) is not bool:
            raise ValueError('calibration_audit must be boolean')
        if self.calibration_audit and (self.variant != 'R1' or self.local_reliability != 'off'
                or self.score_normalization not in {'none', 'tailscale'}
                or self.ablation_feature_matching != 'layerwise'):
            raise ValueError('calibration audit requires plain R1 none/q95 layerwise scoring')
        self._calibration_bags = None
        self._calibration_grids = None

    def _observe_layer_scores(self, distances, scores, *, tau):
        super()._observe_layer_scores(distances, scores, tau=tau)
        if self._calibration_bags is not None:
            self._calibration_bags.append(torch.stack([
                torch.stack(distances), torch.stack(scores)]).detach())

    def predict_batch(self, test_image_paths):
        if not self.calibration_audit:
            return super().predict_batch(test_image_paths)
        predictions = []
        for path in test_image_paths:
            self._calibration_grids = []
            started = time.perf_counter()
            try:
                score, image_score = self.predict(path)
                if len(self._calibration_grids) != len(self.last_resolutions):
                    raise RuntimeError('incomplete calibration audit')
                predictions.append(Prediction(score, image_score, time.perf_counter() - started,
                    {'calibration_audit': {'grids': self._calibration_grids,
                     'resolutions': list(self.last_resolutions), 'layers': list(self.layers)}}))
            finally:
                self._calibration_grids = None
                self._calibration_bags = None
        return predictions

    def _fuse_layers(self, scores, tau=None):
        if self.ablation_shallow_mode == "gated":
            return super()._fuse_layers(scores, tau=tau)
        if len(scores) < 2:
            raise ValueError("shallow ablation requires at least two layers")
        index = min(self.layer_fusion_shallow_idx, len(scores) - 1)
        shallow = scores[index]
        deep = torch.stack([s for i, s in enumerate(scores) if i != index]).mean(0)
        if self._noise_suppress_shallow or self.ablation_shallow_mode == "off":
            return deep
        contribution = shallow
        if self.ablation_shallow_mode == "hard":
            threshold = self.layer_fusion_tau if tau is None else tau
            contribution = shallow * (shallow > threshold)
        return deep + self.layer_fusion_gamma * contribution

    def fit(self, support_paths: list[str], context) -> None:
        self._shared_union_cache = {}
        self._fast_full_bank_cache = None
        super().fit(support_paths, context)
        if not self.fast_shared_bag_distances or self.variant != "R1" or self.bank_bags <= 1:
            return
        # Build cached unions during fit so query timing measures only scoring.
        for resolution, bags in self.bag_banks.items():
            for layer_index in range(len(self.layers)):
                values = [bag["layers"][layer_index] for bag in bags]
                key = tuple(id(value) for value in values)
                self._shared_union_cache[key] = _build_shared_bag_union(
                    values, [bag["token_keys"] for bag in bags], self.device)

    def _extract(self, image, resolution: int, pool=None):
        """Extract FP32 features on the model device when the backbone returns them there."""
        tensor, grid = self.backbone.prepare_image(image, resolution=resolution)
        with torch.no_grad():
            outputs = self.backbone.extract_features(tensor)
            source_layers = self._concat_source_layers or self.layers
            declared = tuple(getattr(self.backbone, "feature_layers", source_layers))
            by_layer = {layer: output for layer, output in zip(declared, outputs)}
            if any(layer not in by_layer for layer in source_layers):
                raise ValueError(f"backbone did not return required layers {list(source_layers)}")
            layers = []
            for layer in source_layers:
                matrix, output_grid = _device_feature_matrix(by_layer[layer], grid, self.device)
                if tuple(output_grid) != tuple(grid):
                    raise ValueError("feature layer grids must match")
                if pool is None:
                    matrix = self._apply_feature_pooling(matrix, output_grid)
                elif pool is True:
                    matrix = self._apply_feature_pooling(matrix, output_grid, enable=True)
                layers.append(F.normalize(matrix, dim=-1))
            if self._concat_source_layers is not None:
                layers = [F.normalize(torch.cat(layers, dim=-1), dim=-1)]
        return layers, tuple(grid)

    def _bank_for(self, resolution: int, excluded: int | None = None):
        bank = super()._bank_for(resolution, excluded=excluded)
        records = [record for record in self.records[resolution]
                   if record["source"] != excluded]
        # The keys let bag scoring share a distance to the same sampled support
        # token across bags while retaining every bag's original membership.
        keys = []
        for record in records:
            keys.extend((int(record["source"]), int(index), 0)
                        for index in record["indices"].tolist())
        bank["token_keys"] = keys
        if excluded is None:
            self._fast_full_bank_cache = (resolution, id(self.records[resolution]), bank)
        return bank

    def _fit_scales_for_resolution(self, resolution: int) -> torch.Tensor:
        records = self.records[resolution]
        if (not self.fast_reuse_calibration_bank or self.loo_estimator != "source"
                or len(records) <= 1):
            return super()._fit_scales_for_resolution(resolution)

        cached = self._fast_full_bank_cache
        if (cached is not None and cached[0] == resolution
                and cached[1] == id(records)):
            bank = cached[2]
        else:
            bank = self._bank_for(resolution)
        owners = torch.as_tensor(bank["owners"], dtype=torch.long, device=self.device)
        values = [[] for _ in self.layers]
        for record in records:
            for layer_index, layer in enumerate(record["layers"]):
                distances = _masked_loo_distances(
                    layer.to(self.device), bank["layers"][layer_index], owners,
                    int(record["source"]), self.chunk, self.knn_k,
                )
                values[layer_index].append(distances.detach().cpu())

        # This is the base calibration reducer, kept local to the optimized
        # multi-source branch. One-shot and token-LOO use the unchanged parent
        # implementation above.
        scales = []
        tails = []
        local = [] if self.local_reliability != "off" else None
        for layer_values in values:
            sample = torch.cat(layer_values)
            median = sample.median()
            mad = (sample - median).abs().median()
            mad_scale = (1.4826 * mad).clamp_min(1e-4)
            tail_dict = self._tail_params(sample)
            if self.layer_fusion == "residual_spike" and self.layer_fusion_tau_mode == "support_quantile":
                if self.score_normalization == "tailscale":
                    divisor = tail_dict["u"].item()
                elif self.score_normalization == "none":
                    divisor = 1.0
                else:
                    divisor = mad_scale.item()
                q_val = torch.quantile(sample.detach().float().cpu(), self.layer_fusion_tau_q).item()
                tail_dict["spike_tau"] = float(q_val / max(divisor, 1e-4))
                if self.image_residual_tau_q is not None:
                    q_img = torch.quantile(sample.detach().float().cpu(), self.image_residual_tau_q).item()
                    tail_dict["image_spike_tau"] = float(q_img / max(divisor, 1e-4))
            tails.append(tail_dict)
            if self.score_normalization == "tailscale":
                scales.append(tails[-1]["u"].clamp_min(1e-4))
            elif self.score_normalization == "none":
                scales.append(torch.ones_like(mad_scale))
            else:
                scales.append(mad_scale)
            if local is not None:
                local.append(sample)
        self._last_tails = tails
        self._last_local = local
        return torch.stack(scales)

    def _score_tokens(self, layers, grid, bank, *, normalize=True):
        if self._calibration_grids is None:
            return self._score_tokens_impl(layers, grid, bank, normalize=normalize)
        self._calibration_bags = []
        try:
            score = self._score_tokens_impl(layers, grid, bank, normalize=normalize)
            values = torch.stack(self._calibration_bags).mean(0)
            self._calibration_grids.append(values.reshape(2, len(self.layers), *grid).cpu().numpy())
            return score
        finally:
            self._calibration_bags = None

    def _score_tokens_impl(self, layers, grid, bank, *, normalize=True):
        if not (self.fast_shared_bag_distances and self.variant == "R1"
                and isinstance(bank, list) and len(bank) > 1):
            return super()._score_tokens(layers, grid, bank, normalize=normalize)

        bags = bank
        neighbor_k = max(self.knn_k, self.local_reliability_k)
        shared_per_layer = []
        for layer_index, query in enumerate(layers):
            values = [bag["layers"][layer_index] for bag in bags]
            cache_key = tuple(id(value) for value in values)
            union_pair = self._shared_union_cache.get(cache_key)
            if union_pair is None:
                union_pair = _build_shared_bag_union(
                    values, [bag["token_keys"] for bag in bags], query.device)
                self._shared_union_cache[cache_key] = union_pair
            shared_per_layer.append(_shared_bag_distances(
                query, union_pair[0], union_pair[1], self.chunk, self.knn_k,
                return_indices=normalize and self.local_reliability != "off",
                index_k=neighbor_k,
            ))

        bag_scores = []
        image_bag_scores = []
        for bag_index, bag_bank in enumerate(bags):
            scores = []
            distances = []
            weights = [] if (normalize and self.local_reliability == "weight") else None
            for layer_index, values in enumerate(bag_bank["layers"]):
                if normalize and self.local_reliability != "off":
                    distance_values, neighbour_indices = shared_per_layer[layer_index]
                    distance = distance_values[bag_index]
                    nn_indices = neighbour_indices[bag_index]
                else:
                    distance = shared_per_layer[layer_index][bag_index]
                distances.append(distance)
                global_scale = self._active_scale(layer_index, values.device, bag_index)
                sigma_local = None
                if normalize and self.local_reliability != "off":
                    loo = self._active_local(layer_index, bag_index).to(values.device)
                    if self.local_reliability != "center":
                        sigma_local = self._local_scale_from_indices(
                            nn_indices, loo, global_scale, self.local_reliability_k,
                            self.local_reliability_shrinkage,
                        )
                if normalize and self.local_reliability == "center":
                    scores.append(self._local_center_score(
                        distance, nn_indices, loo, global_scale))
                elif normalize and self.score_normalization == "tail":
                    scores.append(self._tail_transform(
                        self._active_tail(layer_index, bag_index), distance))
                elif normalize and sigma_local is not None:
                    scores.append(distance / sigma_local)
                else:
                    scores.append(distance / (global_scale if normalize else 1.0))
                if weights is not None:
                    weights.append(1.0 / sigma_local.clamp_min(1e-4).pow(2))

            if weights is not None:
                bag_scores.append(self._weighted_fuse(scores, weights))
            else:
                tau = None
                if self.layer_fusion == "residual_spike" and self.layer_fusion_tau_mode == "support_quantile":
                    active_tail_dict = self._active_tail(self.layer_fusion_shallow_idx, bag_index)
                    tau = active_tail_dict.get("spike_tau", self.layer_fusion_tau)
                self._observe_layer_scores(distances, scores, tau=tau)
                bag_scores.append(self._fuse_layers(scores, tau=tau))
                if self._image_token_records is not None:
                    image_bag_scores.append(self._image_gate_fuse(scores, bag_index))
        self._record_image_tokens(image_bag_scores, grid)
        return torch.stack(bag_scores).mean(0)

    @staticmethod
    def _local_scale_from_indices(neighbour_index, loo, global_scale, k, shrinkage):
        neighbour_index = neighbour_index[:, :k]
        n_eff = neighbour_index.shape[1]
        local_mean = loo[neighbour_index].mean(dim=1)
        return ((n_eff * local_mean + shrinkage * global_scale) /
                (n_eff + shrinkage)).clamp_min(1e-4)

    def object_metadata(self) -> dict:
        metadata = super().object_metadata()
        metadata["ablation_shallow_mode"] = self.ablation_shallow_mode
        metadata["ablation_feature_matching"] = self.ablation_feature_matching
        if self._concat_source_layers is not None:
            metadata["feature_layers"] = list(self._concat_source_layers)
            metadata["score_definition"] += "_feature_concat_matching"
        if self.ablation_shallow_mode != "gated":
            metadata["score_definition"] += f"_shallow_{self.ablation_shallow_mode}"
        metadata["fast_reuse_calibration_bank"] = self.fast_reuse_calibration_bank
        metadata["fast_shared_bag_distances"] = self.fast_shared_bag_distances
        if self.ablation_bank_audit:
            digest = hashlib.sha256()
            calibration = {}
            for resolution in sorted(self.banks):
                bags = self.bag_banks[resolution] if self.bank_bags > 1 else [self.banks[resolution]]
                for bag_index, bank in enumerate(bags):
                    for layer_index, layer in enumerate(bank["layers"]):
                        value = layer.detach().cpu().contiguous().numpy()
                        digest.update(str((resolution, bag_index, layer_index, value.shape, str(value.dtype))).encode())
                        digest.update(value.tobytes())
                scales = self.bag_scales.get(resolution, [self.scales[resolution]])
                tails = self.bag_tails.get(resolution, [self.tails[resolution]])
                calibration[str(resolution)] = {
                    "scales": [[float(v) for v in scale] for scale in scales],
                    "shallow_tau": [float(bag[self.layer_fusion_shallow_idx].get("spike_tau", self.layer_fusion_tau))
                                    for bag in tails],
                }
            metadata["ablation_bank_feature_sha256"] = digest.hexdigest()
            metadata["ablation_calibration"] = calibration
        return metadata
