"""FBPA: pack a fixed normal-support synthetic episode deck into paired updates.

M0/M1/M2 differ only in packing/order. All inputs, randomness and reference
exclusions belong to an episode, not its execution step. Dense state lives for
one update; metadata retains only compact scalar/hash provenance.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fmad.methods.PSMA_MR import _apply_photometric, _sample_photometric_transform
from fmad.methods.tail_contrast import ResidualAdapter
from fmad.methods.transfer_update import TransferUpdate, _FunctionalScorer
from fmad.registry import METHOD_REGISTRY


def episode_schedule(family_balance, source_pairing):
    """Return64 two-episode updates over the same16x4x2 immutable deck."""
    if source_pairing and not family_balance:
        raise ValueError("source_pairing requires family_balance")
    pairs = [(a, a + 1) for a in range(0, 16, 2)]
    if not family_balance:
        return [((a, r, f), (b, r, f))
                for h in range(2) for f in range(2)
                for r in (2 * h, 2 * h + 1) for a, b in pairs]
    batches = []
    for r in range(4):
        for a, b in pairs:
            if source_pairing:
                batches.extend([((a, r, 0), (a, r, 1)),
                                ((b, r, 0), (b, r, 1))])
            else:
                batches.extend([((a, r, 0), (b, r, 1)),
                                ((a, r, 1), (b, r, 0))])
    return batches


def array_digest(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(json.dumps([list(value.shape), str(value.dtype)]).encode())
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


@METHOD_REGISTRY.register("family_balanced_paired")
class FamilyBalancedPaired(TransferUpdate):
    def __init__(self, backbone, config):
        config = {"cross_environment_meta": False, **config}
        super().__init__(backbone, config)
        if self.meta_enabled or not self.transfer_enabled or self.outer_weight != 1.:
            raise ValueError("FBPA requires L3C (transfer on, meta off, outer weight1)")
        # Do not overwrite BoundarySynthetic.enabled: weak selection uses it.
        self.episode_enabled = config.get("episode_enabled", True)
        self.family_balance = config.get("family_balance", True)
        self.source_pairing = config.get("source_pairing", True)
        if any(type(v) is not bool for v in
               (self.episode_enabled, self.family_balance, self.source_pairing)):
            raise ValueError("FBPA switches must be booleans")
        if self.source_pairing and not self.family_balance:
            raise ValueError("source_pairing requires family_balance")
        self.episode_seed = config.get("episode_seed", 0)
        repeats = config.get("episode_repeats", 4)
        if type(self.episode_seed) is not int or self.episode_seed < 0:
            raise ValueError("episode_seed must be a nonnegative integer")
        if type(repeats) is not int or repeats != 4:
            raise ValueError("FBPA requires four episode repeats")
        if config.get("reference_policy", "fixed_pair_excluded") != "fixed_pair_excluded":
            raise ValueError("FBPA requires fixed_pair_excluded references")
        if self.episode_enabled and self.steps != 64:
            raise ValueError("FBPA requires64 real updates; cannot truncate the deck")
        self.episode_manifest = []
        self._fit_extract_calls = None

    def extract(self, image, resolution, flips=(False, False)):
        result = super().extract(image, resolution, flips)
        if self._fit_extract_calls is not None:
            self._fit_extract_calls += 1
        return result

    def _seed(self, source, repeat, family, purpose):
        payload = ["fbpa_v1", self._fit_seed, self.episode_seed,
                   self._source_ids[source], repeat, family, purpose]
        digest = hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).digest()
        return int.from_bytes(digest[:8], "big") % (2**63)

    @torch.no_grad()
    def _cache_view(self, view):
        cached = {}
        for resolution in self.resolutions:
            features, grid = self.extract(view, resolution)
            cached[resolution] = (features.cpu(), grid)
            self._view_extract_calls += 1
        return cached

    def _keyed_episode(self, images, token, excluded, banks, clean_cache):
        source, repeat, family_index = token
        image = images[source]
        anomaly_seed = self._seed(source, repeat, family_index, "anomaly")
        shift_seed = self._seed(source, repeat, None, "shift")
        rank_seed = self._seed(source, repeat, family_index, "ranking")
        anomaly, mask, family, selection = self._episode_anomaly(
            image, np.random.default_rng(anomaly_seed), family_index, source, excluded, banks)
        transform = _sample_photometric_transform(np.random.default_rng(shift_seed), 1.)
        key = (source, repeat)
        if key not in clean_cache:
            shifted = _apply_photometric(image, transform)
            clean_views = [self._cache_view(view) for view in (image, shifted)]
            with torch.no_grad():
                teacher, shifted_teacher = [self.fused(self._branches_to_device(view), banks,
                                                       image.shape[:2], baseline=True).cpu()
                                            for view in clean_views]
                susceptibility = float((shifted_teacher - teacher).abs().mean() /
                                       teacher.abs().mean().clamp_min(1e-6))
            clean_cache[key] = dict(views=clean_views, teacher=teacher,
                                    budget=self.budget_base /
                                    (1. + self.susceptibility_scale * susceptibility),
                                    hashes=[array_digest(image), array_digest(shifted)])
        clean = clean_cache[key]
        shifted_anomaly = _apply_photometric(anomaly, transform)
        cached = clean["views"] + [self._cache_view(view)
                                   for view in (anomaly, shifted_anomaly)]
        provenance = dict(source=source, source_content_sha256=self._source_ids[source],
                          repeat=repeat, family_index=family_index, family=family,
                          anomaly_seed=anomaly_seed, shift_seed=shift_seed, rank_seed=rank_seed,
                          excluded_sources=list(excluded),
                          view_sha256=clean["hashes"] + [array_digest(anomaly),
                                                         array_digest(shifted_anomaly)],
                          mask_sha256=array_digest(mask), **selection)
        episode = dict(cached=cached, teacher=clean["teacher"], mask=mask,
                       shape=image.shape[:2], budget=clean["budget"])
        return episode, provenance

    def fit(self, support_paths, context):
        if not self.episode_enabled:
            return super().fit(support_paths, context)
        if len(support_paths) != 16:
            raise ValueError("FBPA requires exactly16 normal supports")
        source_ids = []
        for path in support_paths:
            digest = hashlib.sha256()
            with Path(path).open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            source_ids.append(digest.hexdigest())
        if len(set(source_ids)) != 16:
            raise ValueError("FBPA requires16 distinct support contents")
        self._fit_extract_calls = 0
        on_cuda = torch.device(self.device).type == "cuda"
        if on_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        self.adapter = None
        images, _, _, _ = self._prepare_fit(support_paths, context)
        self.records.clear()
        self._source_ids, self._fit_seed = source_ids, context.seed
        self._view_extract_calls = 0
        self.episode_manifest = []
        dimension = self.banks[self.resolutions[0]][0].shape[1]
        self.adapter = ResidualAdapter(dimension, self.adapter_rank).to(self.device)
        scorer = _FunctionalScorer(self)
        parameters = dict(scorer.named_parameters())
        self.parameter_count = sum(p.numel() for p in parameters.values())
        optimizer = torch.optim.AdamW(parameters.values(),
                                      lr=float(self.config.get("learning_rate", .001)),
                                      weight_decay=0)
        for step, batch in enumerate(episode_schedule(self.family_balance, self.source_pairing)):
            first_source = batch[0][0]
            excluded = (first_source // 2 * 2, first_source // 2 * 2 + 1)
            assert all(token[0] in excluded for token in batch)
            banks = {r: (bank[0][torch.tensor([s not in excluded for s in bank[2]],
                                             device=self.device)], None,
                         [s for s in bank[2] if s not in excluded])
                     for r, bank in self.banks.items()}
            assert all(bank[0].shape[0] and not set(excluded).intersection(bank[2])
                       for bank in banks.values())
            clean_cache, logs, provenance = {}, [], []
            optimizer.zero_grad(set_to_none=True)
            for token in batch:
                episode, manifest = self._keyed_episode(images, token, excluded, banks, clean_cache)
                sampler = torch.Generator(device=self.device).manual_seed(manifest["rank_seed"])
                loss, metrics = self._episode_loss(scorer, parameters, episode, banks, sampler)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite FBPA episode objective")
                # Same theta for both backwards. Never step or clip between them.
                (loss / 2).backward()
                logs.append(dict(loss=float(loss.detach()), **metrics))
                provenance.append(manifest)
                del episode, loss
            gradient_norm = nn.utils.clip_grad_norm_(parameters.values(), 5., error_if_nonfinite=True)
            optimizer.step()
            row = {key: (logs[0][key] + logs[1][key]) / 2
                   for key in ("loss", "rank_loss", "shift_loss", "contrast_loss")}
            row.update(gradient_norm=float(gradient_norm),
                       source=batch[0][0], outer_source=batch[1][0],
                       synthetic_family=provenance[0]["family"],
                       outer_synthetic_family=provenance[1]["family"],
                       P=logs[0]["P"], outer_P=logs[1]["P"],
                       budget=logs[0]["budget"], outer_budget=logs[1]["budget"],
                       lambda_value=logs[0]["lambda_value"],
                       outer_lambda_value=logs[1]["lambda_value"],
                       boundary_strength=provenance[0]["strength"],
                       outer_boundary_strength=provenance[1]["strength"],
                       boundary_feasible=provenance[0]["feasible"],
                       outer_boundary_feasible=provenance[1]["feasible"],
                       excluded_sources=list(excluded), episode_ids=[list(t) for t in batch],
                       episodes_consumed=2 * (step + 1), unique_views=4 + 2 * len(clean_cache),
                       training_view_extract_calls=self._view_extract_calls,
                       family_balance=self.family_balance, source_pairing=self.source_pairing,
                       cross_environment_meta=False, resolutions=list(self.resolutions))
            self.history.append(row)
            self.log_training(step, **row)
            self.episode_manifest.extend(provenance)
            del clean_cache, banks, sampler
        self.episode_manifest.sort(key=lambda item: (item["source"], item["repeat"], item["family_index"]))
        self.metric.eval()
        self.adapter.eval()
        self._fit_extract_total = self._fit_extract_calls
        self._fit_extract_calls = None
        self._peak_cuda_fit_bytes = torch.cuda.max_memory_allocated(self.device) if on_cuda else None

    def object_metadata(self):
        metadata = super().object_metadata()
        metadata["episode_enabled"] = self.episode_enabled
        if not self.episode_enabled:
            return metadata
        metadata.update(family_balance=self.family_balance, source_pairing=self.source_pairing,
                        episode_seed=self.episode_seed, episode_repeats=4,
                        episode_sources=1 if self.source_pairing else 2,
                        training_bank_exclusion="fixed_pair_excluded",
                        gradient_replay="two_sequential_first_order_backwards_one_adam_step",
                        inner_optimizer=None, meta_gain="per_episode_immediate_feedback",
                        episode_count=len(self.episode_manifest),
                        training_view_extract_calls=self._view_extract_calls,
                        feature_extract_calls_fit=self._fit_extract_total,
                        peak_cuda_fit_allocated_bytes=self._peak_cuda_fit_bytes,
                        episode_manifest=self.episode_manifest,
                        episode_manifest_sha256=hashlib.sha256(json.dumps(
                            self.episode_manifest, sort_keys=True).encode()).hexdigest())
        return metadata
