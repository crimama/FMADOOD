"""Config-driven A0/A1/A2/A3, frozen-backbone local correspondence scoring.

Only selected normal supports reach fit. Synthetic masks are training labels;
query labels never enter this module. Banks retain bounded support tokens only.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fmad.methods.base import BaseMethod
from fmad.methods.PSMA_MR import (
    _apply_photometric, _feature_matrix, _read_rgb,
    _sample_photometric_transform, _synthetic_anomalies,
)
from fmad.registry import METHOD_REGISTRY


def flip_tokens(tokens, grid, horizontal, vertical):
    value = tokens.reshape(*grid, -1)
    axes = ([0] if vertical else []) + ([1] if horizontal else [])
    return value.flip(axes).reshape_as(tokens) if axes else tokens


class ContextKey(nn.Module):
    def __init__(self, dimension, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(dimension, hidden, 3, padding=1),
                                 nn.GELU(), nn.Conv2d(hidden, hidden, 1))

    def forward(self, tokens, grid):
        image = tokens.T.reshape(1, -1, *grid)
        return F.normalize(self.net(image).flatten(2).squeeze(0).T, dim=-1)


@METHOD_REGISTRY.register("contextual_score")
class ContextualScore(BaseMethod):
    def __init__(self, backbone, config):
        super().__init__(backbone, config)
        self.device = torch.device(backbone.device)
        self.flags = {name: config.get(name, default) for name, default in
                      (("score_head", True), ("context_matching", False),
                       ("correspondence_training", False))}
        if any(type(value) is not bool for value in self.flags.values()):
            raise ValueError("ablation switches must be YAML booleans")
        if self.flags["context_matching"] and not self.flags["score_head"]:
            raise ValueError("context_matching requires score_head")
        if self.flags["correspondence_training"] and not self.flags["context_matching"]:
            raise ValueError("correspondence_training requires context_matching")
        self.resolutions = tuple(config.get("resolutions", [448, 672]))
        self.steps = int(config.get("steps", 64))
        self.cap = int(config.get("bank_tokens_per_source", 256))
        self.train_tokens = int(config.get("train_tokens", 128))
        self.topk = int(config.get("candidate_count", 8))
        self.chunk = int(config.get("query_chunk_size", 128))
        if min(self.steps, self.cap, self.train_tokens, self.topk, self.chunk,
               *self.resolutions) <= 0:
            raise ValueError("steps, token budgets and resolutions must be positive")
        self.head = self.key = None
        self.records = {}
        self.banks = {}
        self.history = []

    def extract(self, image, resolution, flips=(False, False)):
        tensor, grid = self.backbone.prepare_image(image, resolution=resolution)
        # Flip after common resize/crop: token correspondences stay exact even
        # when the native image dimensions are not multiples of patch size.
        axes = ([-1] if flips[0] else []) + ([-2] if flips[1] else [])
        if axes:
            tensor = tensor.flip(axes)
        with torch.no_grad():
            outputs = self.backbone.extract_features(tensor)
        features, grid = _feature_matrix(outputs[-1], grid)
        return F.normalize(features.to(self.device), dim=-1), grid

    def keys(self, features, grid):
        return self.key(features, grid) if self.key is not None else features

    def bank(self, resolution, excluded=None):
        values, keys, owners = [], [], []
        for source, features, grid, indices in self.records[resolution]:
            if source == excluded:
                continue
            features = features.to(self.device)
            values.append(features[indices])
            keys.append(self.keys(features, grid)[indices])
            owners.extend([source] * len(indices))
        if not values:
            raise ValueError("source-excluded fitting requires at least two supports")
        return torch.cat(values), torch.cat(keys), owners

    def score(self, features, grid, bank, indices=None):
        values, bank_keys, _ = bank
        query_keys = self.keys(features, grid)
        if indices is not None:
            features, query_keys = features[indices], query_keys[indices]
        scores = []
        for start in range(0, len(features), self.chunk):
            query = features[start:start + self.chunk]
            keys = query_keys[start:start + self.chunk]
            if self.head is None:
                scores.append((1 - query @ values.T).amin(dim=1))
                continue
            similarities = keys @ bank_keys.T
            selected = similarities.topk(min(self.topk, len(values)), dim=1).indices
            references = values[selected]
            delta = query[:, None] - references
            distances = 1 - (query[:, None] * references).sum(-1, keepdim=True)
            # Both appearance and correspondence enter the deployed score.
            correspondence = similarities.gather(1, selected).unsqueeze(-1)
            inputs = torch.cat((delta, delta.abs(), distances, correspondence), dim=-1)
            logits = self.head(inputs).squeeze(-1)
            weights = F.softmax(correspondence.squeeze(-1) / 0.1, dim=1)
            scores.append((weights * logits).sum(dim=1))
        return torch.cat(scores)

    def correspondence_loss(self, first, first_grid, second, second_grid, qflip, rflip):
        if first_grid != second_grid:
            raise ValueError("flip correspondence requires matching grids")
        left = flip_tokens(self.keys(first, first_grid), first_grid, *qflip)
        right = flip_tokens(self.keys(second, second_grid), second_grid, *rflip)
        count = min(self.train_tokens, len(left))
        selected = torch.linspace(0, len(left) - 1, count, device=self.device).long()
        logits = left[selected] @ right.T / 0.1
        reverse = right[selected] @ left.T / 0.1
        return (F.cross_entropy(logits, selected) + F.cross_entropy(reverse, selected)) / 2

    def fit(self, support_paths, context):
        if context.masking or context.rotation:
            raise ValueError("use explicit no_mask_no_rotation common preprocessing")
        if len(support_paths) < 2 or len(set(support_paths)) != len(support_paths):
            raise ValueError("at least two unique support paths required")
        self.records, self.banks, self.history = {}, {}, []
        self.context = context
        rng = np.random.default_rng(context.seed)
        torch.manual_seed(context.seed)
        sampling_rng = torch.Generator(device=self.device).manual_seed(context.seed + 19)
        images = [_read_rgb(path) for path in support_paths]
        self.support_count = len(images)
        for resolution in self.resolutions:
            records = []
            for source, image in enumerate(images):
                features, grid = self.extract(image, resolution)
                indices = torch.from_numpy(rng.choice(len(features), min(self.cap, len(features)),
                                                      replace=False)).long().to(self.device)
                records.append((source, features.cpu(), grid, indices))
            self.records[resolution] = records
        dimension = self.records[self.resolutions[0]][0][1].shape[-1]
        hidden = int(self.config.get("hidden_dim", 64))
        self.head = (nn.Sequential(nn.Linear(2 * dimension + 2, hidden), nn.GELU(),
                                   nn.Linear(hidden, 1)).to(self.device)
                     if self.flags["score_head"] else None)
        self.key = (ContextKey(dimension, hidden).to(self.device)
                    if self.flags["context_matching"] else None)
        parameters = [p for module in (self.head, self.key) if module is not None
                      for p in module.parameters()]
        self.parameter_count = sum(p.numel() for p in parameters)
        if parameters:
            optimizer = torch.optim.AdamW(parameters, lr=float(self.config.get("learning_rate", 0.001)),
                                          weight_decay=0.0)
            # Separate generators prevent A3 augmentation draws changing score training data.
            aug_rng = np.random.default_rng(context.seed + 100003)
            for step in range(self.steps):
                source = (step // len(self.resolutions)) % len(images)
                resolution = self.resolutions[step % len(self.resolutions)]
                image = images[source]
                anomaly, mask, family = _synthetic_anomalies(image, rng, [])[step % 2]
                transform = _sample_photometric_transform(rng, 1.0)
                clean, grid = self.extract(image, resolution)
                optimizer.zero_grad(set_to_none=True)
                bank = self.bank(resolution, excluded=source)
                assert source not in bank[2]
                losses = []
                for view, label in ((image, None), (_apply_photometric(image, transform), None),
                                    (anomaly, mask), (_apply_photometric(anomaly, transform), mask)):
                    features, view_grid = (clean, grid) if view is image else self.extract(view, resolution)
                    target = torch.zeros(len(features), device=self.device)
                    if label is not None:
                        target = F.adaptive_max_pool2d(
                            torch.as_tensor(label, device=self.device, dtype=torch.float32)[None, None],
                            view_grid).flatten().clamp(0, 1)
                    positive = torch.where(target > 0)[0]
                    negative = torch.where(target == 0)[0]
                    def sample(pool, count):
                        return pool[torch.randperm(len(pool), device=self.device,
                                                  generator=sampling_rng)[:count]]
                    selected = torch.cat((sample(positive, self.train_tokens // 2),
                                          sample(negative, self.train_tokens // 2)))
                    logits = self.score(features, view_grid, bank, selected)
                    losses.append(F.binary_cross_entropy_with_logits(logits, target[selected]))
                score_loss = torch.stack(losses).mean()
                correspondence = torch.zeros((), device=self.device)
                if self.flags["correspondence_training"]:
                    views, flips = [], []
                    for _ in range(2):
                        flags = tuple(bool(v) for v in aug_rng.integers(0, 2, size=2))
                        view = _apply_photometric(image, _sample_photometric_transform(aug_rng, 1.0))
                        views.append(self.extract(view, resolution, flags))
                        flips.append(flags)
                    correspondence = self.correspondence_loss(*views[0], *views[1], *flips)
                loss = score_loss + float(self.config.get("correspondence_weight", 0.1)) * correspondence
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, 5.0, error_if_nonfinite=True)
                optimizer.step()
                row = dict(loss=float(loss.detach()), score_loss=float(score_loss.detach()),
                           correspondence_loss=float(correspondence.detach()), source=source,
                           resolution=resolution, synthetic_family=family)
                self.history.append(row)
                self.log_training(step, **row)
        with torch.no_grad():
            self.banks = {resolution: self.bank(resolution) for resolution in self.resolutions}
        # Full support grids are only transient during fit; deployment retains capped banks.
        self.records.clear()
        for module in (self.head, self.key):
            if module is not None:
                module.eval()

    def predict(self, test_image_path):
        image = _read_rgb(test_image_path)
        total = None
        with torch.no_grad():
            for resolution in self.resolutions:
                features, grid = self.extract(image, resolution)
                score = self.score(features, grid, self.banks[resolution])
                native = F.interpolate(score.reshape(1, 1, *grid), size=image.shape[:2],
                                       mode="bilinear", align_corners=False)[0, 0]
                total = native if total is None else total + native
            result = (total / len(self.resolutions)).cpu().numpy()
        if not np.isfinite(result).all():
            raise RuntimeError("nonfinite prediction")
        return result, float(np.max(result))

    def object_metadata(self):
        return dict(ablation=self.flags, trainable_parameters=self.parameter_count,
                    backbone_frozen=True, support_count=self.support_count,
                    source_excluded_score_training=True, correspondence_normal_only=True,
                    steps_completed=len(self.history), resolutions=list(self.resolutions),
                    bank_sizes={str(r): len(b[0]) for r, b in self.banks.items()},
                    score_definition="mean_resolution_logits" if self.head is not None else "mean_cosine_nn",
                    last_training=self.history[-1] if self.history else None)
