"""Bounded contextual diagonal metric; frozen features, exact full-bank NN.

Selection is detached, then the selected quadratic distance is differentiated
(the exact gradient of a minimum almost everywhere). No query/bank/D tensor.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fmad.methods.contextual_score import ContextualScore, flip_tokens
from fmad.methods.PSMA_MR import _read_rgb, _synthetic_anomalies, _apply_photometric, _sample_photometric_transform
from fmad.registry import METHOD_REGISTRY


class BoundedMetric(nn.Module):
    def __init__(self, dimension, contextual, bound=0.2, hidden=32):
        super().__init__()
        self.bound = bound
        self.global_weight = nn.Parameter(torch.zeros(dimension))
        self.encoder = nn.Linear(dimension, hidden) if contextual else None
        self.readout = nn.Linear(hidden, dimension) if contextual else None
        if contextual:
            nn.init.zeros_(self.readout.weight)
            nn.init.zeros_(self.readout.bias)

    def context(self, features, grid):
        image = features.T.reshape(1, -1, *grid)
        total = F.avg_pool2d(image, 3, stride=1, padding=1, count_include_pad=True) * 9 - image
        count = F.avg_pool2d(torch.ones_like(image[:, :1]), 3, stride=1,
                             padding=1, count_include_pad=True) * 9 - 1
        ring = (total / count.clamp_min(1)).flatten(2)[0].T
        return F.gelu(self.encoder(ring))

    def forward(self, features, grid):
        logits = self.global_weight.expand_as(features)
        if self.encoder is not None:
            logits = logits + self.readout(self.context(features, grid))
        return self.bound * torch.tanh(logits)


@METHOD_REGISTRY.register("bcma")
class BCMA(ContextualScore):
    def __init__(self, backbone, config):
        super().__init__(backbone, {**config, "score_head": False,
                                    "context_matching": False, "correspondence_training": False})
        names = ("metric_adaptation", "context_conditioning", "correspondence_regularization")
        self.switches = {key: config.get(key, False) for key in names}
        if any(type(v) is not bool for v in self.switches.values()):
            raise ValueError("BCMA switches must be booleans")
        if self.switches[names[1]] and not self.switches[names[0]]:
            raise ValueError("context requires adaptation")
        if self.switches[names[2]] and not self.switches[names[1]]:
            raise ValueError("correspondence requires context")
        self.bound = float(config.get("metric_bound", 0.2))
        if not 0 < self.bound < 1:
            raise ValueError("metric_bound must be strictly between zero and one")
        self.metric = None
        self.config = config

    def score(self, features, grid, bank, indices=None, baseline=False):
        values = bank[0]
        correction = self.metric(features, grid) if self.metric is not None and not baseline else None
        if indices is not None:
            features = features[indices]
            correction = correction[indices] if correction is not None else None
        output = []
        for start in range(0, len(features), self.chunk):
            query = features[start:start + self.chunk]
            if correction is None:
                output.append((1 - query @ values.T).amin(1))
                continue
            change = correction[start:start + self.chunk]
            with torch.no_grad():
                q, c = query.detach(), change.detach()
                distance = 1 - q @ values.T + 0.5 * (
                    (c * q.square()).sum(1, keepdim=True) + c @ values.square().T
                    - 2 * (c * q) @ values.T)
                selected = distance.argmin(1)
            reference = values[selected]
            distance = 1 - (query * reference).sum(1)
            distance = distance + 0.5 * (change * (query - reference).square()).sum(1)
            output.append(distance)
        return torch.cat(output)

    def fused(self, branches, banks, shape, baseline=False):
        result = None
        for resolution, (features, grid) in branches.items():
            score = self.score(features, grid, banks[resolution], baseline=baseline)
            native = F.interpolate(score.reshape(1, 1, *grid), size=shape,
                                   mode="bilinear", align_corners=False)[0, 0]
            result = native if result is None else result + native
        return result / len(branches)

    def _prepare_fit(self, support_paths, context):
        """Initialize the common frozen support bank and deterministic samplers."""
        if context.masking or context.rotation:
            raise ValueError("BCMA requires no_mask_no_rotation")
        if len(support_paths) < 2 or len(set(support_paths)) != len(support_paths):
            raise ValueError("need at least two unique supports")
        torch.manual_seed(context.seed)
        rng = np.random.default_rng(context.seed)
        augmentation_rng = np.random.default_rng(context.seed + 100003)
        sampler = torch.Generator(device=self.device).manual_seed(context.seed + 19)
        images = [_read_rgb(path) for path in support_paths]
        self.support_count = len(images)
        self.records, self.banks, self.history = {}, {}, []
        self.metric = None
        for resolution in self.resolutions:
            self.records[resolution] = []
            for source, image in enumerate(images):
                features, grid = self.extract(image, resolution)
                indices = torch.as_tensor(rng.choice(len(features), min(self.cap, len(features)),
                                                      replace=False), device=self.device)
                self.records[resolution].append((source, features.cpu(), grid, indices))
        self.banks = {r: self.bank(r) for r in self.resolutions}
        if self.switches["metric_adaptation"]:
            dimension = self.banks[self.resolutions[0]][0].shape[1]
            self.metric = BoundedMetric(dimension, self.switches["context_conditioning"],
                                        self.bound, int(self.config.get("hidden_dim", 32))).to(self.device)
        self.parameter_count = sum(p.numel() for p in self.metric.parameters()) if self.metric else 0
        return images, rng, augmentation_rng, sampler

    def fit(self, support_paths, context):
        images, rng, augmentation_rng, sampler = self._prepare_fit(support_paths, context)
        if self.metric is not None:
            optimizer = torch.optim.AdamW(self.metric.parameters(), lr=float(self.config.get("learning_rate", .001)),
                                          weight_decay=0)
            for step in range(self.steps):
                source = step % len(images)
                family_index = (step // len(images)) % 2
                image = images[source]
                anomaly, mask, family = _synthetic_anomalies(image, rng, [])[family_index]
                transform = _sample_photometric_transform(rng, 1.0)
                banks = {r: (b[0][torch.tensor(b[2], device=self.device) != source], None,
                              [s for s in b[2] if s != source]) for r, b in self.banks.items()}
                assert all(source not in bank[2] for bank in banks.values())
                optimizer.zero_grad(set_to_none=True)
                rows = {"rank_loss": 0., "keep_loss": 0., "shift_loss": 0., "correspondence_loss": 0.}
                normal_scores = []
                baseline_clean = None
                for view_index, view in enumerate((image, _apply_photometric(image, transform),
                                                   anomaly, _apply_photometric(anomaly, transform))):
                    branches = {r: self.extract(view, r) for r in self.resolutions}
                    scores = self.fused(branches, banks, image.shape[:2])
                    if view_index == 0:
                        with torch.no_grad():
                            baseline_clean = self.fused(branches, banks, image.shape[:2], baseline=True)
                        term = (scores - baseline_clean).square().mean()
                        rows["keep_loss"] = float(term.detach())
                        (float(self.config.get("keep_weight", 1.0)) * term).backward()
                    elif view_index == 1:
                        term = (scores - baseline_clean).square().mean()
                        rows["shift_loss"] = float(term.detach())
                        (float(self.config.get("shift_weight", 1.0)) * term).backward()
                    if view_index < 2:
                        normal_scores.append(scores.detach().flatten().topk(min(self.train_tokens, scores.numel())).values)
                    else:
                        foreground = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
                        positive = scores[foreground]
                        selected = torch.randperm(len(positive), device=self.device, generator=sampler)[:self.train_tokens]
                        positive = positive[selected]
                        negative = scores[~foreground]
                        if len(negative):
                            random_ids = torch.randperm(len(negative), device=self.device, generator=sampler)[:self.train_tokens]
                            negative = torch.cat((negative[random_ids], negative.topk(min(self.train_tokens, len(negative))).values))
                        negative = torch.cat((negative, *normal_scores))
                        term = F.relu(float(self.config.get("margin", .05)) - positive[:, None] + negative[None]).mean()
                        rows["rank_loss"] += float(term.detach()) / 2
                        (term / 2).backward()
                    del scores, branches
                if self.switches["correspondence_regularization"]:
                    resolution = self.resolutions[step % len(self.resolutions)]
                    views = []
                    for _ in range(2):
                        flips = tuple(bool(v) for v in augmentation_rng.integers(0, 2, 2))
                        view = _apply_photometric(image, _sample_photometric_transform(augmentation_rng, 1.))
                        features, grid = self.extract(view, resolution, flips)
                        latent = F.normalize(self.metric.context(features, grid), dim=-1)
                        views.append(flip_tokens(latent, grid, *flips))
                    ids = torch.linspace(0, len(views[0]) - 1, min(self.train_tokens, len(views[0])), device=self.device).long()
                    term = (F.cross_entropy(views[0][ids] @ views[1].T / .1, ids)
                            + F.cross_entropy(views[1][ids] @ views[0].T / .1, ids)) / 2
                    rows["correspondence_loss"] = float(term.detach())
                    (float(self.config.get("correspondence_weight", .01)) * term).backward()
                nn.utils.clip_grad_norm_(self.metric.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                if not all(np.isfinite(v) for v in rows.values()):
                    raise RuntimeError("nonfinite BCMA loss")
                rows.update(source=source, synthetic_family=family, resolutions=list(self.resolutions))
                self.history.append(rows)
                self.log_training(step, **rows)
            self.metric.eval()
        self.records.clear()

    def object_metadata(self):
        return dict(ablation=self.switches, metric_bound=self.bound, trainable_parameters=self.parameter_count,
                    backbone_frozen=True, support_count=self.support_count, source_excluded_score_training=True,
                    steps_completed=len(self.history), resolutions=list(self.resolutions),
                    bank_sizes={str(r): len(b[0]) for r, b in self.banks.items()},
                    score_definition="mean_resolution_bounded_diagonal_nn", joint_resolution_loss=True,
                    last_training=self.history[-1] if self.history else None)
