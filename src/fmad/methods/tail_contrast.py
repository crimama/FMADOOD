"""Tail-focused contrast adaptation of frozen support features.

D0 delegates exactly to BCMA B2. D1 enables bidirectional normal-tail ranking;
D2 replaces baseline score imitation with paired normal/defect consistency;
D3 adds a shared low-rank residual transform to both queries and references.

Four detached score maps are temporary loss variables, not saved artifacts.
Their exact vector-Jacobian products are replayed one view at a time, so hard
normal scores receive gradients without retaining four model graphs. Frozen
features are cached on CPU for this step only. No query data enter fitting.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fmad.methods.bcma import BCMA
from fmad.methods.PSMA_MR import (
    _apply_photometric, _sample_photometric_transform, _synthetic_anomalies,
)
from fmad.registry import METHOD_REGISTRY


class ResidualAdapter(nn.Module):
    def __init__(self, dimension, rank):
        super().__init__()
        self.down = nn.Linear(dimension, rank, bias=False)
        self.up = nn.Linear(rank, dimension, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, features):
        return F.normalize(features + self.up(self.down(features)), dim=-1)


def tail_samples(scores, fraction, count, generator):
    """Mine indices without gradients; keep gradients of selected values."""
    flat = scores.flatten()
    if not flat.numel():
        return flat
    k = max(1, math.ceil(flat.numel() * fraction))
    tail_ids = flat.detach().topk(k).indices
    selection = torch.randperm(k, device=flat.device, generator=generator)[:count]
    random_ids = torch.randperm(flat.numel(), device=flat.device, generator=generator)[:count]
    return torch.cat((flat[tail_ids[selection]], flat[random_ids]))


def paired_losses(scores, mask):
    """Same transform for clean/defect pairs; no baseline target or stop-grad."""
    clean, shifted, anomaly, shifted_anomaly = scores
    normal = (shifted - clean).square().mean()
    delta = anomaly[mask] - clean[mask]
    shifted_delta = shifted_anomaly[mask] - shifted[mask]
    contrast = (shifted_delta - delta).square().mean()
    return normal, contrast


def tail_ranking_pairs(scores, mask, fraction, count, generator):
    """Yield the historical sampled pairs, preserving RNG order and gradients."""
    clean_negatives = torch.cat([
        tail_samples(view, fraction, count, generator) for view in scores[:2]
    ])
    for view in scores[2:]:
        positive = view[mask]
        ids = torch.randperm(len(positive), device=view.device, generator=generator)[:count]
        positive = positive[ids]
        negative = torch.cat((clean_negatives,
                              tail_samples(view[~mask], fraction, count, generator)))
        yield positive, negative


def tail_ranking(scores, mask, fraction, count, margin, generator):
    terms = [F.relu(margin - positive[:, None] + negative[None]).mean()
             for positive, negative in tail_ranking_pairs(
                 scores, mask, fraction, count, generator)]
    return torch.stack(terms).mean()


@METHOD_REGISTRY.register("tail_contrast")
class TailContrast(BCMA):
    def __init__(self, backbone, config):
        fixed = dict(metric_adaptation=True, context_conditioning=True,
                     correspondence_regularization=False)
        for name, value in fixed.items():
            if name in config and config[name] is not value:
                raise ValueError(f"tail_contrast requires {name}={value}")
        super().__init__(backbone, {**config, **fixed})
        names = ("hard_normal_ranking", "paired_contrast", "low_rank_adapter")
        self.components = {name: config.get(name, False) for name in names}
        if any(type(value) is not bool for value in self.components.values()):
            raise ValueError("TailContrast switches must be booleans")
        if self.components[names[1]] and not self.components[names[0]]:
            raise ValueError("paired_contrast requires hard_normal_ranking")
        if self.components[names[2]] and not self.components[names[1]]:
            raise ValueError("low_rank_adapter requires paired_contrast")
        self.tail_fraction = float(config.get("normal_tail_fraction", .05))
        self.adapter_rank = int(config.get("adapter_rank", 16))
        if not 0 < self.tail_fraction <= 1 or self.adapter_rank < 1:
            raise ValueError("invalid tail fraction or adapter rank")
        self.adapter = None

    def score(self, features, grid, bank, indices=None, baseline=False):
        if self.adapter is not None and not baseline:
            features = self.adapter(features)
            bank = (self.adapter(bank[0]), bank[1], bank[2])
        return super().score(features, grid, bank, indices, baseline)

    def _branches_to_device(self, branches):
        return {r: (features.to(self.device), grid)
                for r, (features, grid) in branches.items()}

    def extra_parameters(self):
        return []

    def training_anomaly(self, image, rng, family_index, step):
        return _synthetic_anomalies(image, rng, [])[family_index]

    def ranking_loss(self, scores, mask, sampler):
        return tail_ranking(scores, mask, self.tail_fraction, self.train_tokens,
                            float(self.config.get("margin", .05)), sampler)

    def training_diagnostics(self):
        return {}

    def training_step_context(self, scores, baseline_clean, baseline_shift):
        """Optional detached context hook; historical methods are no-ops."""
        return None

    def compose_training_loss(self, scores, mask, baseline_clean, rank, keep,
                              shift, contrast):
        """Return the objective and optional diagnostics for one replay step.

        The default path is deliberately the historical objective.  Methods
        may extend it without duplicating the memory-safe four-view replay.
        """
        if self.components["paired_contrast"]:
            total = (rank + float(self.config.get("shift_weight", 1.)) * shift
                     + float(self.config.get("contrast_weight", 1.)) * contrast)
        else:
            total = (rank + float(self.config.get("keep_weight", 1.)) * keep
                     + float(self.config.get("shift_weight", 1.)) * shift)
        return total, {}

    def training_gradients(self, total, score_leaves):
        """Score-leaf gradients used by the exact replay implementation."""
        return torch.autograd.grad(total, score_leaves)

    def training_update(self, optimizer, parameters, clean_branches, banks, shape,
                        baseline_clean):
        """Apply one update after replay/clipping; default is unchanged AdamW.

        Methods may reuse this step's frozen clean features for an update
        constraint without changing the loss, sampler or backbone extraction.
        """
        optimizer.step()
        return {}

    def fit(self, support_paths, context):
        self.adapter = None
        if not self.components["hard_normal_ranking"]:
            return super().fit(support_paths, context)
        images, rng, _, sampler = self._prepare_fit(support_paths, context)
        self.records.clear()
        parameters = list(self.metric.parameters())
        if self.components["low_rank_adapter"]:
            dimension = self.banks[self.resolutions[0]][0].shape[1]
            self.adapter = ResidualAdapter(dimension, self.adapter_rank).to(self.device)
            parameters += list(self.adapter.parameters())
        parameters += list(self.extra_parameters())
        self.parameter_count = sum(p.numel() for p in parameters)
        optimizer = torch.optim.AdamW(parameters, lr=float(self.config.get("learning_rate", .001)),
                                      weight_decay=0)
        for step in range(self.steps):
            source = step % len(images)
            family_index = (step // len(images)) % 2
            image = images[source]
            anomaly, mask, family = self.training_anomaly(image, rng, family_index, step)
            transform = _sample_photometric_transform(rng, 1.)
            banks = {r: (b[0][torch.tensor(b[2], device=self.device) != source], None,
                          [owner for owner in b[2] if owner != source])
                     for r, b in self.banks.items()}
            assert all(source not in b[2] for b in banks.values())
            cached, score_leaves = [], []
            baseline_clean = None
            baseline_shift = None
            # No graph or persistent map cache: one step of frozen CPU features.
            for view_index, view in enumerate((image, _apply_photometric(image, transform),
                                               anomaly, _apply_photometric(anomaly, transform))):
                with torch.no_grad():
                    branches = {r: self.extract(view, r) for r in self.resolutions}
                    values = self.fused(branches, banks, image.shape[:2])
                    if view_index == 0:
                        baseline_clean = self.fused(branches, banks, image.shape[:2], baseline=True)
                    elif view_index == 1:
                        baseline_shift = self.fused(branches, banks, image.shape[:2], baseline=True)
                    cached.append({r: (features.cpu(), grid)
                                   for r, (features, grid) in branches.items()})
                score_leaves.append(values.detach().requires_grad_(True))
                del branches, values
            foreground = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
            self.training_step_context(score_leaves, baseline_clean, baseline_shift)
            rank = self.ranking_loss(score_leaves, foreground, sampler)
            keep = (score_leaves[0] - baseline_clean).square().mean()
            if self.components["paired_contrast"]:
                shift, contrast = paired_losses(score_leaves, foreground)
            else:
                shift = (score_leaves[1] - baseline_clean).square().mean()
                contrast = rank.new_zeros(())
            total, extra_diagnostics = self.compose_training_loss(
                score_leaves, foreground, baseline_clean, rank, keep, shift, contrast)
            if not torch.isfinite(total):
                raise RuntimeError("nonfinite TailContrast loss")
            gradients = self.training_gradients(total, score_leaves)
            optimizer.zero_grad(set_to_none=True)
            # Exact chain rule; no optimizer step until all four VJPs accumulate.
            for branches, gradient in zip(cached, gradients):
                prediction = self.fused(self._branches_to_device(branches), banks, image.shape[:2])
                prediction.backward(gradient)
                del prediction
            grad_norm = nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
            row = dict(loss=float(total.detach()), rank_loss=float(rank.detach()),
                       keep_loss=float(keep.detach()), keep_loss_optimized=not self.components["paired_contrast"],
                       shift_loss=float(shift.detach()), contrast_loss=float(contrast.detach()),
                       normal_score_gradient_l1=float(sum(g.abs().sum() for g in gradients[:2])),
                       gradient_norm=float(grad_norm), source=source, synthetic_family=family,
                       resolutions=list(self.resolutions), normal_tail_fraction=self.tail_fraction,
                       normal_score_mean=float(score_leaves[0].detach().mean()),
                       anomaly_score_mean=float(score_leaves[2].detach()[foreground].mean()))
            row.update(self.training_diagnostics())
            row.update(extra_diagnostics)
            row.update(self.training_update(optimizer, parameters, cached[0], banks,
                                            image.shape[:2], baseline_clean))
            self.history.append(row)
            self.log_training(step, **row)
            del cached, score_leaves, gradients, banks
        self.metric.eval()
        if self.adapter is not None:
            self.adapter.eval()

    def object_metadata(self):
        metadata = super().object_metadata()
        metadata.update(components=self.components,
                        baseline_equivalence="BCMA_B2_when_all_components_off",
                        adapter_parameters=sum(p.numel() for p in self.adapter.parameters()) if self.adapter else 0,
                        adapter_rank=self.adapter_rank if self.adapter else 0,
                        normal_tail_fraction=self.tail_fraction,
                        gradient_replay="four_view_exact_vjp" if self.components["hard_normal_ranking"] else None,
                        score_definition="mean_resolution_bounded_diagonal_nn" if self.adapter is None
                        else "mean_resolution_low_rank_then_bounded_diagonal_nn")
        return metadata
