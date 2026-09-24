"""L3: exact one-step cross-environment meta-update of the K1 scorer.

L3C sees the same two sources, synthetic families, transforms and excluded
reference bank, but optimizes ordinary pooled ERM. A virtual SGD step is
differentiated exactly; the real update is AdamW. Frozen features/maps live
for one episode only. No target stream or evaluation labels enter fitting.
"""
import math

import torch
from torch import nn
from torch.func import functional_call

from fmad.methods.boundary_synthetic import BoundarySynthetic, boundary_choice
from fmad.methods.PSMA_MR import _apply_photometric, _sample_photometric_transform
from fmad.methods.relational_tail import relational_score_loss
from fmad.methods.tail_contrast import ResidualAdapter, paired_losses, tail_ranking
from fmad.registry import METHOD_REGISTRY


def virtual_sgd(parameters, loss, learning_rate, *, differentiable=True):
    """Functional SGD, without mutating parameters, gradients or Adam state."""
    gradients = torch.autograd.grad(loss, tuple(parameters.values()),
                                    create_graph=differentiable, retain_graph=True)
    return {name: value - learning_rate * gradient
            for (name, value), gradient in zip(parameters.items(), gradients)}, gradients


class _FunctionalScorer(nn.Module):
    """Expose the existing scorer's modules to stateless functional_call.

    The owner is not an nn.Module. Its adapter/metric are the SAME registered
    objects below, so functional_call temporarily substitutes their parameters
    during the ordinary scoring path and restores them after every call.
    """
    def __init__(self, owner):
        super().__init__()
        self.metric = owner.metric
        self.adapter = owner.adapter
        self.owner = owner

    def forward(self, branches, banks, shape):
        return self.owner.fused(branches, banks, shape)


@METHOD_REGISTRY.register("transfer_update")
class TransferUpdate(BoundarySynthetic):
    def __init__(self, backbone, config):
        super().__init__(backbone, config)
        self.transfer_enabled = config.get("transfer_update", True)
        self.meta_enabled = config.get("cross_environment_meta", True)
        if type(self.transfer_enabled) is not bool or type(self.meta_enabled) is not bool:
            raise ValueError("transfer switches must be booleans")
        self.inner_lr = float(config.get("meta_inner_lr", .01))
        self.outer_weight = float(config.get("meta_outer_weight", 1.))
        if not all(math.isfinite(v) and v > 0 for v in (self.inner_lr, self.outer_weight)):
            raise ValueError("meta learning rate and outer weight must be positive and finite")

    def _episode_anomaly(self, image, rng, family_index, source, excluded, banks):
        # Bypass only K1's single-source bank construction, not its generator.
        anomaly, mask, family = super(BoundarySynthetic, self).training_anomaly(
            image, rng, family_index, source)
        if not self.enabled:
            return anomaly, mask, family, dict(strength=1., feasible=-1)
        key = (source, tuple(excluded))
        if key not in self._normal_thresholds:
            self._normal_thresholds[key] = self._teacher_quantile(
                image, banks, 1. - self.tail_fraction)
        target = self._normal_thresholds[key] + self.boundary_margin
        positive = [self._teacher_quantile(self._blend(image, anomaly, strength),
                                          banks, .25, mask)
                    for strength in self.boundary_strengths]
        selected, feasible = boundary_choice(positive, target)
        strength = self.boundary_strengths[selected]
        return self._blend(image, anomaly, strength), mask, family, dict(
            strength=strength, feasible=int(feasible), teacher_target=target,
            teacher_positive=positive[selected])

    def _prepare_episode(self, image, rng, family_index, source, excluded, banks):
        anomaly, mask, family, diagnostics = self._episode_anomaly(
            image, rng, family_index, source, excluded, banks)
        transform = _sample_photometric_transform(rng, 1.)
        views = (image, _apply_photometric(image, transform),
                 anomaly, _apply_photometric(anomaly, transform))
        cached, teachers = [], []
        for index, view in enumerate(views):
            with torch.no_grad():
                branches = {r: self.extract(view, r) for r in self.resolutions}
                if index < 2:
                    teachers.append(self.fused(branches, banks, image.shape[:2],
                                               baseline=True).cpu())
                cached.append({r: (features.cpu(), grid)
                               for r, (features, grid) in branches.items()})
            del branches
        clean, shifted = teachers
        susceptibility = float((shifted - clean).abs().mean() /
                               clean.abs().mean().clamp_min(1e-6))
        return dict(cached=cached, teacher=clean, mask=mask, shape=image.shape[:2],
                    budget=self.budget_base / (1. + self.susceptibility_scale * susceptibility),
                    source=source, family=family, diagnostics=diagnostics)

    def _episode_loss(self, scorer, parameters, episode, banks, sampler, gain=None):
        scores = [functional_call(scorer, parameters,
                                 (self._branches_to_device(branches), banks, episode["shape"]),
                                 strict=True)
                  for branches in episode["cached"]]
        mask = torch.as_tensor(episode["mask"], device=self.device, dtype=torch.bool)
        teacher = episode["teacher"].to(self.device)
        rank = tail_ranking(scores, mask, self.tail_fraction, self.train_tokens,
                            float(self.config.get("margin", .05)), sampler)
        shift, contrast = paired_losses(scores, mask)
        preserve = (float(self.config.get("keep_weight", 1.)) *
                    (scores[0] - teacher).square().mean() +
                    self.relational_beta * relational_score_loss(scores[0], teacher))
        if gain is None:
            gain = max(self._lambda_initial, min(self.lambda_max,
                       float(preserve.detach()) / episode["budget"]))
        loss = (rank + float(self.config.get("shift_weight", 1.)) * shift +
                float(self.config.get("contrast_weight", 1.)) * contrast + gain * preserve)
        return loss, dict(rank_loss=float(rank.detach()), shift_loss=float(shift.detach()),
                          contrast_loss=float(contrast.detach()), P=float(preserve.detach()),
                          budget=episode["budget"], lambda_value=gain)

    def fit(self, support_paths, context):
        if not self.transfer_enabled:
            return super().fit(support_paths, context)
        if len(support_paths) < 3:
            raise ValueError("cross-environment training needs at least three unique supports")
        self.adapter = None
        images, rng, _, sampler = self._prepare_fit(support_paths, context)
        self.records.clear()
        dimension = self.banks[self.resolutions[0]][0].shape[1]
        self.adapter = ResidualAdapter(dimension, self.adapter_rank).to(self.device)
        scorer = _FunctionalScorer(self)
        parameters = dict(scorer.named_parameters())
        self.parameter_count = sum(p.numel() for p in parameters.values())
        optimizer = torch.optim.AdamW(parameters.values(),
                                      lr=float(self.config.get("learning_rate", .001)),
                                      weight_decay=0)
        for step in range(self.steps):
            first, second = step % len(images), (step + 1) % len(images)
            excluded = tuple(sorted((first, second)))
            family = (step // len(images)) % 2
            banks = {r: (bank[0][torch.tensor([owner not in excluded for owner in bank[2]],
                                             device=self.device)], None,
                         [owner for owner in bank[2] if owner not in excluded])
                     for r, bank in self.banks.items()}
            assert all(bank[0].shape[0] and not set(excluded).intersection(bank[2])
                       for bank in banks.values())
            inner = self._prepare_episode(images[first], rng, family, first, excluded, banks)
            outer = self._prepare_episode(images[second], rng, 1 - family, second, excluded, banks)
            optimizer.zero_grad(set_to_none=True)
            inner_loss, inner_log = self._episode_loss(scorer, parameters, inner, banks, sampler)
            outer_rng = sampler.get_state()
            # Freeze the detached K1 feedback gain at theta, not at theta'.
            # Both controls make this preview and consume the same sampler RNG.
            with torch.no_grad():
                outer_before, before_log = self._episode_loss(
                    scorer, parameters, outer, banks, sampler)
            replay_sampler = torch.Generator(device=self.device)
            replay_sampler.set_state(outer_rng)
            if self.meta_enabled:
                virtual, inner_gradients = virtual_sgd(parameters, inner_loss, self.inner_lr)
                virtual_norm = float(torch.linalg.vector_norm(torch.stack([
                    (self.inner_lr * gradient.detach()).norm() for gradient in inner_gradients])))
            else:
                virtual, virtual_norm = parameters, 0.
            outer_loss, outer_log = self._episode_loss(
                scorer, virtual, outer, banks, replay_sampler, gain=before_log["lambda_value"])
            total = (inner_loss + self.outer_weight * outer_loss) / (1. + self.outer_weight)
            if not torch.isfinite(total):
                raise RuntimeError("nonfinite cross-environment objective")
            total.backward()
            gradient_norm = nn.utils.clip_grad_norm_(parameters.values(), 5., error_if_nonfinite=True)
            optimizer.step()
            row = dict(loss=float(total.detach()), rank_loss=(inner_log["rank_loss"] +
                       self.outer_weight * outer_log["rank_loss"]) / (1. + self.outer_weight),
                       source=first, outer_source=second, synthetic_family=inner["family"],
                       outer_synthetic_family=outer["family"],
                       boundary_strength=inner["diagnostics"]["strength"],
                       outer_boundary_strength=outer["diagnostics"]["strength"],
                       boundary_feasible=inner["diagnostics"]["feasible"],
                       outer_boundary_feasible=outer["diagnostics"]["feasible"],
                       inner_loss=float(inner_loss.detach()), outer_loss=float(outer_loss.detach()),
                       outer_loss_before=float(outer_before),
                       outer_virtual_loss_change=float(outer_loss.detach() - outer_before),
                       virtual_step_norm=virtual_norm, cross_environment_meta=self.meta_enabled,
                       gradient_norm=float(gradient_norm), P=inner_log["P"],
                       budget=inner_log["budget"], lambda_value=inner_log["lambda_value"],
                       outer_P=outer_log["P"], outer_budget=outer_log["budget"],
                       outer_lambda_value=outer_log["lambda_value"],
                       excluded_sources=list(excluded), resolutions=list(self.resolutions))
            self.history.append(row)
            self.log_training(step, **row)
            # Do not accumulate autograd graphs or dense tensors across steps.
            del inner, outer, inner_loss, outer_loss, total, outer_before, virtual, banks
            if self.meta_enabled:
                del inner_gradients
        self.metric.eval()
        self.adapter.eval()

    def object_metadata(self):
        return {**super().object_metadata(), "transfer_update": self.transfer_enabled,
                "cross_environment_meta": self.meta_enabled if self.transfer_enabled else False,
                "meta_inner_lr": self.inner_lr, "meta_outer_weight": self.outer_weight,
                "inner_optimizer": "functional_sgd_exact_second_order",
                "real_optimizer": "adamw",
                "episode_sources": 2 if self.transfer_enabled else 1,
                "training_bank_exclusion": "both_episode_sources" if self.transfer_enabled
                else "current_source", "meta_gain": "detached_at_pre_unroll_parameters",
                "virtual_domain_is_sealed_holdout": False}
