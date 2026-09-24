"""K3: project K1's actual Adam proposal onto a normal-budget halfspace.

K1 synthesis, losses and prediction are unchanged. This is a first-order
constraint on the current support image, not a guarantee of finite-step
preservation, representation quality or anomaly detection performance.
"""
import math

import torch

from fmad.methods.boundary_synthetic import BoundarySynthetic
from fmad.registry import METHOD_REGISTRY


@torch.no_grad()
def project_update(delta, gradient, slack):
    """Closest Euclidean update with gradient.dot(update) <= slack >= 0.

    Use a unit normal and double-precision reductions rather than dividing by
    a clamped squared gradient norm, which can silently relax the constraint.
    A zero preservation gradient provides no first-order restriction.
    """
    if (delta.ndim != 1 or delta.shape != gradient.shape
            or not math.isfinite(slack) or slack < 0
            or not torch.isfinite(delta).all() or not torch.isfinite(gradient).all()):
        raise ValueError("projection requires finite matching vectors and nonnegative slack")
    normal = gradient.double()
    norm = normal.norm()
    if norm == 0:
        return delta, False
    normal = normal / norm
    excess = torch.dot(normal, delta.double()) - slack / norm
    if excess <= 0:
        return delta, False
    return (delta.double() - excess * normal).to(delta.dtype), True


@METHOD_REGISTRY.register("projected_update")
class ProjectedUpdate(BoundarySynthetic):
    def __init__(self, backbone, config):
        super().__init__(backbone, config)
        self.projected_update = config.get("projected_update", True)
        if type(self.projected_update) is not bool:
            raise ValueError("projected_update must be a boolean")
        if config.get("student_boundary", False):
            raise ValueError("projected_update retains K1, not K2 candidate selection")

    def training_update(self, optimizer, parameters, clean_branches, banks, shape,
                        baseline_clean):
        if not self.projected_update:
            return super().training_update(optimizer, parameters, clean_branches,
                                           banks, shape, baseline_clean)

        # An extra clean scoring VJP only: no backbone call, sampler draw or
        # cross-step map/feature cache. autograd.grad leaves total-loss .grad
        # (already clipped by the common loop) and Adam's moments untouched.
        clean = self.fused(self._branches_to_device(clean_branches), banks, shape)
        keep = (clean - baseline_clean.detach()).square().mean()
        keep, relation = self.preservation_terms(clean, baseline_clean, keep)
        preserve = float(self.config.get("keep_weight", 1.)) * keep + self.relational_beta * relation
        value = float(preserve.detach())
        if not math.isfinite(value):
            raise RuntimeError("nonfinite projected-update preservation loss")
        gradients = torch.autograd.grad(preserve, parameters, allow_unused=True)
        normal = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                            for p, g in zip(parameters, gradients)]).detach()
        if not torch.isfinite(normal).all():
            raise RuntimeError("nonfinite projected-update preservation gradient")
        del clean, keep, relation, preserve, gradients

        before = torch.cat([p.detach().reshape(-1) for p in parameters])
        optimizer.step()
        proposal = torch.cat([p.detach().reshape(-1) for p in parameters]) - before
        # Below budget spend the remaining *linearized* slack; above budget
        # prevent a first-order increase, not force a return below budget.
        slack = max(self._budget - value, 0.)
        applied, active = project_update(proposal, normal, slack)
        if active:
            with torch.no_grad():
                offset = 0
                for parameter in parameters:
                    count = parameter.numel()
                    parameter.copy_((before[offset:offset + count]
                                     + applied[offset:offset + count]).view_as(parameter))
                    offset += count
        # Report the actual rounded parameter displacement, not an idealized
        # projected vector. Momentum/variance remain those of the Adam proposal.
        actual = torch.cat([p.detach().reshape(-1) for p in parameters]) - before
        if not torch.isfinite(actual).all():
            raise RuntimeError("nonfinite projected-update parameters")
        linear_change = float(torch.dot(normal.double(), actual.double()))
        return dict(
            update_projection_active=int(active),
            update_projection_zero_gradient=int(normal.count_nonzero() == 0),
            update_preservation_replay_error=abs(value - float(self._preserve_loss.detach())),
            update_preservation_gradient_norm=float(normal.double().norm()),
            update_linear_slack=slack,
            update_raw_linear_change=float(torch.dot(normal.double(), proposal.double())),
            update_applied_linear_change=linear_change,
            update_linear_excess=max(linear_change - slack, 0.),
            update_raw_norm=float(proposal.double().norm()),
            update_applied_norm=float(actual.double().norm()),
            update_correction_norm=float((actual - proposal).double().norm()))

    def object_metadata(self):
        return {**super().object_metadata(),
                "projected_update": self.projected_update,
                "update_projection_space": "actual_adam_parameter_displacement"
                if self.projected_update else None,
                "update_projection_constraint": "current_support_first_order_budget_slack"
                if self.projected_update else None,
                "update_projection_optimizer_state": "retain_adam_proposal_moments",
                "update_projection_extra_vjps": int(self.projected_update),
                "update_projection_off_equivalence": "K1; both switches off H1"}
