import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from fmad.methods import register_methods
from fmad.methods.anchored_retrieval import AnchoredRetrieval
from fmad.methods.bcma import BoundedMetric
from fmad.methods.boundary_synthetic import BoundarySynthetic, boundary_choice
from fmad.methods.paired_margin import PairedMargin
from fmad.methods.relational_tail import RelationalTail
from fmad.methods.selective_preservation import SelectivePreservation
from fmad.methods.tail_contrast import ResidualAdapter, paired_losses
from fmad.methods.tail_risk import TailRisk, upper_tail_mean
from fmad.protocol import MethodContext
from fmad.registry import METHOD_REGISTRY
from tests.test_psma_mr_v2 import TinyPSMAV2Backbone, _write_image


VARIANTS = [(SelectivePreservation, "selective_preservation", "I1"),
            (PairedMargin, "paired_margin", "I2"),
            (AnchoredRetrieval, "anchored_retrieval", "I3"),
            (TailRisk, "tail_risk", "J1"),
            (BoundarySynthetic, "boundary_synthetic", "K1")]


def config(**kwargs):
    return dict(resolutions=[8, 12], steps=4, bank_tokens_per_source=4,
                train_tokens=4, hidden_dim=4, adapter_rank=2,
                relational_teacher=True, adaptive_preservation=True,
                immediate_preservation=True, budget_base=.0012, lambda_max=1.,
                **kwargs)


@pytest.mark.parametrize("scores,target,expected", [
    ([.1, .4, .8], .3, (1, True)),
    ([.5, .2, .4], .3, (2, True)),  # No monotone-score assumption.
    ([.1, .2, .25], .3, (2, False)),
    ([.3, .3, .6], .3, (0, True)),
])
def test_boundary_choice_is_closest_feasible_with_full_strength_fallback(scores, target, expected):
    assert boundary_choice(scores, target) == expected


@pytest.mark.parametrize("strengths", [[], [.25], [1., .5], [.5, .5, 1.],
                                       [0., 1.], [float("nan"), 1.]])
def test_boundary_rejects_invalid_strengths(strengths):
    with pytest.raises(ValueError):
        BoundarySynthetic(TinyPSMAV2Backbone(), config(boundary_strengths=strengths))


def test_boundary_generation_uses_frozen_source_exclusion_and_scalar_cache(tmp_path, monkeypatch):
    paths = [_write_image(tmp_path / f"{i}.png", 17 * i) for i in range(2)]
    context = MethodContext("can", 0, 8, False, False)
    model = BoundarySynthetic(TinyPSMAV2Backbone(), config())
    images, rng, _, _ = model._prepare_fit(paths, context)
    calls = []

    def quantile(image, banks, q, mask=None):
        assert all(bank[2] and all(owner == 1 for owner in bank[2])
                   for bank in banks.values())
        calls.append(q)
        return .1 if mask is None else {.25: .12, .5: .18, 1.: .4}[
            model.boundary_strengths[(len(calls) - 2) % 3]]

    monkeypatch.setattr(model, "_teacher_quantile", quantile)
    before = images[0].copy()
    initial_rng = copy.deepcopy(rng.bit_generator.state)
    synthetic, mask, _ = model.training_anomaly(images[0], rng, 0, 0)
    assert model.anomaly_strength == .5 and calls == [.95, .25, .25, .25]
    assert model._normal_thresholds == {0: .1}
    assert all(isinstance(v, (float, int)) for v in model._gradient_diagnostics.values())
    np.testing.assert_array_equal(images[0], before)
    np.testing.assert_array_equal(synthetic[~mask], before[~mask])
    expected_rng = np.random.default_rng()
    expected_rng.bit_generator.state = initial_rng
    reference = RelationalTail(TinyPSMAV2Backbone(), config())
    reference.training_anomaly(images[0], expected_rng, 0, 0)
    assert rng.bit_generator.state == expected_rng.bit_generator.state
    model._prepare_fit(paths, context)
    assert model._normal_thresholds == {}


def test_boundary_teacher_bypasses_adapter_metric_and_retains_no_graph():
    model = BoundarySynthetic(TinyPSMAV2Backbone(), config())
    model.metric = BoundedMetric(8, True, .2, 4)
    model.adapter = ResidualAdapter(8, 2)
    features = F.normalize(torch.randn(4, 8), dim=-1)
    model.extract = lambda image, resolution: (features, (2, 2))
    banks = {r: (features.clone(), None, [1] * 4) for r in model.resolutions}
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    baseline = model._teacher_quantile(image, banks, .95)
    with torch.no_grad():
        for parameter in (*model.metric.parameters(), *model.adapter.parameters()):
            parameter.add_(3.)
    assert model._teacher_quantile(image, banks, .95) == baseline
    assert all(p.grad is None for p in (*model.metric.parameters(), *model.adapter.parameters()))


def test_tail_risk_selects_hard_pairs_with_live_gradients():
    values = torch.tensor([[0., 1.], [2., 4.]], requires_grad=True)
    risk = upper_tail_mean(values, .5)
    assert risk == 3
    risk.backward()
    torch.testing.assert_close(values.grad, torch.tensor([[0., 0.], [.5, .5]]))
    assert upper_tail_mean(values.detach(), 1.) == values.detach().mean()
    empty = torch.empty(0, requires_grad=True)
    upper_tail_mean(empty, .25).backward()
    assert empty.grad.numel() == 0


@pytest.mark.parametrize("fraction", [0., -1., 1.1, float("nan"), float("inf")])
def test_tail_risk_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        TailRisk(TinyPSMAV2Backbone(), config(tail_risk_fraction=fraction))


def test_tail_risk_full_mass_matches_h1_and_preserves_sampler():
    reference = RelationalTail(TinyPSMAV2Backbone(), config())
    model = TailRisk(TinyPSMAV2Backbone(), config(tail_risk_fraction=1.))
    torch.manual_seed(12)
    leaves = [torch.rand(4, 8, requires_grad=True) for _ in range(4)]
    mask = torch.arange(32).reshape(4, 8) < 10
    baseline_rng, risk_rng = (torch.Generator().manual_seed(7) for _ in range(2))
    original = reference.ranking_loss(leaves, mask, baseline_rng)
    risk = model.ranking_loss(leaves, mask, risk_rng)
    torch.testing.assert_close(risk, original)
    assert torch.equal(baseline_rng.get_state(), risk_rng.get_state())
    for a, b in zip(torch.autograd.grad(original, leaves), torch.autograd.grad(risk, leaves)):
        torch.testing.assert_close(a, b)
    model.risk_fraction = .25
    concentrated = model.ranking_loss(leaves, mask, torch.Generator().manual_seed(7))
    assert concentrated >= original
    assert concentrated <= 4 * original + 1e-6


def test_tail_risk_pushes_synthetic_up_and_normal_down():
    model = TailRisk(TinyPSMAV2Backbone(), config())
    leaves = [torch.full((2, 4), .4, requires_grad=True) for _ in range(2)]
    leaves += [torch.full((2, 4), .1, requires_grad=True) for _ in range(2)]
    mask = torch.tensor([[True, True, False, False]] * 2)
    loss = model.ranking_loss(leaves, mask, torch.Generator().manual_seed(1))
    grads = torch.autograd.grad(loss, leaves)
    assert sum(g.sum() for g in grads[:2]) > 0
    assert all((g[mask] <= 0).all() and g[mask].sum() < 0 for g in grads[2:])
    assert all((g[~mask] >= 0).all() for g in grads[2:])
    assert all(isinstance(v, (float, int)) for v in model._gradient_diagnostics.values())


@pytest.mark.parametrize("cls,switch,stage", VARIANTS)
def test_variant_fit_and_disabled_h1_parity(tmp_path, cls, switch, stage):
    paths = [_write_image(tmp_path / f"{i}.png", 17 * i) for i in range(2)]
    context = MethodContext("can", 0, 8, False, False)
    reference = RelationalTail(TinyPSMAV2Backbone(), config())
    reference.fit(paths, context)
    disabled = cls(TinyPSMAV2Backbone(), config(**{switch: False}))
    disabled.fit(paths, context)
    np.testing.assert_array_equal(reference.predict(paths[0])[0], disabled.predict(paths[0])[0])
    assert reference.parameter_count == disabled.parameter_count
    for expected, actual in zip(reference.history, disabled.history):
        assert actual["loss"] == expected["loss"]
    enabled = cls(TinyPSMAV2Backbone(), config())
    before = enabled.backbone.model.weight.detach().clone()
    enabled.fit(paths, context)
    prediction = enabled.predict(paths[0])[0]
    assert prediction.shape == (12, 16) and np.isfinite(prediction).all()
    assert torch.equal(before, enabled.backbone.model.weight)
    assert enabled.backbone.model.weight.grad is None
    assert len(enabled.history) == 4 and not enabled.records
    assert enabled.parameter_count == reference.parameter_count
    assert enabled.object_metadata()[switch] is True
    assert not list(tmp_path.rglob("*.npz")) and not list(tmp_path.rglob("*.pt"))
    if switch == "paired_margin":
        assert enabled._shift_displacement is None


def test_selective_preservation_releases_only_teacher_tail_and_detaches_teacher():
    model = SelectivePreservation(TinyPSMAV2Backbone(), config(normal_tail_fraction=.25))
    teacher = torch.tensor([0., 1., 2., 3.], requires_grad=True)
    clean = torch.tensor([.1, 1.1, 2.1, 8.], requires_grad=True)
    keep, relation = model.preservation_terms(clean, teacher, (clean-teacher).square().mean())
    (keep + relation).backward()
    assert clean.grad[-1] == 0 and clean.grad[:3].abs().sum() > 0
    assert teacher.grad is None and model.preserved_fraction == .75
    flat = torch.ones(4)
    loss, relation = model.preservation_terms(flat, flat, flat.sum()*0)
    assert model.preserved_fraction == 1 and loss == 0 and relation == 0


def test_paired_margin_uses_same_pixel_and_worst_view_with_detached_bounded_target():
    model = PairedMargin(TinyPSMAV2Backbone(), config(margin=.05))
    leaves = [torch.tensor([.1, .2], requires_grad=True),
              torch.tensor([.2, .3], requires_grad=True),
              torch.tensor([.15, .25], requires_grad=True),
              torch.tensor([.22, .32], requires_grad=True)]
    teacher = torch.tensor([.1, .2], requires_grad=True)
    shifted = torch.tensor([.12, 1.], requires_grad=True)
    model.training_step_context(leaves, teacher, shifted)
    loss = model.ranking_loss(leaves, torch.tensor([True, True]), torch.Generator().manual_seed(0))
    # targets [.07,.1], worst uplift [.02,.02].
    assert float(loss) == pytest.approx(.065)
    gradients = torch.autograd.grad(loss, leaves, retain_graph=True)
    assert gradients[1].sum() > 0 and gradients[3].sum() < 0
    assert gradients[0].abs().sum() == 0 and gradients[2].abs().sum() == 0
    assert torch.autograd.grad(loss, (teacher, shifted), allow_unused=True) == (None, None)
    assert model._shift_displacement is None


class ScaleAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([.1, 1.]))

    def forward(self, x):
        return F.normalize(x * self.scale, dim=-1)


def test_anchor_freezes_identity_not_adapter_and_handles_indices():
    model = AnchoredRetrieval(TinyPSMAV2Backbone(), config())
    model.adapter = ScaleAdapter()
    model.metric = BoundedMetric(2, False)
    q = F.normalize(torch.tensor([[1., 1.], [2., 1.]]), dim=-1).requires_grad_(True)
    b = torch.eye(2, requires_grad=True)
    bank = (b, None, [0, 1])
    anchored = model.score(q, (1, 2), bank)
    assert torch.allclose(anchored, 1 - model.adapter(q)[:, 0])
    assert torch.allclose(model.score(q, (1, 2), bank, torch.tensor([1])), anchored[1:])
    frozen = model.score(q, (1, 2), bank, baseline=True)
    assert torch.allclose(frozen, (1-q@b.T).amin(1))
    model.enabled = False
    drifting = model.score(q, (1, 2), bank)
    assert (anchored > drifting).all()
    anchored.sum().backward()
    assert model.adapter.scale.grad.abs().sum() > 0
    assert q.grad.abs().sum() > 0 and b.grad[0].abs().sum() > 0
    assert b.grad[1].abs().sum() == 0


@pytest.mark.parametrize("cls,switch,stage", VARIANTS)
def test_exact_replay_matches_full_graph_for_all_variants(cls, switch, stage):
    torch.manual_seed(4)
    direct = cls(TinyPSMAV2Backbone(), config())
    direct.metric = BoundedMetric(8, True, .2, 4)
    direct.adapter = ResidualAdapter(8, 2)
    with torch.no_grad():
        direct.adapter.up.weight.normal_(std=.02)
    replay = copy.deepcopy(direct)
    inputs = [F.normalize(torch.randn(12, 8), dim=-1) for _ in range(4)]
    bank = (F.normalize(torch.randn(16, 8), dim=-1), None, [])
    mask = torch.arange(12) < 4
    with torch.no_grad():
        clean = direct.score(inputs[0], (3, 4), bank, baseline=True)
        shifted = direct.score(inputs[1], (3, 4), bank, baseline=True)

    def loss(model, leaves):
        model.training_step_context(leaves, clean, shifted)
        rank = model.ranking_loss(leaves, mask, torch.Generator().manual_seed(5))
        keep = (leaves[0]-clean).square().mean()
        shift, contrast = paired_losses(leaves, mask)
        return model.compose_training_loss(leaves, mask, clean, rank, keep, shift, contrast)[0]

    loss(direct, [direct.score(q, (3, 4), bank) for q in inputs]).backward()
    with torch.no_grad():
        values = [replay.score(q, (3, 4), bank) for q in inputs]
    leaves = [value.requires_grad_(True) for value in values]
    gradients = replay.training_gradients(loss(replay, leaves), leaves)
    for q, gradient in zip(inputs, gradients):
        replay.score(q, (3, 4), bank).backward(gradient)
    for original, copied in ((direct.metric, replay.metric), (direct.adapter, replay.adapter)):
        for a, b in zip(original.parameters(), copied.parameters()):
            assert torch.allclose(a.grad, b.grad, atol=2e-7, rtol=1e-4)


@pytest.mark.parametrize("cls,switch,stage", VARIANTS)
def test_registration_config_boundary_and_invalid_switch(cls, switch, stage):
    register_methods(switch)
    assert METHOD_REGISTRY.get(switch) is cls
    with pytest.raises(ValueError):
        cls(TinyPSMAV2Backbone(), config(**{switch: "true"}))
    invalid = config()
    invalid["immediate_preservation"] = False
    with pytest.raises(ValueError):
        cls(TinyPSMAV2Backbone(), invalid)
    root = Path(__file__).resolve().parents[1]/"configs/methods"
    reference = yaml.safe_load((root/"relational_tail_H1.yaml").read_text())
    candidate = yaml.safe_load((root/f"{switch}_{stage}.yaml").read_text())
    assert candidate["method"] == switch
    candidate["method"] = reference["method"]
    assert candidate["method_config"].pop(switch) is True
    assert candidate == reference
