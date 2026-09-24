import copy

import numpy as np
import pytest
import torch

from fmad.methods.operating_region import (
    OperatingRegion, operating_region_loss, operating_region_thresholds,
)
from fmad.methods.boundary_synthetic import BoundarySynthetic
from fmad.protocol import MethodContext
from tests.test_psma_mr_v2 import TinyPSMAV2Backbone, _write_image


def base_config(**kwargs):
    config = dict(resolutions=[8, 12], steps=4, bank_tokens_per_source=4,
                train_tokens=4, hidden_dim=4, adapter_rank=2,
                boundary_synthetic=True, relational_teacher=True,
                adaptive_preservation=True, immediate_preservation=True,
                budget_base=.0012, lambda_max=1.)
    config.update(kwargs)
    return config


def test_quantiles_are_live_and_finite():
    values = torch.tensor([0., 1., 2., 3.], requires_grad=True)
    q = operating_region_thresholds(values, (.01, .025, .05))
    q.sum().backward()
    assert torch.isfinite(q).all() and values.grad is not None
    assert values.grad.abs().sum() > 0


def test_equal_region_weighting():
    # One large region and one single-pixel region must contribute equally.
    scores = [torch.zeros(4, 4, requires_grad=True) for _ in range(4)]
    scores[2] = torch.tensor([[0., 0., 0., 0.], [0., 0., 0., 0.],
                              [0., 0., 0., 0.], [0., 0., 0., 10.]], requires_grad=True)
    scores[3] = scores[2].clone().detach().requires_grad_(True)
    mask = torch.tensor([[True, True, False, False], [True, True, False, False],
                         [False, False, False, False], [False, False, False, True]])
    loss = operating_region_loss(scores, mask, (.05,), .05, .05)
    expected = .5 * (.05 * torch.nn.functional.softplus(torch.tensor(.05/.05))
                     + .05 * torch.nn.functional.softplus(torch.tensor((.05-10.)/.05)))
    torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-5)


def test_no_tail_negative_sampling_and_both_views_gradients():
    scores = [torch.rand(4, 4, requires_grad=True) for _ in range(4)]
    mask = torch.zeros(4, 4, dtype=torch.bool)
    mask[:2, :2] = True
    loss = operating_region_loss(scores, mask, (.01, .025, .05), .05, .05)
    loss.backward()
    assert all(score.grad is not None for score in scores)
    assert (scores[0].grad.abs().sum() + scores[1].grad.abs().sum()) > 0
    assert scores[2].grad[mask].abs().sum() > 0
    assert scores[3].grad[mask].abs().sum() > 0
    assert scores[2].grad[~mask].abs().sum() == 0


@pytest.mark.parametrize("bad", [[], [0.], [.05, .01], [.06], [float("nan")]])
def test_invalid_operating_fprs(bad):
    with pytest.raises(ValueError):
        OperatingRegion(TinyPSMAV2Backbone(), base_config(operating_fprs=bad))


def test_fit_finite_frozen_and_no_cross_step_cache(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", 17 * i) for i in range(2)]
    model = OperatingRegion(TinyPSMAV2Backbone(), base_config(steps=4))
    before = model.backbone.model.weight.detach().clone()
    model.fit(paths, MethodContext("can", 0, 8, False, False))
    assert len(model.history) == 4 and all(np.isfinite(list(r.values())[0]) for r in model.history if r)
    assert torch.equal(before, model.backbone.model.weight)
    assert not model.records and not hasattr(model, "_operating_region_cache")
    assert all("operating_region_count" in row for row in model.history)


def test_disabled_delegates_exactly_to_k1(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", 17 * i) for i in range(2)]
    cfg = base_config(steps=4, operating_region=False)
    backbone = TinyPSMAV2Backbone()
    first, second = OperatingRegion(backbone, cfg), BoundarySynthetic(
        copy.deepcopy(backbone), base_config())
    first.fit(paths, MethodContext("can", 0, 8, False, False))
    second.fit(paths, MethodContext("can", 0, 8, False, False))
    np.testing.assert_array_equal(first.predict(paths[0])[0], second.predict(paths[0])[0])
    assert first.history == second.history


def test_switches_are_independent_and_disabled_keeps_k1_selector(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", 17 * i) for i in range(2)]
    context = MethodContext("can", 0, 8, False, False)
    model = OperatingRegion(TinyPSMAV2Backbone(), base_config(
        operating_region=False, boundary_strengths=[.25, .5, 1.]))
    model._prepare_fit(paths, context)
    calls = []

    def teacher(_image, _banks, _quantile, mask=None):
        if mask is None:
            return .1
        calls.append(True)
        return [.12, .20, .40][len(calls) - 1]

    model._teacher_quantile = teacher
    model.training_anomaly(model._prepare_fit(paths, context)[0][0],
                           np.random.default_rng(0), 0, 0)
    assert model.enabled is True and model.operating_enabled is False
    assert model.anomaly_strength == .5 and len(calls) == 3

    boundary_off = OperatingRegion(TinyPSMAV2Backbone(), base_config(
        operating_region=True, boundary_synthetic=False))
    assert boundary_off.enabled is False and boundary_off.operating_enabled is True
