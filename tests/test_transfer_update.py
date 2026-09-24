import copy
import os

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from fmad.methods.bcma import BoundedMetric
from fmad.methods.boundary_synthetic import BoundarySynthetic
from fmad.methods.tail_contrast import ResidualAdapter
from fmad.methods.transfer_update import TransferUpdate, _FunctionalScorer, virtual_sgd
from fmad.protocol import MethodContext
from tests.test_psma_mr_v2 import TinyPSMAV2Backbone, _write_image


def config(**overrides):
    return dict(resolutions=[8, 12], steps=2, bank_tokens_per_source=4,
                train_tokens=4, hidden_dim=4, adapter_rank=2,
                relational_teacher=True, adaptive_preservation=True,
                immediate_preservation=True, budget_base=.0012, lambda_max=1.,
                **overrides)


def test_virtual_sgd_has_exact_meta_gradient_without_parameter_mutation():
    theta = torch.tensor([.4, -.3], dtype=torch.float64, requires_grad=True)
    original = theta.detach().clone()
    target_a, target_b = torch.tensor([1., 2.]), torch.tensor([-1., .5])
    inner = .5 * (theta - target_a).square().sum()
    updated, _ = virtual_sgd({"theta": theta}, inner, .2)
    outer = .5 * (updated["theta"] - target_b).square().sum()
    derivative, = torch.autograd.grad((inner + outer) / 2, theta)
    expected = ((theta - target_a) + .8 * (updated["theta"] - target_b)) / 2
    torch.testing.assert_close(derivative, expected)
    first_order = ((theta - target_a) + (updated["theta"] - target_b)) / 2
    assert not torch.allclose(derivative, first_order)
    torch.testing.assert_close(theta, original)
    assert theta.grad is None


def test_functional_scorer_parity_state_restoration_and_second_order():
    from torch.func import functional_call

    torch.manual_seed(7)
    owner = TransferUpdate(TinyPSMAV2Backbone(), config())
    owner.metric = BoundedMetric(8, True, .2, 4).double()
    owner.adapter = ResidualAdapter(8, 2).double()
    scorer = _FunctionalScorer(owner)
    parameters = dict(scorer.named_parameters())
    original = {name: value.detach().clone() for name, value in parameters.items()}
    branches = {8: (F.normalize(torch.randn(4, 8, dtype=torch.float64), dim=-1), (2, 2))}
    banks = {8: (F.normalize(torch.randn(7, 8, dtype=torch.float64), dim=-1), None, [2] * 7)}

    def objective(values):
        return functional_call(scorer, values, (branches, banks, (2, 2)), strict=True).square().mean()

    actual = functional_call(scorer, parameters, (branches, banks, (2, 2)), strict=True)
    torch.testing.assert_close(actual, owner.fused(branches, banks, (2, 2)))
    gradients = torch.autograd.grad(objective(parameters), tuple(parameters.values()), create_graph=True)
    vectors = [torch.randn_like(p) * .01 for p in parameters.values()]
    hvp = torch.autograd.grad(sum((g * v).sum() for g, v in zip(gradients, vectors)),
                              tuple(parameters.values()))
    epsilon = 1e-4
    plus = {n: (p.detach() + epsilon * v).requires_grad_()
            for (n, p), v in zip(parameters.items(), vectors)}
    minus = {n: (p.detach() - epsilon * v).requires_grad_()
             for (n, p), v in zip(parameters.items(), vectors)}
    gp = torch.autograd.grad(objective(plus), tuple(plus.values()))
    gm = torch.autograd.grad(objective(minus), tuple(minus.values()))
    for actual_hvp, positive, negative in zip(hvp, gp, gm):
        torch.testing.assert_close(actual_hvp, (positive - negative) / (2 * epsilon),
                                   atol=1e-7, rtol=1e-3)
    for name, value in scorer.named_parameters():
        assert value is parameters[name]
        torch.testing.assert_close(value, original[name])
        assert value.grad is None


@pytest.mark.parametrize("kwargs", [dict(transfer_update="yes"),
                                    dict(cross_environment_meta=1), dict(meta_inner_lr=0),
                                    dict(meta_inner_lr=float("nan")),
                                    dict(meta_outer_weight=-1)])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        TransferUpdate(TinyPSMAV2Backbone(), config(**kwargs))


def test_disabled_is_exact_k1(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", 17 * i) for i in range(2)]
    context = MethodContext("can", 0, 8, False, False)
    reference = BoundarySynthetic(TinyPSMAV2Backbone(), config())
    disabled = TransferUpdate(TinyPSMAV2Backbone(), config(transfer_update=False))
    reference.fit(paths, context)
    disabled.fit(paths, context)
    assert reference.history == disabled.history
    assert reference.parameter_count == disabled.parameter_count
    np.testing.assert_array_equal(reference.predict(paths[0])[0], disabled.predict(paths[0])[0])


def test_two_source_bank_exclusion_and_frozen_exact_meta_fit(tmp_path, monkeypatch):
    paths = [_write_image(tmp_path / f"{i}.png", 17 * i) for i in range(4)]
    context = MethodContext("can", 0, 8, False, False)
    fitted = []
    for enabled in (False, True):
        backbone = TinyPSMAV2Backbone()
        original = backbone.model.weight.detach().clone()
        model = TransferUpdate(backbone, config(cross_environment_meta=enabled))
        previous = model._prepare_episode
        seen = []

        def inspect(image, rng, family_index, source, excluded, banks):
            assert source in excluded and len(excluded) == 2
            assert all(not set(excluded).intersection(b[2]) for b in banks.values())
            episode = previous(image, rng, family_index, source, excluded, banks)
            seen.append((source, excluded, family_index, copy.deepcopy(episode["diagnostics"])))
            return episode

        monkeypatch.setattr(model, "_prepare_episode", inspect)
        model.fit(paths, context)
        prediction = model.predict(paths[0])[0]
        assert np.isfinite(prediction).all() and np.ptp(prediction) > 0
        assert len(model.history) == 2 and not model.records
        assert model.object_metadata()["training_bank_exclusion"] == "both_episode_sources"
        assert model.object_metadata()["episode_sources"] == 2
        assert len(model._normal_thresholds) == 4  # keyed by source AND excluded set
        assert all(isinstance(v, (int, float)) for v in model._normal_thresholds.values())
        torch.testing.assert_close(backbone.model.weight, original)
        assert all(p.grad is None for p in backbone.model.parameters())
        assert all(np.isfinite(row["loss"]) for row in model.history)
        if enabled:
            assert any(row["virtual_step_norm"] > 0 for row in model.history)
        else:
            assert all(abs(row["outer_virtual_loss_change"]) < 1e-7 for row in model.history)
        fitted.append((model, seen, backbone.extract_calls))
    assert fitted[0][1] == fitted[1][1]
    assert fitted[0][2] == fitted[1][2]
    assert fitted[0][0].parameter_count == fitted[1][0].parameter_count
    assert fitted[0][0].history[0]["inner_loss"] == fitted[1][0].history[0]["inner_loss"]


def test_enabled_rejects_insufficient_supports(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", 17 * i) for i in range(2)]
    with pytest.raises(ValueError, match="three unique"):
        TransferUpdate(TinyPSMAV2Backbone(), config()).fit(
            paths, MethodContext("can", 0, 8, False, False))


@pytest.mark.skipif(os.environ.get("FMAD_L3_CUDA_SMOKE") != "1" or not torch.cuda.is_available(),
                    reason="opt-in full-shape CUDA execution check; no dataset or metrics")
def test_full_shape_cuda_second_order_smoke():
    # Sheet-metal aspect ratio: 448/672 branches and native 1056x4224 scores.
    # Frozen RANDOM features only. This tests execution/memory, not a method
    # hypothesis or benchmark; no real image/GT/checkpoint is used or written.
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    backbone = TinyPSMAV2Backbone(1024)
    backbone.device = "cuda:0"
    model = TransferUpdate(backbone, {**config(), "resolutions": [448, 672],
                                      "train_tokens": 128, "hidden_dim": 32,
                                      "adapter_rank": 16})
    model.metric = BoundedMetric(1024, True, .2, 32).cuda()
    model.adapter = ResidualAdapter(1024, 16).cuda()
    scorer = _FunctionalScorer(model)
    parameters = dict(scorer.named_parameters())
    banks = {r: (F.normalize(torch.randn(14 * 256, 1024, device="cuda"), dim=-1),
                 None, [i for i in range(2, 16) for _ in range(256)]) for r in (448, 672)}
    shape = (1056, 4224)
    mask = np.zeros(shape, dtype=bool)
    mask[300:600, 900:1300] = True
    episodes = []
    for _ in range(2):
        views = [{r: (F.normalize(torch.randn(h * w, 1024), dim=-1), (h, w))
                  for r, (h, w) in ((448, (32, 128)), (672, (48, 192)))} for _ in range(4)]
        episodes.append(dict(cached=views, teacher=torch.full(shape, .1), mask=mask,
                             shape=shape, budget=.0012))
    sampler = torch.Generator(device="cuda").manual_seed(19)
    inner, _ = model._episode_loss(scorer, parameters, episodes[0], banks, sampler)
    virtual, _ = virtual_sgd(parameters, inner, model.inner_lr)
    outer, _ = model._episode_loss(scorer, virtual, episodes[1], banks, sampler, gain=1.)
    ((inner + outer) / 2).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters.values())
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"full-shape L3 CUDA peak allocated: {peak:.3f} GiB")
    # Reserve space for frozen ViT weights/feature extraction in the real fit.
    assert peak < 20., "full-shape graph leaves too little space on local24GiB GPU"
