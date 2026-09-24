import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.nn import functional as F

from fmad.methods.bcma import BoundedMetric
from fmad.methods.guarded_metric import GuardedLowRank, GuardedMetric, hard_positive_samples
from fmad.protocol import MethodContext
from tests.test_psma_mr_v2 import TinyPSMAV2Backbone, _write_image


def model_with_metric(protect=False):
    model = GuardedMetric(TinyPSMAV2Backbone(), dict(metric_rank=2, hidden_dim=4, query_chunk_size=3))
    model.metric = BoundedMetric(8, True, .2, 4)
    model.low_rank = GuardedLowRank(8, 2, .3)
    if protect:
        model.low_rank.set_evidence(torch.diag(torch.arange(1., 9)), torch.eye(8))
    return model


@pytest.mark.parametrize("protect", [False, True])
def test_exact_full_bank_metric_bounds_and_gradients(protect):
    torch.manual_seed(17)
    model = model_with_metric(protect)
    q = F.normalize(torch.randn(12, 8), dim=-1)
    bank = (F.normalize(torch.randn(13, 8), dim=-1), None, [])
    initial = model.score(q, (3, 4), bank)
    baseline = model.score(q, (3, 4), bank, baseline=True)
    torch.testing.assert_close(initial, baseline, atol=2e-7, rtol=1e-6)
    initial.sum().backward()
    assert model.low_rank.logits.grad.abs().sum() > 0
    with torch.no_grad():
        model.low_rank.logits.copy_(torch.tensor([-1.3, 1.1]))
        model.metric.readout.weight.normal_()
        model.metric.global_weight.normal_()
    for module in (model.metric, model.low_rank):
        module.zero_grad(set_to_none=True)
    reference = copy.deepcopy(model)
    actual = model.score(q, (3, 4), bank)
    p, n, positive, negative = reference.low_rank.factors()
    residual = q[:, None] - bank[0][None]
    expected_pairs = .5 * (residual.square() * (1 + reference.metric(q, (3, 4)))[:, None]).sum(-1)
    expected_pairs += .5 * ((residual @ p).square() * positive).sum(-1)
    expected_pairs -= .5 * ((residual @ n).square() * negative).sum(-1)
    expected = expected_pairs.min(1).values
    torch.testing.assert_close(actual, expected, atol=3e-7, rtol=2e-6)
    assert torch.all(actual >= .5 * baseline - 1e-6)
    assert torch.all(actual <= 1.5 * baseline + 1e-6)
    actual.sum().backward()
    expected.sum().backward()
    for a, b in ((model.metric, reference.metric), (model.low_rank, reference.low_rank)):
        for x, y in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(x.grad, y.grad, atol=2e-6, rtol=2e-4)
    assert model.low_rank.basis.grad.abs().sum() > 0


def test_protection_preserves_complement_and_defect_veto():
    metric = GuardedLowRank(8, 2, .3)
    shift = torch.diag(torch.tensor([0., 0., 0., 0., 0., 0., 2., 4.]))
    defect = torch.diag(torch.tensor([0., 0., 0., 0., 0., 0., 0., 100.]))
    evidence = metric.set_evidence(shift, defect)
    assert evidence["gate_min"] < .05 and evidence["gate_max"] > .99
    with torch.no_grad():
        metric.logits.fill_(-2.)
    p, n, positive, negative = metric.factors()
    complement = torch.eye(8)[:6]
    torch.testing.assert_close(complement @ n, torch.zeros(6, 2), atol=1e-6, rtol=0)
    assert torch.linalg.matrix_norm(n, ord=2) <= 1.00001
    metric.set_evidence(torch.zeros(8, 8), torch.eye(8))
    assert torch.count_nonzero(metric.nuisance_gate) == 0
    assert torch.count_nonzero(metric.factors()[1]) == 0


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_fit_stages_frozen_backbone_state_reset_and_no_artifacts(tmp_path, stage):
    paths = [_write_image(tmp_path / f"{i}.png", 17*i) for i in range(2)]
    backbone = TinyPSMAV2Backbone()
    before = backbone.model.weight.detach().clone()
    config = dict(resolutions=[8, 12], steps=4, bank_tokens_per_source=4,
                  train_tokens=4, hidden_dim=4, metric_rank=2, evidence_tokens=4,
                  nuisance_protection=stage >= 2, weak_anomaly_training=stage >= 3)
    model = GuardedMetric(backbone, config)
    context = MethodContext("can", 0, 8, False, False)
    model.fit(paths, context)
    pred = model.predict(paths[0])[0]
    assert pred.shape == (12, 16) and np.isfinite(pred).all()
    torch.testing.assert_close(before, backbone.model.weight, atol=0, rtol=0)
    assert backbone.model.weight.grad is None and model.adapter is None
    assert not model.records and len(model.history) == 4
    assert all(len(bank[0]) == 8 for bank in model.banks.values())
    assert all(row["resolutions"] == [8, 12] for row in model.history)
    assert all(row["normal_score_gradient_l1"] > 0 for row in model.history)
    metadata = model.object_metadata()
    assert metadata["low_rank_parameters"] == 18
    assert metadata["original_distance_ratio_bounds"] == [.5, 1.5]
    assert (metadata["evidence"].get("defect_groups") == 8) == (stage >= 2)
    assert all(not row["keep_loss_optimized"] for row in model.history)
    if stage == 3:
        assert {row["anomaly_strength"] for row in model.history} == {.35, .65, 1.}
    history = copy.deepcopy(model.history)
    model.fit(paths, context)
    np.testing.assert_array_equal(pred, model.predict(paths[0])[0])
    assert history == model.history
    assert sorted(p.name for p in tmp_path.iterdir()) == ["0.png", "1.png"]


@pytest.mark.parametrize("protect", [False, True])
def test_replay_gradients_remain_exact(protect):
    torch.manual_seed(6)
    direct = model_with_metric(protect)
    with torch.no_grad():
        direct.low_rank.logits.copy_(torch.tensor([.2, -.3]))
    replay = copy.deepcopy(direct)
    inputs = [F.normalize(torch.randn(12, 8), dim=-1) for _ in range(4)]
    bank = (F.normalize(torch.randn(16, 8), dim=-1), None, [])
    def objective(v):
        return (v[2] - v[0] - .05).square().mean() + (v[3] - v[1]).square().mean()
    objective([direct.score(q, (3, 4), bank) for q in inputs]).backward()
    with torch.no_grad():
        values = [replay.score(q, (3, 4), bank) for q in inputs]
    leaves = [v.requires_grad_(True) for v in values]
    gradients = torch.autograd.grad(objective(leaves), leaves)
    for q, gradient in zip(inputs, gradients):
        replay.score(q, (3, 4), bank).backward(gradient)
    for first, second in ((direct.metric, replay.metric), (direct.low_rank, replay.low_rank)):
        for a, b in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-7, rtol=1e-4)


def test_hard_positive_budget_and_gradients():
    values = torch.arange(100., requires_grad=True)
    selected = hard_positive_samples(values, 16, .25, torch.Generator().manual_seed(1))
    assert len(selected) == 16 and torch.all(selected[:8] < 25)
    selected.sum().backward()
    assert values.grad.sum() == 16


def test_evidence_does_not_change_main_rng_banks_or_parameter_budget(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", 17*i) for i in range(2)]
    models, draws = [], []
    for stage in (1, 2, 3):
        model = GuardedMetric(TinyPSMAV2Backbone(), dict(resolutions=[8, 12], metric_rank=2,
            bank_tokens_per_source=4, evidence_tokens=4, nuisance_protection=stage >= 2,
            weak_anomaly_training=stage >= 3))
        _, rng, _, sampler = model._prepare_fit(paths, MethodContext("can", 0, 8, False, False))
        draws.append((rng.integers(0, 100000, 20).tolist(), torch.randperm(100, generator=sampler).tolist()))
        models.append(model)
    assert draws[0] == draws[1] == draws[2]
    for other in models[1:]:
        for resolution in (8, 12):
            torch.testing.assert_close(models[0].banks[resolution][0], other.banks[resolution][0], atol=0, rtol=0)
        torch.testing.assert_close(models[0].low_rank.basis, other.low_rank.basis, atol=0, rtol=0)
    torch.testing.assert_close(models[1].low_rank.nuisance_basis, models[2].low_rank.nuisance_basis, atol=0, rtol=0)
    assert models[1].evidence == models[2].evidence


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not installed")
def test_cuda_eigensolver_metric_and_backward():
    model = model_with_metric()
    model.metric.to("cuda")
    model.low_rank.to("cuda")
    model.low_rank.set_evidence(torch.eye(8, device="cuda"), torch.eye(8, device="cuda"))
    with torch.no_grad():
        model.low_rank.logits.copy_(torch.tensor([.2, -.3], device="cuda"))
    q = F.normalize(torch.randn(12, 8, device="cuda"), dim=-1)
    bank = (F.normalize(torch.randn(16, 8, device="cuda"), dim=-1), None, [])
    score = model.score(q, (3, 4), bank)
    baseline = model.score(q, (3, 4), bank, baseline=True)
    assert torch.all(score >= .5 * baseline - 1e-6)
    assert torch.all(score <= 1.5 * baseline + 1e-6)
    score.sum().backward()
    assert torch.isfinite(model.low_rank.basis.grad).all()
    assert torch.isfinite(model.low_rank.logits.grad).all()


def test_configs_only_change_declared_components():
    root = Path(__file__).resolve().parents[1] / "configs/methods"
    configs = [yaml.safe_load((root / f"guarded_metric_E{s}.yaml").read_text()) for s in (1, 2, 3)]
    for stage, config in enumerate(configs, 1):
        assert config["output"] == dict(save_anomaly_maps=False, save_weights=False)
        method = config["method_config"]
        assert method.pop("nuisance_protection") == (stage >= 2)
        assert method.pop("weak_anomaly_training") == (stage >= 3)
    assert all(c == configs[0] for c in configs)
    original = yaml.safe_load((root / "tail_contrast_D3.yaml").read_text())
    for key in ("dataset", "backbone", "seed", "device", "protocol", "dataset_config", "backbone_config", "evaluation", "output"):
        assert configs[0][key] == original[key]
    for key in ("resolutions", "steps", "bank_tokens_per_source", "train_tokens", "learning_rate", "margin", "shift_weight", "contrast_weight"):
        assert configs[0]["method_config"][key] == original["method_config"][key]


@pytest.mark.parametrize("config", [dict(low_rank_bound=.8), dict(metric_rank=0),
    dict(nuisance_protection="yes"), dict(weak_anomaly_training=True), dict(low_rank_adapter=True)])
def test_invalid_config(config):
    with pytest.raises(ValueError):
        GuardedMetric(TinyPSMAV2Backbone(), config)
