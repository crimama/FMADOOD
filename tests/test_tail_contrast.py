import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.nn import functional as F

from fmad.methods.bcma import BCMA, BoundedMetric
from fmad.methods.tail_contrast import TailContrast, ResidualAdapter, paired_losses, tail_ranking
from fmad.protocol import MethodContext
from tests.test_psma_mr_v2 import TinyPSMAV2Backbone, _write_image


@pytest.mark.parametrize("stage", range(4))
def test_stages_and_baseline_equivalence(tmp_path, stage):
    paths = [_write_image(tmp_path / f"{i}.png", 17 * i) for i in range(2)]
    backbone = TinyPSMAV2Backbone()
    before = backbone.model.weight.detach().clone()
    config = dict(resolutions=[8, 12], steps=4, bank_tokens_per_source=4,
                  train_tokens=4, hidden_dim=4, adapter_rank=2,
                  hard_normal_ranking=stage > 0, paired_contrast=stage > 1,
                  low_rank_adapter=stage > 2)
    context = MethodContext("can", 0, 8, False, False)
    model = TailContrast(backbone, config)
    model.fit(paths, context)
    pred, _ = model.predict(paths[0])
    assert pred.shape == (12, 16) and np.isfinite(pred).all()
    assert torch.equal(before, backbone.model.weight) and backbone.model.weight.grad is None
    assert not model.records and len(model.history) == 4
    assert all(len(b[0]) == 8 for b in model.banks.values())
    assert {r["synthetic_family"] for r in model.history} == {"rectangle_cutpaste", "scratch_noise"}
    assert all(r["resolutions"] == [8, 12] for r in model.history)
    if stage == 0:
        reference = BCMA(backbone, dict(config, metric_adaptation=True, context_conditioning=True))
        reference.fit(paths, context)
        np.testing.assert_array_equal(pred, reference.predict(paths[0])[0])
        assert model.history == reference.history
    else:
        assert model.history[-1]["normal_score_gradient_l1"] > 0
        assert model.history[-1]["keep_loss_optimized"] == (stage == 1)
    if stage == 3:
        assert model.adapter.up.weight.abs().sum() > 0
        assert model.object_metadata()["adapter_parameters"] == 32


def test_normal_negative_gradient_and_contrast():
    scores = [torch.tensor([[.1, .2], [.3, .4]], requires_grad=True) for _ in range(4)]
    mask = torch.tensor([[True, False], [False, False]])
    loss = tail_ranking(scores, mask, .05, 4, .05, torch.Generator().manual_seed(0))
    gradients = torch.autograd.grad(loss, scores)
    assert gradients[0].sum() > 0 and gradients[1].sum() > 0
    assert gradients[2][mask].sum() < 0 and gradients[3][mask].sum() < 0
    normal, contrast = paired_losses(scores, mask)
    assert normal == 0 and contrast == 0


def test_adapter_identity_and_both_sides_gradient():
    adapter = ResidualAdapter(8, 2)
    q = F.normalize(torch.randn(7, 8), dim=-1)
    b = F.normalize(torch.randn(7, 8), dim=-1)
    assert torch.allclose(adapter(q), q, atol=2e-7)
    loss = (adapter(q) * adapter(b)).sum()
    loss.backward()
    assert adapter.up.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("adapt", [False, True])
def test_exact_view_replay_matches_full_graph(adapt):
    torch.manual_seed(4)
    direct = TailContrast(TinyPSMAV2Backbone(), {})
    direct.metric = BoundedMetric(8, True, .2, 4)
    if adapt:
        direct.adapter = ResidualAdapter(8, 2)
        with torch.no_grad():
            direct.adapter.up.weight.normal_(std=.02)
    replay = copy.deepcopy(direct)
    inputs = [F.normalize(torch.randn(12, 8), dim=-1) for _ in range(4)]
    bank = (F.normalize(torch.randn(16, 8), dim=-1), None, [])
    def objective(values):
        return (values[2] - values[0] - .05).square().mean() + (values[3] - values[1]).square().mean()
    objective([direct.score(q, (3, 4), bank) for q in inputs]).backward()
    with torch.no_grad():
        values = [replay.score(q, (3, 4), bank) for q in inputs]
    leaves = [v.requires_grad_(True) for v in values]
    grads = torch.autograd.grad(objective(leaves), leaves)
    for q, grad in zip(inputs, grads):
        replay.score(q, (3, 4), bank).backward(grad)
    modules = [(direct.metric, replay.metric)]
    if adapt:
        modules.append((direct.adapter, replay.adapter))
    for first, second in modules:
        for a, b in zip(first.parameters(), second.parameters()):
            assert torch.allclose(a.grad, b.grad, atol=2e-7, rtol=1e-4)


def test_configs_only_toggle_components():
    root = Path(__file__).resolve().parents[1] / "configs/methods"
    configs = [yaml.safe_load((root / f"tail_contrast_D{s}.yaml").read_text()) for s in range(4)]
    for s, config in enumerate(configs):
        assert config["output"] == dict(save_anomaly_maps=False, save_weights=False)
        method = config["method_config"]
        assert method.pop("hard_normal_ranking") == (s > 0)
        assert method.pop("paired_contrast") == (s > 1)
        assert method.pop("low_rank_adapter") == (s > 2)
    assert all(c == configs[0] for c in configs)


def test_invalid_combinations():
    with pytest.raises(ValueError):
        TailContrast(TinyPSMAV2Backbone(), dict(paired_contrast=True))
    with pytest.raises(ValueError):
        TailContrast(TinyPSMAV2Backbone(), dict(hard_normal_ranking="true"))
