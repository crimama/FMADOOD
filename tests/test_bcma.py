import numpy as np
import pytest
import torch
from torch.nn import functional as F
from fmad.methods.bcma import BCMA, BoundedMetric
from fmad.methods.contextual_score import ContextualScore
from fmad.protocol import MethodContext
from tests.test_psma_mr_v2 import TinyPSMAV2Backbone, _write_image


def test_metric_bounds_and_gradient():
    model = BCMA(TinyPSMAV2Backbone(), {})
    model.metric = BoundedMetric(8, True, .2, 4)
    q, b = F.normalize(torch.randn(12, 8), dim=-1), F.normalize(torch.randn(15, 8), dim=-1)
    bank = (b, None, [])
    baseline = model.score(q, (3, 4), bank, baseline=True)
    assert torch.allclose(model.score(q, (3, 4), bank), baseline, atol=2e-7)
    with torch.no_grad():
        model.metric.global_weight.normal_()
        model.metric.readout.weight.normal_()
    score = model.score(q, (3, 4), bank)
    assert torch.all(score >= .8 * baseline - 1e-6)
    assert torch.all(score <= 1.2 * baseline + 1e-6)
    explicit = (.5 * (1 + model.metric(q, (3, 4)))[:, None] * (q[:, None] - b[None]).square()).sum(-1).amin(1)
    assert torch.allclose(score, explicit, atol=3e-7)
    score.sum().backward()
    assert model.metric.global_weight.grad.abs().sum() > 0


@pytest.mark.parametrize("stage", range(4))
def test_fit_pipeline_and_family_coverage(tmp_path, stage):
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(2)]
    backbone = TinyPSMAV2Backbone()
    context = MethodContext("can", 0, 8, False, False)
    config = dict(resolutions=[8, 12], steps=4, bank_tokens_per_source=4,
                  train_tokens=4, hidden_dim=4, metric_adaptation=stage > 0,
                  context_conditioning=stage > 1, correspondence_regularization=stage > 2)
    model = BCMA(backbone, config)
    before = backbone.model.weight.detach().clone()
    model.fit(paths, context)
    score, _ = model.predict(paths[0])
    assert score.shape == (12, 16) and np.isfinite(score).all()
    assert torch.equal(before, backbone.model.weight) and backbone.model.weight.grad is None
    assert model.records == {}
    if stage == 0:
        baseline = ContextualScore(backbone, {**config, "score_head": False})
        baseline.fit(paths, context)
        np.testing.assert_array_equal(score, baseline.predict(paths[0])[0])
    else:
        assert len(model.history) == 4
        assert {row["synthetic_family"] for row in model.history} == {"rectangle_cutpaste", "scratch_noise"}
        assert all(row["resolutions"] == [8, 12] for row in model.history)
        assert (model.history[-1]["correspondence_loss"] > 0) == (stage == 3)


def test_switch_validation():
    with pytest.raises(ValueError):
        BCMA(TinyPSMAV2Backbone(), dict(context_conditioning=True))
