from pathlib import Path
import copy

import numpy as np
import pytest
import torch
import yaml

from fmad.methods.relational_tail import RelationalTail, relational_score_loss
from tests.test_psma_mr_v2 import TinyPSMAV2Backbone, _write_image
from fmad.protocol import MethodContext


def test_relational_loss_is_finite_and_teacher_detached():
    student = torch.randn(12, requires_grad=True)
    teacher = torch.randn(12, requires_grad=True)
    loss = relational_score_loss(student, teacher)
    loss.backward()
    assert torch.isfinite(loss) and student.grad is not None and teacher.grad is None


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_rgta_stages_keep_d3_adapter_and_no_artifacts(tmp_path, stage):
    paths = [_write_image(tmp_path / f"{i}.png", 17 * i) for i in range(2)]
    config = dict(resolutions=[8, 12], steps=4, bank_tokens_per_source=4,
                  train_tokens=4, hidden_dim=4, adapter_rank=2,
                  relational_teacher=True, trusted_weak_synthetic=stage >= 2,
                  conflict_safe_gradient=stage >= 3)
    model = RelationalTail(TinyPSMAV2Backbone(), config)
    model.fit(paths, MethodContext("can", 0, 8, False, False))
    pred = model.predict(paths[0])[0]
    assert pred.shape == (12, 16) and np.isfinite(pred).all()
    assert model.adapter is not None and model.adapter.up.weight.abs().sum() > 0
    assert len(model.history) == 4 and not model.records
    assert all(row["keep_loss_optimized"] for row in model.history)
    assert all("relational_loss" in row for row in model.history)
    if stage == 2:
        assert all("synthetic_reliability_mean" in row for row in model.history)
    if stage == 3:
        assert all("conflict_steps" in row for row in model.history)
    if stage >= 2:
        assert {row["anomaly_strength"] for row in model.history} == {.35, .65, 1.}
    assert model.object_metadata()["conflict_projection"] == (
        "score_leaf_pcgrad" if stage == 3 else None)


def test_rgta_replay_projection_is_finite():
    model = RelationalTail(TinyPSMAV2Backbone(), dict(relational_teacher=True,
                                                      conflict_safe_gradient=True))
    leaves = [torch.randn(5, requires_grad=True) for _ in range(4)]
    model._preserve_loss = (leaves[0] ** 2).mean()
    model._anomaly_loss = (-leaves[0] ** 2).mean()
    grads = model.training_gradients(model._preserve_loss + model._anomaly_loss, leaves)
    assert all(g is None or torch.isfinite(g).all() for g in grads)
    assert model._gradient_diagnostics["conflict_steps"] == 1


@pytest.mark.parametrize("preserve_slot, anomaly_slot", [(0, 1), (1, 0), (2, 3), (3, 2)])
def test_rgta_replay_projection_zero_safe_for_none_gradients(preserve_slot, anomaly_slot):
    model = RelationalTail(TinyPSMAV2Backbone(), dict(relational_teacher=True,
                                                      conflict_safe_gradient=True))
    leaves = [torch.randn(4, requires_grad=True) for _ in range(4)]
    model._preserve_loss = (leaves[preserve_slot] ** 2).mean()
    model._anomaly_loss = -(leaves[anomaly_slot] ** 2).mean()
    grads = model.training_gradients(model._preserve_loss + model._anomaly_loss, leaves)
    assert all(torch.isfinite(g).all() for g in grads)


def test_configs_are_matched_and_incremental():
    root = Path(__file__).resolve().parents[1] / "configs/methods"
    configs = [yaml.safe_load((root / f"relational_tail_F{i}.yaml").read_text()) for i in (1, 2, 3)]
    for i, config in enumerate(configs, 1):
        assert config["method"] == "relational_tail"
        assert config["output"] == {"save_anomaly_maps": False, "save_weights": False}
        method = config["method_config"]
        assert method["relational_teacher"] is True
        assert method["trusted_weak_synthetic"] is (i >= 2)
        assert method["conflict_safe_gradient"] is (i >= 3)
    baseline = yaml.safe_load((root / "tail_contrast_D3.yaml").read_text())
    for key in ("dataset", "backbone", "seed", "device", "protocol", "dataset_config",
                "backbone_config", "evaluation", "output"):
        assert configs[0][key] == baseline[key]


def test_gard_budget_context_and_dual_diagnostics():
    model = RelationalTail(TinyPSMAV2Backbone(), dict(
        relational_teacher=True, adaptive_preservation=True,
        budget_base=.2, susceptibility_scale=1., lambda_init=.1,
        lambda_lr=.5, lambda_max=.8))
    scores = [torch.randn(8, requires_grad=True) for _ in range(4)]
    clean = torch.ones(8)
    shifted = torch.ones(8) * 1.5
    model.training_step_context(scores, clean, shifted)
    total, diagnostics = model.compose_training_loss(
        scores, torch.ones(8, dtype=torch.bool), clean,
        torch.tensor(.3, requires_grad=True),
        (scores[0] - clean).square().mean(), torch.tensor(.1), torch.tensor(.1))
    assert torch.isfinite(total)
    assert model._gradient_diagnostics["nu"] > 0
    assert 0 < model._gradient_diagnostics["budget"] < .2
    assert {"P", "budget", "violation", "lambda_value"} <= diagnostics.keys()


def test_gard_synthetic_weight_is_detached_and_configured():
    model = RelationalTail(TinyPSMAV2Backbone(), dict(
        relational_teacher=True, trusted_weak_synthetic=True,
        budget_aware_synthetic=True, anomaly_strengths=[.5]))
    assert model.budget_aware_synthetic
    assert model.features["trusted_weak_synthetic"] is True


def test_gard_dual_scale_is_slow_and_non_saturating():
    model = RelationalTail(TinyPSMAV2Backbone(), dict(
        relational_teacher=True, adaptive_preservation=True, budget_base=.0012,
        lambda_init=.1, lambda_lr=.01, lambda_max=1., violation_ema_decay=.9))
    model._budget = .0012
    values = []
    # Representative F3 P scale, alternating slight violations and slack.
    for p in (.0017, .0004) * 32:
        model._update_dual(p - model._budget)
        values.append(model.lambda_value)
    assert values[-1] != values[0]
    assert 0.01 < min(values) and max(values) < .99


@pytest.mark.parametrize("relative, expected", [(0., .1), (.05, .1), (.5, .5),
                                                (1., 1.), (4., 1.)])
def test_immediate_preservation_has_no_zero_or_delayed_gain(relative, expected):
    model = RelationalTail(TinyPSMAV2Backbone(), dict(
        relational_teacher=True, adaptive_preservation=True,
        immediate_preservation=True, budget_base=.0012,
        lambda_init=.1, lambda_max=1.))
    # Reproduce the state that delayed the old controller even after P > b.
    model.lambda_value = 0.
    model._violation_ema = -1.
    scores = [torch.linspace(0., 1., 8, requires_grad=True) for _ in range(4)]
    keep = torch.tensor(relative * model._budget, requires_grad=True)
    zero = torch.tensor(0.)
    total, diagnostics = model.compose_training_loss(
        scores, torch.ones(8, dtype=torch.bool), scores[0].detach(),
        zero, keep, zero, zero)
    assert diagnostics["dual_lambda"] == pytest.approx(expected)
    assert diagnostics["lambda_value"] == diagnostics["dual_lambda"]
    assert model._violation_ema == -1.  # No historical-state update in this mode.
    total.backward()
    assert keep.grad.item() == pytest.approx(expected)  # Controller gain is detached.


@pytest.mark.parametrize("override", [
    {"adaptive_preservation": False}, {"relational_teacher": False},
    {"conflict_safe_gradient": True}, {"immediate_preservation": "true"},
    {"lambda_init": 0.}, {"lambda_init": 2.}, {"lambda_max": float("nan")},
    {"budget_base": 0.}, {"susceptibility_scale": -1.},
])
def test_immediate_preservation_rejects_incompatible_options(override):
    with pytest.raises(ValueError):
        RelationalTail(TinyPSMAV2Backbone(), dict(
            relational_teacher=True, adaptive_preservation=True,
            immediate_preservation=True, lambda_init=.1, lambda_max=1.) | override)


def test_h1_changes_only_the_preservation_controller():
    root = Path(__file__).resolve().parents[1] / "configs/methods"
    baseline = yaml.safe_load((root / "relational_tail_G1.yaml").read_text())
    candidate = yaml.safe_load((root / "relational_tail_H1.yaml").read_text())
    assert candidate["method_config"].pop("immediate_preservation") is True
    assert candidate == baseline


def test_immediate_preservation_replay_fits_without_extra_artifacts(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", 17 * i) for i in range(2)]
    model = RelationalTail(TinyPSMAV2Backbone(), dict(
        resolutions=[8, 12], steps=4, bank_tokens_per_source=4,
        train_tokens=4, hidden_dim=4, adapter_rank=2,
        relational_teacher=True, adaptive_preservation=True,
        immediate_preservation=True, budget_base=.0012,
        lambda_init=.1, lambda_max=1.))
    model.fit(paths, MethodContext("can", 0, 8, False, False))
    assert len(model.history) == 4 and not model.records
    assert all(.1 <= row["dual_lambda"] <= 1. for row in model.history)
    assert all(row["dual_lambda"] == pytest.approx(
        max(.1, min(1., row["P"] / row["budget"]))) for row in model.history)
    assert model.adapter.up.weight.abs().sum() > 0
    assert model.object_metadata()["immediate_preservation"] is True
    assert np.isfinite(model.predict(paths[0])[0]).all()
