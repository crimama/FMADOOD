import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from fmad.methods import register_methods
from fmad.methods.boundary_synthetic import BoundarySynthetic
from fmad.methods.projected_update import ProjectedUpdate, project_update
from fmad.methods.relational_tail import RelationalTail
from fmad.protocol import MethodContext
from fmad.registry import METHOD_REGISTRY
from tests.test_psma_mr_v2 import TinyPSMAV2Backbone, _write_image
from tests.test_three_directions import config


@pytest.mark.parametrize("scale", [1e-30, 1., 1e20])
@pytest.mark.parametrize("slack_fraction", [0., .3, 2.])
def test_projection_is_minimal_halfspace_correction(scale, slack_fraction):
    delta = torch.tensor([2., -3., .5], dtype=torch.float64)
    gradient = torch.tensor([scale, 0., 0.], dtype=torch.float64)
    slack = slack_fraction * 2 * scale
    projected, active = project_update(delta, gradient, slack)
    expected = delta.clone()
    expected[0] = min(2., slack / scale)
    torch.testing.assert_close(projected, expected)
    assert active == (slack_fraction < 1)
    assert torch.dot(gradient, projected) <= slack + 1e-14 * scale
    assert projected.norm() <= delta.norm()
    # All components orthogonal to the constraint are exactly retained.
    assert torch.equal(projected[1:], delta[1:])


def test_zero_gradient_and_inward_update_are_exact_noops():
    delta = torch.tensor([-1., 2.])
    for gradient in (torch.zeros(2), torch.tensor([1., 0.])):
        projected, active = project_update(delta, gradient, 0.)
        assert projected is delta and not active


@pytest.mark.parametrize("delta,gradient,slack", [
    ([float("nan")], [1.], 0.), ([1.], [float("inf")], 0.),
    ([1.], [1.], -1.), ([1.], [1.], float("nan")),
    ([1., 2.], [1.], 0.),
])
def test_invalid_projection_inputs(delta, gradient, slack):
    with pytest.raises(ValueError):
        project_update(torch.tensor(delta), torch.tensor(gradient), slack)


@pytest.mark.parametrize("switch", ["true", 1, None])
def test_switch_is_strict_boolean(switch):
    with pytest.raises(ValueError):
        ProjectedUpdate(TinyPSMAV2Backbone(), config(projected_update=switch))


def test_rejects_k2_combination():
    with pytest.raises(ValueError, match="K1"):
        ProjectedUpdate(TinyPSMAV2Backbone(), config(student_boundary=True))


@pytest.mark.parametrize("boundary", [False, True])
def test_disabled_fit_exactly_matches_h1_or_k1(tmp_path, boundary):
    paths = [_write_image(tmp_path / f"{i}.png", 17 * i) for i in range(2)]
    context = MethodContext("can", 0, 8, False, False)
    reference_class = BoundarySynthetic if boundary else RelationalTail
    reference = reference_class(TinyPSMAV2Backbone(), config())
    reference.fit(paths, context)
    model = ProjectedUpdate(TinyPSMAV2Backbone(), config(
        projected_update=False, boundary_synthetic=boundary))
    model.fit(paths, context)
    assert model.history == reference.history
    assert model.parameter_count == reference.parameter_count
    assert model.backbone.extract_calls == reference.backbone.extract_calls
    np.testing.assert_array_equal(model.predict(paths[0])[0], reference.predict(paths[0])[0])
    for first, second in ((model.metric, reference.metric), (model.adapter, reference.adapter)):
        for actual, expected in zip(first.parameters(), second.parameters()):
            assert torch.equal(actual, expected)


def test_projects_actual_adam_momentum_not_raw_gradient(monkeypatch):
    model = ProjectedUpdate(TinyPSMAV2Backbone(), config(relational_beta=0.))
    p = torch.nn.Parameter(torch.tensor([1., 0.]))
    unused = torch.nn.Parameter(torch.tensor([2.]))
    parameters = [p, unused]
    optimizer = torch.optim.AdamW(parameters, lr=.1, weight_decay=0.)
    for parameter in parameters:
        optimizer.state[parameter] = dict(step=torch.tensor(1.),
                                         exp_avg=-torch.ones_like(parameter),
                                         exp_avg_sq=torch.ones_like(parameter))
        parameter.grad = torch.ones_like(parameter)
    reference_parameters = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    reference_optimizer = torch.optim.AdamW(reference_parameters, lr=.1, weight_decay=0.)
    reference_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for parameter in reference_parameters:
        parameter.grad = torch.ones_like(parameter)
    before = p.detach().clone()
    reference_optimizer.step()
    assert reference_parameters[0][0] > before[0]  # Momentum overrules -gradient.

    monkeypatch.setattr(model, "fused", lambda *args, **kwargs: p)
    model._preserve_loss = p.square().mean()
    model._budget = float(model._preserve_loss.detach())
    gradients_before = [parameter.grad.clone() for parameter in parameters]
    diagnostics = model.training_update(optimizer, parameters, {}, {}, (2,), torch.zeros(2))
    assert diagnostics["update_projection_active"] == 1
    assert diagnostics["update_raw_linear_change"] > 0
    assert abs(diagnostics["update_applied_linear_change"]) < 1e-7
    assert diagnostics["update_preservation_replay_error"] == 0
    assert p[0] == before[0]
    assert p[1] == reference_parameters[0][1]
    # Orthogonal motion still increases this quadratic loss at second order:
    # the implementation must not be described as a finite-step hard budget.
    assert p.square().mean() > model._budget
    assert torch.equal(unused, reference_parameters[1])
    for actual, expected, grad in zip(parameters, reference_parameters, gradients_before):
        assert torch.equal(actual.grad, grad)
        for name in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(optimizer.state[actual][name], reference_optimizer.state[expected][name])


@pytest.mark.parametrize("preservation_budget", [.0012, 1e-12])
def test_enabled_fit_preserves_synthesis_backbone_and_compact_output(tmp_path, preservation_budget):
    paths = [_write_image(tmp_path / f"{i}.png", 17 * i) for i in range(2)]
    context = MethodContext("can", 0, 8, False, False)
    settings = {**config(), "budget_base": preservation_budget}
    reference = BoundarySynthetic(TinyPSMAV2Backbone(), settings)
    reference.fit(paths, context)
    sampler_state = torch.get_rng_state().clone()
    model = ProjectedUpdate(TinyPSMAV2Backbone(), settings)
    before = model.backbone.model.weight.detach().clone()
    model.fit(paths, context)
    assert torch.equal(sampler_state, torch.get_rng_state())
    assert model.backbone.extract_calls == reference.backbone.extract_calls
    assert model.parameter_count == reference.parameter_count
    assert torch.equal(before, model.backbone.model.weight)
    assert model.backbone.model.weight.grad is None
    assert len(model.history) == 4 and not model.records
    if preservation_budget == 1e-12:
        assert any(row["update_projection_active"] for row in model.history)
    for row, original in zip(model.history, reference.history):
        for key in ("source", "synthetic_family", "boundary_strength", "boundary_positive_q",
                    "boundary_feasible", "boundary_normal_q", "boundary_target"):
            assert row[key] == original[key]
        assert row["update_preservation_replay_error"] < 1e-7
        assert row["update_linear_excess"] < 1e-7
        assert row["update_applied_norm"] <= row["update_raw_norm"] + 1e-6
        assert all(isinstance(value, (int, float)) and np.isfinite(value)
                   for key, value in row.items() if key.startswith("update_"))
    metadata = model.object_metadata()
    assert metadata["projected_update"] is True
    assert metadata["update_projection_space"] == "actual_adam_parameter_displacement"
    assert metadata["update_projection_extra_vjps"] == 1
    pred = model.predict(paths[0])[0]
    assert pred.shape == (12, 16) and np.isfinite(pred).all()
    assert not list(tmp_path.rglob("*.npz")) and not list(tmp_path.rglob("*.pt"))
    assert all(isinstance(v, (int, float)) for v in model._normal_thresholds.values())


def test_config_changes_only_method_switch_and_registers():
    root = Path(__file__).resolve().parents[1] / "configs/methods"
    reference = yaml.safe_load((root / "boundary_synthetic_K1.yaml").read_text())
    candidate = yaml.safe_load((root / "projected_update_K3.yaml").read_text())
    assert candidate["method_config"].pop("projected_update") is True
    assert candidate["method"] == "projected_update"
    candidate["method"] = reference["method"]
    assert candidate == reference
    assert candidate["output"] == dict(save_anomaly_maps=False, save_weights=False)
    register_methods("projected_update")
    assert METHOD_REGISTRY.get("projected_update") is ProjectedUpdate
