import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.nn import functional as F

from fmad.methods import register_methods
from fmad.methods.boundary_synthetic import BoundarySynthetic
from fmad.methods.PSMA_MR import _apply_photometric, _sample_photometric_transform
from fmad.methods.relational_tail import RelationalTail
from fmad.methods.student_boundary import StudentBoundary, student_boundary_stats
from fmad.methods.tail_contrast import ResidualAdapter, tail_ranking, tail_samples
from fmad.protocol import MethodContext
from fmad.registry import METHOD_REGISTRY
from tests.test_psma_mr_v2 import TinyPSMAV2Backbone, _write_image
from tests.test_three_directions import config


def legacy_ranking(scores, mask, fraction, count, margin, generator):
    normal = torch.cat([tail_samples(s, fraction, count, generator) for s in scores[:2]])
    terms = []
    for view in scores[2:]:
        positive = view[mask]
        ids = torch.randperm(len(positive), device=view.device, generator=generator)[:count]
        negative = torch.cat((normal, tail_samples(view[~mask], fraction, count, generator)))
        terms.append(F.relu(margin - positive[ids, None] + negative[None]).mean())
    return torch.stack(terms).mean()


def test_shared_sampling_refactor_preserves_legacy_values_gradients_and_rng():
    scores = [torch.rand(5, 7, requires_grad=True) for _ in range(4)]
    mask = torch.arange(35).reshape(5, 7) < 11
    a, b = [torch.Generator().manual_seed(9) for _ in range(2)]
    expected = legacy_ranking(scores, mask, .2, 5, .05, a)
    actual = tail_ranking(scores, mask, .2, 5, .05, b)
    assert torch.equal(expected, actual) and torch.equal(a.get_state(), b.get_state())
    for x, y in zip(torch.autograd.grad(expected, scores), torch.autograd.grad(actual, scores)):
        assert torch.equal(x, y)


def test_selection_uses_shifted_and_background_negatives_and_preserves_rng():
    mask = torch.tensor([True, True, False, False])
    scores = [torch.full((4,), .1), torch.full((4,), .4),
              torch.tensor([.6, .6, .2, .2]), torch.tensor([.6, .6, .7, .7])]
    sampler = torch.Generator().manual_seed(9)
    state = sampler.get_state().clone()
    gap, loss = student_boundary_stats(scores, mask, .05, 4, .05, sampler)
    assert gap == pytest.approx(-.1)
    assert torch.equal(state, sampler.get_state())
    expected = tail_ranking(scores, mask, .05, 4, .05, sampler)
    assert loss == float(expected)
    scores[3][~mask] = .2
    gap, _ = student_boundary_stats(scores, mask, .05, 4, .05, sampler)
    assert gap == pytest.approx(.2)  # shifted-clean .4, not original-clean .1


@pytest.mark.parametrize("teacher_on", [True, False])
def test_disabled_switch_reproduces_k1_or_h1(tmp_path, teacher_on):
    paths = [_write_image(tmp_path / f"{i}.png", 17 * i) for i in range(2)]
    context = MethodContext("can", 0, 8, False, False)
    reference_cls = BoundarySynthetic if teacher_on else RelationalTail
    reference = reference_cls(TinyPSMAV2Backbone(), config())
    reference.fit(paths, context)
    disabled = StudentBoundary(TinyPSMAV2Backbone(), config(
        student_boundary=False, boundary_synthetic=teacher_on))
    disabled.fit(paths, context)
    np.testing.assert_array_equal(reference.predict(paths[0])[0], disabled.predict(paths[0])[0])
    assert reference.history == disabled.history
    assert reference.parameter_count == disabled.parameter_count


def test_generation_previews_exact_transform_sampler_and_excludes_source(tmp_path, monkeypatch):
    paths = [_write_image(tmp_path / f"{i}.png", 17 * i) for i in range(2)]
    model = StudentBoundary(TinyPSMAV2Backbone(), config())
    images, rng, _, sampler = model._prepare_fit(paths, MethodContext("can", 0, 8, False, False))
    model.adapter = ResidualAdapter(8, 2)
    reference_rng = copy.deepcopy(rng)
    reference = RelationalTail(TinyPSMAV2Backbone(), config())
    anomaly, mask, family = reference.training_anomaly(images[0], reference_rng, 0, 0)
    transform = _sample_photometric_transform(copy.deepcopy(reference_rng), 1.)
    expected_views = [images[0], _apply_photometric(images[0], transform)]
    for strength in model.boundary_strengths:
        candidate = model._blend(images[0], anomaly, strength)
        expected_views.extend([candidate, _apply_photometric(candidate, transform)])
    calls = []

    def score(view, banks):
        assert not torch.is_grad_enabled()
        assert all(bank[2] and set(bank[2]) == {1} for bank in banks.values())
        calls.append(view.copy())
        value = torch.full(mask.shape, .1)
        if len(calls) > 2:
            value[torch.as_tensor(mask)] = [.12, .18, .4][(len(calls) - 3) // 2]
        return value

    monkeypatch.setattr(model, "_student_scores", score)
    before_sampler = sampler.get_state().clone()
    selected, actual_mask, actual_family = model.training_anomaly(images[0], rng, 0, 0)
    assert model.anomaly_strength == .5
    assert rng.bit_generator.state == reference_rng.bit_generator.state
    assert torch.equal(sampler.get_state(), before_sampler)
    assert len(calls) == 8
    for actual, expected in zip(calls, expected_views):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_mask, mask)
    np.testing.assert_array_equal(selected, model._blend(images[0], anomaly, .5))
    assert actual_family == family and model._normal_thresholds == {}
    assert all(type(x) in (int, float) for x in model._gradient_diagnostics.values())
    assert all(p.grad is None for p in model.metric.parameters())


def test_fit_replay_matches_selected_objective_and_preserves_output_policy(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", 17 * i) for i in range(2)]
    model = StudentBoundary(TinyPSMAV2Backbone(), config())
    before = model.backbone.model.weight.detach().clone()
    model.fit(paths, MethodContext("can", 0, 8, False, False))
    assert len(model.history) == 4 and not model.records and not model._normal_thresholds
    assert all(row["student_selection_replay_error"] < 1e-7 for row in model.history)
    assert torch.equal(before, model.backbone.model.weight)
    assert model.backbone.model.weight.grad is None
    assert np.isfinite(model.predict(paths[0])[0]).all()
    reference = BoundarySynthetic(TinyPSMAV2Backbone(), config())
    reference.fit(paths, MethodContext("can", 0, 8, False, False))
    assert reference.parameter_count == model.parameter_count
    assert model.object_metadata()["boundary_teacher"] is None
    assert not list(tmp_path.rglob("*.npz")) and not list(tmp_path.rglob("*.pt"))


@pytest.mark.parametrize("overrides", [{"student_boundary": "true"},
                                        {"boundary_synthetic": False}])
def test_invalid_switches(overrides):
    with pytest.raises(ValueError):
        StudentBoundary(TinyPSMAV2Backbone(), config(**overrides))


def test_config_only_changes_selection_and_registers_method():
    root = Path(__file__).resolve().parents[1] / "configs/methods"
    reference = yaml.safe_load((root / "boundary_synthetic_K1.yaml").read_text())
    candidate = yaml.safe_load((root / "student_boundary_K2.yaml").read_text())
    assert candidate.pop("method") == "student_boundary"
    reference.pop("method")
    assert candidate["method_config"].pop("student_boundary") is True
    assert candidate == reference
    register_methods("student_boundary")
    assert METHOD_REGISTRY.get("student_boundary") is StudentBoundary
