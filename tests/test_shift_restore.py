import copy

import numpy as np
import pytest
import torch
import yaml

from fmad.methods.boundary_synthetic import BoundarySynthetic
from fmad.methods.shift_restore import ShiftRestore, balanced_reconstruction, mask_coverage
from fmad.protocol import MethodContext
from tests.test_psma_mr_v2 import TinyPSMAV2Backbone, _write_image


def config(**kwargs):
    values = dict(resolutions=[8, 12], steps=2, bank_tokens_per_source=4,
                train_tokens=4, hidden_dim=4, adapter_rank=2,
                relational_teacher=True, adaptive_preservation=True,
                immediate_preservation=True, budget_base=.0012, lambda_max=1.,
                shift_restore=True, restoration_target="same_shift")
    values.update(kwargs)
    return values


def test_same_shift_and_canonical_use_different_targets():
    model = ShiftRestore(TinyPSMAV2Backbone(), config())
    source = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    shifted = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    model.restorer = torch.nn.Identity()
    same = model._restore_loss(shifted, shifted)
    canonical = model._restore_loss(shifted, source)
    assert float(same) == pytest.approx(0.)
    assert float(canonical) > 0.


def test_area_coverage_balances_foreground_and_background():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:4, :4] = 1
    coverage = mask_coverage(mask, (2, 2), torch.device("cpu"))
    prediction = torch.zeros(4, 2)
    target = torch.zeros(4, 2)
    target[0] = 2.
    target[3] = 1.
    loss = balanced_reconstruction(prediction, target, coverage.flatten())
    # Region means are (4, 1) and (0, 0), then equally averaged.
    assert float(loss) == pytest.approx(13. / 6.)
    assert float(coverage.sum()) > 0.


def test_tiny_mask_has_explicit_nonempty_coverage():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[0, 0] = 1
    coverage = mask_coverage(mask, (1, 1), torch.device("cpu"))
    assert float(coverage.sum()) > 0.


def test_disabled_shift_restore_delegates_exact_k1(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", 17 * i) for i in range(2)]
    context = MethodContext("can", 0, 8, False, False)
    reference = BoundarySynthetic(TinyPSMAV2Backbone(), config(shift_restore=False))
    disabled = ShiftRestore(TinyPSMAV2Backbone(), config(shift_restore=False))
    reference.fit(paths, context)
    disabled.fit(paths, context)
    np.testing.assert_array_equal(reference.predict(paths[0])[0], disabled.predict(paths[0])[0])
    assert reference.parameter_count == disabled.parameter_count
    for expected, actual in zip(reference.history, disabled.history):
        assert actual["loss"] == expected["loss"]


def test_disabled_restoration_keeps_k1_boundary_selector_enabled():
    settings = config(shift_restore=False, boundary_synthetic=True,
                      boundary_strengths=[.25, .5, 1.])
    model = ShiftRestore(TinyPSMAV2Backbone(), settings)
    assert model.restoration_enabled is False
    assert model.enabled is True


def test_l2_and_l2c_keep_views_and_rng_but_change_target(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", 17 * i) for i in range(2)]
    context = MethodContext("can", 0, 8, False, False)
    l2 = ShiftRestore(TinyPSMAV2Backbone(), config(restoration_target="same_shift"))
    l2c = ShiftRestore(TinyPSMAV2Backbone(), config(restoration_target="canonical"))
    before = copy.deepcopy(l2.config)
    l2.fit(paths, context)
    l2c.fit(paths, context)
    assert l2.history[0]["synthetic_family"] == l2c.history[0]["synthetic_family"]
    assert l2.history[0]["source"] == l2c.history[0]["source"]
    assert l2.history[0]["restoration_target"] == "same_shift"
    assert l2c.history[0]["restoration_target"] == "canonical"
    assert l2.parameter_count == l2c.parameter_count
    for left, right in zip(l2.history, l2c.history):
        for key in ("source", "synthetic_family", "boundary_strength",
                    "boundary_normal_q", "boundary_target", "boundary_positive_q",
                    "boundary_feasible"):
            assert left[key] == right[key]
    assert before["restoration_target"] == "same_shift"


def test_restoration_has_only_module_parameters_and_finite_nonconstant_prediction(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", 17 * i) for i in range(2)]
    context = MethodContext("can", 0, 8, False, False)
    model = ShiftRestore(TinyPSMAV2Backbone(), config())
    frozen = model.backbone.model.weight.detach().clone()
    model.fit(paths, context)
    assert model.metric is None
    torch.testing.assert_close(model.backbone.model.weight, frozen)
    assert model.parameter_count == sum(p.numel() for p in model.restorer.parameters())
    result = model.predict(paths[0])[0]
    assert np.isfinite(result).all()
    assert np.ptp(result) > 0.
    assert model.object_metadata()["score_definition"] == "mean_resolution_squared_l2_restoration_residual"


def test_configs_declare_l2_variants():
    root = __import__("pathlib").Path(__file__).parents[1]
    l2 = yaml.safe_load((root / "configs/methods/shift_restore_L2.yaml").read_text())
    l2c = yaml.safe_load((root / "configs/methods/shift_restore_L2C.yaml").read_text())
    assert l2["method_config"]["restoration_target"] == "same_shift"
    assert l2c["method_config"]["restoration_target"] == "canonical"
