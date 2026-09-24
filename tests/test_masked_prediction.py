import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from fmad.methods.masked_prediction import (
    MaskedPrediction, hide_region, normal_loss, pixel_mask,
    prediction_score, synthetic_margin, tile_masks,
)
from fmad.protocol import MethodContext
from tests.test_psma_mr_v2 import TinyPSMAV2Backbone, _write_image


def settings(**kwargs):
    values = dict(resolutions=[8, 12], steps=2, mask_rows=2, mask_columns=2,
                  guard_tokens=1, train_tokens=4, hidden_dim=8, attention_heads=2,
                  predictor_layers=1, context_pool_grid=2,
                  target_layer=11, context_layer=23,
                  uncertainty_normalization=False, synthetic_supervision=False)
    values.update(kwargs)
    return values


def test_tile_partition_covers_each_patch_exactly_once():
    grid = (7, 13)
    cover = torch.zeros(grid, dtype=torch.int)
    for tile in range(8):
        target, hidden = tile_masks(grid, 2, 4, tile, 1)
        cover += target
        assert torch.all(hidden[target])
    assert torch.all(cover == 1)


def test_hiding_is_invariant_to_every_hidden_pixel_including_style():
    image = torch.randn(3, 16, 24)
    target, hidden = tile_masks((8, 12), 2, 4, 3, 1)
    changed = image.clone()
    changed[:, pixel_mask(hidden, image)] = 1000
    masked_a, style_a = hide_region(image, hidden)
    masked_b, style_b = hide_region(changed, hidden)
    torch.testing.assert_close(masked_a, masked_b, rtol=0, atol=0)
    torch.testing.assert_close(style_a, style_b, rtol=0, atol=0)
    assert not torch.equal(image, masked_a)


def test_bicubic_resize_spillover_stays_inside_one_token_guard():
    from types import SimpleNamespace
    from fmad.backbones.wrapper import DINOv2Wrapper
    # Exercise the actual common preprocessing without loading model weights.
    wrapper = DINOv2Wrapper.__new__(DINOv2Wrapper)
    wrapper.smaller_edge_size = 448
    wrapper._transform_cache = {}
    wrapper.model = SimpleNamespace(patch_size=14)
    image = np.full((140, 196, 3), 110, dtype=np.uint8)
    tensor, grid = wrapper.prepare_image(image)
    _, hidden = tile_masks(grid, 2, 4, 0, 1)
    anomaly = image.copy()
    # Native first tile; resize interpolation crosses its edge before masking.
    anomaly[:70, :49] = 230
    changed, _ = wrapper.prepare_image(anomaly)
    masked_a, style_a = hide_region(tensor, hidden)
    masked_b, style_b = hide_region(changed, hidden)
    torch.testing.assert_close(masked_a, masked_b, rtol=0, atol=0)
    torch.testing.assert_close(style_a, style_b, rtol=0, atol=0)


def test_synthetic_loss_cannot_backpropagate_variance():
    mean = torch.zeros(4, 3, requires_grad=True)
    variance = torch.ones(4, 1, requires_grad=True)
    normal = torch.zeros_like(mean)
    anomaly = torch.full_like(mean, .1)
    loss = synthetic_margin(mean, variance, normal, anomaly, torch.ones(4), .2)
    loss.backward()
    assert mean.grad is not None and mean.grad.abs().sum() > 0
    assert variance.grad is None


def test_normal_loss_trains_variance_and_score_is_not_log_likelihood():
    variance = torch.ones(2, 1, requires_grad=True)
    mean = torch.zeros(2, 3)
    target = torch.full((2, 3), 2.)
    normal_loss(mean, variance, target).backward()
    assert variance.grad is not None
    torch.testing.assert_close(prediction_score(mean, variance, target), torch.full((2,), 4.))


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))])
def test_p_variants_share_episodes_targets_and_extraction_budget(tmp_path, device):
    paths = [_write_image(tmp_path / f"{i}.png", 17*i) for i in range(2)]
    context = MethodContext("can", 0, 8, False, False)
    methods = []
    for stage in range(3):
        backbone = TinyPSMAV2Backbone()
        backbone.device = device
        model = MaskedPrediction(backbone, settings(
            uncertainty_normalization=stage > 0, synthetic_supervision=stage > 1))
        frozen = copy.deepcopy(model.backbone.model.state_dict())
        model.fit(paths, context)
        for name, value in model.backbone.model.state_dict().items():
            torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
        assert all(p.grad is None for p in model.backbone.model.parameters())
        result, score = model.predict(paths[0])
        assert result.shape == (12, 16) and np.isfinite(result).all()
        assert np.ptp(result) > 0 and np.isfinite(score)
        assert model.object_metadata()["bank_sizes"] == {}
        assert model.object_metadata()["steps_completed"] == 2
        assert len(model.history) == 2
        methods.append(model)
    for other in methods[1:]:
        for left, right in zip(methods[0].history, other.history):
            for key in ("source", "tile", "synthetic_family", "synthetic_coverage", "episode_sha256"):
                assert left[key] == right[key]
        assert methods[0].fit_feature_calls == other.fit_feature_calls
    # Each inference makes precisely one observed + four masked calls per resolution.
    for model in methods:
        assert model.feature_calls - model.fit_feature_calls == 10


def test_same_shift_target_is_observed_not_canonical(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", 17*i) for i in range(2)]
    model = MaskedPrediction(TinyPSMAV2Backbone(), settings(steps=1, resolutions=[8]))
    captured = []
    original = model.features
    def record(tensor, grid):
        result = original(tensor, grid)
        captured.append(result[0].clone())
        return result
    model.features = record
    model.fit(paths, MethodContext("can", 0, 8, False, False))
    # Two initialization calls followed by (normal, abnormal, masked) x two views.
    assert not torch.equal(captured[2], captured[5])
    assert model.object_metadata()["feature_target"].startswith("same_view")


def test_variance_bounds_and_inactive_head(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", 17*i) for i in range(2)]
    for enabled in (False, True):
        model = MaskedPrediction(TinyPSMAV2Backbone(), settings(
            steps=1, uncertainty_normalization=enabled))
        model.fit(paths, MethodContext("can", 0, 8, False, False))
        assert all(p.requires_grad == enabled for p in model.predictor.variance_head.parameters())
        with torch.no_grad():
            model.predictor.variance_head.bias.fill_(100.)
            _, var = model.predictor(torch.randn(16, 8), (4, 4), torch.arange(4), torch.zeros(6))
        factor = var / model.predictor.base_variance
        torch.testing.assert_close(factor, torch.full_like(factor, 10. if enabled else 1.))


@pytest.mark.parametrize("override", [dict(uncertainty_normalization="false"),
    dict(synthetic_supervision=True), dict(mask_rows=0), dict(guard_tokens=-1), dict(guard_tokens=0),
    dict(variance_min_factor=0.), dict(variance_max_factor=float("nan")),
    dict(hidden_dim=7), dict(target_layer=10)])
def test_invalid_configs_rejected(override):
    with pytest.raises(ValueError):
        MaskedPrediction(TinyPSMAV2Backbone(), settings(**override))


def test_configs_are_isolated_flags_and_outputs_off():
    root = Path(__file__).parents[1]
    configs = [yaml.safe_load((root / f"configs/methods/masked_prediction_P{i}.yaml").read_text())
               for i in range(3)]
    for i, config in enumerate(configs):
        assert config["output"] == dict(save_anomaly_maps=False, save_weights=False)
        assert config["method_config"]["uncertainty_normalization"] == (i > 0)
        assert config["method_config"]["synthetic_supervision"] == (i > 1)
        config["method_config"].pop("uncertainty_normalization")
        config["method_config"].pop("synthetic_supervision")
    assert configs[0] == configs[1] == configs[2]


def test_mcp_cosine_fit_predict_and_budget(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", 17*i) for i in range(2)]
    model = MaskedPrediction(TinyPSMAV2Backbone(), settings(
        steps=1, resolutions=[8], prediction_objective="cosine"))
    model.fit(paths, MethodContext("can", 0, 8, False, False))
    assert model.fit_feature_calls == 6  # two supports + two views x (target, masked)
    assert 0 <= model.history[-1]["loss"] <= 2
    result, score = model.predict(paths[0])
    assert np.isfinite(result).all() and result.min() >= -1e-6 and result.max() <= 2.000001
    assert np.isfinite(score)
    assert model.object_metadata()["prediction_objective"] == "cosine"
    assert model.object_metadata()["bank_sizes"] == {}
    with pytest.raises(ValueError):
        MaskedPrediction(TinyPSMAV2Backbone(), settings(
            prediction_objective="cosine", uncertainty_normalization=True))
