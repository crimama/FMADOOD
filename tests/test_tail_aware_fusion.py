from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.nn import functional as F

from fmad.methods import register_methods
from fmad.methods.reference_matching import ReferenceMatching
from fmad.methods.tail_aware_fusion import (
    BoundedFusion, TailAwareFusion, sample_native, select_negatives,
)
from fmad.protocol import MethodContext
from fmad.registry import METHOD_REGISTRY
from tests.test_reference_matching import TinyPairBackbone
from tests.test_psma_mr_v2 import _write_image


DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))]


def config(stage):
    return dict(variant="R1", feature_layers=[11, 23], resolutions=[8, 12],
                bank_tokens_per_source=4, query_chunk_size=4, steps=4,
                train_tokens=8, normal_candidates=64, fusion_chunk_size=31,
                learn_fusion=stage > 0, hard_normal_ranking=stage > 1, local_fusion=stage > 2)


@pytest.mark.parametrize("local,parameters", [(False, 1), (True, 65)])
def test_gate_zero_parity_bounds_agreement_and_gradients(local, parameters):
    gate = BoundedFusion(local, .5)
    values = torch.tensor([[1., 3.], [4., 2.], [2., 2.]])
    torch.testing.assert_close(gate(values), values.mean(-1), rtol=0, atol=0)
    assert sum(p.numel() for p in gate.parameters()) == parameters
    gate(values).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in gate.parameters())
    with torch.no_grad():
        if local:
            gate.network[-1].bias.fill_(100)
        else:
            gate.logit.fill_(100)
    weight = gate.weights(values)
    assert (weight >= .25).all() and (weight <= .75).all()
    assert gate(values)[2] == 2
    assert (abs(gate(values) - values.mean(-1)) <= .25 * abs(values[:, 0]-values[:, 1]) + 1e-6).all()


@pytest.mark.parametrize("device", DEVICES)
def test_native_sampling_matches_dense_bilinear_including_edges(device):
    grid = torch.arange(30, dtype=torch.float32, device=device).reshape(2, 3, 5) / 30
    shape = (13, 19)
    indices = np.arange(shape[0] * shape[1])
    expected = F.interpolate(grid[None], size=shape, mode="bilinear", align_corners=False)[0].flatten(1).T
    torch.testing.assert_close(sample_native(grid, indices, shape), expected, rtol=2e-6, atol=2e-6)


def test_hard_mining_selects_current_upper_tail_with_live_gradient():
    values = torch.arange(100, dtype=torch.float32, requires_grad=True)
    indices = select_negatives(values, hard=True, fraction=.05, count=128,
                               rng=np.random.default_rng(0))
    assert set(indices.tolist()) == set(range(95, 100))
    values[indices].sum().backward()
    assert values.grad[:95].sum() == 0 and values.grad[95:].sum() == 5
    uniform = select_negatives(values, hard=False, fraction=.05, count=50,
                               rng=np.random.default_rng(0))
    assert len(uniform.unique()) == 50 and (uniform < 95).any()


@pytest.mark.parametrize("device", DEVICES)
def test_tf0_is_exact_r1_and_zero_gate_matches_native_path(tmp_path, device):
    paths = [_write_image(tmp_path / f"{i}.png", i*17) for i in range(3)]
    context = MethodContext("can", 0, 8, False, False)
    backbone = TinyPairBackbone()
    backbone.device = device
    reference = ReferenceMatching(backbone, config(0))
    method = TailAwareFusion(backbone, config(0))
    reference.fit(paths, context)
    method.fit(paths, context)
    expected = reference.predict(paths[0])[0]
    np.testing.assert_array_equal(method.predict(paths[0])[0], expected)
    for resolution in method.resolutions:
        torch.testing.assert_close(method.scales[resolution], reference.scales[resolution], rtol=0, atol=0)
        assert method.banks[resolution]["owners"] == reference.banks[resolution]["owners"]
    method.components["learn_fusion"] = True
    method.gate = BoundedFusion(False, .5).to(device)
    np.testing.assert_allclose(method.predict(paths[0])[0], expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_fit_triplet_same_episodes_scales_frozen_backbone_and_finite_predictions(tmp_path, device):
    paths = [_write_image(tmp_path / f"{i}.png", i*17) for i in range(3)]
    context = MethodContext("can", 7, 8, False, False)
    methods = []
    for stage in (1, 2, 3):
        backbone = TinyPairBackbone()
        backbone.device = device
        original = copy.deepcopy(backbone.model.state_dict())
        method = TailAwareFusion(backbone, config(stage))
        emitted = []
        method._training_log_sink = lambda step, row: emitted.append((step, row))
        method.fit(paths, context)
        assert [step for step, _ in emitted] == list(range(4))
        assert len(method.history) == len(method.episodes) == 4
        assert all(np.isfinite(row["loss"]) for row in method.history)
        for name, value in original.items():
            torch.testing.assert_close(backbone.model.state_dict()[name], value, rtol=0, atol=0)
        for source in range(3):
            banks = method._source_excluded(source)
            assert all(source not in b["owners"] and len(b["owners"]) == 8 for b in banks.values())
        before = copy.deepcopy(method.gate.state_dict())
        actual, score = method.predict(paths[0])
        assert actual.shape == (12, 16) and np.isfinite(actual).all() and np.isfinite(score)
        assert method.records == {} and method.object_metadata()["dense_maps_accumulated"] is False
        assert method.object_metadata()["fit_backbone_calls"] == 3*2 + 4*4*2
        assert method.parameter_count == (65 if stage == 3 else 1)
        for name, value in before.items():
            torch.testing.assert_close(method.gate.state_dict()[name], value, rtol=0, atol=0)
        assert not any(key in vars(method) for key in ("values", "score_maps", "views", "images"))
        methods.append(method)
    assert methods[0].episodes == methods[1].episodes == methods[2].episodes
    for resolution in methods[0].resolutions:
        for method in methods[1:]:
            torch.testing.assert_close(method.scales[resolution], methods[0].scales[resolution], rtol=0, atol=0)


@pytest.mark.parametrize("bad", [dict(learn_fusion="yes"), dict(hard_normal_ranking=True),
                                  dict(local_fusion=True), dict(fusion_bound=1),
                                  dict(normal_tail_fraction=float("nan")), dict(normal_candidates=0),
                                  dict(steps=True), dict(margin=-1), dict(pair_weight=float("inf")),
                                  dict(variant="R2")])
def test_rejects_invalid_config(bad):
    with pytest.raises(ValueError):
        TailAwareFusion(TinyPairBackbone(), {**config(0), **bad})


def test_config_ablation_boundaries_and_registry():
    register_methods("tail_aware_fusion")
    assert METHOD_REGISTRY.get("tail_aware_fusion") is TailAwareFusion
    root = Path(__file__).resolve().parents[1]
    configs = [yaml.safe_load((root / f"configs/methods/tail_aware_fusion_TF{i}.yaml").read_text())
               for i in range(4)]
    for i, item in enumerate(configs):
        assert item["output"] == {"save_anomaly_maps": False, "save_weights": False}
        assert item["backbone_config"]["feature_layers"] == [11, 23]
        assert item["method_config"]["steps"] == 64
        if i:
            previous = copy.deepcopy(configs[i-1])
            key = ("learn_fusion", "hard_normal_ranking", "local_fusion")[i-1]
            previous["method_config"][key] = True
            assert previous == item


@pytest.mark.skipif(os.environ.get("FMAD_REAL_BACKBONE_SMOKE") != "1",
                    reason="explicit cached-backbone CUDA smoke only")
def test_real_cached_dinov2_full_resolution_fit_and_predict():
    from fmad.backbones.dinov2 import DINOv2Backbone
    paths = [str(p) for p in sorted(Path("/data/mvtec_ad_2/can/train/good").glob("*.png"))[:3]]
    assert len(paths) == 3 and torch.cuda.is_available()
    backbone = DINOv2Backbone("dinov2_vitl14", "cuda:0", feature_layers=[11, 23])
    context = MethodContext("can", 0, 448, False, False)
    root = Path(__file__).resolve().parents[1]
    methods = []
    for stage in range(4):
        settings = yaml.safe_load((root / f"configs/methods/tail_aware_fusion_TF{stage}.yaml").read_text())["method_config"]
        settings["steps"] = 2
        method = TailAwareFusion(backbone, settings)
        method.fit(paths, context)
        actual, score = method.predict(paths[0])
        assert np.isfinite(actual).all() and np.ptp(actual) > 0 and np.isfinite(score)
        if stage == 0:
            reference = ReferenceMatching(backbone, settings)
            reference.fit(paths, context)
            np.testing.assert_array_equal(actual, reference.predict(paths[0])[0])
        else:
            assert len(method.history) == 2 and all(np.isfinite(row["loss"]) for row in method.history)
        print(f"real CUDA TF{stage}: finite/nonconstant; updates={len(method.history)}; parameters={method.parameter_count}")
        methods.append(method)
    assert methods[1].episodes == methods[2].episodes == methods[3].episodes
    for resolution in methods[0].resolutions:
        for method in methods[1:]:
            torch.testing.assert_close(method.scales[resolution], methods[0].scales[resolution], rtol=0, atol=0)
