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
from fmad.methods.correspondence_guard import (
    CorrespondenceGuard, correspondence_cost, guarded_distances,
)
from fmad.methods.mode_reference import ModeReference
from fmad.protocol import MethodContext
from fmad.registry import METHOD_REGISTRY
from tests.test_mode_reference import config as mc_config
from tests.test_psma_mr_v2 import _write_image
from tests.test_query_reference import DEVICES
from tests.test_reference_matching import TinyPairBackbone


ROOT = Path(__file__).resolve().parents[1]


def config(stage):
    return {**mc_config(min(stage, 1)), "mode_guard": stage > 1,
            "identity_guard": stage > 2, "guard_weight": 1.}


@pytest.mark.parametrize("device", DEVICES)
def test_nested_cost_equation_and_noncontiguous_mode_labels(device):
    distances = torch.tensor([[.2, .3, .5, .7], [.8, .6, .3, .2]], device=device)
    labels = torch.tensor([3, 3, 8, 8], device=device)
    mode = correspondence_cost(distances, labels, mode_guard=True, identity_guard=False, weight=1.)
    both = correspondence_cost(distances, labels, mode_guard=True, identity_guard=True, weight=1.)
    torch.testing.assert_close(mode, distances.new_tensor([[0., 0., .3, .3], [.4, .4, 0., 0.]]))
    torch.testing.assert_close(both, distances - distances.amin(1, keepdim=True))
    assert bool((both >= mode - 1e-7).all())
    for guard, weight in [(False, 1.), (True, 0.)]:
        assert correspondence_cost(distances, labels, mode_guard=guard, identity_guard=False, weight=weight).count_nonzero() == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("same_mode", [False, True])
def test_cost_limits_reference_switch_without_altering_candidate(device, same_mode):
    # Constructed geometry, not a benchmark or an anomaly-label assertion.
    query = torch.tensor([[1., 0.]], device=device)
    a = torch.tensor([.3, .6], device=device)
    b = torch.tensor([.28, .05], device=device)
    anchor = torch.stack([a.cos(), a.sin()], 1)
    candidate = torch.stack([b.cos(), b.sin()], 1)
    labels = torch.tensor([0, 0 if same_mode else 1], device=device)
    old = candidate.clone()
    raw, _ = guarded_distances(query, anchor, candidate, labels, chunk=1,
                              mode_guard=False, identity_guard=False, weight=1.)
    mode, _ = guarded_distances(query, anchor, candidate, labels, chunk=1,
                               mode_guard=True, identity_guard=False, weight=1.)
    both, stats = guarded_distances(query, anchor, candidate, labels, chunk=1,
                                   mode_guard=True, identity_guard=True, weight=1.)
    assert float(both[0]) > float(raw[0])
    if same_mode:
        torch.testing.assert_close(mode, raw, rtol=0, atol=0)
    else:
        torch.testing.assert_close(mode, both, rtol=0, atol=0)
    assert stats["changed_choice_tokens"] == stats["mc1_reference_switch_tokens"] == 1
    assert stats["guarded_reference_switch_tokens"] == 0
    torch.testing.assert_close(candidate, old, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_chunked_full_bank_bounds_identity_and_empty_query(device):
    gen = torch.Generator().manual_seed(18)
    q = F.normalize(torch.randn(19, 8, generator=gen).to(device), dim=-1)
    anchor = F.normalize(torch.randn(17, 8, generator=gen).to(device), dim=-1)
    candidate = F.normalize(anchor + .08 * torch.randn(17, 8, generator=gen).to(device), dim=-1)
    labels = torch.arange(17, device=device) % 3
    d0, d1 = 1-q@anchor.T, 1-q@candidate.T
    score_rows = []
    for mode, identity in [(False, False), (True, False), (True, True)]:
        penalty = correspondence_cost(d0, labels, mode_guard=mode, identity_guard=identity, weight=1.)
        expected = (d1 + penalty).amin(1)
        for chunk in [1, 7, 128]:
            actual, stats = guarded_distances(q, anchor, candidate, labels, chunk=chunk,
                                             mode_guard=mode, identity_guard=identity, weight=1.)
            torch.testing.assert_close(actual, expected, atol=3e-7, rtol=1e-6)
            assert stats["tokens"] == len(q)
            assert stats["lower_bound_violations"] == stats["anchor_bound_violations"] == 0
            assert stats["removed_mc1_benefit_sum"] <= stats["mc1_benefit_sum"] + 1e-6
        score_rows.append(expected)
    assert bool((score_rows[1] >= score_rows[0]-1e-7).all())
    assert bool((score_rows[2] >= score_rows[1]-1e-7).all())
    for identity in [False, True]:
        unchanged, _ = guarded_distances(q, anchor, anchor, labels, chunk=7,
                                        mode_guard=True, identity_guard=identity, weight=1.)
        torch.testing.assert_close(unchanged, d0.amin(1), atol=2e-7, rtol=1e-6)
    empty, stats = guarded_distances(q[:0], anchor, candidate, labels, chunk=7,
                                    mode_guard=True, identity_guard=True, weight=1.)
    assert empty.numel() == 0 and all(v == 0 for v in stats.values())


@pytest.mark.parametrize("device", DEVICES)
def test_control_parity_candidate_stats_and_state_isolation(tmp_path, device):
    paths = [_write_image(tmp_path/f"{i}.png", i*27) for i in range(4)]
    ctx = MethodContext("can", 7, 8, False, False)
    backbone = TinyPairBackbone(); backbone.device = device
    weights = copy.deepcopy(backbone.model.state_dict())
    expected = []
    for stage in [0, 1]:
        control = ModeReference(backbone, mc_config(stage)); control.fit(paths, ctx)
        expected.append(control.predict(paths[0])[0])
    outputs, diagnostics = [], []
    for stage in range(4):
        method = CorrespondenceGuard(backbone, config(stage)); method.fit(paths, ctx)
        result, score = method.predict(paths[0]); outputs.append(result)
        diagnostics.append(method.object_metadata())
        if stage < 2:
            np.testing.assert_array_equal(result, expected[stage])
        assert np.isfinite(result).all() and np.isfinite(score)
        method.predict(paths[1]); repeat, repeat_score = method.predict(paths[0])
        np.testing.assert_array_equal(result, repeat); assert score == repeat_score
        assert method.records == {} and method.history == []
        assert method.parameter_count == 0 and not method.object_metadata()["persistent_adapted_state"]
        method.fit(paths, ctx)
        assert all(v == 0 for v in method.guard_stats.values())
    for stage in [2, 3]:
        assert np.all(outputs[stage] >= outputs[stage-1]-2e-5)
        for key in ["normal_geometry", "robust_scales", "bank_sizes", "translation_diagnostics_including_warmup", "mode_diagnostics_including_warmup"]:
            assert diagnostics[stage][key] == diagnostics[1][key]
        assert diagnostics[stage]["correspondence_guard_diagnostics_including_warmup"]["tokens"] > 0
    for key, value in weights.items():
        torch.testing.assert_close(backbone.model.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("override", [
    {"mode_guard": 1}, {"identity_guard": "yes"}, {"mode_guard": True},
    {"identity_guard": True}, {"guard_weight": -1}, {"guard_weight": True},
    {"guard_weight": float("nan")}, {"guard_weight": float("inf")},
    {"normal_subspace": True},
])
def test_invalid_settings(override):
    with pytest.raises(ValueError):
        CorrespondenceGuard(TinyPairBackbone(), {**config(0), **override})


def test_registry_and_single_switch_configs():
    register_methods("correspondence_guard")
    assert METHOD_REGISTRY.get("correspondence_guard") is CorrespondenceGuard
    configs = [yaml.safe_load((ROOT/f"configs/methods/correspondence_guard_CG{i}.yaml").read_text()) for i in range(4)]
    control = yaml.safe_load((ROOT/"configs/methods/mode_reference_MC0.yaml").read_text())
    for i, c in enumerate(configs):
        for key in ["dataset", "backbone", "seed", "device", "dataset_config", "backbone_config", "protocol", "evaluation", "output"]:
            assert c[key] == control[key]
        if i:
            previous = copy.deepcopy(configs[i-1])
            previous["method_config"][("mode_conditioning", "mode_guard", "identity_guard")[i-1]] = True
            assert previous == c


@pytest.mark.skipif(os.environ.get("FMAD_REAL_BACKBONE_SMOKE") != "1", reason="explicit cached-backbone smoke only")
def test_real_cached_backbone_full_resolution_guard():
    from fmad.backbones.dinov2 import DINOv2Backbone
    obj = os.environ.get("FMAD_SMOKE_OBJECT", "can")
    files = sorted(Path(f"/data/mvtec_ad_2/{obj}/train/good").glob("*.png"))
    paths = [str(p) for p in files[:16]]; query = str(files[16])
    assert len(paths) == 16 and torch.cuda.is_available()
    torch.cuda.reset_peak_memory_stats()
    backbone = DINOv2Backbone("dinov2_vitl14", "cuda:0", feature_layers=[11, 23])
    ctx = MethodContext(obj, 0, 448, False, False)
    expected = []
    for i in [0, 1]:
        c = yaml.safe_load((ROOT/f"configs/methods/mode_reference_MC{i}.yaml").read_text())["method_config"]
        method = ModeReference(backbone, c); method.fit(paths, ctx)
        expected.append(method.predict(query)[0]); del method
    previous, diagnostics = None, []
    for i in range(4):
        c = yaml.safe_load((ROOT/f"configs/methods/correspondence_guard_CG{i}.yaml").read_text())["method_config"]
        method = CorrespondenceGuard(backbone, c); method.fit(paths, ctx)
        result, score = method.predict(query); meta = method.object_metadata(); diagnostics.append(meta)
        assert np.isfinite(result).all() and np.ptp(result)>0 and np.isfinite(score)
        assert all(size == 4096 for size in meta["bank_sizes"].values())
        if i < 2:
            np.testing.assert_array_equal(result, expected[i])
        else:
            assert np.all(result >= previous-2e-5)
            for key in ["normal_geometry", "robust_scales", "mode_diagnostics_including_warmup", "translation_diagnostics_including_warmup"]:
                assert meta[key] == diagnostics[1][key]
            g = meta["correspondence_guard_diagnostics_including_warmup"]
            assert g["tokens"]>0 and g["lower_bound_violations"] == g["anchor_bound_violations"] == 0
        print(f"real CUDA {obj} CG{i}: finite, control/candidate invariants pass; {method.guard_stats}")
        previous = result; del method
    assert all(not p.requires_grad for p in backbone._wrapper.model.parameters())
    print(f"peak_reserved_MiB={torch.cuda.max_memory_reserved()/2**20:.1f}")
