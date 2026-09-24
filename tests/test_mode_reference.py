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
from fmad.methods.mode_reference import (
    ModeReference, mode_translations, normal_basis, normal_gate_scales,
    normal_modes, orthogonal_norm,
)
from fmad.methods.query_reference import QueryReference, fit_translation
from fmad.protocol import MethodContext
from fmad.registry import METHOD_REGISTRY
from tests.test_psma_mr_v2 import _write_image
from tests.test_query_reference import DEVICES, config as qr_config
from tests.test_reference_matching import TinyPairBackbone


ROOT = Path(__file__).resolve().parents[1]


def config(stage):
    return {**qr_config(1), "mode_conditioning": stage > 0, "normal_subspace": stage > 1,
            "orthogonal_gate": stage > 2, "mode_count": 2, "mode_kmeans_iters": 3,
            "mode_subspace_rank": 2, "min_mode_tokens": 2, "mode_prior_tokens": 4.}


def fit_modes(residuals, modes, geometry, **overrides):
    settings = dict(ridge=1., max_norm=.1, min_tokens=2, min_mode_tokens=2,
                    prior_tokens=4., constrain=False, gate=False)
    return mode_translations(residuals, modes, geometry, **{**settings, **overrides})


@pytest.mark.parametrize("device", DEVICES)
def test_normal_clusters_and_subspaces_deterministic_and_rng_isolated(device):
    generator = torch.Generator().manual_seed(22)
    values = torch.randn(40, 6, generator=generator).to(device)
    reference = F.normalize(values, dim=-1)
    labels, centers = normal_modes(reference, 4, 5)
    other_labels, other_centers = normal_modes(reference, 4, 5)
    torch.testing.assert_close(labels, other_labels, rtol=0, atol=0)
    torch.testing.assert_close(centers, other_centers, rtol=0, atol=0)
    assert labels.shape == (40,) and torch.isfinite(centers).all()
    one_labels, one_center = normal_modes(reference[:1].repeat(8, 1), 4, 5)
    assert len(one_center) == 1 and one_labels.count_nonzero() == 0
    plane = values.new_zeros(40, 6)
    plane[:, :2] = values[:, :2]
    state = torch.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    basis = normal_basis(plane, 2, 123)
    assert torch.equal(torch.get_rng_state(), state)
    if cuda_state is not None:
        assert torch.equal(torch.cuda.get_rng_state(), cuda_state)
    torch.testing.assert_close(basis.T @ basis, torch.eye(2, device=device), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close((plane @ basis) @ basis.T, plane, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(normal_basis(plane, 2, 123), basis, rtol=0, atol=0)
    assert normal_basis(plane[:1], 2, 0).shape == (6, 0)
    assert normal_basis(torch.zeros_like(plane), 2, 0).shape == (6, 0)


@pytest.mark.parametrize("device", DEVICES)
def test_mode_innovations_break_global_cancellation_and_shrink_to_global(device):
    # Structural unit example, not an anomaly benchmark: opposite modes cancel
    # in QR1, while a normal-mode deformation can represent both changes.
    values = torch.tensor([[.06, 0., 0.]] * 8 + [[-.06, 0., 0.]] * 8, device=device)
    modes = torch.tensor([0] * 8 + [1] * 8, device=device)
    geometry = {"centers": values.new_zeros(2, 3)}
    global_delta, _ = fit_translation(values, keep_fraction=1., ridge=1., max_norm=.1, min_tokens=2)
    deltas, stats, extra = fit_modes(values, modes, geometry)
    torch.testing.assert_close(global_delta, torch.zeros_like(global_delta), atol=1e-8, rtol=0)
    expected = values.new_tensor([[.02, 0., 0.], [-.02, 0., 0.]])
    torch.testing.assert_close(deltas, expected)
    assert extra["active_modes"] == 2 and not stats["fallback"]
    assert (values - deltas[modes]).square().sum() < (values - global_delta).square().sum()
    sparse, _, extra = fit_modes(values, modes, geometry, min_mode_tokens=9)
    torch.testing.assert_close(sparse, global_delta.expand_as(sparse))
    assert extra["fallback_modes"] == 2
    uniform = values.new_tensor([[.04, 0., 0.]]).repeat(16, 1)
    same, _, _ = fit_modes(uniform, modes, geometry)
    torch.testing.assert_close(same, uniform[:2] / 2)
    clipped, _, extra = fit_modes(values * 100, modes, geometry)
    assert float(clipped.norm(dim=1).max()) <= .100001 and extra["clipped_modes"] == 2
    empty, stats, _ = fit_modes(values[:0], modes[:0], geometry)
    assert stats["fallback"] and empty.count_nonzero() == 0


@pytest.mark.parametrize("device", DEVICES)
def test_projection_and_gate_are_direction_sensitive_not_raw_magnitude_trim(device):
    values = torch.tensor([[.04, .06, 0.]] * 8 + [[-.04, -.06, 0.]] * 8, device=device)
    modes = torch.tensor([0] * 8 + [1] * 8, device=device)
    basis = values.new_tensor([[1.], [0.], [0.]])
    geometry = dict(centers=values.new_zeros(2, 3), bases=[basis, basis], scales=values.new_tensor([.1, .1]))
    projected, _, extra = fit_modes(values, modes, geometry, constrain=True)
    assert projected[:, 0].abs().min() > 0 and projected[:, 1:].count_nonzero() == 0
    assert extra["innovation_after_norm_sum"] < extra["innovation_before_norm_sum"]
    probes = values.new_tensor([[1., 0., 0.], [0., .2, 0.]])
    norms = orthogonal_norm(probes, basis)
    torch.testing.assert_close(norms, values.new_tensor([0., .2]))
    weights = 1 / (1 + (norms / .1).square())
    assert float(weights[0]) == 1 and float(weights[1]) < .25
    _, _, gated = fit_modes(values, modes, geometry, constrain=True, gate=True)
    assert 0 < gated["evidence_mass"] < len(values)
    severe = values * 10
    _, _, rejected = fit_modes(severe, modes, geometry, constrain=True, gate=True)
    assert rejected["fallback_modes"] == 2 and rejected["low_gate_tokens"] == len(values)


@pytest.mark.parametrize("device", DEVICES)
def test_normal_gate_excludes_same_support_and_uses_matched_mode(device):
    angles = torch.tensor([0., .03, .6, .65], device=device)
    reference = torch.stack([angles.cos(), angles.sin()], dim=1)
    labels = torch.tensor([0, 0, 1, 1], device=device)
    bases = [reference.new_empty(2, 0), reference.new_empty(2, 0)]
    scales, counts = normal_gate_scales(reference, [0, 0, 1, 1], labels, bases, 2, .9, 1)
    selected = torch.tensor([2, 2, 1, 1], device=device)
    residuals = (reference - reference[selected]).norm(dim=1)
    expected = torch.stack([torch.quantile(residuals[labels[selected] == k], .9) for k in range(2)])
    torch.testing.assert_close(scales, expected)
    assert counts == [2, 2] and float(scales.min()) > .5
    for chunk in (1, 4):
        actual, other_counts = normal_gate_scales(reference, [0, 0, 1, 1], labels, bases, chunk, .9, 1)
        torch.testing.assert_close(actual, scales)
        assert other_counts == counts
    with pytest.raises(ValueError, match="two independent"):
        normal_gate_scales(reference, [0] * 4, labels, bases, 2, .9, 1)


@pytest.mark.parametrize("device", DEVICES)
def test_qr1_parity_no_bank_mutation_or_query_state_and_matched_geometry(tmp_path, device):
    paths = [_write_image(tmp_path / f"{i}.png", i * 27) for i in range(4)]
    context = MethodContext("can", 7, 8, False, False)
    backbone = TinyPairBackbone()
    backbone.device = device
    weights = copy.deepcopy(backbone.model.state_dict())
    baseline = QueryReference(backbone, qr_config(1)); baseline.fit(paths, context)
    expected, expected_score = baseline.predict(paths[0])
    previous = None
    for stage in range(4):
        method = ModeReference(backbone, config(stage)); method.fit(paths, context)
        actual, score = method.predict(paths[0]); method.predict(paths[1])
        repeat, repeat_score = method.predict(paths[0])
        np.testing.assert_array_equal(actual, repeat)
        assert score == repeat_score and np.isfinite(actual).all()
        if stage == 0:
            np.testing.assert_array_equal(actual, expected)
            assert score == expected_score
        else:
            assert method.mode_stats["fits"] == 12
            assert method.translation_stats["fallback_fits"] == 0
        for resolution in method.resolutions:
            torch.testing.assert_close(method.scales[resolution], baseline.scales[resolution], rtol=0, atol=0)
            for bank, original in zip(method.banks[resolution]["layers"], baseline.banks[resolution]["layers"]):
                torch.testing.assert_close(bank, original, rtol=0, atol=0)
            if stage > 1:
                for g, old in zip(method.banks[resolution]["mode_geometry"], previous.banks[resolution]["mode_geometry"]):
                    torch.testing.assert_close(g["labels"], old["labels"], rtol=0, atol=0)
                    torch.testing.assert_close(g["centers"], old["centers"], rtol=0, atol=0)
                    if stage == 3:
                        for b, prior in zip(g["bases"], old["bases"]):
                            torch.testing.assert_close(b, prior, rtol=0, atol=0)
        assert method.records == {} and method.history == []
        assert all(isinstance(v, (int, float)) for v in method.mode_stats.values())
        meta = method.object_metadata()
        assert not meta["dense_maps_accumulated"] and not meta["query_labels_used"]
        assert not meta["persistent_adapted_state"] and meta["trainable_parameters"] == 0
        previous = method
    previous.fit(paths, context)
    assert previous.mode_stats["fits"] == previous.translation_stats["fits"] == 0
    for name, value in weights.items():
        torch.testing.assert_close(backbone.model.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("override", [
    {"mode_conditioning": 1}, {"normal_subspace": True}, {"orthogonal_gate": True},
    {"query_adaptation": False}, {"robust_trim": True}, {"spatial_crossfit": True},
    {"mode_count": 0}, {"mode_count": 2.5}, {"mode_kmeans_iters": 0},
    {"mode_subspace_rank": 0}, {"min_mode_tokens": 0}, {"mode_prior_tokens": 0},
    {"mode_prior_tokens": float("nan")}, {"normal_gate_quantile": 1.1},
])
def test_invalid_settings(override):
    with pytest.raises(ValueError):
        ModeReference(TinyPairBackbone(), {**config(0), **override})


def test_registry_configs_and_one_switch_per_stage():
    register_methods("mode_reference")
    assert METHOD_REGISTRY.get("mode_reference") is ModeReference
    baseline = yaml.safe_load((ROOT / "configs/methods/query_reference_QR1.yaml").read_text())
    configs = [yaml.safe_load((ROOT / f"configs/methods/mode_reference_MC{i}.yaml").read_text()) for i in range(4)]
    for stage, settings in enumerate(configs):
        for key, value in baseline.items():
            if key == "method_config":
                assert all(settings[key][k] == v for k, v in value.items())
            elif key != "method":
                assert settings[key] == value
        if stage:
            previous = copy.deepcopy(configs[stage - 1])
            previous["method_config"][("mode_conditioning", "normal_subspace", "orthogonal_gate")[stage - 1]] = True
            assert previous == settings


@pytest.mark.skipif(os.environ.get("FMAD_REAL_BACKBONE_SMOKE") != "1",
                    reason="explicit real cached-backbone smoke only")
def test_real_cached_backbone_full_resolution_modes():
    from fmad.backbones.dinov2 import DINOv2Backbone
    paths = [str(p) for p in sorted(Path("/data/mvtec_ad_2/can/train/good").glob("*.png"))[:16]]
    assert len(paths) == 16 and torch.cuda.is_available()
    backbone = DINOv2Backbone("dinov2_vitl14", "cuda:0", feature_layers=[11, 23])
    context = MethodContext("can", 0, 448, False, False)
    settings = yaml.safe_load((ROOT / "configs/methods/query_reference_QR1.yaml").read_text())["method_config"]
    baseline = QueryReference(backbone, settings); baseline.fit(paths, context)
    expected = baseline.predict(paths[-1])[0]
    previous_geometry = None
    for stage in range(4):
        settings = yaml.safe_load((ROOT / f"configs/methods/mode_reference_MC{stage}.yaml").read_text())["method_config"]
        method = ModeReference(backbone, settings); method.fit(paths, context)
        actual, score = method.predict(paths[-1])
        assert np.isfinite(actual).all() and np.ptp(actual) > 0 and np.isfinite(score)
        metadata = method.object_metadata()
        assert all(size == 4096 for size in metadata["bank_sizes"].values())
        if stage == 0:
            np.testing.assert_array_equal(actual, expected)
        else:
            assert not np.array_equal(actual, expected)
            assert method.mode_stats["active_modes"] > 0
            assert method.mode_stats["mode_global_difference_norm_sum"] > 0
            assert method.translation_stats["fallback_fits"] == 0
            if stage >= 2:
                assert method.mode_stats["innovation_after_norm_sum"] <= method.mode_stats["innovation_before_norm_sum"] + 1e-6
            if stage == 3:
                assert 0 < method.mode_stats["evidence_mass"] < method.mode_stats["query_tokens"]
        geometry = metadata["normal_geometry"]
        if stage >= 2:
            for resolution, layers in geometry.items():
                for current, old in zip(layers, previous_geometry[resolution]):
                    assert current["labels_sha256"] == old["labels_sha256"]
                    assert current["centers_sha256"] == old["centers_sha256"]
                    if stage == 3:
                        assert current["normal_basis_sha256"] == old["normal_basis_sha256"]
        previous_geometry = geometry
        assert all(not p.requires_grad for p in backbone._wrapper.model.parameters())
        print(f"real CUDA MC{stage}: finite/nonconstant; {method.mode_stats}")
