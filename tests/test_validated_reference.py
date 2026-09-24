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
from fmad.methods.mode_reference import ModeReference
from fmad.methods.query_reference import fit_translation, spatial_folds
from fmad.methods.reference_matching import _distance_chunks
from fmad.methods.validated_reference import (
    ValidatedReference, bounded_deltas, context_innovations, single_mode_gain,
    transfer_admission,
)
from fmad.protocol import MethodContext
from fmad.registry import METHOD_REGISTRY
from tests.test_mode_reference import config as mc_config
from tests.test_psma_mr_v2 import _write_image
from tests.test_query_reference import DEVICES
from tests.test_reference_matching import TinyPairBackbone


ROOT = Path(__file__).resolve().parents[1]


def config(stage):
    return {**mc_config(min(stage, 1)), "innovation_crossfit": stage > 1,
            "transfer_validation": stage > 2, "validation_min_tokens": 2,
            "validation_tokens_per_mode": 4, "validation_min_blocks": 2}


def innovations(residuals, modes, **overrides):
    kwargs = dict(ridge=1., prior=4., minimum=2, min_mode=2)
    return context_innovations(residuals, modes, 2, **{**kwargs, **overrides})


@pytest.mark.parametrize("device", DEVICES)
def test_innovation_equation_sparse_fallback_and_bounds(device):
    values = torch.tensor([[.06, 0.]] * 8 + [[-.06, 0.]] * 8, device=device)
    modes = torch.tensor([0] * 8 + [1] * 8, device=device)
    actual, valid = innovations(values, modes)
    torch.testing.assert_close(actual, values.new_tensor([[.02, 0.], [-.02, 0.]]))
    assert valid.all()
    sparse, valid = innovations(values, modes, min_mode=9)
    assert sparse.count_nonzero() == 0 and not valid.any()
    empty, valid = innovations(values[:0], modes[:0])
    assert empty.count_nonzero() == 0 and not valid.any()
    delta = bounded_deltas(values.new_tensor([.02, .01]), actual * 100, .1)
    assert float(delta.norm(dim=1).max()) <= .100001


@pytest.mark.parametrize("device", DEVICES)
def test_actual_cosine_validation_matches_full_bank_counterfactual(device):
    gen = torch.Generator().manual_seed(83)
    reference = F.normalize(torch.randn(20, 6, generator=gen).to(device), dim=-1)
    query = F.normalize(torch.randn(9, 6, generator=gen).to(device), dim=-1)
    labels = torch.arange(20, device=device) % 2
    baseline = reference.new_tensor([.02, -.01, .01, 0., 0., 0.])
    candidate = baseline + reference.new_tensor([.01, .04, 0., 0., 0., 0.])
    global_reference = F.normalize(reference + baseline, dim=-1)
    changed = global_reference.clone()
    changed[labels == 0] = F.normalize(reference[labels == 0] + candidate, dim=-1)
    before = _distance_chunks(query, global_reference, 4)
    after = _distance_chunks(query, changed, 4)
    expected = float((before - after).mean() / before.mean().clamp_min(1e-6))
    actual = single_mode_gain(query, reference, labels, global_reference, candidate, 0, 4)
    assert actual == pytest.approx(expected, abs=1e-6)
    one = torch.zeros_like(labels)
    assert np.isfinite(single_mode_gain(query, reference, one, global_reference, candidate, 0, 4))


def toy_evidence(device, local_only=False):
    reference = torch.tensor([[1., 0., 0.], [0., 1., 0.]], device=device)
    modes = torch.arange(64, device=device) % 2
    folds = list(spatial_folds((8, 8), 2, 0, device))
    offsets = reference.new_zeros(64, 3)
    offsets[:, 2] = torch.where(modes == 0, .08, -.08)
    if local_only:
        offsets[~folds[0][0]] = 0
    query = F.normalize(reference[modes] + offsets, dim=-1)
    residuals = query - reference[modes]
    baseline, _ = fit_translation(residuals, keep_fraction=1., ridge=1., max_norm=.1, min_tokens=2)
    geometry = dict(centers=reference, labels=torch.arange(2, device=device))
    return query, residuals, modes, reference, geometry, baseline, folds


def admit(data, context, **overrides):
    query, residuals, modes, reference, geometry, baseline, folds = data
    options = dict(ridge=1., prior=4., minimum=2, min_mode=2, max_norm=.1,
                   chunk=8, min_witness=2, witness_cap=4, min_blocks=2, min_gain=.01)
    eligible = torch.ones(2, device=query.device, dtype=torch.bool)
    return transfer_admission(query, residuals, modes, reference, geometry, baseline,
                              context, folds, eligible, **{**options, **overrides})


@pytest.mark.parametrize("device", DEVICES)
def test_replicated_changes_pass_but_local_only_changes_do_not(device):
    # Structural constructed example, not a benchmark or normality guarantee.
    shared = toy_evidence(device)
    local = toy_evidence(device, local_only=True)
    for _, context in shared[-1]:
        accepted, stats = admit(shared, context)
        assert accepted.all() and stats["successful_pairs"] >= 4
        rejected, _ = admit(local, context)
        assert not rejected.any()
        assert stats["witness_tokens"] <= 3 * 2 * 4
    accepted, stats = admit(shared, shared[-1][0][1], min_witness=100)
    assert not accepted.any() and stats["insufficient_modes"] == 2


@pytest.mark.parametrize("device", DEVICES)
def test_exclusion_for_innovation_and_validation_conditional_on_fixed_QR1(device):
    data = toy_evidence(device)
    for scored, context in data[-1]:
        expected, valid = innovations(data[1][context], data[2][context])
        accepted, stats = admit(data, context)
        changed = list(data)
        changed[0] = data[0].clone(); changed[0][scored] += 50
        changed[1] = data[1].clone(); changed[1][scored] -= 100
        actual, other_valid = innovations(changed[1][context], changed[2][context])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(other_valid, valid)
        other_accepted, other_stats = admit(changed, context)
        assert torch.equal(accepted, other_accepted) and stats == other_stats


@pytest.mark.parametrize("device", DEVICES)
def test_control_parity_no_state_bank_or_backbone_mutation(tmp_path, device):
    paths = [_write_image(tmp_path / f"{i}.png", i * 27) for i in range(4)]
    context = MethodContext("can", 7, 8, False, False)
    backbone = TinyPairBackbone(); backbone.device = device
    weights = copy.deepcopy(backbone.model.state_dict())
    for stage in range(4):
        baseline = ModeReference(backbone, mc_config(min(stage, 1)))
        baseline.fit(paths, context)
        expected, expected_score = baseline.predict(paths[0])
        method = ValidatedReference(backbone, config(stage)); method.fit(paths, context)
        result, score = method.predict(paths[0]); method.predict(paths[1])
        repeat, repeat_score = method.predict(paths[0])
        np.testing.assert_array_equal(result, repeat)
        assert score == repeat_score and np.isfinite(result).all()
        if stage < 2:
            np.testing.assert_array_equal(result, expected)
            assert score == expected_score
        else:
            assert method.validation_stats["score_folds"] > 0
            assert method.mode_stats["fits"] == 4 * method.translation_stats["fits"]
            v = method.validation_stats
            assert v["accepted_modes"] + v["rejected_modes"] == v["proposed_modes"]
        for resolution in method.resolutions:
            torch.testing.assert_close(method.scales[resolution], baseline.scales[resolution], rtol=0, atol=0)
            for bank, old in zip(method.banks[resolution]["layers"], baseline.banks[resolution]["layers"]):
                torch.testing.assert_close(bank, old, rtol=0, atol=0)
        meta = method.object_metadata()
        assert not meta["query_labels_used"] and not meta["dense_maps_accumulated"]
        assert meta["trainable_parameters"] == 0 and not meta["persistent_adapted_state"]
        assert method.records == {} and method.history == []
        method.fit(paths, context)
        assert all(value == 0 for value in method.validation_stats.values())
    for key, value in weights.items():
        torch.testing.assert_close(backbone.model.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("override", [
    {"innovation_crossfit": 1}, {"transfer_validation": "yes"},
    {"innovation_crossfit": True}, {"transfer_validation": True},
    {"normal_subspace": True}, {"validation_min_tokens": 0},
    {"validation_tokens_per_mode": 1}, {"validation_min_blocks": 4},
    {"validation_min_blocks": 1}, {"validation_min_gain": 0},
    {"validation_min_gain": float("nan")},
])
def test_invalid_settings(override):
    with pytest.raises(ValueError):
        ValidatedReference(TinyPairBackbone(), {**config(0), **override})


def test_registry_configs_and_sequential_switches():
    register_methods("validated_reference")
    assert METHOD_REGISTRY.get("validated_reference") is ValidatedReference
    configs = [yaml.safe_load((ROOT / f"configs/methods/validated_reference_VR{i}.yaml").read_text()) for i in range(4)]
    baseline = yaml.safe_load((ROOT / "configs/methods/mode_reference_MC0.yaml").read_text())
    for stage, settings in enumerate(configs):
        for key in ("dataset", "backbone", "seed", "device", "dataset_config", "backbone_config", "protocol", "evaluation", "output"):
            assert settings[key] == baseline[key]
        assert not settings["method_config"]["normal_subspace"] and not settings["method_config"]["orthogonal_gate"]
        if stage:
            previous = copy.deepcopy(configs[stage - 1])
            previous["method_config"][("mode_conditioning", "innovation_crossfit", "transfer_validation")[stage - 1]] = True
            assert previous == settings


@pytest.mark.skipif(os.environ.get("FMAD_REAL_BACKBONE_SMOKE") != "1", reason="explicit real cached-backbone smoke only")
def test_real_cached_backbone_full_resolution_validation():
    from fmad.backbones.dinov2 import DINOv2Backbone
    files = sorted(Path("/data/mvtec_ad_2/can/train/good").glob("*.png"))
    paths = [str(p) for p in files[:16]]; query = str(files[16])
    assert len(paths) == 16 and torch.cuda.is_available()
    backbone = DINOv2Backbone("dinov2_vitl14", "cuda:0", feature_layers=[11, 23])
    context = MethodContext("can", 0, 448, False, False)
    expected = []
    for stage in range(2):
        settings = yaml.safe_load((ROOT / f"configs/methods/mode_reference_MC{stage}.yaml").read_text())["method_config"]
        method = ModeReference(backbone, settings); method.fit(paths, context)
        expected.append(method.predict(query)[0])
    geometries = None
    for stage in range(4):
        settings = yaml.safe_load((ROOT / f"configs/methods/validated_reference_VR{stage}.yaml").read_text())["method_config"]
        method = ValidatedReference(backbone, settings); method.fit(paths, context)
        actual, score = method.predict(query)
        assert np.isfinite(actual).all() and np.ptp(actual) > 0 and np.isfinite(score)
        metadata = method.object_metadata()
        assert all(size == 4096 for size in metadata["bank_sizes"].values())
        if stage < 2:
            np.testing.assert_array_equal(actual, expected[stage])
        if stage >= 2:
            assert method.validation_stats["proposed_modes"] > 0
            assert method.translation_stats["fallback_fits"] == 0
            assert metadata["normal_geometry"] == geometries
        if stage == 3:
            assert method.validation_stats["validation_pairs"] > 0
        geometries = metadata["normal_geometry"]
        assert all(not p.requires_grad for p in backbone._wrapper.model.parameters())
        print(f"real CUDA VR{stage}: finite/nonconstant; {method.validation_stats}")
