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
from fmad.methods.query_reference import QueryReference, fit_translation, nearest_residuals
from fmad.methods.reference_consensus import (
    ReferenceConsensus, consensus_residuals, support_groups,
    translation_weights, weighted_translation,
)
from fmad.protocol import MethodContext
from fmad.registry import METHOD_REGISTRY
from tests.test_psma_mr_v2 import _write_image
from tests.test_query_reference import DEVICES, config as qr_config
from tests.test_reference_matching import TinyPairBackbone


ROOT = Path(__file__).resolve().parents[1]


def config(stage):
    return {**qr_config(1), "consensus_weighting": stage > 0,
            "match_translation_norm": stage > 1, "balance_reference_cells": stage > 2,
            "consensus_groups": 4, "consensus_weight_floor": .1}


@pytest.mark.parametrize("device", DEVICES)
def test_disjoint_whole_support_groups_and_chunked_direction_formula(device):
    torch.manual_seed(12)
    owners = [source for source in range(8) for _ in range(3)]
    groups = support_groups(owners, 4, device)
    assert torch.cat(groups).unique().numel() == len(owners)
    assert all(len(indices) == 6 for indices in groups)
    for source in range(8):
        assert sum(any(owners[i] == source for i in indices.tolist()) for indices in groups) == 1
    reference = F.normalize(torch.randn(24, 8, device=device), dim=-1)
    query = F.normalize(torch.randn(19, 8, device=device), dim=-1)
    actual, selected, agreement = consensus_residuals(query, reference, groups, 5)
    torch.testing.assert_close(actual, nearest_residuals(query, reference, 5), rtol=0, atol=0)
    directions = [F.normalize(query - reference[g[(query @ reference[g].T).argmax(1)]], dim=-1)
                  for g in groups]
    expected = torch.stack([(directions[i] * directions[j]).sum(1)
                            for i in range(4) for j in range(i + 1, 4)]).mean(0).clamp(0, 1)
    torch.testing.assert_close(agreement, expected, atol=2e-6, rtol=2e-6)
    for chunk in (1, 19):
        other = consensus_residuals(query, reference, groups, chunk)
        torch.testing.assert_close(other[0], actual)
        torch.testing.assert_close(other[1], selected)
        torch.testing.assert_close(other[2], agreement, atol=2e-6, rtol=2e-6)


def test_consensus_does_not_reject_large_consistent_residuals_and_can_trust_anomalies():
    # Geometry only: a far query agrees across references. This tests both the
    # intended non-magnitude selection and the explicit global-defect loophole.
    bank = torch.tensor([[1., 0., 0.]]).repeat(4, 1)
    groups = support_groups(list(range(4)), 4, "cpu")
    query = F.normalize(torch.tensor([[1., .01, 0.], [1., 1., 0.]]), dim=-1)
    _, identities, agreement = consensus_residuals(query, bank, groups, 2)
    weights = translation_weights(agreement, identities, floor=.1, balance=False)
    torch.testing.assert_close(weights, torch.ones(2))
    # Inconsistent residual directions get only the floor, even if small.
    low = translation_weights(torch.tensor([0.]), torch.tensor([0]), floor=.1, balance=False)
    assert float(low[0]) == pytest.approx(.1)


@pytest.mark.parametrize("device", DEVICES)
def test_weighted_ridge_norm_control_and_cell_occupancy(device):
    residuals = torch.tensor([[.06, 0.], [.06, 0.], [0., .03]], device=device)
    identities = torch.tensor([0, 0, 1], device=device)
    agreement = torch.tensor([1., .5, .1], device=device)
    weights = translation_weights(agreement, identities, floor=.1, balance=False)
    balanced = translation_weights(agreement, identities, floor=.1, balance=True)
    torch.testing.assert_close(balanced, weights / weights.new_tensor([2., 2., 1.]).sqrt())
    options = dict(ridge=1., max_norm=.1, min_tokens=2)
    delta, stats, extras = weighted_translation(residuals, weights, **options, match_norm=False)
    expected = (residuals * weights[:, None]).sum(0) / weights.sum() / 2
    torch.testing.assert_close(delta, expected)
    baseline, _ = fit_translation(residuals, keep_fraction=1., **options)
    matched, _, _ = weighted_translation(residuals, weights, **options, match_norm=True)
    torch.testing.assert_close(matched.norm(), baseline.norm())
    torch.testing.assert_close(F.normalize(matched, dim=0), F.normalize(delta, dim=0))
    assert extras["effective_tokens"] <= len(residuals) and not stats["fallback"]
    uniform, _, _ = weighted_translation(residuals, torch.ones_like(weights), **options, match_norm=False)
    torch.testing.assert_close(uniform, baseline)
    zero, stats, _ = weighted_translation(residuals[:1], weights[:1], **options, match_norm=True)
    assert stats["fallback"] and zero.count_nonzero() == 0
    # Opposite residuals cancel only after weighting: must fall back to QR1.
    cancellation = residuals.new_tensor([[.1, 0.], [-.05, 0.]])
    result, _, extra = weighted_translation(cancellation, weights.new_tensor([1., 2.]),
                                            **options, match_norm=True)
    expected, _ = fit_translation(cancellation, keep_fraction=1., **options)
    torch.testing.assert_close(result, expected)
    assert extra["norm_match_fallback"] == 1
    clipped, _, _ = weighted_translation(residuals * 100, weights, **options, match_norm=False)
    assert float(clipped.norm()) <= .100001


@pytest.mark.parametrize("device", DEVICES)
def test_qr1_parity_frozen_state_scalar_diagnostics_and_reset(tmp_path, device):
    paths = [_write_image(tmp_path / f"{i}.png", i * 27) for i in range(4)]
    context = MethodContext("can", 7, 8, False, False)
    backbone = TinyPairBackbone()
    backbone.device = device
    model_state = copy.deepcopy(backbone.model.state_dict())
    baseline = QueryReference(backbone, qr_config(1))
    baseline.fit(paths, context)
    expected, expected_score = baseline.predict(paths[0])
    for stage in range(4):
        method = ReferenceConsensus(backbone, config(stage))
        method.fit(paths, context)
        actual, score = method.predict(paths[0])
        method.predict(paths[1])
        repeat, repeat_score = method.predict(paths[0])
        np.testing.assert_array_equal(actual, repeat)
        assert score == repeat_score and np.isfinite(actual).all()
        if stage == 0:
            np.testing.assert_array_equal(actual, expected)
            assert score == expected_score
        else:
            assert method.consensus_stats["fits"] == 12
            assert method.translation_stats["fallback_fits"] == 0
        for resolution in method.resolutions:
            torch.testing.assert_close(method.scales[resolution], baseline.scales[resolution], rtol=0, atol=0)
            for a, b in zip(method.banks[resolution]["layers"], baseline.banks[resolution]["layers"]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert method.records == {} and method.history == []
        assert all(isinstance(v, (int, float)) for v in method.consensus_stats.values())
        metadata = method.object_metadata()
        assert metadata["trainable_parameters"] == 0 and not metadata["query_labels_used"]
        assert not metadata["dense_maps_accumulated"] and not metadata["persistent_adapted_state"]
        method.fit(paths, context)
        assert method.consensus_stats["fits"] == method.translation_stats["fits"] == 0
    for name, original in model_state.items():
        torch.testing.assert_close(backbone.model.state_dict()[name], original, rtol=0, atol=0)


@pytest.mark.parametrize("override", [
    {"query_adaptation": False}, {"robust_trim": True}, {"spatial_crossfit": True},
    {"consensus_weighting": "true"}, {"match_translation_norm": 1},
    {"balance_reference_cells": True}, {"consensus_groups": 1},
    {"consensus_groups": 2.5}, {"consensus_weight_floor": 0},
    {"consensus_weight_floor": float("nan")}, {"consensus_weight_floor": 1.1},
])
def test_invalid_config(override):
    with pytest.raises(ValueError):
        ReferenceConsensus(TinyPairBackbone(), {**config(0), **override})


def test_insufficient_independent_supports_rejected():
    with pytest.raises(ValueError, match="normal supports"):
        support_groups([0, 0, 1, 1], 4, "cpu")


def test_registry_and_configs_change_one_switch_only():
    register_methods("reference_consensus")
    assert METHOD_REGISTRY.get("reference_consensus") is ReferenceConsensus
    baseline = yaml.safe_load((ROOT / "configs/methods/query_reference_QR1.yaml").read_text())
    configs = [yaml.safe_load((ROOT / f"configs/methods/reference_consensus_RC{i}.yaml").read_text())
               for i in range(4)]
    for stage, settings in enumerate(configs):
        for name, value in baseline.items():
            if name == "method_config":
                for key, expected in value.items():
                    assert settings[name][key] == expected
            elif name != "method":
                assert settings[name] == value
        if stage:
            previous = copy.deepcopy(configs[stage - 1])
            switch = ("consensus_weighting", "match_translation_norm", "balance_reference_cells")[stage - 1]
            previous["method_config"][switch] = True
            assert settings == previous


@pytest.mark.skipif(os.environ.get("FMAD_REAL_BACKBONE_SMOKE") != "1",
                    reason="explicit cached-backbone CUDA smoke only")
def test_real_cached_backbone_full_resolution_consensus():
    from fmad.backbones.dinov2 import DINOv2Backbone
    paths = [str(p) for p in sorted(Path("/data/mvtec_ad_2/can/train/good").glob("*.png"))[:16]]
    assert len(paths) == 16 and torch.cuda.is_available()
    backbone = DINOv2Backbone("dinov2_vitl14", "cuda:0", feature_layers=[11, 23])
    context = MethodContext("can", 0, 448, False, False)
    reference_config = yaml.safe_load((ROOT / "configs/methods/query_reference_QR1.yaml").read_text())["method_config"]
    baseline = QueryReference(backbone, reference_config)
    baseline.fit(paths, context)
    expected = baseline.predict(paths[-1])[0]
    for stage in range(4):
        settings = yaml.safe_load((ROOT / f"configs/methods/reference_consensus_RC{stage}.yaml").read_text())["method_config"]
        method = ReferenceConsensus(backbone, settings)
        method.fit(paths, context)
        actual, score = method.predict(paths[-1])
        assert np.isfinite(actual).all() and np.ptp(actual) > 0 and np.isfinite(score)
        assert all(size == 4096 for size in method.object_metadata()["bank_sizes"].values())
        if stage == 0:
            np.testing.assert_array_equal(actual, expected)
        else:
            assert not np.array_equal(actual, expected)
            assert method.translation_stats["fallback_fits"] == 0
            assert method.consensus_stats["fits"] == 4
            if stage >= 2:
                assert method.translation_stats["norm_sum"] == pytest.approx(
                    method.consensus_stats["baseline_norm_sum"], abs=1e-6)
        assert all(not p.requires_grad for p in backbone._wrapper.model.parameters())
        print(f"real CUDA RC{stage}: finite/nonconstant; {method.consensus_stats}")
