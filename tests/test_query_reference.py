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
from fmad.methods.query_reference import (
    QueryReference, fit_translation, nearest_residuals, spatial_folds,
)
from fmad.methods.reference_matching import ReferenceMatching, _distance_chunks
from fmad.protocol import MethodContext
from fmad.registry import METHOD_REGISTRY
from tests.test_psma_mr_v2 import _write_image
from tests.test_reference_matching import TinyPairBackbone


DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))]


def config(stage):
    return dict(variant="R1", feature_layers=[11, 23], resolutions=[8, 12],
                bank_tokens_per_source=4, query_chunk_size=8, min_fit_tokens=2,
                query_adaptation=stage > 0, robust_trim=stage > 1,
                spatial_crossfit=stage > 2, crossfit_halo=0)


def fit(values, **overrides):
    options = dict(keep_fraction=1.0, ridge=1.0, max_norm=0.1, min_tokens=2)
    return fit_translation(values, **{**options, **overrides})


@pytest.mark.parametrize("device", DEVICES)
def test_nearest_residuals_use_initial_independent_cosine_matches(device):
    torch.manual_seed(19)
    bank = F.normalize(torch.randn(15, 8, device=device), dim=-1)
    query = F.normalize(torch.randn(21, 8, device=device), dim=-1)
    actual = nearest_residuals(query, bank, 4)
    expected = query - bank[(query @ bank.T).argmax(1)]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("device", DEVICES)
def test_closed_form_ridge_clip_and_trim(device):
    normal = torch.tensor([[0.04, 0.02]], device=device).repeat(6, 1)
    contaminated = torch.cat([normal, normal.new_tensor([[0.8, 0.8], [0.9, 0.9]])])
    plain, plain_stats = fit(contaminated)
    robust, stats = fit(contaminated, keep_fraction=.75)
    torch.testing.assert_close(robust, normal[0] / 2)
    assert stats["retained_tokens"] == 6 and not stats["fallback"]
    assert torch.linalg.vector_norm(plain - normal[0] / 2) > 0.01
    assert plain_stats["clipped"] and torch.linalg.vector_norm(plain) <= .100001
    zero, stats = fit(contaminated[:1])
    assert torch.count_nonzero(zero) == 0 and stats["fallback"]
    zero, stats = fit(contaminated[:0])
    assert torch.count_nonzero(zero) == 0 and stats["fallback"]


@pytest.mark.parametrize("grid,blocks,halo", [((7, 9), 2, 1), ((2, 3), 4, 1), ((8, 8), 2, 0)])
def test_spatial_folds_exact_coverage_and_no_self_fit(grid, blocks, halo):
    coverage = torch.zeros(grid[0] * grid[1], dtype=torch.int64)
    for scored, context in spatial_folds(grid, blocks, halo, "cpu"):
        coverage += scored
        assert not torch.any(scored & context)
        if halo:
            expanded = F.max_pool2d(scored.float().reshape(1, 1, *grid),
                                    halo * 2 + 1, stride=1, padding=halo).bool().flatten()
            assert not torch.any(expanded & context)
    assert torch.all(coverage == 1)


def test_held_block_values_cannot_change_its_own_translation():
    torch.manual_seed(12)
    residuals = torch.randn(64, 8) * .01
    for scored, context in spatial_folds((8, 8), 2, 1, "cpu"):
        expected, _ = fit(residuals[context], keep_fraction=.75)
        changed = residuals.clone()
        changed[scored] += 100
        actual, _ = fit(changed[context], keep_fraction=.75)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_control_parity_bank_preservation_and_no_cross_query_state(tmp_path, device):
    paths = [_write_image(tmp_path / f"{i}.png", i * 27) for i in range(4)]
    context = MethodContext("can", 7, 8, False, False)
    backbone = TinyPairBackbone()
    backbone.device = device
    reference = ReferenceMatching(backbone, config(0))
    reference.fit(paths, context)
    expected, score = reference.predict(paths[0])
    weights = copy.deepcopy(backbone.model.state_dict())
    for stage in range(4):
        method = QueryReference(backbone, config(stage))
        method.fit(paths, context)
        banks = {r: [t.clone() for t in b["layers"]] for r, b in method.banks.items()}
        for resolution in method.resolutions:
            torch.testing.assert_close(method.scales[resolution], reference.scales[resolution],
                                       rtol=0, atol=0)
            for actual, baseline in zip(banks[resolution], reference.banks[resolution]["layers"]):
                torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
        actual, image_score = method.predict(paths[0])
        method.predict(paths[1])
        repeat, repeat_score = method.predict(paths[0])
        np.testing.assert_array_equal(actual, repeat)
        assert image_score == repeat_score
        assert np.isfinite(actual).all() and actual.shape == expected.shape
        if stage == 0:
            np.testing.assert_array_equal(actual, expected)
            assert score == image_score
        else:
            assert method.translation_stats["fits"] > 0
            assert method.translation_stats["fallback_fits"] == 0
        for resolution in method.resolutions:
            for actual_bank, original in zip(method.banks[resolution]["layers"], banks[resolution]):
                torch.testing.assert_close(actual_bank, original, rtol=0, atol=0)
        assert method.records == {} and method.history == []
        metadata = method.object_metadata()
        assert metadata["trainable_parameters"] == 0 and not metadata["query_labels_used"]
        assert not metadata["dense_maps_accumulated"] and not metadata["persistent_adapted_state"]
        assert all(isinstance(v, (int, float)) for v in method.translation_stats.values())
    for key, value in backbone.model.state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)


def test_translation_sign_reduces_shared_shift_and_does_not_reconstruct_outlier():
    # Geometric unit test, not a benchmark or a claim about real defects.
    reference = torch.tensor([[1., 0., 0.]])
    normal = F.normalize(reference + torch.tensor([[0., .08, 0.]]), dim=-1).repeat(12, 1)
    anomaly = F.normalize(reference + torch.tensor([[0., .08, .8]]), dim=-1).repeat(4, 1)
    query = torch.cat([normal, anomaly])
    residuals = nearest_residuals(query, reference, 4)
    delta, _ = fit(residuals, keep_fraction=.75)
    adapted = F.normalize(reference + delta, dim=-1)
    before = _distance_chunks(query, reference, 4)
    after = _distance_chunks(query, adapted, 4)
    assert after[:12].mean() < before[:12].mean()
    assert after[12:].min() > after[:12].max() + .1


@pytest.mark.parametrize("override", [
    {"variant": "R2"}, {"query_adaptation": "false"}, {"robust_trim": True},
    {"spatial_crossfit": True}, {"keep_fraction": 0}, {"keep_fraction": 1.1},
    {"ridge": -1}, {"ridge": float("nan")}, {"max_shift_norm": 1},
    {"max_shift_norm": 0}, {"max_shift_norm": True}, {"crossfit_blocks": 1},
    {"crossfit_halo": -1}, {"min_fit_tokens": 0}, {"min_fit_tokens": 1.5},
])
def test_invalid_settings_rejected(override):
    with pytest.raises(ValueError):
        QueryReference(TinyPairBackbone(), {**config(0), **override})


def test_registry_configs_and_one_component_at_a_time():
    register_methods("query_reference")
    assert METHOD_REGISTRY.get("query_reference") is QueryReference
    root = Path(__file__).resolve().parents[1]
    configs = [yaml.safe_load((root / f"configs/methods/query_reference_QR{i}.yaml").read_text())
               for i in range(4)]
    baseline = yaml.safe_load((root / "configs/methods/reference_matching_R1.yaml").read_text())
    for stage, settings in enumerate(configs):
        assert settings["method"] == "query_reference"
        assert settings["method_config"]["query_adaptation"] == (stage > 0)
        assert settings["method_config"]["robust_trim"] == (stage > 1)
        assert settings["method_config"]["spatial_crossfit"] == (stage > 2)
        for key in ("seed", "dataset", "backbone", "dataset_config", "backbone_config",
                    "protocol", "evaluation", "output"):
            assert settings[key] == baseline[key]
        for key, value in baseline["method_config"].items():
            assert settings["method_config"][key] == value
        if stage:
            previous = copy.deepcopy(configs[stage - 1])
            previous["method_config"][("query_adaptation", "robust_trim", "spatial_crossfit")[stage - 1]] = True
            assert settings == previous


@pytest.mark.skipif(os.environ.get("FMAD_REAL_BACKBONE_SMOKE") != "1",
                    reason="explicit cached-backbone CUDA smoke only")
def test_real_cached_backbone_full_resolution_query_reference():
    from fmad.backbones.dinov2 import DINOv2Backbone
    paths = [str(p) for p in sorted(Path("/data/mvtec_ad_2/can/train/good").glob("*.png"))[:16]]
    assert len(paths) == 16 and torch.cuda.is_available()
    backbone = DINOv2Backbone("dinov2_vitl14", "cuda:0", feature_layers=[11, 23])
    root = Path(__file__).resolve().parents[1]
    context = MethodContext("can", 0, 448, False, False)
    expected = None
    for stage in range(4):
        settings = yaml.safe_load((root / f"configs/methods/query_reference_QR{stage}.yaml").read_text())["method_config"]
        method = QueryReference(backbone, settings)
        method.fit(paths, context)
        actual, score = method.predict(paths[-1])
        assert np.isfinite(actual).all() and np.ptp(actual) > 0 and np.isfinite(score)
        assert all(size == 4096 for size in method.object_metadata()["bank_sizes"].values())
        if stage == 0:
            reference = ReferenceMatching(backbone, settings)
            reference.fit(paths, context)
            expected = reference.predict(paths[-1])[0]
            np.testing.assert_array_equal(actual, expected)
        else:
            assert not np.array_equal(actual, expected)
            assert method.translation_stats["fallback_fits"] == 0
        print(f"real CUDA QR{stage}: finite/nonconstant; {method.translation_stats}")
