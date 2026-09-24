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
from fmad.methods.PSMA_MR import _read_rgb
from fmad.methods.cross_image_reference import (
    BoundedResidual, CompactPool, CrossImageReference, reference_radii,
    source_distances, weighted_tail,
)
from fmad.methods.reference_matching import ReferenceMatching
from fmad.methods.tail_aware_fusion import TailAwareFusion, sample_native
from fmad.protocol import MethodContext
from fmad.registry import METHOD_REGISTRY
from tests.test_psma_mr_v2 import _write_image
from tests.test_reference_matching import TinyPairBackbone


DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))]


def config(stage):
    return dict(variant="R1", feature_layers=[11, 23], resolutions=[8, 12],
                bank_tokens_per_source=4, query_chunk_size=8, steps=10,
                train_tokens=8, normal_candidates=32, head_chunk_size=63,
                learn_residual=stage > 0, global_tail_pool=stage > 1,
                reference_geometry=stage > 2)


@pytest.mark.parametrize("geometry", [False, True])
def test_residual_initial_parity_bounds_and_agreeing_layer_correction(geometry):
    torch.manual_seed(0)
    head = BoundedResidual(geometry, .5)
    assert sum(p.numel() for p in head.parameters()) == 129
    values = torch.tensor([[2., 2., 1., 1., 3., 3.], [1., 4., 2., 3., 2., 5.]])
    torch.testing.assert_close(head(values), values[:, :2].mean(-1), rtol=0, atol=0)
    head(values).sum().backward()
    assert head.network[-1].bias.grad.abs().sum() > 0
    with torch.no_grad():
        head.network[-1].bias.fill_(100)
    torch.testing.assert_close(head(values), values[:, :2].mean(-1) + .5)
    assert head(values)[0] != 2.
    if not geometry:
        changed = values.clone()
        changed[:, 2:] += 100
        torch.testing.assert_close(head(values), head(changed), rtol=0, atol=0)


def test_weighted_tail_fractional_boundary_and_stable_ties():
    ids, probabilities = weighted_tail(np.array([10., 9., 8.]), np.array([1., 9., 10.]), .25)
    np.testing.assert_array_equal(ids, [0, 1])
    np.testing.assert_allclose(probabilities, [.2, .8])
    ids, probabilities = weighted_tail(np.ones(3), np.ones(3), .5)
    np.testing.assert_array_equal(ids, [0, 1])
    np.testing.assert_allclose(probabilities, [2/3, 1/3])
    ids, _ = weighted_tail(np.array([0., 5., 8.]), np.ones(3), .01)
    np.testing.assert_array_equal(ids, [2])


@pytest.mark.parametrize("scores,weights,fraction", [
    ([1., float("nan")], [1., 1.], .05), ([1.], [0.], .05),
    ([], [], .05), ([1.], [1.], 0.), ([1.], [1.], 1.1),
])
def test_weighted_tail_rejects_invalid_population(scores, weights, fraction):
    with pytest.raises(ValueError):
        weighted_tail(scores, weights, fraction)


@pytest.mark.parametrize("device", DEVICES)
def test_source_minima_and_radius_exclude_both_owners(device):
    generator = torch.Generator().manual_seed(31)
    bank = F.normalize(torch.rand(10, 7, generator=generator), dim=1).to(device)
    query = F.normalize(torch.rand(4, 7, generator=generator), dim=1).to(device)
    owners = torch.arange(5, device=device).repeat_interleave(2)
    distance, identity, table = source_distances(query, bank, owners, 5, 3)
    matrix = 1 - query @ bank.T
    torch.testing.assert_close(distance, matrix.amin(1))
    torch.testing.assert_close(identity, matrix.argmin(1))
    for source in range(5):
        torch.testing.assert_close(table[:, source], matrix[:, owners == source].amin(1))
    _, _, reference_table = source_distances(bank, bank, owners, 5, 3)
    radii = reference_radii(reference_table, owners, 3, excluded=1)
    for row in range(10):
        eligible = [s for s in range(5) if s not in (1, int(owners[row]))]
        expected = reference_table[row, eligible].clamp_min(0).topk(3, largest=False).values.mean()
        torch.testing.assert_close(radii[row], expected)
    altered = reference_table.clone()
    altered[:, 1] = -1000
    altered[torch.arange(10, device=device), owners] = -1000
    torch.testing.assert_close(reference_radii(altered, owners, 3, 1), radii, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_cr0_exact_r1_and_triplet_contract(tmp_path, device):
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(5)]
    context = MethodContext("can", 7, 8, False, False)
    backbone = TinyPairBackbone()
    backbone.device = device
    original = copy.deepcopy(backbone.model.state_dict())
    reference = ReferenceMatching(backbone, config(0))
    reference.fit(paths, context)
    expected = reference.predict(paths[0])[0]
    methods = []
    for stage in range(4):
        method = CrossImageReference(backbone, config(stage))
        emitted = []
        method._training_log_sink = lambda step, row: emitted.append((step, row))
        method.fit(paths, context)
        actual, score = method.predict(paths[0])
        assert actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(score)
        assert method.parameter_count == (129 if stage else 0)
        assert method.fit_backbone_calls == 10 + (80 if stage else 0)
        assert method.records == {}
        for resolution in method.resolutions:
            torch.testing.assert_close(method.scales[resolution], reference.scales[resolution], rtol=0, atol=0)
            assert method.banks[resolution]["owners"] == reference.banks[resolution]["owners"]
        if not stage:
            np.testing.assert_array_equal(actual, expected)
            assert not method.tables and not method.history and not emitted
        else:
            assert len(method.episodes) == len(method.history) == 10
            assert [step for step, _ in emitted if step is not None] == list(range(10))
            assert sum(step is None for step, _ in emitted) == 10
            assert all(np.isfinite(row["loss"]) and row["pair_terms"] == 64 for row in method.history)
            if stage == 1:
                assert all(row["cross_source_pair_fraction"] == 0 for row in method.history)
            else:
                assert any(row["cross_source_pair_fraction"] > 0 for row in method.history)
            for source in range(5):
                for resolution in method.resolutions:
                    bank = method._geometry_bank(resolution, source)
                    assert source not in bank["owner_tensor"].tolist()
                    assert len(bank["layers"][0]) == 16
            before = copy.deepcopy(method.head.state_dict())
            method.predict(paths[1])
            for name, value in before.items():
                torch.testing.assert_close(method.head.state_dict()[name], value, rtol=0, atol=0)
            assert not any(key in vars(method) for key in ("normals", "positives", "values", "score_maps", "views", "images"))
            assert method.pool_metadata["allocated_bytes"] <= 128 * 1024**2
            assert method.object_metadata()["descriptor_pool_retained_after_fit"] is False
            # Native descriptor interpolation must agree with the dense convention.
            image = _read_rgb(paths[0])
            grid = method._descriptor_grid(image, 8, method._geometry_bank(8))
            dense = F.interpolate(grid[None], size=image.shape[:2], mode="bilinear", align_corners=False)[0].flatten(1).T
            torch.testing.assert_close(sample_native(grid, np.arange(192), (12, 16)), dense, rtol=3e-6, atol=3e-6)
            with torch.no_grad():
                method.head.network[-1].weight.zero_()
                method.head.network[-1].bias.zero_()
            np.testing.assert_allclose(method.predict(paths[0])[0], expected, rtol=3e-6, atol=3e-6)
            method.head.load_state_dict(before)
        methods.append(method)
    for name, value in original.items():
        torch.testing.assert_close(backbone.model.state_dict()[name], value, rtol=0, atol=0)
    assert methods[1].episodes == methods[2].episodes == methods[3].episodes
    assert methods[1].pool_metadata == methods[2].pool_metadata == methods[3].pool_metadata
    expected_ids = [row["positive_ids_sha256"] for row in methods[1].history]
    for method in methods[2:]:
        assert [row["positive_ids_sha256"] for row in method.history] == expected_ids
    # The source/family/view deck is byte-identical to TBF's established deck.
    for step, episode in enumerate(methods[1].episodes):
        source = step % 5
        image = _read_rgb(paths[source])
        assert TailAwareFusion._episode(methods[1], image, source, step, 7)[3] == episode


def test_pool_memory_cap_and_support_requirements(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(5)]
    context = MethodContext("can", 0, 8, False, False)
    with pytest.raises(ValueError, match="supports"):
        CrossImageReference(TinyPairBackbone(), config(1)).fit(paths[:4], context)
    with pytest.raises(ValueError, match="memory cap"):
        CrossImageReference(TinyPairBackbone(), {**config(1), "steps": 1000,
                                                "descriptor_pool_max_mib": 1}).fit(paths, context)
    pool = CompactPool(1, 2)
    pool.append(np.ones((1, 2, 6), dtype=np.float32), 1., 0, 0)
    with pytest.raises(RuntimeError):
        pool.append(np.ones((1, 2, 6), dtype=np.float32), 1., 0, 0)


@pytest.mark.parametrize("bad", [dict(learn_residual="yes"), dict(global_tail_pool=True),
    dict(reference_geometry=True), dict(residual_bound=float("inf")),
    dict(normal_tail_fraction=float("nan")), dict(normal_candidates=0), dict(steps=True),
    dict(margin=-1), dict(learning_rate=0), dict(variant="R2"), dict(reference_sources=0)])
def test_invalid_config(bad):
    with pytest.raises(ValueError):
        CrossImageReference(TinyPairBackbone(), {**config(0), **bad})


def test_config_boundaries_and_common_pipeline():
    register_methods("cross_image_reference")
    assert METHOD_REGISTRY.get("cross_image_reference") is CrossImageReference
    root = Path(__file__).resolve().parents[1]
    configs = [yaml.safe_load((root / f"configs/methods/cross_image_reference_CR{i}.yaml").read_text()) for i in range(4)]
    baseline = yaml.safe_load((root / "configs/methods/tail_aware_fusion_TF0.yaml").read_text())
    for stage, settings in enumerate(configs):
        for key in ("dataset", "backbone", "seed", "device", "dataset_config", "backbone_config", "protocol", "evaluation", "output"):
            assert settings[key] == baseline[key]
        if stage:
            previous = copy.deepcopy(configs[stage - 1])
            previous["method_config"][("learn_residual", "global_tail_pool", "reference_geometry")[stage - 1]] = True
            assert previous == settings


@pytest.mark.skipif(os.environ.get("FMAD_REAL_BACKBONE_SMOKE") != "1",
                    reason="explicit cached-backbone CUDA smoke only")
def test_real_cached_backbone_full_bank_native_fit_predict():
    from fmad.backbones.dinov2 import DINOv2Backbone
    paths = [str(p) for p in sorted(Path("/data/mvtec_ad_2/can/train/good").glob("*.png"))[:16]]
    assert len(paths) == 16 and torch.cuda.is_available()
    backbone = DINOv2Backbone("dinov2_vitl14", "cuda:0", feature_layers=[11, 23])
    context = MethodContext("can", 0, 448, False, False)
    root = Path(__file__).resolve().parents[1]
    prior = None
    for stage in range(4):
        settings = yaml.safe_load((root / f"configs/methods/cross_image_reference_CR{stage}.yaml").read_text())["method_config"]
        settings["steps"] = 2
        method = CrossImageReference(backbone, settings)
        method.fit(paths, context)
        actual, score = method.predict(paths[0])
        assert np.isfinite(actual).all() and np.ptp(actual) > 0 and np.isfinite(score)
        assert all(size == 4096 for size in method.object_metadata()["bank_sizes"].values())
        if stage == 0:
            reference = ReferenceMatching(backbone, settings)
            reference.fit(paths, context)
            np.testing.assert_array_equal(actual, reference.predict(paths[0])[0])
        else:
            assert len(method.history) == 2 and all(np.isfinite(row["loss"]) for row in method.history)
            evidence = (method.episodes, method.pool_metadata)
            if prior is not None:
                assert evidence == prior
            prior = evidence
        print(f"real CUDA CR{stage}: finite/nonconstant; updates={len(method.history)}; parameters={method.parameter_count}")
