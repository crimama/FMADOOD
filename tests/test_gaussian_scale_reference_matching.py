import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from fmad.methods.gaussian_scale_reference_matching import (
    GaussianScaleReferenceMatching, bilinear_probe_indices,
    fit_gaussian_scale_model, gaussian_subset_scan,
)
from fmad.methods.multiscale_aggregation_reference_matching import (
    MultiscaleAggregationReferenceMatching,
)
from fmad.protocol import MethodContext
from tests.test_psma_mr_v2 import _write_image
from tests.test_reference_matching import TinyTripleBackbone
from tests.test_statistical_reference_matching import config


@pytest.mark.parametrize('grid,side', [((4, 7), 3), ((3, 2), 8), ((8, 8), 4)])
def test_probes_equal_bilinear_score_interpolation(grid, side):
    score = torch.arange(np.prod(grid), dtype=torch.float32).reshape(grid)
    indices, weights = bilinear_probe_indices(grid, side)
    sampled = (score.flatten()[indices] * torch.tensor(weights)).sum(-1)
    expected = F.interpolate(score[None, None], (side, side),
                             mode='bilinear', align_corners=False).flatten()
    torch.testing.assert_close(sampled.float(), expected)


def test_subset_scan_matches_closed_form_and_normal_scale_invariance():
    # With independent unit noise, max subset GLRT includes both weak coherent
    # evidence and the single strong channel; it is one-sided in the shift.
    maps = [torch.tensor([[3., -2., 2.]]), torch.tensor([[0., -1., 2.]])]
    model = dict(mean=[0., 0.], std=[1., 1.], precision=np.eye(2).tolist())
    expected = torch.tensor([[3., 0., np.sqrt(8)]], dtype=torch.float32)
    torch.testing.assert_close(gaussian_subset_scan(maps, model), expected)
    scaled = [maps[0]*7+10, maps[1]*.25-3]
    adjusted = dict(model, mean=[10., -3.], std=[7., .25])
    torch.testing.assert_close(gaussian_subset_scan(scaled, adjusted), expected)


def test_covariance_fit_uses_source_blocks_and_is_finite_for_constant_channel():
    rng = np.random.default_rng(12)
    scores = rng.normal(size=(3, 64, 4))
    scores[..., 1] = scores[..., 0] + .1*rng.normal(size=(3, 64))
    scores[..., 3] = 1.
    model = fit_gaussian_scale_model(scores)
    json.dumps(model, allow_nan=False)
    assert model['source_count'] == 3
    assert 0 <= model['shrinkage'] <= 1
    assert np.linalg.eigvalsh(model['covariance']).min() > 0
    assert not model['independent_probes_claimed']
    with pytest.raises(ValueError):
        fit_gaussian_scale_model(scores[:1])


@pytest.mark.parametrize('mode', ['zscore_order', 'scan_diagonal', 'scan_shrink'])
def test_normal_only_fit_bank_parity_and_no_query_refit(tmp_path, mode, monkeypatch):
    from fmad.methods.reference_matching_fast import ReferenceMatchingFast

    def extract(self, image, resolution, pool=None):
        rng = torch.Generator().manual_seed(int(np.asarray(image).sum()) + resolution)
        return [F.normalize(torch.randn(32, 24, generator=rng), dim=1)
                for _ in range(3)], (4, 8)
    monkeypatch.setattr(ReferenceMatchingFast, '_extract', extract)
    paths = [_write_image(tmp_path / f's{i}.png', 17*i) for i in range(3)]
    query = _write_image(tmp_path / 'q.png', 81)
    cfg = config(statistical_normalization='q95')
    cfg.update(resolutions=[8, 12], calibration_audit=False,
               multiscale_aggregation='scale_order_linear',
               gaussian_scale_fusion=mode, gaussian_probe_side=3, noise_guard_action='lowres')
    context = MethodContext('can', 0, 8, False, False)
    method = GaussianScaleReferenceMatching(TinyTripleBackbone(), cfg)
    method.fit(paths, context)
    baseline = MultiscaleAggregationReferenceMatching(TinyTripleBackbone(), cfg)
    baseline.fit(paths, context)
    assert method.object_metadata()['ablation_bank_feature_sha256'] == (
        baseline.object_metadata()['ablation_bank_feature_sha256'])
    before = json.dumps(method.object_metadata(), allow_nan=False, sort_keys=True)
    amap, score = method.predict(query)
    assert np.isfinite(amap).all() and np.isfinite(score)
    assert json.dumps(method.object_metadata(), allow_nan=False, sort_keys=True) == before
    assert method._probe_features == {} and method._probe_bags == {}
    # The covariance of an allowed noise-guard scale subset is marginalized.
    subset = method._combine_maps([torch.ones((3, 3))], [8])
    assert torch.isfinite(subset).all()


def test_none_order_control_is_supported_without_changing_banks(tmp_path):
    paths = [_write_image(tmp_path / f'{i}.png', i*29) for i in range(2)]
    cfg = config(statistical_normalization='none')
    cfg.update(resolutions=[8, 12], calibration_audit=False,
               multiscale_aggregation='scale_order_linear')
    method = MultiscaleAggregationReferenceMatching(TinyTripleBackbone(), cfg)
    method.fit(paths, MethodContext('can', 0, 8, False, False))
    assert np.isfinite(method.predict(paths[0])[0]).all()
