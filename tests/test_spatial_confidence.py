import copy

import numpy as np
import pytest
import torch
import yaml

from fmad.methods import register_methods
from fmad.methods.query_reference import QueryReference
from fmad.methods.reference_matching import ReferenceMatching
from fmad.methods.spatial_confidence import SpatialConfidence, spatial_confidence
from fmad.protocol import MethodContext
from fmad.registry import METHOD_REGISTRY
from tests.test_psma_mr_v2 import _write_image
from tests.test_query_reference import DEVICES, config
from tests.test_reference_matching import TinyPairBackbone


@pytest.mark.parametrize('device', DEVICES)
def test_uniform_opposite_zero_and_magnitude_sensitive_agreement(device):
    values = torch.tensor([.02, .01], device=device).repeat(10, 10, 1)
    c, stats = spatial_confidence(values.flatten(0, 1), (10, 10), min_tokens=8)
    torch.testing.assert_close(c, torch.tensor(1., device=device))
    assert stats['insufficient_axes'] == 0
    values[6:] *= -1
    assert spatial_confidence(values.flatten(0, 1), (10, 10), min_tokens=8)[0] == 0
    values[:] = 0
    assert spatial_confidence(values.flatten(0, 1), (10, 10), min_tokens=8)[0] == 0
    values[:] = .01
    values[6:] = .03
    c, _ = spatial_confidence(values.flatten(0, 1), (10, 10), min_tokens=8)
    torch.testing.assert_close(c, torch.tensor(.6, device=device))


def test_context_limits_grid_validation_and_repeatability():
    torch.manual_seed(5)
    values = torch.randn(90, 6)
    original = values.clone()
    c, stats = spatial_confidence(values, (9, 10), min_tokens=100)
    assert c == 0 and stats['insufficient_axes'] == 2
    first = spatial_confidence(values, (9, 10), min_tokens=4)
    second = spatial_confidence(values, (9, 10), min_tokens=4)
    assert first == second and 0 <= first[0] <= 1
    torch.testing.assert_close(values, original, rtol=0, atol=0)
    for grid in [(1, 90), (2, 45), (90, 1)]:
        assert spatial_confidence(values, grid, min_tokens=4)[0] == 0
    with pytest.raises(ValueError):
        spatial_confidence(values, (9, 9), min_tokens=4)
    with pytest.raises(ValueError):
        spatial_confidence(values * float('nan'), (9, 10), min_tokens=4)


@pytest.mark.parametrize('enabled,force_identity', [(False, False), (True, True), (True, False)])
def test_baseline_parity_immutable_fit_and_query_reset(tmp_path, enabled, force_identity):
    paths = [_write_image(tmp_path / f'{i}.png', 29 * i) for i in range(4)]
    cfg = config(1)
    if force_identity:
        cfg['min_fit_tokens'] = 1000
    context = MethodContext('can', 0, 8, False, False)
    backbone = TinyPairBackbone()
    weights = copy.deepcopy(backbone.model.state_dict())
    method = SpatialConfidence(backbone, dict(cfg, confidence_enabled=enabled))
    baseline = QueryReference(TinyPairBackbone(), cfg)
    original = ReferenceMatching(TinyPairBackbone(), config(0))
    for item in [method, baseline, original]:
        item.fit(paths[:2], context)
    banks = copy.deepcopy(method.banks)
    scales = copy.deepcopy(method.scales)
    first = method.predict(paths[2])
    method.predict(paths[3])
    second = method.predict(paths[2])
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1] == second[1] and np.isfinite(first[0]).all()
    # Force insufficient context explicitly; do not assume the tiny backbone
    # downsamples spatial grids. Disabled must delegate exactly to QR1.
    if force_identity or not enabled:
        expected = (original if force_identity else baseline).predict(paths[2])
        np.testing.assert_array_equal(first[0], expected[0])
    for record in method.confidence_records:
        for estimate in record['estimates']:
            assert 0 <= estimate['confidence'] <= 1
            assert estimate['applied_norm'] <= estimate['qr1_norm'] + 1e-7
    for r in method.resolutions:
        torch.testing.assert_close(method.scales[r], scales[r], rtol=0, atol=0)
        torch.testing.assert_close(method.scales[r], baseline.scales[r], rtol=0, atol=0)
        for a, b, ref in zip(method.banks[r]['layers'], banks[r]['layers'], baseline.banks[r]['layers']):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(a, ref, rtol=0, atol=0)
    for key, value in weights.items():
        torch.testing.assert_close(backbone.model.state_dict()[key], value, rtol=0, atol=0)
    metadata = method.object_metadata()
    assert metadata['trainable_parameters'] == 0 and not metadata['query_labels_used']
    assert len(method.confidence_records) == (3 if enabled else 0)
    method.fit(paths[:2], context)
    assert not method.confidence_records


@pytest.mark.parametrize('device', DEVICES)
def test_full_confidence_exactly_reproduces_qr1_and_norm_is_bounded(device):
    cfg = config(1)
    baseline = QueryReference(TinyPairBackbone(), cfg)
    method = SpatialConfidence(TinyPairBackbone(), cfg)
    torch.manual_seed(9)
    reference = torch.nn.functional.normalize(torch.randn(12, 8, device=device), dim=-1)
    query = torch.nn.functional.normalize(torch.randn(1, 8, device=device), dim=-1).repeat(100, 1)
    layers, bank = [query, query], {'layers': [reference, reference]}
    before = [r.clone() for r in bank['layers']]
    actual = method._score_tokens(layers, (10, 10), bank, normalize=False)
    expected = baseline._score_tokens(layers, (10, 10), bank, normalize=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for item in method._query_confidence:
        assert item['confidence'] == 1 and item['applied_norm'] <= .100001
    for a, b in zip(bank['layers'], before):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize('overrides', [dict(confidence_enabled='yes'),
    dict(query_adaptation=False), dict(robust_trim=True), dict(spatial_crossfit=True),
    dict(translation_control='random'), dict(scale_fusion='max')])
def test_invalid_settings(overrides):
    with pytest.raises(ValueError):
        SpatialConfidence(TinyPairBackbone(), {**config(1), **overrides})


def test_registered_configs_keep_comparison_conditions():
    from pathlib import Path
    register_methods('spatial_confidence')
    assert METHOD_REGISTRY.get('spatial_confidence') is SpatialConfidence
    root = Path(__file__).resolve().parents[1] / 'configs/methods'
    for shot in [1, 16]:
        candidate = yaml.safe_load((root / f'SC1_s{shot}.yaml').read_text())
        baseline = yaml.safe_load((root / f'PG_B_s{shot}.yaml').read_text())
        for key in ['dataset','backbone','seed','dataset_config','backbone_config','protocol','evaluation','output']:
            assert candidate[key] == baseline[key]
        assert candidate['method_config']['confidence_enabled'] is True
        assert not candidate['output']['save_anomaly_maps']
        assert not candidate['output']['save_weights']
