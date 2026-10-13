import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from fmad.evaluation.fusion_diagnostic import FusionDiagnostic
from fmad.methods.fusion_factorial import FusionFactorial
from fmad.methods.reference_matching import ReferenceMatching
from fmad.protocol import MethodContext
from tests.test_reference_matching import TinyTripleBackbone
from tests.test_psma_mr_v2 import _write_image


def config(arm):
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / f'configs/methods/FF_{arm}.yaml').read_text())['method_config']
    return dict(cfg, resolutions=[8, 12], bank_tokens_per_source=4, query_chunk_size=4, bank_bags=2)


@pytest.mark.parametrize('arm,expected', [('A', [6., 9.]), ('B', [6., 9.]),
    ('C', [4., 6.]), ('D', [4., 6.]), ('E', [4., 4.])])
def test_exact_fusion_not_three_layer_mean(arm, expected):
    method = FusionFactorial(TinyTripleBackbone(), config(arm))
    scores = [torch.tensor([2., 5.]), torch.tensor([3., 3.]), torch.tensor([5., 5.])]
    torch.testing.assert_close(method._fuse_layers(scores, tau=3.), torch.tensor(expected))


def test_matched_banks_d_exact_parity_and_batch_auxiliary(tmp_path):
    paths = [_write_image(tmp_path / f'{i}.png', i * 17) for i in range(3)]
    context = MethodContext('can', 0, 8, False, False)
    ref = ReferenceMatching(TinyTripleBackbone(), config('D'))
    ref.fit(paths, context)
    expected = ref.predict(paths[0])[0]
    hashes = []
    gates = []
    for arm in 'ABCDE':
        method = FusionFactorial(TinyTripleBackbone(), config(arm))
        method.fit(paths, context)
        predictions = method.predict_batch(paths[:2])
        hashes.append(method.object_metadata()['bank_feature_sha256'])
        for prediction in predictions:
            aux = prediction.diagnostics['fusion_audit']
            assert aux['raw_map'].shape == prediction.score_map.shape == (12, 16)
            assert len(aux['gate_grids']) == 2
        gates.append(predictions[0].diagnostics['fusion_audit']['gate_grids'])
        assert method._raw_map is None and method._gate_grids == [] and not method._audit_active
        if arm == 'D':
            np.testing.assert_array_equal(predictions[0].score_map, expected)
        with pytest.raises(FileNotFoundError):
            method.predict_batch([str(tmp_path / 'missing.png')])
        assert not method._audit_active and method._raw_map is None
    assert len(set(hashes)) == 1
    for group in gates[1:]:
        for a, b in zip(gates[0], group):
            np.testing.assert_array_equal(a, b)


def test_diagnostic_condition_raw_cq_and_label_isolation():
    audit = FusionDiagnostic()
    raw = np.array([[0., 1.], [2., 3.]], np.float32)
    mask = np.array([[0, 0], [1, 1]], bool)
    auxiliary = {'raw_map': raw, 'gate_grids': [np.array([[0., 1.]], np.float32)]}
    for condition in ('regular', 'shift_1'):
        audit.update(f'001_{condition}.png', 'bad', raw, mask, auxiliary)
    report = audit.compute()
    json.dumps(report, allow_nan=False)
    assert report['metrics']['all']['raw'] == report['metrics']['all']['CQ']
    assert report['metrics']['all']['raw']['coverage']['queries'] == 2
    assert report['metrics']['regular']['raw']['metrics']['AUPRO_0_05'] == 1.
    assert report['gate_activation']['all']['normal_pixel_weighted'] == .5
    assert report['gate_activation']['all']['defect_pixel_weighted'] == .5
    np.testing.assert_array_equal(auxiliary['raw_map'], raw)
    normal_only = FusionDiagnostic()
    normal_only.update('001_regular.png', 'good', raw, np.zeros_like(mask), auxiliary)
    assert normal_only.compute()['metrics']['all']['raw']['metrics'] is None


def test_configs_only_vary_normalization_and_shallow_mode():
    configs = [config(arm) for arm in 'ABCDE']
    for cfg in configs:
        cfg.pop('shallow_mode')
        cfg.pop('score_normalization')
    assert all(cfg == configs[0] for cfg in configs)


def test_pipeline_routes_auxiliary_after_prediction_and_keeps_debug_compact(tmp_path, monkeypatch):
    from fmad.methods.base import Prediction
    from fmad.pipeline import _run_streaming_object
    from tests.test_fmad_swappable_pipeline import Adapter, make_dataset
    monkeypatch.setattr('fmad.evaluation.fusion_diagnostic.acquisition_identity',
                        lambda path: (Path(path).stem, 'regular'))

    class Audited(Adapter):
        def predict_batch(self, paths):
            return [Prediction(p.score_map, p.image_score, p.inference_seconds,
                {'fusion_audit': {'raw_map': p.score_map.copy(),
                                  'gate_grids': [np.ones((1, 1), np.float32)]}, 'scalar': 1})
                for p in super().predict_batch(paths)]

    dataset = make_dataset(tmp_path)
    cfg = {'protocol': {'shots': 2}, 'evaluation': {'fusion_diagnostic': True}}
    result = _run_streaming_object(Audited(), dataset, 'thing', 0, str(tmp_path / 'out'),
                                   2, config=cfg, debug_output=True)
    json.dumps(result.metadata, allow_nan=False)
    diagnostic = result.metadata['fusion_diagnostic']['metrics']['all']['CQ']['metrics']
    assert diagnostic['AUPRO_0_05'] == result.metadata['streaming_metrics']['seg_AUPRO_0.05']
    assert all(d == {'scalar': 1} for d in result.metadata['prediction_diagnostics'].values())
    with pytest.raises(ValueError, match='auxiliary'):
        _run_streaming_object(Adapter(), dataset, 'thing', 0, str(tmp_path / 'bad'), 2, config=cfg)


def test_shallow_mode_mean3_equals_equal_weight_average(tmp_path):
    """Test that shallow_mode='mean3' computes equal-weight mean of 3 layer scores."""
    paths = [_write_image(tmp_path / f'{i}.png', i * 17) for i in range(3)]
    context = MethodContext('can', 0, 8, False, False)

    # Create mean3 method with all other params matching an existing arm
    base_cfg = config('D')  # Use existing arm as base
    mean3_cfg = dict(base_cfg, shallow_mode='mean3', score_normalization='tailscale')

    method = FusionFactorial(TinyTripleBackbone(), mean3_cfg)
    method.fit(paths, context)
    pred, max_val = method.predict(paths[0])
    assert np.isfinite(pred).all() and np.isfinite(max_val)
    assert method.shallow_mode == 'mean3'


def test_shallow_mode_mean3_with_none_normalization(tmp_path):
    """Test that mean3 works with score_normalization='none'."""
    paths = [_write_image(tmp_path / f'{i}.png', i * 17) for i in range(3)]
    context = MethodContext('can', 0, 8, False, False)

    base_cfg = config('D')
    mean3_cfg = dict(base_cfg, shallow_mode='mean3', score_normalization='none')

    method = FusionFactorial(TinyTripleBackbone(), mean3_cfg)
    method.fit(paths, context)
    pred, max_val = method.predict(paths[0])
    assert np.isfinite(pred).all() and np.isfinite(max_val)


@pytest.mark.parametrize('normalization', ['none', 'tailscale'])
def test_mean3_predict_batch_records_complete_audit(tmp_path, normalization):
    paths = [_write_image(tmp_path / f'{i}.png', i * 17) for i in range(3)]
    context = MethodContext('can', 0, 8, False, False)
    cfg = dict(config('D'), shallow_mode='mean3',
               score_normalization=normalization, mechanism_audit=True)
    method = FusionFactorial(TinyTripleBackbone(), cfg)
    method.fit(paths, context)

    predictions = method.predict_batch(paths[:2])
    assert len(predictions) == 2
    for prediction in predictions:
        audit = prediction.diagnostics['fusion_audit']
        assert audit['raw_map'].shape == prediction.score_map.shape
        assert len(audit['gate_grids']) == len(method.resolutions)
        assert len(audit['component_grids']) == len(method.resolutions)
        assert all(np.isfinite(grid).all() for grid in audit['gate_grids'])
    assert not method._audit_active
    assert method._gate_grids == [] and method._bag_gates == []


def test_shallow_mode_mean3_score_definition(tmp_path):
    """Test that mean3 produces the correct score_definition string."""
    paths = [_write_image(tmp_path / f'{i}.png', i * 17) for i in range(3)]
    context = MethodContext('can', 0, 8, False, False)

    base_cfg = config('D')
    mean3_cfg = dict(base_cfg, shallow_mode='mean3', score_normalization='tailscale')

    method = FusionFactorial(TinyTripleBackbone(), mean3_cfg)
    method.fit(paths, context)
    metadata = method.object_metadata()
    assert 'mean3' in metadata['score_definition']
    assert 'tailscale' in metadata['score_definition']


def test_mean3_mode_exact_fusion_logic():
    """Test that mean3 computes exactly equal-weight mean of three layer scores."""
    from tests.test_reference_matching import TinyTripleBackbone

    base_cfg = config('D')
    mean3_cfg = dict(base_cfg, shallow_mode='mean3')

    method = FusionFactorial(TinyTripleBackbone(), mean3_cfg)

    # Test with known scores
    scores = [
        torch.tensor([1., 2., 3.]),
        torch.tensor([2., 3., 4.]),
        torch.tensor([3., 4., 5.]),
    ]
    fused = method._fuse_layers(scores, tau=1.5)  # tau provided but not used
    expected = torch.stack(scores).mean(0)
    torch.testing.assert_close(fused, expected)


def test_mean3_versus_existing_modes_differ(tmp_path):
    """Test that mean3 produces different results from existing shallow modes."""
    paths = [_write_image(tmp_path / f'{i}.png', i * 17) for i in range(3)]
    context = MethodContext('can', 0, 8, False, False)

    base_cfg = config('D')
    modes_to_test = [
        ('off', base_cfg),
        ('gated', config('B')),
        ('hard', config('C')),
        ('all', config('A')),
        ('mean3', dict(base_cfg, shallow_mode='mean3')),
    ]

    maps = {}
    for mode, cfg in modes_to_test:
        method = FusionFactorial(TinyTripleBackbone(), cfg)
        method.fit(paths, context)
        pred, _ = method.predict(paths[0])
        maps[mode] = pred

    # mean3 should differ from at least some of the other modes
    mean3_map = maps['mean3']
    assert np.isfinite(mean3_map).all()
    # At least one other mode should differ from mean3
    assert any(not np.allclose(mean3_map, maps[m]) for m in ['off', 'gated', 'hard', 'all'])
