import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
import torch

from fmad.evaluation.source_shift import (SourceShiftDiagnostic, operating_stats,
    region_moments, select_calibration, threshold_code)
from fmad.evaluation.streaming import _ordered_float16_codes, _label8
from fmad.methods.base import Prediction
from fmad.methods.fusion_factorial import FusionFactorial
from fmad.protocol import MethodContext
from tests.test_fusion_factorial import config
from tests.test_reference_matching import TinyTripleBackbone
from tests.test_psma_mr_v2 import _write_image


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_large_native_projection_matches_prediction_backend(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    generator = torch.Generator().manual_seed(17)
    grid = torch.rand((80, 160), generator=generator).to(device)
    raw = torch.nn.functional.interpolate(grid[None, None], size=(1600, 3200),
        mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
    auxiliary = {'raw_map': raw, 'component_device': device,
        'component_grids': [{'deep': grid.cpu().numpy(),
                            'shallow_used': np.zeros((80, 160), np.float32)}]}
    audit = SourceShiftDiagnostic({s: {'0.01': 32768} for s in ('raw', 'CQ')}, {})
    audit.update('001_regular.png', 'good', raw, np.zeros(raw.shape, bool), auxiliary)
    assert audit.max_reconstruction_error == 0.


def test_histogram_threshold_strict_ties_and_operating_counts():
    values = np.arange(100, dtype=np.float32).reshape(10, 10)
    hist = np.bincount(_ordered_float16_codes(values).reshape(-1), minlength=65536)
    code = threshold_code(hist, .05)
    mask = values >= 90
    labels, count = _label8(mask)
    result = operating_stats(values, mask, labels, count, {'t': code})['t']
    assert result['FPR'] == 0 and result['TPR'] == result['PRO'] == .5
    assert result['true_positive_pixels'] == 5
    tied = np.bincount(_ordered_float16_codes(np.ones(100)).reshape(-1), minlength=65536)
    assert tied[threshold_code(tied, .01) + 1:].sum() == 0
    assert region_moments(values, np.zeros_like(mask))['defect_minus_normal'] is None


def test_pro_is_region_balanced_not_pixel_tpr():
    mask = np.zeros((5, 5), bool)
    mask[0, 0] = True
    mask[3:, 3:] = True
    score = np.zeros((5, 5), np.float32)
    score[0, 0] = 2
    labels, count = _label8(mask)
    code = int(_ordered_float16_codes(np.array([1.]))[0])
    result = operating_stats(score, mask, labels, count, {'t': code})['t']
    assert count == 2 and result['TPR'] == .2 and result['PRO'] == .5


def normal_paths(tmp_path, n=18):
    directory = tmp_path / 'can' / 'train' / 'good'
    directory.mkdir(parents=True)
    return [_write_image(directory / f'{i:03d}_regular.png', i * 7) for i in range(n)]


def test_calibration_excludes_actual_support_and_preserves_method_state(tmp_path):
    paths = normal_paths(tmp_path)
    support = paths[5:7]
    chosen = select_calibration(paths, support, 'can')
    assert len(chosen) == 16 and not set(chosen).intersection(support)

    class Method:
        mechanism_audit = True
        calls = []
        def predict_batch(self, paths):
            self.calls.extend(paths)
            score = np.arange(12, dtype=np.float32).reshape(3, 4)
            return [Prediction(score, 11., .01, {'fusion_audit': {'raw_map': score.copy()}})]

    method = Method()
    audit = SourceShiftDiagnostic.calibrate(method, paths, support, 'can')
    assert method.calls == chosen
    assert len(audit.calibration['thresholds']['CQ']['0.01']['leave_one_image_out_values']) == 16
    assert audit.calibration['thresholds']['CQ']['0.01']['calibration_FPR'] <= .01
    assert audit.thresholds['raw'] == audit.thresholds['CQ']
    # Byte-identical duplicate support content is rejected even under another name.
    duplicate = Path(chosen[0])
    Image.open(support[0]).save(duplicate)
    with pytest.raises(ValueError, match='duplicate content'):
        select_calibration(paths, support, 'can')
    with pytest.raises(ValueError, match='insufficient'):
        select_calibration(paths[:5], paths[:1], 'can')


@pytest.mark.parametrize('arm', list('BCDE'))
def test_instrumentation_exact_parity_components_and_query_cleanup(tmp_path, arm):
    paths = [_write_image(tmp_path / f'{i}.png', i * 17) for i in range(3)]
    method = FusionFactorial(TinyTripleBackbone(), config(arm))
    method.fit(paths, MethodContext('can', 0, 8, False, False))
    expected = method.predict_batch(paths[:1])[0].score_map
    bank_hash = method.object_metadata()['bank_feature_sha256']
    method.mechanism_audit = True
    prediction = method.predict_batch(paths[:1])[0]
    np.testing.assert_array_equal(prediction.score_map, expected)
    assert method.object_metadata()['bank_feature_sha256'] == bank_hash
    aux = prediction.diagnostics['fusion_audit']
    assert len(aux['component_grids']) == 2 and len(aux['component_grids'][0]) == 10
    thresholds = {s: {'0.01': 32768} for s in ('raw', 'CQ')}
    audit = SourceShiftDiagnostic(thresholds, {})
    mask = np.zeros(expected.shape, bool)
    mask[1:3, 1:3] = True
    for condition in ('regular', 'shift_1'):
        audit.update(f'001_{condition}.png', 'bad', expected, mask, aux)
    report = audit.compute()
    json.dumps(report, allow_nan=False)
    assert report['component_reconstruction_max_abs_error'] < 1e-3
    changes = report['paired_regular_shift1'][0]['component_mean_change']
    assert all(change == 0 for row in changes.values() for change in row.values())
    if arm == 'E':
        assert report['records'][0]['components']['shallow_used']['normal']['mean'] == 0
    assert method._component_bags == method._component_grids == []
    with pytest.raises(FileNotFoundError):
        method.predict_batch([str(tmp_path / 'missing.png')])
    assert method._component_bags == method._component_grids == [] and not method._audit_active


def test_pairing_fails_closed_and_shift_changes_are_after_prediction():
    x = np.array([[0., 1.], [2., 3.]], np.float32)
    mask = np.array([[0, 0], [1, 1]], bool)
    code = int(_ordered_float16_codes(np.array([1.]))[0])
    audit = SourceShiftDiagnostic({s: {'0.01': code} for s in ('raw', 'CQ')}, {})
    aux = {'raw_map': x, 'component_grids': [{'deep': x, 'shallow_used': np.zeros_like(x)}]}
    audit.update('001_regular.png', 'bad', x, mask, aux)
    with pytest.raises(ValueError, match='not paired'):
        audit.compute()
    shifted = {'raw_map': x + 1, 'component_grids': [{'deep': x + 1, 'shallow_used': np.zeros_like(x)}]}
    audit.update('001_shift_1.png', 'bad', x + 1, mask, shifted)
    result = audit.compute()
    assert result['paired_regular_shift1'][0]['component_mean_change']['deep']['normal'] == 1.
    assert result['paired_regular_shift1'][0]['operating_change']['CQ']['0.01']['FPR'] == .5
    assert result['condition_operating_points']['shift_1']['all']['CQ']['0.01']['PRO'] == 1.


def test_common_pipeline_calibrates_before_queries_without_changing_predictions(tmp_path):
    from fmad.datasets.mvtec_ad2 import MVTecAD2Dataset
    from fmad.pipeline import _run_streaming_object
    normal_paths(tmp_path)
    for kind, scene in (('good', '101'), ('bad', '102')):
        directory = tmp_path / 'can' / 'test_public' / kind
        directory.mkdir(parents=True)
        for condition in ('regular', 'shift_1'):
            _write_image(directory / f'{scene}_{condition}.png', 170)
            if kind == 'bad':
                gt = tmp_path / 'can' / 'test_public' / 'ground_truth' / kind
                gt.mkdir(parents=True, exist_ok=True)
                mask = np.zeros((12, 16), np.uint8)
                mask[1:3, 1:3] = 255
                Image.fromarray(mask).save(gt / f'{scene}_{condition}_mask.png')
    dataset = MVTecAD2Dataset(str(tmp_path), {'preprocess': 'no_mask_no_rotation'})
    cfg = {'protocol': {'shots': 2, 'warmup_iters': 0},
           'evaluation': {'fusion_diagnostic': True, 'source_shift_diagnostic': True}}
    method = FusionFactorial(TinyTripleBackbone(), dict(config('D'), mechanism_audit=True))
    result = _run_streaming_object(method, dataset, 'can', 0, str(tmp_path / 'audited'), 1, config=cfg)
    report = result.metadata['source_shift_diagnostic']
    assert report['calibration']['count'] == 16 and len(report['records']) == 4
    assert len(report['paired_regular_shift1']) == 2
    assert not set(report['calibration']['paths']).intersection(result.metadata['protocol']['support_paths'])
    json.dumps(result.metadata, allow_nan=False)
    ref = FusionFactorial(TinyTripleBackbone(), config('D'))
    baseline = _run_streaming_object(ref, dataset, 'can', 0, str(tmp_path / 'ref'), 1,
        config={'protocol': cfg['protocol']})
    assert result.metadata['streaming_metrics'] == baseline.metadata['streaming_metrics']
    assert result.metadata['bank_feature_sha256'] == baseline.metadata['bank_feature_sha256']
