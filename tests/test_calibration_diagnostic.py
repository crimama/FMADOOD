import numpy as np
import pytest

from fmad.evaluation.calibration_diagnostic import CalibrationDiagnostic
from fmad.methods.reference_matching_fast import ReferenceMatchingFast
from fmad.protocol import MethodContext
from tests.test_reference_matching import TinyTripleBackbone
from tests.test_psma_mr_v2 import _write_image


@pytest.mark.parametrize('normalization', ['none', 'tailscale'])
def test_audit_preserves_predictions_and_reconstructs_fusion(tmp_path, normalization):
    cfg = dict(variant='R1', feature_layers=[5, 11, 23], resolutions=[8, 12],
        bank_tokens_per_source=4, query_chunk_size=4, bank_bags=2, knn_k=1,
        score_normalization=normalization, layer_fusion='residual_spike',
        layer_fusion_tau_mode='support_quantile', layer_fusion_gamma=.125,
        ablation_shallow_mode='all', prediction_postprocess='none',
        fast_reuse_calibration_bank=True, fast_shared_bag_distances=True)
    support = [_write_image(tmp_path / f's{i}.png', i * 17) for i in range(3)]
    query = _write_image(tmp_path / 'query.png', 81)
    context = MethodContext('can', 0, 8, False, False)
    base = ReferenceMatchingFast(TinyTripleBackbone(), cfg)
    audit = ReferenceMatchingFast(TinyTripleBackbone(), {**cfg, 'calibration_audit': True})
    base.fit(support, context)
    audit.fit(support, context)
    expected = base.predict_batch([query])[0]
    actual = audit.predict_batch([query])[0]
    np.testing.assert_array_equal(actual.score_map, expected.score_map)
    assert actual.image_score == expected.image_score
    aux = actual.diagnostics['calibration_audit']
    import torch
    import torch.nn.functional as F
    native = []
    for raw, normalized in aux['grids']:
        if normalization == 'none':
            np.testing.assert_array_equal(raw, normalized)
        combined = normalized[1:].mean(0) + .125 * normalized[0]
        native.append(F.interpolate(torch.from_numpy(combined)[None, None],
            size=actual.score_map.shape, mode='bilinear', align_corners=False)[0, 0].numpy())
    np.testing.assert_allclose(np.mean(native, axis=0), actual.score_map, atol=1e-6, rtol=1e-6)
    diagnostic = CalibrationDiagnostic()
    mask = np.zeros_like(actual.score_map, dtype=bool)
    mask[:4, :4] = True
    diagnostic.update(query, actual.score_map, mask, aux)
    result = diagnostic.compute()
    assert result['labels_used_only_after_prediction']
    assert set(result['samples']) == {'normal', 'defect'}
    assert len(result['columns']) == 13
    assert audit._calibration_grids is None
    with pytest.raises(FileNotFoundError):
        audit.predict_batch([str(tmp_path / 'missing.png')])
    assert audit._calibration_grids is None


def test_samples_are_paired_and_reservoir_bounded():
    diagnostic = CalibrationDiagnostic()
    raw = np.arange(16, dtype=np.float32).reshape(4, 4)
    grid = np.stack([np.stack([raw] * 3), np.stack([raw / 2] * 3)])
    for index in range(12):
        diagnostic.update(f'/good/{index}.png', np.ones((32, 32), np.float32),
            np.zeros((32, 32), bool), {'grids': [grid], 'resolutions': [448], 'layers': [5, 11, 23]})
    result = diagnostic.compute()
    values = np.array(result['samples']['normal'])
    assert values.shape == (2048, 7)
    np.testing.assert_allclose(values[:, 0] / 2, values[:, 3])
