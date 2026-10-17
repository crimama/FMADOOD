import numpy as np
import pytest
import torch

from fmad.methods.reference_matching import ReferenceMatching
from fmad.methods.reference_matching_fast import ReferenceMatchingFast
from fmad.evaluation.calibration_diagnostic import CalibrationDiagnostic
from fmad.protocol import MethodContext
from tests.test_reference_matching import TinyTripleBackbone
from tests.test_psma_mr_v2 import _write_image


@pytest.mark.parametrize('rho', [0., .5, 1.])
def test_centered_normalization_reference_fast_and_native_diagnostic(tmp_path, monkeypatch, rho):
    def extract(self, image, resolution, pool=None):
        # Diverse unit features keep the comparison away from the denominator
        # floor; the floor and negative-score case is tested separately.
        rng = torch.Generator().manual_seed(int(np.asarray(image).sum()) + resolution)
        return [torch.nn.functional.normalize(torch.randn(16, 32, generator=rng), dim=1)
                for _ in range(3)], (4, 4)
    monkeypatch.setattr(ReferenceMatching, '_extract', extract)
    monkeypatch.setattr(ReferenceMatchingFast, '_extract', extract)
    cfg = dict(variant='R1', feature_layers=[5, 11, 23], resolutions=[8],
        bank_tokens_per_source=4, query_chunk_size=4, bank_bags=2, knn_k=1,
        score_normalization='tailscale', tailscale_center_rho=rho,
        layer_fusion='residual_spike', layer_fusion_tau_mode='support_quantile',
        layer_fusion_gamma=.125, prediction_postprocess='none')
    paths = [_write_image(tmp_path / f's{i}.png', i * 17) for i in range(3)]
    query = _write_image(tmp_path / 'q.png', 81)
    context = MethodContext('can', 0, 8, False, False)
    reference = ReferenceMatching(TinyTripleBackbone(), cfg)
    fast = ReferenceMatchingFast(TinyTripleBackbone(), {**cfg,
        'fast_reuse_calibration_bank': True, 'fast_shared_bag_distances': True,
        'ablation_shallow_mode': 'gated', 'calibration_audit': True})
    reference.fit(paths, context)
    fast.fit(paths, context)
    expected = reference.predict(query)[0]
    actual = fast.predict_batch([query])[0]
    np.testing.assert_allclose(actual.score_map, expected, atol=1e-6, rtol=1e-6)
    aux = actual.diagnostics['calibration_audit']
    for b, bag in enumerate(fast.bag_tails[8]):
        for l, tail in enumerate(bag):
            q50 = torch.quantile(tail['sample'], .5)
            scale = fast.bag_scales[8][b][l]
            torch.testing.assert_close(scale, (tail['u'] - rho * q50).clamp_min(1e-4))
            raw = torch.from_numpy(aux['native_bags'][0][b, 0, l])
            norm = aux['native_bags'][0][b, 1, l]
            np.testing.assert_allclose(norm, ((raw - rho * q50) / scale).numpy(), atol=1e-6, rtol=1e-6)
            q99 = torch.quantile(tail['sample'], fast.layer_fusion_tau_q)
            assert tail['spike_tau'] == pytest.approx(float((q99-rho*q50)/scale), abs=1e-6)
    diagnostic = CalibrationDiagnostic()
    diagnostic.update(query, actual.score_map, np.zeros_like(actual.score_map, bool), aux)
    native = diagnostic.compute()['native_bags']
    assert len(native['samples']) == 6
    assert all(v['images'] == 1 for v in native['counts'].values())
    assert fast._calibration_native_bags is None
    if rho == 0:
        original = ReferenceMatching(TinyTripleBackbone(), {k: v for k, v in cfg.items() if k != 'tailscale_center_rho'})
        original.fit(paths, context)
        np.testing.assert_array_equal(expected, original.predict(query)[0])


def test_negative_scores_remain_and_zero_width_is_finite():
    params = {'sample': torch.tensor([.2, .2, .2]), 'u': torch.tensor(.2)}
    method = ReferenceMatching(TinyTripleBackbone(), dict(variant='R1',
        feature_layers=[5, 11, 23], score_normalization='tailscale', tailscale_center_rho=1.))
    method._set_tail_center(params)
    result = method._centered_tail_score(torch.tensor([.1, .2, .3]), params, method._tailscale_divisor(params))
    assert torch.isfinite(result).all()
    assert result[0] < 0 and result[1] == 0 and result[2] > 0


@pytest.mark.parametrize('rho', [True, -1, 1.1, float('nan')])
def test_invalid_rho(rho):
    with pytest.raises(ValueError, match='tailscale_center_rho'):
        ReferenceMatching(TinyTripleBackbone(), dict(variant='R1', tailscale_center_rho=rho))
