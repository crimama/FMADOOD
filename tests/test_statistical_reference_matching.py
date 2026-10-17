import json

import numpy as np
import pytest
import torch
from scipy import stats

from fmad.evaluation.calibration_diagnostic import CalibrationDiagnostic
from fmad.methods.reference_matching import ReferenceMatching
from fmad.methods.reference_matching_fast import ReferenceMatchingFast
from fmad.methods.statistical_reference_matching import (
    NORMALIZERS, StatisticalReferenceMatching, _qn, fit_normalizer, transform_normalizer,
    geometric_axis_factors, quantile_scale,
)
from fmad.protocol import MethodContext
from tests.test_psma_mr_v2 import _write_image
from tests.test_reference_matching import TinyTripleBackbone


def config(**overrides):
    return dict(variant='R1', feature_layers=[5, 11, 23], resolutions=[8],
        bank_tokens_per_source=16, query_chunk_size=8, bank_bags=2, knn_k=1,
        score_normalization='tail', layer_fusion='residual_spike',
        layer_fusion_tau_mode='support_quantile', layer_fusion_gamma=.125,
        prediction_postprocess='none', ablation_shallow_mode='all',
        fast_reuse_calibration_bank=True, fast_shared_bag_distances=True,
        ablation_bank_audit=True, calibration_audit=True, **overrides)


def tail(sample):
    parent = ReferenceMatching(TinyTripleBackbone(), dict(variant='R1', feature_layers=[5, 11, 23]))
    return parent._tail_params(sample)


@pytest.mark.parametrize('name', [name for name in NORMALIZERS if name not in ('quantile_anchor', 'quantile_barycenter')])
@pytest.mark.parametrize('constant', [False, True])
def test_finite_monotonic_and_json_fit(name, constant):
    sample = (torch.full((256,), .2) if constant else
              torch.tensor(np.random.default_rng(13).gamma(4, .04, 512), dtype=torch.float32))
    t = tail(sample)
    p = fit_normalizer(t['sample'], name, t)
    json.dumps(p, allow_nan=False)
    query = torch.linspace(0, 2, 513)
    output = transform_normalizer(query, t, p)
    assert output.shape == query.shape
    assert torch.isfinite(output).all()
    assert torch.all(output[1:] >= output[:-1] - 1e-5)
    # Metadata uses scalar input, not just vector queries.
    assert torch.isfinite(transform_normalizer(torch.tensor(.25), t, p))


def test_upper_width_and_empirical_ties():
    sample = torch.linspace(.1, .5, 500)
    t = tail(sample)
    p = fit_normalizer(sample, 'upper_width', t)
    q80, q95 = torch.quantile(sample, torch.tensor([.8, .95]))
    torch.testing.assert_close(transform_normalizer(torch.stack([q80, q95]), t, p),
                               torch.tensor([0., 1.]), atol=1e-5, rtol=1e-5)
    repeated = torch.tensor([.1, .1, .2, .3])
    t = tail(repeated)
    p = fit_normalizer(repeated, 'empirical_survival', t)
    result = transform_normalizer(torch.tensor([.1, .2, .4, 1.]), t, p)
    expected = -torch.log(torch.tensor([1., 3/5, 1/5, 1/5]))
    torch.testing.assert_close(result, expected)


def test_qn_matches_pair_order_statistic():
    x = np.array([.1, .11, .2, .3, .5, .8])
    pairs = sorted(abs(x[i] - x[j]) for i in range(len(x)) for j in range(i + 1, len(x)))
    h = len(x) // 2 + 1
    expected = 2.219144465985076 * pairs[h * (h - 1) // 2 - 1]
    assert _qn(x)[0] == pytest.approx(expected)


def test_multi_quantile_scale_matches_q95_endpoint_and_image_groups(tmp_path, diverse_extract):
    target = np.array([.4, .65, .85, 1.])
    q = 2 * target
    assert quantile_scale(q, target) == pytest.approx(2.)
    supports = [_write_image(tmp_path / f's{i}.png', 17*i) for i in range(3)]
    context = MethodContext('can', 0, 8, False, False)
    for name in ('multi_quantile_scale', 'image_median_quantile_scale'):
        method = StatisticalReferenceMatching(TinyTripleBackbone(),
            config(statistical_normalization=name))
        method.fit(supports, context)
        for bags in method.bag_tails.values():
            for bag in bags:
                for tail in bag:
                    params = tail['statistical']
                    assert params['scale'] > 0
                    if name == 'image_median_quantile_scale':
                        assert len(params['source_scales']) == len(supports)


def test_gamma_and_lognormal_match_scipy_survival_and_extreme_gamma_is_finite():
    sample = torch.tensor(np.random.default_rng(4).gamma(3, .04, 800), dtype=torch.float32)
    t = tail(sample)
    query = torch.tensor([.01, .1, .4, 1.])
    for name in ('gamma_survival', 'lognormal_survival'):
        p = fit_normalizer(sample, name, t)
        actual = transform_normalizer(query, t, p).numpy()
        if name == 'gamma_survival':
            expected = -stats.gamma.logsf(query.numpy(), p['gamma_shape'], scale=p['gamma_scale'])
        else:
            expected = -stats.lognorm.logsf(query.numpy(), p['log_sigma'], scale=np.exp(p['log_mu']))
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    p = dict(kind='gamma_survival', gamma_shape=2., gamma_scale=1e-5)
    actual = transform_normalizer(torch.tensor([.01, .1, 2.]), t, p)
    assert torch.isfinite(actual).all() and torch.all(actual[1:] > actual[:-1])
    assert torch.isfinite(transform_normalizer(torch.tensor(2.), t, p))


@pytest.fixture
def diverse_extract(monkeypatch):
    def extract(self, image, resolution, pool=None):
        rng = torch.Generator().manual_seed(int(np.asarray(image).sum()) + resolution)
        return [torch.nn.functional.normalize(torch.randn(32, 24, generator=rng), dim=1)
                for _ in range(3)], (4, 8)
    monkeypatch.setattr(ReferenceMatchingFast, '_extract', extract)


@pytest.mark.parametrize('name', NORMALIZERS)
def test_end_to_end_native_audit_and_normal_only_fit(tmp_path, diverse_extract, name):
    supports = [_write_image(tmp_path / f's{i}.png', 17*i) for i in range(3)]
    query = _write_image(tmp_path / 'q.png', 81)
    method = StatisticalReferenceMatching(TinyTripleBackbone(), config(statistical_normalization=name))
    method.fit(supports, MethodContext('can', 0, 8, False, False))
    before = json.dumps(method.object_metadata(), sort_keys=True, allow_nan=False)
    prediction = method.predict_batch([query])[0]
    assert np.isfinite(prediction.score_map).all()
    auxiliary = prediction.diagnostics['calibration_audit']
    assert auxiliary['native_bags'][0].shape == (2, 2, 3, 4, 8)
    diagnostic = CalibrationDiagnostic()
    diagnostic.update(query, prediction.score_map, np.zeros_like(prediction.score_map, bool), auxiliary)
    assert len(diagnostic.compute()['native_bags']['samples']) == 6
    assert json.dumps(method.object_metadata(), sort_keys=True, allow_nan=False) == before
    assert method.records == {}
    if name == 'shrink_q95':
        for layer in range(3):
            ps = [b[layer]['statistical'] for b in method.bag_tails[8]]
            spread_before = np.ptp([np.log(p['unshrunk_scale']) for p in ps])
            spread_after = np.ptp([np.log(p['scale']) for p in ps])
            assert spread_after == pytest.approx(.5 * spread_before)


@pytest.mark.parametrize('name,parent_name', [('none', 'none'), ('q95', 'tailscale'),
                                            ('mad', 'mad'), ('tail', 'tail')])
def test_existing_controls_preserve_banks_and_predictions(tmp_path, diverse_extract, name, parent_name):
    supports = [_write_image(tmp_path / f's{i}.png', 17*i) for i in range(3)]
    query = _write_image(tmp_path / 'q.png', 81)
    context = MethodContext('can', 0, 8, False, False)
    cfg = config(statistical_normalization=name)
    actual = StatisticalReferenceMatching(TinyTripleBackbone(), cfg)
    expected = ReferenceMatchingFast(TinyTripleBackbone(),
        {**cfg, 'score_normalization': parent_name, 'calibration_audit': False})
    actual.fit(supports, context)
    expected.fit(supports, context)
    assert (actual.object_metadata()['ablation_bank_feature_sha256'] ==
            expected.object_metadata()['ablation_bank_feature_sha256'])
    np.testing.assert_allclose(actual.predict(query)[0], expected.predict(query)[0], atol=1e-6, rtol=1e-6)


def test_registry_and_invalid_configuration():
    from fmad.methods import register_methods
    from fmad.registry import METHOD_REGISTRY
    register_methods('statistical_reference_matching')
    assert METHOD_REGISTRY.get('statistical_reference_matching') is StatisticalReferenceMatching
    for key, value in [('statistical_normalization', 'unknown'), ('local_reliability', 'scale'),
                       ('tailscale_center_rho', .5), ('ablation_shallow_mode', 'gated')]:
        cfg = config()
        cfg[key] = value
        with pytest.raises(ValueError):
            StatisticalReferenceMatching(TinyTripleBackbone(), cfg)


def test_axis_factors_centering_separability_and_degenerate_fallback():
    a = np.array([.5, 1., 2.])
    c = np.array([.25, 1., 4., 1.])
    bags = np.array([.5, 1., 2., 1.])
    q = .2 * c[:, None, None] * bags[None, :, None] * a[None, None, :]
    factors = geometric_axis_factors(q)
    np.testing.assert_allclose(factors['layer'], a)
    np.testing.assert_allclose(factors['resolution'], c)
    assert factors['observations'] == 48
    assert np.exp(np.log(factors['layer']).mean()) == pytest.approx(1.)
    assert np.exp(np.log(factors['resolution']).mean()) == pytest.approx(1.)
    for q in (np.zeros((4, 4, 3)), np.full((4, 4, 3), .2)):
        f = geometric_axis_factors(q)
        np.testing.assert_allclose(f['layer'], 1.)
        np.testing.assert_allclose(f['resolution'], 1.)
    q[0, 0, 0] = np.nan
    f = geometric_axis_factors(q)
    assert f['fallback'] == 'nonfinite_reference_using_identity'
    np.testing.assert_allclose(f['layer'], 1.)


@pytest.mark.parametrize('name', ['axis_layer', 'axis_resolution', 'axis_layer_resolution'])
def test_axis_fit_applies_only_requested_factors(tmp_path, diverse_extract, name):
    supports = [_write_image(tmp_path / f's{i}.png', 17*i) for i in range(3)]
    cfg = config(statistical_normalization=name)
    cfg['resolutions'] = [8, 16]
    method = StatisticalReferenceMatching(TinyTripleBackbone(), cfg)
    method.fit(supports, MethodContext('can', 0, 8, False, False))
    f = method.axis_factors
    scales = []
    for ri, (r, bags) in enumerate(method.bag_tails.items()):
        for bi, bag in enumerate(bags):
            for li, t in enumerate(bag):
                expected = (f['layer'][li] if name != 'axis_resolution' else 1.)
                expected *= f['resolution'][ri] if name != 'axis_layer' else 1.
                p = t['statistical']
                assert p['scale'] == pytest.approx(expected)
                assert float(method.bag_scales[r][bi][li]) == pytest.approx(expected)
                torch.testing.assert_close(method._tail_transform(t, torch.tensor(.4)),
                                           torch.tensor(.4 / expected))
                scales.append(p['scale'])
    assert np.exp(np.log(scales).mean()) == pytest.approx(1.)


@pytest.mark.parametrize('name', ['quantile_anchor', 'quantile_barycenter'])
def test_cross_observation_transport_is_support_fitted_and_aligns_knots(tmp_path, diverse_extract, name):
    supports = [_write_image(tmp_path / f's{i}.png', 17*i) for i in range(3)]
    query = _write_image(tmp_path / 'q.png', 81)
    cfg = config(statistical_normalization=name)
    cfg['resolutions'] = [8, 16]
    method = StatisticalReferenceMatching(TinyTripleBackbone(), cfg)
    method.fit(supports, MethodContext('can', 0, 8, False, False))
    metadata_before = json.dumps(method.object_metadata(), sort_keys=True, allow_nan=False)
    fitted = [t for bags in method.bag_tails.values() for bag in bags for t in bag]
    assert len(fitted) == 12
    assert all(t['statistical']['barycenter_observations'] == len(fitted) for t in fitted)
    for t in fitted:
        p = t['statistical']
        assert p['fallback'] is None
        source = torch.tensor(p['source_knots'], dtype=torch.float64) * p['scale']
        mapped = transform_normalizer(source, t, p)
        np.testing.assert_allclose(mapped.numpy(), p['target_knots'], atol=1e-6)
        query_distances = torch.linspace(0, 5, 1024)
        scores = transform_normalizer(query_distances, t, p)
        assert torch.isfinite(scores).all()
        assert torch.all(scores[1:] >= scores[:-1] - 1e-6)
    assert np.isfinite(method.predict(query)[0]).all()
    assert json.dumps(method.object_metadata(), sort_keys=True, allow_nan=False) == metadata_before


@pytest.mark.parametrize('alpha_l,alpha_r,control', [
    (0., 0., 'none'), (1., 0., 'axis_layer'),
    (0., 1., 'axis_resolution'), (1., 1., 'axis_layer_resolution'),
    (.5, 0., None), (0., .5, None), (.5, .5, None)])
def test_partial_axes_same_bank_scoring(tmp_path, diverse_extract, alpha_l, alpha_r, control):
    supports = [_write_image(tmp_path / f's{i}.png', 17*i) for i in range(3)]
    query = _write_image(tmp_path / 'q.png', 81)
    context = MethodContext('can', 0, 8, False, False)
    cfg = config(statistical_normalization='axis_layer_resolution')
    cfg['resolutions'] = [8, 16]
    actual = StatisticalReferenceMatching(TinyTripleBackbone(),
        {**cfg, 'axis_alpha_layer': alpha_l, 'axis_alpha_resolution': alpha_r})
    expected = StatisticalReferenceMatching(TinyTripleBackbone(),
        {**cfg, 'statistical_normalization': control or 'none'})
    actual.fit(supports, context)
    expected.fit(supports, context)
    assert actual.object_metadata()['ablation_bank_feature_sha256'] == expected.object_metadata()['ablation_bank_feature_sha256']
    factors = actual.axis_factors
    for ri, (r, bags) in enumerate(actual.bag_tails.items()):
        for bi, bag in enumerate(bags):
            for li, t in enumerate(bag):
                scale = factors['layer'][li] ** alpha_l * factors['resolution'][ri] ** alpha_r
                assert np.isfinite(scale) and scale > 0
                assert t['statistical']['scale'] == pytest.approx(scale)
                assert float(actual.bag_scales[r][bi][li]) == pytest.approx(scale)
                if control is None:
                    # Oracle: same bank, sqrt factors independent of exponent code.
                    oracle = np.sqrt(factors['layer'][li] if alpha_l else 1.)
                    oracle *= np.sqrt(factors['resolution'][ri] if alpha_r else 1.)
                    expected.bag_tails[r][bi][li]['statistical']['scale'] = oracle
                    expected.bag_scales[r][bi][li] = oracle
    before = json.dumps(actual.object_metadata(), sort_keys=True, allow_nan=False)
    np.testing.assert_array_equal(actual.predict(query)[0], expected.predict(query)[0])
    assert json.dumps(actual.object_metadata(), sort_keys=True, allow_nan=False) == before


@pytest.mark.parametrize('value', [-.01, 1.01, float('nan'), float('inf')])
@pytest.mark.parametrize('key', ['axis_alpha_layer', 'axis_alpha_resolution'])
def test_invalid_axis_exponents_are_rejected(value, key):
    cfg = config(statistical_normalization='axis_layer_resolution')
    cfg[key] = value
    with pytest.raises(ValueError, match='exponents'):
        StatisticalReferenceMatching(TinyTripleBackbone(), cfg)


def test_partial_axis_identity_fallback(tmp_path, diverse_extract, monkeypatch):
    import fmad.methods.statistical_reference_matching as module
    original = module.geometric_axis_factors
    def nonfinite(q95):
        values = q95.copy()
        values[0, 0, 0] = np.nan
        return original(values)
    monkeypatch.setattr(module, 'geometric_axis_factors', nonfinite)
    supports = [_write_image(tmp_path / f's{i}.png', 17*i) for i in range(3)]
    method = StatisticalReferenceMatching(TinyTripleBackbone(), config(
        statistical_normalization='axis_layer_resolution',
        axis_alpha_layer=.5, axis_alpha_resolution=.5))
    method.fit(supports, MethodContext('can', 0, 8, False, False))
    assert method.axis_factors['fallback'] == 'nonfinite_reference_using_identity'
    assert all(t['statistical']['scale'] == 1.
               for bags in method.bag_tails.values() for bag in bags for t in bag)
