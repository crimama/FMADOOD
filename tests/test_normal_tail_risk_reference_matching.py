import json

import numpy as np
import pytest

from fmad.methods.normal_tail_risk_reference_matching import (
    NormalTailRiskReferenceMatching, fit_tail_risk, empirical_cvar,
)
from fmad.methods.statistical_reference_matching import StatisticalReferenceMatching
from fmad.protocol import MethodContext
from tests.test_statistical_reference_matching import config, diverse_extract
from tests.test_reference_matching import TinyTripleBackbone
from tests.test_psma_mr_v2 import _write_image


def test_discrete_cvar_uses_empirical_quantile_not_linear_interpolation():
    assert empirical_cvar([1.,2.],.95)==pytest.approx(2.)
    assert empirical_cvar([1.,2.,3.,4.],.5)==pytest.approx(3.5)


@pytest.mark.parametrize('fixed',[False,True])
@pytest.mark.parametrize('excess',[False,True])
def test_lp_tail_risk_reduces_empirical_risk_and_respects_constraints(fixed,excess):
    z=np.random.default_rng(3).gamma(3,.4,(3,19,4,3))
    weights,audit=fit_tail_risk(z,fixed_layer=fixed,excess=excess)
    assert weights.min()>=.125-1e-10 and weights.max()<=8+1e-10
    assert audit['fitted_risk']<=audit['baseline_risk']+1e-8
    if fixed:
        np.testing.assert_allclose(weights.mean(0),1.,atol=1e-10)
    else:
        assert weights.mean()==pytest.approx(1.)
    json.dumps(audit,allow_nan=False)
    # Source order cannot change the optimum risk.
    _,reverse=fit_tail_risk(z[::-1],fixed_layer=fixed,excess=excess)
    assert reverse['fitted_risk']==pytest.approx(audit['fitted_risk'],abs=1e-8)


def test_zero_excess_preserves_identity():
    weights,audit=fit_tail_risk(np.zeros((2,16,4,3)),excess=True)
    np.testing.assert_array_equal(weights,np.ones((4,3)))
    assert audit['fallback']=='zero_signal_retains_Q95'


@pytest.mark.parametrize('mode',['off','score_free','score_fixed_layer','excess_free','excess_fixed_layer'])
def test_normal_only_fit_preserves_banks_and_prediction_state(tmp_path,diverse_extract,mode):
    supports=[_write_image(tmp_path/f's{i}.png',17*i) for i in range(3)]
    query=_write_image(tmp_path/'query.png',81)
    cfg=config(statistical_normalization='q95',tail_probe_side=2,tail_risk_mode=mode)
    method=NormalTailRiskReferenceMatching(TinyTripleBackbone(),cfg)
    control=StatisticalReferenceMatching(TinyTripleBackbone(),config(statistical_normalization='q95'))
    context=MethodContext('can',0,8,False,False)
    method.fit(supports,context);control.fit(supports,context)
    meta=method.object_metadata()
    assert meta['ablation_bank_feature_sha256']==control.object_metadata()['ablation_bank_feature_sha256']
    assert method._probe_features=={} and method._probe_bags=={} and method.records=={}
    before=json.dumps(meta,sort_keys=True,allow_nan=False)
    actual=method.predict_batch([query])[0].score_map
    assert np.isfinite(actual).all()
    assert before==json.dumps(method.object_metadata(),sort_keys=True,allow_nan=False)
    if mode=='off':
        np.testing.assert_allclose(actual,control.predict_batch([query])[0].score_map,atol=1e-6,rtol=1e-6)
    if mode.endswith('fixed_layer'):
        np.testing.assert_allclose(np.mean(meta['normal_tail_risk_calibration']['multipliers'],axis=0),1.,atol=1e-10)


def test_registry_and_one_shot_rejected():
    from fmad.methods import register_methods
    from fmad.registry import METHOD_REGISTRY
    register_methods('normal_tail_risk_reference_matching')
    assert METHOD_REGISTRY.get('normal_tail_risk_reference_matching') is NormalTailRiskReferenceMatching
    method=NormalTailRiskReferenceMatching(TinyTripleBackbone(),config(statistical_normalization='q95'))
    with pytest.raises(ValueError,match='>=2'):
        method.fit(['normal.png'],MethodContext('can',0,8,False,False))


def test_four_resolution_fit_learns_exactly_twelve_shared_bag_scales(tmp_path,diverse_extract):
    supports=[_write_image(tmp_path/f'n{i}.png',i*19) for i in range(2)]
    cfg=config(statistical_normalization='q95',tail_probe_side=2,
               tail_risk_mode='excess_fixed_layer')
    cfg['resolutions']=[8,12,16,20]
    method=NormalTailRiskReferenceMatching(TinyTripleBackbone(),cfg)
    method.fit(supports,MethodContext('can',0,8,False,False))
    audit=method.object_metadata()['normal_tail_risk_calibration']
    assert np.asarray(audit['multipliers']).shape==(4,3)
    assert audit['effective_calibration_degrees_of_freedom']==9
    assert audit['source_count']==2
    for ri,r in enumerate(cfg['resolutions']):
        for bag in method.bag_tails[r]:
            np.testing.assert_allclose([t['statistical']['fusion_multiplier'] for t in bag],
                                       audit['multipliers'][ri])
