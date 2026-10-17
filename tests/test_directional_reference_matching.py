import math

import numpy as np
import pytest
import torch
from scipy import special,stats

from fmad.methods.directional_reference_matching import (
    DirectionalReferenceMatching,fit_directional_concentration,
    log_vmf_uniform_ratio_constant,
)
from fmad.methods.statistical_reference_matching import StatisticalReferenceMatching
from fmad.protocol import MethodContext
from tests.test_psma_mr_v2 import _write_image
from tests.test_reference_matching import TinyTripleBackbone
from tests.test_statistical_reference_matching import config


@pytest.mark.parametrize('kappa',[0.,.001,1.,100.,10000.])
def test_normalizer_matches_analytic_3d_sphere(kappa):
    expected=0. if kappa==0 else (math.log(kappa)-kappa+math.log(2.)
                                -math.log(-math.expm1(-2*kappa)))
    assert log_vmf_uniform_ratio_constant(3,kappa)==pytest.approx(expected,abs=1e-9)


def test_high_dimensional_bessel_underflow_and_uniform_limit():
    assert log_vmf_uniform_ratio_constant(1024,.001)==pytest.approx(-.001**2/2048,abs=1e-12)
    assert np.isfinite(log_vmf_uniform_ratio_constant(1024,1e7))


def test_mle_recovers_known_directional_concentration():
    rng=np.random.default_rng(6)
    mean=np.array([1.,0.,0.])
    for kappa in [3.,30.]:
        x=stats.vonmises_fisher(mean,kappa).rvs(3000,random_state=rng)
        fitted,audit=fit_directional_concentration([x[:,:1]],3)
        assert fitted==pytest.approx(kappa,rel=.12)
        assert audit['normal_heldout_nll']<0


@pytest.mark.parametrize('mode',['mixture','concentration_nn'])
def test_same_bank_identity_and_brute_force_query_score(tmp_path,mode):
    paths=[_write_image(tmp_path/f'{i}.png',i*37) for i in range(2)]
    cfg=config(statistical_normalization='q95');cfg['calibration_audit']=False
    context=MethodContext('can',0,8,False,False)
    baseline=StatisticalReferenceMatching(TinyTripleBackbone(),cfg)
    baseline.fit(paths,context)
    model=DirectionalReferenceMatching(TinyTripleBackbone(),dict(cfg,directional_score=mode))
    model.fit(paths,context)
    assert model.object_metadata()['ablation_bank_feature_sha256']==baseline.object_metadata()['ablation_bank_feature_sha256']
    q=[torch.nn.functional.normalize(torch.randn(9,x.shape[1]),dim=1)
       for x in model.bag_banks[8][0]['layers']]
    result=model._score_tokens(q,(3,3),model.bag_banks[8])
    expected=[]
    for bank in model.bag_banks[8]:
        scores=[]
        for li,(query,b) in enumerate(zip(q,bank['layers'])):
            p=model.directional_parameters[8][li]; k=p['concentration']
            sims=query.double() @ b.double().T
            if mode=='mixture':
                score=-p['log_uniform_ratio_constant']-torch.logsumexp(k*sims,dim=1)+math.log(len(b))
            else: score=k*(1-sims.amax(1))
            scores.append(score)
        expected.append((scores[1]+scores[2])/2+.125*scores[0])
    torch.testing.assert_close(result.double(),torch.stack(expected).mean(0),rtol=2e-5,atol=2e-4)
    before=[dict(p) for p in model.directional_parameters[8]]
    amap,score=model.predict(paths[0])
    assert np.isfinite(amap).all() and np.isfinite(score)
    assert model.directional_parameters[8]==before
