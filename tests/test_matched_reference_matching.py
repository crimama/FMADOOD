import numpy as np
import pytest
import torch

from fmad.methods.matched_reference_matching import MatchedReferenceMatching
from fmad.methods.statistical_reference_matching import StatisticalReferenceMatching
from fmad.protocol import MethodContext
from tests.test_reference_matching import TinyTripleBackbone
from tests.test_psma_mr_v2 import _write_image
from tests.test_statistical_reference_matching import config


@pytest.mark.parametrize('mode', ['matched', 'pooled_control'])
def test_fold_banks_divisors_and_scores_equal_physical_exclusion(tmp_path, mode):
    paths = [_write_image(tmp_path / f'{i}.png', i * 37) for i in range(2)]
    cfg = config(statistical_normalization='q95')
    cfg['calibration_audit'] = False
    context = MethodContext('can', 0, 8, False, False)
    baseline = StatisticalReferenceMatching(TinyTripleBackbone(), cfg)
    baseline.fit(paths, context)
    model = MatchedReferenceMatching(TinyTripleBackbone(), dict(cfg, reference_calibration=mode))
    model.fit(paths, context)
    assert model.full_bank_feature_sha256 == baseline.object_metadata()['ablation_bank_feature_sha256']
    r = 8
    assert model.bank_bags == 4
    for fi, bank in enumerate(model.bag_banks[r]):
        bi, source = divmod(fi, 2)
        original = baseline.bag_banks[r][bi]
        owner = torch.as_tensor(original['owners'])
        assert set(bank['owners']) == {1-source}
        assert all(k[0] != source for k in bank['token_keys'])
        for li, layer in enumerate(bank['layers']):
            torch.testing.assert_close(layer, original['layers'][li][owner != source])
            query = original['layers'][li][owner == source]
            sample = (1-query @ layer.T).amin(1)
            divisor = (float(torch.quantile(sample, .95).clamp_min(1e-4)) if mode=='matched'
                       else baseline.bag_tails[r][bi][li]['statistical']['scale'])
            assert model.bag_tails[r][fi][li]['statistical']['scale'] == pytest.approx(divisor, abs=2e-6)
    query = [torch.nn.functional.normalize(torch.randn(9, x.shape[1]), dim=1)
             for x in model.bag_banks[r][0]['layers']]
    model._current_scales = model.bag_scales[r]
    model._current_tails = model.bag_tails[r]
    model._current_local = None
    actual = model._score_tokens(query, (3,3), model.bag_banks[r])
    folds=[]
    for bank, tails in zip(model.bag_banks[r],model.bag_tails[r]):
        scores=[(1-q @ b.T).amin(1)/t['statistical']['scale']
                for q,b,t in zip(query,bank['layers'],tails)]
        folds.append((scores[1]+scores[2])/2 + .125*scores[0])
    torch.testing.assert_close(actual,torch.stack(folds).mean(0),rtol=2e-5,atol=2e-5)
    amap, score = model.predict(paths[0])
    assert np.isfinite(amap).all() and np.isfinite(score)
    assert model.object_metadata()['query_used_for_fit'] is False
    model.fit(paths, context)
    assert model.bank_bags == 4  # refitting must not multiply folds again


def test_rejects_one_shot_before_fitting():
    cfg=config(); cfg['calibration_audit']=False
    model=MatchedReferenceMatching(TinyTripleBackbone(),cfg)
    with pytest.raises(ValueError,match='at least two'):
        model.fit(['one.png'],MethodContext('can',0,8,False,False))
