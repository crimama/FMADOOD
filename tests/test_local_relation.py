import numpy as np
import pytest
import torch
from torch.nn import functional as F

from fmad.methods.local_relation import LocalRelation, star_descriptor, candidate_relations
from fmad.methods.reference_matching import ReferenceMatching
from fmad.protocol import MethodContext
from tests.test_reference_matching import TinyTripleBackbone
from tests.test_psma_mr_v2 import _write_image


def config(**changes):
    return dict(variant="R1", feature_layers=[5,11,23], resolutions=[8,12],
                bank_tokens_per_source=4, query_chunk_size=4, bank_bags=2,
                score_normalization="tailscale", layer_fusion="residual_spike",
                layer_fusion_tau_mode="support_quantile", layer_fusion_tau_q=.99,
                **changes)


def test_star_rotation_and_constant_field():
    x=F.normalize(torch.randn(4,4,16),dim=-1)
    a=star_descriptor(x.reshape(-1,16),(4,4)).reshape(4,4,8)
    b=star_descriptor(x.rot90(1,[0,1]).reshape(-1,16),(4,4)).reshape(4,4,8)
    torch.testing.assert_close(a.rot90(1,[0,1]),b)
    constant=F.normalize(torch.ones(16,16),dim=-1)
    torch.testing.assert_close(star_descriptor(constant,(4,4)),torch.zeros(16,8))


def test_shuffled_null_preserves_rng_and_breaks_locality():
    x=F.normalize(torch.randn(30,16),dim=-1)
    state=torch.random.get_rng_state().clone()
    a=star_descriptor(x,(5,6),shuffled=True)
    assert torch.equal(state,torch.random.get_rng_state())
    torch.testing.assert_close(a,star_descriptor(x,(5,6),shuffled=True))
    assert not torch.allclose(a,star_descriptor(x,(5,6)))


@pytest.mark.parametrize("shots",[1,3])
def test_zero_weight_exact_and_active_finite(tmp_path,shots):
    paths=[_write_image(tmp_path/f"{i}.png",i*37) for i in range(shots)]
    query=_write_image(tmp_path/'query.png',83)
    ctx=MethodContext("can",0,8,False,False)
    base=ReferenceMatching(TinyTripleBackbone(),config())
    zero=LocalRelation(TinyTripleBackbone(),config(relation_weight=0))
    base.fit(paths,ctx);zero.fit(paths,ctx)
    np.testing.assert_array_equal(base.predict(query)[0],zero.predict(query)[0])
    for res in base.bag_banks:
        for b,z in zip(base.bag_banks[res],zero.bag_banks[res]):
            for x,y in zip(b['layers'],z['layers']):
                assert torch.equal(x,y)
            assert torch.isfinite(z['relation_tau']) and z['relation_tau']>0
    zero.relation_weight=.25
    actual=zero.predict(query)[0]
    assert np.isfinite(actual).all()
    assert (actual>=base.predict(query)[0]-1e-6).all()
    assert zero._relation_sources=={}
    assert zero.object_metadata()['trainable_parameters']==0


def test_candidate_alternative_can_explain_relation():
    q=torch.tensor([[1.,0.]])
    bank=torch.tensor([[1.,0.],[.99,.1]])
    d,r=candidate_relations(q,bank,torch.zeros(1,8),torch.stack([torch.ones(8),torch.zeros(8)]),2)
    assert d.shape==r.shape==(1,2)
    assert r[0,0]==1 and r[0,1]==0


def test_entire_source_is_excluded_from_calibration():
    method=LocalRelation(TinyTripleBackbone(),config())
    # Within each source the relation descriptors are identical. Across sources
    # they differ by 1; source exclusion must recover tau=1, not zero.
    layers=[torch.tensor([[1.,0.],[0.,1.]])]*3
    method.records={8:[{'source':s,'layers':layers,'indices':torch.arange(2),'grid':(1,2)} for s in range(2)]}
    method._relation_sources={8:[torch.zeros(2,8),torch.ones(2,8)]}
    bank=method._bank_for(8)
    assert bank['relation_tau']==1
