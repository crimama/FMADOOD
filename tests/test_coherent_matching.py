import numpy as np
import pytest
import torch

from fmad.methods.coherent_matching import CoherentMatching, coherence_gap, window_indices, quantile_transfer
from fmad.methods.reference_matching import ReferenceMatching
from fmad.protocol import MethodContext
from tests.test_local_relation import config
from tests.test_reference_matching import TinyTripleBackbone
from tests.test_psma_mr_v2 import _write_image


def test_shared_correspondence_vs_spliced_structure():
    # Both normal windows are internally uniform; mixture preserves every
    # individual feature but cannot be explained by one shared reference.
    a = torch.tensor([1.,0.]).repeat(5,1)
    b = torch.tensor([0.,1.]).repeat(5,1)
    reference = torch.stack([a,b])[None]
    normal = coherence_gap(a[None], reference)
    disrupted = a.clone(); disrupted[2:] = b[2:]
    assert normal.item() == 0
    assert coherence_gap(disrupted[None], reference).item() == pytest.approx(.4)
    assert coherence_gap(disrupted[None], reference.flip(1)).item() == pytest.approx(.4)


def test_boundary_and_translation_of_windows():
    indices = window_indices((3,4), "cpu")
    assert indices[0].tolist() == [0,0,4,0,1]
    assert indices[5].tolist() == [5,1,9,4,6]
    assert indices.min() == 0 and indices.max() == 11


def test_magnitude_control_matches_normal_empirical_distribution():
    distances = torch.tensor([.1,.2,.3,.4])
    penalties = torch.tensor([0.,0.,.1,.5])
    torch.testing.assert_close(quantile_transfer(distances, distances, penalties), penalties)
    assert quantile_transfer(torch.tensor([2.]), distances, penalties).item() == .5


@pytest.mark.parametrize("mode", ["relaxed", "joint", "magnitude"])
def test_fit_predict_same_bank_frozen_finite(tmp_path, mode):
    paths = [_write_image(tmp_path/f"{i}.png",i*37) for i in range(3)]
    query = _write_image(tmp_path/"query.png",83)
    context = MethodContext("can",0,8,False,False)
    baseline = ReferenceMatching(TinyTripleBackbone(),config())
    method = CoherentMatching(TinyTripleBackbone(),config(coherence_mode=mode))
    baseline.fit(paths,context); method.fit(paths,context)
    expected = baseline.predict(query)[0]
    actual = method.predict(query)[0]
    assert np.isfinite(actual).all()
    if mode == "relaxed":
        np.testing.assert_array_equal(actual,expected)
    else:
        assert (actual >= expected-1e-6).all()
    for resolution,bags in method.bag_banks.items():
        for a,b in zip(bags,baseline.bag_banks[resolution]):
            for x,y in zip(a["layers"],b["layers"]):
                assert torch.equal(x,y)
            assert a["coherence_tau"] >= 1e-4
    assert not method._window_sources
    assert method.object_metadata()["trainable_parameters"] == 0


def test_single_support_rejected():
    method = CoherentMatching(TinyTripleBackbone(),config())
    with pytest.raises(ValueError,match=">=2"):
        method.fit(["unused"],MethodContext("can",0,8,False,False))


def test_gap_nonnegative_and_single_candidate_zero():
    q = torch.nn.functional.normalize(torch.randn(11,5,7),dim=-1)
    r = torch.nn.functional.normalize(torch.randn(11,3,5,7),dim=-1)
    assert (coherence_gap(q,r)>=0).all()
    torch.testing.assert_close(coherence_gap(q,r[:,:1]),torch.zeros(11))
