import pytest
import torch
from fmad.methods.query_reference import controlled_translation, QueryReference
from tests.test_reference_matching import TinyPairBackbone

def test_direction_controls_preserve_norm_rng_and_identity():
    delta=torch.linspace(-.1,.1,32)
    state=torch.random.get_rng_state().clone()
    assert controlled_translation(delta,'fitted',7) is delta
    torch.testing.assert_close(controlled_translation(delta,'opposite',7),-delta)
    random=controlled_translation(delta,'random',7)
    torch.testing.assert_close(random.norm(),delta.norm())
    torch.testing.assert_close(random,controlled_translation(delta,'random',7))
    assert not torch.allclose(random,delta)
    assert torch.equal(state,torch.random.get_rng_state())
    torch.testing.assert_close(controlled_translation(torch.zeros(32),'random',7),torch.zeros(32))

def test_invalid_direction_configuration():
    with pytest.raises(ValueError):
        QueryReference(TinyPairBackbone(),dict(translation_control='random'))
    with pytest.raises(ValueError):controlled_translation(torch.zeros(3),'unknown',0)
