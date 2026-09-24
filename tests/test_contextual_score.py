import numpy as np
import pytest
import torch

from fmad.methods.contextual_score import ContextualScore, flip_tokens
from fmad.protocol import MethodContext
from tests.test_psma_mr_v2 import TinyPSMAV2Backbone, _write_image


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_stages_fit_predict_no_leakage(tmp_path, stage):
    paths = [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(3)]
    backbone = TinyPSMAV2Backbone()
    before = backbone.model.weight.detach().clone()
    method = ContextualScore(backbone, dict(
        score_head=stage >= 1, context_matching=stage >= 2,
        correspondence_training=stage >= 3, resolutions=[8, 12], steps=2,
        bank_tokens_per_source=4, train_tokens=8, hidden_dim=4, candidate_count=2))
    method.fit(paths, MethodContext("can", 0, 8, False, False))
    prediction, score = method.predict(paths[0])
    assert prediction.shape == (12, 16)
    assert np.isfinite(prediction).all() and np.isfinite(score)
    assert torch.equal(backbone.model.weight, before)
    assert backbone.model.weight.grad is None
    assert method.records == {}
    assert method.object_metadata()["source_excluded_score_training"]
    assert all(len(bank[0]) == 12 for bank in method.banks.values())
    assert (method.history[-1]["correspondence_loss"] > 0) if stage == 3 else True
    if stage == 0:
        assert method.head is None and method.key is None and not method.history
    elif stage < 3:
        assert all(row["correspondence_loss"] == 0 for row in method.history)


def test_flip_inverse_and_correspondence():
    tokens = torch.arange(24).reshape(6, 4)
    for horizontal in (False, True):
        for vertical in (False, True):
            flipped = flip_tokens(tokens, (2, 3), horizontal, vertical)
            assert torch.equal(flip_tokens(flipped, (2, 3), horizontal, vertical), tokens)


def test_invalid_switch_dependency():
    with pytest.raises(ValueError, match="requires"):
        ContextualScore(TinyPSMAV2Backbone(), dict(correspondence_training=True))


def test_loo_excludes_all_source_tokens():
    method = ContextualScore(TinyPSMAV2Backbone(), dict(score_head=False))
    method.records = {8: [(i, torch.randn(4, 8), (2, 2), torch.arange(4)) for i in range(3)]}
    bank = method.bank(8, excluded=1)
    assert bank[2] == [0] * 4 + [2] * 4
