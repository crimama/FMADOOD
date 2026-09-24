import copy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from fmad.evaluation.feature_mechanism import (
    ARMS, cosine, image_evidence, paired_alignment, score_summary, token_evidence,
)
from fmad.methods.query_reference import QueryReference
from fmad.protocol import MethodContext
from tests.test_psma_mr_v2 import _write_image
from tests.test_query_reference import config
from tests.test_reference_matching import TinyPairBackbone


def test_token_controls_fixed_and_five_nn_are_direct_cosine_scores():
    torch.manual_seed(4)
    q, b = (F.normalize(torch.randn(n, 8), dim=-1) for n in (12, 9))
    original = b.clone()
    scores, trace = token_evidence(q, b, chunk=3, ridge=1., max_norm=.1,
                                   min_tokens=2, seed=7)
    index = (q @ b.T).argmax(1)
    shifted = F.normalize(b + trace["deltas"]["fitted"], dim=-1)
    d = 1 - q @ shifted.T
    torch.testing.assert_close(scores["fitted"], d.min(1).values)
    torch.testing.assert_close(scores["fixed"], d[torch.arange(len(q)), index])
    torch.testing.assert_close(scores["fitted5"], d.topk(5, largest=False).values.mean(1))
    assert bool((scores["fixed"] >= scores["fitted"] - 1e-6).all())
    torch.testing.assert_close(b, original, rtol=0, atol=0)
    for arm in ("random", "opposite"):
        torch.testing.assert_close(trace["deltas"][arm].norm(), trace["deltas"]["fitted"].norm())


def test_missing_cosine_and_empty_defect_are_not_zero():
    assert cosine(torch.zeros(2), torch.ones(2)) is None
    summary = score_summary(np.ones((2, 2)), np.zeros((2, 2), dtype=bool))
    assert summary == {"normal_mean": 1., "defect_mean": None, "margin": None}


def test_paired_identical_images_have_zero_displacement_and_missing_direction():
    q, b = (F.normalize(torch.randn(n, 8), dim=-1) for n in (12, 9))
    _, trace = token_evidence(q, b, chunk=3, ridge=1., max_norm=.1, min_tokens=2, seed=7)
    evidence = paired_alignment(trace, trace, b)
    assert evidence["baseline_residual_displacement_mse"] == 0
    for arm in ("fitted", "random", "opposite"):
        assert evidence[arm]["differential_cosine"] is None
        assert evidence[arm]["residual_displacement_mse"] == 0


def test_replay_matches_actual_qr0_qr1_and_preserves_banks(tmp_path):
    paths = [_write_image(tmp_path / f"{i}.png", i * 27) for i in range(4)]
    method = QueryReference(TinyPairBackbone(), config(1))
    method.fit(paths[:3], MethodContext("can", 0, 8, False, False))
    banks = copy.deepcopy(method.banks)
    maps, traces = image_evidence(method, paths[3], retain_trace=True)
    np.testing.assert_allclose(maps["fitted"], method.predict(paths[3])[0], rtol=1e-6, atol=1e-6)
    method.query_adaptation = False
    np.testing.assert_allclose(maps["baseline"], method.predict(paths[3])[0], rtol=1e-6, atol=1e-6)
    assert set(maps) == set(ARMS) and len(traces) == 4
    for resolution in banks:
        for before, after in zip(banks[resolution]["layers"], method.banks[resolution]["layers"]):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
