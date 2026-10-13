import numpy as np
import pytest
import torch
from torch.nn import functional as F

from fmad.methods.reference_matching import ReferenceMatching, _distance_chunks
from fmad.methods.reference_matching_fast import (
    ReferenceMatchingFast,
    _build_shared_bag_union,
    _masked_loo_distances,
    _shared_bag_distances,
)
from fmad.protocol import MethodContext
from tests.test_reference_matching import TinyPairBackbone
from tests.test_psma_mr_v2 import _write_image


def test_masked_loo_matches_physical_source_exclusion_with_k_and_repeated_features():
    bank = F.normalize(torch.tensor([
        [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.8, 0.2, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.8, 0.2], [0.0, 0.0, 1.0],
    ]), dim=1)
    owners = torch.tensor([0, 0, 1, 1, 2, 2])
    queries = [bank[owners == owner] for owner in range(3)]
    for owner, query in enumerate(queries):
        expected = _distance_chunks(query, bank[owners != owner], chunk=1, k=3)
        actual = _masked_loo_distances(query, bank, owners, owner, chunk=1, k=3)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    # Effective k is clamped to the count of eligible tokens, even when the
    # excluded source contributed more tokens than all remaining sources.
    imbalanced = torch.cat([bank[:4], bank[4:5]])
    imbalanced_owners = torch.tensor([0, 0, 0, 0, 1])
    expected = _distance_chunks(bank[4:5], bank[:4], chunk=1, k=8)
    actual = _masked_loo_distances(bank[4:5], imbalanced, imbalanced_owners, 1, 1, 8)
    torch.testing.assert_close(actual, expected)


def test_shared_union_preserves_each_bag_membership_duplicates_and_neighbour_indices():
    first = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]), dim=1)
    second = F.normalize(torch.tensor([[0.0, 1.0], [1.0, 0.0]]), dim=1)
    banks = [first, second]
    # The same sampled identity overlaps across bags, and bag 0 deliberately
    # repeats one sampled index to verify that membership multiplicity remains.
    keys = [
        [(0, 5, 0), (0, 9, 0), (0, 5, 0)],
        [(0, 9, 0), (0, 5, 0)],
    ]
    union, members = _build_shared_bag_union(banks, keys, torch.device("cpu"))
    assert union.shape[0] == 2
    assert members[0].tolist() == [0, 1, 0]
    assert members[1].tolist() == [1, 0]

    query = F.normalize(torch.tensor([[1.0, 0.0], [0.7, 0.3], [0.0, 1.0]]), dim=1)
    actual_values, actual_indices = _shared_bag_distances(
        query, union, members, chunk=2, k=2, return_indices=True, index_k=3)
    for index, bank in enumerate(banks):
        expected_values = _distance_chunks(query, bank, chunk=2, k=2)
        _, expected_indices = _distance_chunks(query, bank, chunk=2, k=3, return_indices=True)
        torch.testing.assert_close(actual_values[index], expected_values)
        assert torch.equal(actual_indices[index], expected_indices)


def _common_config(**updates):
    config = {
        "variant": "R1",
        "feature_layers": [11, 23],
        "resolutions": [8],
        "bank_tokens_per_source": 6,
        "query_chunk_size": 2,
        "knn_k": 3,
        "score_normalization": "tailscale",
        "layer_fusion": "residual_spike",
        "layer_fusion_tau_mode": "support_quantile",
        "layer_fusion_tau_q": 0.9,
    }
    return {**config, **updates}


def _assert_calibration_equal(reference, fast):
    assert torch.allclose(reference.scales[8], fast.scales[8], rtol=1e-6, atol=1e-7)
    assert len(reference.bag_scales[8]) == len(fast.bag_scales[8])
    for ref_scale, fast_scale in zip(reference.bag_scales[8], fast.bag_scales[8]):
        torch.testing.assert_close(ref_scale, fast_scale, rtol=1e-6, atol=1e-7)
    for ref_bag, fast_bag in zip(reference.bag_tails[8], fast.bag_tails[8]):
        for ref_tail, fast_tail in zip(ref_bag, fast_bag):
            for name in ("sample", "u", "beta", "p_u"):
                torch.testing.assert_close(ref_tail[name], fast_tail[name], rtol=1e-6, atol=1e-7)
            if "spike_tau" in ref_tail:
                assert fast_tail["spike_tau"] == pytest.approx(ref_tail["spike_tau"], rel=1e-6, abs=1e-7)


def test_fast_source_mask_and_bag_union_match_baseline_calibration_and_maps(tmp_path):
    paths = [_write_image(tmp_path / f"support-{i}.png", 17 * i) for i in range(3)]
    context = MethodContext("can", 19, 8, False, False)
    config = _common_config(bank_bags=3)
    baseline = ReferenceMatching(TinyPairBackbone(), config)
    fast = ReferenceMatchingFast(TinyPairBackbone(), {
        **config,
        "fast_reuse_calibration_bank": True,
        "fast_shared_bag_distances": True,
    })
    baseline.fit(paths, context)
    fast.fit(paths, context)

    _assert_calibration_equal(baseline, fast)
    assert fast._shared_union_cache
    assert fast.object_metadata()["fast_reuse_calibration_bank"] is True
    assert fast.object_metadata()["fast_shared_bag_distances"] is True
    baseline_map, baseline_score = baseline.predict(paths[0])
    fast_map, fast_score = fast.predict(paths[0])
    np.testing.assert_allclose(fast_map, baseline_map, rtol=1e-5, atol=1e-6)
    assert fast_score == pytest.approx(baseline_score, rel=1e-5, abs=1e-6)


def test_fast_one_shot_keeps_parent_leave_token_out_behavior(tmp_path):
    path = _write_image(tmp_path / "one.png", 37)
    context = MethodContext("can", 7, 8, False, False)
    config = _common_config()
    baseline = ReferenceMatching(TinyPairBackbone(), config)
    fast = ReferenceMatchingFast(TinyPairBackbone(), {
        **config,
        "fast_reuse_calibration_bank": True,
    })
    baseline.fit([path], context)
    fast.fit([path], context)
    torch.testing.assert_close(fast.scales[8], baseline.scales[8])
    for ref_tail, fast_tail in zip(baseline.tails[8], fast.tails[8]):
        for name in ("sample", "u", "beta", "p_u"):
            torch.testing.assert_close(fast_tail[name], ref_tail[name])


def test_local_center_uses_support_neighbour_bias_and_preserves_signed_scores():
    method = ReferenceMatchingFast(TinyPairBackbone(), _common_config(
        local_reliability="center", local_center_alpha=0.5, local_reliability_k=2))
    loo = torch.tensor([0.1, 0.1, 0.9, 0.9])
    indices = torch.tensor([[0, 1], [2, 3]])
    distance = torch.tensor([0.1, 0.1])
    actual = method._local_center_score(distance, indices, loo, torch.tensor(0.5))
    torch.testing.assert_close(actual, torch.tensor([0.6, -0.2]))
    method.local_center_alpha = 0.0
    assert torch.equal(method._local_center_score(distance, indices, loo, torch.tensor(0.5)),
                       distance / 0.5)


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5])
def test_local_center_shared_bags_match_unoptimized_and_zero_matches_q95(tmp_path, alpha):
    paths = [_write_image(tmp_path / f"support-{i}.png", 17 * i) for i in range(3)]
    query = _write_image(tmp_path / "query.png", 79)
    context = MethodContext("can", 19, 8, False, False)
    config = _common_config(bank_bags=3, ablation_shallow_mode="all",
                            local_reliability="center", local_center_alpha=alpha,
                            local_reliability_k=8)
    reference = ReferenceMatchingFast(TinyPairBackbone(), config)
    fast = ReferenceMatchingFast(TinyPairBackbone(), {
        **config, "fast_reuse_calibration_bank": True, "fast_shared_bag_distances": True})
    reference.fit(paths, context)
    fast.fit(paths, context)
    _assert_calibration_equal(reference, fast)
    reference_map, reference_score = reference.predict(query)
    fast_map, fast_score = fast.predict(query)
    np.testing.assert_allclose(fast_map, reference_map, rtol=1e-5, atol=1e-6)
    assert fast_score == pytest.approx(reference_score, rel=1e-5, abs=1e-6)
    if alpha == 0.0:
        q95 = ReferenceMatchingFast(TinyPairBackbone(), {
            **config, "local_reliability": "off", "local_center_alpha": 0.0})
        q95.fit(paths, context)
        q95_map, q95_score = q95.predict(query)
        np.testing.assert_array_equal(fast_map, q95_map)
        assert fast_score == q95_score
