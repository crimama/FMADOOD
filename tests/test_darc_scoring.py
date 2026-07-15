from __future__ import annotations

import numpy as np

from flow_tte.darc_scoring import (
    FusionInputs,
    LocalCandidateSet,
    fuse_confidence,
    g0_cosine_scores,
    leave_one_image_out_g0,
    local_min_cosine_scores,
    r1_cosine_scores,
    upper_tail_evidence,
)


def test_upper_tail_evidence_includes_equal_reference_scores() -> None:
    # Given: a reference tail with duplicate values at the queried threshold.
    reference = np.asarray([1.0, 2.0, 2.0, 4.0], dtype=np.float32)
    values = np.asarray([0.0, 2.0, 3.0, 5.0], dtype=np.float32)

    # When: exact finite-sample upper-tail evidence is computed.
    evidence = upper_tail_evidence(reference, values)

    # Then: ties count in the upper tail and the +1 bounds remain finite.
    expected = -np.log(np.asarray([1.0, 0.8, 0.4, 0.2], dtype=np.float32))
    assert np.allclose(evidence, expected)


def test_g0_cosine_scores_matches_global_nearest_neighbor_across_chunks() -> None:
    # Given: two queries and a memory bank whose best cosine matches are known.
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    memory = np.asarray([[2.0, 0.0], [0.0, -3.0]], dtype=np.float32)

    # When: G0 scores one query per chunk.
    scores = g0_cosine_scores(query, memory, chunk_size=1)

    # Then: each score is one minus the best normalized cosine similarity.
    assert np.allclose(scores, np.asarray([0.0, 1.0], dtype=np.float32))


def test_leave_one_image_out_g0_excludes_same_image_features() -> None:
    # Given: two normal images whose sole features are orthogonal.
    feature_images = (
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray([[0.0, 1.0]], dtype=np.float32),
    )

    # When: each image is scored against the other image only.
    scores = leave_one_image_out_g0(feature_images, chunk_size=1)

    # Then: self-distance zero is unavailable and both distances are one.
    assert len(scores) == 2
    assert np.allclose(scores[0], np.asarray([1.0], dtype=np.float32))
    assert np.allclose(scores[1], np.asarray([1.0], dtype=np.float32))


def test_local_min_cosine_requires_three_valid_supports() -> None:
    # Given: one query has three valid supports while another has only two.
    query = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    values = np.tile(
        np.asarray([[[[1.0, 0.0], [0.0, 1.0]]] * 3], dtype=np.float32),
        (2, 1, 1, 1),
    )
    valid = np.ones((2, 3, 2), dtype=np.bool_)
    valid[1, 2] = False

    # When: local candidate distances are reduced across supports and candidates.
    result = local_min_cosine_scores(query, LocalCandidateSet(values=values, valid=valid))

    # Then: only the query backed by at least three supports is usable.
    assert np.array_equal(result.valid, np.asarray([True, False], dtype=np.bool_))
    assert result.scores[0] == 0.0


def test_r1_cosine_scores_builds_robust_cross_support_prototype() -> None:
    # Given: three supports each have a majority candidate at the query direction.
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    values = np.asarray(
        [
            [
                [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]],
                [[2.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [3.0, 0.0], [0.0, -1.0]],
            ],
        ],
        dtype=np.float32,
    )
    candidates = LocalCandidateSet(values=values, valid=np.ones((1, 3, 3), dtype=np.bool_))

    # When: R1 forms per-support medians and their geometric median.
    result = r1_cosine_scores(query, candidates)

    # Then: the robust prototype follows the repeated normal direction.
    assert result.valid[0]
    assert result.scores[0] < 1e-6


def test_r1_cosine_scores_falls_back_with_fewer_than_three_supports() -> None:
    # Given: only two support prototypes for a micro query.
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    values = np.asarray(
        [[[[1.0, 0.0]], [[0.0, 1.0]]]],
        dtype=np.float32,
    )
    candidates = LocalCandidateSet(values=values, valid=np.ones((1, 2, 1), dtype=np.bool_))

    # When: R1 attempts robust prototype construction.
    result = r1_cosine_scores(query, candidates)

    # Then: it marks the result invalid for deterministic coarse fallback.
    assert not result.valid[0]


def test_confidence_fusion_uses_coarse_score_for_invalid_micro_pixels() -> None:
    # Given: equal branch scores but only the first micro pixel is valid.
    inputs = FusionInputs(
        coarse_evidence=np.asarray([1.0, 1.0], dtype=np.float32),
        micro_evidence=np.asarray([4.0, 4.0], dtype=np.float32),
        confidence=np.asarray([0.5, 0.5], dtype=np.float32),
        micro_valid=np.asarray([True, False], dtype=np.bool_),
    )

    # When: confidence-weighted max fusion is applied.
    fused = fuse_confidence(inputs)

    # Then: valid micro evidence can win and invalid micro evidence cannot leak in.
    assert np.allclose(fused, np.asarray([2.0, 1.0], dtype=np.float32))
