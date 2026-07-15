from __future__ import annotations

import numpy as np
import pytest

from flow_tte.darc_gate2_scoring import (
    calibrate_rung_evidence,
    r1_cosine_scores_chunked,
    score_rungs,
)
from flow_tte.darc_gate2_scoring_types import (
    RungNormalReferences,
    RungScores,
    RungScoringConfig,
    RungScoringInput,
    SupportValidityAudit,
)
from flow_tte.darc_scoring import (
    LocalCandidateSet,
    g0_cosine_scores,
    r1_cosine_scores,
)
from flow_tte.metrics import MetricInputError as ScoringInputError


def test_chunked_r1_matches_scalar_with_randomized_invalid_candidates() -> None:
    # Given: randomized local candidates containing masked rows and non-finite entries.
    rng = np.random.default_rng(20260710)
    query = rng.normal(size=(17, 8)).astype(np.float32)
    values = rng.normal(size=(17, 5, 9, 8)).astype(np.float32)
    valid = rng.random(size=(17, 5, 9)) > 0.2
    values[1, 2, 3, 0] = np.nan
    values[4, 1, 5, 2] = np.inf
    valid[7, 2:] = False
    candidates = LocalCandidateSet(values=values, valid=valid)

    # When: R1 is evaluated by the scalar reference and by the chunked implementation.
    expected = r1_cosine_scores(query, candidates)
    actual = r1_cosine_scores_chunked(query, candidates, chunk_size=4)

    # Then: validity and residuals preserve the frozen scalar semantics.
    assert np.array_equal(actual.valid, expected.valid)
    assert np.allclose(actual.scores, expected.scores, rtol=1e-6, atol=1e-6)


def test_score_rungs_preserves_identity_l0_l1_and_token_population() -> None:
    # Given: identity and aligned candidate tensors are the same ordered population.
    query = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    values = np.repeat(query[:, None, None, :], repeats=4, axis=1)
    values = np.repeat(values, repeats=3, axis=2)
    valid = np.ones((3, 4, 3), dtype=np.bool_)
    candidates = LocalCandidateSet(values=values, valid=valid)
    inputs = RungScoringInput(
        query=query,
        global_memory=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        identity_candidates=candidates,
        aligned_candidates=candidates,
    )

    # When: all three rungs score the shared query-token population.
    result = score_rungs(inputs, RungScoringConfig(g0_chunk_size=2, r1_chunk_size=2))

    # Then: identity L0/L1 agree and no token is removed from any output.
    assert np.array_equal(result.l0, result.l1)
    assert result.g0.shape == result.l0.shape == result.l1.shape == result.r1.shape == (3,)
    assert not np.any(result.common_fallback)


def test_score_rungs_uses_shared_support_intersection_and_one_g0_fallback() -> None:
    # Given: each arm has three supports but their ordered-slot intersection has only two.
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    values = np.repeat(query[:, None, None, :], repeats=4, axis=1)
    identity_valid = np.ones((2, 4, 1), dtype=np.bool_)
    identity_valid[1, 3] = False
    identity = LocalCandidateSet(
        values=values,
        valid=identity_valid,
    )
    aligned_valid = np.ones((2, 4, 1), dtype=np.bool_)
    aligned_valid[1, 0] = False
    aligned = LocalCandidateSet(values=values, valid=aligned_valid)
    inputs = RungScoringInput(
        query=query,
        global_memory=np.asarray([[-1.0, 0.0], [0.0, -1.0]], dtype=np.float32),
        identity_candidates=identity,
        aligned_candidates=aligned,
    )

    # When: the rung ladder applies its deterministic fallback.
    result = score_rungs(inputs, RungScoringConfig(g0_chunk_size=1, r1_chunk_size=1))

    # Then: the failed token remains and receives one identical G0 residual in all arms.
    assert np.array_equal(result.common_fallback, np.asarray([False, True]))
    assert result.l0[1] == result.l1[1] == result.r1[1] == result.g0[1]
    assert np.array_equal(
        result.support_validity.shared[1],
        np.asarray([False, True, True, False], dtype=np.bool_),
    )


def test_score_rungs_rejects_unequal_candidate_populations() -> None:
    # Given: L0 and L1 have different ordered support counts.
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    identity = LocalCandidateSet(
        values=np.ones((1, 3, 1, 2), dtype=np.float32),
        valid=np.ones((1, 3, 1), dtype=np.bool_),
    )
    aligned = LocalCandidateSet(
        values=np.ones((1, 4, 1, 2), dtype=np.float32),
        valid=np.ones((1, 4, 1), dtype=np.bool_),
    )
    inputs = RungScoringInput(
        query=query,
        global_memory=query,
        identity_candidates=identity,
        aligned_candidates=aligned,
    )

    # When/Then: unequal rung populations are invalid instead of silently truncated.
    with pytest.raises(ScoringInputError, match="identical ordered populations"):
        score_rungs(inputs, RungScoringConfig(g0_chunk_size=1, r1_chunk_size=1))


def test_score_rungs_returns_finite_values_for_nonfinite_local_candidates() -> None:
    # Given: every local candidate for one token is non-finite but G0 remains valid.
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    values = np.repeat(query[:, None, None, :], repeats=3, axis=1)
    values[1] = np.nan
    candidates = LocalCandidateSet(
        values=values,
        valid=np.ones((2, 3, 1), dtype=np.bool_),
    )
    inputs = RungScoringInput(
        query=query,
        global_memory=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        identity_candidates=candidates,
        aligned_candidates=candidates,
    )

    # When: the shared rung scorer processes the invalid local token.
    result = score_rungs(inputs, RungScoringConfig(g0_chunk_size=2, r1_chunk_size=2))

    # Then: it falls back without emitting NaN/Inf or changing population size.
    assert result.common_fallback[1]
    residuals = (result.g0, result.l0, result.l1, result.r1)
    assert all(np.all(np.isfinite(values_)) for values_ in residuals)


def test_chunked_r1_is_deterministic_across_chunk_sizes() -> None:
    # Given: a fixed random input with five complete support populations.
    rng = np.random.default_rng(99)
    query = rng.normal(size=(13, 6)).astype(np.float32)
    values = rng.normal(size=(13, 5, 9, 6)).astype(np.float32)
    candidates = LocalCandidateSet(
        values=values,
        valid=np.ones((13, 5, 9), dtype=np.bool_),
    )

    # When: the same R1 computation uses different scheduling chunks.
    one = r1_cosine_scores_chunked(query, candidates, chunk_size=1)
    seven = r1_cosine_scores_chunked(query, candidates, chunk_size=7)

    # Then: chunk boundaries do not affect validity or float32 residuals.
    assert np.array_equal(one.valid, seven.valid)
    assert np.array_equal(one.scores, seven.scores)


def test_sparse_g0_mode_preserves_all_rung_outputs() -> None:
    # Given: one valid local token and one token requiring the shared fallback.
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    values = np.repeat(query[:, None, None, :], repeats=3, axis=1)
    valid = np.ones((2, 3, 1), dtype=np.bool_)
    valid[1, 2] = False
    candidates = LocalCandidateSet(values=values, valid=valid)
    inputs = RungScoringInput(
        query=query,
        global_memory=np.asarray([[-1.0, 0.0], [0.0, -1.0]], dtype=np.float32),
        identity_candidates=candidates,
        aligned_candidates=candidates,
    )

    # When: reference mode computes complete G0 and query mode computes fallback rows only.
    complete = score_rungs(
        inputs,
        RungScoringConfig(g0_chunk_size=1, r1_chunk_size=1, complete_g0=True),
    )
    sparse = score_rungs(
        inputs,
        RungScoringConfig(g0_chunk_size=1, r1_chunk_size=1, complete_g0=False),
    )

    # Then: all scientific rung outputs agree and omitted G0 rows are explicitly marked.
    assert np.array_equal(sparse.common_fallback, complete.common_fallback)
    assert np.array_equal(sparse.l0, complete.l0)
    assert np.array_equal(sparse.l1, complete.l1)
    assert np.array_equal(sparse.r1, complete.r1)
    assert np.array_equal(sparse.g0_valid, sparse.common_fallback)
    assert np.all(complete.g0_valid)


def test_precomputed_g0_matches_internal_scoring_without_reading_memory() -> None:
    # Given: exact full-population G0 residuals and complete identity candidates.
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    memory = np.asarray([[-1.0, 0.0], [0.0, -1.0]], dtype=np.float32)
    precomputed = g0_cosine_scores(query, memory, chunk_size=1)
    values = np.repeat(query[:, None, None, :], repeats=3, axis=1)
    candidates = LocalCandidateSet(values, np.ones((2, 3, 1), dtype=np.bool_))
    expected = score_rungs(
        RungScoringInput(query, memory, candidates, candidates),
        RungScoringConfig(g0_chunk_size=1, r1_chunk_size=1),
    )

    # When: the same residuals are injected while the global memory is intentionally empty.
    actual = score_rungs(
        RungScoringInput(
            query,
            np.empty((0, 2), dtype=np.float32),
            candidates,
            candidates,
            precomputed_g0=precomputed,
        ),
        RungScoringConfig(g0_chunk_size=1, r1_chunk_size=1, complete_g0=False),
    )

    # Then: precomputation bypasses memory scoring and preserves every rung exactly.
    assert np.array_equal(actual.g0, expected.g0)
    assert np.array_equal(actual.l0, expected.l0)
    assert np.array_equal(actual.l1, expected.l1)
    assert np.array_equal(actual.r1, expected.r1)
    assert np.all(actual.g0_valid)


def test_evidence_uses_g0_reference_only_on_common_fallback_tokens() -> None:
    # Given: one local token and one common-fallback token with distinct rung references.
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    values = np.repeat(query[:, None, None, :], repeats=3, axis=1)
    valid = np.ones((2, 3, 1), dtype=np.bool_)
    valid[1, 2] = False
    candidates = LocalCandidateSet(values=values, valid=valid)
    scores = score_rungs(
        RungScoringInput(
            query=query,
            global_memory=np.asarray([[-1.0, 0.0], [0.0, -1.0]], dtype=np.float32),
            identity_candidates=candidates,
            aligned_candidates=candidates,
        ),
        RungScoringConfig(g0_chunk_size=2, r1_chunk_size=2),
    )
    references = RungNormalReferences(
        g0=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        l0=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        l1=np.asarray([0.5, 0.5, 0.5], dtype=np.float32),
        r1=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
    )

    # When: each rung is calibrated against its frozen normal-only tail.
    evidence = calibrate_rung_evidence(scores, references)

    # Then: fallback evidence is rung-independent G0 evidence and remains finite.
    assert evidence.l0[1] == evidence.l1[1] == evidence.r1[1] == evidence.g0[1]
    assert np.array_equal(evidence.g0_valid, scores.g0_valid)
    evidence_arrays = (evidence.g0, evidence.l0, evidence.l1, evidence.r1)
    assert all(np.all(np.isfinite(values_)) for values_ in evidence_arrays)


def test_evidence_uses_only_g0_when_local_reference_domains_are_empty() -> None:
    # Given: an all-fallback query and the corresponding empty local normal domains.
    residuals = np.asarray([0.25, 1.25], dtype=np.float32)
    fallback = np.ones(2, dtype=np.bool_)
    support = np.zeros((2, 3), dtype=np.bool_)
    scores = RungScores(
        g0=residuals,
        g0_valid=np.ones(2, dtype=np.bool_),
        l0=residuals,
        l1=residuals,
        r1=residuals,
        common_fallback=fallback,
        support_validity=SupportValidityAudit(support, support, support, support),
    )
    references = RungNormalReferences(
        g0=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        l0=np.empty(0, dtype=np.float32),
        l1=np.empty(0, dtype=np.float32),
        r1=np.empty(0, dtype=np.float32),
    )

    # When: evidence is calibrated through the registered common-fallback path.
    evidence = calibrate_rung_evidence(scores, references)

    # Then: no undefined local tail is evaluated and every rung equals finite G0 evidence.
    assert np.array_equal(evidence.l0, evidence.g0)
    assert np.array_equal(evidence.l1, evidence.g0)
    assert np.array_equal(evidence.r1, evidence.g0)
    assert np.all(np.isfinite(evidence.g0))


def test_evidence_rejects_nonfallback_tokens_without_local_reference_domains() -> None:
    # Given: one nonfallback token but no empirical local normal reference population.
    residuals = np.asarray([0.25, 1.25], dtype=np.float32)
    fallback = np.asarray([True, False], dtype=np.bool_)
    support = np.zeros((2, 3), dtype=np.bool_)
    scores = RungScores(
        g0=residuals,
        g0_valid=np.ones(2, dtype=np.bool_),
        l0=residuals,
        l1=residuals,
        r1=residuals,
        common_fallback=fallback,
        support_validity=SupportValidityAudit(support, support, support, support),
    )
    references = RungNormalReferences(
        g0=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        l0=np.empty(0, dtype=np.float32),
        l1=np.empty(0, dtype=np.float32),
        r1=np.empty(0, dtype=np.float32),
    )

    # When/Then: an undefined nonfallback local tail remains an invalid gate input.
    with pytest.raises(ScoringInputError, match="nonfallback tokens require local references"):
        calibrate_rung_evidence(scores, references)


def test_evidence_requires_g0_reference_when_no_g0_score_is_valid() -> None:
    # Given: sparse G0 scoring has no valid query values, but local rungs are usable.
    residuals = np.asarray([0.25, 1.25], dtype=np.float32)
    support = np.ones((2, 3), dtype=np.bool_)
    scores = RungScores(
        g0=np.zeros(2, dtype=np.float32),
        g0_valid=np.zeros(2, dtype=np.bool_),
        l0=residuals,
        l1=residuals,
        r1=residuals,
        common_fallback=np.zeros(2, dtype=np.bool_),
        support_validity=SupportValidityAudit(support, support, support, support),
    )
    local_reference = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    invalid_g0_references = (
        np.empty(0, dtype=np.float32),
        np.asarray([np.nan], dtype=np.float32),
    )

    # When/Then: the registered global normal reference remains finite and mandatory.
    for g0_reference in invalid_g0_references:
        references = RungNormalReferences(
            g0=g0_reference,
            l0=local_reference,
            l1=local_reference,
            r1=local_reference,
        )
        with pytest.raises(ScoringInputError, match="G0 reference must be non-empty and finite"):
            calibrate_rung_evidence(scores, references)


def test_evidence_rejects_mismatched_rung_score_populations() -> None:
    # Given: a hand-built public input has one local rung on a shorter token population.
    support = np.ones((2, 3), dtype=np.bool_)
    scores = RungScores(
        g0=np.zeros(2, dtype=np.float32),
        g0_valid=np.zeros(2, dtype=np.bool_),
        l0=np.asarray([0.25], dtype=np.float32),
        l1=np.asarray([0.25, 1.25], dtype=np.float32),
        r1=np.asarray([0.25, 1.25], dtype=np.float32),
        common_fallback=np.zeros(2, dtype=np.bool_),
        support_validity=SupportValidityAudit(support, support, support, support),
    )
    reference = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    references = RungNormalReferences(reference, reference, reference, reference)

    # When/Then: malformed input fails at the public boundary instead of leaking IndexError.
    with pytest.raises(ScoringInputError, match="rung score arrays must have identical shapes"):
        calibrate_rung_evidence(scores, references)
