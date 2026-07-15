from __future__ import annotations

import pytest
import numpy as np
import torch

from flow_tte.config import ScoreConfig
from flow_tte.memory import TorchMemoryBank
from flow_tte.scoring import ScoreCalibration, ScoreInputs, score_flow_memory
from flow_tte_phase3_scorer_suite import (
    NativeMapRecord,
    evaluate_native_records,
    fit_density_profiles,
    fit_mahalanobis,
    fit_pca,
    fit_scorer_suite,
    mahalanobis_distances,
    pca_rank_for_explained_variance,
    pca_residual_distances,
    score_scorer_suite,
)
from flow_tte_gap_decomposition import oracle_f1


def test_density_normalization_equalizes_regions_with_different_bank_density() -> None:
    bank_features = torch.tensor(
        [[0.0], [1.0], [2.0], [3.0], [100.0], [110.0], [120.0], [130.0]],
        dtype=torch.float32,
    )
    queries = torch.tensor([[0.4], [102.5]], dtype=torch.float32)
    profile = fit_density_profiles(bank_features, k_values=(3,), chunk_size=3)[3]
    bank = TorchMemoryBank()
    bank.fit(bank_features)

    nearest = bank.query(queries, k=1, chunk_size=1)
    raw_distances = nearest.distances[:, 0]
    normalized = raw_distances / (profile.rho[nearest.indices[:, 0]] + 1e-8)

    assert torch.equal(raw_distances, torch.tensor([0.4, 2.5]))
    assert raw_distances[0] < raw_distances[1]
    assert torch.equal(profile.rho[nearest.indices[:, 0]], torch.tensor([2.0, 20.0]))
    assert normalized[0] > normalized[1]
    assert profile.floor == pytest.approx(1.0)
    assert profile.clamp_count == 0


def test_density_floor_clip_and_large_k_are_stable() -> None:
    support = (torch.arange(220, dtype=torch.float32).square() / 100.0).unsqueeze(1)
    profiles = fit_density_profiles(support, k_values=(3, 20, 500), chunk_size=31)
    assert profiles[3].clamp_count > 0
    assert profiles[500].effective_k == support.shape[0] - 1
    assert torch.all(profiles[3].rho >= profiles[3].floor)

    config = ScoreConfig(query_chunk_size=32, calibration_sample_size=64)
    suite = fit_scorer_suite(support, config)
    inputs = ScoreInputs(
        query_z=torch.tensor([[1e6]], dtype=torch.float32),
        nll=torch.zeros(1),
        nll_penalty=torch.zeros(1),
        image_indices=torch.zeros(1, dtype=torch.long),
        n_images=1,
    )
    result = score_scorer_suite(inputs, suite)["density_normalized_1nn_k3"]
    assert result.distances.item() == pytest.approx(suite.score_caps["density_normalized_1nn_k3"])


@pytest.mark.parametrize(
    ("k", "expected"),
    [
        (2, [1.5, 1.0, 1.0, 1.5]),
        (3, [2.0, 1.0, 1.0, 2.0]),
    ],
)
def test_rho_excludes_self_and_uses_mathematical_median(
    k: int,
    expected: list[float],
) -> None:
    bank_features = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float32)

    profile = fit_density_profiles(bank_features, k_values=(k,), chunk_size=2)[k]

    assert torch.equal(profile.rho, torch.tensor(expected))
    assert profile.floor == pytest.approx(1.0)
    assert profile.clamp_count == 0


def test_pca_rank_is_smallest_rank_reaching_ninety_percent() -> None:
    bank_features = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [-4.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=torch.float32,
    )

    state = fit_pca(bank_features, target=0.90)
    residual = pca_residual_distances(
        torch.tensor([[0.0, 0.0, 3.0]], dtype=torch.float32),
        state,
        chunk_size=1,
    )
    boundary_rank, boundary_explained = pca_rank_for_explained_variance(
        torch.tensor([3.0, 1.0]),
        target=0.90,
    )

    assert state.rank == 2
    assert state.explained_variance == pytest.approx(40.0 / 42.0)
    assert torch.allclose(residual, torch.tensor([3.0]), atol=1e-6)
    assert boundary_rank == 1
    assert boundary_explained == pytest.approx(0.90)


def test_mahalanobis_and_pca_are_finite_and_honor_energy_target() -> None:
    generator = torch.Generator().manual_seed(9)
    support = torch.randn((40, 6), generator=generator)
    queries = torch.randn((7, 6), generator=generator)
    mahalanobis = mahalanobis_distances(queries, fit_mahalanobis(support), chunk_size=3)
    pca = fit_pca(support, target=0.95, max_components=6)
    residual = pca_residual_distances(queries, pca, chunk_size=2)

    assert mahalanobis.shape == residual.shape == (7,)
    assert torch.isfinite(mahalanobis).all()
    assert torch.isfinite(residual).all()
    assert pca.rank <= 6
    assert pca.explained_variance >= 0.95


def test_raw_1nn_suite_path_exactly_matches_existing_scorer() -> None:
    generator = torch.Generator().manual_seed(123)
    support = torch.randn((31, 7), generator=generator)
    queries = torch.randn((13, 7), generator=generator)
    density_penalty = torch.randn((13,), generator=generator)
    config = ScoreConfig(
        distance_weight=1.0,
        density_weight=0.25,
        top_percent=0.25,
        query_chunk_size=5,
        calibration_sample_size=23,
        use_squared_distance=False,
    )
    inputs = ScoreInputs(
        query_z=queries,
        nll=torch.zeros(13),
        nll_penalty=density_penalty,
        image_indices=torch.zeros(13, dtype=torch.long),
        n_images=1,
    )
    bank = TorchMemoryBank()
    bank.fit(support)
    calibration = ScoreCalibration.fit(support, config)

    expected = score_flow_memory(inputs, bank, config, calibration)
    suite = fit_scorer_suite(support, config)
    observed = score_scorer_suite(inputs, suite)["raw_1nn"]

    assert torch.equal(observed.distances, expected.distances)
    assert torch.equal(observed.distance_scores, expected.distance_scores)
    assert torch.equal(observed.patch_scores, expected.patch_scores)
    assert torch.equal(observed.image_scores, expected.image_scores)
    assert observed.image_score == expected.image_score


def test_in_run_evaluator_matches_gap_decomposition_float16_oracle() -> None:
    generator = np.random.default_rng(17)
    records = []
    for index, split in enumerate(("good", "good", "bad", "bad")):
        score = generator.normal(size=(5, 7)).astype(np.float16)
        gt = np.zeros((5, 7), dtype=np.bool_)
        if split == "bad":
            gt[index : index + 2, 2:5] = True
        records.append(NativeMapRecord(split=split, score=score, gt=gt, stem=str(index)))
    labels = np.concatenate([record.gt.ravel() for record in records])
    scores = np.concatenate([record.score.ravel() for record in records])

    expected = oracle_f1(labels, scores, cast_float16=True)
    observed = evaluate_native_records(records)

    assert observed["pooled_oracle_f1_float16"] == expected["f1"]
    assert observed["oracle_threshold_float16"] == expected["threshold"]
    assert np.isfinite(observed["pooled_pauroc_0.05_float16"])
    assert np.isfinite(observed["normal_image_mean_fpr_at_oracle"])
    assert np.isfinite(observed["bad_image_oracle_mean_float16"])
