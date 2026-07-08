from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt
import torch

from flow_tte.config import ExpansionConfig, FlowConfig, FlowTTEConfig, ScoreConfig
from flow_tte.flow import PatchNormalizingFlow
from flow_tte.losses import mean_nll, tail_aware_nll
from flow_tte.memory import TorchMemoryBank
from flow_tte.pipeline import FlowTTE
from flow_tte.tensors import as_patch_batch

FloatArray = npt.NDArray[np.float32]


def normal_array(
    rng: np.random.Generator,
    loc: float,
    scale: float,
    size: Tuple[int, ...],
) -> FloatArray:
    return rng.normal(loc=loc, scale=scale, size=size).astype(np.float32)


def test_affine_flow_is_invertible() -> None:
    flow = PatchNormalizingFlow(dim=6, n_coupling_layers=4, hidden_multiplier=2, clamp=1.5)
    x = torch.arange(66, dtype=torch.float32).reshape(11, 6) / 10.0

    z, logdet = flow.forward(x)
    x_recon, reverse_logdet = flow.forward(z, reverse=True)

    assert torch.allclose(x, x_recon, atol=1e-5)
    assert torch.allclose(logdet + reverse_logdet, torch.zeros_like(logdet), atol=1e-5)


def test_tail_aware_loss_focuses_high_nll_patches() -> None:
    z = torch.tensor([[0.0], [0.0], [0.0], [8.0]], dtype=torch.float32)
    logdet = torch.zeros(4)

    standard = mean_nll(z, logdet)
    tail = tail_aware_nll(z, logdet, tail_weight=1.0, tail_top_k_ratio=0.25)

    assert tail > standard


def test_score_then_expand_scores_before_absorbing_batch() -> None:
    rng = np.random.default_rng(11)
    support = normal_array(rng, loc=0.0, scale=0.3, size=(64, 4))
    batch = normal_array(rng, loc=0.0, scale=0.3, size=(16, 4))
    config = FlowTTEConfig(
        flow=FlowConfig(n_coupling_layers=2, n_epochs=2, batch_size=32),
        expansion=ExpansionConfig(budget=1.25, density_quantile=0.95, random_seed=1),
        score=ScoreConfig(density_weight=0.2, top_percent=0.25),
        device="cpu",
    )
    pipeline = FlowTTE(config)
    _ = pipeline.fit(support)

    result = pipeline.score_then_expand(batch)

    assert result.memory_size_before == 64
    assert result.memory_size_after >= result.memory_size_before
    assert result.patch_scores.shape == (16,)
    assert result.nll.shape == (16,)
    assert result.image_scores.shape == (1,)


def test_score_then_expand_preserves_image_spatial_shape() -> None:
    rng = np.random.default_rng(17)
    support = normal_array(rng, loc=0.0, scale=0.3, size=(2, 4, 4, 4))
    batch = normal_array(rng, loc=0.0, scale=0.3, size=(3, 4, 4, 4))
    config = FlowTTEConfig(
        flow=FlowConfig(n_coupling_layers=2, n_epochs=2, batch_size=16, seed=3),
        expansion=ExpansionConfig(budget=1.25, density_quantile=0.95, random_seed=1),
        score=ScoreConfig(density_weight=0.2, top_percent=0.25, query_chunk_size=8),
        device="cpu",
    )
    pipeline = FlowTTE(config)
    _ = pipeline.fit(support)

    result = pipeline.score_then_expand(batch)

    assert result.memory_size_before == 32
    assert result.patch_scores.shape == (3, 4, 4)
    assert result.nll.shape == (3, 4, 4)
    assert result.selected_mask.shape == (3, 4, 4)
    assert result.image_scores.shape == (3,)


def test_patch_batch_keeps_image_membership_for_spatial_inputs() -> None:
    features = np.zeros((2, 3, 5), dtype=np.float32)

    batch = as_patch_batch(features, torch.device("cpu"))

    assert batch.flat_features.shape == (6, 5)
    assert torch.equal(batch.image_indices, torch.tensor([0, 0, 0, 1, 1, 1]))
    assert batch.restore(torch.arange(6, dtype=torch.float32)).shape == (2, 3)


def test_training_is_reproducible_for_fixed_seed() -> None:
    rng = np.random.default_rng(19)
    support = normal_array(rng, loc=0.0, scale=0.25, size=(2, 8, 4))
    config = FlowTTEConfig(
        flow=FlowConfig(n_coupling_layers=2, n_epochs=2, batch_size=8, seed=7),
        expansion=ExpansionConfig(budget=1.25, density_quantile=0.90, random_seed=4),
        score=ScoreConfig(density_weight=0.2, top_percent=0.25),
        device="cpu",
    )

    first = FlowTTE(config).fit(support)
    second = FlowTTE(config).fit(support)

    assert first.density_threshold == second.density_threshold
    assert first.losses == second.losses


def test_chunked_memory_query_matches_full_query() -> None:
    memory = torch.arange(85, dtype=torch.float32).reshape(17, 5) / 10.0
    query = torch.arange(35, dtype=torch.float32).reshape(7, 5) / 7.0
    bank = TorchMemoryBank()
    bank.fit(memory)

    full = bank.query(query, k=3, chunk_size=64)
    chunked = bank.query(query, k=3, chunk_size=2)

    assert torch.allclose(full.distances, chunked.distances)
    assert torch.equal(full.indices, chunked.indices)


def test_flow_density_gate_rejects_far_candidates() -> None:
    rng = np.random.default_rng(13)
    support = normal_array(rng, loc=0.0, scale=0.2, size=(96, 4))
    normal = normal_array(rng, loc=0.0, scale=0.2, size=(8, 4))
    far = normal_array(rng, loc=5.0, scale=0.1, size=(2, 4))
    batch: FloatArray = np.concatenate((normal, far), axis=0).astype(np.float32)
    config = FlowTTEConfig(
        flow=FlowConfig(n_coupling_layers=2, n_epochs=2, batch_size=32),
        expansion=ExpansionConfig(budget=1.5, density_quantile=0.95, random_seed=2),
        score=ScoreConfig(density_weight=0.2, top_percent=0.2),
        device="cpu",
    )
    pipeline = FlowTTE(config)
    _ = pipeline.fit(support)

    result = pipeline.score_then_expand(batch)

    assert not result.selected_mask[-1]
    assert not result.selected_mask[-2]


def test_nf_nll_score_mode_uses_raw_density_score() -> None:
    rng = np.random.default_rng(23)
    support = normal_array(rng, loc=0.0, scale=0.2, size=(96, 4))
    normal = normal_array(rng, loc=0.0, scale=0.2, size=(8, 4))
    far = normal_array(rng, loc=4.0, scale=0.1, size=(2, 4))
    batch: FloatArray = np.concatenate((normal, far), axis=0).astype(np.float32)
    config = FlowTTEConfig(
        flow=FlowConfig(n_coupling_layers=2, n_epochs=2, batch_size=32, seed=23),
        expansion=ExpansionConfig(budget=1.0, density_quantile=0.95, random_seed=2),
        score=ScoreConfig(score_mode="nf_nll", density_weight=0.0, top_percent=0.2),
        device="cpu",
    )
    pipeline = FlowTTE(config)
    _ = pipeline.fit(support)

    result = pipeline.score_static(batch)

    assert result.memory_size_after == result.memory_size_before
    assert np.allclose(result.patch_scores, result.nll)
    assert result.patch_scores[-1] > result.patch_scores[0]
    assert result.patch_scores[-2] > result.patch_scores[1]


def test_identity_transform_mode_uses_standardized_features_as_latents() -> None:
    rng = np.random.default_rng(47)
    support = normal_array(rng, loc=0.0, scale=0.2, size=(32, 4))
    batch = normal_array(rng, loc=0.0, scale=0.2, size=(8, 4))
    config = FlowTTEConfig(
        flow=FlowConfig(
            n_coupling_layers=2,
            n_epochs=2,
            batch_size=16,
            seed=47,
            transform_mode="identity",
        ),
        expansion=ExpansionConfig(budget=1.0, density_quantile=0.95, random_seed=2),
        score=ScoreConfig(density_weight=0.0, top_percent=0.25),
        device="cpu",
    )
    pipeline = FlowTTE(config)
    stats = pipeline.fit(support)

    result = pipeline.score_static(batch)

    assert len(stats.losses) == 1
    assert result.patch_scores.shape == (8,)
    assert result.nll.shape == (8,)
    assert result.memory_size_after == result.memory_size_before
