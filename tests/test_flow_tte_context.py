from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt
import pytest
import torch

from flow_tte.config import ExpansionConfig, FlowConfig, FlowTTEConfig, ScoreConfig
from flow_tte.flow import PatchNormalizingFlow
from flow_tte.memory import TorchMemoryBank
from flow_tte.pipeline import FlowTTE
from flow_tte.scoring import ScoreCalibration

FloatArray = npt.NDArray[np.float32]


def normal_array(
    rng: np.random.Generator,
    loc: float,
    scale: float,
    size: Tuple[int, ...],
) -> FloatArray:
    return rng.normal(loc=loc, scale=scale, size=size).astype(np.float32)


def two_image_contexts() -> FloatArray:
    return np.array(
        [
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        ],
        dtype=np.float32,
    )


def test_conditional_affine_flow_is_invertible() -> None:
    flow = PatchNormalizingFlow(
        dim=6,
        n_coupling_layers=4,
        hidden_multiplier=2,
        clamp=1.5,
        condition_dim=3,
    )
    x = torch.arange(66, dtype=torch.float32).reshape(11, 6) / 10.0
    condition = torch.arange(33, dtype=torch.float32).reshape(11, 3) / 10.0

    z, logdet = flow.forward(x, condition=condition)
    x_recon, reverse_logdet = flow.forward(z, reverse=True, condition=condition)

    assert torch.allclose(x, x_recon, atol=1e-5)
    assert torch.allclose(logdet + reverse_logdet, torch.zeros_like(logdet), atol=1e-5)


def test_memory_query_context_penalty_changes_nearest_patch() -> None:
    memory = torch.tensor([[0.0, 0.0], [0.05, 0.0]], dtype=torch.float32)
    contexts = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    query = torch.tensor([[0.04, 0.0]], dtype=torch.float32)
    query_contexts = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    bank = TorchMemoryBank()
    bank.fit(memory, contexts=contexts)

    latent_only = bank.query(query, k=1, query_contexts=query_contexts, context_weight=0.0)
    context_penalized = bank.query(query, k=1, query_contexts=query_contexts, context_weight=1.0)

    assert int(latent_only.indices[0, 0]) == 1
    assert int(context_penalized.indices[0, 0]) == 0


def test_score_calibration_can_use_deterministic_sample_cap() -> None:
    features = torch.arange(40, dtype=torch.float32).reshape(20, 2)

    calibration = ScoreCalibration.fit(
        features,
        ScoreConfig(calibration_sample_size=5, query_chunk_size=4),
    )

    assert calibration.distance_std > 0.0


def test_memory_query_top_m_context_routes_to_context_group() -> None:
    memory = torch.tensor(
        [[1.0, 0.0], [1.1, 0.0], [0.01, 0.0], [0.02, 0.0]],
        dtype=torch.float32,
    )
    contexts = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    query = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    query_contexts = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    bank = TorchMemoryBank()
    bank.fit(memory, contexts=contexts)

    latent_only = bank.query(query, k=1, query_contexts=query_contexts)
    routed = bank.query(query, k=1, query_contexts=query_contexts, context_top_m=1)

    assert int(latent_only.indices[0, 0]) == 2
    assert int(routed.indices[0, 0]) == 0


def test_flow_tte_scores_with_context_penalty() -> None:
    rng = np.random.default_rng(29)
    support = normal_array(rng, loc=0.0, scale=0.3, size=(2, 4, 4))
    support_contexts = two_image_contexts()
    batch = normal_array(rng, loc=0.0, scale=0.3, size=(1, 4, 4))
    batch_contexts: FloatArray = np.array([[[1.0, 0.0]] * 4], dtype=np.float32)
    config = FlowTTEConfig(
        flow=FlowConfig(n_coupling_layers=2, n_epochs=2, batch_size=8, seed=29),
        expansion=ExpansionConfig(budget=1.0, density_quantile=0.95, random_seed=1),
        score=ScoreConfig(
            density_weight=0.2,
            context_mode="soft_penalty",
            context_weight=0.1,
            top_percent=0.25,
        ),
        device="cpu",
    )
    pipeline = FlowTTE(config)
    _ = pipeline.fit(support, support_contexts=support_contexts)

    result = pipeline.score_static(batch, batch_contexts=batch_contexts)

    assert result.patch_scores.shape == (1, 4)
    assert result.image_scores.shape == (1,)
    assert result.memory_size_after == result.memory_size_before


def test_flow_tte_scores_with_context_top_m_routing() -> None:
    rng = np.random.default_rng(31)
    support = normal_array(rng, loc=0.0, scale=0.3, size=(2, 4, 4))
    support_contexts = two_image_contexts()
    batch = normal_array(rng, loc=0.0, scale=0.3, size=(1, 4, 4))
    batch_contexts: FloatArray = np.array([[[1.0, 0.0]] * 4], dtype=np.float32)
    config = FlowTTEConfig(
        flow=FlowConfig(n_coupling_layers=2, n_epochs=2, batch_size=8, seed=31),
        expansion=ExpansionConfig(budget=1.0, density_quantile=0.95, random_seed=1),
        score=ScoreConfig(
            density_weight=0.2,
            context_mode="top_m",
            context_top_m=1,
            top_percent=0.25,
        ),
        device="cpu",
    )
    pipeline = FlowTTE(config)
    _ = pipeline.fit(support, support_contexts=support_contexts)

    result = pipeline.score_static(batch, batch_contexts=batch_contexts)

    assert result.patch_scores.shape == (1, 4)
    assert result.image_scores.shape == (1,)
    assert result.memory_size_after == result.memory_size_before


def test_flow_tte_conditional_nf_requires_contexts() -> None:
    rng = np.random.default_rng(37)
    support = normal_array(rng, loc=0.0, scale=0.3, size=(2, 4, 4))
    config = FlowTTEConfig(
        flow=FlowConfig(
            n_coupling_layers=2,
            n_epochs=2,
            batch_size=8,
            seed=37,
            condition_mode="context",
        ),
        expansion=ExpansionConfig(budget=1.0, density_quantile=0.95, random_seed=1),
        score=ScoreConfig(density_weight=0.2, top_percent=0.25),
        device="cpu",
    )
    pipeline = FlowTTE(config)

    with pytest.raises(RuntimeError, match="condition context"):
        _ = pipeline.fit(support)


def test_flow_tte_scores_with_conditional_nf() -> None:
    rng = np.random.default_rng(41)
    support = normal_array(rng, loc=0.0, scale=0.3, size=(2, 4, 4))
    support_contexts = two_image_contexts()
    batch = normal_array(rng, loc=0.0, scale=0.3, size=(1, 4, 4))
    batch_contexts: FloatArray = np.array([[[1.0, 0.0]] * 4], dtype=np.float32)
    config = FlowTTEConfig(
        flow=FlowConfig(
            n_coupling_layers=2,
            n_epochs=2,
            batch_size=8,
            seed=41,
            condition_mode="context",
        ),
        expansion=ExpansionConfig(budget=1.0, density_quantile=0.95, random_seed=1),
        score=ScoreConfig(density_weight=0.2, top_percent=0.25),
        device="cpu",
    )
    pipeline = FlowTTE(config)
    _ = pipeline.fit(support, support_contexts=support_contexts)

    result = pipeline.score_static(batch, batch_contexts=batch_contexts)

    assert result.patch_scores.shape == (1, 4)
    assert result.image_scores.shape == (1,)
    assert result.memory_size_after == result.memory_size_before


def test_flow_tte_splits_condition_and_memory_contexts() -> None:
    rng = np.random.default_rng(43)
    support = normal_array(rng, loc=0.0, scale=0.3, size=(2, 4, 4))
    support_condition_contexts = np.ones((2, 4, 3), dtype=np.float32)
    support_memory_contexts = two_image_contexts()
    batch = normal_array(rng, loc=0.0, scale=0.3, size=(1, 4, 4))
    batch_condition_contexts = np.ones((1, 4, 3), dtype=np.float32)
    batch_memory_contexts: FloatArray = np.array([[[1.0, 0.0]] * 4], dtype=np.float32)
    config = FlowTTEConfig(
        flow=FlowConfig(
            n_coupling_layers=2,
            n_epochs=2,
            batch_size=8,
            seed=43,
            condition_mode="context",
        ),
        expansion=ExpansionConfig(budget=1.0, density_quantile=0.95, random_seed=1),
        score=ScoreConfig(
            density_weight=0.2,
            context_mode="top_m",
            context_top_m=1,
            top_percent=0.25,
        ),
        device="cpu",
    )
    pipeline = FlowTTE(config)
    _ = pipeline.fit(
        support,
        support_contexts=support_condition_contexts,
        memory_contexts=support_memory_contexts,
    )
    assert pipeline.memory is not None
    assert pipeline.memory.m0_contexts is not None
    assert pipeline.memory.m0_contexts.shape[1] == 2

    result = pipeline.score_static(
        batch,
        batch_contexts=batch_condition_contexts,
        memory_contexts=batch_memory_contexts,
    )

    assert result.patch_scores.shape == (1, 4)
    assert result.image_scores.shape == (1,)
