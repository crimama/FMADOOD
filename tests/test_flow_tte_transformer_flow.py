from __future__ import annotations

import numpy as np
import torch

from flow_tte.config import FlowConfig, ScoreConfig
from flow_tte.memory import TorchMemoryBank
from flow_tte.scoring import ScoreCalibration, ScoreInputs, score_flow_memory
from flow_tte.transformer_flow import TransformerFlowDensityEstimator, TransformerNormalizingFlow


def test_transformer_flow_is_invertible() -> None:
    flow = TransformerNormalizingFlow(
        dim=6,
        n_coupling_layers=4,
        model_dim=8,
        clamp=1.5,
    )
    x = torch.arange(2 * 12 * 6, dtype=torch.float32).reshape(2, 12, 6) / 100.0

    z, logdet = flow.forward(x)
    x_recon, reverse_logdet = flow.forward(z, reverse=True)

    assert torch.allclose(x, x_recon, atol=1e-5)
    assert torch.allclose(logdet + reverse_logdet, torch.zeros_like(logdet), atol=1e-5)


def test_transformer_flow_is_invertible_with_context_tokens() -> None:
    flow = TransformerNormalizingFlow(
        dim=6,
        n_coupling_layers=4,
        model_dim=8,
        clamp=1.5,
        context_dim=6,
    )
    x = torch.arange(2 * 12 * 6, dtype=torch.float32).reshape(2, 12, 6) / 100.0
    context = torch.arange(2 * 5 * 6, dtype=torch.float32).reshape(2, 5, 6) / 50.0

    z, logdet = flow.forward(x, context_tokens=context)
    x_recon, reverse_logdet = flow.forward(z, context_tokens=context, reverse=True)

    assert z.shape == x.shape
    assert torch.allclose(x, x_recon, atol=1e-5)
    assert torch.allclose(logdet + reverse_logdet, torch.zeros_like(logdet), atol=1e-5)


def test_transformer_estimator_outputs_memory_scorer_compatible_tensors() -> None:
    rng = np.random.default_rng(17)
    support_maps = tuple(
        rng.normal(loc=0.0, scale=0.2, size=(3, 4, 6)).astype(np.float32)
        for _ in range(3)
    )
    query_map = rng.normal(loc=0.1, scale=0.2, size=(3, 4, 6)).astype(np.float32)
    estimator = TransformerFlowDensityEstimator(
        dim=6,
        config=FlowConfig(n_coupling_layers=2, n_epochs=2, batch_size=24, seed=17),
        device="cpu",
    )

    _ = estimator.fit(support_maps, density_quantile=0.9)
    support_eval = estimator.evaluate_many(support_maps)
    query_eval = estimator.evaluate(query_map)
    bank = TorchMemoryBank()
    bank.fit(support_eval.z)
    config = ScoreConfig(density_weight=0.0, top_percent=0.25, query_chunk_size=8)
    calibration = ScoreCalibration.fit(support_eval.z, config)
    image_indices = torch.zeros(query_eval.z.shape[0], dtype=torch.long)

    result = score_flow_memory(
        inputs=ScoreInputs(
            query_z=query_eval.z,
            nll=query_eval.nll,
            nll_penalty=estimator.density_penalty(query_eval.nll),
            image_indices=image_indices,
            n_images=1,
        ),
        bank=bank,
        config=config,
        calibration=calibration,
    )

    assert query_eval.spatial_shape == (3, 4)
    assert result.patch_scores.shape == (12,)
    assert result.image_scores.shape == (1,)


def test_transformer_estimator_accepts_context_tokens_without_scoring_context() -> None:
    rng = np.random.default_rng(23)
    support_maps = tuple(
        rng.normal(loc=0.0, scale=0.2, size=(2, 3, 6)).astype(np.float32)
        for _ in range(2)
    )
    support_contexts = tuple(
        rng.normal(loc=0.0, scale=0.2, size=(5, 6)).astype(np.float32)
        for _ in range(2)
    )
    query_map = rng.normal(loc=0.1, scale=0.2, size=(2, 3, 6)).astype(np.float32)
    query_context = rng.normal(loc=0.1, scale=0.2, size=(5, 6)).astype(np.float32)
    estimator = TransformerFlowDensityEstimator(
        dim=6,
        config=FlowConfig(n_coupling_layers=2, n_epochs=1, batch_size=12, seed=23),
        device="cpu",
        context_dim=6,
    )

    _ = estimator.fit(support_maps, density_quantile=0.9, context_tokens=support_contexts)
    support_eval = estimator.evaluate_many(support_maps, context_tokens=support_contexts)
    query_eval = estimator.evaluate(query_map, context_tokens=query_context)

    assert support_eval.z.shape == (12, 6)
    assert query_eval.z.shape == (6, 6)
    assert query_eval.spatial_shape == (2, 3)


def test_transformer_estimator_accepts_dummy_register_tokens() -> None:
    rng = np.random.default_rng(29)
    support_maps = tuple(
        rng.normal(loc=0.0, scale=0.2, size=(2, 3, 6)).astype(np.float32)
        for _ in range(2)
    )
    estimator = TransformerFlowDensityEstimator(
        dim=6,
        config=FlowConfig(n_coupling_layers=2, n_epochs=1, batch_size=12, seed=29),
        device="cpu",
        dummy_token_count=4,
        dummy_trainable=False,
    )

    _ = estimator.fit(support_maps, density_quantile=0.9)
    support_eval = estimator.evaluate_many(support_maps)

    assert support_eval.z.shape == (12, 6)
