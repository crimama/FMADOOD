from __future__ import annotations

import numpy as np
import torch

from flow_tte.config import FlowConfig, ScoreConfig
from flow_tte.conv2d_flow import Conv2DFlowDensityEstimator, Conv2DNormalizingFlow
from flow_tte.memory import TorchMemoryBank
from flow_tte.scoring import ScoreCalibration, ScoreInputs, score_flow_memory


def test_conv2d_flow_is_invertible() -> None:
    flow = Conv2DNormalizingFlow(
        channels=6,
        n_coupling_layers=4,
        hidden_channels=8,
        clamp=1.5,
    )
    x = torch.arange(2 * 6 * 3 * 4, dtype=torch.float32).reshape(2, 6, 3, 4) / 100.0

    z, logdet = flow.forward(x)
    x_recon, reverse_logdet = flow.forward(z, reverse=True)

    assert torch.allclose(x, x_recon, atol=1e-5)
    assert torch.allclose(logdet + reverse_logdet, torch.zeros_like(logdet), atol=1e-5)


def test_decoflow_spatial_context_flow_is_invertible() -> None:
    # Given: an asymmetric coupling flow whose scale subnet receives local context.
    flow = Conv2DNormalizingFlow(
        channels=6,
        n_coupling_layers=2,
        hidden_channels=8,
        clamp=1.5,
        spatial_context=True,
    )
    generator = torch.Generator().manual_seed(19)
    with torch.no_grad():
        for parameter in flow.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.03)
    x = torch.arange(2 * 6 * 3 * 4, dtype=torch.float32).reshape(2, 6, 3, 4) / 100.0

    # When: the spatial flow is applied and exactly reversed.
    z, logdet = flow.forward(x)
    x_recon, reverse_logdet = flow.forward(z, reverse=True)

    # Then: context injection preserves coupling invertibility and its Jacobian.
    assert not torch.allclose(x, z)
    assert not torch.allclose(logdet, torch.zeros_like(logdet))
    assert torch.allclose(x, x_recon, atol=1e-5)
    assert torch.allclose(logdet + reverse_logdet, torch.zeros_like(logdet), atol=1e-5)


def test_conv2d_estimator_outputs_memory_scorer_compatible_tensors() -> None:
    rng = np.random.default_rng(7)
    support_maps = tuple(
        rng.normal(loc=0.0, scale=0.2, size=(3, 4, 6)).astype(np.float32) for _ in range(3)
    )
    query_map = rng.normal(loc=0.1, scale=0.2, size=(3, 4, 6)).astype(np.float32)
    estimator = Conv2DFlowDensityEstimator(
        dim=6,
        config=FlowConfig(n_coupling_layers=2, n_epochs=2, batch_size=24, seed=7),
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


def test_spatial_context_estimator_trains_and_evaluates_finite_scores() -> None:
    rng = np.random.default_rng(23)
    support_maps = tuple(
        rng.normal(loc=0.0, scale=0.2, size=(3, 4, 6)).astype(np.float32) for _ in range(3)
    )
    estimator = Conv2DFlowDensityEstimator(
        dim=6,
        config=FlowConfig(
            n_coupling_layers=2,
            n_epochs=2,
            batch_size=24,
            seed=23,
            spatial_context=True,
        ),
        device="cpu",
    )

    stats = estimator.fit(support_maps, density_quantile=0.9)
    evaluation = estimator.evaluate_many(support_maps)

    assert len(stats.losses) == 2
    assert np.isfinite(stats.losses).all()
    assert evaluation.z.shape == (36, 6)
    assert evaluation.nll.shape == (36,)
    assert torch.isfinite(evaluation.z).all()
    assert torch.isfinite(evaluation.nll).all()
