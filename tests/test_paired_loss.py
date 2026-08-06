from __future__ import annotations

import torch

from flow_tte.paired_loss import (
    clean_identity_loss,
    deterministic_neighbor_retention,
    effective_rank,
    effective_rank_ratio,
    identity_anchor_loss,
    latent_norm_preservation_loss,
    neighbor_identity_retention,
    normalized_huber_pair_loss,
    normalized_pair_huber_loss,
    raw_pair_distance_loss,
    raw_paired_canonicalization_loss,
)


def test_raw_pair_loss_is_radial_sensitive_while_normalized_control_is_not() -> None:
    target = torch.eye(4)
    scaled = target * 2.0
    assert raw_paired_canonicalization_loss(scaled, target) > 0.4
    assert normalized_huber_pair_loss(scaled, target) < 1e-6


def test_random_pair_has_larger_raw_loss_than_matched_pair() -> None:
    generator = torch.Generator().manual_seed(0)
    target = torch.randn(16, 8, generator=generator)
    matched = target + 0.01 * torch.randn(16, 8, generator=generator)
    random_target = target.roll(shifts=1, dims=0)
    assert raw_paired_canonicalization_loss(matched, target) < raw_paired_canonicalization_loss(
        matched, random_target,
    )


def test_losses_are_finite_and_provide_gradients() -> None:
    latent = torch.randn(3, 5, 7, requires_grad=True)
    teacher = torch.randn(3, 5, 7)
    loss = (
        raw_paired_canonicalization_loss(latent, teacher)
        + normalized_huber_pair_loss(latent, teacher)
        + identity_anchor_loss(latent, teacher)
        + 0.1 * latent_norm_preservation_loss(latent, teacher)
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
    assert float(latent.grad.abs().sum()) > 0.0


def test_identity_and_norm_losses_use_feature_dimension_scaling() -> None:
    target = torch.zeros(2, 4)
    latent = torch.ones(2, 4)

    assert torch.allclose(identity_anchor_loss(latent, target), torch.tensor(1.0))
    assert torch.allclose(latent_norm_preservation_loss(latent, target), torch.tensor(1.0))


def test_effective_rank_and_ratio_detect_collapse() -> None:
    full = torch.eye(8)
    collapsed = torch.ones(8, 8)
    assert effective_rank(full) > effective_rank(collapsed)
    assert effective_rank_ratio(collapsed, full) < 0.5


def test_neighbor_retention_is_identity_sensitive() -> None:
    base = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    retained = deterministic_neighbor_retention(base, base, k=2)
    shuffled = deterministic_neighbor_retention(base.roll(shifts=3, dims=0), base, k=2)
    assert retained == 1.0
    assert shuffled < retained


def test_runner_contract_aliases_match_core_implementations() -> None:
    latent = torch.randn(5, 3)
    teacher = torch.randn(5, 3)
    assert torch.allclose(
        raw_pair_distance_loss(latent, teacher),
        raw_paired_canonicalization_loss(latent, teacher),
    )
    assert torch.allclose(
        normalized_pair_huber_loss(latent, teacher),
        normalized_huber_pair_loss(latent, teacher),
    )
    assert torch.allclose(
        clean_identity_loss(latent, teacher),
        identity_anchor_loss(latent, teacher),
    )
    assert torch.allclose(
        neighbor_identity_retention(latent, teacher, k=2),
        deterministic_neighbor_retention(latent, teacher, k=2),
    )
