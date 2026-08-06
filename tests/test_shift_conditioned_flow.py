from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

import flow_tte.shift_conditioned as shift_conditioned
from flow_tte.flow import PatchNormalizingFlow
from flow_tte.shift_conditioned import (
    PairedRelativeShiftPredictor,
    RegisterShiftDescriptor,
    ShiftConditionedPatchLocalFlow,
    baseline_identity_mse,
    normalized_huber_invariance_loss,
    normalized_nll_loss,
    residual_magnitude_loss,
    teacher_feature_knn_kl_loss,
)


def _base_flow(dim: int = 6) -> PatchNormalizingFlow:
    torch.manual_seed(7)
    return PatchNormalizingFlow(dim=dim, n_coupling_layers=3, hidden_multiplier=2, clamp=1.2)


def _patches(batch_size: int = 2, patch_count: int = 5, dim: int = 6) -> torch.Tensor:
    return torch.arange(batch_size * patch_count * dim, dtype=torch.float32).reshape(
        batch_size,
        patch_count,
        dim,
    ) / 100.0


def _conditioned_flow(base: PatchNormalizingFlow) -> ShiftConditionedPatchLocalFlow:
    torch.manual_seed(11)
    return ShiftConditionedPatchLocalFlow(base, patch_dim=6)


def _make_residual_branch_nonzero(flow: ShiftConditionedPatchLocalFlow) -> None:
    with torch.no_grad():
        for idx, head in enumerate(flow.residual_heads):
            output = head.delta
            output.weight.fill_(0.001 * float(idx + 1))
            output.bias.fill_(0.01 * float(idx + 1))


def test_zero_residual_is_exactly_base_equivalent_and_freezes_base() -> None:
    base = _base_flow()
    flow = _conditioned_flow(base)
    patches = _patches()
    flat = patches.reshape(-1, patches.shape[-1])

    conditioned = flow.forward(patches)
    base_z, base_logdet = base.forward(flat)

    assert torch.equal(conditioned.z.reshape_as(base_z), base_z)
    assert torch.equal(conditioned.logdet.reshape_as(base_logdet), base_logdet)
    assert max(
        (conditioned.z.reshape_as(base_z) - base_z).abs().max().item(),
        (conditioned.logdet.reshape_as(base_logdet) - base_logdet).abs().max().item(),
    ) < 1e-6
    assert all(not parameter.requires_grad for parameter in base.parameters())


def test_nonzero_residual_is_invertible_and_logdet_cancels() -> None:
    base = _base_flow()
    flow = _conditioned_flow(base)
    _make_residual_branch_nonzero(flow)
    patches = _patches()

    forward = flow.forward(patches)
    reverse = flow.forward(forward.z, reverse=True, descriptor=forward.descriptor)

    assert not torch.allclose(forward.z, patches)
    assert not torch.allclose(forward.logdet, torch.zeros_like(forward.logdet))
    assert torch.allclose(reverse.z, patches, atol=1e-5)
    assert torch.allclose(
        forward.logdet + reverse.logdet,
        torch.zeros_like(forward.logdet),
        atol=1e-5,
    )


def test_reverse_requires_forward_descriptor() -> None:
    base = _base_flow()
    flow = _conditioned_flow(base)
    _make_residual_branch_nonzero(flow)
    patches = _patches()

    forward = flow.forward(patches)

    with pytest.raises(RuntimeError, match="reverse requires"):
        _ = flow.forward(forward.z, reverse=True)
    reverse = flow.forward(forward.z, reverse=True, descriptor=forward.descriptor)
    assert torch.allclose(reverse.z, patches, atol=1e-5)


def test_gradient_liveness_at_zero_initialization_reaches_residual_and_register_path() -> None:
    base = _base_flow()
    flow = _conditioned_flow(base)
    predictor = PairedRelativeShiftPredictor(num_transform_types=3, regression_dim=2)
    patches = _patches().requires_grad_(requires_grad=True)
    augmented = (patches + 0.05).detach().requires_grad_(requires_grad=True)

    output = flow.forward(patches)
    augmented_descriptor = flow.encode_shift(augmented).descriptor
    logits, regression = predictor.forward(output.descriptor, augmented_descriptor)
    labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    target = torch.zeros(2, 2)
    shift_loss = functional.binary_cross_entropy_with_logits(logits, labels)
    shift_loss = shift_loss + functional.smooth_l1_loss(regression, target)
    loss = normalized_nll_loss(output.z, output.logdet) + shift_loss
    loss.backward()

    residual_grad = flow.residual_heads[0].delta.weight.grad
    register_grad = flow.descriptor.register_queries.grad
    projection_grad = flow.descriptor.patch_projection.weight.grad
    assert residual_grad is not None
    assert register_grad is not None
    assert projection_grad is not None
    assert residual_grad.abs().sum() > 0
    assert register_grad.abs().sum() > 0
    assert projection_grad.abs().sum() > 0


def test_descriptor_extractor_is_deterministic_and_leaves_patches_unchanged() -> None:
    torch.manual_seed(13)
    extractor = RegisterShiftDescriptor(patch_dim=6)
    patches = _patches()

    first = extractor.forward(patches)
    second = extractor.forward(patches)

    assert torch.equal(first.patch_tokens, patches)
    assert torch.equal(second.patch_tokens, patches)
    assert torch.allclose(first.descriptor, second.descriptor)
    assert torch.allclose(first.registers, second.registers)


def test_descriptor_shuffle_changes_output_after_nonzero_branch() -> None:
    base = _base_flow()
    flow = _conditioned_flow(base)
    _make_residual_branch_nonzero(flow)
    patches = _patches(batch_size=3)
    descriptor = flow.encode_shift(patches).descriptor
    shuffled = descriptor.roll(shifts=1, dims=0)

    original = flow.forward(patches, descriptor=descriptor)
    changed = flow.forward(patches, descriptor=shuffled)

    assert not torch.allclose(original.z, changed.z)


def test_paired_shift_predictor_and_loss_helpers_are_practical() -> None:
    torch.manual_seed(17)
    predictor = PairedRelativeShiftPredictor(
        descriptor_dim=64,
        num_transform_types=5,
        regression_dim=3,
    )
    original = torch.randn(4, 64)
    augmented = original + 0.1 * torch.randn(4, 64)
    logits, regression = predictor.forward(original, augmented)
    labels = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ],
    )
    target = torch.zeros(4, 3)
    cls_loss = functional.binary_cross_entropy_with_logits(logits, labels)
    reg_loss = functional.smooth_l1_loss(regression, target)
    invariant = normalized_huber_invariance_loss(original, augmented)
    teacher_kl = teacher_feature_knn_kl_loss(augmented, original, max_samples=4, k=2)
    identity = baseline_identity_mse(augmented, original)
    residual = residual_magnitude_loss({"residual_magnitude": torch.tensor(0.25)})
    nll = normalized_nll_loss(torch.zeros(2, 3, 6), torch.zeros(2, 3))

    assert logits.shape == (4, 5)
    assert regression.shape == (4, 3)
    assert cls_loss >= 0
    assert reg_loss >= 0
    assert invariant >= 0
    assert teacher_kl >= 0
    assert identity > 0
    assert residual.item() == 0.25
    assert nll > 0


def test_module_does_not_import_raw_or_normalized_memory_scorer() -> None:
    imported = set(shift_conditioned.__dict__)

    assert "ScoreCalibration" not in imported
    assert "ScoreInputs" not in imported
    assert "score_flow_memory" not in imported
