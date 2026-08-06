from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, cast

import torch
import torch.nn.functional as functional
from torch import nn
from typing_extensions import final, override

from flow_tte.flow import PatchNormalizingFlow
from flow_tte.losses import patch_nll


@dataclass(frozen=True)
class ShiftDescriptorOutput:
    """Image-level shift descriptor plus unchanged patch tokens."""

    descriptor: torch.Tensor
    registers: torch.Tensor
    patch_tokens: torch.Tensor


@dataclass(frozen=True)
class ShiftConditionedFlowOutput:
    """Output of a shift-conditioned flow pass."""

    z: torch.Tensor
    logdet: torch.Tensor
    descriptor: torch.Tensor
    diagnostics: Mapping[str, torch.Tensor]


@final
class RegisterShiftDescriptor(nn.Module):
    """Extract a 64D per-image shift descriptor with learned register cross-attention.

    The input patch tensor is returned unchanged in ``ShiftDescriptorOutput.patch_tokens``.
    This module only summarizes patches into learned register tokens; it does not update,
    reorder, or mix the patch features consumed by the flow.
    """

    def __init__(  # noqa: PLR0913
        self,
        patch_dim: int,
        register_count: int = 4,
        model_dim: int = 128,
        num_heads: int = 4,
        ffn_dim: int = 256,
        descriptor_dim: int = 64,
    ) -> None:
        super().__init__()
        if patch_dim <= 0:
            raise RuntimeError("patch_dim must be positive")
        self.patch_dim = patch_dim
        self.register_count = register_count
        self.model_dim = model_dim
        self.descriptor_dim = descriptor_dim
        self.patch_projection = nn.Linear(patch_dim, model_dim)
        self.register_queries = nn.Parameter(torch.empty(register_count, model_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, model_dim),
        )
        self.ffn_norm = nn.LayerNorm(model_dim)
        self.descriptor_head = nn.Linear(model_dim, descriptor_dim)
        nn.init.normal_(self.register_queries, mean=0.0, std=0.02)

    @override
    def forward(self, patches: torch.Tensor) -> ShiftDescriptorOutput:
        if patches.ndim != 3:
            raise RuntimeError("patches must have shape [batch, patches, dim]")
        if patches.shape[-1] != self.patch_dim:
            raise RuntimeError("patch feature dimension mismatch")
        tokens = cast("torch.Tensor", self.patch_projection(patches))
        queries = self.register_queries.unsqueeze(0).expand(patches.shape[0], -1, -1)
        attended = cast(
            "Tuple[torch.Tensor, Optional[torch.Tensor]]",
            self.attn(queries, tokens, tokens, need_weights=False),
        )[0]
        registers = cast("torch.Tensor", self.attn_norm(queries + attended))
        registers = cast("torch.Tensor", self.ffn_norm(registers + self.ffn(registers)))
        descriptor = cast("torch.Tensor", self.descriptor_head(registers.mean(dim=1)))
        return ShiftDescriptorOutput(
            descriptor=descriptor,
            registers=registers,
            patch_tokens=patches,
        )


@final
class _ResidualFilmHead(nn.Module):
    """Descriptor FiLM plus zero-initialized delta head for one coupling block."""

    def __init__(  # noqa: PLR0913
        self,
        descriptor_dim: int,
        output_dim: int,
        hidden_dim: int,
        initial_gain: float,
        max_gain: float,
        film_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0.0 < initial_gain < max_gain:
            raise RuntimeError("initial_gain must be in (0, max_gain)")
        self.output_dim = output_dim
        self.max_gain = max_gain
        self.film_scale = film_scale
        self.norm = nn.LayerNorm(descriptor_dim)
        self.film = nn.Linear(descriptor_dim, hidden_dim * 2)
        self.delta = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        ratio = initial_gain / max_gain
        self.gain_logit = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio))))

    def _gain(self) -> torch.Tensor:
        return self.max_gain * torch.sigmoid(self.gain_logit)

    @override
    def forward(
        self,
        hidden: torch.Tensor,
        descriptor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        film_input = cast("torch.Tensor", self.norm(descriptor))
        film = cast("torch.Tensor", self.film(film_input))
        alpha, beta = film.chunk(2, dim=-1)
        alpha = self.film_scale * torch.tanh(alpha)
        beta = self.film_scale * torch.tanh(beta)
        h_tilde = hidden * (1.0 + alpha) + beta
        delta = cast("torch.Tensor", self.delta(h_tilde))
        return self._gain() * delta, delta


@final
class ShiftConditionedPatchLocalFlow(nn.Module):
    """Freeze a trained patch flow and add descriptor-conditioned residual blocks.

    The base ``PatchNormalizingFlow`` is traversed block-by-block.  Each frozen coupling
    MLP hidden state receives descriptor FiLM, then a zero-initialized residual head
    predicts delta scale/shift logits.  Patch rows stay independent after the descriptor
    is computed, and a single bounded nonzero gain is initialized to 0.05 and capped at
    0.1.  Zero residual heads give exact frozen-base parity.
    """

    def __init__(
        self,
        base_flow: PatchNormalizingFlow,
        patch_dim: int,
        descriptor: Optional[RegisterShiftDescriptor] = None,
        initial_gain: float = 0.05,
        max_gain: float = 0.1,
    ) -> None:
        super().__init__()
        self.base_flow = base_flow
        for parameter in self.base_flow.parameters():
            parameter.requires_grad = False
        self.patch_dim = patch_dim
        self.descriptor = descriptor or RegisterShiftDescriptor(patch_dim)
        self.residual_heads = nn.ModuleList(
            [
                _ResidualFilmHead(
                    descriptor_dim=self.descriptor.descriptor_dim,
                    output_dim=patch_dim * 2,
                    hidden_dim=block.net.hidden.out_features,
                    initial_gain=initial_gain,
                    max_gain=max_gain,
                )
                for block in self.base_flow.blocks
            ],
        )

    def encode_shift(self, patches: torch.Tensor) -> ShiftDescriptorOutput:
        """Return a per-image descriptor without modifying patch tokens."""

        return self.descriptor.forward(patches)

    def _prepare_condition(
        self,
        condition: Optional[torch.Tensor],
        batch_size: int,
        patches_per_image: int,
    ) -> Optional[torch.Tensor]:
        if condition is None:
            return None
        if condition.ndim == 3:
            if condition.shape[:2] != (batch_size, patches_per_image):
                raise RuntimeError("condition image and patch dimensions must match patches")
            return condition.reshape(batch_size * patches_per_image, condition.shape[-1])
        if condition.ndim == 2:
            if condition.shape[0] != batch_size * patches_per_image:
                raise RuntimeError("flat condition row count must match flattened patches")
            return condition
        raise RuntimeError("condition must be [batch, patches, dim] or [rows, dim]")

    def _repeat_descriptor(self, descriptor: torch.Tensor, patches_per_image: int) -> torch.Tensor:
        return (
            descriptor.unsqueeze(1)
            .expand(descriptor.shape[0], patches_per_image, descriptor.shape[-1])
            .reshape(descriptor.shape[0] * patches_per_image, descriptor.shape[-1])
        )

    def _block_forward(
        self,
        block_idx: int,
        x: torch.Tensor,
        descriptor_rows: torch.Tensor,
        condition: Optional[torch.Tensor],
        reverse: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        block = self.base_flow.blocks[block_idx]
        if block.spatial_net is not None:
            raise RuntimeError(
                "shift-conditioned residual flow does not support spatial_net blocks",
            )
        mask = block.mask
        identity = x * mask
        net_input = identity
        if block.condition_dim > 0:
            if condition is None:
                raise RuntimeError("conditional base flow requires condition context")
            net_input = torch.cat([identity, condition], dim=-1)
        elif condition is not None:
            net_input = identity
        hidden = cast("torch.Tensor", block.net.activation(block.net.input(net_input)))
        hidden = cast("torch.Tensor", block.net.activation(block.net.hidden(hidden)))
        base_raw = cast("torch.Tensor", block.net.output(hidden))
        delta_raw, delta_pre_gain = cast(
            "_ResidualFilmHead",
            self.residual_heads[block_idx],
        ).forward(hidden, descriptor_rows)
        raw_scale, raw_shift = (base_raw + delta_raw).chunk(2, dim=-1)
        active = 1.0 - mask
        scale = block.clamp * torch.tanh(raw_scale) * active
        shift = raw_shift * active
        if reverse:
            y = identity + active * ((x - shift) * torch.exp(-scale))
            logdet = -scale.sum(dim=-1)
        else:
            y = identity + active * (x * torch.exp(scale) + shift)
            logdet = scale.sum(dim=-1)
        delta_scale, delta_shift = delta_pre_gain.chunk(2, dim=-1)
        residual_l2 = delta_scale.square().mean() + delta_shift.square().mean()
        return y, logdet, residual_l2, delta_raw.detach().square().mean()

    @override
    def forward(
        self,
        patches: torch.Tensor,
        reverse: bool = False,
        descriptor: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        spatial_shape: Optional[Tuple[int, int]] = None,
    ) -> ShiftConditionedFlowOutput:
        if patches.ndim != 3:
            raise RuntimeError("patches must have shape [batch, patches, dim]")
        if patches.shape[-1] != self.patch_dim:
            raise RuntimeError("patch feature dimension mismatch")
        if reverse and descriptor is None:
            raise RuntimeError("reverse requires the descriptor used for the forward pass")
        batch_size, patches_per_image, feature_dim = patches.shape
        encoded = self.encode_shift(patches) if descriptor is None else None
        image_descriptor = encoded.descriptor if encoded is not None else descriptor
        if image_descriptor is None:
            raise RuntimeError("descriptor was not provided")
        if image_descriptor.shape != (batch_size, self.descriptor.descriptor_dim):
            raise RuntimeError("descriptor shape must be [batch, descriptor_dim]")
        has_spatial_base = any(block.spatial_net is not None for block in self.base_flow.blocks)
        if spatial_shape is not None and has_spatial_base:
            raise RuntimeError(
                "shift-conditioned residual flow does not support spatial_net blocks",
            )
        out = patches.reshape(batch_size * patches_per_image, feature_dim)
        flat_condition = self._prepare_condition(condition, batch_size, patches_per_image)
        descriptor_rows = self._repeat_descriptor(image_descriptor, patches_per_image)
        logdet = torch.zeros(out.shape[0], device=out.device, dtype=out.dtype)
        residual_l2 = out.new_zeros(())
        applied_residual_l2 = out.new_zeros(())
        block_indices: Sequence[int]
        if reverse:
            block_indices = range(len(self.base_flow.blocks) - 1, -1, -1)
        else:
            block_indices = range(len(self.base_flow.blocks))
        for block_idx in block_indices:
            out, block_logdet, block_residual_l2, block_applied_l2 = self._block_forward(
                block_idx=block_idx,
                x=out,
                descriptor_rows=descriptor_rows,
                condition=flat_condition,
                reverse=reverse,
            )
            logdet = logdet + block_logdet
            residual_l2 = residual_l2 + block_residual_l2
            applied_residual_l2 = applied_residual_l2 + block_applied_l2
        z = out.reshape(batch_size, patches_per_image, feature_dim)
        diagnostics: Dict[str, torch.Tensor] = {
            "residual_magnitude": residual_l2,
            "residual_l2": residual_l2,
            "applied_residual_l2": applied_residual_l2,
            "descriptor_l2": image_descriptor.norm(dim=-1).mean(),
        }
        return ShiftConditionedFlowOutput(
            z=z,
            logdet=logdet.reshape(batch_size, patches_per_image),
            descriptor=image_descriptor,
            diagnostics=diagnostics,
        )


@final
class PairedRelativeShiftPredictor(nn.Module):
    """Predict transform labels and continuous parameters from ``g_aug - g_orig``."""

    def __init__(
        self,
        descriptor_dim: int = 64,
        hidden_dim: int = 128,
        num_transform_types: int = 8,
        regression_dim: int = 4,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(descriptor_dim)
        self.input = nn.Linear(descriptor_dim, hidden_dim)
        self.activation = nn.GELU()
        self.type_head = nn.Linear(hidden_dim, num_transform_types)
        self.regression_head = nn.Linear(hidden_dim, regression_dim)

    @override
    def forward(
        self,
        g_orig: torch.Tensor,
        g_aug: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if g_orig.shape != g_aug.shape:
            raise RuntimeError("paired descriptors must have identical shape")
        hidden = cast("torch.Tensor", self.norm(g_aug - g_orig))
        hidden = cast("torch.Tensor", self.input(hidden))
        hidden = cast("torch.Tensor", self.activation(hidden))
        type_logits = cast("torch.Tensor", self.type_head(hidden))
        regression = cast("torch.Tensor", self.regression_head(hidden))
        return type_logits, regression


def normalized_huber_invariance_loss(
    original: torch.Tensor,
    augmented: torch.Tensor,
    delta: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Smooth-L1 invariance loss after per-row L2 normalization."""

    original_norm = functional.normalize(original, dim=-1, eps=eps)
    augmented_norm = functional.normalize(augmented, dim=-1, eps=eps)
    return functional.smooth_l1_loss(original_norm, augmented_norm, beta=delta)


def teacher_feature_knn_kl_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    max_samples: int = 256,
    k: int = 10,
    temperature: float = 0.1,
) -> torch.Tensor:
    """KL between teacher and student kNN neighborhoods on a deterministic sample."""

    if student_features.shape != teacher_features.shape:
        raise RuntimeError("student and teacher features must have identical shape")
    if student_features.ndim != 2:
        raise RuntimeError("features must be 2D [rows, dim]")
    n_rows = student_features.shape[0]
    if n_rows < 2:
        return student_features.new_zeros(())
    sample_count = min(max_samples, n_rows)
    if sample_count == n_rows:
        indices = torch.arange(n_rows, device=student_features.device)
    else:
        indices = torch.linspace(
            0,
            n_rows - 1,
            sample_count,
            device=student_features.device,
        ).round().long()
    student = functional.normalize(student_features.index_select(0, indices), dim=-1)
    teacher = functional.normalize(teacher_features.index_select(0, indices), dim=-1)
    teacher_logits_all = teacher @ teacher.T / temperature
    student_logits_all = student @ student.T / temperature
    eye = torch.eye(sample_count, device=teacher_logits_all.device, dtype=torch.bool)
    teacher_logits_all = teacher_logits_all.masked_fill(eye, float("-inf"))
    student_logits_all = student_logits_all.masked_fill(eye, float("-inf"))
    k_eff = min(k, sample_count - 1)
    neighbor_idx = torch.topk(teacher_logits_all, k_eff, largest=True, dim=-1).indices
    teacher_logits = teacher_logits_all.gather(1, neighbor_idx)
    student_logits = student_logits_all.gather(1, neighbor_idx)
    teacher_probs = functional.softmax(teacher_logits, dim=-1).detach()
    student_log_probs = functional.log_softmax(student_logits, dim=-1)
    return functional.kl_div(student_log_probs, teacher_probs, reduction="batchmean")


def baseline_identity_mse(candidate: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
    """MSE penalty that keeps a residual branch close to its frozen-base identity."""

    return functional.mse_loss(candidate, baseline)


def residual_magnitude_loss(
    diagnostics: Mapping[str, torch.Tensor],
    key: str = "residual_magnitude",
) -> torch.Tensor:
    """Return the residual magnitude diagnostic as a loss term."""

    if key not in diagnostics:
        raise RuntimeError("residual diagnostic is missing")
    return diagnostics[key]


def normalized_nll_loss(z: torch.Tensor, logdet: torch.Tensor) -> torch.Tensor:
    """Mean patch NLL normalized by feature dimension."""

    return (patch_nll(z, logdet) / max(z.shape[-1], 1)).mean()
