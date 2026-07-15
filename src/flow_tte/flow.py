from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from typing_extensions import final, override


@final
class CouplingNet(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.input: nn.Linear = nn.Linear(dim + condition_dim, hidden_dim)
        self.hidden: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.output: nn.Linear = nn.Linear(hidden_dim, dim * 2)
        self.activation: nn.GELU = nn.GELU()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation.forward(self.input.forward(x))
        out = self.activation.forward(self.hidden.forward(out))
        return self.output.forward(out)


@final
class SpatialContextCouplingNet(nn.Module):
    """MLP coupling whose scale branch receives a gated local spatial context."""

    def __init__(self, dim: int, hidden_dim: int, mask: torch.Tensor) -> None:
        super().__init__()
        active_dim = int((1.0 - mask).sum().item())
        identity_dim = int(mask.sum().item())
        self.context = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.register_buffer("identity_mask", mask.reshape(1, dim))
        self.context_scale = nn.Parameter(torch.tensor(-2.1972246))
        self.scale_net = CouplingNet(identity_dim, hidden_dim, identity_dim)
        self.shift_net = CouplingNet(identity_dim, hidden_dim, 0)
        self.active_dim = active_dim
        _ = nn.init.zeros_(self.scale_net.output.weight)
        _ = nn.init.zeros_(self.scale_net.output.bias)
        _ = nn.init.zeros_(self.shift_net.output.weight)
        _ = nn.init.zeros_(self.shift_net.output.bias)

    def forward(
        self, identity: torch.Tensor, spatial_shape: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        height, width = spatial_shape
        n_images = identity.shape[0] // (height * width)
        context_input = identity.reshape(
            n_images, height, width, identity.shape[-1],
        ).permute(0, 3, 1, 2)
        context = self.context(context_input)
        context = context * (0.2 * torch.sigmoid(self.context_scale))
        context = context.permute(0, 2, 3, 1).reshape(-1, context.shape[1])
        context = context[:, self.identity_mask[0].bool()]
        identity_selected = identity[:, self.identity_mask[0].bool()]
        scale_shift = self.scale_net.forward(torch.cat([identity_selected, context], dim=-1))
        raw_scale, _ = scale_shift.chunk(2, dim=-1)
        raw_shift = self.shift_net.forward(identity_selected).chunk(2, dim=-1)[1]
        return raw_scale, raw_shift


@final
class AffineCouplingBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        mask: torch.Tensor,
        clamp: float,
        condition_dim: int = 0,
        spatial_context: bool = False,
    ) -> None:
        super().__init__()
        self.clamp: float = clamp
        self.condition_dim: int = condition_dim
        self.mask: torch.Tensor
        self.register_buffer("mask", mask.reshape(1, dim))
        self.net: CouplingNet = CouplingNet(dim, hidden_dim, condition_dim)
        self.spatial_net = (
            SpatialContextCouplingNet(dim, hidden_dim, mask) if spatial_context else None
        )
        _ = nn.init.zeros_(self.net.output.weight)
        _ = nn.init.zeros_(self.net.output.bias)

    @override
    def forward(
        self,
        x: torch.Tensor,
        reverse: bool = False,
        condition: Optional[torch.Tensor] = None,
        spatial_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = self.mask
        identity = x * mask
        net_input = identity
        if self.condition_dim > 0:
            if condition is None:
                raise RuntimeError("Conditional flow block requires condition context")
            if condition.ndim != 2:
                raise RuntimeError("Conditional flow context must be 2D")
            if condition.shape[0] != x.shape[0]:
                raise RuntimeError("Conditional flow context row count must match features")
            if condition.shape[1] != self.condition_dim:
                raise RuntimeError("Conditional flow context dimension mismatch")
            net_input = torch.cat([identity, condition], dim=-1)
        if self.spatial_net is not None:
            if spatial_shape is None:
                raise RuntimeError("Spatial context flow requires spatial shape")
            raw_scale, raw_shift = self.spatial_net.forward(identity, spatial_shape)
            selected = self.spatial_net.identity_mask[0].bool()
            full_scale = torch.zeros_like(x)
            full_shift = torch.zeros_like(x)
            full_scale[:, selected] = raw_scale
            full_shift[:, selected] = raw_shift
            raw_scale, raw_shift = full_scale, full_shift
        else:
            scale_shift = self.net.forward(net_input)
            raw_scale, raw_shift = scale_shift.chunk(2, dim=-1)
        active = 1.0 - mask
        scale = self.clamp * torch.tanh(raw_scale) * active
        shift = raw_shift * active

        if reverse:
            y = identity + active * ((x - shift) * torch.exp(-scale))
            logdet = -scale.sum(dim=-1)
            return y, logdet

        y = identity + active * (x * torch.exp(scale) + shift)
        logdet = scale.sum(dim=-1)
        return y, logdet


@final
class PatchNormalizingFlow(nn.Module):
    def __init__(
        self,
        dim: int,
        n_coupling_layers: int,
        hidden_multiplier: int,
        clamp: float,
        condition_dim: int = 0,
        spatial_context: bool = False,
    ) -> None:
        super().__init__()
        if condition_dim < 0:
            raise RuntimeError("condition_dim must be non-negative")
        self.condition_dim: int = condition_dim
        hidden_dim = max(dim * hidden_multiplier, 8)
        blocks: list[AffineCouplingBlock] = []
        base = torch.arange(dim, dtype=torch.float32) % 2.0
        for idx in range(n_coupling_layers):
            mask = base if idx % 2 == 0 else 1.0 - base
            blocks.append(
                AffineCouplingBlock(dim, hidden_dim, mask, clamp, condition_dim, spatial_context),
            )
        self.blocks: Tuple[AffineCouplingBlock, ...] = tuple(blocks)
        self.registered_blocks: nn.ModuleList = nn.ModuleList(blocks)

    @override
    def forward(
        self,
        x: torch.Tensor,
        reverse: bool = False,
        condition: Optional[torch.Tensor] = None,
        spatial_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        blocks = reversed(self.blocks) if reverse else self.blocks
        out = x
        for block in blocks:
            out, block_logdet = block.forward(
                out, reverse=reverse, condition=condition, spatial_shape=spatial_shape,
            )
            logdet = logdet + block_logdet
        return out, logdet
