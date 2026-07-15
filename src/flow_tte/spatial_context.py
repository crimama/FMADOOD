from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from typing_extensions import final, override


@final
class SpatialContextCouplingNet(nn.Module):
    """DeCoFlow-style asymmetric subnet: context for scale, identity for shift."""

    def __init__(self, channels: int, active_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.context_conv: nn.Conv2d = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )
        _ = nn.init.zeros_(self.context_conv.weight)
        if self.context_conv.bias is not None:
            _ = nn.init.zeros_(self.context_conv.bias)
        self.context_scale: nn.Parameter = nn.Parameter(torch.tensor(0.0))
        self.scale_net: nn.Sequential = self._build_net(
            channels * 2,
            hidden_channels,
            active_channels,
        )
        self.shift_net: nn.Sequential = self._build_net(
            channels,
            hidden_channels,
            active_channels,
        )

    @staticmethod
    def _build_net(in_channels: int, hidden_channels: int, out_channels: int) -> nn.Sequential:
        net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )
        output = net[-1]
        if not isinstance(output, nn.Conv2d):
            raise TypeError("Spatial context subnet output must be Conv2d")
        _ = nn.init.zeros_(output.weight)
        if output.bias is not None:
            _ = nn.init.zeros_(output.bias)
        return net

    @override
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        context = 0.2 * torch.sigmoid(self.context_scale) * self.context_conv.forward(x)
        scale = self.scale_net.forward(torch.cat([x, context], dim=1))
        shift = self.shift_net.forward(x)
        return scale, shift
