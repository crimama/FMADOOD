from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from typing_extensions import final, override

from flow_tte.config import FlowConfig
from flow_tte.losses import patch_nll, tail_aware_nll
from flow_tte.spatial_context import SpatialContextCouplingNet
from flow_tte.tensors import resolve_device
from flow_tte.trainer import FeatureStandardizer, FlowTrainingStats, _seed_torch


@dataclass(frozen=True)
class Conv2DFlowEvaluation:
    z: torch.Tensor
    nll: torch.Tensor
    spatial_shape: Tuple[int, int]


@final
class Conv2DCouplingNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )
        last = self.net[-1]
        if not isinstance(last, nn.Conv2d):
            raise TypeError("Conv2DCouplingNet final module must be Conv2d")
        _ = nn.init.zeros_(last.weight)
        if last.bias is not None:
            _ = nn.init.zeros_(last.bias)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.forward(x)


@final
class Conv2DAffineCouplingBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        clamp: float,
        flip: bool,
        spatial_context: bool = False,
    ) -> None:
        super().__init__()
        if channels < 2:
            raise RuntimeError("Conv2D flow requires at least two channels")
        self.clamp: float = clamp
        self.flip: bool = flip
        self.left_channels: int = channels // 2
        self.right_channels: int = channels - self.left_channels
        input_channels = self.right_channels if flip else self.left_channels
        active_channels = self.left_channels if flip else self.right_channels
        self.net: Conv2DCouplingNet | None = None
        self.spatial_net: SpatialContextCouplingNet | None = None
        if spatial_context:
            self.spatial_net = SpatialContextCouplingNet(
                input_channels,
                active_channels,
                hidden_channels,
            )
        else:
            self.net = Conv2DCouplingNet(
                input_channels,
                active_channels * 2,
                hidden_channels,
            )

    @override
    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        left, right = torch.split(x, [self.left_channels, self.right_channels], dim=1)
        identity = right if self.flip else left
        active = left if self.flip else right
        if self.spatial_net is not None:
            raw_scale, raw_shift = self.spatial_net.forward(identity)
        elif self.net is not None:
            raw_scale, raw_shift = self.net.forward(identity).chunk(2, dim=1)
        else:
            raise RuntimeError("Conv2D coupling subnet is not configured")
        scale = self.clamp * torch.tanh(raw_scale)
        if reverse:
            transformed = (active - raw_shift) * torch.exp(-scale)
            logdet = -scale.sum(dim=1)
        else:
            transformed = active * torch.exp(scale) + raw_shift
            logdet = scale.sum(dim=1)
        if self.flip:
            return torch.cat([transformed, identity], dim=1), logdet
        return torch.cat([identity, transformed], dim=1), logdet


@final
class Conv2DNormalizingFlow(nn.Module):
    def __init__(
        self,
        channels: int,
        n_coupling_layers: int,
        hidden_channels: int,
        clamp: float,
        spatial_context: bool = False,
    ) -> None:
        super().__init__()
        blocks = [
            Conv2DAffineCouplingBlock(
                channels=channels,
                hidden_channels=hidden_channels,
                clamp=clamp,
                flip=idx % 2 == 1,
                spatial_context=spatial_context,
            )
            for idx in range(n_coupling_layers)
        ]
        self.blocks: Tuple[Conv2DAffineCouplingBlock, ...] = tuple(blocks)
        self.registered_blocks: nn.ModuleList = nn.ModuleList(blocks)

    @override
    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet = torch.zeros(
            x.shape[0],
            x.shape[2],
            x.shape[3],
            device=x.device,
            dtype=x.dtype,
        )
        blocks = reversed(self.blocks) if reverse else self.blocks
        out = x
        for block in blocks:
            out, block_logdet = block.forward(out, reverse=reverse)
            logdet = logdet + block_logdet
        return out, logdet


@final
class Conv2DFlowDensityEstimator:
    def __init__(self, dim: int, config: FlowConfig, device: str) -> None:
        self.config: FlowConfig = config
        self.device: torch.device = resolve_device(device)
        _seed_torch(config.seed, self.device)
        hidden_channels = min(256, max(32, dim // max(config.hidden_multiplier, 1)))
        self.flow: Conv2DNormalizingFlow = Conv2DNormalizingFlow(
            channels=dim,
            n_coupling_layers=config.n_coupling_layers,
            hidden_channels=hidden_channels,
            clamp=config.clamp,
            spatial_context=config.spatial_context,
        ).to(self.device)
        self.standardizer: FeatureStandardizer | None = None
        self.train_nll_mean: float = 0.0
        self.train_nll_std: float = 1.0
        self.density_threshold: float = 0.0

    def fit(
        self,
        feature_maps: Sequence[np.ndarray],
        density_quantile: float,
    ) -> FlowTrainingStats:
        x = self._maps_to_tensor(feature_maps)
        if self.config.standardize:
            flat = self._flatten_channels_last(x)
            self.standardizer = FeatureStandardizer.fit(flat)
            x = self._restore_channels_first(self.standardizer.transform(flat), x.shape)
        if self.config.transform_mode == "identity":
            train_nll = self._identity_nll(x)
            return self._store_stats(
                [float(train_nll.mean().detach().cpu())],
                train_nll,
                density_quantile,
            )

        image_batch_size = max(1, self.config.batch_size // (x.shape[2] * x.shape[3]))
        generator = torch.Generator(device=self.device)
        _ = generator.manual_seed(self.config.seed)
        optimizer = torch.optim.AdamW(self.flow.parameters(), lr=self.config.lr)
        losses: list[float] = []
        _ = self.flow.train()
        for _epoch in range(self.config.n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            order = torch.randperm(x.shape[0], generator=generator, device=self.device)
            for start in range(0, x.shape[0], image_batch_size):
                image_batch = x[order[start : start + image_batch_size]]
                z, logdet = self.flow.forward(image_batch)
                nll = tail_aware_nll(
                    z.permute(0, 2, 3, 1),
                    logdet,
                    tail_weight=self.config.tail_weight,
                    tail_top_k_ratio=self.config.tail_top_k_ratio,
                )
                loss = nll + self.config.lambda_logdet * logdet.square().mean()
                optimizer.zero_grad()
                loss.backward()
                _ = clip_grad_norm_(self.flow.parameters(), max_norm=0.5)
                optimizer.step()
                epoch_loss += float(nll.detach().cpu())
                n_batches += 1
            losses.append(epoch_loss / max(n_batches, 1))
        train_nll = self.evaluate_many(feature_maps).nll
        return self._store_stats(losses, train_nll, density_quantile)

    def evaluate_many(self, feature_maps: Sequence[np.ndarray]) -> Conv2DFlowEvaluation:
        x = self._maps_to_tensor(feature_maps)
        if self.standardizer is not None:
            flat = self._flatten_channels_last(x)
            x = self._restore_channels_first(self.standardizer.transform(flat), x.shape)
        if self.config.transform_mode == "identity":
            z = x
            nll_map = self._identity_nll(x)
        else:
            _ = self.flow.eval()
            with torch.no_grad():
                z, logdet = self.flow.forward(x)
                nll_map = patch_nll(z.permute(0, 2, 3, 1), logdet)
        height, width = int(x.shape[2]), int(x.shape[3])
        return Conv2DFlowEvaluation(
            z=z.permute(0, 2, 3, 1).reshape(-1, z.shape[1]),
            nll=nll_map.reshape(-1),
            spatial_shape=(height, width),
        )

    def evaluate(self, feature_map: np.ndarray) -> Conv2DFlowEvaluation:
        return self.evaluate_many((feature_map,))

    def density_penalty(self, nll: torch.Tensor) -> torch.Tensor:
        return torch.relu((nll - self.train_nll_mean) / self.train_nll_std)

    def _store_stats(
        self,
        losses: list[float],
        train_nll: torch.Tensor,
        density_quantile: float,
    ) -> FlowTrainingStats:
        self.train_nll_mean = float(train_nll.mean().detach().cpu())
        self.train_nll_std = float(train_nll.std(unbiased=False).clamp_min(1e-6).detach().cpu())
        self.density_threshold = float(torch.quantile(train_nll, density_quantile).detach().cpu())
        return FlowTrainingStats(
            losses=losses,
            train_nll_mean=self.train_nll_mean,
            train_nll_std=self.train_nll_std,
            density_threshold=self.density_threshold,
        )

    def _maps_to_tensor(self, feature_maps: Sequence[np.ndarray]) -> torch.Tensor:
        if not feature_maps:
            raise RuntimeError("Conv2D flow requires at least one feature map")
        array = np.stack(feature_maps, axis=0).astype(np.float32, copy=False)
        if array.ndim != 4:
            raise RuntimeError("Conv2D flow expects feature maps shaped HxWxC")
        return torch.as_tensor(array, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2)

    @staticmethod
    def _flatten_channels_last(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])

    @staticmethod
    def _restore_channels_first(flat: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        return flat.reshape(shape[0], shape[2], shape[3], shape[1]).permute(0, 3, 1, 2)

    @staticmethod
    def _identity_nll(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x.square().sum(dim=1)
