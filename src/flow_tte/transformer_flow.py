from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from typing_extensions import final, override

from flow_tte.config import FlowConfig
from flow_tte.losses import patch_nll, tail_aware_nll
from flow_tte.trainer import FeatureStandardizer, FlowTrainingStats, _seed_torch
from flow_tte.tensors import resolve_device


@dataclass(frozen=True)
class TransformerFlowEvaluation:
    z: torch.Tensor
    nll: torch.Tensor
    spatial_shape: Tuple[int, int]


@final
class TransformerCouplingNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        model_dim: int,
        context_dim: Optional[int] = None,
        dummy_token_count: int = 0,
        dummy_trainable: bool = False,
    ) -> None:
        super().__init__()
        if dummy_token_count < 0:
            raise RuntimeError("dummy_token_count must be non-negative")
        heads = _head_count(model_dim)
        self.input: nn.Linear = nn.Linear(in_dim, model_dim)
        self.context_input: Optional[nn.Linear] = (
            nn.Linear(context_dim, model_dim) if context_dim is not None else None
        )
        dummy_tokens = torch.empty(dummy_token_count, model_dim)
        if dummy_token_count:
            _ = nn.init.normal_(dummy_tokens, mean=0.0, std=0.02)
        if dummy_trainable:
            self.dummy_tokens: torch.Tensor = nn.Parameter(dummy_tokens)
        else:
            self.register_buffer("dummy_tokens", dummy_tokens)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder: nn.TransformerEncoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.output: nn.Linear = nn.Linear(model_dim, out_dim)
        _ = nn.init.zeros_(self.output.weight)
        _ = nn.init.zeros_(self.output.bias)

    @override
    def forward(self, x: torch.Tensor, context_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        patch_tokens = self.input.forward(x)
        prefix_tokens = []
        if context_tokens is not None:
            if self.context_input is None:
                raise RuntimeError("Transformer flow received context tokens without context_input")
            if context_tokens.shape[0] != x.shape[0]:
                raise RuntimeError("Transformer flow context batch size does not match patch batch")
            prefix_tokens.append(self.context_input.forward(context_tokens))
        elif self.context_input is not None:
            raise RuntimeError("Transformer flow context tokens are required for this estimator")
        if self.dummy_tokens.shape[0]:
            prefix_tokens.append(self.dummy_tokens.unsqueeze(0).expand(x.shape[0], -1, -1))
        if prefix_tokens:
            encoded_input = torch.cat([*prefix_tokens, patch_tokens], dim=1)
            encoded = self.encoder.forward(encoded_input)
            patch_encoded = encoded[:, -x.shape[1] :]
        else:
            patch_encoded = self.encoder.forward(patch_tokens)
        return self.output.forward(patch_encoded)


@final
class TransformerAffineCouplingBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        model_dim: int,
        clamp: float,
        flip: bool,
        context_dim: Optional[int] = None,
        dummy_token_count: int = 0,
        dummy_trainable: bool = False,
    ) -> None:
        super().__init__()
        if dim < 2:
            raise RuntimeError("Transformer flow requires at least two channels")
        self.clamp: float = clamp
        self.flip: bool = flip
        self.left_dim: int = dim // 2
        self.right_dim: int = dim - self.left_dim
        input_dim = self.right_dim if flip else self.left_dim
        active_dim = self.left_dim if flip else self.right_dim
        self.net: TransformerCouplingNet = TransformerCouplingNet(
            input_dim,
            active_dim * 2,
            model_dim,
            context_dim=context_dim,
            dummy_token_count=dummy_token_count,
            dummy_trainable=dummy_trainable,
        )

    @override
    def forward(
        self,
        x: torch.Tensor,
        context_tokens: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        left, right = torch.split(x, [self.left_dim, self.right_dim], dim=-1)
        identity = right if self.flip else left
        active = left if self.flip else right
        raw_scale, raw_shift = self.net.forward(identity, context_tokens=context_tokens).chunk(2, dim=-1)
        scale = self.clamp * torch.tanh(raw_scale)
        if reverse:
            transformed = (active - raw_shift) * torch.exp(-scale)
            logdet = -scale.sum(dim=-1)
        else:
            transformed = active * torch.exp(scale) + raw_shift
            logdet = scale.sum(dim=-1)
        if self.flip:
            return torch.cat([transformed, identity], dim=-1), logdet
        return torch.cat([identity, transformed], dim=-1), logdet


@final
class TransformerNormalizingFlow(nn.Module):
    def __init__(
        self,
        dim: int,
        n_coupling_layers: int,
        model_dim: int,
        clamp: float,
        context_dim: Optional[int] = None,
        dummy_token_count: int = 0,
        dummy_trainable: bool = False,
    ) -> None:
        super().__init__()
        blocks = [
            TransformerAffineCouplingBlock(
                dim=dim,
                model_dim=model_dim,
                clamp=clamp,
                flip=idx % 2 == 1,
                context_dim=context_dim,
                dummy_token_count=dummy_token_count,
                dummy_trainable=dummy_trainable,
            )
            for idx in range(n_coupling_layers)
        ]
        self.blocks: Tuple[TransformerAffineCouplingBlock, ...] = tuple(blocks)
        self.registered_blocks: nn.ModuleList = nn.ModuleList(blocks)

    @override
    def forward(
        self,
        x: torch.Tensor,
        context_tokens: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
        blocks = reversed(self.blocks) if reverse else self.blocks
        out = x
        for block in blocks:
            out, block_logdet = block.forward(out, context_tokens=context_tokens, reverse=reverse)
            logdet = logdet + block_logdet
        return out, logdet


@final
class TransformerFlowDensityEstimator:
    def __init__(
        self,
        dim: int,
        config: FlowConfig,
        device: str,
        context_dim: Optional[int] = None,
        dummy_token_count: int = 0,
        dummy_trainable: bool = False,
    ) -> None:
        self.config: FlowConfig = config
        self.device: torch.device = resolve_device(device)
        _seed_torch(config.seed, self.device)
        model_dim = _model_dim(dim)
        self.flow: TransformerNormalizingFlow = TransformerNormalizingFlow(
            dim=dim,
            n_coupling_layers=config.n_coupling_layers,
            model_dim=model_dim,
            clamp=config.clamp,
            context_dim=context_dim,
            dummy_token_count=dummy_token_count,
            dummy_trainable=dummy_trainable,
        ).to(self.device)
        self.standardizer: FeatureStandardizer | None = None
        self.train_nll_mean: float = 0.0
        self.train_nll_std: float = 1.0
        self.density_threshold: float = 0.0

    def fit(
        self,
        feature_maps: Sequence[np.ndarray],
        density_quantile: float,
        context_tokens: Optional[Sequence[np.ndarray]] = None,
    ) -> FlowTrainingStats:
        x = self._maps_to_tensor(feature_maps)
        context_x = self._contexts_to_tensor(context_tokens, expected_count=x.shape[0])
        if self.config.standardize:
            flat = x.reshape(-1, x.shape[-1])
            self.standardizer = FeatureStandardizer.fit(flat)
            x = self.standardizer.transform(flat).reshape_as(x)
            context_x = self._standardize_contexts(context_x)
        if self.config.transform_mode == "identity":
            train_nll = self._identity_nll(x)
            return self._store_stats([float(train_nll.mean().detach().cpu())], train_nll, density_quantile)

        image_batch_size = max(1, self.config.batch_size // x.shape[1])
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
                batch_indices = order[start : start + image_batch_size]
                image_batch = x[batch_indices]
                context_batch = context_x[batch_indices] if context_x is not None else None
                z, logdet = self.flow.forward(image_batch, context_tokens=context_batch)
                nll = tail_aware_nll(
                    z,
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
        train_nll = self.evaluate_many(feature_maps, context_tokens=context_tokens).nll
        return self._store_stats(losses, train_nll, density_quantile)

    def evaluate_many(
        self,
        feature_maps: Sequence[np.ndarray],
        context_tokens: Optional[Sequence[np.ndarray]] = None,
    ) -> TransformerFlowEvaluation:
        x = self._maps_to_tensor(feature_maps)
        context_x = self._contexts_to_tensor(context_tokens, expected_count=x.shape[0])
        if self.standardizer is not None:
            x = self.standardizer.transform(x.reshape(-1, x.shape[-1])).reshape_as(x)
            context_x = self._standardize_contexts(context_x)
        if self.config.transform_mode == "identity":
            z = x
            nll = self._identity_nll(x)
        else:
            _ = self.flow.eval()
            with torch.no_grad():
                z, logdet = self.flow.forward(x, context_tokens=context_x)
                nll = patch_nll(z, logdet)
        height, width = int(feature_maps[0].shape[0]), int(feature_maps[0].shape[1])
        return TransformerFlowEvaluation(
            z=z.reshape(-1, z.shape[-1]),
            nll=nll.reshape(-1),
            spatial_shape=(height, width),
        )

    def evaluate(
        self,
        feature_map: np.ndarray,
        context_tokens: Optional[np.ndarray] = None,
    ) -> TransformerFlowEvaluation:
        contexts = (context_tokens,) if context_tokens is not None else None
        return self.evaluate_many((feature_map,), context_tokens=contexts)

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
        return FlowTrainingStats(losses, self.train_nll_mean, self.train_nll_std, self.density_threshold)

    def _maps_to_tensor(self, feature_maps: Sequence[np.ndarray]) -> torch.Tensor:
        if not feature_maps:
            raise RuntimeError("Transformer flow requires at least one feature map")
        array = np.stack(feature_maps, axis=0).astype(np.float32, copy=False)
        if array.ndim != 4:
            raise RuntimeError("Transformer flow expects feature maps shaped HxWxC")
        return torch.as_tensor(array, dtype=torch.float32, device=self.device).reshape(
            array.shape[0],
            array.shape[1] * array.shape[2],
            array.shape[3],
        )

    def _contexts_to_tensor(
        self,
        context_tokens: Optional[Sequence[np.ndarray]],
        expected_count: int,
    ) -> Optional[torch.Tensor]:
        if context_tokens is None:
            return None
        array = np.stack(tuple(context_tokens), axis=0).astype(np.float32, copy=False)
        if array.ndim != 3:
            raise RuntimeError("Transformer context tokens must be shaped BxTxC")
        if array.shape[0] != expected_count:
            raise RuntimeError("Transformer context token count does not match feature map count")
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def _standardize_contexts(self, context_x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if context_x is None or self.standardizer is None:
            return context_x
        if context_x.shape[-1] != self.standardizer.mean.shape[-1]:
            raise RuntimeError("Transformer context token dimension must match patch feature dimension")
        return self.standardizer.transform(context_x.reshape(-1, context_x.shape[-1])).reshape_as(context_x)

    @staticmethod
    def _identity_nll(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x.square().sum(dim=-1)


def _model_dim(dim: int) -> int:
    return min(256, max(32, dim // 8))


def _head_count(model_dim: int) -> int:
    for candidate in (8, 4, 2):
        if model_dim % candidate == 0:
            return candidate
    return 1
