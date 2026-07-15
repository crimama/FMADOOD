from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from typing_extensions import final

from flow_tte.config import FlowConfig
from flow_tte.flow import PatchNormalizingFlow
from flow_tte.losses import patch_nll, tail_aware_nll
from flow_tte.tensors import (
    FeatureArray,
    PatchBatch,
    as_2d_float_tensor,
    as_patch_batch,
    resolve_device,
)


@dataclass(frozen=True)
class FeatureStandardizer:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, features: torch.Tensor) -> "FeatureStandardizer":
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        return cls(mean=mean, std=std)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.mean) / self.std


@dataclass(frozen=True)
class FlowTrainingStats:
    losses: List[float]
    train_nll_mean: float
    train_nll_std: float
    density_threshold: float


@dataclass(frozen=True)
class FlowEvaluation:
    batch: PatchBatch
    z: torch.Tensor
    nll: torch.Tensor

    @property
    def spatial_shape(self) -> Tuple[int, ...]:
        return self.batch.patch_shape


def _seed_torch(seed: int, device: torch.device) -> None:
    _ = torch.random.manual_seed(seed)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@final
class FlowDensityEstimator:
    def __init__(
        self,
        dim: int,
        config: FlowConfig,
        device: str,
        condition_dim: int = 0,
    ) -> None:
        self.config: FlowConfig = config
        self.device: torch.device = resolve_device(device)
        _seed_torch(config.seed, self.device)
        self.condition_dim: int = condition_dim if config.condition_mode == "context" else 0
        self.flow: PatchNormalizingFlow = PatchNormalizingFlow(
            dim=dim,
            n_coupling_layers=config.n_coupling_layers,
            hidden_multiplier=config.hidden_multiplier,
            clamp=config.clamp,
            condition_dim=self.condition_dim,
            spatial_context=config.spatial_context,
        ).to(self.device)
        self.standardizer: Optional[FeatureStandardizer] = None
        self.context_standardizer: Optional[FeatureStandardizer] = None
        self.train_nll_mean: float = 0.0
        self.train_nll_std: float = 1.0
        self.density_threshold: float = 0.0
        self.spatial_shape: Optional[Tuple[int, int]] = None

    def fit(
        self,
        features: FeatureArray,
        density_quantile: float,
        contexts: Optional[FeatureArray] = None,
    ) -> FlowTrainingStats:
        patch_batch = as_patch_batch(features, self.device)
        self.spatial_shape = (
            (patch_batch.patch_shape[0], patch_batch.patch_shape[1])
            if len(patch_batch.patch_shape) == 2
            else None
        )
        x = patch_batch.flat_features
        if self.config.standardize:
            self.standardizer = FeatureStandardizer.fit(x)
            x = self.standardizer.transform(x)
        condition = self._prepare_condition(
            contexts=contexts,
            n_rows=int(x.shape[0]),
            fit=True,
        )
        if self.config.transform_mode == "identity":
            train_nll = self._identity_nll(x)
            self.train_nll_mean = float(train_nll.mean().detach().cpu())
            self.train_nll_std = float(
                train_nll.std(unbiased=False).clamp_min(1e-6).detach().cpu(),
            )
            self.density_threshold = float(
                torch.quantile(train_nll, density_quantile).detach().cpu(),
            )
            return FlowTrainingStats(
                losses=[self.train_nll_mean],
                train_nll_mean=self.train_nll_mean,
                train_nll_std=self.train_nll_std,
                density_threshold=self.density_threshold,
            )

        image_features = x.reshape(patch_batch.n_images, patch_batch.patches_per_image, x.shape[1])
        image_conditions = None
        if condition is not None:
            image_conditions = condition.reshape(
                patch_batch.n_images,
                patch_batch.patches_per_image,
                condition.shape[1],
            )
        image_batch_size = max(1, self.config.batch_size // patch_batch.patches_per_image)
        loader_generator = torch.Generator(device=self.device)
        _ = loader_generator.manual_seed(self.config.seed)
        optimizer = torch.optim.AdamW(self.flow.parameters(), lr=self.config.lr)
        losses: List[float] = []
        _ = self.flow.train()
        for _epoch in range(self.config.n_epochs):
            if patch_batch.flat_input and x.shape[0] > self.config.batch_size:
                losses.append(self._fit_flat_batch_microbatched(x, condition, optimizer))
                continue
            epoch_loss = 0.0
            n_batches = 0
            order = torch.randperm(
                patch_batch.n_images,
                generator=loader_generator,
                device=self.device,
            )
            for start in range(0, patch_batch.n_images, image_batch_size):
                batch_indices = order[start : start + image_batch_size]
                image_batch = image_features[batch_indices]
                flat_batch = image_batch.reshape(-1, image_batch.shape[-1])
                flat_condition = None
                if image_conditions is not None:
                    flat_condition = image_conditions[batch_indices].reshape(
                        -1,
                        image_conditions.shape[-1],
                    )
                z_flat, logdet_flat = self._forward_flow(flat_batch, flat_condition)
                z = z_flat.reshape(image_batch.shape[0], image_batch.shape[1], z_flat.shape[-1])
                logdet = logdet_flat.reshape(image_batch.shape[0], image_batch.shape[1])
                nll = tail_aware_nll(
                    z,
                    logdet,
                    tail_weight=self.config.tail_weight,
                    tail_top_k_ratio=self.config.tail_top_k_ratio,
                )
                logdet_reg = logdet.square().mean()
                loss = nll + self.config.lambda_logdet * logdet_reg
                optimizer.zero_grad()
                loss.backward()
                _ = clip_grad_norm_(self.flow.parameters(), max_norm=0.5)
                optimizer.step()
                epoch_loss += float(nll.detach().cpu())
                n_batches += 1
            losses.append(epoch_loss / max(n_batches, 1))

        # Training statistics only need the scalar NLL per patch.  Re-running
        # evaluate() here also materializes and concatenates every latent, which
        # unnecessarily doubles the peak memory for full-normal support sets.
        train_nll = self._nll_in_chunks(x, condition)
        self.train_nll_mean = float(train_nll.mean().detach().cpu())
        self.train_nll_std = float(train_nll.std(unbiased=False).clamp_min(1e-6).detach().cpu())
        self.density_threshold = float(
            torch.quantile(train_nll, density_quantile).detach().cpu(),
        )
        return FlowTrainingStats(
            losses=losses,
            train_nll_mean=self.train_nll_mean,
            train_nll_std=self.train_nll_std,
            density_threshold=self.density_threshold,
        )

    def evaluate(
        self,
        features: FeatureArray,
        contexts: Optional[FeatureArray] = None,
    ) -> FlowEvaluation:
        batch = as_patch_batch(features, self.device)
        if len(batch.patch_shape) == 2:
            self.spatial_shape = (batch.patch_shape[0], batch.patch_shape[1])
        x = batch.flat_features
        if self.standardizer is not None:
            x = self.standardizer.transform(x)
        condition = self._prepare_condition(
            contexts=contexts,
            n_rows=int(x.shape[0]),
            fit=False,
        )
        if self.config.transform_mode == "identity":
            return FlowEvaluation(batch=batch, z=x, nll=self._identity_nll(x))
        _ = self.flow.eval()
        with torch.no_grad():
            z, logdet = self._forward_in_chunks(x, condition)
            nll = patch_nll(z, logdet)
        return FlowEvaluation(batch=batch, z=z, nll=nll)

    def evaluate_many(self, features: FeatureArray) -> FlowEvaluation:
        return self.evaluate(features)

    def _fit_flat_batch_microbatched(
        self,
        features: torch.Tensor,
        condition: Optional[torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> float:
        n_rows = int(features.shape[0])
        nll_parts: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, n_rows, self.config.batch_size):
                end = start + self.config.batch_size
                chunk_condition = None if condition is None else condition[start:end]
                z, logdet = self._forward_flow(features[start:end], chunk_condition)
                nll_parts.append(patch_nll(z, logdet))
        nll = torch.cat(nll_parts, dim=0)
        tail_count = max(1, int(n_rows * self.config.tail_top_k_ratio))
        tail_indices = torch.topk(nll, tail_count).indices
        nll_weights = torch.full_like(nll, (1.0 - self.config.tail_weight) / n_rows)
        nll_weights[tail_indices] += self.config.tail_weight / tail_count

        optimizer.zero_grad()
        for start in range(0, n_rows, self.config.batch_size):
            end = start + self.config.batch_size
            chunk_condition = None if condition is None else condition[start:end]
            z, logdet = self._forward_flow(features[start:end], chunk_condition)
            chunk_nll = patch_nll(z, logdet)
            loss = (chunk_nll * nll_weights[start:end]).sum()
            loss = loss + self.config.lambda_logdet * logdet.square().sum() / n_rows
            loss.backward()
        _ = clip_grad_norm_(self.flow.parameters(), max_norm=0.5)
        optimizer.step()
        return float((nll * nll_weights).sum().detach().cpu())

    def _forward_in_chunks(
        self,
        features: torch.Tensor,
        condition: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.spatial_context:
            return self._forward_flow(features, condition)
        if features.shape[0] <= self.config.batch_size:
            return self._forward_flow(features, condition)
        z_parts: List[torch.Tensor] = []
        logdet_parts: List[torch.Tensor] = []
        for start in range(0, features.shape[0], self.config.batch_size):
            end = start + self.config.batch_size
            chunk_condition = None if condition is None else condition[start:end]
            z, logdet = self._forward_flow(features[start:end], chunk_condition)
            z_parts.append(z)
            logdet_parts.append(logdet)
        return torch.cat(z_parts, dim=0), torch.cat(logdet_parts, dim=0)

    def _nll_in_chunks(
        self,
        features: torch.Tensor,
        condition: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute exact patch NLLs without retaining the full latent matrix."""
        nll_parts: List[torch.Tensor] = []
        _ = self.flow.eval()
        with torch.no_grad():
            if self.config.spatial_context:
                z, logdet = self._forward_flow(features, condition)
                return patch_nll(z, logdet)
            for start in range(0, features.shape[0], self.config.batch_size):
                end = start + self.config.batch_size
                chunk_condition = None if condition is None else condition[start:end]
                z, logdet = self._forward_flow(features[start:end], chunk_condition)
                nll_parts.append(patch_nll(z, logdet))
        return torch.cat(nll_parts, dim=0)

    def _forward_flow(
        self,
        features: torch.Tensor,
        condition: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.spatial_context:
            return self.flow.forward(
                features,
                condition=condition,
                spatial_shape=self.spatial_shape,
            )
        return self.flow.forward(features, condition=condition)

    def transform(
        self,
        features: FeatureArray,
        contexts: Optional[FeatureArray] = None,
    ) -> torch.Tensor:
        return self.evaluate(features, contexts=contexts).z

    def nll(self, features: FeatureArray, contexts: Optional[FeatureArray] = None) -> torch.Tensor:
        return self.evaluate(features, contexts=contexts).nll

    def density_penalty(self, nll: torch.Tensor) -> torch.Tensor:
        centered = (nll - self.train_nll_mean) / self.train_nll_std
        return torch.relu(centered)

    def _prepare_condition(
        self,
        contexts: Optional[FeatureArray],
        n_rows: int,
        fit: bool,
    ) -> Optional[torch.Tensor]:
        if self.condition_dim == 0:
            return None
        if contexts is None:
            raise RuntimeError("Conditional NF requires condition context")
        condition = as_2d_float_tensor(contexts, self.device)
        if condition.shape[0] != n_rows:
            raise RuntimeError("condition context row count must match patch features")
        if condition.shape[1] != self.condition_dim:
            raise RuntimeError("condition context dimension mismatch")
        if fit:
            self.context_standardizer = FeatureStandardizer.fit(condition)
        if self.context_standardizer is None:
            raise RuntimeError("Conditional NF context standardizer is not fitted")
        return self.context_standardizer.transform(condition)

    @staticmethod
    def _identity_nll(features: torch.Tensor) -> torch.Tensor:
        return 0.5 * features.square().sum(dim=1)
