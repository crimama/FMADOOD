"""Official SubspaceAD few-shot PCA adapter for the shared AD 2 pipeline.

The feature settings and scoring follow CLendering/SubspaceAD at the pinned
CVPR 2026 release. Support selection and evaluation remain owned by the shared
pipeline; fitting only receives its manifest-selected normal support paths.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Iterator, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from fmad.methods.base import BaseMethod
from fmad.protocol import MethodContext
from fmad.registry import METHOD_REGISTRY


_OFFICIAL_REPOSITORY = "https://github.com/CLendering/SubspaceAD"
_OFFICIAL_COMMIT = "ef56d5c8ab2f1feb7dda1c93b25cc3f73f0960d7"
_OFFICIAL_LICENSE = "Apache-2.0"
_LAYERS = tuple(range(22, 29))
_FEATURE_DIM = 1536
_PATCH_SIZE = 14


def _streaming_pca(
    feature_generator: Callable[[], Iterator[np.ndarray]],
    *,
    feature_dim: int,
    total_tokens: int,
    num_batches: int,
    device: torch.device,
    explained_variance: float,
    eps: float,
) -> dict:
    """Two-pass float64 PCA matching the official SubspaceAD implementation."""
    if feature_dim <= 0 or total_tokens <= 1 or num_batches <= 0:
        raise ValueError("SubspaceAD PCA requires non-empty support feature batches")

    dtype = torch.float64
    mean = torch.zeros(feature_dim, dtype=dtype, device=device)
    observed_batches = 0
    for batch in feature_generator():
        tensor = torch.from_numpy(np.asarray(batch)).to(device, dtype=dtype)
        if tensor.ndim != 2 or tensor.shape[1] != feature_dim:
            raise ValueError("SubspaceAD feature batch has an unexpected shape")
        mean += tensor.sum(dim=0)
        observed_batches += 1
    if observed_batches != num_batches:
        raise ValueError(
            f"SubspaceAD mean pass produced {observed_batches} batches; expected {num_batches}"
        )
    mean /= total_tokens

    covariance = torch.zeros((feature_dim, feature_dim), dtype=dtype, device=device)
    observed_batches = 0
    for batch in feature_generator():
        tensor = torch.from_numpy(np.asarray(batch)).to(device, dtype=dtype)
        centered = tensor - mean
        covariance += centered.T @ centered
        observed_batches += 1
    if observed_batches != num_batches:
        raise ValueError(
            "SubspaceAD covariance pass produced "
            f"{observed_batches} batches; expected {num_batches}"
        )
    covariance /= total_tokens - 1

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    variance_sum = eigenvalues.sum()
    if not torch.isfinite(variance_sum) or variance_sum <= 0:
        raise ValueError("SubspaceAD PCA found no positive finite support variance")
    cumulative = torch.cumsum(eigenvalues, dim=0) / variance_sum
    target = torch.tensor(explained_variance, dtype=dtype, device=device)
    component_count = int(torch.searchsorted(cumulative, target).item()) + 1
    component_count = min(component_count, eigenvectors.shape[1])

    mean_np = mean.cpu().numpy().astype(np.float64)
    components_np = eigenvectors[:, :component_count].cpu().numpy().astype(np.float64)
    eigenvalues_np = eigenvalues[:component_count].cpu().numpy().astype(np.float64)
    retained_variance = float(cumulative[component_count - 1].item())
    del covariance, eigenvectors, eigenvalues, order, cumulative, variance_sum
    return {
        "mu": mean_np,
        "components": components_np,
        "eigvals": eigenvalues_np,
        "k": component_count,
        "eps": eps,
        "explained_variance": retained_variance,
    }


@METHOD_REGISTRY.register("subspacead")
class SubspaceAD(BaseMethod):
    """Few-shot SubspaceAD with support-only fitting and v13-native maps."""

    def __init__(self, backbone, config: dict):
        super().__init__(backbone, config)
        if getattr(backbone, "model_name", None) != "dinov2_vitg14_hf":
            raise ValueError("SubspaceAD requires the pinned HF DINOv2-G/14 backbone")
        if tuple(getattr(backbone, "feature_layers", ())) != _LAYERS:
            raise ValueError("SubspaceAD requires DINOv2-G/14 blocks 22 through 28")

        self.pca_ev = float(config.get("pca_ev", 0.99))
        self.aug_count = int(config.get("aug_count", 30))
        self.rotation_degrees = tuple(config.get("rotation_degrees", (0, 345)))
        self.drop_k = int(config.get("drop_k", 0))
        self.eps = float(config.get("eps", 1e-6))
        self.smoothing_sigma = float(config.get("smoothing_sigma", 4.0))
        self._rotation = transforms.RandomRotation(degrees=self.rotation_degrees)
        self._pca: dict | None = None
        self._resolution: int | None = None
        self._support_count = 0
        self._support_views = 0
        self._total_tokens = 0
        self._feature_dim = _FEATURE_DIM
        self._pca_components = 0
        self._retained_variance = 0.0
        self._current_object: str | None = None

        if not 0 < self.pca_ev <= 1:
            raise ValueError("method_config.pca_ev must be in (0, 1]")
        if self.aug_count < 0:
            raise ValueError("method_config.aug_count must be non-negative")
        if len(self.rotation_degrees) != 2 or self.rotation_degrees != (0, 345):
            raise ValueError("SubspaceAD uses the official rotation range [0, 345]")
        if self.drop_k < 0:
            raise ValueError("method_config.drop_k must be non-negative")
        if self.eps <= 0 or self.smoothing_sigma <= 0:
            raise ValueError("SubspaceAD eps and smoothing sigma must be positive")
        if config.get("score_method", "reconstruction") != "reconstruction":
            raise ValueError("This SubspaceAD comparison uses PCA reconstruction scoring")

    def _extract_image_features(
        self, image: Image.Image, resolution: int
    ) -> tuple[np.ndarray, tuple[int, int]]:
        image_tensor, grid_size = self.backbone.prepare_image(image, resolution=resolution)
        layer_features = self.backbone.extract_features(image_tensor)
        if len(layer_features) != len(_LAYERS):
            raise ValueError("HF DINOv2-G/14 returned an unexpected number of layers")
        expected_tokens = grid_size[0] * grid_size[1]
        features = []
        for values in layer_features:
            layer = np.asarray(values, dtype=np.float32)
            if layer.shape != (expected_tokens, _FEATURE_DIM):
                raise ValueError(
                    f"SubspaceAD expected {(expected_tokens, _FEATURE_DIM)} features, "
                    f"found {layer.shape}"
                )
            if not np.isfinite(layer).all():
                raise ValueError("SubspaceAD received non-finite DINOv2-G/14 features")
            features.append(layer)
        return np.mean(np.stack(features, axis=0), axis=0), grid_size

    def _feature_generator(self, support_paths: Sequence[str]) -> Iterator[np.ndarray]:
        if self._resolution is None:
            raise RuntimeError("SubspaceAD fit() must set a resolution before feature extraction")
        for path in support_paths:
            with Image.open(path) as opened:
                support_image = opened.convert("RGB")
            original_features, _ = self._extract_image_features(
                support_image, self._resolution
            )
            yield original_features
            for _ in range(self.aug_count):
                augmented = self._rotation(support_image)
                augmented_features, _ = self._extract_image_features(
                    augmented, self._resolution
                )
                yield augmented_features

    def fit(self, support_paths: list[str], context: MethodContext) -> None:
        if not support_paths:
            raise ValueError("SubspaceAD requires manifest-selected normal support images")
        if context.masking or context.rotation:
            raise ValueError("SubspaceAD v13 uses no dataset masking or common-pipeline rotation")
        if context.resolution % _PATCH_SIZE:
            raise ValueError("SubspaceAD resolution must be divisible by DINOv2-G patch size 14")

        self._resolution = context.resolution
        self._current_object = context.object_name
        grid_edge = context.resolution // _PATCH_SIZE
        tokens_per_view = grid_edge * grid_edge
        self._support_count = len(support_paths)
        self._support_views = self._support_count * (1 + self.aug_count)
        self._total_tokens = self._support_views * tokens_per_view
        batch_count = self._support_views

        self._pca = _streaming_pca(
            lambda: self._feature_generator(support_paths),
            feature_dim=_FEATURE_DIM,
            total_tokens=self._total_tokens,
            num_batches=batch_count,
            device=torch.device(self.backbone.device),
            explained_variance=self.pca_ev,
            eps=self.eps,
        )
        self._pca_components = int(self._pca["k"])
        self._retained_variance = float(self._pca["explained_variance"])
        self.log_training(
            stage="subspacead_pca_ready",
            object=context.object_name,
            support_images=self._support_count,
            support_views=self._support_views,
            support_tokens=self._total_tokens,
            feature_dim=_FEATURE_DIM,
            pca_components=self._pca_components,
            retained_variance=self._retained_variance,
        )

    def _reconstruction_scores(self, features: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("SubspaceAD fit() must precede prediction")
        x = np.asarray(features, dtype=np.float32)
        mean = np.asarray(self._pca["mu"], dtype=x.dtype)
        components = np.asarray(self._pca["components"], dtype=x.dtype)
        centered = x - mean
        coordinates = centered @ components
        if self.drop_k:
            coordinates[:, : min(self.drop_k, coordinates.shape[1])] = 0.0
        reconstructed = coordinates @ components.T + mean
        return np.sum((x - reconstructed) ** 2, axis=1, dtype=np.float32)

    def predict(self, test_image_path: str) -> tuple[np.ndarray, float]:
        if self._pca is None or self._resolution is None:
            raise RuntimeError("SubspaceAD fit() must precede prediction")
        with Image.open(test_image_path) as opened:
            image = opened.convert("RGB")
        original_shape = (image.height, image.width)
        features, grid_size = self._extract_image_features(image, self._resolution)
        patch_scores = self._reconstruction_scores(features).reshape(grid_size)
        score_map = cv2.resize(
            patch_scores,
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        score_map = cv2.GaussianBlur(
            score_map.astype(np.float32, copy=False),
            (3, 3),
            sigmaX=self.smoothing_sigma,
        )
        return score_map, float(np.max(score_map))

    def object_metadata(self) -> dict:
        return {
            "method": "subspacead",
            "official_repository": _OFFICIAL_REPOSITORY,
            "official_commit": _OFFICIAL_COMMIT,
            "official_license": _OFFICIAL_LICENSE,
            "support_count": self._support_count,
            "support_views": self._support_views,
            "support_tokens": self._total_tokens,
            "resolution": self._resolution,
            "feature_dim": self._feature_dim,
            "feature_layers": list(_LAYERS),
            "augmentation": {
                "name": "RandomRotation",
                "count_per_support": self.aug_count,
                "degrees": list(self.rotation_degrees),
            },
            "pca": {
                "explained_variance_target": self.pca_ev,
                "components": self._pca_components,
                "retained_variance": self._retained_variance,
                "fit_dtype": "float64",
            },
            "score_method": "pca_reconstruction_residual",
            "map_resize": "bilinear_to_native_image_size",
            "map_smoothing_sigma": self.smoothing_sigma,
            "image_score": "score_map_max_shared_contract",
        }
