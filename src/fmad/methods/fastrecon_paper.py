"""Paper-faithful FastRecon adapter for the public v13 comparison.

This implementation follows the ICCV 2023 method settings (ImageNet ResNet-50,
368x368 input, 15% greedy coreset, and lambda=0.3).  The support-only regression
is evaluated as a row-space projection so it remains defined when the coreset
contains more vectors than the feature dimension.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.random_projection import SparseRandomProjection
from torchvision import models, transforms

from fmad.backbones.base import BaseBackbone
from fmad.backbones.provenance import backbone_state_sha256
from fmad.methods.base import BaseMethod
from fmad.protocol import MethodContext
from fmad.registry import BACKBONE_REGISTRY, METHOD_REGISTRY


_INPUT_SIZE = 368
_FEATURE_GRID = 46
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _embedding_concat(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Concatenate layer2 and layer3 features using PatchCore's spatial tiling."""
    batch, channels_x, height_x, width_x = x.shape
    batch_y, channels_y, height_y, width_y = y.shape
    if batch != batch_y or height_x % height_y or width_x % width_y:
        raise ValueError("ResNet feature maps have incompatible spatial dimensions")
    scale_h, scale_w = height_x // height_y, width_x // width_y
    if scale_h != scale_w:
        raise ValueError("FastRecon expects an isotropic layer2/layer3 scale ratio")
    scale = scale_h

    patches = F.unfold(x, kernel_size=scale, stride=scale)
    patches = patches.reshape(batch, channels_x, scale * scale, height_y, width_y)
    merged = torch.empty(
        batch,
        channels_x + channels_y,
        scale * scale,
        height_y,
        width_y,
        device=x.device,
        dtype=x.dtype,
    )
    for offset in range(scale * scale):
        merged[:, :, offset] = torch.cat((patches[:, :, offset], y), dim=1)
    merged = merged.reshape(batch, (channels_x + channels_y) * scale * scale, -1)
    return F.fold(
        merged,
        kernel_size=scale,
        output_size=(height_x, width_x),
        stride=scale,
    )


@BACKBONE_REGISTRY.register("fastrecon_resnet50")
class FastReconResNet50(BaseBackbone):
    """ImageNet ResNet-50 feature extractor used by the FastRecon paper."""

    def __init__(
        self,
        model_name: str,
        device: str,
        smaller_edge_size: int = _INPUT_SIZE,
        weights: str = "IMAGENET1K_V1",
        resolution_ablation: bool = False,
    ) -> None:
        if smaller_edge_size != _INPUT_SIZE and not resolution_ablation:
            raise ValueError("FastRecon paper settings require a fixed 368x368 input")
        if smaller_edge_size <= 0 or smaller_edge_size % 16:
            raise ValueError("FastRecon input must be a positive multiple of 16")
        if weights != "IMAGENET1K_V1":
            raise ValueError("FastRecon paper settings require ResNet50 IMAGENET1K_V1 weights")
        super().__init__(model_name, device, smaller_edge_size)
        self.device = str(device)
        self.paper_input_size = int(smaller_edge_size)
        self.resolution_ablation = bool(resolution_ablation)
        self.pipeline_resolution_hint: int | None = None
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.paper_input_size, self.paper_input_size),
                    interpolation=transforms.InterpolationMode.LANCZOS,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.to(self.device)
        self.state_sha256 = backbone_state_sha256(self.model)

    def set_resolution(self, smaller_edge_size: int) -> None:
        """Record the shared dataset resolution while retaining the paper's 368 input."""
        self.pipeline_resolution_hint = int(smaller_edge_size)

    def prepare_image(self, img, *, resolution: int | None = None):
        if resolution is not None and resolution != self.paper_input_size:
            raise ValueError("FastRecon preprocessing must match its configured input size")
        if isinstance(img, (str, bytes, Path)):
            with Image.open(img) as opened:
                image = opened.convert("RGB")
        elif isinstance(img, Image.Image):
            image = img.convert("RGB")
        elif isinstance(img, np.ndarray):
            image = Image.fromarray(np.asarray(img, dtype=np.uint8)).convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(img).__name__}")
        tensor = self.transform(image)
        grid = self.paper_input_size // 8
        return tensor, (grid, grid)

    def _patch_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if image_tensor.ndim != 4 or tuple(image_tensor.shape[-2:]) != (
            self.paper_input_size,
            self.paper_input_size,
        ):
            raise ValueError("FastRecon expects CHW/BCHW images at its configured input size")
        image_tensor = image_tensor.to(self.device)
        with torch.inference_mode():
            model = self.model
            x = model.conv1(image_tensor)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            layer2 = model.layer2(x)
            layer3 = model.layer3(layer2)
            layer2 = F.avg_pool2d(layer2, kernel_size=3, stride=1, padding=1)
            layer3 = F.avg_pool2d(layer3, kernel_size=3, stride=1, padding=1)
            embedding = _embedding_concat(layer2, layer3)
            return embedding[0].permute(1, 2, 0).reshape(-1, embedding.shape[1])

    def extract_features(self, image_tensor) -> list[np.ndarray]:
        return [self._patch_features(image_tensor).cpu().numpy()]

    def extract_cls_features(self, image_tensor) -> torch.Tensor:
        return self._patch_features(image_tensor).mean(dim=0)

    def compute_background_mask(
        self, features, grid_size, threshold: float = 1.0, masking_type=False, **kwargs
    ) -> np.ndarray:
        del features, threshold, masking_type, kwargs
        return np.ones(grid_size[0] * grid_size[1], dtype=bool)

    def warmup(self, image_path: str, iters: int = 25) -> None:
        if iters <= 0:
            return
        image_tensor, _ = self.prepare_image(image_path)
        for _ in range(iters):
            self._patch_features(image_tensor)


def _greedy_coreset(
    features: np.ndarray,
    center_count: int,
    seed: int,
    projection_eps: float,
) -> np.ndarray:
    """Select deterministic greedy k-center indices after sparse random projection."""
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("FastRecon support feature pool must be a non-empty matrix")
    if center_count < 1 or center_count > features.shape[0]:
        raise ValueError("FastRecon coreset size must be within the support feature pool")

    projector = SparseRandomProjection(
        n_components="auto",
        eps=projection_eps,
        dense_output=True,
        random_state=seed,
    )
    projected = np.asarray(projector.fit_transform(features), dtype=np.float32)
    squared_norms = np.einsum("ij,ij->i", projected, projected, dtype=np.float64)
    rng = np.random.default_rng(seed)
    selected = np.empty(center_count, dtype=np.int64)
    selected_mask = np.zeros(features.shape[0], dtype=bool)
    nearest_squared = np.full(features.shape[0], np.inf, dtype=np.float64)
    selected_index = int(rng.integers(features.shape[0]))

    for center_index in range(center_count):
        selected[center_index] = selected_index
        selected_mask[selected_index] = True
        center = projected[selected_index]
        distances = squared_norms + float(np.dot(center, center)) - 2.0 * (
            projected @ center
        )
        np.maximum(distances, 0.0, out=distances)
        np.minimum(nearest_squared, distances, out=nearest_squared)
        nearest_squared[selected_mask] = -1.0
        if center_index + 1 < center_count:
            selected_index = int(np.argmax(nearest_squared))

    return selected


@METHOD_REGISTRY.register("fastrecon_paper")
class FastReconPaper(BaseMethod):
    """FastRecon's feature regression and anomaly scoring from the ICCV paper."""

    def __init__(self, backbone: FastReconResNet50, config: dict):
        if not isinstance(backbone, FastReconResNet50):
            raise TypeError("fastrecon_paper requires backbone: fastrecon_resnet50")
        super().__init__(backbone, config)
        self.device = torch.device(backbone.device)
        self.input_size = int(config.get("input_size", _INPUT_SIZE))
        self.reg_lambda = float(config.get("reg_lambda", 0.3))
        self.coreset_ratio = float(config.get("coreset_sampling_ratio", 0.15))
        self.projection_eps = float(config.get("projection_eps", 0.9))
        self.smooth_sigma = float(config.get("smooth_sigma", 4.0))
        if self.input_size != _INPUT_SIZE and not backbone.resolution_ablation:
            raise ValueError("FastRecon paper settings require input_size=368")
        if self.input_size != backbone.paper_input_size:
            raise ValueError("FastRecon method and backbone input sizes must match")
        if self.reg_lambda != 0.3:
            raise ValueError("FastRecon paper settings require reg_lambda=0.3")
        if self.coreset_ratio != 0.15:
            raise ValueError("FastRecon paper settings require a 0.15 coreset sampling ratio")
        if self.projection_eps <= 0 or self.smooth_sigma < 0:
            raise ValueError("projection_eps must be positive and smooth_sigma non-negative")

        self._selected_support_paths: list[str] = []
        self._coreset: torch.Tensor | None = None
        self._row_basis: torch.Tensor | None = None
        self._support_mean: torch.Tensor | None = None
        self._grid_size: tuple[int, int] | None = None
        self._feature_dim: int | None = None
        self._support_patch_count = 0
        self._coreset_rank = 0
        self._dataset_resolution: int | None = None

    def fit(self, support_paths: list[str], context: MethodContext) -> None:
        if not support_paths:
            raise ValueError("support_paths must not be empty")
        self._selected_support_paths = list(support_paths)
        self._dataset_resolution = context.resolution
        support_maps: list[np.ndarray] = []

        with torch.inference_mode():
            for step, image_path in enumerate(support_paths, start=1):
                image_tensor, grid_size = self.backbone.prepare_image(image_path)
                patch_features = self.backbone.extract_features(image_tensor)[0]
                patch_features = np.asarray(patch_features, dtype=np.float32)
                if patch_features.shape[0] != grid_size[0] * grid_size[1]:
                    raise ValueError("FastRecon feature count does not match its spatial grid")
                if self._grid_size is not None and grid_size != self._grid_size:
                    raise ValueError("FastRecon support feature grids must match")
                self._grid_size = grid_size
                support_maps.append(patch_features)
                self.log_training(
                    step=step,
                    stage="fastrecon_paper_support_features",
                    support_total=len(support_paths),
                )

        stacked = np.stack(support_maps, axis=0)
        self._feature_dim = int(stacked.shape[-1])
        self._support_patch_count = int(stacked.shape[0] * stacked.shape[1])
        self._support_mean = torch.as_tensor(
            stacked.mean(axis=0, dtype=np.float64).astype(np.float32),
            device=self.device,
        )
        support_pool = stacked.reshape(-1, stacked.shape[-1])
        center_count = max(1, int(support_pool.shape[0] * self.coreset_ratio))
        selected = _greedy_coreset(
            support_pool,
            center_count=center_count,
            seed=context.seed,
            projection_eps=self.projection_eps,
        )
        self._coreset = torch.as_tensor(
            support_pool[selected], dtype=torch.float32, device=self.device
        )

        # Eq. (6) has the unique prediction
        #   Qbar = Proj_row(S)((Q + lambda*mu)/(1+lambda)).
        # An SVD basis evaluates this Moore-Penrose least-squares solution
        # without inverting S S^T, which is singular when centers > channels.
        _, singular_values, right_vectors = torch.linalg.svd(
            self._coreset, full_matrices=False
        )
        if singular_values.numel() == 0:
            raise ValueError("FastRecon coreset has no non-zero feature directions")
        tolerance = (
            torch.finfo(singular_values.dtype).eps
            * max(self._coreset.shape)
            * singular_values[0]
        )
        rank = int(torch.count_nonzero(singular_values > tolerance).item())
        if rank == 0:
            raise ValueError("FastRecon coreset has zero numerical rank")
        self._coreset_rank = rank
        self._row_basis = right_vectors[:rank].transpose(0, 1).contiguous()
        self.log_training(
            step=len(support_paths),
            stage="fastrecon_paper_fit_complete",
            support_patch_count=self._support_patch_count,
            coreset_size=center_count,
            coreset_rank=rank,
        )

    def predict(self, test_image_path: str) -> tuple[np.ndarray, float]:
        if (
            self._coreset is None
            or self._row_basis is None
            or self._support_mean is None
            or self._grid_size is None
        ):
            raise RuntimeError("FastReconPaper.fit() must be called before predict()")

        with Image.open(test_image_path) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        image_tensor, grid_size = self.backbone.prepare_image(image)
        if grid_size != self._grid_size:
            raise ValueError("FastRecon query feature grid differs from support grid")
        query_features = torch.as_tensor(
            self.backbone.extract_features(image_tensor)[0],
            dtype=torch.float32,
            device=self.device,
        )
        target = (query_features + self.reg_lambda * self._support_mean) / (
            1.0 + self.reg_lambda
        )
        reconstructed = (target @ self._row_basis) @ self._row_basis.transpose(0, 1)
        patch_scores = (query_features - reconstructed).square().sum(dim=-1)
        score_grid = patch_scores.reshape(grid_size).cpu().numpy().astype(np.float32)

        paper_map = cv2.resize(
            score_grid,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        if self.smooth_sigma > 0:
            paper_map = gaussian_filter(paper_map, sigma=self.smooth_sigma)
        native_map = cv2.resize(
            paper_map,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        native_map = np.asarray(native_map, dtype=np.float32)
        # The public v13 protocol defines I-AUROC from the maximum final map value.
        return native_map, float(np.max(native_map))

    def object_metadata(self) -> dict:
        return {
            "method": "fastrecon_paper",
            "paper": "FastRecon ICCV 2023",
            "backbone": "torchvision_resnet50_imagenet1k_v1",
            "backbone_state_sha256": self.backbone.state_sha256,
            "paper_input_size": self.input_size,
            "paper_default_input_size": _INPUT_SIZE,
            "resolution_ablation": self.backbone.resolution_ablation,
            "reg_lambda": self.reg_lambda,
            "coreset_sampling_ratio": self.coreset_ratio,
            "projection_eps": self.projection_eps,
            "smooth_sigma": self.smooth_sigma,
            "support_count": len(self._selected_support_paths),
            "image_score_rule": "maximum_of_final_native_map_v13",
            "support_patch_count": self._support_patch_count,
            "coreset_size": None if self._coreset is None else int(self._coreset.shape[0]),
            "feature_dim": self._feature_dim,
            "coreset_numerical_rank": self._coreset_rank,
            "solver": "svd_row_space_projection_moore_penrose",
            "rank_deficient_formula_extension": True,
            "pipeline_resolution_hint": self._dataset_resolution,
        }

    def predict_batch(self, test_image_paths: Sequence[str]):
        return super().predict_batch(test_image_paths)
