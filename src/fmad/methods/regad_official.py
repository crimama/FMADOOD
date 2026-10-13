"""Faithful inference bridge for the released official RegAD implementation.

The upstream RegAD test path depends on global CLI state and writes maps at its
224-pixel input size.  This adapter keeps the official STN, test-time support
augmentations, positional Gaussian and Mahalanobis map while fitting only from
the support paths supplied by the shared pipeline.
"""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter

from fmad.methods.base import BaseMethod, Prediction
from fmad.protocol import MethodContext
from fmad.registry import METHOD_REGISTRY


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load upstream RegAD module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resize_to_tensor(path: str, size: int) -> torch.Tensor:
    """Official 224-pixel Lanczos resize and ToTensor, with square AD2 geometry."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        # Official MVTec images are square, so Resize(224) yields this same size.
        # AD2 includes non-square inputs; the fixed square is required by the
        # released STN's hard-coded 224-pixel fully connected geometry.
        image = image.resize((size, size), resample=Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.uint8)
    return torch.from_numpy(array.copy()).permute(2, 0, 1).to(torch.float32).div_(255.0)


def _embedding_concat(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Upstream RegAD's patch-preserving concatenation, using its exact ordering."""
    batch, channels_x, height_x, width_x = x.shape
    _, channels_y, height_y, width_y = y.shape
    if height_x % height_y or width_x % width_y:
        raise ValueError("RegAD feature levels must have integer spatial ratios")
    kernel_h, kernel_w = height_x // height_y, width_x // width_y
    if kernel_h != kernel_w:
        raise ValueError("RegAD feature levels must have square spatial ratios")
    kernel = kernel_h
    unfolded = F.unfold(x, kernel_size=kernel, dilation=1, stride=kernel)
    unfolded = unfolded.view(batch, channels_x, -1, height_y, width_y)
    merged = torch.empty(
        (batch, channels_x + channels_y, kernel * kernel, height_y, width_y),
        dtype=x.dtype,
        device=x.device,
    )
    for offset in range(kernel * kernel):
        merged[:, :, offset] = torch.cat((unfolded[:, :, offset], y), dim=1)
    folded = merged.reshape(batch, -1, height_y * width_y)
    return F.fold(folded, kernel_size=kernel, output_size=(height_x, width_x), stride=kernel)


def _support_augmentations(image: torch.Tensor, kornia) -> torch.Tensor:
    """Generate the official 22-view support stack for one resized image."""
    image = image.unsqueeze(0)
    views = [image]

    def affine(matrix: torch.Tensor) -> torch.Tensor:
        matrix = matrix.unsqueeze(0).repeat(image.shape[0], 1, 1).to(torch.float32)
        grid = F.affine_grid(matrix, image.size())
        return F.grid_sample(image, grid, padding_mode="reflection")

    for angle in (
        -np.pi / 4,
        -3 * np.pi / 16,
        -np.pi / 8,
        -np.pi / 16,
        np.pi / 16,
        np.pi / 8,
        3 * np.pi / 16,
        np.pi / 4,
    ):
        theta = torch.tensor(angle)
        matrix = torch.stack(
            (
                torch.stack((torch.cos(theta), -torch.sin(theta), torch.tensor(0.0))),
                torch.stack((torch.sin(theta), torch.cos(theta), torch.tensor(0.0))),
            )
        )
        views.append(affine(matrix))

    for a, b in (
        (0.2, 0.2),
        (-0.2, 0.2),
        (-0.2, -0.2),
        (0.2, -0.2),
        (0.1, 0.1),
        (-0.1, 0.1),
        (-0.1, -0.1),
        (0.1, -0.1),
    ):
        matrix = torch.tensor([[1.0, 0.0, a], [0.0, 1.0, b]], dtype=torch.float32)
        views.append(affine(matrix))

    views.append(kornia.geometry.transform.hflip(image))
    grey = kornia.color.rgb_to_grayscale(image).repeat(1, 3, 1, 1)
    views.append(grey)
    for degrees in (90.0, 180.0, 270.0):
        views.append(
            kornia.geometry.transform.rotate(
                image, angle=torch.tensor(degrees), padding_mode="reflection"
            )
        )
    return torch.cat(views, dim=0)


@METHOD_REGISTRY.register("regad_official")
class RegADOfficial(BaseMethod):
    """Official RegAD's released 2-shot source model with shared-pipeline support."""

    def __init__(self, backbone, config: dict):
        super().__init__(backbone, config)
        self.source_root = Path(config["source_root"]).expanduser()
        self.dependency_root = Path(config["dependency_root"]).expanduser()
        self.checkpoint_path = Path(config["checkpoint_path"]).expanduser()
        self.expected_commit = str(config["source_commit"])
        self.expected_checkpoint_sha256 = str(config["checkpoint_sha256"]).lower()
        self.source_archive_sha256 = str(config.get("checkpoint_archive_sha256", ""))
        self.stn_mode = str(config.get("stn_mode", "rotation_scale"))
        self.image_size = int(config.get("image_size", 224))
        self.support_batch_size = int(config.get("support_batch_size", 16))
        self.spatial_chunk_size = int(config.get("covariance_spatial_chunk", 64))
        self.covariance_ridge = float(config.get("covariance_ridge", 0.01))
        self.smoothing_sigma = float(config.get("gaussian_sigma", 4.0))
        self.device = torch.device(getattr(backbone, "device", "cuda:0"))

        if self.image_size != 224:
            raise ValueError("the released RegAD registration model requires 224-pixel input")
        if self.support_batch_size <= 0 or self.spatial_chunk_size <= 0:
            raise ValueError("RegAD support and covariance chunk sizes must be positive")
        if self.covariance_ridge != 0.01 or self.smoothing_sigma != 4.0:
            raise ValueError("the official RegAD covariance ridge and smoothing are fixed")
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("regad_official requires the assigned CUDA device")

        self._stn = None
        self._encoder = None
        self._predictor = None
        self._kornia = None
        self._source_revision = None
        self._checkpoint_hash = None
        self._object_name = None
        self._support_paths: list[str] = []
        self._support_mean: torch.Tensor | None = None
        self._inverse_covariance: torch.Tensor | None = None
        self._feature_shape: tuple[int, int, int] | None = None
        self._support_view_count = 0

    def _load_official_models(self) -> None:
        if self._stn is not None:
            return
        if not self.source_root.is_dir() or not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"RegAD source/checkpoint unavailable: {self.source_root} / {self.checkpoint_path}"
            )
        if not self.dependency_root.is_dir():
            raise FileNotFoundError(f"isolated RegAD dependency path is missing: {self.dependency_root}")

        revision = subprocess.run(
            ["git", "-C", str(self.source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision != self.expected_commit:
            raise RuntimeError(
                f"RegAD source revision mismatch: expected {self.expected_commit}, got {revision}"
            )
        checkpoint_hash = _sha256(self.checkpoint_path)
        if checkpoint_hash != self.expected_checkpoint_sha256:
            raise RuntimeError(
                "RegAD checkpoint SHA256 mismatch: "
                f"expected {self.expected_checkpoint_sha256}, got {checkpoint_hash}"
            )

        stn_module = _load_module(
            self.source_root / "models" / "stn.py", "_fmad_regad_official_stn"
        )
        siamese_module = _load_module(
            self.source_root / "models" / "siamese.py", "_fmad_regad_official_siamese"
        )
        args = SimpleNamespace(stn_mode=self.stn_mode)
        stn = stn_module.stn_net(args, pretrained=False)
        encoder = siamese_module.Encoder()
        predictor = siamese_module.Predictor()
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        for key, model in (("STN", stn), ("ENC", encoder), ("PRED", predictor)):
            if key not in checkpoint:
                raise KeyError(f"released RegAD checkpoint is missing {key}")
            model.load_state_dict(checkpoint[key], strict=True)

        # RegAD's upstream STN uses Tensor.cuda() for inverse grids.  Setting the
        # current device makes that operation follow the explicit assigned slot.
        torch.cuda.set_device(self.device)
        self._stn = stn.to(self.device).eval()
        self._encoder = encoder.eval()
        self._predictor = predictor.eval()
        for model in (self._stn, self._encoder, self._predictor):
            model.requires_grad_(False)

        if str(self.dependency_root) not in sys.path:
            sys.path.insert(0, str(self.dependency_root))
        import kornia

        self._kornia = kornia
        self._source_revision = revision
        self._checkpoint_hash = checkpoint_hash

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        assert self._stn is not None
        images = images.to(self.device, non_blocking=True)
        with torch.no_grad():
            self._stn(images)
        layer1 = self._stn.stn1_output
        layer2 = self._stn.stn2_output
        layer3 = self._stn.stn3_output
        embeddings = _embedding_concat(layer1, layer2)
        embeddings = _embedding_concat(embeddings, layer3)
        if embeddings.shape[1:] != (448, 56, 56):
            raise ValueError(f"unexpected RegAD embedding shape: {tuple(embeddings.shape)}")
        return embeddings

    def _fit_support_gaussian(self, augmented_support: torch.Tensor) -> None:
        view_count = int(augmented_support.shape[0])
        if view_count < 2:
            raise ValueError("RegAD positional covariance requires at least two support views")

        feature_sum = None
        for start in range(0, view_count, self.support_batch_size):
            batch = augmented_support[start : start + self.support_batch_size]
            features = self._extract_features(batch)
            batch_sum = features.sum(dim=0)
            feature_sum = batch_sum if feature_sum is None else feature_sum + batch_sum
            self.log_training(
                step=start + len(batch),
                stage="regad_official_support_mean",
                support_view_count=view_count,
            )
        assert feature_sum is not None
        mean = feature_sum / float(view_count)
        channels, height, width = mean.shape
        positions = height * width
        mean_flat = mean.permute(1, 2, 0).reshape(positions, channels).contiguous()

        # The released implementation builds a CxCx(HW) covariance tensor.
        # Keep the same positional covariance while accumulating support and
        # spatial chunks so the 16-shot AD2 extension does not retain 352
        # full-resolution feature maps or a full augmentation batch on the GPU.
        covariance_sum = torch.zeros(
            (positions, channels, channels), dtype=mean.dtype, device=self.device
        )
        for start in range(0, view_count, self.support_batch_size):
            batch = augmented_support[start : start + self.support_batch_size]
            features = self._extract_features(batch)
            flat = features.permute(0, 2, 3, 1).reshape(len(batch), positions, channels)
            for p_start in range(0, positions, self.spatial_chunk_size):
                p_end = min(positions, p_start + self.spatial_chunk_size)
                centered = flat[:, p_start:p_end] - mean_flat[p_start:p_end].unsqueeze(0)
                centered = centered.permute(1, 0, 2).contiguous()
                covariance_sum[p_start:p_end].add_(
                    torch.bmm(centered.transpose(1, 2), centered)
                )
            self.log_training(
                step=start + len(batch),
                stage="regad_official_support_covariance",
                support_view_count=view_count,
            )

        covariance = covariance_sum.div_(float(view_count - 1))
        covariance.diagonal(dim1=1, dim2=2).add_(self.covariance_ridge)
        del covariance_sum

        inverse = torch.empty_like(covariance)
        inverse_chunk = 16
        for start in range(0, positions, inverse_chunk):
            end = min(positions, start + inverse_chunk)
            inverse[start:end] = torch.linalg.inv(covariance[start:end])
        del covariance

        self._support_mean = mean_flat
        self._inverse_covariance = inverse
        self._feature_shape = (channels, height, width)
        self._support_view_count = view_count

    def fit(self, support_paths: list[str], context: MethodContext) -> None:
        if not support_paths:
            raise ValueError("support_paths must not be empty")
        self._load_official_models()
        assert self._kornia is not None
        self._object_name = context.object_name
        self._support_paths = list(support_paths)
        self._support_mean = None
        self._inverse_covariance = None

        # RegAD's released test transform creates 22 augmented views per
        # support image.  All targets use one fixed released 2-shot bottle
        # checkpoint; target support count affects this Gaussian only.
        augmented = [
            _support_augmentations(_resize_to_tensor(path, self.image_size), self._kornia)
            for path in support_paths
        ]
        augmented_support = torch.cat(augmented, dim=0)
        self._fit_support_gaussian(augmented_support)
        del augmented_support, augmented
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, test_image_path: str) -> tuple[np.ndarray, float]:
        if (
            self._stn is None
            or self._support_mean is None
            or self._inverse_covariance is None
            or self._feature_shape is None
        ):
            raise RuntimeError("regad_official fit() must run before prediction")
        started = time.perf_counter()
        original_shape = None
        with Image.open(test_image_path) as image:
            original_shape = (image.height, image.width)
        query = _resize_to_tensor(test_image_path, self.image_size).unsqueeze(0)
        features = self._extract_features(query)
        channels, height, width = self._feature_shape
        query_flat = features[0].permute(1, 2, 0).reshape(height * width, channels)
        delta = query_flat - self._support_mean
        transformed = torch.bmm(
            self._inverse_covariance, delta.unsqueeze(-1)
        ).squeeze(-1)
        distances = torch.sqrt(torch.sum(delta * transformed, dim=1).clamp_min_(0.0))
        low_resolution = distances.reshape(1, 1, height, width)
        official_map = F.interpolate(
            low_resolution,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )[0, 0].cpu().numpy().astype(np.float32, copy=False)
        official_map = gaussian_filter(official_map, sigma=self.smoothing_sigma)

        # The common AD2 evaluator requires a native query-size map.  The map
        # is first formed and smoothed in the official 224-pixel domain.
        native_map = F.interpolate(
            torch.from_numpy(official_map).unsqueeze(0).unsqueeze(0),
            size=original_shape,
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy().astype(np.float32, copy=False)
        return native_map, float(np.max(native_map))

    def predict_batch(self, test_image_paths: Sequence[str]) -> list[Prediction]:
        predictions = []
        for path in test_image_paths:
            started = time.perf_counter()
            score_map, image_score = self.predict(path)
            predictions.append(
                Prediction(
                    score_map=score_map,
                    image_score=image_score,
                    inference_seconds=time.perf_counter() - started,
                )
            )
        return predictions

    def object_metadata(self) -> dict:
        return {
            "implementation": "MediaBrain-SJTU/RegAD official test path adapter",
            "official_repo_commit": self._source_revision or self.expected_commit,
            "official_source_root": str(self.source_root),
            "released_checkpoint": str(self.checkpoint_path),
            "released_checkpoint_sha256": self._checkpoint_hash or self.expected_checkpoint_sha256,
            "released_checkpoint_archive_sha256": self.source_archive_sha256,
            "checkpoint_source_model": "MVTec bottle, 2-shot, rotation_scale",
            "checkpoint_selection": "official source MVTec test image AUROC and pixel AUROC",
            "source_test_selected": True,
            "ad2_labels_used_for_checkpoint_or_method_selection": False,
            "target_fit_support_count": len(self._support_paths),
            "support_augmentation_views_per_image": 22,
            "support_view_count": self._support_view_count,
            "support_resize": [self.image_size, self.image_size],
            "non_square_resize_extension": "stretch to square 224x224 for official STN geometry",
            "covariance": "per-position unbiased sample covariance over augmented support views plus 0.01 I",
            "score_map": "official positional Mahalanobis distance, bilinear 224 upsample, Gaussian sigma 4, bilinear lift to native AD2 shape",
            "feature_shape": list(self._feature_shape) if self._feature_shape else None,
        }
