"""DINOv2 preprocessing and feature extraction wrapper."""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA

from torchvision import transforms

from fmad.backbones.base import _validate_resolution
from fmad.backbones.loader import load_dinov2_model


class DINOv2Wrapper:
    def __init__(
        self,
        model_name: str,
        device: str,
        smaller_edge_size: int = 224,
        half_precision: bool = False,
        feature_layers: Optional[Sequence[int]] = None,
    ):
        defaults = (2, 5, 8, 11) if "_vits14" in model_name else (5, 11, 17, 23)
        self.feature_layers = tuple(feature_layers or defaults)
        self.device = device
        self.smaller_edge_size = _validate_resolution(smaller_edge_size)
        self.half_precision = half_precision
        self.model_name = model_name
        self._transform_cache: dict[int, transforms.Compose] = {}
        self.transform = self._get_transform(self.smaller_edge_size)
        self.model = self.load_model()
        depth = len(getattr(self.model, "blocks", ()))
        if not self.feature_layers or min(self.feature_layers) < 0:
            raise ValueError("feature_layers must contain non-negative indices")
        if depth and max(self.feature_layers) >= depth:
            raise ValueError(f"feature layer index must be less than model depth {depth}")

    def load_model(self):
        model = load_dinov2_model(self.model_name)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        self.transform = self._get_transform(self.smaller_edge_size)
        return model.to(self.device)

    def _get_transform(self, resolution: int) -> transforms.Compose:
        resolution = _validate_resolution(resolution)
        transform = self._transform_cache.get(resolution)
        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        size=resolution,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                        antialias=True,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
            self._transform_cache[resolution] = transform
        return transform

    def set_resolution(self, smaller_edge_size: int) -> None:
        """Change the default preprocessing resolution without reloading DINOv2."""
        self.smaller_edge_size = _validate_resolution(smaller_edge_size)
        self.transform = self._get_transform(self.smaller_edge_size)

    def prepare_image(self, img, *, resolution: int | None = None):
        if isinstance(img, (str, bytes)):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        target_resolution = (
            self.smaller_edge_size if resolution is None else _validate_resolution(resolution)
        )
        image_tensor = self._get_transform(target_resolution)(img)
        height, width = image_tensor.shape[1:]
        patch_size = int(self.model.patch_size)
        cropped_width, cropped_height = width - width % patch_size, height - height % patch_size
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]
        return image_tensor, (cropped_height // patch_size, cropped_width // patch_size)

    def extract_features(self, image_tensor):
        with torch.inference_mode():
            image_batch = image_tensor.unsqueeze(0).to(self.device)
            if self.half_precision:
                image_batch = image_batch.half()
            tokens = self.model.get_intermediate_layers(image_batch, list(self.feature_layers))
            return [token.squeeze().float().cpu().numpy() for token in tokens]

    def extract_cls_features(self, image_tensor):
        with torch.inference_mode():
            image_batch = image_tensor.unsqueeze(0).to(self.device)
            if self.half_precision:
                image_batch = image_batch.half()
            _, cls_token = self.model.get_intermediate_layers(
                image_batch, n=1, return_class_token=True
            )[0]
        return cls_token.squeeze(0)

    def get_embedding_visualization(self, tokens, grid_size, resized_mask=None, normalize=True):
        pca = PCA(n_components=3, svd_solver="randomized")
        if resized_mask is not None:
            tokens = tokens[resized_mask]
        reduced = pca.fit_transform(tokens.astype(np.float32))
        if resized_mask is not None:
            restored = np.zeros((*resized_mask.shape, 3), dtype=reduced.dtype)
            restored[resized_mask] = reduced
            reduced = restored
        reduced = reduced.reshape((*grid_size, -1))
        if normalize:
            return (reduced - np.min(reduced)) / (np.max(reduced) - np.min(reduced))
        return reduced

    def compute_background_mask_from_image(self, image, threshold=10, masking_type=None):
        image_tensor, grid_size = self.prepare_image(image)
        return self.compute_background_mask(
            self.extract_features(image_tensor), grid_size, threshold, masking_type
        )

    def compute_background_mask(
        self, img_features, grid_size, threshold=1, masking_type=False, kernel_size=3
    ):
        first_pc = PCA(n_components=1, svd_solver="randomized").fit_transform(
            img_features.astype(np.float32)
        )
        if masking_type:
            mask = first_pc.squeeze() > threshold
            masked, unmasked = img_features[mask], img_features[~mask]
            masked_variance = np.median(np.var(masked, axis=0)) if masked.size else 0
            unmasked_variance = np.median(np.var(unmasked, axis=0)) if unmasked.size else 0
            if masked_variance < unmasked_variance:
                mask = ~mask
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        else:
            mask = np.ones_like(first_pc, dtype=bool)
        return mask.squeeze()
