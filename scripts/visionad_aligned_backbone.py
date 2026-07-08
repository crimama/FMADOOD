# /// script
# requires-python = ">=3.8"
# dependencies = []
# ///
# pyright: reportMissingImports=false
"""DINOv2-Register backbone adapter with VisionAD-style preprocessing."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as vision_tf

ImageInput = Union[str, Path, np.ndarray]


class VisionADAlignedBackbone:
    def __init__(
        self,
        model_name: str,
        device: str,
        image_size: int,
        crop_size: int,
        feature_layers: Sequence[int],
    ) -> None:
        from src.dinov2_loader import load_dinov2_model  # noqa: PLC0415

        self.model_name = model_name
        self.device = device
        self.image_size = image_size
        self.crop_size = crop_size
        self.feature_layers = tuple(feature_layers)
        self.model = load_dinov2_model(model_name).to(device).eval()
        self.patch_size = int(self.model.patch_size)

    def set_resolution(self, smaller_edge_size: int) -> None:
        _ = smaller_edge_size

    def prepare_image(self, img: ImageInput) -> Tuple[torch.Tensor, Tuple[int, int]]:
        image = self._center_crop(self._to_pil(img))
        image_tensor = vision_tf.to_tensor(image)
        image_tensor = vision_tf.normalize(
            image_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        height, width = image_tensor.shape[1:]
        cropped_height = height - height % self.patch_size
        cropped_width = width - width % self.patch_size
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]
        return image_tensor, (
            cropped_height // self.patch_size,
            cropped_width // self.patch_size,
        )

    def extract_features(self, image_tensor: torch.Tensor) -> List[np.ndarray]:
        with torch.inference_mode():
            batch = image_tensor.unsqueeze(0).to(self.device)
            outputs = self.model.get_intermediate_layers(batch, list(self.feature_layers))
            return [self._patch_tokens(output) for output in outputs]

    @staticmethod
    def _to_pil(img: ImageInput) -> Image.Image:
        if isinstance(img, Path):
            return Image.open(img).convert("RGB")
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        return Image.fromarray(img)

    def _center_crop(self, image: Image.Image) -> Image.Image:
        resized = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        left = (self.image_size - self.crop_size) // 2
        upper = (self.image_size - self.crop_size) // 2
        return resized.crop((left, upper, left + self.crop_size, upper + self.crop_size))

    @staticmethod
    def _patch_tokens(output: object) -> np.ndarray:
        tokens = output[0] if isinstance(output, tuple) else output
        if not isinstance(tokens, torch.Tensor):
            message = f"Expected tensor features, got {type(tokens).__name__}"
            raise TypeError(message)
        return tokens.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
