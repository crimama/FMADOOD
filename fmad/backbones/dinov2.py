"""DINOv2 backbone wrapper — delegates to src/backbones.py."""

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from fmad.backbones.base import BaseBackbone
from fmad.registry import BACKBONE_REGISTRY

# Ensure src/ is importable
_SRC_DIR = str(Path(__file__).resolve().parents[2] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@BACKBONE_REGISTRY.register("dinov2_vitl14")
@BACKBONE_REGISTRY.register("dinov2_vits14")
class DINOv2Backbone(BaseBackbone):
    """DINOv2 backbone using src.backbones.DINOv2Wrapper under the hood."""

    def __init__(self, model_name: str, device: str, smaller_edge_size: int = 448):
        super().__init__(model_name, device, smaller_edge_size)
        self._wrapper = None

    def _ensure_loaded(self) -> None:
        if self._wrapper is None:
            from src.backbones import DINOv2Wrapper
            self._wrapper = DINOv2Wrapper(
                self.model_name, self.device, self.smaller_edge_size
            )

    def set_resolution(self, smaller_edge_size: int) -> None:
        if smaller_edge_size != self.smaller_edge_size:
            self.smaller_edge_size = smaller_edge_size
            # Reload with new resolution
            from src.backbones import DINOv2Wrapper
            self._wrapper = DINOv2Wrapper(
                self.model_name, self.device, self.smaller_edge_size
            )

    def warmup(self, image_path: str, iters: int = 25) -> None:
        """CUDA warmup with a sample image."""
        self._ensure_loaded()
        for _ in range(iters):
            img_tensor, _ = self._wrapper.prepare_image(image_path)
            self._wrapper.extract_features(img_tensor)

    def prepare_image(self, img) -> Tuple:
        self._ensure_loaded()
        return self._wrapper.prepare_image(img)

    def extract_features(self, image_tensor) -> List[np.ndarray]:
        self._ensure_loaded()
        return self._wrapper.extract_features(image_tensor)

    def extract_cls_features(self, image_tensor) -> torch.Tensor:
        self._ensure_loaded()
        return self._wrapper.extract_cls_features(image_tensor)

    def compute_background_mask(self, features, grid_size,
                                 threshold: float = 1.0,
                                 masking_type=False, **kwargs) -> np.ndarray:
        self._ensure_loaded()
        return self._wrapper.compute_background_mask(
            features, grid_size, threshold=threshold,
            masking_type=masking_type, **kwargs
        )
