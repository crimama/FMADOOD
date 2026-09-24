"""Base class for feature extraction backbones."""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


def _validate_resolution(resolution: int) -> int:
    """Validate a positive integer short-edge resolution."""
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution <= 0:
        raise ValueError("resolution must be a positive integer")
    return resolution


class BaseBackbone(ABC):
    """Abstract base for vision backbone models."""

    def __init__(self, model_name: str, device: str, smaller_edge_size: int = 448):
        self.model_name = model_name
        self.device = device
        self.smaller_edge_size = _validate_resolution(smaller_edge_size)

    @abstractmethod
    def prepare_image(
        self, img, *, resolution: int | None = None
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Preprocess image for the model.

        ``resolution`` selects the short-edge size for this call only.  A
        ``None`` value preserves the backbone's configured default.

        Returns:
            (image_tensor, grid_size)
        """

    @abstractmethod
    def extract_features(self, image_tensor) -> List[np.ndarray]:
        """Extract multi-layer patch features.

        Returns:
            List of feature arrays, one per layer.
        """

    @abstractmethod
    def extract_cls_features(self, image_tensor):
        """Extract CLS token features for sample selection."""

    @abstractmethod
    def compute_background_mask(
        self, features, grid_size, threshold: float = 1.0, masking_type=False, **kwargs
    ) -> np.ndarray:
        """Compute foreground/background mask from features.

        Returns:
            Boolean mask array.
        """

    def set_resolution(self, smaller_edge_size: int) -> None:
        """Update the default input resolution without reloading the model."""
        self.smaller_edge_size = _validate_resolution(smaller_edge_size)
