"""Base class for anomaly detection methods."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class ObjectResult:
    """Immutable result for a single object's anomaly detection run."""
    object_name: str
    anomaly_maps: Dict[str, np.ndarray]     # {sample_key: anomaly_map}
    anomaly_scores: Dict[str, float]        # {sample_key: score}
    time_memorybank: float                  # seconds
    inference_times: Dict[str, float]       # {sample_key: seconds}
    metadata: Dict = field(default_factory=dict)


class BaseMethod(ABC):
    """Abstract base for anomaly detection methods.

    Subclasses implement fit() and predict() to define the method's pipeline.
    The framework calls run_object() which orchestrates the full flow.
    """

    def __init__(self, backbone, config: dict):
        self.backbone = backbone
        self.config = config

    @abstractmethod
    def fit(self, train_image_paths: list, object_name: str, seed: int = 0) -> None:
        """Build memory bank / reference from training images.

        Args:
            train_image_paths: Paths to normal training images.
            object_name: Name of the object category.
            seed: Random seed for reproducibility.
        """

    @abstractmethod
    def predict(self, test_image_path: str) -> tuple:
        """Predict anomaly map for a single test image.

        Args:
            test_image_path: Path to the test image.

        Returns:
            (anomaly_map, anomaly_score): numpy array and float score.
        """

    @abstractmethod
    def run_object(self, dataset, object_name: str, seed: int,
                   output_dir: str, save_maps: bool = True) -> ObjectResult:
        """Run full pipeline for one object: fit + predict all test images.

        Args:
            dataset: BaseDataset instance.
            object_name: Object category name.
            seed: Random seed.
            output_dir: Directory to save anomaly maps.
            save_maps: Whether to save tiff/jpg outputs.

        Returns:
            ObjectResult with all predictions and timings.
        """
