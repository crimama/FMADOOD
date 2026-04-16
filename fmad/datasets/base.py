"""Base class for anomaly detection datasets."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ObjectInfo:
    """Immutable metadata for a single dataset object."""
    name: str
    anomaly_types: List[str]
    resolution: int
    masking: bool
    rotation: bool


class BaseDataset(ABC):
    """Abstract base for anomaly detection datasets."""

    def __init__(self, data_root: str, config: dict):
        self.data_root = data_root
        self.config = config

    @abstractmethod
    def get_objects(self) -> List[ObjectInfo]:
        """Return list of all object categories with metadata."""

    @abstractmethod
    def get_train_images(self, object_name: str) -> List[str]:
        """Return sorted list of training image paths for an object."""

    @abstractmethod
    def get_test_images(self, object_name: str, split: str = "test_public") -> Dict[str, List[str]]:
        """Return test images grouped by anomaly type.

        Returns:
            {anomaly_type: [image_path, ...]}
        """

    @abstractmethod
    def get_ground_truth_dir(self, object_name: str, split: str = "test_public") -> Optional[str]:
        """Return path to ground truth directory, or None if unavailable."""

    def get_object_info(self, object_name: str) -> ObjectInfo:
        """Get metadata for a specific object."""
        for obj in self.get_objects():
            if obj.name == object_name:
                return obj
        raise KeyError(f"Object '{object_name}' not found in dataset.")
