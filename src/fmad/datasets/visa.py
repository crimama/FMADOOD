"""ViSA (Visual Anomaly) dataset implementation."""

import os
from typing import Dict, List, Optional

from fmad.datasets.base import BaseDataset, ObjectInfo
from fmad.registry import DATASET_REGISTRY

_OBJECTS = [
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
]


@DATASET_REGISTRY.register("visa")
class VisADataset(BaseDataset):
    """ViSA benchmark dataset (1cls pytorch standard format)."""

    def __init__(self, data_root: str, config: dict):
        super().__init__(data_root, config)
        self.default_resolution = config.get("resolution", 448)
        self._filter_objects = config.get("objects", None)
        if self._filter_objects is not None:
            if not isinstance(self._filter_objects, list) or not self._filter_objects:
                raise ValueError("dataset_config.objects must be a non-empty list")
            unknown = set(self._filter_objects) - set(_OBJECTS)
            if unknown:
                raise ValueError(f"Unknown ViSA objects: {sorted(unknown)}")

    def get_objects(self) -> List[ObjectInfo]:
        objects = []
        for obj in _OBJECTS:
            if self._filter_objects and obj not in self._filter_objects:
                continue
            test_dir = os.path.join(self.data_root, obj, "test")
            types = []
            if os.path.isdir(test_dir):
                types = [
                    d for d in sorted(os.listdir(test_dir))
                    if os.path.isdir(os.path.join(test_dir, d)) and d != "good"
                ]
            if not types:
                types = ["bad"]
            objects.append(
                ObjectInfo(
                    name=obj,
                    anomaly_types=types,
                    resolution=self.default_resolution,
                    masking=False,
                    rotation=False,
                )
            )
        return objects

    def get_train_images(self, object_name: str) -> List[str]:
        train_dir = os.path.join(self.data_root, object_name, "train", "good")
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"Train directory not found: {train_dir}")
        return sorted(
            os.path.join(train_dir, f)
            for f in os.listdir(train_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        )

    def get_test_images(self, object_name: str, split: str = "test") -> Dict[str, List[str]]:
        test_dir = os.path.join(self.data_root, object_name, "test")
        if not os.path.isdir(test_dir):
            raise FileNotFoundError(f"Test directory not found: {test_dir}")

        result = {}
        for subdir in sorted(os.listdir(test_dir)):
            subdir_path = os.path.join(test_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            images = sorted(
                os.path.join(subdir_path, f)
                for f in os.listdir(subdir_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            )
            if images:
                result[subdir] = images
        return result

    def get_ground_truth_dir(self, object_name: str, split: str = "test") -> Optional[str]:
        gt_dir = os.path.join(self.data_root, object_name, "ground_truth")
        return gt_dir if os.path.isdir(gt_dir) else None
