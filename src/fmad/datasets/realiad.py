"""Real-IAD C1 single-view dataset in the VisA-style folder layout.

The tree is produced by scripts/prepare_realiad_c1.py from the official
realiad_jsons_sv split: <object>/train/good, <object>/test/{good,bad} and
<object>/ground_truth/bad/<stem>.png.
"""

import os
from typing import Dict, List, Optional

from fmad.datasets.base import BaseDataset, ObjectInfo
from fmad.registry import DATASET_REGISTRY

_OBJECTS = [
    "audiojack", "bottle_cap", "button_battery", "end_cap", "eraser",
    "fire_hood", "mint", "mounts", "pcb", "phone_battery",
    "plastic_nut", "plastic_plug", "porcelain_doll", "regulator",
    "rolled_strip_base", "sim_card_set", "switch", "tape", "terminalblock",
    "toothbrush", "toy", "toy_brick", "transistor1", "u_block", "usb",
    "usb_adaptor", "vcpill", "wooden_beads", "woodstick", "zipper",
]
_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


@DATASET_REGISTRY.register("realiad")
class RealIADDataset(BaseDataset):
    """Real-IAD C1 (single view, 1024x1024 release)."""

    def __init__(self, data_root: str, config: dict):
        super().__init__(data_root, config)
        self.default_resolution = config.get("resolution", 448)
        self._filter_objects = config.get("objects", None)
        if self._filter_objects is not None:
            if not isinstance(self._filter_objects, list) or not self._filter_objects:
                raise ValueError("dataset_config.objects must be a non-empty list")
            unknown = set(self._filter_objects) - set(_OBJECTS)
            if unknown:
                raise ValueError(f"Unknown Real-IAD objects: {sorted(unknown)}")

    def get_objects(self) -> List[ObjectInfo]:
        return [
            ObjectInfo(name=obj, anomaly_types=["bad"], resolution=self.default_resolution,
                       masking=False, rotation=False)
            for obj in _OBJECTS
            if not self._filter_objects or obj in self._filter_objects
        ]

    def get_train_images(self, object_name: str) -> List[str]:
        train_dir = os.path.join(self.data_root, object_name, "train", "good")
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"Train directory not found: {train_dir}")
        return sorted(os.path.join(train_dir, f) for f in os.listdir(train_dir)
                      if f.lower().endswith(_EXTENSIONS))

    def get_test_images(self, object_name: str, split: str = "test") -> Dict[str, List[str]]:
        test_dir = os.path.join(self.data_root, object_name, "test")
        if not os.path.isdir(test_dir):
            raise FileNotFoundError(f"Test directory not found: {test_dir}")
        result = {}
        for subdir in sorted(os.listdir(test_dir)):
            path = os.path.join(test_dir, subdir)
            if not os.path.isdir(path):
                continue
            images = sorted(os.path.join(path, f) for f in os.listdir(path)
                            if f.lower().endswith(_EXTENSIONS))
            if images:
                result[subdir] = images
        return result

    def get_ground_truth_dir(self, object_name: str, split: str = "test") -> Optional[str]:
        gt_dir = os.path.join(self.data_root, object_name, "ground_truth")
        return gt_dir if os.path.isdir(gt_dir) else None
