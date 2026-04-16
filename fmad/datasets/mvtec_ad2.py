"""MVTec AD 2 dataset implementation."""

import os
from typing import Dict, List, Optional

from fmad.datasets.base import BaseDataset, ObjectInfo
from fmad.registry import DATASET_REGISTRY

# Per-object defaults for MVTec AD 2
_OBJECTS = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]

_ANOMALY_TYPES = {o: ["bad"] for o in _OBJECTS}

_RESOLUTION = {
    "can": 672, "fabric": 672, "fruit_jelly": 672, "rice": 672,
    "sheet_metal": 448, "vial": 672, "wallplugs": 672, "walnuts": 672,
}

_MASKING_AGNOSTIC = {
    "can": False, "fabric": False, "fruit_jelly": False, "rice": False,
    "sheet_metal": False, "vial": True, "wallplugs": True, "walnuts": False,
}

_ROTATION_AGNOSTIC = {o: True for o in _OBJECTS}
_ROTATION_INFORMED = {o: False for o in _OBJECTS}


def _get_preprocess_settings(preprocess: str):
    if preprocess == "agnostic":
        return dict(_MASKING_AGNOSTIC), dict(_ROTATION_AGNOSTIC)
    elif preprocess == "informed":
        return dict(_MASKING_AGNOSTIC), dict(_ROTATION_INFORMED)
    elif preprocess == "masking_only":
        return dict(_MASKING_AGNOSTIC), {o: False for o in _OBJECTS}
    elif preprocess == "no_mask_no_rotation":
        return {o: False for o in _OBJECTS}, {o: False for o in _OBJECTS}
    else:
        raise ValueError(f"Unknown preprocess mode: {preprocess}")


@DATASET_REGISTRY.register("mvtec_ad2")
class MVTecAD2Dataset(BaseDataset):
    """MVTec Anomaly Detection 2 dataset with distribution shift."""

    def __init__(self, data_root: str, config: dict):
        super().__init__(data_root, config)
        preprocess = config.get("preprocess", "agnostic")
        self._masking, self._rotation = _get_preprocess_settings(preprocess)
        self._filter_objects = config.get("objects", None)

    def get_objects(self) -> List[ObjectInfo]:
        objects = []
        for name in _OBJECTS:
            if self._filter_objects and name not in self._filter_objects:
                continue
            objects.append(ObjectInfo(
                name=name,
                anomaly_types=_ANOMALY_TYPES[name],
                resolution=_RESOLUTION[name],
                masking=self._masking[name],
                rotation=self._rotation[name],
            ))
        return objects

    def get_train_images(self, object_name: str) -> List[str]:
        train_dir = os.path.join(self.data_root, object_name, "train", "good")
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"Train directory not found: {train_dir}")
        return sorted(
            os.path.join(train_dir, f) for f in os.listdir(train_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        )

    def get_test_images(self, object_name: str, split: str = "test_public") -> Dict[str, List[str]]:
        test_dir = os.path.join(self.data_root, object_name, split)
        if not os.path.isdir(test_dir):
            raise FileNotFoundError(f"Test directory not found: {test_dir}")

        result = {}
        for subdir in sorted(os.listdir(test_dir)):
            subdir_path = os.path.join(test_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            if subdir == "ground_truth":
                continue
            images = sorted(
                os.path.join(subdir_path, f) for f in os.listdir(subdir_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            )
            if images:
                result[subdir] = images
        return result

    def get_ground_truth_dir(self, object_name: str, split: str = "test_public") -> Optional[str]:
        gt_dir = os.path.join(self.data_root, object_name, split, "ground_truth")
        return gt_dir if os.path.isdir(gt_dir) else None
