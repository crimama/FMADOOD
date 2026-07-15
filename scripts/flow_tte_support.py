# /// script
# requires-python = ">=3.8"
# dependencies = []
# ///
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Sequence, Tuple

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import torch

_EPS = 1e-12
_FIXED_JSON_PREFIX = "fixed_json="
SUPERADD_FULL_SUPPORT_POLICY = "superadd_full_7of8"
_SUPERAD_ROTATION_ANGLES = {
    "superad_rot000": 0,
    "superad_rot045": 45,
    "superad_rot090": 90,
    "superad_rot135": 135,
    "superad_rot180": 180,
    "superad_rot225": 225,
    "superad_rot270": 270,
    "superad_rot315": 315,
}
_RIGHT_ANGLE_ROTATIONS = {"rot90": 1, "rot180": 2, "rot270": 3}


class ClsBackboneLike(Protocol):
    def prepare_image(self, img: np.ndarray) -> Tuple["torch.Tensor", Tuple[int, int]]: ...

    def extract_cls_features(self, image_tensor: "torch.Tensor") -> "torch.Tensor": ...


def select_support_paths(
    paths: Sequence[Path],
    shots: int,
    policy: str,
    seed: int,
) -> Tuple[Path, ...]:
    if policy == SUPERADD_FULL_SUPPORT_POLICY:
        return tuple(path for index, path in enumerate(paths) if index % 8 != 0)
    if is_fixed_support_policy(policy):
        return select_support_paths_from_json(paths, shots, policy)
    if policy == "first":
        return tuple(paths[:shots])
    if policy == "visionad_seeded_random":
        if len(paths) < shots:
            return tuple(paths)
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(paths), size=shots, replace=False)
        return tuple(paths[int(index)] for index in indices)
    message = f"Unknown support selection policy: {policy}"
    raise RuntimeError(message)


def select_superadd_threshold_paths(paths: Sequence[Path]) -> Tuple[Path, ...]:
    """Return the official sorted-index 1/8 normal threshold split."""
    return tuple(path for index, path in enumerate(paths) if index % 8 == 0)


def greedy_coreset_indices(
    features: npt.NDArray[np.float32],
    shots: int,
) -> Tuple[int, ...]:
    if shots <= 0:
        return ()
    if len(features) <= shots:
        return tuple(range(len(features)))
    normalized = features.astype(np.float32, copy=False)
    norms = np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), _EPS)
    normalized = normalized / norms
    center = normalized.mean(axis=0, keepdims=True)
    selected = [int(np.argmin(np.sum((normalized - center) ** 2, axis=1)))]
    min_distances = np.sum((normalized - normalized[selected[0]]) ** 2, axis=1)
    while len(selected) < shots:
        next_index = int(np.argmax(min_distances))
        selected.append(next_index)
        distances = np.sum((normalized - normalized[next_index]) ** 2, axis=1)
        min_distances = np.minimum(min_distances, distances)
    return tuple(selected)


def select_support_paths_for_backbone(
    backbone: ClsBackboneLike,
    paths: Sequence[Path],
    shots: int,
    policy: str,
    seed: int,
) -> Tuple[Path, ...]:
    if is_cls_coreset_policy(policy):
        return select_support_paths_by_cls_coreset(backbone, paths, shots)
    return select_support_paths(paths, shots=shots, policy=policy, seed=seed)


def is_cls_coreset_policy(policy: str) -> bool:
    return policy in {
        "cls_greedy_coreset",
        "dinov2_cls_greedy_coreset",
        "dinov3_cls_greedy_coreset",
    }


def is_fixed_support_policy(policy: str) -> bool:
    return policy.startswith(_FIXED_JSON_PREFIX)


def select_support_paths_from_json(
    paths: Sequence[Path],
    shots: int,
    policy: str,
) -> Tuple[Path, ...]:
    manifest_path = Path(policy[len(_FIXED_JSON_PREFIX) :])
    object_name = infer_object_name_from_train_paths(paths)
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw_manifest, dict):
        message = f"Fixed support manifest must be an object: {manifest_path}"
        raise TypeError(message)
    raw_paths = raw_manifest.get(object_name)
    if not isinstance(raw_paths, list):
        message = f"Fixed support manifest has no list for object: {object_name}"
        raise TypeError(message)
    selected_paths = tuple(Path(item) for item in raw_paths if isinstance(item, str))
    if len(selected_paths) != len(raw_paths):
        message = f"{object_name}: fixed support paths must all be strings"
        raise RuntimeError(message)
    if len(selected_paths) != shots:
        message = f"{object_name}: fixed support count {len(selected_paths)} != shots {shots}"
        raise RuntimeError(message)
    train_path_by_text = {str(path): path for path in paths}
    missing = [str(path) for path in selected_paths if str(path) not in train_path_by_text]
    if missing:
        message = f"{object_name}: fixed support paths are not in train/good: {missing}"
        raise RuntimeError(message)
    return tuple(train_path_by_text[str(path)] for path in selected_paths)


def infer_object_name_from_train_paths(paths: Sequence[Path]) -> str:
    if not paths:
        message = "Cannot infer object name from an empty train path list"
        raise RuntimeError(message)
    return paths[0].parents[2].name


def select_support_paths_by_cls_coreset(
    backbone: ClsBackboneLike,
    paths: Sequence[Path],
    shots: int,
) -> Tuple[Path, ...]:
    if len(paths) <= shots:
        return tuple(paths)
    features = []
    for path in paths:
        image_tensor, _ = backbone.prepare_image(read_rgb(path))
        cls_features = backbone.extract_cls_features(image_tensor)
        features.append(cls_features.detach().cpu().numpy().reshape(-1))
    feature_matrix = np.stack(features, axis=0).astype(np.float32, copy=False)
    indices = greedy_coreset_indices(feature_matrix, shots)
    return tuple(paths[index] for index in indices)


def read_rgb(path: Path) -> np.ndarray:
    cv2_module = cv2
    if cv2_module is None:
        message = "OpenCV is required to read support/query images"
        raise RuntimeError(message)
    bgr = cv2_module.imread(str(path), cv2_module.IMREAD_COLOR)
    if bgr is None:
        message = f"Could not read image: {path}"
        raise RuntimeError(message)
    return cv2_module.cvtColor(bgr, cv2_module.COLOR_BGR2RGB)


def rotate_rgb_like_superad(image: np.ndarray, angle: int) -> np.ndarray:
    cv2_module = cv2
    if cv2_module is None:
        message = "OpenCV is required for SuperAD support rotations"
        raise RuntimeError(message)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rotation = cv2_module.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2_module.warpAffine(
        image,
        rotation,
        image.shape[1::-1],
        flags=cv2_module.INTER_LINEAR,
        borderMode=cv2_module.BORDER_DEFAULT,
    )


def transform_rgb(image: np.ndarray, transform_name: str) -> np.ndarray:
    superad_angle = _SUPERAD_ROTATION_ANGLES.get(transform_name)
    if superad_angle is not None:
        return rotate_rgb_like_superad(image, superad_angle)
    if transform_name == "identity":
        return image
    right_angle_turns = _RIGHT_ANGLE_ROTATIONS.get(transform_name)
    if right_angle_turns is not None:
        return np.ascontiguousarray(np.rot90(image, k=right_angle_turns))
    if transform_name == "flip_vertical":
        return np.ascontiguousarray(np.flip(image, axis=0))
    if transform_name == "flip_horizontal":
        return np.ascontiguousarray(np.flip(image, axis=1))
    message = f"Unknown support transform: {transform_name}"
    raise RuntimeError(message)


def merge_layer_features(layer_features: Sequence[np.ndarray], feature_fusion: str) -> np.ndarray:
    normalized = [normalize_layer_features(layer) for layer in layer_features]
    if feature_fusion == "layer_norm_mean":
        return np.mean(np.stack(normalized, axis=0), axis=0).astype(np.float32, copy=False)
    if feature_fusion == "visionad_mean_l2":
        merged = np.mean(
            np.stack([layer.astype(np.float32, copy=False) for layer in layer_features], axis=0),
            axis=0,
        )
        norm = np.maximum(np.linalg.norm(merged, axis=1, keepdims=True), _EPS)
        return (merged / norm).astype(np.float32, copy=False)
    message = f"Unknown feature fusion: {feature_fusion}"
    raise RuntimeError(message)


def normalize_layer_features(layer: np.ndarray) -> np.ndarray:
    values = layer.astype(np.float32, copy=False)
    norm = np.maximum(np.linalg.norm(values, axis=1, keepdims=True), _EPS)
    return (values / norm).astype(np.float32, copy=False)
