"""Independent paper-faithful adapter for Retrieval-based Anomaly Detection.

The implementation follows the RAD paper's frozen multi-layer feature memory
and the equations documented by the authors. It does not import or copy code
from the unlicensed RAD repository. Dataset selection, query enumeration, and
streaming evaluation remain owned by the shared pipeline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from fmad.methods.base import BaseMethod, Prediction
from fmad.protocol import MethodContext
from fmad.registry import METHOD_REGISTRY


_RAD_PAPER = "https://arxiv.org/abs/2601.22763v3"
_RAD_REPOSITORY = "https://github.com/longkukuhi/RAD"
_RAD_COMMIT = "59847f4f55082c31649d4e0a2dd0ec329b932633"
_RAD_LAYERS = (3, 6, 9, 11)
_RAD_OBJECTS = (
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
)


@METHOD_REGISTRY.register("rad")
class RAD(BaseMethod):
    """Multi-category RAD memory with the shared MVTec AD 2 query/evaluator.

    Every object fit call uses the same all-category support manifest. The
    common runner passes only the current object's support paths, so this
    adapter verifies that subset and loads the remaining manifest entries
    directly. It never enumerates train directories or reads labels.
    """

    def __init__(self, backbone, config: dict):
        super().__init__(backbone, config)
        if getattr(backbone, "model_name", None) != "dinov3_vitb16":
            raise ValueError("RAD requires the native DINOv3 ViT-B/16 backbone")
        if tuple(getattr(backbone, "feature_layers", ())) != _RAD_LAYERS:
            raise ValueError("RAD requires normalized DINOv3 block features [3, 6, 9, 11]")
        if not bool(getattr(backbone, "normalize_intermediate_features", False)):
            raise ValueError("RAD requires final-layer normalization on every selected block")

        manifest = config.get("support_manifest")
        if not isinstance(manifest, str) or not manifest:
            raise ValueError("method_config.support_manifest is required for RAD's global bank")
        self._manifest_path = Path(manifest)
        self._manifest_bytes = self._manifest_path.read_bytes()
        self._manifest_digest = hashlib.sha256(self._manifest_bytes).hexdigest()
        payload = json.loads(self._manifest_bytes)
        self._objects = tuple(config.get("support_objects", _RAD_OBJECTS))
        if (not self._objects or len(set(self._objects)) != len(self._objects)
                or any(not isinstance(name, str) or not name for name in self._objects)):
            raise ValueError("RAD support_objects must be unique non-empty object names")
        if not isinstance(payload, dict) or set(payload) != set(self._objects):
            raise ValueError("RAD's support manifest must contain exactly the declared support_objects")
        self._manifest = payload

        self.input_size = int(config.get("input_size", 512))
        self.crop_size = int(config.get("crop_size", 448))
        self.map_size = int(config.get("map_size", 256))
        self.image_neighbors = int(config.get("image_neighbors", 150))
        self.position_radius = int(config.get("position_radius", 1))
        self.patch_chunk_size = int(config.get("patch_chunk_size", 64))
        self.layer_weights = tuple(float(x) for x in config.get("layer_weights", (0.25,) * 4))
        if self.input_size <= 0 or self.crop_size <= 0 or self.crop_size > self.input_size:
            raise ValueError("RAD input_size/crop_size must be positive, with crop_size <= input_size")
        if self.input_size % 16 or self.crop_size % 16:
            raise ValueError("RAD's square input and crop sizes must be multiples of the ViT patch size")
        if self.map_size <= 0 or self.image_neighbors <= 0 or self.patch_chunk_size <= 0:
            raise ValueError("RAD map size, image-neighbor count, and patch chunk must be positive")
        if self.position_radius < 0:
            raise ValueError("RAD position_radius must be non-negative")
        if len(self.layer_weights) != len(_RAD_LAYERS) or any(w < 0 for w in self.layer_weights):
            raise ValueError("RAD requires four non-negative layer weights")
        total_weight = sum(self.layer_weights)
        if total_weight <= 0:
            raise ValueError("RAD layer weights must sum to a positive value")
        self.layer_weights = tuple(w / total_weight for w in self.layer_weights)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self._image_transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self._bank_key = None
        self._cls_bank = None
        self._patch_banks = None
        self._grid_size = (self.crop_size // 16, self.crop_size // 16)
        self._global_support_counts = {}
        self._effective_image_neighbors = 0
        self._current_object = None
        self._bank_reused = False

    def _data_root(self, support_paths: Sequence[str], object_name: str) -> Path:
        if not support_paths:
            raise ValueError("RAD needs selected normal support images")
        first = Path(support_paths[0]).resolve()
        if first.parts[-4:-1] != (object_name, "train", "good"):
            raise ValueError(f"Unexpected normal support path for {object_name}: {first}")
        return first.parents[3]

    def _manifest_paths(self, data_root: Path) -> dict[str, list[Path]]:
        resolved: dict[str, list[Path]] = {}
        for object_name in self._objects:
            entries = self._manifest[object_name]
            if not isinstance(entries, list) or not entries:
                raise ValueError(f"RAD support manifest has no entries for {object_name}")
            paths = []
            for value in entries:
                listed = Path(str(value))
                if listed.parts[-4:-1] != (object_name, "train", "good"):
                    raise ValueError(f"Manifest path is not normal {object_name} support: {value}")
                path = data_root / object_name / "train" / "good" / listed.name
                if not path.is_file():
                    raise FileNotFoundError(f"Manifest support image is unavailable: {path}")
                paths.append(path.resolve())
            if len({p.name for p in paths}) != len(paths):
                raise ValueError(f"RAD support manifest contains duplicate {object_name} images")
            resolved[object_name] = paths
        return resolved

    def _preprocess(self, image_path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            original_shape = (image.height, image.width)
            tensor = self._image_transform(image)
        return tensor, original_shape

    def _extract_one(self, image_path: Path, device: torch.device):
        image_tensor, original_shape = self._preprocess(image_path)
        patch_features = self.backbone.extract_features(image_tensor)
        if len(patch_features) != len(_RAD_LAYERS):
            raise ValueError("DINOv3 returned a different number of feature layers than RAD expects")
        patches = []
        for layer, values in zip(_RAD_LAYERS, patch_features, strict=True):
            feature = torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device)
            if feature.ndim != 2 or feature.shape[0] != self._grid_size[0] * self._grid_size[1]:
                raise ValueError(f"RAD layer {layer} does not match the configured square patch grid")
            if not torch.isfinite(feature).all():
                raise ValueError(f"RAD layer {layer} contains non-finite support/query features")
            patches.append(F.normalize(feature, dim=-1))

        cls_feature = torch.as_tensor(
            self.backbone.extract_cls_features(image_tensor),
            dtype=torch.float32,
            device=device,
        ).reshape(-1)
        if cls_feature.numel() != patches[-1].shape[-1] or not torch.isfinite(cls_feature).all():
            raise ValueError("RAD requires a finite final-layer CLS feature matching ViT-B/16 width")
        return F.normalize(cls_feature, dim=0), patches, original_shape

    def fit(self, support_paths: list[str], context: MethodContext) -> None:
        if context.object_name not in self._objects:
            raise ValueError(f"Unsupported RAD object: {context.object_name}")
        if context.masking or context.rotation:
            raise ValueError("RAD public-v13 uses the declared no-mask, no-rotation protocol")

        data_root = self._data_root(support_paths, context.object_name)
        paths_by_object = self._manifest_paths(data_root)
        expected = paths_by_object[context.object_name]
        provided = [Path(path).resolve() for path in support_paths]
        if provided != expected:
            raise ValueError(
                f"{context.object_name}: pipeline-selected support paths differ from RAD's global manifest"
            )
        shot_counts = {len(paths) for paths in paths_by_object.values()}
        if len(shot_counts) != 1:
            raise ValueError("RAD's cross-category memory requires the same shot count per object")

        device = torch.device(self.backbone.device)
        key = (
            self._manifest_digest,
            str(data_root),
            self.backbone.model_name,
            self.backbone.model_id,
            self._grid_size,
            self.input_size,
            self.crop_size,
        )
        self._current_object = context.object_name
        self._bank_reused = key == self._bank_key and self._cls_bank is not None
        if self._bank_reused:
            self.log_training(stage="rad_global_bank_reused", support_total=sum(shot_counts) * len(self._objects))
            return

        cls_features = []
        layer_features = [[] for _ in _RAD_LAYERS]
        counts = {}
        with torch.inference_mode():
            step = 0
            for object_name in self._objects:
                counts[object_name] = len(paths_by_object[object_name])
                for image_path in paths_by_object[object_name]:
                    cls_feature, patches, _ = self._extract_one(image_path, device)
                    cls_features.append(cls_feature)
                    for index, feature in enumerate(patches):
                        layer_features[index].append(feature)
                    step += 1
                    self.log_training(
                        step=step,
                        stage="rad_global_reference_features",
                        support_total=sum(shot_counts) * len(self._objects),
                    )

        self._cls_bank = torch.stack(cls_features, dim=0)
        self._patch_banks = [torch.stack(features, dim=0) for features in layer_features]
        self._global_support_counts = counts
        self._effective_image_neighbors = min(self.image_neighbors, len(cls_features))
        self._bank_key = key
        self.log_training(
            stage="rad_global_memory_ready",
            support_images=len(cls_features),
            image_neighbors=self._effective_image_neighbors,
            patches_per_image=self._grid_size[0] * self._grid_size[1],
        )

    def _position_indices(self, device: torch.device) -> torch.Tensor:
        height, width = self._grid_size
        rows = torch.arange(height, device=device).repeat_interleave(width)
        cols = torch.arange(width, device=device).repeat(height)
        offsets = torch.arange(-self.position_radius, self.position_radius + 1, device=device)
        rr = (rows[:, None, None] + offsets[None, :, None]).clamp(0, height - 1)
        cc = (cols[:, None, None] + offsets[None, None, :]).clamp(0, width - 1)
        return (rr * width + cc).reshape(rows.numel(), -1)

    def _patch_distances(
        self,
        query_features: torch.Tensor,
        bank: torch.Tensor,
        selected_images: torch.Tensor,
        position_indices: torch.Tensor,
    ) -> torch.Tensor:
        query_count = query_features.shape[0]
        output = torch.empty(query_count, dtype=query_features.dtype, device=query_features.device)
        bank = bank.index_select(0, selected_images)
        for start in range(0, query_count, self.patch_chunk_size):
            end = min(start + self.patch_chunk_size, query_count)
            positions = position_indices[start:end]
            # Shape: [retrieved image, query patch, nearby patch, channel].
            candidates = bank[:, positions.reshape(-1), :].reshape(
                bank.shape[0], end - start, positions.shape[1], bank.shape[-1]
            )
            similarities = torch.einsum(
                "pd,kpnd->kpn", query_features[start:end], candidates
            )
            output[start:end] = 1.0 - torch.amax(similarities, dim=(0, 2))
        return output

    @staticmethod
    def _gaussian_kernel(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        coords = torch.arange(5, dtype=dtype, device=device)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        distance = (grid_x - 2.0).square() + (grid_y - 2.0).square()
        kernel = torch.exp(-distance / 2.0)
        kernel /= kernel.sum()
        return kernel.reshape(1, 1, 5, 5)

    def _map_to_query_frame(self, patch_scores: torch.Tensor, original_shape: tuple[int, int]):
        device = patch_scores.device
        score_map = F.interpolate(
            patch_scores.reshape(1, 1, *self._grid_size),
            size=(self.map_size, self.map_size),
            mode="bilinear",
            align_corners=False,
        )
        score_map = F.conv2d(score_map, self._gaussian_kernel(device, score_map.dtype), padding=2)

        # Undo the official square resize/center crop geometrically, leaving
        # scores outside the sampled center field at the neutral value zero.
        crop_map = F.interpolate(
            score_map,
            size=(self.crop_size, self.crop_size),
            mode="bilinear",
            align_corners=False,
        )
        square = torch.zeros(
            (1, 1, self.input_size, self.input_size), dtype=crop_map.dtype, device=device
        )
        offset = (self.input_size - self.crop_size) // 2
        square[:, :, offset : offset + self.crop_size, offset : offset + self.crop_size] = crop_map
        query_frame = F.interpolate(
            square,
            size=original_shape,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return query_frame

    def predict_batch(self, test_image_paths: Sequence[str]) -> list[Prediction]:
        if self._cls_bank is None or self._patch_banks is None:
            raise RuntimeError("RAD fit() must build its all-category normal support memory first")
        outputs = []
        device = self._cls_bank.device
        position_indices = self._position_indices(device)
        with torch.inference_mode():
            for image_path in test_image_paths:
                started = time.perf_counter()
                cls_feature, patches, original_shape = self._extract_one(Path(image_path), device)
                image_similarity = self._cls_bank @ cls_feature
                neighbors = torch.topk(
                    image_similarity,
                    k=self._effective_image_neighbors,
                    largest=True,
                    sorted=False,
                ).indices

                layer_maps = [
                    self._patch_distances(query, bank, neighbors, position_indices)
                    for query, bank in zip(patches, self._patch_banks, strict=True)
                ]
                patch_scores = torch.stack(layer_maps, dim=0)
                weights = torch.as_tensor(self.layer_weights, dtype=patch_scores.dtype, device=device)
                patch_scores = (patch_scores * weights[:, None]).sum(dim=0)
                score_map = self._map_to_query_frame(patch_scores, original_shape)
                image_score = float(torch.max(score_map).item())
                outputs.append(
                    Prediction(
                        score_map=score_map.float().cpu().numpy(),
                        image_score=image_score,
                        inference_seconds=time.perf_counter() - started,
                        diagnostics={
                            "rad_effective_image_neighbors": self._effective_image_neighbors,
                            "rad_bank_images": int(self._cls_bank.shape[0]),
                        },
                    )
                )
        return outputs

    def object_metadata(self) -> dict:
        return {
            "method": "rad",
            "paper": _RAD_PAPER,
            "official_repository": _RAD_REPOSITORY,
            "official_commit": _RAD_COMMIT,
            "official_repository_license": "no license file or license grant found at pinned commit",
            "implementation_status": "independent paper-faithful reimplementation; no repository code copied",
            "support_manifest_sha256": self._manifest_digest,
            "global_support_counts": self._global_support_counts,
            "global_support_total": sum(self._global_support_counts.values()),
            "support_manifest_cross_category_memory": True,
            "feature_layers_zero_based": list(_RAD_LAYERS),
            "normalized_intermediate_features": True,
            "position_radius": self.position_radius,
            "configured_image_neighbors": self.image_neighbors,
            "effective_image_neighbors": self._effective_image_neighbors,
            "layer_weights": list(self.layer_weights),
            "preprocess": {
                "square_resize": self.input_size,
                "center_crop": self.crop_size,
                "map_resize_before_shared_evaluation": self.map_size,
                "query_frame": "center-crop predictions mapped back; unobserved margins score zero",
            },
            "image_score": "maximum of final query-frame map per public-v13 contract",
            "bank_reused_for_object": self._bank_reused,
            "backbone_model_id": getattr(self.backbone, "model_id", None),
        }
