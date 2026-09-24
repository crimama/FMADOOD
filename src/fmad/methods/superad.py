"""SuperAD multi-layer kNN over the common protocol's fixed support set."""

from typing import Dict, List, Tuple
import json
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch

from fmad.methods.base import BaseMethod
from fmad.protocol import MethodContext
from fmad.registry import METHOD_REGISTRY


def _mean_top1p(arr: np.ndarray) -> float:
    k = max(1, int(len(arr) * 0.01))
    return float(np.sort(arr)[-k:].mean())


def _augment_image(image: np.ndarray) -> list[np.ndarray]:
    center = tuple(np.array(image.shape[1::-1]) / 2)
    return [
        cv2.warpAffine(
            image,
            cv2.getRotationMatrix2D(center, angle, 1.0),
            image.shape[1::-1],
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_DEFAULT,
        )
        for angle in (0, 45, 90, 135, 180, 225, 270, 315)
    ]


@METHOD_REGISTRY.register("superad")
class SuperAD(BaseMethod):
    """Multi-layer DINOv2 kNN anomaly detection over selected support images."""

    def __init__(self, backbone, config: dict):
        super().__init__(backbone, config)
        self.knn_metric = config.get("knn_metric", "L2_normalized")
        self.k_neighbors = config.get("k_neighbors", 1)
        self.faiss_on_cpu = config.get("faiss_on_cpu", False)
        if self.knn_metric not in {"L2_normalized", "L2"}:
            raise ValueError("knn_metric must be L2_normalized or L2")
        if (
            isinstance(self.k_neighbors, bool)
            or not isinstance(self.k_neighbors, (int, np.integer))
            or self.k_neighbors < 1
        ):
            raise ValueError("k_neighbors must be a positive integer")
        self.k_neighbors = int(self.k_neighbors)

        # State built during fit()
        self._knn_indices: Dict[str, faiss.Index] = {}
        self._grid_size = None
        self._masking = False
        self._mask_ref_images = config.get("mask_ref_images", False)
        self._selected_reference_paths: List[str] = []
        self._faiss_resources = []

    def fit(self, support_paths: list[str], context: MethodContext) -> None:
        """Build multi-layer indices from pipeline-selected support images."""
        self._masking, self._rotation = context.masking, context.rotation
        self._selected_reference_paths = list(support_paths)
        if not support_paths:
            raise ValueError("support paths must not be empty")

        # Stage 2: multi-layer features → FAISS index per layer
        feature_refs: Dict[str, List[np.ndarray]] = {}
        with torch.inference_mode():
            for step, img_path in enumerate(support_paths, start=1):
                img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                aug_images = _augment_image(img_rgb) if self._rotation else [img_rgb]

                for aug in aug_images:
                    img_tensor, self._grid_size = self.backbone.prepare_image(aug)
                    feats_list = self.backbone.extract_features(img_tensor)
                    mask = self.backbone.compute_background_mask(
                        feats_list[0],
                        self._grid_size,
                        threshold=1,
                        masking_type=(self._mask_ref_images and self._masking),
                    )
                    for idx, feats in enumerate(feats_list):
                        key = f"layer{idx}"
                        if key not in feature_refs:
                            feature_refs[key] = []
                        feature_refs[key].append(feats[mask])
                self.log_training(
                    step=step, stage="reference_features", support_total=len(support_paths)
                )

        self._knn_indices = {}
        self._faiss_resources = []
        for layer_name, feats_list in feature_refs.items():
            layer_feats = np.concatenate(feats_list, axis=0).astype("float32")
            if layer_feats.shape[0] < self.k_neighbors:
                raise ValueError("reference memory has fewer features than k_neighbors")
            if not np.isfinite(layer_feats).all():
                raise ValueError("reference features must be finite")
            if self.knn_metric == "L2_normalized":
                faiss.normalize_L2(layer_feats)

            use_gpu = not self.faiss_on_cpu and str(
                getattr(self.backbone, "device", "cpu")
            ).startswith("cuda")
            if not use_gpu:
                index = (
                    faiss.IndexFlatIP(layer_feats.shape[1])
                    if self.knn_metric == "L2_normalized"
                    else faiss.IndexFlatL2(layer_feats.shape[1])
                )
            else:
                if not torch.cuda.is_available() or not hasattr(faiss, "StandardGpuResources"):
                    raise RuntimeError(
                        "CUDA FAISS requested but unavailable; set faiss_on_cpu=true for CPU search"
                    )
                res = faiss.StandardGpuResources()
                self._faiss_resources.append(res)
                gpu_config = faiss.GpuIndexFlatConfig()
                device_index = torch.device(self.backbone.device).index
                gpu_config.device = (
                    torch.cuda.current_device() if device_index is None else device_index
                )
                index = (
                    faiss.GpuIndexFlatIP(res, layer_feats.shape[1], gpu_config)
                    if self.knn_metric == "L2_normalized"
                    else faiss.GpuIndexFlatL2(res, layer_feats.shape[1], gpu_config)
                )

            index.add(layer_feats)
            self._knn_indices[layer_name] = index
        self.log_training(
            stage="memory_bank_ready",
            reference_vectors={
                layer: int(index.ntotal) for layer, index in self._knn_indices.items()
            },
        )

    def save_checkpoint(self, directory: Path) -> None:
        """Persist fitted memory, not duplicate frozen pretrained backbone weights."""
        if not self._knn_indices:
            raise RuntimeError("fit must complete before checkpoint saving")
        for layer, index in self._knn_indices.items():
            cpu_index = faiss.index_gpu_to_cpu(index) if self._faiss_resources else index
            faiss.write_index(cpu_index, str(directory / f"{layer}.faiss"))
        metadata = {
            "schema_version": 1,
            "kind": "faiss_memory_bank",
            "method_class": type(self).__name__,
            "knn_metric": self.knn_metric,
            "k_neighbors": self.k_neighbors,
            "masking": self._masking,
            "rotation": self._rotation,
            "feature_layers": list(self.backbone.feature_layers),
            "selected_reference_paths": self._selected_reference_paths,
            "backbone": {
                "model_name": self.backbone.model_name,
                "provenance": getattr(self.backbone, "provenance", None),
            },
            "indices": {layer: f"{layer}.faiss" for layer in self._knn_indices},
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    def predict(self, test_image_path: str) -> Tuple[np.ndarray, float]:
        """Predict anomaly map for a single test image."""
        if not self._knn_indices:
            raise RuntimeError("fit must complete before prediction")
        img_rgb = cv2.cvtColor(cv2.imread(test_image_path), cv2.COLOR_BGR2RGB)
        img_tensor, query_grid = self.backbone.prepare_image(img_rgb)
        self._grid_size = query_grid
        feats_list = self.backbone.extract_features(img_tensor)

        mask = self.backbone.compute_background_mask(
            feats_list[0], self._grid_size, threshold=1, masking_type=self._masking
        )

        dists_per_layer = []
        for num, feats in enumerate(feats_list):
            masked_feats = feats[mask].astype("float32")
            if self.knn_metric == "L2_normalized":
                faiss.normalize_L2(masked_feats)

            dists, _ = self._knn_indices[f"layer{num}"].search(masked_feats, k=self.k_neighbors)
            if self.k_neighbors > 1:
                dists = dists.mean(axis=1)
            if self.knn_metric == "L2_normalized":
                dists = 1 - dists  # cosine distance

            dmap = np.zeros_like(mask, dtype=float)
            dmap[mask] = dists.squeeze()
            dmap = dmap.reshape(self._grid_size)
            dmap_resized = cv2.resize(dmap, (img_rgb.shape[1], img_rgb.shape[0]))
            dists_per_layer.append(dmap_resized)

        anomaly_map = np.mean(dists_per_layer, axis=0)
        score = _mean_top1p(anomaly_map.flatten())
        return anomaly_map, score

    def object_metadata(self) -> Dict:
        """Return model provenance without GT or evaluator-owned fields."""
        return {
            "selected_reference_count": len(self._selected_reference_paths),
            "selected_reference_paths": self._selected_reference_paths,
            "feature_layers": list(self.backbone.feature_layers),
        }


@METHOD_REGISTRY.register("rn_fmlk")
class RNFMLK(SuperAD):
    """Final-layer local kNN hard baseline over the same fixed references."""

    def __init__(self, backbone, config: dict):
        if len(backbone.feature_layers) != 1:
            raise ValueError("RN-FMLK requires exactly one final backbone feature layer")
        super().__init__(backbone, config)
