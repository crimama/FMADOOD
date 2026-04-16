"""SuperAD method — multi-layer kNN with coreset selection."""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import faiss
import numpy as np
import tifffile as tiff
import torch
from tqdm import tqdm

from fmad.methods.base import BaseMethod, ObjectResult
from fmad.registry import METHOD_REGISTRY

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.sampler import GreedyCoresetSampler
from src.utils import augment_image, cvt2heatmap, heatmap_on_image, min_max_norm


def _mean_top1p(arr: np.ndarray) -> float:
    k = max(1, int(len(arr) * 0.01))
    return float(np.sort(arr)[-k:].mean())


@METHOD_REGISTRY.register("superad")
class SuperAD(BaseMethod):
    """SuperAD: multi-layer DINOv2 kNN anomaly detection with coreset selection."""

    def __init__(self, backbone, config: dict):
        super().__init__(backbone, config)
        self.shots = config.get("shots", 16)
        self.knn_metric = config.get("knn_metric", "L2_normalized")
        self.k_neighbors = config.get("k_neighbors", 1)
        self.faiss_on_cpu = config.get("faiss_on_cpu", False)
        self.warmup_iters = config.get("warmup_iters", 25)

        # State built during fit()
        self._knn_indices: Dict[str, faiss.Index] = {}
        self._grid_size = None
        self._masking = False
        self._mask_ref_images = config.get("mask_ref_images", False)

    def fit(self, train_image_paths: list, object_name: str, seed: int = 0) -> None:
        """Build memory bank: coreset selection + multi-layer FAISS indices."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Stage 1: CLS features → coreset selection
        cls_features = []
        valid_names = []
        with torch.inference_mode():
            for img_path in tqdm(train_image_paths, desc="Extracting CLS features", leave=False):
                img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                img_tensor, _ = self.backbone.prepare_image(img_rgb)
                cls_feat = self.backbone.extract_cls_features(img_tensor)
                cls_features.append(cls_feat.squeeze().cpu())
                valid_names.append(img_path)

        cls_features = torch.stack(cls_features).to(device)
        sampler = GreedyCoresetSampler(percentage=0.1, device=device, dimension_to_project_features_to=1024)
        selected_indices = sampler.run(cls_features)
        selected_paths = [valid_names[idx] for idx in selected_indices]

        # Stage 2: multi-layer features → FAISS index per layer
        feature_refs: Dict[str, List[np.ndarray]] = {}
        with torch.inference_mode():
            for img_path in tqdm(selected_paths, desc="Extracting reference features", leave=False):
                img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                aug_images = augment_image(img_rgb) if self._rotation else [img_rgb]

                for aug in aug_images:
                    img_tensor, self._grid_size = self.backbone.prepare_image(aug)
                    feats_list = self.backbone.extract_features(img_tensor)
                    mask = self.backbone.compute_background_mask(
                        feats_list[0], self._grid_size, threshold=1,
                        masking_type=(self._mask_ref_images and self._masking)
                    )
                    for idx, feats in enumerate(feats_list):
                        key = f"layer{idx}"
                        if key not in feature_refs:
                            feature_refs[key] = []
                        feature_refs[key].append(feats[mask])

        self._knn_indices = {}
        for layer_name, feats_list in feature_refs.items():
            layer_feats = np.concatenate(feats_list, axis=0).astype("float32")
            if self.knn_metric == "L2_normalized":
                faiss.normalize_L2(layer_feats)

            if self.faiss_on_cpu:
                index = faiss.IndexFlatL2(layer_feats.shape[1])
            else:
                res = faiss.StandardGpuResources()
                index = faiss.GpuIndexFlatIP(res, layer_feats.shape[1])

            index.add(layer_feats)
            self._knn_indices[layer_name] = index

    def predict(self, test_image_path: str) -> Tuple[np.ndarray, float]:
        """Predict anomaly map for a single test image."""
        img_rgb = cv2.cvtColor(cv2.imread(test_image_path), cv2.COLOR_BGR2RGB)
        img_tensor, _ = self.backbone.prepare_image(img_rgb)
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
            dists = 1 - dists  # cosine distance

            dmap = np.zeros_like(mask, dtype=float)
            dmap[mask] = dists.squeeze()
            dmap = dmap.reshape(self._grid_size)
            dmap_resized = cv2.resize(dmap, (img_rgb.shape[1], img_rgb.shape[0]))
            dists_per_layer.append(dmap_resized)

        anomaly_map = np.mean(dists_per_layer, axis=0)
        score = _mean_top1p(anomaly_map.flatten())
        return anomaly_map, score

    def run_object(self, dataset, object_name: str, seed: int,
                   output_dir: str, save_maps: bool = True) -> ObjectResult:
        """Run full SuperAD pipeline for one object."""
        obj_info = dataset.get_object_info(object_name)
        self._masking = obj_info.masking
        self._rotation = obj_info.rotation

        # Set backbone resolution
        self.backbone.set_resolution(obj_info.resolution)

        # Warmup
        train_images = dataset.get_train_images(object_name)
        self.backbone.warmup(train_images[0], iters=self.warmup_iters)

        # Fit
        start = time.time()
        self.fit(train_images, object_name, seed)
        time_memorybank = time.time() - start

        # Predict
        test_images = dataset.get_test_images(object_name)
        anomaly_maps = {}
        anomaly_scores = {}
        inference_times = {}

        for anomaly_type, img_paths in tqdm(test_images.items(), desc=f"Processing {object_name}"):
            if save_maps:
                os.makedirs(os.path.join(output_dir, "anomaly_maps", object_name, "test", anomaly_type), exist_ok=True)
                os.makedirs(os.path.join(output_dir, "anomaly_maps", object_name, "test_hm_on_img", anomaly_type), exist_ok=True)

            for img_path in img_paths:
                t0 = time.time()
                amap, score = self.predict(img_path)
                elapsed = time.time() - t0

                sample_key = f"{anomaly_type}/{os.path.basename(img_path)}"
                anomaly_maps[sample_key] = amap
                anomaly_scores[sample_key] = score
                inference_times[sample_key] = elapsed

                if save_maps:
                    fname = os.path.splitext(os.path.basename(img_path))[0]
                    tiff.imwrite(
                        os.path.join(output_dir, "anomaly_maps", object_name, "test", anomaly_type, f"{fname}.tiff"),
                        amap,
                    )
                    amap_norm = min_max_norm(amap)
                    heatmap = cvt2heatmap(amap_norm * 255)
                    img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                    hm_on_img = heatmap_on_image(heatmap, img_rgb)
                    cv2.imwrite(
                        os.path.join(output_dir, "anomaly_maps", object_name, "test_hm_on_img", anomaly_type, f"{fname}.jpg"),
                        hm_on_img,
                    )

        return ObjectResult(
            object_name=object_name,
            anomaly_maps=anomaly_maps,
            anomaly_scores=anomaly_scores,
            time_memorybank=time_memorybank,
            inference_times=inference_times,
        )
