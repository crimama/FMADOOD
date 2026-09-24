"""PatchCore baseline method for few-shot anomaly detection.

Implements the standard PatchCore (CVPR 2022) memory-bank nearest-neighbor
anomaly detection and localization. This implementation is completely isolated
from other methods and strictly adheres to the BaseMethod interface.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
from pathlib import Path
import time

import cv2
import faiss
import numpy as np
import torch

from fmad.methods.base import BaseMethod, Prediction
from fmad.protocol import MethodContext
from fmad.registry import METHOD_REGISTRY


def _greedy_coreset_sampling(
    features: np.ndarray,
    sampling_ratio: float,
    seed: int = 0,
) -> np.ndarray:
    """Greedy k-center coreset subsampling for memory bank reduction.

    Args:
        features: (N, D) float32 array.
        sampling_ratio: Fraction of features to retain in (0.0, 1.0].
        seed: Random seed for starting center.

    Returns:
        Subsampled (M, D) float32 array where M = int(N * sampling_ratio).
    """
    n_samples = len(features)
    target_count = max(1, int(n_samples * sampling_ratio))
    if target_count >= n_samples:
        return features

    rng = np.random.default_rng(seed)
    selected_indices = [int(rng.integers(0, n_samples))]

    # Distances to the closest selected center
    min_distances = np.linalg.norm(features - features[selected_indices[0]], axis=1)

    for _ in range(1, target_count):
        farthest_idx = int(np.argmax(min_distances))
        selected_indices.append(farthest_idx)
        new_dists = np.linalg.norm(features - features[farthest_idx], axis=1)
        min_distances = np.minimum(min_distances, new_dists)

    return features[selected_indices]


@METHOD_REGISTRY.register("patchcore")
class PatchCore(BaseMethod):
    """PatchCore: Towards Total Recall in Industrial Anomaly Detection (CVPR 2022).

    Extracts patch-level feature embeddings from a frozen backbone, constructs
    a coreset memory bank from the support set, and computes anomaly maps via
    k-nearest neighbor distances.
    """

    def __init__(self, backbone, config: dict):
        super().__init__(backbone, config)
        self.coreset_sampling_ratio = float(config.get("coreset_sampling_ratio", 1.0))
        self.k_neighbors = int(config.get("k_neighbors", 1))
        self.knn_metric = str(config.get("knn_metric", "L2_normalized"))
        self.faiss_on_cpu = bool(config.get("faiss_on_cpu", False))
        self.top_k_percent = float(config.get("top_k_percent", 0.01))

        if self.knn_metric not in {"L2_normalized", "L2"}:
            raise ValueError("knn_metric must be L2_normalized or L2")
        if self.k_neighbors < 1:
            raise ValueError("k_neighbors must be a positive integer")

        # Memory bank state
        self._memory_bank: np.ndarray | None = None
        self._faiss_index: faiss.Index | None = None
        self._grid_size: Tuple[int, int] | None = None
        self._selected_support_paths: List[str] = []
        self._faiss_resources: list = []

    def fit(self, support_paths: list[str], context: MethodContext) -> None:
        """Construct memory bank from support images without rotation augmentation."""
        if not support_paths:
            raise ValueError("support_paths must not be empty")

        self._selected_support_paths = list(support_paths)
        masking = context.masking

        patch_features_list: List[np.ndarray] = []

        with torch.inference_mode():
            for step, img_path in enumerate(support_paths, start=1):
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    raise FileNotFoundError(f"Cannot read support image: {img_path}")
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                img_tensor, self._grid_size = self.backbone.prepare_image(img_rgb)
                feats_list = self.backbone.extract_features(img_tensor)

                # Concatenate or average features across selected backbone layers
                # Standard PatchCore aggregates mid-to-high level representations
                if len(feats_list) == 1:
                    patch_feats = feats_list[0]
                else:
                    patch_feats = np.concatenate(feats_list, axis=-1)

                if masking:
                    mask = self.backbone.compute_background_mask(
                        feats_list[0],
                        self._grid_size,
                        threshold=1,
                        masking_type=True,
                    )
                    patch_feats = patch_feats[mask]

                patch_features_list.append(patch_feats)
                self.log_training(
                    step=step,
                    stage="patchcore_feature_extraction",
                    support_total=len(support_paths),
                )

        # Concatenate all support patch features
        all_features = np.concatenate(patch_features_list, axis=0).astype("float32")
        if not np.isfinite(all_features).all():
            raise ValueError("Extracted patch features must be finite")

        # Coreset subsampling if configured < 1.0
        if self.coreset_sampling_ratio < 1.0:
            all_features = _greedy_coreset_sampling(
                all_features,
                sampling_ratio=self.coreset_sampling_ratio,
                seed=context.seed,
            )

        self._memory_bank = all_features

        if self.knn_metric == "L2_normalized":
            faiss.normalize_L2(self._memory_bank)

        # Build FAISS Index
        feature_dim = self._memory_bank.shape[1]
        use_gpu = not self.faiss_on_cpu and str(
            getattr(self.backbone, "device", "cpu")
        ).startswith("cuda")

        self._faiss_resources = []
        if not use_gpu:
            index = (
                faiss.IndexFlatIP(feature_dim)
                if self.knn_metric == "L2_normalized"
                else faiss.IndexFlatL2(feature_dim)
            )
        else:
            if not torch.cuda.is_available() or not hasattr(faiss, "StandardGpuResources"):
                index = (
                    faiss.IndexFlatIP(feature_dim)
                    if self.knn_metric == "L2_normalized"
                    else faiss.IndexFlatL2(feature_dim)
                )
            else:
                res = faiss.StandardGpuResources()
                self._faiss_resources.append(res)
                gpu_config = faiss.GpuIndexFlatConfig()
                device_index = torch.device(self.backbone.device).index
                gpu_config.device = (
                    torch.cuda.current_device() if device_index is None else device_index
                )
                index = (
                    faiss.GpuIndexFlatIP(res, feature_dim, gpu_config)
                    if self.knn_metric == "L2_normalized"
                    else faiss.GpuIndexFlatL2(res, feature_dim, gpu_config)
                )

        index.add(self._memory_bank)
        self._faiss_index = index

        self.log_training(
            stage="patchcore_memory_bank_ready",
            total_patches=int(self._faiss_index.ntotal),
            feature_dim=feature_dim,
        )

    def predict(self, test_image_path: str) -> tuple[np.ndarray, float]:
        """Compute anomaly map and score for a single query image."""
        if self._faiss_index is None or self._grid_size is None:
            raise RuntimeError("PatchCore fit() must be called before predict()")

        img_bgr = cv2.imread(test_image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read query image: {test_image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        with torch.inference_mode():
            img_tensor, grid_size = self.backbone.prepare_image(img_rgb)
            feats_list = self.backbone.extract_features(img_tensor)

            if len(feats_list) == 1:
                query_feats = feats_list[0]
            else:
                query_feats = np.concatenate(feats_list, axis=-1)

            query_feats = query_feats.astype("float32")
            if self.knn_metric == "L2_normalized":
                faiss.normalize_L2(query_feats)

            # Query kNN
            dists, _ = self._faiss_index.search(query_feats, self.k_neighbors)
            if self.k_neighbors > 1:
                dists = dists.mean(axis=1)

            if self.knn_metric == "L2_normalized":
                dists = 1.0 - dists  # Cosine distance

        # Reshape to spatial grid and interpolate to original resolution
        dmap = dists.reshape(grid_size)
        anomaly_map = cv2.resize(dmap, (w, h), interpolation=cv2.INTER_LINEAR)

        # Image-level score: top-1% average or maximum
        k_top = max(1, int(len(dists) * self.top_k_percent))
        image_score = float(np.sort(dists.flatten())[-k_top:].mean())

        return anomaly_map, image_score

    def predict_batch(self, test_image_paths: Sequence[str]) -> list[Prediction]:
        predictions: list[Prediction] = []
        for path in test_image_paths:
            started = time.perf_counter()
            score_map, image_score = self.predict(path)
            predictions.append(
                Prediction(
                    score_map=score_map,
                    image_score=image_score,
                    inference_seconds=time.perf_counter() - started,
                )
            )
        return predictions

    def object_metadata(self) -> Dict:
        return {
            "method": "patchcore",
            "coreset_sampling_ratio": self.coreset_sampling_ratio,
            "k_neighbors": self.k_neighbors,
            "knn_metric": self.knn_metric,
            "memory_bank_size": int(self._faiss_index.ntotal) if self._faiss_index else 0,
            "selected_support_count": len(self._selected_support_paths),
        }
