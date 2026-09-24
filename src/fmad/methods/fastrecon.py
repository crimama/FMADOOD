"""FastRecon: Few-shot Industrial Anomaly Detection via Fast Feature Reconstruction (ICCV 2023).

Reconstructs query patch features as a regularized linear combination of their
nearest neighbors in the support memory bank using a closed-form ridge solver:
    w* = (S^T S + lambda * I)^{-1} S^T q
Anomaly score is the reconstruction residual ||q - S w*||_2.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import time

import cv2
import faiss
import numpy as np
import torch

from fmad.methods.base import BaseMethod, Prediction
from fmad.protocol import MethodContext
from fmad.registry import METHOD_REGISTRY


@METHOD_REGISTRY.register("fastrecon")
class FastRecon(BaseMethod):
    """FastRecon (ICCV 2023) baseline for few-shot anomaly detection."""

    def __init__(self, backbone, config: dict):
        super().__init__(backbone, config)
        self.k_neighbors = int(config.get("k_neighbors", 10))
        self.reg_lambda = float(config.get("reg_lambda", 0.01))
        self.knn_metric = str(config.get("knn_metric", "L2_normalized"))
        self.faiss_on_cpu = bool(config.get("faiss_on_cpu", False))
        self.top_k_percent = float(config.get("top_k_percent", 0.01))

        self._memory_bank: torch.Tensor | None = None
        self._faiss_index: faiss.Index | None = None
        self._grid_size: Tuple[int, int] | None = None
        self._selected_support_paths: List[str] = []
        self._faiss_resources: list = []
        self.device = torch.device(
            getattr(backbone, "device", "cuda" if torch.cuda.is_available() else "cpu")
        )

    def fit(self, support_paths: list[str], context: MethodContext) -> None:
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
                    stage="fastrecon_support_features",
                    support_total=len(support_paths),
                )

        all_features = np.concatenate(patch_features_list, axis=0).astype("float32")
        self._memory_bank = torch.from_numpy(all_features).to(self.device)

        # Build FAISS Index for nearest neighbor retrieval (GPU accelerated if available)
        search_features = all_features.copy()
        if self.knn_metric == "L2_normalized":
            faiss.normalize_L2(search_features)

        dim = search_features.shape[1]
        use_gpu = not self.faiss_on_cpu and str(self.device).startswith("cuda")
        self._faiss_resources = []

        if not use_gpu or not torch.cuda.is_available() or not hasattr(faiss, "StandardGpuResources"):
            index = (
                faiss.IndexFlatIP(dim)
                if self.knn_metric == "L2_normalized"
                else faiss.IndexFlatL2(dim)
            )
        else:
            res = faiss.StandardGpuResources()
            self._faiss_resources.append(res)
            gpu_config = faiss.GpuIndexFlatConfig()
            device_index = torch.device(self.device).index
            gpu_config.device = (
                torch.cuda.current_device() if device_index is None else device_index
            )
            index = (
                faiss.GpuIndexFlatIP(res, dim, gpu_config)
                if self.knn_metric == "L2_normalized"
                else faiss.GpuIndexFlatL2(res, dim, gpu_config)
            )

        index.add(search_features)
        self._faiss_index = index

    def predict(self, test_image_path: str) -> tuple[np.ndarray, float]:
        if self._faiss_index is None or self._grid_size is None or self._memory_bank is None:
            raise RuntimeError("FastRecon fit() must be called before predict()")

        img_bgr = cv2.imread(test_image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read query image: {test_image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        with torch.inference_mode():
            img_tensor, grid_size = self.backbone.prepare_image(img_rgb)
            feats_list = self.backbone.extract_features(img_tensor)

            if len(feats_list) == 1:
                query_feats_np = feats_list[0].astype("float32")
            else:
                query_feats_np = np.concatenate(feats_list, axis=-1).astype("float32")

            search_query = query_feats_np.copy()
            if self.knn_metric == "L2_normalized":
                faiss.normalize_L2(search_query)

            # Retrieve top-K neighbors from memory bank for each query patch
            k = min(self.k_neighbors, self._faiss_index.ntotal)
            _, indices = self._faiss_index.search(search_query, k)  # (N_query, K)

            q_tensor = torch.from_numpy(query_feats_np).to(self.device)  # (N, D)
            idx_tensor = torch.from_numpy(indices).to(self.device)        # (N, K)

            # S: (N, K, D)
            S = self._memory_bank[idx_tensor]

            # Fast Batched Ridge Reconstruction:
            # S: (N, K, D), S_T: (N, D, K)
            # A = S @ S^T + lambda * I -> (N, K, K)
            # b = S @ q -> (N, K, 1)
            # w = A^{-1} b -> (N, K, 1)
            # q_recon = S^T w -> (N, D)
            # residual = ||q - q_recon||_2
            S_T = S.transpose(1, 2)  # (N, D, K)
            A = torch.bmm(S, S_T)    # (N, K, K)
            eye = torch.eye(k, device=self.device).unsqueeze(0)
            A = A + self.reg_lambda * eye

            b = torch.bmm(S, q_tensor.unsqueeze(-1))  # (N, K, 1)

            # Solve A w = b
            w_coeff = torch.linalg.solve(A, b)        # (N, K, 1)

            q_recon = torch.bmm(S_T, w_coeff).squeeze(-1)  # (N, D)
            residuals = torch.norm(q_tensor - q_recon, dim=-1).cpu().numpy()  # (N,)

        # Reshape to 2D spatial anomaly map
        dmap = residuals.reshape(grid_size)
        anomaly_map = cv2.resize(dmap, (w, h), interpolation=cv2.INTER_LINEAR)

        k_top = max(1, int(len(residuals) * self.top_k_percent))
        image_score = float(np.sort(residuals)[-k_top:].mean())

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
            "method": "fastrecon",
            "k_neighbors": self.k_neighbors,
            "reg_lambda": self.reg_lambda,
            "selected_support_count": len(self._selected_support_paths),
        }
