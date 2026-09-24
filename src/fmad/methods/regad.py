"""RegAD: Registration Based Few-Shot Anomaly Detection (ECCV 2022 Oral).

Alings query patch features with the best matching support image features via
dense spatial correspondence / registration, and computes pixel anomaly scores
from the feature discrepancy between the aligned pair.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from fmad.methods.base import BaseMethod, Prediction
from fmad.protocol import MethodContext
from fmad.registry import METHOD_REGISTRY


@METHOD_REGISTRY.register("regad")
class RegAD(BaseMethod):
    """Registration-based Few-Shot Anomaly Detection (ECCV 2022)."""

    def __init__(self, backbone, config: dict):
        super().__init__(backbone, config)
        self.top_k_percent = float(config.get("top_k_percent", 0.01))
        self.smooth_sigma = float(config.get("smooth_sigma", 4.0))

        self._support_feature_maps: List[torch.Tensor] = []
        self._support_global_embeddings: List[torch.Tensor] = []
        self._grid_size: Tuple[int, int] | None = None
        self._selected_support_paths: List[str] = []
        self.device = torch.device(
            getattr(backbone, "device", "cuda" if torch.cuda.is_available() else "cpu")
        )

    def fit(self, support_paths: list[str], context: MethodContext) -> None:
        if not support_paths:
            raise ValueError("support_paths must not be empty")

        self._selected_support_paths = list(support_paths)
        self._support_feature_maps = []
        self._support_global_embeddings = []

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

                h_g, w_g = self._grid_size
                dim = patch_feats.shape[-1]
                # (1, C, H, W)
                feat_map = torch.from_numpy(patch_feats.reshape(h_g, w_g, dim)).permute(2, 0, 1).unsqueeze(0).to(self.device).float()
                feat_map = F.normalize(feat_map, p=2, dim=1)

                global_emb = feat_map.mean(dim=(-2, -1), keepdim=True)
                global_emb = F.normalize(global_emb, p=2, dim=1)

                self._support_feature_maps.append(feat_map)
                self._support_global_embeddings.append(global_emb)

                self.log_training(
                    step=step,
                    stage="regad_support_registration_bank",
                    support_total=len(support_paths),
                )

    def predict(self, test_image_path: str) -> tuple[np.ndarray, float]:
        if not self._support_feature_maps or self._grid_size is None:
            raise RuntimeError("RegAD fit() must be called before predict()")

        img_bgr = cv2.imread(test_image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read query image: {test_image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        with torch.inference_mode():
            img_tensor, grid_size = self.backbone.prepare_image(img_rgb)
            feats_list = self.backbone.extract_features(img_tensor)

            if len(feats_list) == 1:
                patch_feats = feats_list[0]
            else:
                patch_feats = np.concatenate(feats_list, axis=-1)

            h_g, w_g = grid_size
            dim = patch_feats.shape[-1]
            # Query feature map: (1, C, H, W)
            query_map = torch.from_numpy(patch_feats.reshape(h_g, w_g, dim)).permute(2, 0, 1).unsqueeze(0).to(self.device).float()
            query_map = F.normalize(query_map, p=2, dim=1)
            query_global = F.normalize(query_map.mean(dim=(-2, -1), keepdim=True), p=2, dim=1)

            # Step 1: Select best-matching support reference via global similarity
            sims = [
                float((query_global * supp_glob).sum())
                for supp_glob in self._support_global_embeddings
            ]
            best_idx = int(np.argmax(sims))
            best_supp_map = self._support_feature_maps[best_idx]  # (1, C, H, W)

            # Step 2: Dense Feature Registration (Softmax-weighted spatial correspondence)
            # Flatten spatial dims: Q is (C, N), S is (C, N) where N = H*W
            Q_flat = query_map.squeeze(0).flatten(1)         # (C, N)
            S_flat = best_supp_map.squeeze(0).flatten(1)     # (C, N)

            # Pairwise cosine correlation matrix: (N, N)
            corr = torch.matmul(Q_flat.t(), S_flat)  # (N_query, N_support)

            # Softmax alignment over support locations (temperature=0.1)
            attn = F.softmax(corr / 0.1, dim=-1)     # (N_query, N_support)
            aligned_S = torch.matmul(attn, S_flat.t()).t()  # (C, N_query)
            aligned_S = F.normalize(aligned_S, p=2, dim=0)

            # Cosine discrepancy between query patch and its aligned registered support patch
            cos_sim = (Q_flat * aligned_S).sum(dim=0)  # (N_query,)
            discrepancy = (1.0 - cos_sim).reshape(h_g, w_g).cpu().numpy()

        # Gaussian smoothing
        if self.smooth_sigma > 0:
            ksize = int(self.smooth_sigma * 4) | 1
            discrepancy = cv2.GaussianBlur(discrepancy, (ksize, ksize), self.smooth_sigma)

        anomaly_map = cv2.resize(discrepancy, (w, h), interpolation=cv2.INTER_LINEAR)

        k_top = max(1, int(discrepancy.size * self.top_k_percent))
        image_score = float(np.sort(discrepancy.flatten())[-k_top:].mean())

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
            "method": "regad",
            "support_count": len(self._selected_support_paths),
        }
