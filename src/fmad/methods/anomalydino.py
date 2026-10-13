"""AnomalyDINO inference adapted to the shared MVTec AD 2 evaluator.

The feature, one-nearest-neighbour cosine distance, and Gaussian map steps
follow dammsi/AnomalyDINO, commit b9d1c2648e3a5247437d4d953d907a8f3d994457
(`src/detection.py` and `src/utils.py`). Support selection and evaluation are
owned by this project's common pipeline. No labels enter fitting or inference.
"""

from __future__ import annotations

import time
from typing import Sequence

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

try:
    import faiss
except ModuleNotFoundError:
    faiss = None

from fmad.methods.base import BaseMethod, Prediction
from fmad.protocol import MethodContext
from fmad.registry import METHOD_REGISTRY


@METHOD_REGISTRY.register("anomalydino")
class AnomalyDINO(BaseMethod):
    """Official single-layer DINOv2 patch retrieval with shared I-AUROC score."""

    def __init__(self, backbone, config: dict):
        super().__init__(backbone, config)
        self._index = None
        self._resources = None
        self._support_count = 0
        self._feature_count = 0
        self._bank_tensor = None
        self._faiss_on_cpu = bool(config.get("faiss_on_cpu", False))
        if int(config.get("k_neighbors", 1)) != 1:
            raise ValueError("The registered AnomalyDINO comparison uses official 1-NN")
        if tuple(getattr(backbone, "feature_layers", ())) != (23,):
            raise ValueError("AnomalyDINO requires the ViT-L final patch-token layer (23)")

    def _features(self, image_path: str) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor, grid = self.backbone.prepare_image(image)
        features = self.backbone.extract_features(tensor)[0].astype("float32")
        if not np.isfinite(features).all():
            raise ValueError("Non-finite AnomalyDINO patch feature")
        return features, grid, image.shape[:2]

    def fit(self, support_paths: list[str], context: MethodContext) -> None:
        if not support_paths:
            raise ValueError("AnomalyDINO needs normal support images")
        if context.masking or context.rotation:
            raise ValueError("This comparison declares no support masking or rotation")
        with torch.inference_mode():
            bank = np.concatenate([self._features(path)[0] for path in support_paths])
        norms = np.linalg.norm(bank, axis=1, keepdims=True)
        bank /= np.maximum(norms, 1e-12)
        if not self._faiss_on_cpu and faiss is not None and torch.cuda.is_available() and hasattr(faiss, "StandardGpuResources"):
            self._resources = faiss.StandardGpuResources()
            gpu_config = faiss.GpuIndexFlatConfig()
            gpu_config.device = torch.device(self.backbone.device).index or 0
            self._index = faiss.GpuIndexFlatL2(self._resources, bank.shape[1], gpu_config)
        elif self._faiss_on_cpu and faiss is not None:
            self._index = faiss.IndexFlatL2(bank.shape[1])
        else:
            # faiss-cpu alone should not force expensive CPU search when a GPU
            # backbone is in use; exact normalized cosine search stays on CUDA.
            self._bank_tensor = torch.from_numpy(bank).to(self.backbone.device)
        if self._index is not None:
            self._index.add(bank)
        self._support_count = len(support_paths)
        self._feature_count = len(bank)
        self.log_training(stage="anomalydino_memory_bank", support_total=self._support_count,
                          patch_count=self._feature_count)

    def predict_batch(self, test_image_paths: Sequence[str]) -> list[Prediction]:
        if self._index is None and self._bank_tensor is None:
            raise RuntimeError("AnomalyDINO fit() must precede prediction")
        outputs = []
        with torch.inference_mode():
            for path in test_image_paths:
                start = time.perf_counter()
                query, grid, shape = self._features(path)
                query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
                if self._index is not None:
                    squared_l2, _ = self._index.search(query, 1)
                    patch_dist = squared_l2[:, 0] / 2
                else:
                    # Equivalent exact 1-NN cosine distance when FAISS is absent.
                    query_tensor = torch.from_numpy(query).to(self.backbone.device)
                    chunks = []
                    for block in query_tensor.split(128):
                        chunks.append((1 - (block @ self._bank_tensor.T).max(dim=1).values).cpu())
                    patch_dist = torch.cat(chunks).numpy()
                patch_dist = patch_dist.reshape(grid)
                score_map = cv2.resize(patch_dist, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
                score_map = gaussian_filter(score_map, sigma=4)
                # The manuscript's common I-AUROC contract uses map maximum.
                image_score = float(np.max(score_map))
                outputs.append(Prediction(score_map=score_map, image_score=image_score,
                                          inference_seconds=time.perf_counter() - start))
        return outputs

    def object_metadata(self) -> dict:
        return {"method": "anomalydino", "official_commit": "b9d1c2648e3a5247437d4d953d907a8f3d994457",
                "support_count": self._support_count, "memory_patches": self._feature_count,
                "feature_layer": 23, "knn": 1, "masking": False, "rotation": False,
                "search_backend": "faiss" if self._index is not None else "torch_exact_cosine",
                "image_score": "score_map_max_shared_contract"}
