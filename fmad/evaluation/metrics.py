"""Unified evaluation — wraps src/post_eval.py for metric computation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.flow_tte_postprocess_core import binary_mask_metrics, variant_profile
from src.post_eval import eval_segmentation, parse_dataset_files, read_tiff


class Evaluator:
    """Evaluates only the objects that were actually run (avoids missing-file errors)."""

    def __init__(self, config: dict):
        self.pro_integration_limit = config.get("pro_integration_limit", 0.05)
        self.eval_segm = config.get("eval_segm", True)
        self.binary_postprocess = config.get("binary_postprocess", "none")
        self.morphology_line_length = int(config.get("morphology_line_length", 17))
        self.morphology_angle_count = int(config.get("morphology_angle_count", 16))

    def evaluate_run(self, dataset_name: str, data_root: str,
                     anomaly_maps_dir: str, output_dir: str,
                     seed: int = 0,
                     objects: List[str] | None = None) -> Dict:
        """Evaluate objects for a completed run.

        Args:
            dataset_name: Dataset identifier (e.g. "MVTec_AD_2").
            data_root: Root path to dataset.
            anomaly_maps_dir: Directory containing {object}/test/*.tiff
            output_dir: Where to write metrics JSON.
            seed: Seed index.
            objects: List of objects to evaluate (auto-detect if None).

        Returns:
            Dict with per-object and mean metrics.
        """
        # Auto-detect objects from anomaly_maps directory
        if objects is None:
            objects = sorted(
                d for d in os.listdir(anomaly_maps_dir)
                if os.path.isdir(os.path.join(anomaly_maps_dir, d))
            )

        evaluation_dict = {}
        auroc_ls, f1_ls = [], []

        for obj in objects:
            obj_test_dir = os.path.join(anomaly_maps_dir, obj, "test")
            if not os.path.isdir(obj_test_dir):
                print(f"  Skipping {obj}: no test directory")
                continue

            print(f"=== Evaluate {obj} ===")
            gt_filenames, prediction_filenames = parse_dataset_files(
                object_name=obj,
                dataset_base_dir=data_root,
                anomaly_maps_dir=anomaly_maps_dir,
                dataset=dataset_name,
            )

            if self.eval_segm and gt_filenames:
                auroc_px, f1_px, best_thre = eval_segmentation(
                    gt_filenames,
                    prediction_filenames,
                    pro_integration_limit=self.pro_integration_limit,
                    delete_tiff_files=False,
                )
                evaluation_dict[obj] = {
                    "seg_AUROC": auroc_px,
                    "seg_F1": f1_px,
                    "best_thre": best_thre,
                }
                if self.binary_postprocess != "none":
                    scores = [np.asarray(read_tiff(path), dtype=np.float32) for path in prediction_filenames]
                    gt_masks = [
                        np.zeros(score.shape, dtype=np.bool_)
                        if gt_path is None
                        else np.asarray(Image.open(gt_path)) > 0
                        for gt_path, score in zip(gt_filenames, scores)
                    ]
                    processed = binary_mask_metrics(
                        scores,
                        gt_masks,
                        best_thre,
                        variant_profile(self.binary_postprocess),
                        self.morphology_line_length,
                        self.morphology_angle_count,
                    )
                    evaluation_dict[obj].update({
                        "seg_F1_raw": f1_px,
                        "seg_F1": processed.f1,
                        "binary_postprocess": self.binary_postprocess,
                        "binary_postprocess_threshold_source": "raw_best_thre",
                        "binary_postprocess_precision": processed.precision,
                        "binary_postprocess_recall": processed.recall,
                        "binary_postprocess_positive_area": processed.positive_area,
                    })
                    f1_px = processed.f1
                auroc_ls.append(auroc_px)
                f1_ls.append(f1_px)

        if auroc_ls:
            evaluation_dict["mean_segmentation_au_roc"] = float(np.mean(auroc_ls))
            evaluation_dict["mean_segmentation_f1"] = float(np.mean(f1_ls))

        # Save metrics
        metrics_path = os.path.join(output_dir, f"metrics_seed={seed}.json")
        with open(metrics_path, "w") as f:
            json.dump(evaluation_dict, f, indent=2)

        return evaluation_dict
