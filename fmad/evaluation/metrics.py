"""Unified evaluation — wraps src/post_eval.py for metric computation."""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.post_eval import parse_dataset_files, eval_segmentation


class Evaluator:
    """Evaluates only the objects that were actually run (avoids missing-file errors)."""

    def __init__(self, config: dict):
        self.pro_integration_limit = config.get("pro_integration_limit", 0.05)
        self.eval_segm = config.get("eval_segm", True)

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
