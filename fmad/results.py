"""Results manager — standardized saving and TSV accumulation."""

import csv
import json
import os
from datetime import datetime
from typing import Dict, List

from fmad.methods.base import ObjectResult

# TSV columns for the accumulated summary
_TSV_COLUMNS = [
    "timestamp", "method", "dataset", "backbone", "seed",
    "object", "seg_AUROC", "seg_F1", "best_thre",
    "mean_inference_time", "memorybank_time",
]


class ResultsManager:
    """Manages per-run outputs and accumulated summary TSV."""

    def __init__(self, base_dir: str, method: str, dataset: str,
                 backbone: str, seed: int):
        self.method = method
        self.dataset = dataset
        self.backbone = backbone
        self.seed = seed
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Per-run directory: results/{method}/{dataset}/{backbone}/seed={seed}/
        self.run_dir = os.path.join(
            base_dir, method, dataset, backbone, f"seed={seed}"
        )
        os.makedirs(self.run_dir, exist_ok=True)

        # Accumulated summary file
        self.summary_path = os.path.join(base_dir, "summary.tsv")

    def save_config(self, config: dict) -> None:
        """Save full config snapshot for reproducibility."""
        import yaml
        with open(os.path.join(self.run_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    def save_object_times(self, results: List[ObjectResult]) -> None:
        """Save per-sample inference times as CSV."""
        csv_path = os.path.join(self.run_dir, "time_measurements.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["object", "sample", "anomaly_score",
                             "memorybank_time", "inference_time"])
            for r in results:
                for sample_key, score in r.anomaly_scores.items():
                    writer.writerow([
                        r.object_name, sample_key, f"{score:.5f}",
                        f"{r.time_memorybank:.5f}",
                        f"{r.inference_times[sample_key]:.5f}",
                    ])

    def save_metrics(self, metrics: dict) -> None:
        """Save per-run metrics JSON."""
        with open(os.path.join(self.run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    def append_summary(self, metrics: dict, results: List[ObjectResult]) -> None:
        """Append per-object rows to the accumulated summary TSV."""
        write_header = not os.path.exists(self.summary_path)

        times_by_object = {}
        for r in results:
            if r.inference_times:
                times_by_object[r.object_name] = {
                    "mean_inference": sum(r.inference_times.values()) / len(r.inference_times),
                    "memorybank": r.time_memorybank,
                }

        with open(self.summary_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_TSV_COLUMNS, delimiter="\t")
            if write_header:
                writer.writeheader()

            for obj_name, obj_metrics in metrics.items():
                if not isinstance(obj_metrics, dict):
                    continue
                if "seg_AUROC" not in obj_metrics and "best_thre" not in obj_metrics:
                    continue
                timing = times_by_object.get(obj_name, {})
                writer.writerow({
                    "timestamp": self.timestamp,
                    "method": self.method,
                    "dataset": self.dataset,
                    "backbone": self.backbone,
                    "seed": self.seed,
                    "object": obj_name,
                    "seg_AUROC": f"{obj_metrics.get('seg_AUROC', 0):.4f}",
                    "seg_F1": f"{obj_metrics.get('seg_F1', 0):.4f}",
                    "best_thre": f"{obj_metrics.get('best_thre', 0):.6f}",
                    "mean_inference_time": f"{timing.get('mean_inference', 0):.4f}",
                    "memorybank_time": f"{timing.get('memorybank', 0):.4f}",
                })
