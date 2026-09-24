#!/usr/bin/env python3
"""Summarize ICCV 2023 distribution shift benchmark results for MVTec AD (16-shot DINOv2).

Computes Macro-average metrics across 15 classes for:
- In-Distribution (Clean)
- Brightness (Severity 3)
- Contrast (Severity 3)
- Defocus Blur (Severity 3)
- Gaussian Noise (Severity 3)
- Average OOD (across 4 corruptions)
"""

import json
from pathlib import Path
import numpy as np

CLASSES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]


def load_metrics_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def extract_class_metrics(data: dict):
    results = {}
    for cls in CLASSES:
        if cls in data:
            results[cls] = {
                "image_auroc": data[cls].get("image_AUROC", 0.0) * 100,
                "pixel_auroc": data[cls].get("seg_AUROC_1.00", 0.0) * 100,
                "aupro_030": data[cls].get("seg_AUPRO_0.30", 0.0) * 100,
                "aupro_005": data[cls].get("seg_AUPRO_0.05", 0.0) * 100,
                "f1_oracle": data[cls].get("seg_F1_oracle", 0.0) * 100,
            }
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-metrics", type=str, default="results/benchmark_runs/mvtec/dinov2_s0/reference_matching/mvtec/dinov2_vitl14/seed=0/run-e2cc38a2170e5b46-5cc83870beda4fb3891e3a4b2c308364/metrics.json")
    parser.add_argument("--shift-dir", type=str, default="results/remote_runs/dsba3/dist_shift/mvtec")
    args = parser.parse_args()

    clean_path = Path(args.clean_metrics)
    shift_base = Path(args.shift_dir)

    shifts = ["brightness", "contrast", "defocus_blur", "gaussian_noise"]
    all_runs = {}

    if clean_path.exists():
        all_runs["Clean (ID)"] = extract_class_metrics(load_metrics_json(clean_path))

    for shift in shifts:
        shift_pattern = list((shift_base / f"dinov2_{shift}").glob("**/metrics.json"))
        if shift_pattern:
            all_runs[shift] = extract_class_metrics(load_metrics_json(shift_pattern[0]))

    print("\n" + "=" * 90)
    print("MVTec AD 16-shot DINOv2 under Distribution Shift (ICCV 2023 Protocol)")
    print("=" * 90)

    # Print summary table of means
    metrics_keys = [
        ("image_auroc", "Image-AUROC (%)"),
        ("pixel_auroc", "Pixel-AUROC (%)"),
        ("aupro_030", "AUPRO@0.30 (%)"),
        ("aupro_005", "AUPRO@0.05 (%)"),
        ("f1_oracle", "Oracle F1 (%)"),
    ]

    header = f"{'Setting':<20}" + "".join(f"{name:>16}" for _, name in metrics_keys)
    print(header)
    print("-" * len(header))

    ood_means = {k: [] for k, _ in metrics_keys}

    for name, run_data in all_runs.items():
        row_str = f"{name:<20}"
        for m_key, _ in metrics_keys:
            vals = [run_data[c][m_key] for c in CLASSES if c in run_data]
            mean_val = np.mean(vals) if vals else 0.0
            row_str += f"{mean_val:>16.2f}"
            if name != "Clean (ID)":
                ood_means[m_key].append(mean_val)
        print(row_str)

    # Print Average OOD
    if any(ood_means.values()):
        row_str = f"{'Average OOD':<20}"
        for m_key, _ in metrics_keys:
            mean_ood = np.mean(ood_means[m_key]) if ood_means[m_key] else 0.0
            row_str += f"{mean_ood:>16.2f}"
        print("-" * len(header))
        print(row_str)

    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
