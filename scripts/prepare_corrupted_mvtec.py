#!/usr/bin/env python3
"""Generate corrupted MVTec AD test sets following the ICCV 2023 protocol:
'Anomaly Detection under Distribution Shift' (Cao et al., ICCV 2023).

Protocol:
- Corruptions: brightness, contrast, defocus_blur, gaussian_noise
- Severity: 3
- Support/Train: clean normal (symlinked from original MVTec AD)
- Ground Truth: exact mask (symlinked from original MVTec AD)
- Test: all test images (both good and defective) corrupted with severity 3
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from PIL import Image
import numpy as np
from imagecorruptions import corrupt

ALL_CLASSES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

CORRUPTIONS = ["brightness", "contrast", "defocus_blur", "gaussian_noise"]


def process_image(src_path: Path, dst_path: Path, corruption: str, severity: int):
    if dst_path.exists():
        return
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src_path).convert("RGB")
    img_np = np.array(img)
    corrupted_np = corrupt(img_np, corruption_name=corruption, severity=severity)
    corrupted_img = Image.fromarray(corrupted_np)
    corrupted_img.save(dst_path)


def generate_corrupted_dataset(
    src_root: Path,
    dst_root: Path,
    corruption: str,
    severity: int = 3,
    classes: list[str] = ALL_CLASSES,
    num_workers: int = 8,
):
    print(f"=== Generating {corruption} (severity {severity}) ===")
    target_root = dst_root / corruption
    tasks = []

    for cls_name in classes:
        cls_src = src_root / cls_name
        cls_dst = target_root / cls_name
        cls_dst.mkdir(parents=True, exist_ok=True)

        # 1. Symlink train
        train_src = cls_src / "train"
        train_dst = cls_dst / "train"
        if not train_dst.exists():
            train_dst.symlink_to(train_src)

        # 2. Symlink ground_truth
        gt_src = cls_src / "ground_truth"
        gt_dst = cls_dst / "ground_truth"
        if not gt_dst.exists():
            gt_dst.symlink_to(gt_src)

        # 3. Collect test images to corrupt
        test_src = cls_src / "test"
        test_dst = cls_dst / "test"

        for sub_dir in test_src.iterdir():
            if not sub_dir.is_dir():
                continue
            for img_file in sub_dir.glob("*.png"):
                dst_img_file = test_dst / sub_dir.name / img_file.name
                tasks.append((img_file, dst_img_file, corruption, severity))

    print(f"Total test images to process for {corruption}: {len(tasks)}")

    # Run parallel image corruption
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_image, src, dst, corr, sev)
            for src, dst, corr, sev in tasks
        ]
        done = 0
        for f in futures:
            f.result()
            done += 1
            if done % 200 == 0 or done == len(tasks):
                print(f"[{corruption}] Processed {done}/{len(tasks)} images...")

    print(f"=== Completed {corruption} at {target_root} ===\n")


def main():
    parser = argparse.ArgumentParser(description="Generate corrupted MVTec AD datasets")
    parser.add_argument("--src-root", type=str, required=True, help="Path to original MVTec AD")
    parser.add_argument("--dst-root", type=str, required=True, help="Destination directory for corrupted datasets")
    parser.add_argument("--corruptions", nargs="+", default=CORRUPTIONS, help="List of corruptions to generate")
    parser.add_argument("--severity", type=int, default=3, help="Corruption severity (1-5)")
    parser.add_argument("--classes", nargs="+", default=ALL_CLASSES, help="List of classes to process")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")

    args = parser.parse_args()
    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()

    if not src_root.exists():
        print(f"Error: source root {src_root} does not exist", file=sys.stderr)
        sys.exit(1)

    dst_root.mkdir(parents=True, exist_ok=True)

    for corruption in args.corruptions:
        generate_corrupted_dataset(
            src_root=src_root,
            dst_root=dst_root,
            corruption=corruption,
            severity=args.severity,
            classes=args.classes,
            num_workers=args.workers,
        )


if __name__ == "__main__":
    main()
