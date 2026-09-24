"""Script to compute 5-seed arithmetic mean ensemble of TIFF anomaly maps for MVTec AD 2 private test."""

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import numpy as np
import tifffile
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ensemble_tiffs")

OBJECTS = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]


def average_single_image(args_tuple):
    file_rel_path, seed_paths, out_path = args_tuple
    out_path.parent.mkdir(parents=True, exist_ok=True)

    maps = []
    for sp in seed_paths:
        m = tifffile.imread(str(sp))
        maps.append(m.astype(np.float32))

    # Compute arithmetic mean across seeds
    avg_map = np.mean(maps, axis=0).astype(np.float16)

    # Save as float16 TIFF
    tifffile.imwrite(str(out_path), avg_map)

    return {
        "rel_path": str(file_rel_path),
        "pixel_max": float(avg_map.max()),
        "pixel_mean": float(avg_map.mean()),
    }


def ensemble_seeds(
    seed_dirs: List[Path],
    output_dir: Path,
    split: str = "test_private",
    max_workers: int = 16,
):
    logger.info(f"Ensembling {len(seed_dirs)} seeds into {output_dir}")
    for idx, sd in enumerate(seed_dirs):
        logger.info(f"  Seed [{idx}]: {sd}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_anomaly_root = output_dir / "anomaly_images"

    tasks = []
    # Identify files from the first seed
    ref_seed_root = seed_dirs[0] / "anomaly_images"
    for obj in OBJECTS:
        obj_dir = ref_seed_root / obj / split
        if not obj_dir.exists():
            raise FileNotFoundError(f"Missing directory: {obj_dir}")

        tiff_files = sorted(list(obj_dir.glob("*.tiff")))
        if not tiff_files:
            raise ValueError(f"No TIFF files found in {obj_dir}")

        for tf in tiff_files:
            rel_file = tf.relative_to(ref_seed_root)
            seed_file_paths = [sd / "anomaly_images" / rel_file for sd in seed_dirs]
            for sfp in seed_file_paths:
                if not sfp.is_file():
                    raise FileNotFoundError(f"Missing paired seed file: {sfp}")

            out_file_path = out_anomaly_root / rel_file
            tasks.append((rel_file, seed_file_paths, out_file_path))

    logger.info(f"Total images to ensemble: {len(tasks)}")

    scores_record = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for res in pool.map(average_single_image, tasks):
            scores_record[res["rel_path"]] = {
                "pixel_max": res["pixel_max"],
                "pixel_mean": res["pixel_mean"],
            }

    summary = {
        "split": split,
        "num_seeds": len(seed_dirs),
        "seed_sources": [str(sd) for sd in seed_dirs],
        "total_images": len(tasks),
        "samples": scores_record,
    }
    summary_path = output_dir / f"ensemble_summary_{split}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Ensemble complete! {len(tasks)} TIFFs generated in {out_anomaly_root}")
    logger.info(f"Summary saved to {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble multiple seed TIFF anomaly maps.")
    parser.add_argument("--seed-dirs", nargs="+", required=True, help="List of run output directories for each seed")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for ensembled TIFF maps")
    parser.add_argument("--split", type=str, default="test_private", help="Split name")
    parser.add_argument("--workers", type=int, default=16, help="Worker threads")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_paths = [Path(p) for p in args.seed_dirs]
    ensemble_seeds(
        seed_dirs=seed_paths,
        output_dir=Path(args.output_dir),
        split=args.split,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
