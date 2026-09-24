"""Script to run inference on MVTec AD 2 test_private (or test_private_mixed) and save anomaly maps."""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from PIL import Image

import fmad.backbones.dinov2  # noqa: F401
import fmad.backbones.dinov3  # noqa: F401
import fmad.datasets  # noqa: F401
from fmad.methods import register_methods
from fmad.datasets.base import BaseDataset
from fmad.datasets.mvtec_ad2 import MVTecAD2Dataset
from fmad.methods.base import BaseMethod
from fmad.protocol import MethodContext, select_support_paths, seed_generators
from fmad.registry import BACKBONE_REGISTRY, DATASET_REGISTRY, METHOD_REGISTRY

# Try importing tifffile for faster/native TIFF saving; fallback to PIL
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("private_inference")


def save_tiff_map(path: Path, score_map: np.ndarray):
    """Save 2D score map as float16 TIFF adhering to MVTec AD 2 official submission rules."""
    path.parent.mkdir(parents=True, exist_ok=True)
    score_map_f16 = score_map.astype(np.float16)
    if HAS_TIFFFILE:
        tifffile.imwrite(str(path), score_map_f16)
    else:
        Image.fromarray(score_map_f16).save(str(path), format="TIFF")


def save_thresholded_png(path: Path, score_map: np.ndarray, threshold: float):
    """Save binary mask as 0/255 uint8 PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    binary = (score_map > threshold).astype(np.uint8) * 255
    Image.fromarray(binary).save(str(path), format="PNG")


def run_private_object(
    method: BaseMethod,
    dataset: BaseDataset,
    object_name: str,
    seed: int,
    output_dir: Path,
    split: str = "test_private",
    batch_size: int = 1,
    protocol_cfg: dict | None = None,
    save_thresholded: bool = True,
    threshold: Optional[float] = None,
) -> dict:
    protocol_cfg = protocol_cfg or {}
    shots = protocol_cfg.get("shots", 16)
    manifest = protocol_cfg.get("fixed_reference_manifest")
    warmup_iters = protocol_cfg.get("warmup_iters", 1)

    seed_generators(seed)
    obj_info = dataset.get_object_info(object_name)
    train_paths = sorted(dataset.get_train_images(object_name))
    support_paths, _ = select_support_paths(
        train_paths,
        shots,
        seed,
        manifest_path=Path(manifest) if manifest else None,
        object_name=object_name,
    )

    context = MethodContext(
        object_name, seed, obj_info.resolution, obj_info.masking, obj_info.rotation
    )
    if hasattr(method.backbone, "set_resolution"):
        method.backbone.set_resolution(context.resolution)
    if hasattr(method.backbone, "warmup") and support_paths:
        method.backbone.warmup(support_paths[0], iters=warmup_iters)

    logger.info(f"[{object_name}] Fitting support bank ({len(support_paths)} images)...")
    fit_start = time.perf_counter()
    method.fit(support_paths, context)
    fit_seconds = time.perf_counter() - fit_start
    logger.info(f"[{object_name}] Fit complete in {fit_seconds:.2f}s")

    # Load test split images
    grouped = dataset.get_test_images(object_name, split=split)
    queries = sorted(
        [
            (anomaly_type, image_path)
            for anomaly_type, image_paths in grouped.items()
            for image_path in image_paths
        ],
        key=lambda item: Path(item[1]).name,
    )
    if not queries:
        raise ValueError(f"{object_name}: No queries found for split {split}")

    logger.info(f"[{object_name}] Inferring {len(queries)} images on split={split}...")

    tiff_dir = output_dir / "anomaly_images" / object_name / split
    png_dir = output_dir / "anomaly_images_thresholded" / object_name / split

    scores_record = {}
    infer_start = time.perf_counter()

    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]
        predictions = method.predict_batch([path for _, path in batch])

        for (_, image_path), prediction in zip(batch, predictions, strict=True):
            stem = Path(image_path).stem
            score_map = np.asarray(prediction.score_map, dtype=np.float32)

            # 1. Save TIFF
            tiff_path = tiff_dir / f"{stem}.tiff"
            save_tiff_map(tiff_path, score_map)

            # 2. Save Thresholded PNG if requested
            if save_thresholded and threshold is not None:
                png_path = png_dir / f"{stem}.png"
                save_thresholded_png(png_path, score_map, threshold)

            scores_record[stem] = {
                "image_score": float(prediction.image_score),
                "pixel_max": float(score_map.max()),
                "pixel_mean": float(score_map.mean()),
                "inference_seconds": float(prediction.inference_seconds),
            }

    infer_seconds = time.perf_counter() - infer_start
    mean_img_sec = infer_seconds / len(queries)
    logger.info(
        f"[{object_name}] Inference complete: {len(queries)} images in {infer_seconds:.2f}s ({mean_img_sec:.3f}s/img)"
    )

    return {
        "object": object_name,
        "split": split,
        "count": len(queries),
        "fit_seconds": fit_seconds,
        "infer_seconds": infer_seconds,
        "mean_seconds_per_image": mean_img_sec,
        "threshold_used": threshold,
        "sample_scores": scores_record,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run private test inference and save TIFF anomaly maps.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--data-root", type=str, required=True, help="Root path to MVTec AD 2")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save output maps and results")
    parser.add_argument("--device", type=str, default="cuda:0", help="Execution device (e.g. cuda:0)")
    parser.add_argument("--objects", nargs="+", default=None, help="Objects to evaluate (default: all 8)")
    parser.add_argument("--split", type=str, default="test_private", help="Split name (default: test_private)")
    parser.add_argument("--thresholds-json", type=str, default=None, help="JSON file mapping object to threshold")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Build dataset
    dataset_name = config.get("dataset", "mvtec_ad2")
    dataset_cls = DATASET_REGISTRY.get(dataset_name)
    dataset = dataset_cls(args.data_root, config.get("dataset_config", {}))

    # Build backbone
    backbone_name = config.get("backbone", "dinov2_vitl14")
    backbone_cls = BACKBONE_REGISTRY.get(backbone_name)
    backbone_cfg = dict(config.get("backbone_config", {}))
    backbone = backbone_cls(
        model_name=backbone_name,
        device=str(device),
        **backbone_cfg,
    )

    # Build method
    method_name = config.get("method", "reference_matching")
    register_methods(method_name)
    method_cls = METHOD_REGISTRY.get(method_name)
    method_cfg = config.get("method_config", {})
    method = method_cls(backbone, method_cfg)

    # Targets
    all_objects = [obj.name for obj in dataset.get_objects()]
    target_objects = args.objects if args.objects is not None else all_objects

    # Load thresholds if provided
    thresholds = {}
    if args.thresholds_json and os.path.isfile(args.thresholds_json):
        with open(args.thresholds_json, "r") as f:
            thresholds = json.load(f)
        logger.info(f"Loaded thresholds: {thresholds}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol_cfg = config.get("protocol", {})
    overall_summary = {
        "config": args.config,
        "seed": args.seed,
        "backbone": backbone_name,
        "method": method_name,
        "split": args.split,
        "device": str(device),
        "objects": {},
    }

    total_start = time.perf_counter()
    for obj_name in target_objects:
        obj_thresh = thresholds.get(obj_name, None)
        res = run_private_object(
            method=method,
            dataset=dataset,
            object_name=obj_name,
            seed=args.seed,
            output_dir=output_dir,
            split=args.split,
            batch_size=args.batch_size,
            protocol_cfg=protocol_cfg,
            save_thresholded=(obj_thresh is not None),
            threshold=obj_thresh,
        )
        overall_summary["objects"][obj_name] = res

    overall_summary["total_seconds"] = time.perf_counter() - total_start
    summary_path = output_dir / f"summary_seed{args.seed}_{args.split}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2)
    logger.info(f"All done in {overall_summary['total_seconds']:.2f}s! Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
