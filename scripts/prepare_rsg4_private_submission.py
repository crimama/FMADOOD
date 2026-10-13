"""Produce one RSG4 MVTec AD 2 private submission shard with a normal-only threshold."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import struct
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

import fmad.backbones.dinov2  # noqa: F401
import fmad.datasets  # noqa: F401
from fmad.methods import register_methods
from fmad.protocol import MethodContext, seed_generators, select_support_paths
from fmad.registry import BACKBONE_REGISTRY, DATASET_REGISTRY, METHOD_REGISTRY


SPLITS = ("test_private", "test_private_mixed")
THRESHOLD_PERCENTILE = 95
THRESHOLD_FACTOR = 1.421


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prediction_map(method, image_path: str) -> np.ndarray:
    prediction = method.predict_batch([image_path])[0]
    score = np.asarray(prediction.score_map, dtype=np.float32)
    if score.ndim != 2 or not np.isfinite(score).all():
        raise ValueError(f"Invalid prediction for {image_path}: shape={score.shape}")
    return score


def save_float16_tiff(path: Path, score: np.ndarray) -> None:
    """Write one little-endian, single-strip IEEE float16 grayscale TIFF."""
    height, width = score.shape
    image = np.asarray(score, dtype="<f2", order="C")
    tags = (
        (256, 4, width),  # ImageWidth
        (257, 4, height),  # ImageLength
        (258, 3, 16),  # BitsPerSample
        (259, 3, 1),  # Compression: none
        (262, 3, 1),  # PhotometricInterpretation: BlackIsZero
        (273, 4, 8 + 2 + 10 * 12 + 4),  # StripOffsets
        (277, 3, 1),  # SamplesPerPixel
        (278, 4, height),  # RowsPerStrip
        (279, 4, image.nbytes),  # StripByteCounts
        (339, 3, 3),  # SampleFormat: IEEE floating point
    )
    with path.open("wb") as stream:
        stream.write(struct.pack("<2sHIH", b"II", 42, 8, len(tags)))
        for tag, value_type, value in tags:
            stream.write(struct.pack("<HHI", tag, value_type, 1))
            if value_type == 3:
                stream.write(struct.pack("<H", value) + b"\0\0")
            else:
                stream.write(struct.pack("<I", value))
        stream.write(struct.pack("<I", 0))
        stream.write(image.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index must be between 0 and shard-count - 1")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    protocol = config["protocol"]
    seed_generators(seed)

    dataset_cls = DATASET_REGISTRY.get(config["dataset"])
    dataset = dataset_cls(str(args.data_root), config.get("dataset_config", {}))
    object_info = dataset.get_object_info(args.object)
    train_paths = sorted(dataset.get_train_images(args.object))
    support_paths, _ = select_support_paths(
        train_paths,
        int(protocol["shots"]),
        seed,
        manifest_path=Path(protocol["fixed_reference_manifest"]),
        object_name=args.object,
    )
    support_set = set(support_paths)
    if len(support_set) != len(support_paths) or not support_set.issubset(train_paths):
        raise ValueError("Support paths are duplicated or outside train/good")
    remaining = [path for path in train_paths if path not in support_set]
    calibration_paths = remaining[::8]
    if not calibration_paths or support_set.intersection(calibration_paths):
        raise ValueError("Calibration set must be nonempty and disjoint from support")

    device = torch.device(args.device)
    backbone_cls = BACKBONE_REGISTRY.get(config["backbone"])
    backbone = backbone_cls(model_name=config["backbone"], device=str(device), **config.get("backbone_config", {}))
    register_methods(config["method"])
    method_cls = METHOD_REGISTRY.get(config["method"])
    method = method_cls(backbone, config["method_config"])
    context = MethodContext(
        args.object, seed, object_info.resolution, object_info.masking, object_info.rotation
    )
    if hasattr(backbone, "set_resolution"):
        backbone.set_resolution(context.resolution)
    if hasattr(backbone, "warmup"):
        backbone.warmup(support_paths[0], iters=int(protocol.get("warmup_iters", 1)))

    logging.info("%s: fitting %d support images", args.object, len(support_paths))
    start = time.perf_counter()
    method.fit(support_paths, context)
    fit_seconds = time.perf_counter() - start

    # SuperADD computes one class threshold from every pixel of held-out
    # normal-image predictions. The RSG4 support manifest remains fixed; its
    # calibration images are taken from the other train/good images.
    calibration_maps = []
    for index, path in enumerate(calibration_paths, 1):
        calibration_maps.append(prediction_map(method, path).ravel())
        if index % 10 == 0:
            logging.info("%s: calibrated %d/%d", args.object, index, len(calibration_paths))
    calibration_pixels = np.concatenate(calibration_maps)
    percentile = float(np.percentile(calibration_pixels, THRESHOLD_PERCENTILE))
    threshold = percentile * THRESHOLD_FACTOR
    del calibration_maps, calibration_pixels
    logging.info("%s: normal p95=%.8g, threshold=%.8g", args.object, percentile, threshold)

    counts = {}
    for split in SPLITS:
        images = sorted(
            [path for paths in dataset.get_test_images(args.object, split).values() for path in paths],
            key=lambda path: Path(path).name,
        )
        if not images or len({Path(path).stem for path in images}) != len(images):
            raise ValueError(f"Empty split or duplicate image stems: {args.object}/{split}")
        images = images[args.shard_index :: args.shard_count]
        tiff_dir = args.output_root / "anomaly_images" / args.object / split
        png_dir = args.output_root / "anomaly_images_thresholded" / args.object / split
        tiff_dir.mkdir(parents=True, exist_ok=True)
        png_dir.mkdir(parents=True, exist_ok=True)
        for index, path in enumerate(images, 1):
            score = prediction_map(method, path)
            tiff_path = tiff_dir / f"{Path(path).stem}.tiff"
            png_path = png_dir / f"{Path(path).stem}.png"
            save_float16_tiff(tiff_path, score)
            binary = (score > threshold).astype(np.uint8) * 255
            Image.fromarray(binary, mode="L").save(png_path)
            if index % 25 == 0 or index == len(images):
                logging.info("%s/%s: %d/%d", args.object, split, index, len(images))
        counts[split] = len(images)

    summary = {
        "object": args.object,
        "seed": seed,
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "support_manifest": protocol["fixed_reference_manifest"],
        "support_manifest_sha256": file_sha256(Path(protocol["fixed_reference_manifest"])),
        "support_images": [Path(path).name for path in support_paths],
        "calibration_images": [Path(path).name for path in calibration_paths],
        "calibration_rule": "sorted train/good excluding 16-shot support, every eighth image",
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "threshold_factor": THRESHOLD_FACTOR,
        "normal_p95": percentile,
        "threshold": threshold,
        "binary_rule": "score > threshold; no SuperADD morphology",
        "splits": counts,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "fit_seconds": fit_seconds,
    }
    summary_dir = args.output_root / "provenance"
    summary_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_shard{args.shard_index}of{args.shard_count}" if args.shard_count > 1 else ""
    (summary_dir / f"{args.object}{suffix}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
