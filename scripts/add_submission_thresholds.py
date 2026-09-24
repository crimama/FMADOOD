"""Create complete binary MVTec AD 2 submission maps from TIFF score quantiles."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


OBJECTS = (
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
)
SPLITS = ("test_private", "test_private_mixed")


def exact_quantile(paths: list[Path], quantile: float) -> tuple[float, int]:
    """Compute NumPy's linear quantile exactly from float16 value counts."""
    counts: Counter[float] = Counter()
    total = 0
    for path in paths:
        image = tifffile.imread(path)
        if image.ndim != 2 or image.dtype != np.float16:
            raise ValueError(f"Expected a 2D float16 TIFF: {path}")
        if not np.isfinite(image).all():
            raise ValueError(f"Non-finite score found: {path}")
        values, value_counts = np.unique(image, return_counts=True)
        counts.update(dict(zip(values.astype(float), value_counts.astype(int))))
        total += image.size

    if not total:
        raise ValueError("No pixels found")

    ordered = sorted(counts.items())
    rank = (total - 1) * quantile
    lower_rank = int(np.floor(rank))
    upper_rank = int(np.ceil(rank))

    def value_at(target: int) -> float:
        cumulative = 0
        for value, count in ordered:
            cumulative += count
            if cumulative > target:
                return value
        raise AssertionError("Quantile rank exceeded pixel count")

    lower = value_at(lower_rank)
    upper = value_at(upper_rank)
    fraction = rank - lower_rank
    return lower + (upper - lower) * fraction, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 < args.quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1")

    score_root = args.submission / "anomaly_images"
    binary_root = args.submission / "anomaly_images_thresholded"
    metadata: dict[str, object] = {
        "quantile": args.quantile,
        "scope": "per_class_across_test_private_and_test_private_mixed_pixels",
        "comparison": "score > threshold",
        "classes": {},
    }

    for object_name in OBJECTS:
        paths = [
            path
            for split in SPLITS
            for path in sorted((score_root / object_name / split).glob("*.tiff"))
        ]
        threshold, pixel_count = exact_quantile(paths, args.quantile)
        positive_pixels = 0
        for source in paths:
            split = source.parent.name
            destination = binary_root / object_name / split / f"{source.stem}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            score = tifffile.imread(source)
            binary = (score > threshold).astype(np.uint8) * 255
            positive_pixels += int(np.count_nonzero(binary))
            Image.fromarray(binary, mode="L").save(destination, format="PNG")

        metadata["classes"][object_name] = {
            "threshold": threshold,
            "pixel_count": pixel_count,
            "positive_pixel_count": positive_pixels,
            "positive_fraction": positive_pixels / pixel_count,
            "image_count": len(paths),
        }
        print(
            f"{object_name}: threshold={threshold:.10g}, "
            f"positive_fraction={positive_pixels / pixel_count:.6%}, "
            f"images={len(paths)}"
        )

    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
