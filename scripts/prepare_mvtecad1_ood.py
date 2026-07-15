#!/usr/bin/env python3
"""Create the severity-3 MVTec-OOD roots used by the ADShift benchmark."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


CORRUPTIONS = ("brightness", "contrast", "defocus_blur", "gaussian_noise")


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        return
    destination.symlink_to(source)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--severity", type=int, default=3)
    parser.add_argument(
        "--corruptions",
        nargs="+",
        choices=CORRUPTIONS,
        default=list(CORRUPTIONS),
    )
    args = parser.parse_args()

    from imagecorruptions import corrupt

    source_root = args.source_root.resolve()
    for corruption_name in args.corruptions:
        output_root = args.output_parent / f"mvtec_{corruption_name}_s{args.severity}"
        for object_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
            link(object_dir / "train", output_root / object_dir.name / "train")
            ground_truth = object_dir / "ground_truth"
            if ground_truth.is_dir():
                link(ground_truth, output_root / object_dir.name / "ground_truth")
            for image_path in sorted((object_dir / "test").glob("*/*")):
                if not image_path.is_file():
                    continue
                relative = image_path.relative_to(source_root)
                destination = output_root / relative
                if destination.is_file():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(image_path) as image:
                    rgb = np.asarray(image.convert("RGB"))
                transformed = corrupt(
                    rgb,
                    corruption_name=corruption_name,
                    severity=args.severity,
                )
                Image.fromarray(transformed).save(destination)
        marker = output_root / "ood_manifest.txt"
        marker.write_text(
            f"source_root={source_root}\ncorruption={corruption_name}\nseverity={args.severity}\n"
            "implementation=imagecorruptions==1.1.2\n",
            encoding="utf-8",
        )
        print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
