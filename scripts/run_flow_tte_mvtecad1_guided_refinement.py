"""Apply the frozen RGB guided-r8 refinement to classic MVTec AD1 maps."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import tifffile as tiff

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (_REPO_ROOT, _REPO_ROOT / "src", Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from flow_tte_mvtec_classic import (  # noqa: E402
    ClassicEvaluationConfig,
    ClassicMVTecDataset,
    VisADataset,
    evaluate_classic_mvtec,
)
from src.flow_tte_phase2_refinement import load_half_guidance, transform_score  # noqa: E402

_VARIANT = "guided_r8_eps1e-2"


def parse_objects(raw: str) -> tuple[str, ...]:
    objects = tuple(part for part in raw.replace(",", " ").split() if part)
    if not objects:
        raise SystemExit("--objects must contain at least one object")
    return objects


def refine_maps(
    dataset: ClassicMVTecDataset,
    source_root: Path,
    output_root: Path,
) -> int:
    written = 0
    for object_name in dataset.objects:
        for anomaly_type, image_paths in dataset.get_test_images(object_name).items():
            for image_path_text in image_paths:
                image_path = Path(image_path_text)
                relative = Path(object_name) / "test" / anomaly_type / f"{image_path.stem}.tiff"
                source_path = source_root / "anomaly_maps" / relative
                if not source_path.is_file():
                    raise FileNotFoundError(f"Source map not found: {source_path}")
                score = np.asarray(tiff.imread(source_path), dtype=np.float32)
                guidance = load_half_guidance(image_path, score.shape)
                refined = transform_score(score, guidance, _VARIANT)
                if refined.dtype != np.float32 or refined.shape != score.shape:
                    raise RuntimeError(f"Invalid refined map contract: {relative}")
                if not np.all(np.isfinite(refined)):
                    raise RuntimeError(f"Non-finite refined map: {relative}")
                destination = output_root / "anomaly_maps" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                tiff.imwrite(destination, refined)
                written += 1
    return written


def cleanup_maps(root: Path) -> None:
    maps = root / "anomaly_maps"
    if maps.exists():
        shutil.rmtree(maps)


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset-kind", choices=("mvtec_ad1", "visa"), default="mvtec_ad1")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-percent", type=float, default=0.01)
    parser.add_argument("--cleanup-source-maps", action="store_true")
    parser.add_argument("--cleanup-output-maps", action="store_true")
    args = parser.parse_args(list(argv))

    objects = parse_objects(args.objects)
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_cls = VisADataset if args.dataset_kind == "visa" else ClassicMVTecDataset
    dataset = dataset_cls(args.data_root, objects, resolution=448)
    written = refine_maps(dataset, source_root, output_root)
    metrics = evaluate_classic_mvtec(
        ClassicEvaluationConfig(
            dataset=dataset,
            output_root=output_root,
            objects=objects,
            pro_integration_limit=0.05,
            seed=args.seed,
            image_top_fraction=args.top_percent,
            include_legacy_segmentation_metrics=False,
        ),
    )
    manifest = {
        "target_dataset": f"{dataset.dataset_name} single-image",
        "source_root": str(source_root),
        "objects": list(objects),
        "refinement": _VARIANT,
        "work_scale": 0.5,
        "guided_radius_at_work_scale": 8,
        "guided_epsilon": 0.01,
        "written_map_count": written,
        "upstream_change": "none",
        "metrics": metrics,
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.cleanup_source_maps:
        cleanup_maps(source_root)
        (source_root / "cleanup_evidence.txt").write_text(
            "cleanup_anomaly_maps=true\n",
            encoding="utf-8",
        )
    if args.cleanup_output_maps:
        cleanup_maps(output_root)
    (output_root / "cleanup_evidence.txt").write_text(
        "cleanup_source_maps="
        f"{str(args.cleanup_source_maps).lower()}\n"
        "cleanup_output_maps="
        f"{str(args.cleanup_output_maps).lower()}\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
