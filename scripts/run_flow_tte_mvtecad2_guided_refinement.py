"""Apply the frozen RGB guided-r8 refinement to MVTec AD2 anomaly maps."""

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

from fmad.datasets.mvtec_ad2 import MVTecAD2Dataset  # noqa: E402
from fmad.evaluation.metrics import Evaluator  # noqa: E402
from src.flow_tte_phase2_refinement import load_half_guidance, transform_score  # noqa: E402

RGB_GUIDE_VARIANT = "guided_r8_eps1e-2"


def parse_objects(raw: str) -> tuple[str, ...]:
    objects = tuple(part for part in raw.replace(",", " ").split() if part)
    if not objects:
        raise SystemExit("--objects must contain at least one object")
    return objects


def refine_maps(
    dataset: MVTecAD2Dataset,
    source_root: Path,
    output_root: Path,
) -> int:
    written = 0
    for object_info in dataset.get_objects():
        object_name = object_info.name
        for anomaly_type, image_paths in dataset.get_test_images(object_name).items():
            for image_path_text in image_paths:
                image_path = Path(image_path_text)
                relative = Path(object_name) / "test" / anomaly_type / f"{image_path.stem}.tiff"
                source_path = source_root / "anomaly_maps" / relative
                if not source_path.is_file():
                    raise FileNotFoundError(f"Source map not found: {source_path}")
                destination = output_root / "anomaly_maps" / relative
                refine_map_file(image_path, source_path, destination)
                written += 1
    return written


def refine_map_file(image_path: Path, source_path: Path, destination: Path) -> None:
    score = np.asarray(tiff.imread(source_path), dtype=np.float32)
    guidance = load_half_guidance(image_path, score.shape)
    refined = transform_score(score, guidance, RGB_GUIDE_VARIANT)
    if refined.dtype != np.float32 or refined.shape != score.shape:
        raise RuntimeError(f"Invalid refined map contract: {source_path}")
    if not np.all(np.isfinite(refined)):
        raise RuntimeError(f"Non-finite refined map: {source_path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(destination, refined)


def cleanup_maps(root: Path) -> None:
    maps = root / "anomaly_maps"
    if maps.exists():
        shutil.rmtree(maps)


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--binary-postprocess",
        choices=("none", "closefill", "closefill_erode"),
        default="closefill_erode",
    )
    parser.add_argument("--morphology-line-length", type=int, default=17)
    parser.add_argument("--morphology-angle-count", type=int, default=16)
    parser.add_argument("--cleanup-source-maps", action="store_true")
    parser.add_argument("--cleanup-output-maps", action="store_true")
    args = parser.parse_args(list(argv))

    objects = parse_objects(args.objects)
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset = MVTecAD2Dataset(
        args.data_root,
        {"objects": list(objects), "preprocess": "no_mask_no_rotation"},
    )
    written = refine_maps(dataset, source_root, output_root)
    evaluator = Evaluator({
        "pro_integration_limit": 0.05,
        "binary_postprocess": args.binary_postprocess,
        "morphology_line_length": args.morphology_line_length,
        "morphology_angle_count": args.morphology_angle_count,
    })
    metrics = evaluator.evaluate_run(
        dataset_name="MVTec_AD_2",
        data_root=args.data_root,
        anomaly_maps_dir=str(output_root / "anomaly_maps"),
        output_dir=str(output_root),
        seed=args.seed,
        objects=list(objects),
    )
    (output_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "target_dataset": "MVTec AD2 single-image",
        "split": "test_public/good,bad",
        "source_root": str(source_root),
        "objects": list(objects),
        "refinement": RGB_GUIDE_VARIANT,
        "work_scale": 0.5,
        "guided_radius_at_work_scale": 8,
        "guided_epsilon": 0.01,
        "binary_postprocess": args.binary_postprocess,
        "morphology_line_length": args.morphology_line_length,
        "morphology_angle_count": args.morphology_angle_count,
        "binary_threshold_source": "each continuous map's raw_best_thre",
        "ground_truth_used_by_refinement": False,
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
