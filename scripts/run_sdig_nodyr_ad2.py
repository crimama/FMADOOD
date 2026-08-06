#!/usr/bin/env python3
"""Run the raw SDIG-NODyR Stage A candidate and intrinsic controls on AD2."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import tifffile as tiff
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "src", Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from fmad.backbones.dinov2 import DINOv2Backbone  # noqa: E402
from fmad.datasets.mvtec_ad2 import MVTecAD2Dataset  # noqa: E402
from fmad.evaluation.metrics import Evaluator  # noqa: E402
from flow_tte.nodyr import fit_nodyr_models, score_nodyr_models  # noqa: E402
from src.post_eval import eval_segmentation  # noqa: E402

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
VARIANTS = (
    "nodyr",
    "state_angle",
    "state_l2",
    "raw_delta_l2",
    "shuffle_nodyr",
    "unit_delta_l2_equiv",
)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--support-json", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone-model", default="dinov2_vitl14")
    parser.add_argument("--layers", default="15,23")
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-good", type=int, default=0)
    parser.add_argument("--max-bad", type=int, default=0)
    parser.add_argument("--source-ids", default="")
    parser.add_argument("--host-gpu-id", required=True)
    parser.add_argument("--container-cuda-slot", default="0")
    parser.add_argument("--cleanup-maps", action="store_true")
    args = parser.parse_args(list(argv))
    args.objects = tuple(part for part in args.objects.replace(",", " ").split() if part)
    args.layers = tuple(int(part) for part in args.layers.replace(",", " ").split())
    if not args.objects or any(item not in OBJECTS for item in args.objects):
        raise SystemExit(f"--objects must be a non-empty subset of {OBJECTS}")
    if len(args.layers) != 2 or not 0 <= args.layers[0] < args.layers[1] < 24:
        raise SystemExit("--layers must contain two increasing DINOv2 ViT-L indices below 24")
    if args.ridge <= 0.0:
        raise SystemExit("--ridge must be positive")
    if args.max_good < 0 or args.max_bad < 0:
        raise SystemExit("sample limits must be non-negative")
    args.source_ids = tuple(part for part in args.source_ids.replace(",", " ").split() if part)
    return args


def load_support_manifest(path: Path, data_root: Path) -> dict[str, tuple[Path, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("support manifest must be a JSON object")
    result: dict[str, tuple[Path, ...]] = {}
    for object_name in OBJECTS:
        raw = payload.get(object_name)
        if not isinstance(raw, list) or len(raw) != 16 or not all(isinstance(x, str) for x in raw):
            raise ValueError(f"{object_name}: expected exactly 16 support paths")
        rebased = tuple(data_root / object_name / "train" / "good" / Path(x).name for x in raw)
        if len({path.resolve() for path in rebased}) != 16:
            raise ValueError(f"{object_name}: support paths must be unique")
        missing = [str(item) for item in rebased if not item.is_file()]
        if missing:
            raise FileNotFoundError(f"{object_name}: missing support paths: {missing[:3]}")
        result[object_name] = rebased
    return result


def extract_pair(backbone: DINOv2Backbone, image_path: Path) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    image_tensor, grid = backbone.prepare_image(str(image_path))
    features = backbone.extract_features(image_tensor)
    if len(features) != 2 or features[0].shape != features[1].shape:
        raise RuntimeError(f"invalid paired feature shapes for {image_path}: {[x.shape for x in features]}")
    expected = int(grid[0]) * int(grid[1])
    if features[0].ndim != 2 or features[0].shape[0] != expected:
        raise RuntimeError(f"feature/grid mismatch for {image_path}: {features[0].shape} vs {grid}")
    first = torch.as_tensor(features[0], dtype=torch.float32, device=backbone.device)
    second = torch.as_tensor(features[1], dtype=torch.float32, device=backbone.device)
    return first, second, (int(grid[0]), int(grid[1]))


def select_test_paths(
    paths: Iterable[str],
    limit: int,
    source_ids: tuple[str, ...],
) -> tuple[Path, ...]:
    selected = tuple(Path(item) for item in sorted(paths))
    if source_ids:
        selected = tuple(item for item in selected if item.stem.partition("_")[0] in source_ids)
    return selected[:limit] if limit > 0 else selected


def condition_from_stem(stem: str) -> str:
    head, separator, tail = stem.partition("_")
    if separator and head.isdigit() and tail:
        return tail
    return "unknown"


def structured_support_permutation(
    grids: list[tuple[int, int]],
    object_name: str,
    device: str,
) -> torch.Tensor:
    """Break true pairing while preserving image and toroidal spatial marginals."""
    if len(grids) < 2 or len(set(grids)) != 1:
        raise ValueError("structured shuffle requires at least two equal support grids")
    height, width = grids[0]
    digest = hashlib.sha256(f"sdig-nodyr-20260717:{object_name}".encode()).digest()
    y_offset = 1 + int.from_bytes(digest[:4], "big") % max(1, height - 1)
    x_offset = 1 + int.from_bytes(digest[4:8], "big") % max(1, width - 1)
    indices = torch.arange(len(grids) * height * width, dtype=torch.long).reshape(
        len(grids), height, width,
    )
    return indices.roll(shifts=(-1, y_offset, x_offset), dims=(0, 1, 2)).reshape(-1).to(device)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else None


def summarize_score_tails(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in VARIANTS:
        variant_result: dict[str, Any] = {}
        for split in ("good", "bad"):
            split_result: dict[str, Any] = {}
            conditions = sorted({str(row["condition"]) for row in records if row["split"] == split})
            for condition in conditions:
                selected = [
                    row for row in records
                    if row["split"] == split and row["condition"] == condition
                ]
                split_result[condition] = {
                    "image_count": len(selected),
                    "mean_image_score_mean": float(np.mean([row[f"{variant}_mean"] for row in selected])),
                    "mean_image_score_p95": float(np.mean([row[f"{variant}_p95"] for row in selected])),
                    "mean_image_score_max": float(np.mean([row[f"{variant}_max"] for row in selected])),
                }
            variant_result[split] = split_result
        result[variant] = variant_result
    return result


def evaluate_conditions(
    data_root: Path,
    object_name: str,
    records: list[dict[str, Any]],
    variant: str,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["condition"]), []).append(record)
    result: dict[str, dict[str, float]] = {}
    for condition, items in sorted(grouped.items()):
        gt_paths: list[str | None] = []
        prediction_paths: list[str] = []
        for item in items:
            split = str(item["split"])
            stem = str(item["stem"])
            if split == "bad":
                gt = data_root / object_name / "test_public" / "ground_truth" / "bad" / f"{stem}_mask.png"
                if not gt.is_file():
                    raise FileNotFoundError(gt)
                gt_paths.append(str(gt))
            else:
                gt_paths.append(None)
            prediction_paths.append(str(Path(item[f"map_{variant}"]).with_suffix("")))
        if not any(item is not None for item in gt_paths):
            continue
        auroc, f1, threshold = eval_segmentation(
            gt_paths,
            prediction_paths,
            pro_integration_limit=0.05,
            delete_tiff_files=False,
        )
        result[condition] = {
            "seg_AUROC_0.05": float(auroc),
            "seg_F1": float(f1),
            "best_threshold": float(threshold),
            "image_count": len(items),
        }
    return result


def materialize_variant_maps(
    output_root: Path,
    object_name: str,
    records: list[dict[str, Any]],
    variant: str,
) -> None:
    """Expand compact patch scores for one variant, bounding peak disk use."""
    for record in records:
        cache_path = Path(str(record["score_cache"]))
        with np.load(cache_path, allow_pickle=False) as cached:
            patch_scores = np.asarray(cached[variant], dtype=np.float32)
        grid = (int(record["grid_height"]), int(record["grid_width"]))
        if patch_scores.shape != (grid[0] * grid[1],):
            raise RuntimeError(f"{cache_path}/{variant}: cached patch-score shape mismatch")
        score_map = cv2.resize(
            patch_scores.reshape(grid),
            (int(record["image_width"]), int(record["image_height"])),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32, copy=False)
        if not np.all(np.isfinite(score_map)):
            raise RuntimeError(f"{cache_path}/{variant}: non-finite materialized map")
        map_path = (
            output_root
            / "variants"
            / variant
            / "anomaly_maps"
            / object_name
            / "test"
            / str(record["split"])
            / f"{record['stem']}.tiff"
        )
        map_path.parent.mkdir(parents=True, exist_ok=True)
        tiff.imwrite(map_path, score_map)
        record[f"map_{variant}"] = str(map_path)
        record[f"map_sha256_{variant}"] = hashlib.sha256(map_path.read_bytes()).hexdigest()


def run_object(
    args: argparse.Namespace,
    dataset: MVTecAD2Dataset,
    backbone: DINOv2Backbone,
    support_paths: tuple[Path, ...],
    object_name: str,
) -> dict[str, Any]:
    object_info = next(item for item in dataset.get_objects() if item.name == object_name)
    backbone.set_resolution(object_info.resolution)
    support_first: list[torch.Tensor] = []
    support_second: list[torch.Tensor] = []
    support_grids: list[tuple[int, int]] = []
    for path in support_paths:
        first, second, grid = extract_pair(backbone, path)
        support_first.append(first)
        support_second.append(second)
        support_grids.append(grid)
    first_all = torch.cat(support_first, dim=0)
    second_all = torch.cat(support_second, dim=0)
    shuffle_permutation = structured_support_permutation(
        support_grids, object_name, str(first_all.device),
    )
    models = fit_nodyr_models(
        first_all,
        second_all,
        ridge=args.ridge,
        seed=args.seed,
        shuffle_permutation=shuffle_permutation,
    )

    tests = dataset.get_test_images(object_name, split="test_public")
    selected = {
        "good": select_test_paths(tests.get("good", ()), args.max_good, args.source_ids),
        "bad": select_test_paths(tests.get("bad", ()), args.max_bad, args.source_ids),
    }
    if not selected["good"] or not selected["bad"]:
        raise RuntimeError(f"{object_name}: both good and bad test populations are required")

    output_root = Path(args.output_root)
    score_cache_root = output_root / "patch_score_cache" / object_name
    rows: list[dict[str, Any]] = []
    max_equivalence_error = 0.0
    for split in ("good", "bad"):
        for image_path in selected[split]:
            first, second, grid = extract_pair(backbone, image_path)
            scores = score_nodyr_models(models, first, second)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"failed to read query image: {image_path}")
            row: dict[str, Any] = {
                "object": object_name,
                "split": split,
                "condition": condition_from_stem(image_path.stem),
                "stem": image_path.stem,
                "grid_height": grid[0],
                "grid_width": grid[1],
                "image_height": int(image.shape[0]),
                "image_width": int(image.shape[1]),
            }
            transition_norm = scores.pop("raw_transition_norm").detach().float().cpu().numpy()
            candidate = scores["nodyr"].detach().float().cpu().numpy()
            row["transition_norm_mean"] = float(np.mean(transition_norm))
            row["transition_norm_p05"] = float(np.percentile(transition_norm, 5))
            row["nodyr_transition_norm_corr"] = safe_correlation(candidate, transition_norm)
            cache_payload: dict[str, np.ndarray] = {}
            for variant in VARIANTS:
                patch_scores = scores[variant].detach().float().cpu().numpy()
                if patch_scores.shape != (grid[0] * grid[1],) or not np.all(np.isfinite(patch_scores)):
                    raise RuntimeError(f"{object_name}/{image_path.name}/{variant}: invalid patch scores")
                if float(np.ptp(patch_scores)) <= 1e-8:
                    raise RuntimeError(f"{object_name}/{image_path.name}/{variant}: collapsed patch scores")
                if variant == "nodyr" and (
                    float(np.min(patch_scores)) < -2e-5 or float(np.max(patch_scores)) > 2.0 + 2e-5
                ):
                    raise RuntimeError(f"{object_name}/{image_path.name}: NODyR score outside [0,2]")
                cache_payload[variant] = patch_scores.astype(np.float32, copy=False)
                row[f"{variant}_mean"] = float(np.mean(patch_scores))
                row[f"{variant}_p95"] = float(np.percentile(patch_scores, 95))
                row[f"{variant}_max"] = float(np.max(patch_scores))
            cache_path = score_cache_root / split / f"{image_path.stem}.npz"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, **cache_payload)
            row["score_cache"] = str(cache_path)
            equivalence = np.max(
                np.abs(
                    scores["unit_delta_l2_equiv"].detach().float().cpu().numpy()
                    - 2.0 * candidate
                ),
            )
            max_equivalence_error = max(max_equivalence_error, float(equivalence))
            rows.append(row)

    if max_equivalence_error > 2e-5:
        raise RuntimeError(f"unit-delta equivalence audit failed: {max_equivalence_error}")
    full_population = args.max_good == 0 and args.max_bad == 0 and not args.source_ids
    variant_metrics: dict[str, Any] = {}
    condition_metrics: dict[str, Any] = {}
    evaluator = Evaluator({"pro_integration_limit": 0.05, "binary_postprocess": "none"})
    for variant in VARIANTS:
        materialize_variant_maps(output_root, object_name, rows, variant)
        variant_root = output_root / "variants" / variant
        if full_population:
            metrics = evaluator.evaluate_run(
                dataset_name="MVTec_AD_2",
                data_root=str(args.data_root),
                anomaly_maps_dir=str(variant_root / "anomaly_maps"),
                output_dir=str(variant_root),
                seed=args.seed,
                objects=[object_name],
            )
            (variant_root / "metrics.json").write_text(
                json.dumps(metrics, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            variant_metrics[variant] = metrics[object_name]
        if args.cleanup_maps:
            maps = variant_root / "anomaly_maps"
            if maps.exists():
                shutil.rmtree(maps)
    write_rows(output_root / f"{object_name}_per_image.tsv", rows)
    score_cache_parent = output_root / "patch_score_cache"
    if score_cache_parent.exists():
        shutil.rmtree(score_cache_parent)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_dataset": "MVTec AD2 single-image",
        "data_root": str(args.data_root),
        "object": object_name,
        "split": "test_public/good,bad",
        "full_population": full_population,
        "source_ids": list(args.source_ids),
        "test_good_count": len(selected["good"]),
        "test_bad_count": len(selected["bad"]),
        "map_count": len(rows) * len(VARIANTS),
        "finite_map_coverage": 1.0,
        "map_materialization": "one_variant_at_a_time_float32_tiff",
        "peak_materialized_variant_count": 1,
        "backbone": args.backbone_model,
        "feature_layers": list(args.layers),
        "resolution": object_info.resolution,
        "ridge": args.ridge,
        "seed": args.seed,
        "support_count": len(support_paths),
        "support_patch_count": int(first_all.shape[0]),
        "support_grid": list(support_grids[0]),
        "support_paths": [str(item) for item in support_paths],
        "support_manifest": str(args.support_json),
        "host_gpu_id": str(args.host_gpu_id),
        "container_cuda_slot": str(args.container_cuda_slot),
        "variants": list(VARIANTS),
        "independent_nulls": ["state_angle", "state_l2", "raw_delta_l2", "shuffle_nodyr"],
        "shuffle_protocol": "cyclic_cross_image_plus_sha256_toroidal_spatial_offset",
        "shuffle_permutation_sha256": hashlib.sha256(
            shuffle_permutation.detach().cpu().numpy().tobytes(),
        ).hexdigest(),
        "equivalence_control": "unit_delta_l2_equiv == 2 * nodyr",
        "max_equivalence_error": max_equivalence_error,
        "variant_metrics": variant_metrics,
        "condition_metrics": condition_metrics,
        "condition_metrics_role": "not_run_unpreregistered_duplicate_pixel_evaluation",
        "condition_score_tails": summarize_score_tails(rows),
        "score_calibration": "none",
        "fusion": "none",
        "query_dependent_update": False,
        "ground_truth_used_by_scorer": False,
        "evaluation_role": "test_public_shadow_only",
    }
    summary_path = output_root / f"{object_name}_object_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.cleanup_maps:
        (output_root / f"{object_name}_cleanup_evidence.txt").write_text(
            "cleanup_anomaly_maps=true\n",
            encoding="utf-8",
        )
    del models, first_all, second_all
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for remote metric execution")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    support_path = Path(args.support_json)
    support = load_support_manifest(support_path, Path(args.data_root))
    dataset = MVTecAD2Dataset(
        args.data_root,
        {"objects": list(args.objects), "preprocess": "no_mask_no_rotation"},
    )
    backbone = DINOv2Backbone(
        model_name=args.backbone_model,
        device=args.device,
        smaller_edge_size=672,
        feature_layers=args.layers,
    )
    summaries = [
        run_object(args, dataset, backbone, support[object_name], object_name)
        for object_name in args.objects
    ]
    run_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "support_manifest_sha256": hashlib.sha256(support_path.read_bytes()).hexdigest(),
        "objects": list(args.objects),
        "host_gpu_id": str(args.host_gpu_id),
        "container_cuda_slot": str(args.container_cuda_slot),
        "summaries": [f"{item['object']}_object_summary.json" for item in summaries],
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
