#!/usr/bin/env python3
"""Re-evaluate preserved Same16 dense maps with the isolated snapshot evaluator."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

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
TARGETS = {
    "basic": Path("basic/objects"),
    "a6": Path("mechanism/arms/a6/objects"),
    "a7": Path("mechanism/arms/a7/objects"),
    "a6_l17": Path("mechanism/arms/a6_l17/objects"),
    "l17": Path("mechanism/arms/l17/objects"),
}
REQUIRED_METRICS = ("seg_AUROC_0.05", "seg_AUPRO_0.05", "seg_F1_raw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--object", choices=OBJECTS, required=True)
    parser.add_argument("--target", choices=tuple(TARGETS))
    return parser.parse_args()


def import_snapshot_evaluator(snapshot: Path) -> type:
    snapshot = snapshot.resolve()
    sys.path.insert(0, str(snapshot))
    metrics_module = importlib.import_module("fmad.evaluation.metrics")
    post_eval_module = importlib.import_module("src.post_eval")
    for module in (metrics_module, post_eval_module):
        module_path = Path(module.__file__).resolve()
        if not module_path.is_relative_to(snapshot):
            raise RuntimeError(
                f"evaluator import escaped snapshot: {module.__name__}={module_path}",
            )
    return metrics_module.Evaluator


def validate_payload(payload: dict[str, Any], object_name: str) -> None:
    row = payload.get(object_name)
    if not isinstance(row, dict):
        raise ValueError(f"missing object metrics: {object_name}")
    for metric in REQUIRED_METRICS:
        value = row.get(metric)
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"invalid {object_name}/{metric}: {value}")
    if row.get("seg_F1") != row.get("seg_F1_raw"):
        raise ValueError(f"raw F1 contract violated for {object_name}")
    coverage = row.get("coverage")
    if not isinstance(coverage, dict):
        raise ValueError(f"missing coverage for {object_name}")
    expected = coverage.get("ground_truth_map_count")
    for key in ("prediction_map_count", "shape_match_count"):
        if coverage.get(key) != expected:
            raise ValueError(f"coverage mismatch for {object_name}: {key}")
    if not isinstance(expected, int) or expected <= 0:
        raise ValueError(f"empty evaluation population for {object_name}")
    if coverage.get("non_finite_prediction_count") != 0:
        raise ValueError(f"non-finite predictions for {object_name}")


def preserve_legacy_metrics(output_root: Path) -> None:
    for filename in ("metrics.json", "metrics_seed=0.json"):
        source = output_root / filename
        legacy = source.with_name(f"{source.stem}.legacy_workspace_evaluator.json")
        if source.is_file() and not legacy.exists():
            shutil.copy2(source, legacy)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def evaluate_target(
    evaluator_cls: type,
    root: Path,
    data_root: Path,
    object_name: str,
    target: str,
) -> dict[str, Any]:
    output_root = root / TARGETS[target] / object_name
    anomaly_maps = output_root / "anomaly_maps"
    if not anomaly_maps.is_dir():
        raise FileNotFoundError(f"missing dense maps: {anomaly_maps}")
    preserve_legacy_metrics(output_root)
    evaluator = evaluator_cls({
        "pro_integration_limit": 0.05,
        "binary_postprocess": "none",
    })
    with tempfile.TemporaryDirectory(
        prefix=".reevaluation-",
        dir=output_root,
    ) as temporary:
        payload = evaluator.evaluate_run(
            dataset_name="MVTec_AD_2",
            data_root=str(data_root),
            anomaly_maps_dir=str(anomaly_maps),
            output_dir=temporary,
            seed=0,
            objects=[object_name],
        )
        validate_payload(payload, object_name)
        atomic_write_json(output_root / "metrics_seed=0.json", payload)
        atomic_write_json(output_root / "metrics.json", payload)
    return payload[object_name]


def main() -> int:
    args = parse_args()
    evaluator_cls = import_snapshot_evaluator(args.snapshot)
    summary: dict[str, Any] = {
        "object": args.object,
        "snapshot": str(args.snapshot.resolve()),
        "targets": {},
    }
    selected_targets = (args.target,) if args.target else tuple(TARGETS)
    for target in selected_targets:
        print(f"=== Re-evaluate {target}/{args.object} ===", flush=True)
        row = evaluate_target(
            evaluator_cls,
            args.root,
            args.data_root,
            args.object,
            target,
        )
        summary["targets"][target] = {
            metric: row[metric] for metric in REQUIRED_METRICS
        }
    if args.target:
        marker = (
            args.root
            / "reevaluation"
            / "targets"
            / args.target
            / f"{args.object}.json"
        )
    else:
        marker = args.root / "reevaluation" / f"{args.object}.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(marker, summary)
    print(f"COMPLETE {args.object}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
