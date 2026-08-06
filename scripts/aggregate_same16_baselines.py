#!/usr/bin/env python3
"""Validate and aggregate the frozen same-reference AD2 baseline run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

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
METHODS = ("basic", "superad", "rn_fmlk")
METRICS = ("seg_AUROC_0.05", "seg_F1_raw", "seg_AUPRO_0.05")
FROZEN_SUPPORT_SHA256 = "7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--support-manifest", type=Path, required=True)
    return parser.parse_args(argv)


def read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def method_run_dir(root: Path, method: str, object_name: str) -> Path:
    object_root = root / method / "objects" / object_name
    if method == "basic":
        return object_root
    return object_root / method / "mvtec_ad2" / "dinov2_vitl14" / "seed=0"


def selected_reference_paths(
    method: str,
    manifest: Dict[str, Any],
    object_name: str,
) -> List[str]:
    if method == "basic":
        diagnostics = manifest.get("object_diagnostics")
        if not isinstance(diagnostics, list) or len(diagnostics) != 1:
            raise ValueError(f"{method}/{object_name}: invalid object diagnostics")
        paths = diagnostics[0].get("selected_support_paths")
    else:
        objects = manifest.get("objects")
        if not isinstance(objects, dict) or object_name not in objects:
            raise ValueError(f"{method}/{object_name}: missing object metadata")
        paths = objects[object_name].get("selected_reference_paths")
    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
        raise ValueError(f"{method}/{object_name}: invalid selected reference paths")
    return paths


def canonical_reference_ids(paths: List[str], object_name: str) -> List[str]:
    """Return root-independent, split-strict identities for support paths."""
    identities = []
    expected_suffix = (object_name, "train", "good")
    for value in paths:
        parts = Path(value).parts
        if len(parts) < 4 or tuple(parts[-4:-1]) != expected_suffix:
            raise ValueError(
                f"{object_name}: support reference is outside {object_name}/train/good: {value}",
            )
        identities.append("/".join((*expected_suffix, parts[-1])))
    return identities


def validate_object_metrics(
    method: str,
    object_name: str,
    metrics: Dict[str, Any],
) -> Dict[str, float]:
    row = metrics.get(object_name)
    if not isinstance(row, dict):
        raise ValueError(f"{method}/{object_name}: missing metric row")
    values = {}
    for metric in METRICS:
        value = row.get(metric)
        if not isinstance(value, (float, int)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{method}/{object_name}: invalid {metric}: {value}")
        values[metric] = float(value)
    if row.get("seg_F1") != row.get("seg_F1_raw"):
        raise ValueError(f"{method}/{object_name}: strict row must use raw F1")
    coverage = row.get("coverage")
    if not isinstance(coverage, dict):
        raise ValueError(f"{method}/{object_name}: missing coverage")
    prediction_count = coverage.get("prediction_map_count")
    if not isinstance(prediction_count, int) or prediction_count <= 0:
        raise ValueError(f"{method}/{object_name}: invalid prediction coverage")
    if coverage.get("ground_truth_map_count") != prediction_count:
        raise ValueError(f"{method}/{object_name}: GT/prediction count mismatch")
    if coverage.get("shape_match_count") != prediction_count:
        raise ValueError(f"{method}/{object_name}: map shape mismatch")
    if coverage.get("nonconstant_prediction_map_count") != prediction_count:
        raise ValueError(f"{method}/{object_name}: constant prediction map")
    if coverage.get("non_finite_prediction_count") != 0:
        raise ValueError(f"{method}/{object_name}: non-finite predictions")
    values["prediction_map_count"] = float(prediction_count)
    return values


def aggregate(
    root: Path,
    support_manifest_path: Path,
    expected_support_sha256: str = FROZEN_SUPPORT_SHA256,
) -> Dict[str, Any]:
    support_bytes = support_manifest_path.read_bytes()
    support_payload = json.loads(support_bytes)
    if not isinstance(support_payload, dict) or set(support_payload) != set(OBJECTS):
        raise ValueError("support manifest must contain exactly all eight objects")
    support_sha256 = hashlib.sha256(support_bytes).hexdigest()
    if support_sha256 != expected_support_sha256:
        raise ValueError("support manifest does not match the frozen SHA-256")
    per_method: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method in METHODS:
        per_object: Dict[str, Dict[str, float]] = {}
        for object_name in OBJECTS:
            run_dir = method_run_dir(root, method, object_name)
            metrics = read_json(run_dir / "metrics.json")
            manifest = read_json(run_dir / "run_manifest.json")
            actual_paths = selected_reference_paths(method, manifest, object_name)
            if method == "basic":
                actual_support_sha256 = manifest.get("support_manifest_sha256")
            else:
                actual_support_sha256 = manifest["objects"][object_name].get(
                    "reference_manifest_sha256",
                )
            if actual_support_sha256 != support_sha256:
                raise ValueError(
                    f"{method}/{object_name}: support manifest SHA-256 mismatch",
                )
            expected_paths = support_payload[object_name]
            if not isinstance(expected_paths, list) or not all(
                isinstance(value, str) for value in expected_paths
            ):
                raise ValueError(f"{object_name}: invalid support manifest row")
            if canonical_reference_ids(
                actual_paths,
                object_name,
            ) != canonical_reference_ids(expected_paths, object_name):
                raise ValueError(
                    f"{method}/{object_name}: fixed reference order mismatch",
                )
            per_object[object_name] = validate_object_metrics(
                method,
                object_name,
                metrics,
            )
        per_method[method] = per_object

    means = {
        method: {
            metric: sum(per_method[method][obj][metric] for obj in OBJECTS) / len(OBJECTS)
            for metric in METRICS
        }
        for method in METHODS
    }
    deltas = {
        baseline: {metric: means["basic"][metric] - means[baseline][metric] for metric in METRICS}
        for baseline in ("superad", "rn_fmlk")
    }
    return {
        "status": "COMPLETE_COMPARABLE_PUBLIC_SHADOW",
        "claim_scope": "MVTec AD2 test_public public-shadow development evidence",
        "objects": list(OBJECTS),
        "support_manifest": str(support_manifest_path),
        "support_manifest_sha256": support_sha256,
        "metric_contract": {
            "seg_AUROC_0.05": "sklearn standardized partial pixel AUROC",
            "seg_F1_raw": "pooled original-resolution oracle best-threshold F1",
            "seg_AUPRO_0.05": "connected-component PRO normalized over FPR [0,0.05]",
        },
        "per_method": per_method,
        "means": means,
        "basic_minus_baseline": deltas,
        "comparable": True,
    }


def write_outputs(root: Path, payload: Dict[str, Any]) -> None:
    (root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = ["method\tseg_AUROC_0.05\tseg_F1_raw\tseg_AUPRO_0.05"]
    means = payload["means"]
    for method in METHODS:
        lines.append(
            "\t".join(
                [
                    method,
                    f"{means[method]['seg_AUROC_0.05']:.9f}",
                    f"{means[method]['seg_F1_raw']:.9f}",
                    f"{means[method]['seg_AUPRO_0.05']:.9f}",
                ]
            ),
        )
    (root / "summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = aggregate(args.root, args.support_manifest)
    write_outputs(args.root, payload)
    print(json.dumps(payload["means"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
