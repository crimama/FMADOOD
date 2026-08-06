"""Validate and aggregate the frozen A6/dominant-layer public-shadow run."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

from scripts.aggregate_same16_baselines import (
    FROZEN_SUPPORT_SHA256,
    METRICS,
    OBJECTS,
    canonical_reference_ids,
    read_json,
    selected_reference_paths,
    validate_object_metrics,
)

ARMS = ("basic", "a6", "a7", "a6_l17", "l17")
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 0
COLLAPSE_BOUNDS = {
    "seg_AUROC_0.05": -0.02,
    "seg_F1_raw": -0.03,
    "seg_AUPRO_0.05": -0.02,
}
MATERIAL_BOUNDS = {
    "seg_AUROC_0.05": 0.005,
    "seg_F1_raw": 0.010,
    "seg_AUPRO_0.05": 0.010,
}
ARM_CONTRACTS: Dict[str, Dict[str, Any]] = {
    "basic": {
        "backbone": "dinov2_vitl14 layer mean [5, 11, 17, 23]",
        "normality_mode": "fused",
        "flow_transform_mode": "flow",
        "dvt_denoise_mode": "position_mean",
        "density_weight": 0.25,
    },
    "a6": {
        "backbone": "dinov2_vitl14 layer mean [5, 11, 17, 23]",
        "normality_mode": "fused",
        "flow_transform_mode": "flow",
        "dvt_denoise_mode": "none",
        "density_weight": 0.0,
    },
    "a7": {
        "backbone": "dinov2_vitl14 layer mean [5, 11, 17, 23]",
        "normality_mode": "raw_nn",
        "flow_transform_mode": "identity",
        "dvt_denoise_mode": "none",
        "density_weight": 0.0,
    },
    "a6_l17": {
        "backbone": "dinov2_vitl14 layer mean [17]",
        "normality_mode": "fused",
        "flow_transform_mode": "flow",
        "dvt_denoise_mode": "none",
        "density_weight": 0.0,
    },
    "l17": {
        "backbone": "dinov2_vitl14 layer mean [17]",
        "normality_mode": "raw_nn",
        "flow_transform_mode": "identity",
        "dvt_denoise_mode": "none",
        "density_weight": 0.0,
    },
}
COMPARISONS = {
    "a6_minus_basic": ("a6", "basic"),
    "a6_minus_a7": ("a6", "a7"),
    "a6_l17_minus_a6": ("a6_l17", "a6"),
    "l17_minus_a7": ("l17", "a7"),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--support-manifest", type=Path, required=True)
    return parser.parse_args(argv)


def arm_run_dir(root: Path, arm: str, object_name: str) -> Path:
    if arm == "basic":
        return root / "basic" / "objects" / object_name
    return root / "mechanism" / "arms" / arm / "objects" / object_name


def _validate_manifest(
    arm: str,
    object_name: str,
    manifest: Dict[str, Any],
    expected_paths: List[str],
    expected_support_sha256: str,
) -> None:
    actual_paths = selected_reference_paths("basic", manifest, object_name)
    if manifest.get("support_manifest_sha256") != expected_support_sha256:
        message = f"{arm}/{object_name}: support manifest SHA-256 mismatch"
        raise ValueError(message)
    if canonical_reference_ids(
        actual_paths,
        object_name,
    ) != canonical_reference_ids(expected_paths, object_name):
        message = f"{arm}/{object_name}: fixed reference order mismatch"
        raise ValueError(message)
    expected = ARM_CONTRACTS[arm]
    for key, value in expected.items():
        if manifest.get(key) != value:
            message = f"{arm}/{object_name}: {key}={manifest.get(key)!r}, expected {value!r}"
            raise ValueError(message)
    common = {
        "shots": 16,
        "seed": 0,
        "feature_fusion": "layer_norm_mean",
        "support_transforms": ["identity"],
        "expansion_budget": 1.0,
        "loo_standardization": "off",
        "rgb_guide": "none",
        "binary_postprocess": "none",
        "cleanup_maps": False,
    }
    for key, value in common.items():
        if manifest.get(key) != value:
            message = f"{arm}/{object_name}: {key}={manifest.get(key)!r}, expected {value!r}"
            raise ValueError(message)


def _percentile(values: List[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_interval(values: List[float]) -> List[float]:
    generator = random.Random(BOOTSTRAP_SEED)  # noqa: S311 - deterministic bootstrap
    count = len(values)
    means = [
        sum(values[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(BOOTSTRAP_REPLICATES)
    ]
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def build_comparison_payload(
    left: str,
    right: str,
    per_arm: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Any]:
    per_object = {
        object_name: {
            metric: per_arm[left][object_name][metric] - per_arm[right][object_name][metric]
            for metric in METRICS
        }
        for object_name in OBJECTS
    }
    macro = {
        metric: sum(per_object[object_name][metric] for object_name in OBJECTS) / len(OBJECTS)
        for metric in METRICS
    }
    intervals = {
        metric: _bootstrap_interval(
            [per_object[object_name][metric] for object_name in OBJECTS],
        )
        for metric in METRICS
    }
    positive_objects = {
        metric: sum(per_object[object_name][metric] > 0.0 for object_name in OBJECTS)
        for metric in METRICS
    }
    collapsed_objects = {
        metric: [
            object_name
            for object_name in OBJECTS
            if per_object[object_name][metric] < COLLAPSE_BOUNDS[metric]
        ]
        for metric in METRICS
    }
    supported_metrics = [
        metric
        for metric in METRICS
        if macro[metric] >= MATERIAL_BOUNDS[metric]
        and intervals[metric][0] > 0.0
        and positive_objects[metric] >= 6
        and not collapsed_objects[metric]
    ]
    return {
        "left": left,
        "right": right,
        "per_object": per_object,
        "macro": macro,
        "object_bootstrap_95ci": intervals,
        "positive_object_count": positive_objects,
        "collapsed_objects": collapsed_objects,
        "supported_metrics": supported_metrics,
    }


def build_gates(comparisons: Dict[str, Dict[str, Any]]) -> Dict[str, bool]:
    """Apply the frozen promotion gates to independently checkable evidence."""
    simplification = comparisons["a6_minus_basic"]
    flow_without_dvt = comparisons["a6_minus_a7"]
    dominant_l17 = comparisons["a6_l17_minus_a6"]
    a6_safe = (
        simplification["macro"]["seg_AUROC_0.05"] >= -0.005
        and simplification["macro"]["seg_F1_raw"] >= 0.010
        and simplification["macro"]["seg_AUPRO_0.05"] >= -0.005
        and not any(simplification["collapsed_objects"].values())
    )
    flow_without_dvt_supported = bool(flow_without_dvt["supported_metrics"]) and not any(
        flow_without_dvt["collapsed_objects"].values(),
    )
    dominant_l17_supported = bool(dominant_l17["supported_metrics"]) and not any(
        dominant_l17["collapsed_objects"].values(),
    )
    return {
        "a6_simplification_safe": a6_safe,
        "flow_without_dvt_supported": flow_without_dvt_supported,
        "dominant_l17_repair_supported": dominant_l17_supported,
    }


def aggregate(
    root: Path,
    support_manifest_path: Path,
    expected_support_sha256: str = FROZEN_SUPPORT_SHA256,
) -> Dict[str, Any]:
    support_bytes = support_manifest_path.read_bytes()
    support_sha256 = hashlib.sha256(support_bytes).hexdigest()
    if support_sha256 != expected_support_sha256:
        raise ValueError("support manifest does not match the frozen SHA-256")
    support_payload = json.loads(support_bytes)
    if not isinstance(support_payload, dict) or set(support_payload) != set(OBJECTS):
        raise ValueError("support manifest must contain exactly all eight objects")
    per_arm: Dict[str, Dict[str, Dict[str, float]]] = {}
    for arm in ARMS:
        per_object: Dict[str, Dict[str, float]] = {}
        for object_name in OBJECTS:
            run_dir = arm_run_dir(root, arm, object_name)
            metrics = read_json(run_dir / "metrics.json")
            manifest = read_json(run_dir / "run_manifest.json")
            expected_paths = support_payload[object_name]
            if not isinstance(expected_paths, list) or not all(
                isinstance(path, str) for path in expected_paths
            ):
                message = f"{object_name}: invalid support manifest row"
                raise ValueError(message)
            _validate_manifest(
                arm,
                object_name,
                manifest,
                expected_paths,
                support_sha256,
            )
            per_object[object_name] = validate_object_metrics(
                arm,
                object_name,
                metrics,
            )
        per_arm[arm] = per_object

    means = {
        arm: {
            metric: sum(per_arm[arm][object_name][metric] for object_name in OBJECTS) / len(OBJECTS)
            for metric in METRICS
        }
        for arm in ARMS
    }
    comparisons = {
        name: build_comparison_payload(left, right, per_arm)
        for name, (left, right) in COMPARISONS.items()
    }
    gates = build_gates(comparisons)
    return {
        "status": "COMPLETE_PUBLIC_SHADOW",
        "claim_scope": "MVTec AD2 test_public public-shadow development evidence",
        "objects": list(OBJECTS),
        "support_manifest": str(support_manifest_path),
        "support_manifest_sha256": support_sha256,
        "arms": list(ARMS),
        "arm_contracts": ARM_CONTRACTS,
        "metric_contract": {
            "seg_AUROC_0.05": "sklearn standardized partial pixel AUROC",
            "seg_F1_raw": "pooled original-resolution oracle best-threshold F1",
            "seg_AUPRO_0.05": "connected-component PRO normalized over FPR [0,0.05]",
        },
        "bootstrap": {
            "unit": "object",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "per_arm": per_arm,
        "means": means,
        "comparisons": comparisons,
        "gates": gates,
    }


def write_outputs(root: Path, payload: Dict[str, Any]) -> None:
    (root / "a6_fusion_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = ["arm\tseg_AUROC_0.05\tseg_F1_raw\tseg_AUPRO_0.05"]
    means = payload["means"]
    lines.extend(
        [
            "\t".join(
                [
                    arm,
                    f"{means[arm]['seg_AUROC_0.05']:.9f}",
                    f"{means[arm]['seg_F1_raw']:.9f}",
                    f"{means[arm]['seg_AUPRO_0.05']:.9f}",
                ],
            )
            for arm in ARMS
        ],
    )
    (root / "a6_fusion_summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = aggregate(args.root, args.support_manifest)
    write_outputs(args.root, payload)
    print(json.dumps(payload["gates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
