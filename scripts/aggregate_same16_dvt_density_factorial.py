#!/usr/bin/env python3
"""Validate and aggregate the Same16 DVT x density factorial diagnostic."""

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

ARMS = ("a6", "dvt_only", "density_only", "basic")
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
    "a6": {
        "backbone": "dinov2_vitl14 layer mean [5, 11, 17, 23]",
        "normality_mode": "fused",
        "flow_transform_mode": "flow",
        "dvt_denoise_mode": "none",
        "density_weight": 0.0,
    },
    "dvt_only": {
        "backbone": "dinov2_vitl14 layer mean [5, 11, 17, 23]",
        "normality_mode": "fused",
        "flow_transform_mode": "flow",
        "dvt_denoise_mode": "position_mean",
        "density_weight": 0.0,
    },
    "density_only": {
        "backbone": "dinov2_vitl14 layer mean [5, 11, 17, 23]",
        "normality_mode": "fused",
        "flow_transform_mode": "flow",
        "dvt_denoise_mode": "none",
        "density_weight": 0.25,
    },
    "basic": {
        "backbone": "dinov2_vitl14 layer mean [5, 11, 17, 23]",
        "normality_mode": "fused",
        "flow_transform_mode": "flow",
        "dvt_denoise_mode": "position_mean",
        "density_weight": 0.25,
    },
}
COMMON_CONTRACT: Dict[str, Any] = {
    "shots": 16,
    "seed": 0,
    "feature_fusion": "layer_norm_mean",
    "support_transforms": ["identity"],
    "support_selection_seed": 0,
    "flow_epochs": 3,
    "coupling_layers": 2,
    "hidden_multiplier": 1,
    "flow_lr": 0.0002,
    "flow_clamp": 1.9,
    "tail_weight": 0.3,
    "tail_top_k_ratio": 0.05,
    "lambda_logdet": 0.001,
    "density_quantile": 0.9,
    "expansion_budget": 1.0,
    "distance_weight": 1.0,
    "score_mode": "latent_distance",
    "top_percent": 0.01,
    "query_chunk_size": 512,
    "calibration_sample_size": 0,
    "loo_standardization": "off",
    "latent_bank_subsample": "none",
    "dvt_denoise_alpha": 1.0,
    "context_source": "none",
    "flow_context_source": "auto",
    "memory_context_source": "auto",
    "context_mode": "none",
    "context_weight": 0.0,
    "context_top_m": 1,
    "flow_condition_mode": "none",
    "transformer_context_mode": "none",
    "rgb_guide": "none",
    "threshold_calibration_mode": "none",
    "binary_postprocess": "none",
    "cleanup_maps": False,
}
EFFECTS = {
    "dvt_effect_density0": {"terms": (("dvt_only", 1.0), ("a6", -1.0))},
    "dvt_effect_density025": {"terms": (("basic", 1.0), ("density_only", -1.0))},
    "density_effect_no_dvt": {"terms": (("density_only", 1.0), ("a6", -1.0))},
    "density_effect_dvt": {"terms": (("basic", 1.0), ("dvt_only", -1.0))},
    "dvt_average_main_effect": {
        "terms": (
            ("dvt_only", 0.5),
            ("basic", 0.5),
            ("a6", -0.5),
            ("density_only", -0.5),
        ),
    },
    "density_average_main_effect": {
        "terms": (
            ("density_only", 0.5),
            ("basic", 0.5),
            ("a6", -0.5),
            ("dvt_only", -0.5),
        ),
    },
    "interaction": {
        "terms": (
            ("basic", 1.0),
            ("dvt_only", -1.0),
            ("density_only", -1.0),
            ("a6", 1.0),
        ),
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factorial-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--support-manifest", type=Path, required=True)
    return parser.parse_args(argv)


def arm_run_dir(factorial_root: Path, baseline_root: Path, arm: str, object_name: str) -> Path:
    if arm == "basic":
        return baseline_root / "basic" / "objects" / object_name
    if arm == "a6":
        return baseline_root / "mechanism" / "arms" / "a6" / "objects" / object_name
    return factorial_root / "mechanism" / "arms" / arm / "objects" / object_name


def _validate_manifest(
    arm: str,
    object_name: str,
    manifest: Dict[str, Any],
    expected_paths: List[str],
    expected_support_sha256: str,
) -> None:
    actual_paths = selected_reference_paths("basic", manifest, object_name)
    if manifest.get("support_manifest_sha256") != expected_support_sha256:
        raise ValueError(f"{arm}/{object_name}: support manifest SHA-256 mismatch")
    if canonical_reference_ids(actual_paths, object_name) != canonical_reference_ids(
        expected_paths,
        object_name,
    ):
        raise ValueError(f"{arm}/{object_name}: fixed reference order mismatch")
    expected = ARM_CONTRACTS[arm]
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"{arm}/{object_name}: {key}={manifest.get(key)!r}, expected {value!r}",
            )
    for key, value in COMMON_CONTRACT.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"{arm}/{object_name}: {key}={manifest.get(key)!r}, expected {value!r}",
            )


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


def build_effect_payload(
    terms: Sequence[tuple[str, float]],
    per_arm: Dict[str, Dict[str, Dict[str, float]]],
    *,
    interaction: bool = False,
) -> Dict[str, Any]:
    per_object = {
        object_name: {
            metric: sum(
                coefficient * per_arm[arm][object_name][metric]
                for arm, coefficient in terms
            )
            for metric in METRICS
        }
        for object_name in OBJECTS
    }
    macro = {
        metric: sum(per_object[object_name][metric] for object_name in OBJECTS) / len(OBJECTS)
        for metric in METRICS
    }
    intervals = {
        metric: _bootstrap_interval([per_object[object_name][metric] for object_name in OBJECTS])
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
    any_cross_metric_collapse = any(collapsed_objects[metric] for metric in METRICS)
    if interaction:
        supported_metrics = [
            metric
            for metric in METRICS
            if abs(macro[metric]) >= MATERIAL_BOUNDS[metric]
            and (intervals[metric][0] > 0.0 or intervals[metric][1] < 0.0)
        ]
    else:
        supported_metrics = [
            metric
            for metric in METRICS
            if macro[metric] >= MATERIAL_BOUNDS[metric]
            and intervals[metric][0] > 0.0
            and positive_objects[metric] >= 6
            and not any_cross_metric_collapse
        ]
    return {
        "terms": [{"arm": arm, "coefficient": coefficient} for arm, coefficient in terms],
        "per_object": per_object,
        "macro": macro,
        "object_bootstrap_95ci": intervals,
        "positive_object_count": positive_objects,
        "collapsed_objects": collapsed_objects,
        "supported_metrics": supported_metrics,
    }


def aggregate(
    factorial_root: Path,
    baseline_root: Path,
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
            run_dir = arm_run_dir(factorial_root, baseline_root, arm, object_name)
            metrics = read_json(run_dir / "metrics.json")
            manifest = read_json(run_dir / "run_manifest.json")
            expected_paths = support_payload[object_name]
            if not isinstance(expected_paths, list) or not all(
                isinstance(path, str) for path in expected_paths
            ):
                raise ValueError(f"{object_name}: invalid support manifest row")
            _validate_manifest(arm, object_name, manifest, expected_paths, support_sha256)
            per_object[object_name] = validate_object_metrics(arm, object_name, metrics)
        per_arm[arm] = per_object

    means = {
        arm: {
            metric: sum(per_arm[arm][object_name][metric] for object_name in OBJECTS)
            / len(OBJECTS)
            for metric in METRICS
        }
        for arm in ARMS
    }
    effects = {
        name: build_effect_payload(
            effect["terms"],
            per_arm,
            interaction=name == "interaction",
        )
        for name, effect in EFFECTS.items()
    }
    factor_credit = {
        "dvt": {
            "effect": "dvt_average_main_effect",
            "supported_metrics": effects["dvt_average_main_effect"]["supported_metrics"],
            "credited": bool(effects["dvt_average_main_effect"]["supported_metrics"]),
        },
        "density": {
            "effect": "density_average_main_effect",
            "supported_metrics": effects["density_average_main_effect"]["supported_metrics"],
            "credited": bool(effects["density_average_main_effect"]["supported_metrics"]),
        },
        "interaction": {
            "effect": "interaction",
            "supported_metrics": effects["interaction"]["supported_metrics"],
            "supported": bool(effects["interaction"]["supported_metrics"]),
        },
    }
    credited_factors = [
        factor
        for factor in ("dvt", "density")
        if factor_credit[factor]["credited"]
    ]
    if factor_credit["interaction"]["supported"]:
        attribution_verdict = "DVT_DENSITY_INTERACTION"
    elif credited_factors == ["dvt"]:
        attribution_verdict = "DVT_MAIN_EFFECT"
    elif credited_factors == ["density"]:
        attribution_verdict = "DENSITY_MAIN_EFFECT"
    else:
        attribution_verdict = "MIXED_OR_INCONCLUSIVE_ATTRIBUTION"
    return {
        "status": "COMPLETE_DVT_DENSITY_FACTORIAL_DIAGNOSTIC",
        "claim_scope": "MVTec AD2 test_public public-shadow diagnostic-only mechanism evidence",
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
        "effects": effects,
        "factor_credit": factor_credit,
        "attribution_verdict": attribution_verdict,
        "strict_method_claim": False,
        "diagnostic_only": True,
    }


def write_outputs(root: Path, payload: Dict[str, Any]) -> None:
    (root / "dvt_density_factorial_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = ["kind\tname\tseg_AUROC_0.05\tseg_F1_raw\tseg_AUPRO_0.05"]
    means = payload["means"]
    for arm in ARMS:
        lines.append(
            "\t".join(
                [
                    "arm",
                    arm,
                    f"{means[arm]['seg_AUROC_0.05']:.9f}",
                    f"{means[arm]['seg_F1_raw']:.9f}",
                    f"{means[arm]['seg_AUPRO_0.05']:.9f}",
                ],
            ),
        )
    for name, effect in payload["effects"].items():
        macro = effect["macro"]
        lines.append(
            "\t".join(
                [
                    "effect",
                    name,
                    f"{macro['seg_AUROC_0.05']:.9f}",
                    f"{macro['seg_F1_raw']:.9f}",
                    f"{macro['seg_AUPRO_0.05']:.9f}",
                ],
            ),
        )
    (root / "dvt_density_factorial_summary.tsv").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = aggregate(args.factorial_root, args.baseline_root, args.support_manifest)
    write_outputs(args.factorial_root, payload)
    print(json.dumps(payload["factor_credit"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
