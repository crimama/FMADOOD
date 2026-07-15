#!/usr/bin/env python3
"""Run every Phase-3 scorer from one FlowTTE latent evaluation per image.

Native-resolution float16 maps live only in RAM. The only outputs are compact
JSON metrics; this runner never creates anomaly-map TIFFs.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "src", Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from flow_tte_mvtec_ad2_core import (  # noqa: E402
    BackboneLike,
    DatasetLike,
    FeatureExtractionConfig,
    FeatureMap,
    RunConfig,
    apply_feature_denoiser,
    artifact_l2_mean,
    build_pipeline,
    collect_support_feature_maps,
    extract_feature_map_from_rgb,
    flatten_support_feature_maps,
    read_rgb,
    resolve_flow_context_source,
    resolve_memory_context_source,
    stream_test_images,
    support_extraction_config,
    tiling_config,
)
from flow_tte_support import select_support_paths_for_backbone  # noqa: E402
from run_flow_tte_mvtec_ad2 import add_import_paths, build_runtime, parse_args  # noqa: E402

from flow_tte.denoising import fit_feature_denoiser  # noqa: E402
from flow_tte.scoring import ScoreInputs  # noqa: E402
from flow_tte.trainer import FlowDensityEstimator  # noqa: E402
from flow_tte_gap_decomposition import load_gt  # noqa: E402
from flow_tte_phase3_scorer_suite import (  # noqa: E402
    VARIANT_NAMES,
    NativeMapRecord,
    ScorerSuiteState,
    evaluate_native_records,
    fit_scorer_suite,
    score_scorer_suite,
)

CLAIM_SCOPE = "AD2-public-shadow-diagnostic"
ANCHOR_F1 = {
    "can": 0.000710,
    "fabric": 0.697949,
    "fruit_jelly": 0.476761,
    "rice": 0.712533,
    "vial": 0.439581,
    "wallplugs": 0.665702,
    "sheet_metal": 0.516126,
    "walnuts": 0.735719,
}
ANCHOR_MEAN_F1 = 0.530635


def _jsonable(value: Any) -> Any:  # noqa: ANN401
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_running_leaderboard(output_root: Path) -> None:
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_root / "objects").glob("*.json"))
    ]
    columns = ("variant", "objects", "mean_f1", "mean_pauroc_0.05", "mean_normal_fpr", "mean_bad_image_oracle")
    lines = ["\t".join(columns)]
    for variant in VARIANT_NAMES:
        rows = [payload["variants"][variant] for payload in payloads]
        if not rows:
            continue
        values = (
            variant,
            str(len(rows)),
            f"{np.mean([row['pooled_oracle_f1_float16'] for row in rows]):.10g}",
            f"{np.mean([row['pooled_pauroc_0.05_float16'] for row in rows]):.10g}",
            f"{np.mean([row['normal_image_mean_fpr_at_oracle'] for row in rows]):.10g}",
            f"{np.mean([row['bad_image_oracle_mean_float16'] for row in rows]):.10g}",
        )
        lines.append("\t".join(values))
    path = output_root / "leaderboard.tsv"
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_config(config: RunConfig) -> None:
    checks = (
        (config.normality_mode == "fused", "--normality-mode fused"),
        (resolve_flow_context_source(config) == "none", "unconditioned flow latents"),
        (resolve_memory_context_source(config) == "none", "--context-mode none"),
        (config.score_field_calibration_mode == "none", "score-field calibration none"),
        (config.score_field_foreground_mode == "none", "score-field foreground mode none"),
        (config.expansion_budget == 1.0, "anchor --expansion-budget 1.0"),
    )
    for valid, requirement in checks:
        if not valid:
            message = f"Phase-3 scorer suite requires {requirement}"
            raise SystemExit(message)


def _score_maps(
    feature_map: FeatureMap,
    estimator: FlowDensityEstimator,
    suite: ScorerSuiteState,
) -> dict[str, np.ndarray]:
    """Run the only flow evaluation for an image, then all scorer variants."""
    evaluation = estimator.evaluate(feature_map.values[np.newaxis, ...])
    results = score_scorer_suite(
        ScoreInputs(
            query_z=evaluation.z,
            nll=evaluation.nll,
            nll_penalty=estimator.density_penalty(evaluation.nll),
            image_indices=evaluation.batch.image_indices,
            n_images=evaluation.batch.n_images,
            query_contexts=None,
        ),
        suite,
    )
    if set(results) != set(VARIANT_NAMES):
        raise RuntimeError("scorer suite returned an incomplete variant set")
    width, height = feature_map.image_shape[1], feature_map.image_shape[0]
    maps: dict[str, np.ndarray] = {}
    for name in VARIANT_NAMES:
        patches = evaluation.batch.restore(results[name].patch_scores)[0]
        patch_map = patches.detach().cpu().numpy().astype(np.float32, copy=False)
        native = cv2.resize(patch_map, (width, height), interpolation=cv2.INTER_LINEAR)
        maps[name] = native.astype(np.float16)
    return maps


def run_object_suite(
    dataset: DatasetLike,
    backbone: BackboneLike,
    object_name: str,
    config: RunConfig,
) -> dict[str, Any]:
    started = time.time()
    print(f"[phase3][{object_name}] start", flush=True)
    info = dataset.get_object_info(object_name)
    backbone.set_resolution(config.backbone_resolution or info.resolution)
    train_paths = [Path(path) for path in dataset.get_train_images(object_name)]
    selected = select_support_paths_for_backbone(
        backbone,
        train_paths,
        shots=config.shots,
        policy=config.support_selection,
        seed=config.support_selection_seed,
    )
    if len(selected) < config.shots:
        message = f"{object_name}: train/good has fewer than {config.shots} images"
        raise SystemExit(message)
    print(f"[phase3][{object_name}] support_selected count={len(selected)}", flush=True)
    support_maps = collect_support_feature_maps(
        backbone,
        selected,
        support_extraction_config(config, "none"),
    )
    print(f"[phase3][{object_name}] support_features views={len(support_maps)}", flush=True)
    denoiser = fit_feature_denoiser(
        config.dvt_denoise_mode,
        [feature_map.values for feature_map in support_maps],
        alpha=config.dvt_denoise_alpha,
    )
    support = flatten_support_feature_maps(
        support_maps,
        require_contexts=False,
        feature_denoiser=denoiser,
    )
    pipeline = build_pipeline(config)
    training = pipeline.fit(support.values)
    if pipeline.estimator is None or pipeline.memory is None:
        raise RuntimeError("FlowTTE fit did not initialize estimator and memory")
    suite = fit_scorer_suite(pipeline.memory.bank.features, pipeline.config.score)
    print(
        f"[phase3][{object_name}] suite_fit bank={pipeline.memory.bank.size()} "
        f"pca_rank={suite.metadata['pca']['rank']}",
        flush=True,
    )

    test_images = dataset.get_test_images(object_name, split="test_public")
    items = stream_test_images(test_images)
    records: dict[str, list[NativeMapRecord]] = {name: [] for name in VARIANT_NAMES}
    for index, item in enumerate(items):
        feature_map = apply_feature_denoiser(
            extract_feature_map_from_rgb(
                backbone,
                read_rgb(item.path),
                FeatureExtractionConfig(
                    feature_fusion=config.feature_fusion,
                    context_source="none",
                    tiling=tiling_config(config),
                ),
            ),
            denoiser,
        )
        maps = _score_maps(feature_map, pipeline.estimator, suite)
        gt = load_gt(
            config.data_root,
            object_name,
            item.anomaly_type,
            item.path.stem,
            feature_map.image_shape,
        )
        for name in VARIANT_NAMES:
            records[name].append(NativeMapRecord(item.anomaly_type, maps[name], gt, item.path.stem))
        if index == 0:
            print(
                f"[phase3][{object_name}] first_image_scored stem={item.path.stem} "
                f"variants={len(maps)}",
                flush=True,
            )
        if (index + 1) % 10 == 0 or index + 1 == len(items):
            print(f"[phase3][{object_name}] images_scored={index + 1}/{len(items)}", flush=True)

    variants = {name: evaluate_native_records(records[name]) for name in VARIANT_NAMES}
    raw = variants["raw_1nn"]
    for metrics in variants.values():
        f1_drop = float(raw["pooled_oracle_f1_float16"]) - float(
            metrics["pooled_oracle_f1_float16"],
        )
        ap_drop = float(raw["pooled_pixel_ap_float16"]) - float(
            metrics["pooled_pixel_ap_float16"],
        )
        metrics["floor_vs_raw_1nn"] = {
            "f1_drop": f1_drop,
            "ap_drop": ap_drop,
            "violation": bool(f1_drop > 0.02 or ap_drop > 0.02),
        }

    payload = {
        "schema": "flowtte-phase3-scorer-suite-object-v1",
        "claim_scope": CLAIM_SCOPE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "object": object_name,
        "variants": variants,
        "scorer_metadata": suite.metadata,
        "identity_parity": {
            "reference_source": "phase3_plan_anchor_float16_pooled_oracle",
            "reference_seg_f1": ANCHOR_F1[object_name],
            "observed_seg_f1": float(raw["pooled_oracle_f1_float16"]),
            "delta": float(raw["pooled_oracle_f1_float16"]) - ANCHOR_F1[object_name],
            "tolerance": 1e-3,
            "pass": abs(float(raw["pooled_oracle_f1_float16"]) - ANCHOR_F1[object_name]) <= 1e-3,
        },
        "diagnostics": {
            "resolution": int(info.resolution),
            "train_good_count": len(train_paths),
            "test_good_count": len(test_images.get("good", [])),
            "test_bad_count": sum(
                len(paths) for split, paths in test_images.items() if split != "good"
            ),
            "selected_support_count": len(selected),
            "selected_support_paths": [str(path) for path in selected],
            "processed_test_count": len(items),
            "initial_memory_size": pipeline.memory.bank.size(),
            "final_memory_size": pipeline.memory.bank.size(),
            "train_nll_mean": training.train_nll_mean,
            "train_nll_std": training.train_nll_std,
            "density_threshold": training.density_threshold,
            "dvt_denoise_mode": config.dvt_denoise_mode,
            "dvt_denoise_alpha": config.dvt_denoise_alpha,
            "dvt_artifact_l2_mean": artifact_l2_mean(denoiser),
            "flow_evaluations_per_test_image": 1,
            "anomaly_map_tiffs_written": 0,
            "native_map_storage_dtype": "float16",
            "elapsed_seconds": time.time() - started,
        },
    }
    _write_json_atomic(config.output_root / "objects" / f"{object_name}.json", payload)
    _write_running_leaderboard(config.output_root)
    print(
        f"[phase3][{object_name}] complete raw_f1="
        f"{raw['pooled_oracle_f1_float16']:.6f} elapsed={time.time() - started:.1f}s",
        flush=True,
    )
    return payload


def main(argv: Sequence[str]) -> int:
    config = parse_args(argv)
    _validate_config(config)
    torch.manual_seed(config.seed)
    config.output_root.mkdir(parents=True, exist_ok=True)
    add_import_paths(config)
    dataset, backbone = build_runtime(config)
    payloads = [run_object_suite(dataset, backbone, name, config) for name in config.objects]
    manifest = {
        "schema": "flowtte-phase3-scorer-suite-chunk-v1",
        "claim_scope": CLAIM_SCOPE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "objects": [payload["object"] for payload in payloads],
        "variants": list(VARIANT_NAMES),
        "config": asdict(config),
        "metrics_only": True,
        "anomaly_map_tiffs_written": 0,
        "identity_parity_reference": {"objects": ANCHOR_F1, "mean": ANCHOR_MEAN_F1},
        "calibration_deviation": {
            "shrinkage_mahalanobis": "in-sample support distances; optimistic bias",
            "global_pca_residual": "in-sample support distances; optimistic bias",
        },
        "anchor_formula_note": (
            "normality_mode=fused uses z-scored latent distance + 0.25*density_penalty; "
            "the anchor CLI residual_weight=0.25 is inert in this code path"
        ),
    }
    _write_json_atomic(config.output_root / "chunk_manifest.json", manifest)
    print(json.dumps({"objects": manifest["objects"], "status": "complete"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
