#!/usr/bin/env python3
"""Run the preregistered two-view patch-grid shift smoke on MVTec AD2.

The runner fits an anchor-compatible unshifted support pipeline and a second
pipeline whose support tensors are shifted by eight resized pixels. Query view
1 features are extracted once, then scored with the two support/DVT states.
Only compact JSON metrics are retained; native anomaly maps remain in RAM.
"""
from __future__ import annotations

import argparse
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

from flow_tte.denoising import fit_feature_denoiser  # noqa: E402
from flow_tte_gridshift_2view import (  # noqa: E402
    align_shifted_native_map,
    combine_aligned_maps,
    shift_resized_tensor_right_down,
)
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
from flow_tte_phase2_refinement import evaluate_variant  # noqa: E402
from flow_tte_gap_decomposition import load_gt  # noqa: E402
from flow_tte_superadd_preprocess import apply_brightness  # noqa: E402
from flow_tte_support import (  # noqa: E402
    merge_layer_features,
    select_support_paths_for_backbone,
)
from run_flow_tte_mvtec_ad2 import add_import_paths, build_runtime, parse_args  # noqa: E402

CLAIM_SCOPE = "AD2-public-shadow-diagnostic"
VARIANT_NAMES = (
    "view0_only",
    "view1_arm_A",
    "arm_A_mean",
    "arm_A_max",
    "arm_C_mean",
    "arm_C_max",
)
SHIFT_YX = (8, 8)


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


def _parse_runner_args(argv: Sequence[str]) -> tuple[RunConfig, Path, tuple[int, int]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--anchor-root", required=True)
    parser.add_argument("--grid-shift-pixels", type=int, default=8)
    runner, remaining = parser.parse_known_args(list(argv))
    if runner.grid_shift_pixels != 8:
        raise SystemExit("grid-shift smoke requires --grid-shift-pixels 8")
    return parse_args(remaining), Path(runner.anchor_root), SHIFT_YX


def _validate_config(config: RunConfig) -> None:  # noqa: PLR0915
    """Reject any invocation that is not the frozen Phase-3 anchor config."""
    brightness = config.support_brightness_range
    checks = (
        (config.shots == 16, "--shots 16"),
        (config.seed == 0, "--seed 0"),
        (config.flow_epochs == 3, "--flow-epochs 3"),
        (config.coupling_layers == 2, "--coupling-layers 2"),
        (config.hidden_multiplier == 1, "--hidden-multiplier 1"),
        (config.flow_lr == 2e-4, "--flow-lr 2e-4"),
        (config.flow_clamp == 1.9, "--flow-clamp 1.9"),
        (config.tail_weight == 0.3, "--tail-weight 0.3"),
        (config.tail_top_k_ratio == 0.05, "--tail-top-k-ratio 0.05"),
        (config.lambda_logdet == 2e-2, "--lambda-logdet 2e-2"),
        (config.density_quantile == 0.90, "--density-quantile 0.90"),
        (config.expansion_budget == 1.0, "--expansion-budget 1.0"),
        (config.distance_weight == 1.0, "--distance-weight 1.0"),
        (config.density_weight == 0.25, "--density-weight 0.25"),
        (config.score_mode == "latent_distance", "--score-mode latent_distance"),
        (config.residual_weight == 0.25, "--residual-weight 0.25"),
        (config.top_percent == 0.01, "--top-percent 0.01"),
        (config.query_chunk_size == 512, "--query-chunk-size 512"),
        (config.calibration_sample_size == 4096, "--calibration-sample-size 4096"),
        (config.pro_integration_limit == 0.05, "--pro-integration-limit 0.05"),
        (config.backbone_model == "dinov3_vith16plus", "DINOv3-H/16+ backbone"),
        (config.backbone_resolution is None, "--backbone-resolution 0"),
        (config.feature_layers == (7, 15, 23, 31), "--feature-layers 7,15,23,31"),
        (config.tile_patch_size == 0, "--tile-patch-size 0"),
        (config.tile_overlap == 0, "--tile-overlap 0"),
        (config.image_resize_factor == 1.0, "--image-resize-factor 1.0"),
        (brightness.min_factor == 0.80, "support brightness minimum 0.80"),
        (brightness.max_factor == 1.20, "support brightness maximum 1.20"),
        (config.support_selection.startswith("fixed_json="), "fixed support JSON"),
        (config.support_selection.endswith("dinov3_noctx_support_paths.json"), "anchor support JSON"),
        (config.support_selection_seed == 0, "support-selection seed 0"),
        (config.support_transforms == ("identity",), "identity support transform"),
        (config.feature_fusion == "layer_norm_mean", "layer_norm_mean feature fusion"),
        (config.normality_mode == "fused", "fused normality mode"),
        (config.context_source == "none", "--context-source none"),
        (config.flow_context_source == "auto", "--flow-context-source auto"),
        (config.memory_context_source == "auto", "--memory-context-source auto"),
        (resolve_flow_context_source(config) == "none", "unconditioned flow"),
        (resolve_memory_context_source(config) == "none", "unconditioned memory"),
        (config.context_mode == "none", "--context-mode none"),
        (config.context_weight == 0.0, "--context-weight 0.0"),
        (config.context_top_m == 1, "--context-top-m 1"),
        (config.flow_condition_mode == "none", "--flow-condition-mode none"),
        (config.transformer_context_mode == "none", "transformer context none"),
        (config.flow_transform_mode == "flow", "--flow-transform-mode flow"),
        (config.dvt_denoise_mode == "position_mean", "DVT position_mean"),
        (config.dvt_denoise_alpha == 1.0, "DVT alpha 1.0"),
        (config.score_field_calibration_mode == "none", "score-field calibration none"),
        (config.score_field_calibration_alpha == 1.0, "score-field alpha 1.0"),
        (config.score_field_position_std_floor == 0.25, "score-field std floor 0.25"),
        (config.score_field_foreground_mode == "none", "score-field foreground none"),
        (config.score_field_foreground_quantile == 0.20, "foreground quantile 0.20"),
        (config.score_field_background_multiplier == 0.50, "background multiplier 0.50"),
        (config.score_field_foreground_smooth_kernel == 5, "foreground smooth kernel 5"),
        (config.score_field_support_score_quantile == 0.90, "support-score quantile 0.90"),
        (not config.cleanup_maps, "metrics-only map lifecycle"),
        (not config.use_squared_distance, "unsquared memory distance"),
    )
    failed = [requirement for valid, requirement in checks if not valid]
    if failed:
        raise SystemExit("grid-shift runner config mismatch: " + "; ".join(failed))


def _feature_map_from_prepared(
    backbone: BackboneLike,
    image_tensor: torch.Tensor,
    grid_size: tuple[int, int],
    image_shape: tuple[int, int],
    feature_fusion: str,
) -> FeatureMap:
    layers = backbone.extract_features(image_tensor)
    if not layers:
        raise RuntimeError("backbone returned no shifted-view features")
    merged = merge_layer_features(layers, feature_fusion)
    height, width = grid_size
    return FeatureMap(
        values=merged.reshape(height, width, merged.shape[-1]),
        image_shape=image_shape,
    )


def _extract_shifted_feature_map(
    backbone: BackboneLike,
    image: np.ndarray,
    feature_fusion: str,
    offset_yx: tuple[int, int],
) -> tuple[FeatureMap, tuple[int, int]]:
    image_tensor, grid_size = backbone.prepare_image(image)
    shifted = shift_resized_tensor_right_down(image_tensor, offset_yx)
    feature_map = _feature_map_from_prepared(
        backbone,
        shifted,
        grid_size,
        (int(image.shape[0]), int(image.shape[1])),
        feature_fusion,
    )
    return feature_map, (int(image_tensor.shape[-2]), int(image_tensor.shape[-1]))


def _collect_shifted_support_maps(
    backbone: BackboneLike,
    paths: Sequence[Path],
    config: RunConfig,
    offset_yx: tuple[int, int],
) -> list[FeatureMap]:
    """Mirror anchor support preprocessing, shifting only after prepare_image."""
    extraction = support_extraction_config(config, "none")
    if extraction.transform_names != ("identity",):
        raise RuntimeError("grid-shift runner only supports identity support transforms")
    feature_maps = []
    for index, path in enumerate(paths):
        image = read_rgb(path)
        factor = extraction.brightness_range.factor_for(index, extraction.brightness_seed)
        image = apply_brightness(image, factor)
        feature_map, _ = _extract_shifted_feature_map(
            backbone,
            image,
            config.feature_fusion,
            offset_yx,
        )
        feature_maps.append(feature_map)
    return feature_maps


def _native_score_map(feature_map: FeatureMap, pipeline: Any) -> np.ndarray:  # noqa: ANN401
    # Use the literal retained-anchor call path. The frozen expansion budget is
    # 1.0, so its reservoir has zero extra capacity and cannot mutate.
    result = pipeline.score_then_expand(feature_map.values[np.newaxis, ...])
    patch_scores = np.asarray(result.patch_scores[0], dtype=np.float32)
    native = cv2.resize(
        patch_scores,
        (feature_map.image_shape[1], feature_map.image_shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.asarray(native, dtype=np.float32)


def _capture_rng_state() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    cpu = torch.random.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return cpu, cuda


def _restore_rng_state(state: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    cpu, cuda = state
    torch.random.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)


def _anchor_f1(anchor_root: Path, object_name: str) -> float:
    summary_path = anchor_root / "summary_gapdecomp_anchor.json"
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in payload.get("objects", []):
            if row.get("object") == object_name:
                return float(row["seg_F1"])
    for path in sorted(anchor_root.glob("chunks/*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get(object_name)
        if isinstance(values, dict) and "seg_F1" in values:
            return float(values["seg_F1"])
    raise FileNotFoundError(f"anchor F1 for {object_name} not found under {anchor_root}")


def _write_running_leaderboard(output_root: Path) -> None:
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_root / "objects").glob("*.json"))
    ]
    lines = ["variant\tobjects\tmean_f1\tmean_pauroc_0.05\tmean_component_recall\tmean_normal_fpr"]
    for variant in VARIANT_NAMES:
        rows = [payload["variants"][variant] for payload in payloads]
        if not rows:
            continue
        lines.append(
            "\t".join(
                (
                    variant,
                    str(len(rows)),
                    f"{np.mean([row['pooled_oracle_f1'] for row in rows]):.10g}",
                    f"{np.mean([row['seg_AUROC_0.05'] for row in rows]):.10g}",
                    f"{np.mean([row['gt_component_recall'] for row in rows]):.10g}",
                    f"{np.mean([row['normal_image_fpr']['mean_per_image'] for row in rows]):.10g}",
                )
            )
        )
    path = output_root / "leaderboard.tsv"
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_object_smoke(
    dataset: DatasetLike,
    backbone: BackboneLike,
    object_name: str,
    config: RunConfig,
    anchor_root: Path,
    offset_yx: tuple[int, int],
) -> dict[str, Any]:
    started = time.time()
    print(f"[gridshift][{object_name}] start", flush=True)
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
        raise SystemExit(f"{object_name}: train/good has fewer than {config.shots} images")

    original_support_maps = collect_support_feature_maps(
        backbone,
        selected,
        support_extraction_config(config, "none"),
    )
    shifted_support_maps = _collect_shifted_support_maps(backbone, selected, config, offset_yx)
    original_dvt = fit_feature_denoiser(
        config.dvt_denoise_mode,
        [feature_map.values for feature_map in original_support_maps],
        alpha=config.dvt_denoise_alpha,
    )
    shifted_dvt = fit_feature_denoiser(
        config.dvt_denoise_mode,
        [feature_map.values for feature_map in shifted_support_maps],
        alpha=config.dvt_denoise_alpha,
    )
    original_support = flatten_support_feature_maps(
        original_support_maps,
        require_contexts=False,
        feature_denoiser=original_dvt,
    )
    shifted_support = flatten_support_feature_maps(
        shifted_support_maps,
        require_contexts=False,
        feature_denoiser=shifted_dvt,
    )
    # Fit arm C from the exact same pre-fit random state as arm A, then restore
    # arm A's post-fit state. This both matches the requested seed/settings for
    # the independent refit and preserves anchor RNG progression for the next
    # object in a two-object shard.
    pre_fit_rng = _capture_rng_state()
    pipeline_a = build_pipeline(config)
    training_a = pipeline_a.fit(original_support.values)
    post_a_fit_rng = _capture_rng_state()
    _restore_rng_state(pre_fit_rng)
    try:
        pipeline_c = build_pipeline(config)
        training_c = pipeline_c.fit(shifted_support.values)
    finally:
        _restore_rng_state(post_a_fit_rng)
    print(
        f"[gridshift][{object_name}] support_fit views={len(original_support_maps)} "
        f"bank_A={pipeline_a.memory.bank.size() if pipeline_a.memory else 0} "
        f"bank_C={pipeline_c.memory.bank.size() if pipeline_c.memory else 0}",
        flush=True,
    )

    test_images = dataset.get_test_images(object_name, split="test_public")
    items = stream_test_images(test_images)
    records: list[dict[str, Any]] = []
    scores: dict[str, list[np.ndarray]] = {name: [] for name in VARIANT_NAMES}
    good_pixel_deltas: list[np.ndarray] = []
    extraction_config = FeatureExtractionConfig(
        feature_fusion=config.feature_fusion,
        context_source="none",
        tiling=tiling_config(config),
    )
    for index, item in enumerate(items):
        image = read_rgb(item.path)
        view0 = apply_feature_denoiser(
            extract_feature_map_from_rgb(backbone, image, extraction_config),
            original_dvt,
        )
        shifted_raw, resized_shape = _extract_shifted_feature_map(
            backbone,
            image,
            config.feature_fusion,
            offset_yx,
        )
        # The expensive shifted query extraction above is shared by arms A/C;
        # only the support-fitted DVT transform and anchor-path scorer differ.
        shifted_a = apply_feature_denoiser(shifted_raw, original_dvt)
        shifted_c = apply_feature_denoiser(shifted_raw, shifted_dvt)
        map0 = _native_score_map(view0, pipeline_a).astype(np.float16)
        map1_a = align_shifted_native_map(
            _native_score_map(shifted_a, pipeline_a),
            resized_shape,
            offset_yx,
        ).astype(np.float16)
        map1_c = align_shifted_native_map(
            _native_score_map(shifted_c, pipeline_c),
            resized_shape,
            offset_yx,
        ).astype(np.float16)
        image_maps = {
            "view0_only": map0,
            "view1_arm_A": map1_a,
            "arm_A_mean": combine_aligned_maps([map0, map1_a], "mean").astype(np.float16),
            "arm_A_max": combine_aligned_maps([map0, map1_a], "max").astype(np.float16),
            "arm_C_mean": combine_aligned_maps([map0, map1_c], "mean").astype(np.float16),
            "arm_C_max": combine_aligned_maps([map0, map1_c], "max").astype(np.float16),
        }
        gt = load_gt(
            config.data_root,
            object_name,
            item.anomaly_type,
            item.path.stem,
            view0.image_shape,
        )
        records.append(
            {
                "split": item.anomaly_type,
                "gt": gt,
                "stem": item.path.stem,
                "rgb_path": str(item.path),
            }
        )
        for name, score_map in image_maps.items():
            scores[name].append(score_map)
        if item.anomaly_type == "good":
            good_pixel_deltas.append(
                map1_a.astype(np.float32).ravel() - map0.astype(np.float32).ravel()
            )
        if index == 0:
            print(
                f"[gridshift][{object_name}] first_image_scored stem={item.path.stem} "
                f"variants={len(image_maps)}",
                flush=True,
            )
        if (index + 1) % 10 == 0 or index + 1 == len(items):
            print(f"[gridshift][{object_name}] images_scored={index + 1}/{len(items)}", flush=True)

    variants = {name: evaluate_variant(records, scores[name]) for name in VARIANT_NAMES}
    observed = float(variants["view0_only"]["pooled_oracle_f1"])
    reference = _anchor_f1(anchor_root, object_name)
    parity = {
        "reference_seg_f1": reference,
        "observed_seg_f1": observed,
        "delta": observed - reference,
        "tolerance": 0.0,
        "pass": observed == reference,
    }
    drift = float(np.median(np.concatenate(good_pixel_deltas))) if good_pixel_deltas else float("nan")
    payload = {
        "schema": "flowtte-gridshift-2view-object-v1",
        "claim_scope": CLAIM_SCOPE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "object": object_name,
        "variants": variants,
        "identity_parity": parity,
        "phase_drift_good_pixel_median_delta": drift,
        "diagnostics": {
            "resolution": int(info.resolution),
            "train_good_count": len(train_paths),
            "test_good_count": len(test_images.get("good", [])),
            "test_bad_count": sum(len(paths) for split, paths in test_images.items() if split != "good"),
            "selected_support_count": len(selected),
            "selected_support_paths": [str(path) for path in selected],
            "processed_test_count": len(items),
            "shift_resized_yx": list(offset_yx),
            "query_shifted_feature_extractions_per_image": 1,
            "support_pipeline_A": {
                "dvt_artifact_l2_mean": artifact_l2_mean(original_dvt),
                "train_nll_mean": training_a.train_nll_mean,
                "train_nll_std": training_a.train_nll_std,
                "density_threshold": training_a.density_threshold,
            },
            "support_pipeline_C": {
                "dvt_artifact_l2_mean": artifact_l2_mean(shifted_dvt),
                "train_nll_mean": training_c.train_nll_mean,
                "train_nll_std": training_c.train_nll_std,
                "density_threshold": training_c.density_threshold,
            },
            "native_map_storage_dtype": "float16",
            "anomaly_map_tiffs_written": 0,
            "elapsed_seconds": time.time() - started,
        },
    }
    _write_json_atomic(config.output_root / "objects" / f"{object_name}.json", payload)
    _write_running_leaderboard(config.output_root)
    if not parity["pass"]:
        _write_json_atomic(config.output_root / "parity_failure.json", payload["identity_parity"])
        raise SystemExit(
            f"view0 parity failed for {object_name}: observed={observed:.17g} "
            f"reference={reference:.17g} delta={observed - reference:.17g}"
        )
    print(
        f"[gridshift][{object_name}] complete view0_f1={observed:.6f} "
        f"drift={drift:.6g} elapsed={time.time() - started:.1f}s",
        flush=True,
    )
    return payload


def main(argv: Sequence[str]) -> int:
    config, anchor_root, offset_yx = _parse_runner_args(argv)
    _validate_config(config)
    if not anchor_root.is_dir():
        raise SystemExit(f"anchor root does not exist: {anchor_root}")
    torch.manual_seed(config.seed)
    config.output_root.mkdir(parents=True, exist_ok=True)
    add_import_paths(config)
    dataset, backbone = build_runtime(config)
    payloads = [
        run_object_smoke(dataset, backbone, name, config, anchor_root, offset_yx)
        for name in config.objects
    ]
    manifest = {
        "schema": "flowtte-gridshift-2view-chunk-v1",
        "claim_scope": CLAIM_SCOPE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "objects": [payload["object"] for payload in payloads],
        "variants": list(VARIANT_NAMES),
        "shift_resized_yx": list(offset_yx),
        "anchor_root": str(anchor_root),
        "config": asdict(config),
        "metrics_only": True,
        "anomaly_map_tiffs_written": 0,
    }
    _write_json_atomic(config.output_root / "chunk_manifest.json", manifest)
    print(json.dumps({"objects": manifest["objects"], "status": "complete"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
