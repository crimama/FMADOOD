#!/usr/bin/env python3
"""Run one metrics-only FlowTTE resolution-ladder shard.

Resolution is the only experimental variable. All stages use full-frame
shorter-edge resize through ``backbone_resolution``; tiling stays disabled.
The position-mean DVT and patch-wise MLP flow are independently refit from the
same fixed 16 support images at every resolution (same structure/seed/settings,
new resolution-specific weights). Native float16 maps remain in RAM only.
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
from flow_tte_gap_decomposition import load_gt  # noqa: E402
from flow_tte_mvtec_ad2_core import (  # noqa: E402
    BackboneLike, DatasetLike, FeatureExtractionConfig, FeatureMap, RunConfig,
    apply_feature_denoiser, artifact_l2_mean, build_pipeline,
    collect_support_feature_maps, extract_feature_map_from_rgb,
    flatten_support_feature_maps, read_rgb, resolve_flow_context_source,
    resolve_memory_context_source, stream_test_images, support_extraction_config,
    tiling_config,
)
from flow_tte_phase3_scorer_suite import NativeMapRecord  # noqa: E402
from flow_tte_resolution_ladder import (  # noqa: E402
    RESOLUTIONS, evaluate_resolution_records, token_grid_shape,
)
from flow_tte_support import select_support_paths_for_backbone  # noqa: E402
from run_flow_tte_mvtec_ad2 import add_import_paths, build_runtime, parse_args  # noqa: E402

CLAIM_SCOPE = "AD2-public-shadow-diagnostic-resolution-causality"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _parse(argv: Sequence[str]) -> tuple[RunConfig, Path, int]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--anchor-root", required=True)
    parser.add_argument("--ladder-resolution", type=int, required=True, choices=RESOLUTIONS)
    own, remaining = parser.parse_known_args(list(argv))
    return parse_args(remaining), Path(own.anchor_root), own.ladder_resolution


def _validate(config: RunConfig, resolution: int) -> None:  # noqa: C901
    brightness = config.support_brightness_range
    expected_resolution = None if resolution == 672 else resolution
    checks = (
        (config.shots == 16, "shots=16"), (config.seed == 0, "seed=0"),
        (config.flow_epochs == 3, "flow epochs=3"),
        (config.coupling_layers == 2, "coupling layers=2"),
        (config.hidden_multiplier == 1, "hidden multiplier=1"),
        (config.flow_lr == 2e-4, "flow lr=2e-4"),
        (config.flow_clamp == 1.9, "flow clamp=1.9"),
        (config.tail_weight == 0.3, "tail weight=.3"),
        (config.tail_top_k_ratio == 0.05, "tail ratio=.05"),
        (config.lambda_logdet == 2e-2, "lambda logdet=.02"),
        (config.expansion_budget == 1.0, "expansion budget=1"),
        (config.distance_weight == 1.0, "distance weight=1"),
        (config.density_weight == 0.25, "density weight=.25"),
        (config.score_mode == "latent_distance", "latent distance scorer"),
        (config.calibration_sample_size == 4096, "calibration=4096"),
        (config.backbone_model == "dinov3_vith16plus", "H+/16 backbone"),
        (config.backbone_resolution == expected_resolution, "stage backbone resolution"),
        (config.feature_layers == (7, 15, 23, 31), "layers 7,15,23,31"),
        (config.tile_patch_size == 0 and config.tile_overlap == 0, "tiling disabled"),
        (config.image_resize_factor == 1.0, "resize factor=1"),
        (brightness.min_factor == .8 and brightness.max_factor == 1.2, "brightness .8,1.2"),
        (config.support_selection.startswith("fixed_json=") and config.support_selection.endswith("dinov3_noctx_support_paths.json"), "fixed support JSON"),
        (config.support_transforms == ("identity",), "identity support"),
        (config.feature_fusion == "layer_norm_mean", "layer norm mean"),
        (config.normality_mode == "fused", "fused mode"),
        (resolve_flow_context_source(config) == "none", "unconditioned flow"),
        (resolve_memory_context_source(config) == "none", "unconditioned memory"),
        (config.context_mode == "none", "context none"),
        (config.flow_transform_mode == "flow", "patch-wise MLP flow"),
        (config.dvt_denoise_mode == "position_mean" and config.dvt_denoise_alpha == 1.0, "DVT position mean alpha=1"),
        (config.score_field_calibration_mode == "none", "score field none"),
        (not config.use_squared_distance, "raw Euclidean 1-NN"),
    )
    failed = [label for ok, label in checks if not ok]
    if failed:
        raise SystemExit("resolution ladder config mismatch: " + "; ".join(failed))
    maximum_chunk = 512 if resolution <= 896 else 256
    if config.query_chunk_size > maximum_chunk:
        raise SystemExit(f"resolution {resolution} query chunk must be <= {maximum_chunk}")


def _anchor_f1(root: Path, object_name: str) -> float:
    for path in sorted(root.glob("chunks/*/metrics.json")):
        row = json.loads(path.read_text()).get(object_name)
        if isinstance(row, dict) and "seg_F1" in row:
            return float(row["seg_F1"])
    raise FileNotFoundError(f"exact anchor seg_F1 unavailable for {object_name} under {root}")


def _native_score_map(feature_map: FeatureMap, pipeline: Any) -> np.ndarray:
    # Literal retained-anchor scoring call; budget=1.0 prevents memory growth.
    result = pipeline.score_then_expand(feature_map.values[np.newaxis, ...])
    patch = np.asarray(result.patch_scores[0], dtype=np.float32)
    native = cv2.resize(patch, feature_map.image_shape[::-1], interpolation=cv2.INTER_LINEAR)
    return native.astype(np.float16)


def _write_leaderboard(root: Path) -> None:
    payloads = [json.loads(p.read_text()) for p in sorted((root / "objects").glob("*.json"))]
    columns = ("object", "resolution", "pooled_oracle_f1", "oracle_threshold", "pixel_ap", "pauroc_0.05", "component_recall", "small_component_recall", "boundary_f1_t0", "boundary_f1_t4", "boundary_f1_t8", "normal_mean_fpr", "runtime_seconds", "peak_gpu_allocated_bytes")
    lines = ["\t".join(columns)]
    for payload in payloads:
        m, d = payload["metrics"], payload["diagnostics"]
        lines.append("\t".join(map(str, (
            payload["object"], payload["resolution"], m["pooled_oracle_f1_float16"],
            m["oracle_threshold_float16"], m["pooled_pixel_ap_float16"],
            m["pooled_pauroc_0.05_float16"], m["gt_component_recall_at_oracle"],
            m["small_defect_component_recall_at_oracle"],
            m["boundary_tolerant_f1_native_px"]["0"]["f1"],
            m["boundary_tolerant_f1_native_px"]["4"]["f1"],
            m["boundary_tolerant_f1_native_px"]["8"]["f1"],
            m["normal_image_mean_fpr_at_oracle"], d["elapsed_seconds"],
            d["peak_gpu_allocated_bytes"],
        ))))
    tmp = root / ".leaderboard.tsv.tmp"
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(root / "leaderboard.tsv")


def run_object(dataset: DatasetLike, backbone: BackboneLike, name: str, config: RunConfig, resolution: int, anchor_root: Path) -> dict[str, Any]:
    started = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    info = dataset.get_object_info(name)
    backbone.set_resolution(config.backbone_resolution or info.resolution)
    train_paths = [Path(p) for p in dataset.get_train_images(name)]
    selected = select_support_paths_for_backbone(backbone, train_paths, shots=16, policy=config.support_selection, seed=config.support_selection_seed)
    support_maps = collect_support_feature_maps(backbone, selected, support_extraction_config(config, "none"))
    expected_grid = token_grid_shape(resolution)
    observed_grids = sorted({tuple(m.values.shape[:2]) for m in support_maps})
    if observed_grids != [expected_grid]:
        raise RuntimeError(f"{name}: expected token grid {expected_grid}, got {observed_grids}")
    denoiser = fit_feature_denoiser("position_mean", [m.values for m in support_maps], alpha=1.0)
    support = flatten_support_feature_maps(support_maps, require_contexts=False, feature_denoiser=denoiser)
    pipeline = build_pipeline(config)
    training = pipeline.fit(support.values)
    records: list[NativeMapRecord] = []
    test_images = dataset.get_test_images(name, split="test_public")
    extraction = FeatureExtractionConfig(feature_fusion=config.feature_fusion, context_source="none", tiling=tiling_config(config))
    for item in stream_test_images(test_images):
        fmap = apply_feature_denoiser(extract_feature_map_from_rgb(backbone, read_rgb(item.path), extraction), denoiser)
        score = _native_score_map(fmap, pipeline)
        gt = load_gt(config.data_root, name, item.anomaly_type, item.path.stem, fmap.image_shape)
        records.append(NativeMapRecord(item.anomaly_type, score, gt, item.path.stem))
    metrics = evaluate_resolution_records(records)
    observed = float(metrics["pooled_oracle_f1_float16"])
    parity = None
    if resolution == 672:
        reference = _anchor_f1(anchor_root, name)
        parity = {"reference_seg_f1": reference, "observed_seg_f1": observed, "tolerance": 0.0, "pass": observed == reference}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    payload = {
        "schema": "flowtte-resolution-ladder-object-v1", "claim_scope": CLAIM_SCOPE,
        "created_utc": datetime.now(timezone.utc).isoformat(), "object": name,
        "resolution": resolution, "metrics": metrics, "anchor_parity": parity,
        "refit_provenance": {
            "dvt": "position_mean alpha=1.0 refit from this resolution's support features",
            "flow": "patch-wise MLP structure/seed/settings fixed; weights refit at this resolution",
        },
        "diagnostics": {
            "token_grid": list(expected_grid), "full_frame": True, "tile_rule": "disabled at all stages",
            "query_chunk_size": config.query_chunk_size, "selected_support_paths": [str(p) for p in selected],
            "dvt_artifact_l2_mean": artifact_l2_mean(denoiser), "train_nll_mean": training.train_nll_mean,
            "elapsed_seconds": time.time() - started,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
            "native_map_storage_dtype": "float16", "anomaly_map_tiffs_written": 0,
        },
    }
    _write_json_atomic(config.output_root / "objects" / f"{name}.json", payload)
    _write_leaderboard(config.output_root)
    return payload


def main(argv: Sequence[str]) -> int:
    config, anchor_root, resolution = _parse(argv)
    _validate(config, resolution)
    if resolution == 672 and not anchor_root.is_dir():
        raise SystemExit(f"anchor root unavailable: {anchor_root}")
    torch.manual_seed(0)
    config.output_root.mkdir(parents=True, exist_ok=True)
    add_import_paths(config)
    dataset, backbone = build_runtime(config)
    payloads = [run_object(dataset, backbone, n, config, resolution, anchor_root) for n in config.objects]
    failed = [p["object"] for p in payloads if p["anchor_parity"] and not p["anchor_parity"]["pass"]]
    manifest = {"schema": "flowtte-resolution-ladder-chunk-v1", "objects": list(config.objects), "resolution": resolution, "config": asdict(config), "metrics_only": True, "anomaly_map_tiffs_written": 0, "parity_failures": failed}
    _write_json_atomic(config.output_root / "chunk_manifest.json", manifest)
    if failed:
        _write_json_atomic(config.output_root / "parity_failure.json", {"objects": failed})
        return 42
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
