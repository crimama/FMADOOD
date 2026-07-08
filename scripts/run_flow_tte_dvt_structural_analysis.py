# /// script
# requires-python = ">=3.8"
# dependencies = []
# ///
# pyright: reportMissingImports=false
"""Structural diagnostics for the DVT-style FlowTTE position denoising probe.

Usage:
  python scripts/run_flow_tte_dvt_structural_analysis.py \
    --data-root /home/hunim/Volume/DATA/mvtec_ad_2 \
    --output-root /workspace/results_remote/flowtte_dvt_structural_analysis \
    --support-json skill_graph/experiments/.../dinov3_noctx_support_paths.json \
    --objects can,fabric
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (_REPO_ROOT, _REPO_ROOT / "src", Path(__file__).resolve().parent):
    _path_text = str(_path)
    if _path_text not in sys.path:
        sys.path.insert(0, _path_text)

from dinov3_backbone import DINOv3Backbone  # noqa: E402
from flow_tte_dvt_structural_utils import (  # noqa: E402
    FloatArray,
    high_mask,
    high_region_share,
    low_rank_energy_summary,
    normalize_minmax,
    safe_corrcoef,
    summarize_values,
    top_percent_mean,
)
from flow_tte_mvtec_ad2_core import (  # noqa: E402
    DatasetLike,
    FeatureExtractionConfig,
    extract_feature_map_from_rgb,
)
from flow_tte_register_analysis_extract import build_dataset, stream_test_images  # noqa: E402
from flow_tte_register_analysis_types import (  # noqa: E402
    JsonValue,
    latent_volume_summary,
    load_support_paths,
    write_tsv,
)
from flow_tte_support import read_rgb  # noqa: E402

from flow_tte.config import ExpansionConfig, FlowConfig, FlowTTEConfig, ScoreConfig  # noqa: E402
from flow_tte.denoising import PositionMeanArtifactDenoiser  # noqa: E402
from flow_tte.pipeline import FlowTTE  # noqa: E402
from flow_tte.scoring import ScoreInputs, score_flow_memory  # noqa: E402
from flow_tte.tensors import resolve_device, to_numpy  # noqa: E402


@dataclass(frozen=True)
class StructuralConfig:
    data_root: Path
    output_root: Path
    project_root: Path
    fsad_root: Path
    support_json: Path
    objects: Tuple[str, ...]
    device: str
    seed: int
    alpha: float
    top_percent: float
    test_images_per_split: int
    support_sample_patches: int
    query_chunk_size: int


@dataclass(frozen=True)
class SupportState:
    raw_maps: Tuple[FloatArray, ...]
    denoised_maps: Tuple[FloatArray, ...]
    raw_features: FloatArray
    denoised_features: FloatArray
    denoiser: PositionMeanArtifactDenoiser
    foreground_score: FloatArray


@dataclass(frozen=True)
class Scenario:
    name: str
    pipeline: FlowTTE
    support_features: FloatArray
    query_denoised: bool


ScoreMaps = Mapping[str, FloatArray]


def parse_args(argv: Sequence[str]) -> StructuralConfig:
    parser = argparse.ArgumentParser(description="Analyze DVT-style FlowTTE structure.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--project-root", default="/workspace")
    parser.add_argument("--fsad-root", default="/workspace/fsad_tta")
    parser.add_argument("--support-json", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--top-percent", type=float, default=0.01)
    parser.add_argument("--test-images-per-split", type=int, default=12)
    parser.add_argument("--support-sample-patches", type=int, default=4096)
    parser.add_argument("--query-chunk-size", type=int, default=256)
    args = parser.parse_args(list(argv))
    objects = tuple(part for part in args.objects.replace(",", " ").split() if part)
    if not objects:
        raise SystemExit("--objects must contain at least one object")
    if args.alpha < 0.0:
        raise SystemExit("--alpha must be non-negative")
    if args.top_percent <= 0.0 or args.top_percent > 1.0:
        raise SystemExit("--top-percent must be in (0, 1]")
    if args.test_images_per_split <= 0:
        raise SystemExit("--test-images-per-split must be positive")
    if args.support_sample_patches <= 1:
        raise SystemExit("--support-sample-patches must be greater than 1")
    if args.query_chunk_size <= 0:
        raise SystemExit("--query-chunk-size must be positive")
    return StructuralConfig(
        data_root=Path(args.data_root),
        output_root=Path(args.output_root),
        project_root=Path(args.project_root),
        fsad_root=Path(args.fsad_root),
        support_json=Path(args.support_json),
        objects=objects,
        device=args.device,
        seed=args.seed,
        alpha=float(args.alpha),
        top_percent=float(args.top_percent),
        test_images_per_split=int(args.test_images_per_split),
        support_sample_patches=int(args.support_sample_patches),
        query_chunk_size=int(args.query_chunk_size),
    )


def add_import_paths(config: StructuralConfig) -> None:
    for path in (config.project_root, config.fsad_root / "src", config.fsad_root / "scripts"):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def make_pipeline(config: StructuralConfig) -> FlowTTE:
    return FlowTTE(
        FlowTTEConfig(
            flow=FlowConfig(
                n_coupling_layers=2,
                hidden_multiplier=1,
                transform_mode="flow",
                condition_mode="none",
                n_epochs=3,
                lr=2e-4,
                clamp=1.9,
                tail_weight=0.3,
                tail_top_k_ratio=0.05,
                lambda_logdet=1e-3,
                batch_size=512,
                seed=config.seed,
            ),
            expansion=ExpansionConfig(
                budget=1.0,
                density_quantile=0.90,
                random_seed=config.seed,
            ),
            score=ScoreConfig(
                distance_weight=1.0,
                density_weight=0.25,
                score_mode="latent_distance",
                context_mode="none",
                top_percent=config.top_percent,
                query_chunk_size=512,
            ),
            device=config.device,
        ),
    )


def extract_support_state(
    backbone: DINOv3Backbone,
    support_paths: Sequence[Path],
    alpha: float,
) -> SupportState:
    raw_maps: List[FloatArray] = []
    foreground_maps: List[FloatArray] = []
    for path in support_paths:
        rgb = read_rgb(path)
        feature_map = extract_feature_map_from_rgb(
            backbone,
            rgb,
            FeatureExtractionConfig(feature_fusion="layer_norm_mean"),
        )
        raw_maps.append(feature_map.values.astype(np.float32, copy=False))
        height, width = int(feature_map.values.shape[0]), int(feature_map.values.shape[1])
        foreground_maps.append(rgb_foreground_proxy(rgb, (height, width)))
    denoiser = PositionMeanArtifactDenoiser.fit(raw_maps, alpha=alpha)
    denoised_maps = tuple(denoiser.transform(feature_map) for feature_map in raw_maps)
    raw_features = flatten_maps(raw_maps)
    denoised_features = flatten_maps(denoised_maps)
    foreground_score = np.mean(np.stack(foreground_maps, axis=0), axis=0).astype(
        np.float32,
        copy=False,
    )
    return SupportState(
        raw_maps=tuple(raw_maps),
        denoised_maps=denoised_maps,
        raw_features=raw_features,
        denoised_features=denoised_features,
        denoiser=denoiser,
        foreground_score=foreground_score,
    )


def rgb_foreground_proxy(
    image: npt.NDArray[np.uint8],
    grid_shape: Tuple[int, int],
) -> FloatArray:
    height, width = grid_shape
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    red = resized[:, :, 0].astype(np.float32)
    green = resized[:, :, 1].astype(np.float32)
    blue = resized[:, :, 2].astype(np.float32)
    gray = (0.299 * red + 0.587 * green + 0.114 * blue).astype(np.float32, copy=False)
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]], axis=0)
    background = float(np.median(border))
    return normalize_minmax(np.abs(gray - background).astype(np.float32, copy=False))


def flatten_maps(feature_maps: Sequence[FloatArray]) -> FloatArray:
    return np.concatenate(
        [feature_map.reshape(-1, feature_map.shape[-1]) for feature_map in feature_maps],
        axis=0,
    ).astype(np.float32, copy=False)


def fit_scenarios(config: StructuralConfig, support: SupportState) -> Tuple[Scenario, ...]:
    raw_pipeline = make_pipeline(config)
    _ = raw_pipeline.fit(support.raw_features)
    denoised_pipeline = make_pipeline(config)
    _ = denoised_pipeline.fit(support.denoised_features)
    return (
        Scenario(
            name="raw_support_raw_query",
            pipeline=raw_pipeline,
            support_features=support.raw_features,
            query_denoised=False,
        ),
        Scenario(
            name="denoised_support_raw_query",
            pipeline=denoised_pipeline,
            support_features=support.denoised_features,
            query_denoised=False,
        ),
        Scenario(
            name="raw_support_denoised_query",
            pipeline=raw_pipeline,
            support_features=support.raw_features,
            query_denoised=True,
        ),
        Scenario(
            name="denoised_support_denoised_query",
            pipeline=denoised_pipeline,
            support_features=support.denoised_features,
            query_denoised=True,
        ),
    )


def artifact_overlap_row(object_name: str, support: SupportState) -> Dict[str, str]:
    artifact = support.denoiser.artifact
    artifact_norm = np.linalg.norm(artifact, axis=-1).astype(np.float32, copy=False)
    foreground = support.foreground_score
    high_artifact = high_mask(artifact_norm, 80.0)
    foreground_high = high_mask(foreground, 80.0)
    fg_high_mean = float(np.mean(foreground[high_artifact]))
    fg_low_mean = float(np.mean(foreground[~high_artifact]))
    rank_summary = low_rank_energy_summary(artifact, ranks=(1, 3, 5, 10))
    return {
        "object": object_name,
        "artifact_l2_mean": fmt_float(float(np.mean(artifact_norm))),
        "artifact_l2_p95": fmt_float(float(np.percentile(artifact_norm, 95))),
        "artifact_foreground_corr": fmt_float(safe_corrcoef(artifact_norm, foreground)),
        "artifact_top20_foreground_top20_share": fmt_float(
            high_region_share(artifact_norm, foreground, 80.0),
        ),
        "artifact_top20_foreground_mean": fmt_float(fg_high_mean),
        "artifact_low80_foreground_mean": fmt_float(fg_low_mean),
        "foreground_top20_artifact_top20_share": fmt_float(
            float(np.mean(high_artifact[foreground_high])),
        ),
        "artifact_effective_rank": fmt_float(rank_summary["effective_rank"]),
        "artifact_top1_energy_share": fmt_float(rank_summary["top1_energy_share"]),
        "artifact_top3_energy_share": fmt_float(rank_summary["top3_energy_share"]),
        "artifact_top5_energy_share": fmt_float(rank_summary["top5_energy_share"]),
        "artifact_top10_energy_share": fmt_float(rank_summary["top10_energy_share"]),
    }


def support_compactness_rows(
    object_name: str,
    config: StructuralConfig,
    support: SupportState,
    scenarios: Sequence[Scenario],
) -> List[Dict[str, str]]:
    raw_latent = require_pipeline_latent(scenarios[0].pipeline, support.raw_features)
    denoised_latent = require_pipeline_latent(scenarios[-1].pipeline, support.denoised_features)
    entries = (
        ("raw_feature", support.raw_features),
        ("denoised_feature", support.denoised_features),
        ("raw_latent", raw_latent),
        ("denoised_latent", denoised_latent),
    )
    rows: List[Dict[str, str]] = []
    for name, features in entries:
        sample = sample_rows(features, config.support_sample_patches, config.seed + len(name))
        volume = latent_volume_summary(sample)
        loo = leave_one_out_nn(sample, config.device)
        rows.append(
            {
                "object": object_name,
                "space": name,
                "sample_count": str(sample.shape[0]),
                "mean_variance": fmt_float(volume.mean_variance),
                "mean_log_variance": fmt_float(volume.mean_log_variance),
                "effective_rank": fmt_float(volume.effective_rank),
                "loo_nn_mean": fmt_float(float(np.mean(loo))),
                "loo_nn_std": fmt_float(float(np.std(loo))),
                "loo_nn_p95": fmt_float(float(np.percentile(loo, 95))),
            },
        )
    return rows


def require_pipeline_latent(pipeline: FlowTTE, features: FloatArray) -> FloatArray:
    estimator = pipeline.estimator
    if estimator is None:
        raise RuntimeError("Pipeline estimator is missing")
    return to_numpy(estimator.transform(features))


def sample_rows(features: FloatArray, limit: int, seed: int) -> FloatArray:
    if features.shape[0] <= limit:
        return features.astype(np.float32, copy=False)
    rng = np.random.default_rng(seed)
    indices = rng.choice(features.shape[0], size=limit, replace=False)
    return features[indices].astype(np.float32, copy=False)


def leave_one_out_nn(features: FloatArray, device_name: str) -> FloatArray:
    device = resolve_device(device_name)
    values = torch.as_tensor(features, dtype=torch.float32, device=device)
    distances = torch.cdist(values, values, p=2.0)
    distances.fill_diagonal_(float("inf"))
    return to_numpy(torch.min(distances, dim=1).values)


def select_test_items(
    dataset: DatasetLike,
    object_name: str,
    per_split: int,
) -> Tuple[Tuple[str, Path], ...]:
    good: List[Tuple[str, Path]] = []
    bad: List[Tuple[str, Path]] = []
    for item in stream_test_images(dataset, object_name):
        target = good if item.split == "good" else bad
        if len(target) < per_split:
            target.append((item.split, item.path))
    return tuple(good + bad)


def score_components(pipeline: FlowTTE, feature_map: FloatArray) -> Dict[str, FloatArray]:
    estimator = pipeline.estimator
    memory = pipeline.memory
    calibration = pipeline.score_calibration
    if estimator is None or memory is None or calibration is None:
        raise RuntimeError("Pipeline must be fitted before scoring")
    evaluation = estimator.evaluate(feature_map[np.newaxis, ...])
    density_penalty = estimator.density_penalty(evaluation.nll)
    result = score_flow_memory(
        inputs=ScoreInputs(
            query_z=evaluation.z,
            nll=evaluation.nll,
            nll_penalty=density_penalty,
            image_indices=evaluation.batch.image_indices,
            n_images=evaluation.batch.n_images,
        ),
        bank=memory.bank,
        config=pipeline.config.score,
        calibration=calibration,
    )
    return {
        "latent_distance": to_numpy(evaluation.batch.restore(result.distances))[0],
        "latent_distance_norm": to_numpy(evaluation.batch.restore(result.distance_scores))[0],
        "density_penalty": to_numpy(evaluation.batch.restore(result.density_penalty))[0],
        "nll": to_numpy(evaluation.batch.restore(evaluation.nll))[0],
        "final_score": to_numpy(evaluation.batch.restore(result.patch_scores))[0],
    }


def feature_nn_map(
    query_map: FloatArray,
    support_features: FloatArray,
    device_name: str,
    query_chunk_size: int,
) -> FloatArray:
    height, width, feature_dim = query_map.shape
    query = query_map.reshape(-1, feature_dim).astype(np.float32, copy=False)
    device = resolve_device(device_name)
    support = torch.as_tensor(support_features, dtype=torch.float32, device=device)
    outputs: List[torch.Tensor] = []
    for start in range(0, query.shape[0], query_chunk_size):
        stop = min(start + query_chunk_size, query.shape[0])
        chunk = torch.as_tensor(query[start:stop], dtype=torch.float32, device=device)
        distances = torch.cdist(chunk, support, p=2.0)
        outputs.append(torch.min(distances, dim=1).values.detach().cpu())
    values = torch.cat(outputs, dim=0).numpy().astype(np.float32, copy=False)
    return values.reshape(height, width)


def accumulate_image_scores(
    buckets: Dict[Tuple[str, str, str], List[float]],
    split: str,
    scenario: Scenario,
    maps: ScoreMaps,
    top_percent: float,
) -> None:
    for component, score_map in maps.items():
        key = (scenario.name, split, component)
        buckets.setdefault(key, []).append(top_percent_mean(score_map, top_percent))


def summarize_score_buckets(
    object_name: str,
    buckets: Mapping[Tuple[str, str, str], Sequence[float]],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for key in sorted(buckets):
        scenario_name, split, component = key
        count, mean, std, p50, p95 = summarize_values(buckets[key])
        rows.append(
            {
                "object": object_name,
                "scenario": scenario_name,
                "split": split,
                "component": component,
                "image_count": str(count),
                "top_image_mean": fmt_float(mean),
                "top_image_std": fmt_float(std),
                "top_image_p50": fmt_float(p50),
                "top_image_p95": fmt_float(p95),
            },
        )
    return rows


def side_ablation_rows(
    object_name: str,
    score_rows: Sequence[Dict[str, str]],
) -> List[Dict[str, str]]:
    indexed: Dict[Tuple[str, str, str], float] = {}
    for row in score_rows:
        indexed[(row["scenario"], row["split"], row["component"])] = float(row["top_image_mean"])
    rows: List[Dict[str, str]] = []
    scenarios = sorted({row["scenario"] for row in score_rows})
    components = ("feature_nn", "latent_distance_norm", "density_penalty", "final_score")
    for scenario_name in scenarios:
        for component in components:
            good = indexed.get((scenario_name, "good", component), float("nan"))
            bad = indexed.get((scenario_name, "bad", component), float("nan"))
            rows.append(
                {
                    "object": object_name,
                    "scenario": scenario_name,
                    "component": component,
                    "good_top_mean": fmt_float(good),
                    "bad_top_mean": fmt_float(bad),
                    "delta_bad_good": fmt_float(bad - good),
                },
            )
    return rows


def analyze_object(
    config: StructuralConfig,
    dataset: DatasetLike,
    backbone: DINOv3Backbone,
    support_paths: Sequence[Path],
    object_name: str,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    info = dataset.get_object_info(object_name)
    backbone.set_resolution(info.resolution)
    support = extract_support_state(backbone, support_paths, alpha=config.alpha)
    scenarios = fit_scenarios(config, support)
    compactness_rows = support_compactness_rows(object_name, config, support, scenarios)
    artifact_rows = [artifact_overlap_row(object_name, support)]

    buckets: Dict[Tuple[str, str, str], List[float]] = {}
    for split, path in select_test_items(dataset, object_name, config.test_images_per_split):
        raw_map = extract_feature_map_from_rgb(
            backbone,
            read_rgb(path),
            FeatureExtractionConfig(feature_fusion="layer_norm_mean"),
        ).values.astype(np.float32, copy=False)
        denoised_map = support.denoiser.transform(raw_map)
        for scenario in scenarios:
            query_map = denoised_map if scenario.query_denoised else raw_map
            component_maps = dict(score_components(scenario.pipeline, query_map))
            component_maps["feature_nn"] = feature_nn_map(
                query_map,
                scenario.support_features,
                config.device,
                config.query_chunk_size,
            )
            accumulate_image_scores(
                buckets,
                split=split,
                scenario=scenario,
                maps=component_maps,
                top_percent=config.top_percent,
            )
    score_rows = summarize_score_buckets(object_name, buckets)
    return artifact_rows, compactness_rows, score_rows, side_ablation_rows(object_name, score_rows)


def write_manifest(config: StructuralConfig) -> None:
    payload: Dict[str, JsonValue] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "FlowTTE DVT structural analysis",
        "target_dataset": "MVTec AD2 single-image",
        "data_root": str(config.data_root),
        "support_json": str(config.support_json),
        "objects": list(config.objects),
        "backbone": "dinov3_vitl16 layer_norm_mean [5,11,17,23]",
        "flow": "2 coupling layers, hidden multiplier 1, 3 epochs, lr 2e-4, clamp 1.9",
        "score": "latent distance + 0.25 density penalty, no TTE expansion",
        "dvt_probe": "support-fitted position_mean artifact field",
        "alpha": config.alpha,
        "top_percent": config.top_percent,
        "test_images_per_split": config.test_images_per_split,
        "support_sample_patches": config.support_sample_patches,
        "config": asdict(config),
    }
    (config.output_root / "run_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def fmt_float(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return f"{value:.8f}"


def main(argv: Sequence[str]) -> int:
    config = parse_args(argv)
    torch.manual_seed(config.seed)
    config.output_root.mkdir(parents=True, exist_ok=True)
    add_import_paths(config)
    dataset = build_dataset(config.data_root, config.objects)
    support_by_object = load_support_paths(config.support_json)
    backbone = DINOv3Backbone("dinov3_vitl16", device=config.device, smaller_edge_size=672)

    artifact_rows: List[Dict[str, str]] = []
    compactness_rows: List[Dict[str, str]] = []
    score_rows: List[Dict[str, str]] = []
    side_rows: List[Dict[str, str]] = []
    for object_name in config.objects:
        object_artifact, object_compactness, object_scores, object_side = analyze_object(
            config,
            dataset,
            backbone,
            support_by_object[object_name],
            object_name,
        )
        artifact_rows.extend(object_artifact)
        compactness_rows.extend(object_compactness)
        score_rows.extend(object_scores)
        side_rows.extend(object_side)

    write_tsv(config.output_root / "artifact_overlap.tsv", artifact_rows)
    write_tsv(config.output_root / "support_compactness.tsv", compactness_rows)
    write_tsv(config.output_root / "score_decomposition_summary.tsv", score_rows)
    write_tsv(config.output_root / "side_ablation_summary.tsv", side_rows)
    write_manifest(config)
    print(json.dumps({"output_root": str(config.output_root), "objects": list(config.objects)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
