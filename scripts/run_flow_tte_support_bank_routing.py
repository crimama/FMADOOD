#!/usr/bin/env python3
"""Run the frozen support-only acquisition-bank routing diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from flow_tte_mvtec_ad2_core import (  # noqa: E402
    BackboneLike,
    FeatureExtractionConfig,
    FeatureMap,
    RunConfig,
    apply_feature_denoiser,
    build_pipeline,
    build_score_config,
    extract_layer_feature_maps_from_rgb,
    flatten_support_feature_maps,
    merge_layer_features,
    read_rgb,
    save_prediction,
    select_support_paths_for_backbone,
    stream_test_images,
)
from run_flow_tte_mvtec_ad2 import build_runtime  # noqa: E402
from run_flow_tte_mvtec_ad2 import parse_args as parse_default_args  # noqa: E402

from flow_tte.cfntr import (  # noqa: E402
    RGBTransformSpec,
    apply_rgb_transform,
    make_axis_transform,
)
from flow_tte.config import ScoreConfig  # noqa: E402
from flow_tte.denoising import fit_feature_denoiser  # noqa: E402
from flow_tte.memory import TorchMemoryBank  # noqa: E402
from flow_tte.pipeline import FlowTTE  # noqa: E402

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
ARMS = ("identity", "union", "patchwise_min", "routed")
BANK_SPECS = (
    ("identity", None),
    ("exposure_neg", make_axis_transform("exposure", sign=-1)),
    ("exposure_pos", make_axis_transform("exposure", sign=1)),
    ("gamma_neg", make_axis_transform("gamma", sign=-1)),
    ("gamma_pos", make_axis_transform("gamma", sign=1)),
    ("white_balance_neg", make_axis_transform("white_balance", sign=-1)),
    ("white_balance_pos", make_axis_transform("white_balance", sign=1)),
)


@dataclass(frozen=True)
class RouteDecision:
    nearest_bank: str
    selected_bank: str
    nearest_distance: float
    second_distance: float
    margin: float
    confidence_threshold: float
    identity_fallback: bool


def normalize_vector(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("prototype vector must have positive finite norm")
    return (vector / norm).astype(np.float32, copy=False)


def prototype(values: Sequence[np.ndarray]) -> np.ndarray:
    if not values:
        raise ValueError("prototype requires at least one vector")
    return normalize_vector(np.mean(np.stack(values, axis=0), axis=0))


def cosine_distances(vector: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    normalized = normalize_vector(vector)
    matrix = np.asarray(prototypes, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != normalized.size:
        raise ValueError("prototype matrix dimension mismatch")
    return (1.0 - matrix @ normalized).astype(np.float32, copy=False)


def leave_one_view_margin_threshold(
    views: Mapping[str, Sequence[np.ndarray]],
    bank_names: Sequence[str],
) -> tuple[float, dict[str, Any]]:
    if any(len(views[name]) < 2 for name in bank_names):
        raise ValueError("each bank requires at least two support views")
    base = np.stack([prototype(views[name]) for name in bank_names], axis=0)
    correct_margins: list[float] = []
    total = 0
    correct = 0
    for bank_index, bank_name in enumerate(bank_names):
        for held_out in range(len(views[bank_name])):
            candidate = base.copy()
            candidate[bank_index] = prototype(
                [
                    value
                    for index, value in enumerate(views[bank_name])
                    if index != held_out
                ],
            )
            distances = cosine_distances(views[bank_name][held_out], candidate)
            order = np.argsort(distances, kind="stable")
            total += 1
            if int(order[0]) == bank_index:
                correct += 1
                correct_margins.append(float(distances[order[1]] - distances[order[0]]))
    force_identity_fallback = not correct_margins
    if force_identity_fallback:
        # With no correct leave-one-view routes there is no support-only evidence
        # that this acquisition classifier is identifiable.  A maximal finite
        # threshold makes every finite query margin fall back to the frozen
        # identity bank instead of inventing a data-dependent cutoff.
        threshold = float(np.finfo(np.float64).max)
    else:
        threshold = float(
            np.quantile(np.asarray(correct_margins, dtype=np.float64), 0.05),
        )
    return threshold, {
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
        "correct_margin_count": len(correct_margins),
        "correct_margin_p05": threshold,
        "correct_margin_mean": (
            float(np.mean(correct_margins)) if correct_margins else None
        ),
        "force_identity_fallback": force_identity_fallback,
    }


def route_bank(
    vector: np.ndarray,
    prototypes: np.ndarray,
    bank_names: Sequence[str],
    confidence_threshold: float,
) -> RouteDecision:
    distances = cosine_distances(vector, prototypes)
    order = np.argsort(distances, kind="stable")
    nearest = int(order[0])
    margin = float(distances[order[1]] - distances[nearest])
    fallback = margin < confidence_threshold
    selected = "identity" if fallback else str(bank_names[nearest])
    return RouteDecision(
        nearest_bank=str(bank_names[nearest]),
        selected_bank=selected,
        nearest_distance=float(distances[nearest]),
        second_distance=float(distances[order[1]]),
        margin=margin,
        confidence_threshold=confidence_threshold,
        identity_fallback=fallback,
    )


def fused_and_early(
    backbone: BackboneLike,
    image: np.ndarray,
) -> tuple[FeatureMap, np.ndarray]:
    layered = extract_layer_feature_maps_from_rgb(
        backbone,
        image,
        FeatureExtractionConfig(feature_fusion="layer_norm_mean", context_source="none"),
    )
    layer_rows = [
        layer.values.reshape(-1, layer.values.shape[-1]) for layer in layered.layers
    ]
    fused_rows = merge_layer_features(layer_rows, "layer_norm_mean")
    height, width = layered.layers[0].values.shape[:2]
    fused = FeatureMap(
        values=fused_rows.reshape(height, width, fused_rows.shape[-1]),
        image_shape=layered.layers[0].image_shape,
    )
    early = normalize_vector(layered.layers[0].values.mean(axis=(0, 1)))
    return fused, early


def build_default_config(args: argparse.Namespace) -> RunConfig:
    return parse_default_args([
        "--data-root", args.data_root,
        "--output-root", args.output_root,
        "--project-root", args.project_root,
        "--fsad-root", args.fsad_root,
        "--objects", args.objects,
        "--shots", "16", "--seed", "0", "--device", args.device,
        "--flow-epochs", "3", "--coupling-layers", "2",
        "--hidden-multiplier", "1", "--flow-lr", "2e-4",
        "--flow-clamp", "1.9", "--flow-transform-mode", "flow",
        "--tail-weight", "0.3", "--tail-top-k-ratio", "0.05",
        "--lambda-logdet", "1e-3", "--density-quantile", "0.90",
        "--expansion-budget", "1.0", "--distance-weight", "1.0",
        "--density-weight", "0.25", "--score-mode", "latent_distance",
        "--top-percent", "0.01", "--query-chunk-size", "512",
        "--calibration-sample-size", "0", "--loo-standardization", "on",
        "--latent-bank-subsample", "none", "--pro-integration-limit", "0.05",
        "--rgb-guide", "none", "--binary-postprocess", "none",
        "--backbone-model", "dinov3_vith16plus", "--backbone-resolution", "0",
        "--feature-layers", "7,15,23,31", "--tile-patch-size", "0",
        "--tile-overlap", "0", "--image-resize-factor", "1.0",
        "--support-brightness-range", "1.0,1.0",
        "--support-selection", f"fixed_json={args.support_json}",
        "--support-selection-seed", "0", "--support-transforms", "identity",
        "--feature-fusion", "layer_norm_mean", "--normality-mode", "fused",
        "--dvt-denoise-mode", "position_mean", "--dvt-denoise-alpha", "1.0",
        "--context-source", "none", "--flow-context-source", "auto",
        "--memory-context-source", "auto", "--context-mode", "none",
        "--flow-condition-mode", "none", "--transformer-context-mode", "none",
        "--score-field-calibration-mode", "none",
        "--score-field-foreground-mode", "none",
    ])


def transformed_support_image(
    image: np.ndarray,
    spec: RGBTransformSpec | None,
) -> np.ndarray:
    return image if spec is None else apply_rgb_transform(image, spec)


def score_banks(
    pipeline: FlowTTE,
    feature_map: FeatureMap,
    banks: Mapping[str, TorchMemoryBank],
    selected_bank: str,
    score_config: ScoreConfig,
) -> tuple[dict[str, np.ndarray], float]:
    estimator = pipeline.estimator
    calibration = pipeline.score_calibration
    if estimator is None or calibration is None:
        raise RuntimeError("pipeline must be fitted before scoring")
    values = feature_map.values[np.newaxis, ...]
    evaluation = estimator.evaluate(values)
    density_penalty = estimator.density_penalty(evaluation.nll)
    per_bank_distances = {
        name: bank.query(
            evaluation.z,
            k=1,
            chunk_size=score_config.query_chunk_size,
            squared=score_config.use_squared_distance,
        ).distances[:, 0]
        for name, bank in banks.items()
    }

    def score(distance: torch.Tensor) -> torch.Tensor:
        return (
            score_config.distance_weight * calibration.normalize_distance(distance)
            + score_config.density_weight * density_penalty
        )

    identity_score = score(per_bank_distances["identity"])
    routed_score = score(per_bank_distances[selected_bank])
    minimum_distance = torch.stack(list(per_bank_distances.values()), dim=0).min(dim=0).values
    patchwise_score = score(minimum_distance)
    production = pipeline.score_static(values).patch_scores.reshape(-1)
    parity_error = float(
        np.max(np.abs(production - identity_score.detach().cpu().numpy())),
    )
    spatial_shape = feature_map.values.shape[:2]
    return {
        "identity": identity_score.detach().cpu().numpy().reshape(spatial_shape),
        "union": patchwise_score.detach().cpu().numpy().reshape(spatial_shape),
        "patchwise_min": patchwise_score.detach().cpu().numpy().reshape(spatial_shape),
        "routed": routed_score.detach().cpu().numpy().reshape(spatial_shape),
    }, parity_error


def run_object(args: argparse.Namespace, object_name: str) -> dict[str, Any]:  # noqa: PLR0915
    started = time.time()
    config = build_default_config(args)
    dataset, backbone = build_runtime(config)
    info = dataset.get_object_info(object_name)
    backbone.set_resolution(config.backbone_resolution or info.resolution)
    train_paths = [Path(path) for path in dataset.get_train_images(object_name)]
    support_paths = select_support_paths_for_backbone(
        backbone,
        train_paths,
        shots=16,
        policy=f"fixed_json={args.support_json}",
        seed=0,
    )
    if len(support_paths) != 16:
        raise RuntimeError("frozen support selection did not return exactly 16 paths")

    bank_names = tuple(name for name, _spec in BANK_SPECS)
    support_maps: dict[str, list[FeatureMap]] = {name: [] for name in bank_names}
    support_views: dict[str, list[np.ndarray]] = {name: [] for name in bank_names}
    for path in support_paths:
        image = read_rgb(path)
        for bank_name, spec in BANK_SPECS:
            feature_map, early = fused_and_early(
                backbone,
                transformed_support_image(image, spec),
            )
            support_maps[bank_name].append(feature_map)
            support_views[bank_name].append(early)

    identity_maps = support_maps["identity"]
    denoiser = fit_feature_denoiser(
        "position_mean",
        [feature_map.values for feature_map in identity_maps],
        alpha=1.0,
    )
    identity_features = flatten_support_feature_maps(
        identity_maps,
        require_contexts=False,
        feature_denoiser=denoiser,
    )
    pipeline = build_pipeline(config)
    stats = pipeline.fit(identity_features.values)
    if pipeline.estimator is None:
        raise RuntimeError("identity NF did not fit")

    banks: dict[str, TorchMemoryBank] = {}
    device = torch.device(config.device)
    for bank_name in bank_names:
        features = flatten_support_feature_maps(
            support_maps[bank_name],
            require_contexts=False,
            feature_denoiser=denoiser,
        ).values
        latent = torch.as_tensor(
            pipeline.transform_features(features),
            dtype=torch.float32,
            device=device,
        ).reshape(-1, features.shape[-1])
        bank = TorchMemoryBank()
        bank.fit(latent)
        banks[bank_name] = bank

    routing_prototypes = np.stack(
        [prototype(support_views[name]) for name in bank_names],
        axis=0,
    )
    confidence_threshold, support_routing = leave_one_view_margin_threshold(
        support_views,
        bank_names,
    )
    score_config = build_score_config(config)
    routes: list[dict[str, Any]] = []
    max_parity_error = 0.0
    test_images = dataset.get_test_images(object_name, split="test_public")
    items = stream_test_images(test_images)
    for item in items:
        image = read_rgb(item.path)
        raw_map, early = fused_and_early(backbone, image)
        feature_map = apply_feature_denoiser(raw_map, denoiser)
        decision = route_bank(
            early,
            routing_prototypes,
            bank_names,
            confidence_threshold,
        )
        scores, parity_error = score_banks(
            pipeline,
            feature_map,
            banks,
            decision.selected_bank,
            score_config,
        )
        max_parity_error = max(max_parity_error, parity_error)
        for arm, patch_scores in scores.items():
            score_map = cv2.resize(
                patch_scores.astype(np.float32, copy=False),
                (feature_map.image_shape[1], feature_map.image_shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            save_prediction(
                Path(args.output_root) / "objects" / object_name / arm,
                object_name,
                item,
                score_map,
            )
        routes.append({"stem": item.path.stem, **asdict(decision)})

    if max_parity_error > 1e-5:
        message = f"identity scorer parity error exceeded tolerance: {max_parity_error}"
        raise RuntimeError(message)
    object_root = Path(args.output_root) / "objects" / object_name
    object_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "object": object_name,
        "protocol": "support-bank-routing-stage2-20260718",
        "support_paths": [str(path) for path in support_paths],
        "bank_names": list(bank_names),
        "bank_size": {name: bank.size() for name, bank in banks.items()},
        "union_bank_size": sum(bank.size() for bank in banks.values()),
        "union_patchwise_equivalence": (
            "exact: nearest neighbor in the concatenated union equals the per-patch "
            "minimum of bank-wise nearest-neighbor distances"
        ),
        "support_routing": support_routing,
        "confidence_threshold": confidence_threshold,
        "routes": routes,
        "identity_scorer_max_abs_parity_error": max_parity_error,
        "train_nll_mean": stats.train_nll_mean,
        "train_nll_std": stats.train_nll_std,
        "density_threshold": stats.density_threshold,
        "query_transformation": "none",
        "query_memory_update": False,
        "test_labels_used_by_routing": False,
        "elapsed_seconds": time.time() - started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    (object_root / "routing.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--support-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--project-root", default="/workspace")
    parser.add_argument("--fsad-root", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    objects = tuple(value for value in args.objects.replace(",", " ").split() if value)
    if not objects or any(value not in OBJECTS for value in objects):
        parser.error(f"--objects must be a nonempty subset of {OBJECTS}")
    args.object_names = objects
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_cli(argv)
    summaries = [run_object(args, object_name) for object_name in args.object_names]
    output_root = Path(args.output_root)
    for summary in summaries:
        object_name = str(summary["object"])
        destination = output_root / "objects" / object_name / "runner_manifest.json"
        destination.write_text(
            json.dumps({
                "protocol": "support-bank-routing-stage2-20260718",
                "objects": [object_name],
                "arms": list(ARMS),
                "summary": summary,
                "claim_boundary": "diagnostic only; no TTA or query adaptation",
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
