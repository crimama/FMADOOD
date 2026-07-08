# /// script
# requires-python = ">=3.8"
# dependencies = []
# ///
# pyright: reportMissingImports=false
"""CLI wrapper for the remote MVTec AD2 FlowTTE diagnostic."""
from __future__ import annotations

# allow: SIZE_OK — explicit remote experiment CLI mirrors the protocol surface;
# split after the structural-context options stabilize.
import argparse
import importlib
import json
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (_REPO_ROOT, _REPO_ROOT / "src", Path(__file__).resolve().parent):
    _path_text = str(_path)
    if _path_text not in sys.path:
        sys.path.insert(0, _path_text)

from dinov3_backbone import DINOv3Backbone, is_dinov3_model_name  # noqa: E402
from flow_tte_mvtec_ad2_core import (  # noqa: E402
    BackboneLike,
    DatasetLike,
    ObjectDiagnostics,
    RunConfig,
    resolve_flow_context_source,
    resolve_memory_context_source,
    resolve_score_context_mode,
    run_object,
)
from flow_tte_superadd_preprocess import parse_brightness_range, parse_feature_layers  # noqa: E402
from flow_tte_support import is_fixed_support_policy  # noqa: E402

JsonScalar = Union[str, int, float, bool, None]
JsonValue = Union[JsonScalar, Dict[str, "JsonValue"], List["JsonValue"]]


def parse_args(argv: Sequence[str]) -> RunConfig:
    parser = argparse.ArgumentParser(description="Run FlowTTE on MVTec AD2 TESTpublic.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--project-root", default="/workspace")
    parser.add_argument("--fsad-root", default="/workspace/fsad_tta")
    parser.add_argument("--objects", default="can,rice")
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--flow-epochs", type=int, default=5)
    parser.add_argument("--coupling-layers", type=int, default=2)
    parser.add_argument("--hidden-multiplier", type=int, default=1)
    parser.add_argument("--flow-lr", type=float, default=2e-4)
    parser.add_argument("--flow-clamp", type=float, default=1.9)
    parser.add_argument(
        "--flow-transform-mode",
        choices=("flow", "identity"),
        default="flow",
    )
    parser.add_argument("--tail-weight", type=float, default=0.3)
    parser.add_argument("--tail-top-k-ratio", type=float, default=0.05)
    parser.add_argument("--lambda-logdet", type=float, default=1e-3)
    parser.add_argument("--density-quantile", type=float, default=0.90)
    parser.add_argument("--expansion-budget", type=float, default=1.25)
    parser.add_argument("--distance-weight", type=float, default=1.0)
    parser.add_argument("--density-weight", type=float, default=0.25)
    parser.add_argument(
        "--score-mode",
        choices=("latent_distance", "nf_nll"),
        default="latent_distance",
    )
    parser.add_argument("--top-percent", type=float, default=0.01)
    parser.add_argument("--query-chunk-size", type=int, default=512)
    parser.add_argument("--pro-integration-limit", type=float, default=0.05)
    parser.add_argument("--cleanup-maps", action="store_true")
    parser.add_argument("--use-squared-distance", action="store_true")
    parser.add_argument("--backbone-model", default="dinov2_vitl14")
    add_superadd_alignment_args(parser)
    parser.add_argument("--support-selection", default="first")
    parser.add_argument("--support-selection-seed", type=int, default=0)
    parser.add_argument("--support-transforms", default="identity")
    parser.add_argument("--feature-fusion", default="layer_norm_mean")
    parser.add_argument(
        "--normality-mode",
        choices=(
            "fused",
            "layer_wise",
            "raw_nn",
            "raw_layer_wise",
            "raw_nn_nf_residual",
            "foreground_raw_nn",
        ),
        default="fused",
    )
    parser.add_argument("--residual-weight", type=float, default=0.25)
    parser.add_argument(
        "--dvt-denoise-mode",
        choices=("none", "position_mean"),
        default="none",
    )
    parser.add_argument("--dvt-denoise-alpha", type=float, default=1.0)
    parser.add_argument(
        "--context-source",
        choices=("none", "cls", "register", "cls_register"),
        default="none",
    )
    parser.add_argument(
        "--flow-context-source",
        choices=("auto", "none", "cls", "register", "cls_register"),
        default="auto",
    )
    parser.add_argument(
        "--memory-context-source",
        choices=("auto", "none", "cls", "register", "cls_register"),
        default="auto",
    )
    parser.add_argument(
        "--context-mode",
        choices=("auto", "none", "soft_penalty", "top_m"),
        default="auto",
    )
    parser.add_argument("--context-weight", type=float, default=0.0)
    parser.add_argument("--context-top-m", type=int, default=1)
    parser.add_argument(
        "--flow-condition-mode",
        choices=("none", "context"),
        default="none",
    )
    add_score_field_args(parser)
    args = parser.parse_args(list(argv))
    objects = tuple(part for part in args.objects.replace(",", " ").split() if part)
    validate_args(args, objects)
    return RunConfig(
        data_root=Path(args.data_root),
        output_root=Path(args.output_root),
        project_root=Path(args.project_root),
        fsad_root=Path(args.fsad_root),
        objects=objects,
        shots=args.shots,
        seed=args.seed,
        device=args.device,
        flow_epochs=args.flow_epochs,
        coupling_layers=args.coupling_layers,
        hidden_multiplier=args.hidden_multiplier,
        flow_lr=args.flow_lr,
        flow_clamp=args.flow_clamp,
        flow_transform_mode=args.flow_transform_mode,
        tail_weight=args.tail_weight,
        tail_top_k_ratio=args.tail_top_k_ratio,
        lambda_logdet=args.lambda_logdet,
        density_quantile=args.density_quantile,
        expansion_budget=args.expansion_budget,
        distance_weight=args.distance_weight,
        density_weight=args.density_weight,
        score_mode=args.score_mode,
        top_percent=args.top_percent,
        query_chunk_size=args.query_chunk_size,
        pro_integration_limit=args.pro_integration_limit,
        cleanup_maps=args.cleanup_maps,
        use_squared_distance=args.use_squared_distance,
        flow_condition_mode=args.flow_condition_mode,
        backbone_model=args.backbone_model,
        backbone_resolution=(
            args.backbone_resolution if args.backbone_resolution > 0 else None
        ),
        feature_layers=parse_feature_layers(args.feature_layers),
        tile_patch_size=args.tile_patch_size,
        tile_overlap=args.tile_overlap,
        image_resize_factor=args.image_resize_factor,
        support_brightness_range=parse_brightness_range(args.support_brightness_range),
        feature_fusion=args.feature_fusion,
        support_selection=args.support_selection,
        support_selection_seed=args.support_selection_seed,
        support_transforms=parse_transform_tuple(args.support_transforms),
        dvt_denoise_mode=args.dvt_denoise_mode,
        dvt_denoise_alpha=args.dvt_denoise_alpha,
        normality_mode=args.normality_mode,
        residual_weight=args.residual_weight,
        context_source=args.context_source,
        flow_context_source=args.flow_context_source,
        memory_context_source=args.memory_context_source,
        context_mode=args.context_mode,
        context_weight=args.context_weight,
        context_top_m=args.context_top_m,
        score_field_calibration_mode=args.score_field_calibration_mode,
        score_field_calibration_alpha=args.score_field_calibration_alpha,
        score_field_position_std_floor=args.score_field_position_std_floor,
        score_field_foreground_mode=args.score_field_foreground_mode,
        score_field_foreground_quantile=args.score_field_foreground_quantile,
        score_field_background_multiplier=args.score_field_background_multiplier,
        score_field_foreground_smooth_kernel=args.score_field_foreground_smooth_kernel,
    )


def validate_args(args: argparse.Namespace, objects: Tuple[str, ...]) -> None:
    if not objects:
        raise SystemExit("--objects must contain at least one object")
    if args.shots <= 0:
        raise SystemExit("--shots must be positive")
    if args.context_top_m <= 0:
        raise SystemExit("--context-top-m must be positive")
    if args.dvt_denoise_alpha < 0.0:
        raise SystemExit("--dvt-denoise-alpha must be non-negative")
    if args.residual_weight < 0.0:
        raise SystemExit("--residual-weight must be non-negative")
    validate_score_field_args(args)
    validate_superadd_alignment_args(args)
    flow_context_source = args.flow_context_source
    if flow_context_source == "auto":
        flow_context_source = args.context_source
    memory_context_source = args.memory_context_source
    if memory_context_source == "auto":
        memory_context_source = args.context_source
    if memory_context_source == "none" and args.context_mode in ("soft_penalty", "top_m"):
        raise SystemExit("--context-mode requires a memory context source")
    if flow_context_source == "none" and args.flow_condition_mode == "context":
        raise SystemExit("--flow-condition-mode context requires a flow context source")


def add_score_field_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--score-field-calibration-mode",
        choices=("none", "support_position_center", "support_position_zscore"),
        default="none",
    )
    parser.add_argument("--score-field-calibration-alpha", type=float, default=1.0)
    parser.add_argument("--score-field-position-std-floor", type=float, default=0.25)
    parser.add_argument(
        "--score-field-foreground-mode",
        choices=("none", "support_feature_energy"),
        default="none",
    )
    parser.add_argument("--score-field-foreground-quantile", type=float, default=0.20)
    parser.add_argument("--score-field-background-multiplier", type=float, default=0.50)
    parser.add_argument("--score-field-foreground-smooth-kernel", type=int, default=5)


def validate_score_field_args(args: argparse.Namespace) -> None:
    if args.score_field_calibration_alpha < 0.0:
        raise SystemExit("--score-field-calibration-alpha must be non-negative")
    if args.score_field_position_std_floor <= 0.0:
        raise SystemExit("--score-field-position-std-floor must be positive")
    if not 0.0 <= args.score_field_foreground_quantile <= 1.0:
        raise SystemExit("--score-field-foreground-quantile must be in [0, 1]")
    if not 0.0 <= args.score_field_background_multiplier <= 1.0:
        raise SystemExit("--score-field-background-multiplier must be in [0, 1]")
    if args.score_field_foreground_smooth_kernel <= 0:
        raise SystemExit("--score-field-foreground-smooth-kernel must be positive")


def add_superadd_alignment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backbone-resolution", type=int, default=0)
    parser.add_argument("--feature-layers", default="5,11,17,23")
    parser.add_argument("--tile-patch-size", type=int, default=0)
    parser.add_argument("--tile-overlap", type=int, default=0)
    parser.add_argument("--image-resize-factor", type=float, default=1.0)
    parser.add_argument("--support-brightness-range", default="1.0,1.0")


def validate_superadd_alignment_args(args: argparse.Namespace) -> None:
    if args.backbone_resolution < 0:
        raise SystemExit("--backbone-resolution must be non-negative")
    if args.tile_patch_size < 0:
        raise SystemExit("--tile-patch-size must be non-negative")
    if args.tile_overlap < 0:
        raise SystemExit("--tile-overlap must be non-negative")
    if args.tile_patch_size > 0 and args.tile_overlap >= args.tile_patch_size:
        raise SystemExit("--tile-overlap must be smaller than --tile-patch-size")
    if args.image_resize_factor <= 0.0:
        raise SystemExit("--image-resize-factor must be positive")


def parse_transform_tuple(raw: str) -> Tuple[str, ...]:
    if raw == "visionad_rot_flip":
        return ("identity", "rot90", "rot180", "rot270", "flip_vertical", "flip_horizontal")
    values = tuple(part for part in raw.replace(",", " ").split() if part)
    if not values:
        raise SystemExit("support transform list must not be empty")
    return values


def add_import_paths(config: RunConfig) -> None:
    for path in (config.project_root, config.fsad_root / "src"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def build_runtime(config: RunConfig) -> Tuple[DatasetLike, BackboneLike]:
    dataset_module = importlib.import_module("fmad.datasets.mvtec_ad2")
    dataset_cls = dataset_module.MVTecAD2Dataset
    dataset = dataset_cls(
        data_root=str(config.data_root),
        config={"objects": list(config.objects), "preprocess": "no_mask_no_rotation"},
    )
    if is_dinov3_model_name(config.backbone_model):
        backbone = DINOv3Backbone(
            model_name=config.backbone_model,
            device=config.device,
            smaller_edge_size=672,
            feature_layers=config.feature_layers,
        )
    else:
        backbone_module = importlib.import_module("fmad.backbones.dinov2")
        backbone_cls = backbone_module.DINOv2Backbone
        backbone = backbone_cls(
            model_name=config.backbone_model,
            device=config.device,
            smaller_edge_size=672,
        )
    return dataset, backbone


def evaluate(config: RunConfig) -> Dict[str, JsonValue]:
    metrics_module = importlib.import_module("fmad.evaluation.metrics")
    evaluator_cls = metrics_module.Evaluator
    evaluator = evaluator_cls({"pro_integration_limit": config.pro_integration_limit})
    metrics = evaluator.evaluate_run(
        dataset_name="MVTec_AD_2",
        data_root=str(config.data_root),
        anomaly_maps_dir=str(config.output_root / "anomaly_maps"),
        output_dir=str(config.output_root),
        seed=config.seed,
        objects=list(config.objects),
    )
    (config.output_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def write_manifest(
    config: RunConfig,
    diagnostics: Sequence[ObjectDiagnostics],
    metrics: Dict[str, JsonValue],
) -> None:
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_dataset": "MVTec AD2 single-image",
        "data_root": str(config.data_root),
        "method": f"flow_tte_{config.normality_mode}",
        "method_family": "normal_only_adaptation_with_anti_absorption",
        "split": "test_public/good,bad",
        "objects": list(config.objects),
        "shots": config.shots,
        "support_policy": config.support_selection,
        "support_selection_seed": config.support_selection_seed,
        "support_transforms": list(config.support_transforms),
        "dvt_denoise_mode": config.dvt_denoise_mode,
        "dvt_denoise_alpha": config.dvt_denoise_alpha,
        "stream_order": "test image basename, then anomaly type",
        "preprocess": "no_mask_no_rotation",
        "backbone": f"{config.backbone_model} layer mean {list(config.feature_layers)}",
        "backbone_resolution": config.backbone_resolution,
        "tile_patch_size": config.tile_patch_size,
        "tile_overlap": config.tile_overlap,
        "image_resize_factor": config.image_resize_factor,
        "support_brightness_range": [
            config.support_brightness_range.min_factor,
            config.support_brightness_range.max_factor,
        ],
        "feature_fusion": config.feature_fusion,
        "normality_mode": config.normality_mode,
        "residual_weight": config.residual_weight,
        "flow_epochs": config.flow_epochs,
        "coupling_layers": config.coupling_layers,
        "hidden_multiplier": config.hidden_multiplier,
        "flow_lr": config.flow_lr,
        "flow_clamp": config.flow_clamp,
        "flow_transform_mode": config.flow_transform_mode,
        "tail_weight": config.tail_weight,
        "tail_top_k_ratio": config.tail_top_k_ratio,
        "lambda_logdet": config.lambda_logdet,
        "density_quantile": config.density_quantile,
        "expansion_budget": config.expansion_budget,
        "distance_weight": config.distance_weight,
        "density_weight": config.density_weight,
        "score_mode": config.score_mode,
        "flow_condition_mode": config.flow_condition_mode,
        "context_source": config.context_source,
        "flow_context_source": config.flow_context_source,
        "memory_context_source": config.memory_context_source,
        "resolved_flow_context_source": resolve_flow_context_source(config),
        "resolved_memory_context_source": resolve_memory_context_source(config),
        "context_mode": resolve_score_context_mode(config),
        "context_weight": config.context_weight,
        "context_top_m": config.context_top_m,
        "score_field_calibration_mode": config.score_field_calibration_mode,
        "score_field_calibration_alpha": config.score_field_calibration_alpha,
        "score_field_position_std_floor": config.score_field_position_std_floor,
        "score_field_foreground_mode": config.score_field_foreground_mode,
        "score_field_foreground_quantile": config.score_field_foreground_quantile,
        "score_field_background_multiplier": config.score_field_background_multiplier,
        "score_field_foreground_smooth_kernel": config.score_field_foreground_smooth_kernel,
        "primary_metrics": ["seg_AUROC_0.05", "seg_F1"],
        "reference_budget_matched": (
            config.shots == 16
            and (
                (
                    config.support_selection == "dinov2_cls_greedy_coreset"
                    and config.backbone_model == "dinov2_vitl14"
                )
                or is_fixed_support_policy(config.support_selection)
            )
        ),
        "strict_table1_claim_comparable": False,
        "strict_method_claim_supported": False,
        "claim_scope": (
            "all-eight-object few-shot diagnostic"
            if len(config.objects) == 8
            else "reduced-object few-shot diagnostic only"
        ),
        "object_diagnostics": [asdict(item) for item in diagnostics],
        "metrics": metrics,
    }
    (config.output_root / "run_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def cleanup_maps(output_root: Path) -> None:
    maps_dir = output_root / "anomaly_maps"
    if maps_dir.exists():
        shutil.rmtree(maps_dir)
    (output_root / "cleanup_evidence.txt").write_text(
        "cleanup_anomaly_maps=true\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str]) -> int:
    config = parse_args(argv)
    torch.manual_seed(config.seed)
    config.output_root.mkdir(parents=True, exist_ok=True)
    add_import_paths(config)
    dataset, backbone = build_runtime(config)
    diagnostics = [
        run_object(dataset, backbone, object_name, config)
        for object_name in config.objects
    ]
    metrics = evaluate(config)
    write_manifest(config, diagnostics, metrics)
    if config.cleanup_maps:
        cleanup_maps(config.output_root)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
