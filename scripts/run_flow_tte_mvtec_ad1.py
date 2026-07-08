# /// script
# requires-python = ">=3.8"
# dependencies = []
# ///
# pyright: reportMissingImports=false
"""CLI wrapper for the remote classic MVTec AD1 FlowTTE diagnostic."""
from __future__ import annotations

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

from flow_tte_mvtec_ad2_core import (  # noqa: E402
    BackboneLike,
    ObjectDiagnostics,
    RunConfig,
    run_object,
)
from flow_tte_mvtec_classic import (  # noqa: E402
    ClassicEvaluationConfig,
    ClassicMVTecDataset,
    evaluate_classic_mvtec,
)

JsonScalar = Union[str, int, float, bool, None]
JsonValue = Union[JsonScalar, Dict[str, "JsonValue"], List["JsonValue"]]


def parse_args(argv: Sequence[str]) -> RunConfig:
    parser = argparse.ArgumentParser(description="Run FlowTTE on classic MVTec AD1.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--project-root", default="/workspace")
    parser.add_argument("--fsad-root", default="/workspace/fsad_tta")
    parser.add_argument("--objects", default="bottle,hazelnut")
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--flow-epochs", type=int, default=3)
    parser.add_argument("--coupling-layers", type=int, default=2)
    parser.add_argument("--hidden-multiplier", type=int, default=1)
    parser.add_argument("--flow-lr", type=float, default=2e-4)
    parser.add_argument("--flow-clamp", type=float, default=1.9)
    parser.add_argument("--tail-weight", type=float, default=0.3)
    parser.add_argument("--tail-top-k-ratio", type=float, default=0.05)
    parser.add_argument("--lambda-logdet", type=float, default=1e-3)
    parser.add_argument("--density-quantile", type=float, default=0.90)
    parser.add_argument("--expansion-budget", type=float, default=1.25)
    parser.add_argument("--distance-weight", type=float, default=1.0)
    parser.add_argument("--density-weight", type=float, default=0.25)
    parser.add_argument("--top-percent", type=float, default=0.01)
    parser.add_argument("--query-chunk-size", type=int, default=512)
    parser.add_argument("--pro-integration-limit", type=float, default=0.05)
    parser.add_argument("--cleanup-maps", action="store_true")
    parser.add_argument("--use-squared-distance", action="store_true")
    parser.add_argument("--backbone-model", default="dinov2_vitl14")
    parser.add_argument("--preprocess-recipe", default="fmad_shorter_edge")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--crop-size", type=int, default=448)
    parser.add_argument("--feature-layers", default="5,11,17,23")
    parser.add_argument("--feature-fusion", default="layer_norm_mean")
    parser.add_argument("--support-selection", default="first")
    parser.add_argument("--support-selection-seed", type=int, default=0)
    parser.add_argument("--support-transforms", default="identity")
    args = parser.parse_args(list(argv))
    objects = tuple(part for part in args.objects.replace(",", " ").split() if part)
    if not objects:
        raise SystemExit("--objects must contain at least one object")
    if args.shots <= 0:
        raise SystemExit("--shots must be positive")
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
        tail_weight=args.tail_weight,
        tail_top_k_ratio=args.tail_top_k_ratio,
        lambda_logdet=args.lambda_logdet,
        density_quantile=args.density_quantile,
        expansion_budget=args.expansion_budget,
        distance_weight=args.distance_weight,
        density_weight=args.density_weight,
        top_percent=args.top_percent,
        query_chunk_size=args.query_chunk_size,
        pro_integration_limit=args.pro_integration_limit,
        cleanup_maps=args.cleanup_maps,
        use_squared_distance=args.use_squared_distance,
        backbone_model=args.backbone_model,
        preprocess_recipe=args.preprocess_recipe,
        image_size=args.image_size,
        crop_size=args.crop_size,
        feature_layers=parse_int_tuple(args.feature_layers),
        feature_fusion=args.feature_fusion,
        support_selection=args.support_selection,
        support_selection_seed=args.support_selection_seed,
        support_transforms=parse_transform_tuple(args.support_transforms),
    )


def parse_int_tuple(raw: str) -> Tuple[int, ...]:
    values = tuple(int(part) for part in raw.replace(",", " ").split() if part)
    if not values:
        raise SystemExit("integer list must not be empty")
    return values


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


def build_runtime(config: RunConfig) -> Tuple[ClassicMVTecDataset, BackboneLike]:
    dataset = ClassicMVTecDataset(
        data_root=str(config.data_root),
        objects=config.objects,
        resolution=config.crop_size,
    )
    if config.preprocess_recipe == "visionad_official":
        backbone_module = importlib.import_module("visionad_aligned_backbone")
        backbone_cls = backbone_module.VisionADAlignedBackbone
        backbone = backbone_cls(
            model_name=config.backbone_model,
            device=config.device,
            image_size=config.crop_size if config.image_size <= 0 else config.image_size,
            crop_size=config.crop_size,
            feature_layers=config.feature_layers,
        )
        return dataset, backbone
    backbone_module = importlib.import_module("fmad.backbones.dinov2")
    backbone_cls = backbone_module.DINOv2Backbone
    backbone = backbone_cls(
        model_name=config.backbone_model,
        device=config.device,
        smaller_edge_size=config.crop_size,
    )
    return dataset, backbone


def evaluate(config: RunConfig, dataset: ClassicMVTecDataset) -> Dict[str, JsonValue]:
    return evaluate_classic_mvtec(
        ClassicEvaluationConfig(
            dataset=dataset,
            output_root=config.output_root,
            objects=config.objects,
            pro_integration_limit=config.pro_integration_limit,
            seed=config.seed,
            image_top_fraction=config.top_percent,
        ),
    )


def write_manifest(
    config: RunConfig,
    diagnostics: Sequence[ObjectDiagnostics],
    metrics: Dict[str, JsonValue],
) -> None:
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_dataset": "MVTec AD1 classic single-image",
        "data_root": str(config.data_root),
        "method": "flow_tte_nf",
        "method_family": "normal_only_adaptation_with_anti_absorption",
        "split": "test/good,defect_types",
        "objects": list(config.objects),
        "shots": config.shots,
        "support_policy": config.support_selection,
        "support_selection_seed": config.support_selection_seed,
        "support_transforms": list(config.support_transforms),
        "stream_order": "test image basename, then anomaly type",
        "preprocess": config.preprocess_recipe,
        "image_size": config.image_size,
        "crop_size": config.crop_size,
        "backbone": config.backbone_model,
        "feature_layers": list(config.feature_layers),
        "feature_fusion": config.feature_fusion,
        "flow_epochs": config.flow_epochs,
        "coupling_layers": config.coupling_layers,
        "hidden_multiplier": config.hidden_multiplier,
        "flow_lr": config.flow_lr,
        "flow_clamp": config.flow_clamp,
        "tail_weight": config.tail_weight,
        "tail_top_k_ratio": config.tail_top_k_ratio,
        "lambda_logdet": config.lambda_logdet,
        "density_quantile": config.density_quantile,
        "expansion_budget": config.expansion_budget,
        "distance_weight": config.distance_weight,
        "density_weight": config.density_weight,
        "top_percent": config.top_percent,
        "image_top_fraction": config.top_percent,
        "use_squared_distance": config.use_squared_distance,
        "primary_metrics": ["seg_AUROC_0.05", "seg_F1"],
        "superad_baseline_source": "BLOCKED_BASELINE",
        "superad_comparable": False,
        "strict_method_claim_supported": False,
        "claim_scope": "reduced-object few-shot diagnostic only",
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
    metrics = evaluate(config, dataset)
    write_manifest(config, diagnostics, metrics)
    if config.cleanup_maps:
        cleanup_maps(config.output_root)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
