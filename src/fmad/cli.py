#!/usr/bin/env python3
"""Unified CLI entry point for FMAD-OOD experiments.

Usage:
    python run.py --method superad --dataset mvtec_ad2 --backbone dinov2_vitl14 \
                  --data-root /path/to/mvtec_ad_2 --output-dir ./results

    python run.py --config configs/default.yaml

    python run.py --config configs/default.yaml --seed 1 --objects can,fabric
"""

import argparse
import sys
from copy import deepcopy
import yaml
from fmad.protocol import validate_protocol
from fmad.results import validate_output_options

from fmad.registry import METHOD_REGISTRY, DATASET_REGISTRY, BACKBONE_REGISTRY

# Dataset/backbone packages are lightweight registration surfaces.  Concrete
# methods are loaded in main() so config tooling can run without optional
# native method dependencies such as FAISS.
import fmad.methods  # noqa: F401
import fmad.datasets  # noqa: F401
import fmad.backbones  # noqa: F401


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="FMAD-OOD: Foundation Model Anomaly Detection under Distribution Shift"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")

    # Core selection
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Registered method name (default: superad; PSMA_MR is available).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=f"Dataset name. Available: {DATASET_REGISTRY.list()}",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help=f"Backbone name. Available: {BACKBONE_REGISTRY.list()}",
    )

    # Paths
    parser.add_argument(
        "--data-root", type=str, default=None, help="Root directory of the dataset."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Base directory for results output."
    )

    # Experiment
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--objects",
        type=str,
        default=None,
        help="Comma-separated list of objects to run (default: all).",
    )
    parser.add_argument("--device", type=str, default=None)

    # Method-specific overrides
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--knn-metric", type=str, default=None)
    parser.add_argument("--k-neighbors", type=int, default=None)
    parser.add_argument("--faiss-on-cpu", action="store_true", default=None)
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=None,
        help="PSMA_MR short-edge resolutions (comma- or space-separated; default: 448 672).",
    )
    parser.add_argument(
        "--adapter-rank",
        type=int,
        default=None,
        help="PSMA_MR shared adapter rank (default: 4).",
    )
    parser.add_argument("--preprocess", type=str, default=None)
    parser.add_argument("--warmup-iters", type=int, default=None)
    parser.add_argument("--fixed-reference-manifest", default=None)
    parser.add_argument("--debug-output", action="store_true", default=None)
    parser.add_argument("--save-anomaly-maps", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-weights", action=argparse.BooleanOptionalAction, default=None)

    return parser.parse_args(argv)


def load_config(args) -> dict:
    """Build config from YAML + CLI overrides."""
    config = {}

    # Load YAML if provided
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    for section in ("method_config", "backbone_config", "dataset_config", "evaluation", "protocol"):
        if section in config and not isinstance(config[section], dict):
            raise ValueError(f"{section} must be a mapping")

    # Migrate the old location while keeping experiment controls pipeline-owned.
    protocol = config.setdefault("protocol", {})
    method_config = config.setdefault("method_config", {})
    for key in ("shots", "fixed_reference_manifest", "warmup_iters"):
        if key in method_config:
            value = method_config.pop(key)
            if key in protocol and protocol[key] != value:
                raise ValueError(f"conflicting protocol.{key} and method_config.{key}")
            protocol[key] = value
    method_config.pop("save_heatmaps", None)

    # CLI overrides (only set values override YAML)
    def _override(key, value):
        if value is not None:
            config[key] = value

    _override("method", args.method)
    _override("dataset", args.dataset)
    _override("backbone", args.backbone)
    _override("data_root", args.data_root)
    _override("output_dir", args.output_dir)
    _override("seed", args.seed)
    _override("device", args.device)

    if args.objects:
        config.setdefault("dataset_config", {})["objects"] = args.objects.split(",")

    # Method config overrides
    method_overrides = {}
    if args.shots is not None:
        protocol["shots"] = args.shots
    if args.knn_metric is not None:
        method_overrides["knn_metric"] = args.knn_metric
    if args.k_neighbors is not None:
        method_overrides["k_neighbors"] = args.k_neighbors
    if args.faiss_on_cpu is not None:
        method_overrides["faiss_on_cpu"] = args.faiss_on_cpu
    if args.resolutions is not None:
        values = []
        for token in args.resolutions:
            values.extend(token.split(","))
        try:
            resolutions = [int(value) for value in values if value.strip()]
        except ValueError as error:
            raise ValueError("resolutions must contain positive integers") from error
        if not resolutions or any(value <= 0 for value in resolutions):
            raise ValueError("resolutions must contain positive integers")
        if len(set(resolutions)) != len(resolutions):
            raise ValueError("resolutions must not contain duplicates")
        method_overrides["resolutions"] = resolutions
    if args.adapter_rank is not None:
        if args.adapter_rank <= 0:
            raise ValueError("adapter_rank must be a positive integer")
        method_overrides["adapter_rank"] = args.adapter_rank
    if args.preprocess is not None:
        config.setdefault("dataset_config", {})["preprocess"] = args.preprocess
    if args.warmup_iters is not None:
        protocol["warmup_iters"] = args.warmup_iters
    if args.fixed_reference_manifest is not None:
        protocol["fixed_reference_manifest"] = args.fixed_reference_manifest
    if args.debug_output is not None:
        config["debug_output"] = args.debug_output
    output = config.setdefault("output", {})
    if not isinstance(output, dict):
        raise ValueError("output must be a mapping")
    for key in ("save_anomaly_maps", "save_weights"):
        value = getattr(args, key)
        if value is not None:
            output[key] = value
        output.setdefault(key, False)

    if method_overrides:
        config.setdefault("method_config", {}).update(method_overrides)

    # Defaults
    config.setdefault("method", "superad")
    config.setdefault("dataset", "mvtec_ad2")
    config.setdefault("backbone", "dinov2_vitl14")
    config.setdefault("seed", 0)
    config.setdefault("device", "cuda:0")
    config.setdefault("output_dir", "./results")
    config.setdefault("method_config", {})
    config.setdefault("backbone_config", {})
    config.setdefault("dataset_config", {})
    config.setdefault("evaluation", {})
    config.setdefault("debug_output", False)
    defaults = {
        "protocol": {"shots": 16, "fixed_reference_manifest": None, "warmup_iters": 25},
        "method_config": {
            "knn_metric": "L2_normalized",
            "k_neighbors": 1,
            "faiss_on_cpu": False,
            "mask_ref_images": False,
        },
        "dataset_config": {"preprocess": "agnostic"},
        "evaluation": {
            "streaming": True,
            "inference_batch_size": 1,
            "metric_contract": "mvtec_ad2_float16_streaming_v1",
            "pro_integration_limit": 0.05,
        },
    }
    for section, values in defaults.items():
        for key, value in values.items():
            config[section].setdefault(key, deepcopy(value))
    if config["method"] == "PSMA_MR":
        config["method_config"].setdefault("resolutions", [448, 672])
        config["method_config"].setdefault("adapter_rank", 4)
    validate_protocol(config["protocol"])
    validate_output_options(config)

    return config


def resolve_runtime_device(device: str) -> str:
    """Preserve the requested process-visible CUDA slot without remapping it."""
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return "cuda"
    if device.startswith("cuda:") and device.split(":", 1)[1].isdigit():
        return device
    raise ValueError(f"unsupported device: {device}")


def main():
    args = parse_args()
    config = load_config(args)

    # Validate required
    if not config.get("data_root"):
        print("Error: --data-root is required.", file=sys.stderr)
        sys.exit(1)

    runtime_device = resolve_runtime_device(config["device"])

    # Load runtime dependencies only after CLI parsing and validation.
    import fmad.backbones  # noqa: F401
    from fmad.methods import register_methods
    from fmad.pipeline import run_pipeline
    from fmad.protocol import seed_generators

    seed_generators(config["seed"])
    register_methods(config["method"])

    # Instantiate dataset
    dataset_cls = DATASET_REGISTRY.get(config["dataset"])
    dataset = dataset_cls(
        data_root=config["data_root"],
        config=config.get("dataset_config", {}),
    )

    # Instantiate backbone
    backbone_cls = BACKBONE_REGISTRY.get(config["backbone"])
    backbone_config = dict(config.get("backbone_config", {}))
    backbone = backbone_cls(
        model_name=config["backbone"],
        device=runtime_device,
        **backbone_config,
    )

    # Instantiate method
    method_cls = METHOD_REGISTRY.get(config["method"])
    method = method_cls(
        backbone=backbone,
        config=config.get("method_config", {}),
    )

    # Run pipeline
    run_pipeline(
        method=method,
        dataset=dataset,
        backbone=backbone,
        config=config,
        output_dir=config["output_dir"],
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
