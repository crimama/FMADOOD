#!/usr/bin/env python3
"""Unified CLI entry point for FMAD-OOD experiments.

Usage:
    python run.py --method superad --dataset mvtec_ad2 --backbone dinov2_vitl14 \
                  --data-root /path/to/mvtec_ad_2 --output-dir ./results

    python run.py --config configs/default.yaml

    python run.py --config configs/default.yaml --seed 1 --objects can,fabric
"""

import argparse
import os
import sys
import yaml

# Ensure project root is on path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fmad.registry import METHOD_REGISTRY, DATASET_REGISTRY, BACKBONE_REGISTRY

# Import to trigger registration
import fmad.methods   # noqa: F401
import fmad.datasets  # noqa: F401
import fmad.backbones  # noqa: F401

from fmad.pipeline import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="FMAD-OOD: Foundation Model Anomaly Detection under Distribution Shift"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file.")

    # Core selection
    parser.add_argument("--method", type=str, default=None,
                        help=f"Method name. Available: {METHOD_REGISTRY.list()}")
    parser.add_argument("--dataset", type=str, default=None,
                        help=f"Dataset name. Available: {DATASET_REGISTRY.list()}")
    parser.add_argument("--backbone", type=str, default=None,
                        help=f"Backbone name. Available: {BACKBONE_REGISTRY.list()}")

    # Paths
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root directory of the dataset.")
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Base directory for results output.")

    # Experiment
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--objects", type=str, default=None,
                        help="Comma-separated list of objects to run (default: all).")
    parser.add_argument("--device", type=str, default="cuda:0")

    # Method-specific overrides
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--knn-metric", type=str, default=None)
    parser.add_argument("--k-neighbors", type=int, default=None)
    parser.add_argument("--faiss-on-cpu", action="store_true", default=None)
    parser.add_argument("--preprocess", type=str, default=None)
    parser.add_argument("--warmup-iters", type=int, default=None)

    return parser.parse_args()


def load_config(args) -> dict:
    """Build config from YAML + CLI overrides."""
    config = {}

    # Load YAML if provided
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}

    # CLI overrides (only set values override YAML)
    _override = lambda key, val: config.__setitem__(key, val) if val is not None else None

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
        method_overrides["shots"] = args.shots
    if args.knn_metric is not None:
        method_overrides["knn_metric"] = args.knn_metric
    if args.k_neighbors is not None:
        method_overrides["k_neighbors"] = args.k_neighbors
    if args.faiss_on_cpu is not None:
        method_overrides["faiss_on_cpu"] = args.faiss_on_cpu
    if args.preprocess is not None:
        config.setdefault("dataset_config", {})["preprocess"] = args.preprocess
    if args.warmup_iters is not None:
        method_overrides["warmup_iters"] = args.warmup_iters

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
    config.setdefault("dataset_config", {})
    config.setdefault("evaluation", {})

    return config


def main():
    args = parse_args()
    config = load_config(args)

    # Validate required
    if not config.get("data_root"):
        print("Error: --data-root is required.", file=sys.stderr)
        sys.exit(1)

    # Set CUDA device
    device_str = config["device"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_str.split(":")[-1]) if ":" in device_str else "0"

    # Instantiate dataset
    dataset_cls = DATASET_REGISTRY.get(config["dataset"])
    dataset = dataset_cls(
        data_root=config["data_root"],
        config=config.get("dataset_config", {}),
    )

    # Instantiate backbone
    backbone_cls = BACKBONE_REGISTRY.get(config["backbone"])
    backbone = backbone_cls(
        model_name=config["backbone"],
        device="cuda",
    )

    # Instantiate method
    method_cls = METHOD_REGISTRY.get(config["method"])
    method = method_cls(
        backbone=backbone,
        config=config.get("method_config", {}),
    )

    # Run pipeline
    metrics = run_pipeline(
        method=method,
        dataset=dataset,
        backbone=backbone,
        config=config,
        output_dir=config["output_dir"],
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
