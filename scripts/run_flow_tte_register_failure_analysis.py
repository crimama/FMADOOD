# /// script
# requires-python = ">=3.8"
# dependencies = []
# ///
# pyright: reportMissingImports=false
"""Run FlowTTE register/CLS failure-mode analysis on MVTec AD2."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (_REPO_ROOT, _REPO_ROOT / "src", Path(__file__).resolve().parent):
    _path_text = str(_path)
    if _path_text not in sys.path:
        sys.path.insert(0, _path_text)

from dinov3_backbone import DINOv3Backbone  # noqa: E402
from flow_tte_register_analysis_extract import build_dataset  # noqa: E402
from flow_tte_register_analysis_metrics import analyze_object  # noqa: E402
from flow_tte_register_analysis_types import (  # noqa: E402
    AnalysisConfig,
    JsonValue,
    load_support_paths,
    write_tsv,
)


def parse_args(argv: Sequence[str]) -> AnalysisConfig:
    parser = argparse.ArgumentParser(description="Analyze FlowTTE register failure modes.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--project-root", default="/workspace")
    parser.add_argument("--fsad-root", default="/workspace/fsad_tta")
    parser.add_argument("--support-json", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patch-samples-per-image", type=int, default=128)
    parser.add_argument("--context-top-m", type=int, default=4)
    args = parser.parse_args(list(argv))
    objects = tuple(part for part in args.objects.replace(",", " ").split() if part)
    if not objects:
        raise SystemExit("--objects must contain at least one object")
    if args.patch_samples_per_image <= 0:
        raise SystemExit("--patch-samples-per-image must be positive")
    if args.context_top_m <= 0:
        raise SystemExit("--context-top-m must be positive")
    return AnalysisConfig(
        data_root=Path(args.data_root),
        output_root=Path(args.output_root),
        project_root=Path(args.project_root),
        fsad_root=Path(args.fsad_root),
        support_json=Path(args.support_json),
        objects=objects,
        device=args.device,
        seed=args.seed,
        patch_samples_per_image=args.patch_samples_per_image,
        context_top_m=args.context_top_m,
    )


def add_import_paths(config: AnalysisConfig) -> None:
    for path in (config.project_root, config.fsad_root / "src"):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def write_manifest(config: AnalysisConfig) -> None:
    manifest: Dict[str, JsonValue] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_dataset": "MVTec AD2 single-image",
        "data_root": str(config.data_root),
        "objects": list(config.objects),
        "support_json": str(config.support_json),
        "device": config.device,
        "seed": config.seed,
        "patch_samples_per_image": config.patch_samples_per_image,
        "context_top_m": config.context_top_m,
        "analysis": "FlowTTE register failure Phase A",
        "config": asdict(config),
    }
    (config.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str]) -> int:
    config = parse_args(argv)
    torch.manual_seed(config.seed)
    config.output_root.mkdir(parents=True, exist_ok=True)
    add_import_paths(config)
    dataset = build_dataset(config.data_root, config.objects)
    support_by_object = load_support_paths(config.support_json)
    backbone = DINOv3Backbone("dinov3_vitl16", device=config.device, smaller_edge_size=672)
    context_rows: List[Dict[str, str]] = []
    retrieval_rows: List[Dict[str, str]] = []
    nf_rows: List[Dict[str, str]] = []
    for object_name in config.objects:
        object_context, object_retrieval, object_nf = analyze_object(
            config,
            dataset,
            backbone,
            support_by_object[object_name],
            object_name,
        )
        context_rows.extend(object_context)
        retrieval_rows.extend(object_retrieval)
        nf_rows.extend(object_nf)
    write_tsv(config.output_root / "context_metrics.tsv", context_rows)
    write_tsv(config.output_root / "retrieval_metrics.tsv", retrieval_rows)
    write_tsv(config.output_root / "nf_distortion_metrics.tsv", nf_rows)
    write_manifest(config)
    print(json.dumps({"output_root": str(config.output_root), "objects": list(config.objects)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
