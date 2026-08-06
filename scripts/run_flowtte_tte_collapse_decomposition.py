#!/usr/bin/env python3
# ruff: noqa: ANN401, E501, EM102, EXE001, PLC0415, PLR0913, RUF046
"""Decompose the historical `flow_tte_budget125` test-time-expansion collapse.

The 2026-07-07 component ablation recorded `-0.027955` pAUROC / `-0.017479` F1
for `expansion_budget=1.25` and closed the test-time branch.  This runner keeps
Baseline1 frozen and varies only the memory-expansion protocol so the single
pooled number can be split into contamination, calibration, non-stationarity,
and shift-adaptation terms.

Access contract: TESTpublic images, TESTpublic ground truth (diagnostic arms
only), no TESTprivate, no threshold export, no method promotion.  Metrics are
patch-grid mechanism metrics and are not comparable with the frozen contract.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "src", Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_flowtte_paired_loss_diagnostic as base  # noqa: E402

from flow_tte.config import ExpansionConfig, FlowConfig, ScoreConfig  # noqa: E402
from flow_tte.scoring import ScoreCalibration  # noqa: E402
from flow_tte.trainer import FlowDensityEstimator  # noqa: E402
from flow_tte.tte_collapse import ArmSpec, ImageRecord, run_arm, stream_order  # noqa: E402
from flow_tte.tte_collapse_metrics import compute_arm_metrics  # noqa: E402

PROTOCOL_ID = "flowtte-tte-collapse-decomposition-20260805"
DEFAULT_OBJECTS = ("fabric", "wallplugs", "vial", "can")

ARMS: tuple[ArmSpec, ...] = (
    ArmSpec("A0_static", 1.0, protocol="static", note="Baseline1 frozen bank"),
    ArmSpec("A1_replicate", 1.25, note="historical budget 1.25 reservoir expansion"),
    ArmSpec("A2_oracle_clean", 1.25, candidate_policy="gt_normal", note="M1 removed"),
    ArmSpec("A3_good_images", 1.25, candidate_policy="good_images", note="M1 removed at image level"),
    ArmSpec("A4_two_pass", 1.25, protocol="two_pass", note="M3 removed"),
    ArmSpec("A5_two_pass_recal", 1.25, protocol="two_pass", recalibrate=True, note="M2+M3 removed"),
    ArmSpec("A6_two_pass_clean", 1.25, candidate_policy="gt_normal", protocol="two_pass", note="M1+M3 removed"),
    ArmSpec("A7_ceiling", 1.25, candidate_policy="gt_normal", protocol="two_pass", recalibrate=True, note="M4 benefit ceiling"),
    ArmSpec("A8_cap105", 1.05, protocol="two_pass", note="capacity dose 0.05"),
    ArmSpec("A9_cap200", 2.00, protocol="two_pass", note="capacity dose 1.00"),
    ArmSpec("A10_order_s101", 1.25, order_seed=101, note="M3 order variance"),
    ArmSpec("A11_order_s202", 1.25, order_seed=202, note="M3 order variance"),
    ArmSpec("A12_order_s303", 1.25, order_seed=303, note="M3 order variance"),
)


class RunnerError(RuntimeError):
    """Raised when the frozen diagnostic contract cannot be satisfied."""


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--support-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--objects", default=",".join(DEFAULT_OBJECTS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=0, help="0 means the full split")
    parser.add_argument("--backbone-model", default="dinov3_vith16plus")
    parser.add_argument("--feature-layers", default="", help="comma list; empty keeps Baseline1")
    parser.add_argument("--lambda-logdet", type=float, default=-1.0, help="<0 keeps Baseline1")
    parser.add_argument("--density-weight", type=float, default=0.0)
    parser.add_argument("--arms", default="", help="comma list of arm ids; empty runs all")
    parser.add_argument("--config-label", default="baseline1_core")
    parser.add_argument("--dvt-mode", choices=("position_mean", "none"), default="position_mean")
    args = parser.parse_args(list(argv))
    args.objects = tuple(part for part in args.objects.replace(",", " ").split() if part)
    if not args.objects:
        parser.error("--objects must name at least one object")
    if any(name not in base.OBJECTS for name in args.objects):
        parser.error(f"--objects must be a subset of {base.OBJECTS}")
    args.feature_layers = tuple(
        int(part) for part in args.feature_layers.replace(",", " ").split()
    ) or tuple(base.LAYERS)
    if len(args.feature_layers) != len(base.LAYERS):
        parser.error(f"--feature-layers must name exactly {len(base.LAYERS)} layers")
    args.arms = tuple(part for part in args.arms.replace(",", " ").split())
    known = {spec.arm_id for spec in ARMS}
    if any(name not in known for name in args.arms):
        parser.error(f"--arms must be a subset of {sorted(known)}")
    if args.density_weight < 0.0:
        parser.error("--density-weight must be non-negative")
    return args


def selected_arms(args: argparse.Namespace) -> tuple[ArmSpec, ...]:
    if not args.arms:
        return ARMS
    chosen = set(args.arms) | {"A0_static"}
    return tuple(spec for spec in ARMS if spec.arm_id in chosen)


def make_backbone(args: argparse.Namespace) -> Any:
    from dinov3_backbone import DINOv3Backbone

    return DINOv3Backbone(
        model_name=args.backbone_model,
        device=args.device,
        smaller_edge_size=672,
        feature_layers=args.feature_layers,
    )


def flow_overrides(args: argparse.Namespace, stage_a_flow: dict[str, Any]) -> dict[str, Any]:
    if args.lambda_logdet < 0.0:
        return stage_a_flow
    return {**stage_a_flow, "lambda_logdet": args.lambda_logdet}


def frozen_config(args: argparse.Namespace) -> dict[str, Any]:
    device = args.device
    stage_a = base.frozen_config(device)
    return {
        "protocol_id": PROTOCOL_ID,
        "baseline_method": "Baseline1-core",
        "historical_reference": {
            "run": "flowtte_component_ablation_all8_20260707_v1",
            "arm": "flow_tte_budget125",
            "delta_auroc": -0.027955,
            "delta_f1": -0.017479,
        },
        "device": device,
        "config_label": args.config_label,
        "backbone": args.backbone_model,
        "layers": list(args.feature_layers),
        "feature_fusion": stage_a["feature_fusion"],
        "native_size": stage_a["native_size"],
        "dvt": stage_a["dvt"] if args.dvt_mode == "position_mean" else {"mode": "none"},
        "stage_a_flow": flow_overrides(args, stage_a["stage_a_flow"]),
        "density_weight": args.density_weight,
        "score": {"latent_metric": "raw_euclidean", "nn": "1NN", "standardization": "support_LOO"},
        "arms": [asdict(spec) for spec in ARMS],
        "metric_grid": "dinov3_patch_grid",
        "metric_contract_comparable": False,
        "test_public_access": True,
        "test_public_gt_access": True,
        "test_private_access": False,
        "threshold_export": False,
        "method_promotion": False,
    }


def condition_of(path: Path) -> str:
    stem = path.stem
    _, _, suffix = stem.partition("_")
    return suffix or "regular"


def list_test_items(data_root: Path, object_name: str, max_images: int) -> tuple[tuple[Path, bool], ...]:
    root = data_root / object_name / "test_public"
    good = sorted((root / "good").glob("*.png"))
    bad = sorted((root / "bad").glob("*.png"))
    if not good or not bad:
        raise RunnerError(f"{object_name}: empty TESTpublic split")
    items = [(path, False) for path in bad] + [(path, True) for path in good]
    items.sort(key=lambda item: (item[0].name, "good" if item[1] else "bad"))
    if max_images > 0:
        items = items[:max_images]
    return tuple(items)


def patch_defect_labels(mask_path: Path, grid: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    if not mask_path.exists():
        raise RunnerError(f"missing ground-truth mask: {mask_path}")
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
    height, width = grid
    rows = np.array_split(np.arange(mask.shape[0]), height)
    columns = np.array_split(np.arange(mask.shape[1]), width)
    labels = np.zeros((height, width), dtype=bool)
    for row_index, row_slice in enumerate(rows):
        band = mask[row_slice, :]
        for column_index, column_slice in enumerate(columns):
            labels[row_index, column_index] = bool(band[:, column_slice].any())
    return labels.reshape(-1)


def mask_path_for(data_root: Path, object_name: str, image_path: Path) -> Path:
    return data_root / object_name / "test_public" / "ground_truth" / "bad" / f"{image_path.stem}_mask.png"


def fit_support(
    backbone: Any,
    entries: Sequence[Any],
    config: dict[str, Any],
    device: str,
    dvt_mode: str = "position_mean",
) -> tuple[FlowDensityEstimator, np.ndarray | None, torch.Tensor, float]:
    raw = []
    for index, entry in enumerate(entries):
        image = base.read_rgb(entry.path)
        brightness = base.deterministic_support_brightness(index, seed=0)
        adjusted = base.apply_photometric_transform(
            image,
            base.PhotometricTransformSpec(
                name="support_brightness",
                active_axes=("brightness",),
                brightness=brightness,
            ),
        )
        raw.append(base.extract_feature_map(backbone, adjusted, entry.relative_path, "support_brightness_identity"))
    if dvt_mode == "position_mean":
        support_maps, artifact = base.fit_dvt(tuple(raw))
    else:
        support_maps, artifact = tuple(raw), None
    estimator = FlowDensityEstimator(
        dim=int(support_maps[0].values.shape[-1]),
        config=FlowConfig(**dict(config["stage_a_flow"])),
        device=device,
    )
    stats = estimator.fit(
        np.stack([item.values for item in support_maps], axis=0),
        density_quantile=float(ExpansionConfig().density_quantile),
    )
    latents = estimator.transform(np.stack([item.values for item in support_maps], axis=0))
    return estimator, artifact, latents.detach(), float(stats.density_threshold)


def build_records(
    backbone: Any,
    estimator: FlowDensityEstimator,
    artifact: np.ndarray | None,
    data_root: Path,
    object_name: str,
    items: Sequence[tuple[Path, bool]],
) -> tuple[ImageRecord, ...]:
    records: list[ImageRecord] = []
    for path, is_good in items:
        image = base.read_rgb(path)
        feature_map = base.extract_feature_map(backbone, image, str(path), "identity")
        values = feature_map.values
        if artifact is not None:
            values = (values - artifact).astype(np.float32, copy=False)
        evaluation = estimator.evaluate(values[np.newaxis, ...])
        grid = feature_map.grid
        if is_good:
            defect = np.zeros(grid[0] * grid[1], dtype=bool)
        else:
            defect = patch_defect_labels(mask_path_for(data_root, object_name, path), grid)
        records.append(
            ImageRecord(
                relative_path=str(path.relative_to(data_root)),
                condition=condition_of(path),
                is_good=bool(is_good),
                latents=evaluation.z.detach(),
                nll=evaluation.nll.detach(),
                defect=defect,
                density_penalty=estimator.density_penalty(evaluation.nll).detach(),
            ),
        )
    return tuple(records)


def object_phase(
    args: argparse.Namespace,
    backbone: Any,
    config: dict[str, Any],
    object_name: str,
    entries: Sequence[Any],
) -> dict[str, Any]:
    started = time.time()
    data_root = Path(args.data_root)
    estimator, artifact, support_latents, density_threshold = fit_support(
        backbone, entries, config, args.device, args.dvt_mode,
    )
    items = list_test_items(data_root, object_name, args.max_images)
    records = build_records(backbone, estimator, artifact, data_root, object_name, items)
    score_config = ScoreConfig(
        density_weight=float(args.density_weight),
        query_chunk_size=4096,
        calibration_sample_size=4096,
    )
    calibration = ScoreCalibration.fit(support_latents, score_config, None)

    defects = [record.defect for record in records]
    conditions = [record.condition for record in records]
    is_good = [record.is_good for record in records]
    rows: list[dict[str, Any]] = []
    for spec in selected_arms(args):
        result = run_arm(
            spec,
            support_latents,
            records,
            density_threshold,
            calibration,
            score_config,
            density_weight=float(args.density_weight),
        )
        metrics = compute_arm_metrics(
            result.scores,
            defects,
            conditions,
            is_good,
            stream_order(len(records), spec.order_seed),
        )
        capacity = int(round((spec.budget - 1.0) * int(support_latents.shape[0])))
        rows.append(
            {
                "arm_id": spec.arm_id,
                "note": spec.note,
                "uses_labels": spec.uses_labels,
                "budget": spec.budget,
                "protocol": spec.protocol,
                "candidate_policy": spec.candidate_policy,
                "recalibrate": spec.recalibrate,
                "order_seed": spec.order_seed,
                "capacity_rows": capacity,
                "absorbed_rows": result.absorbed_rows,
                "contaminated_rows": result.contaminated_rows,
                "contamination_fraction": (result.contaminated_rows / result.absorbed_rows) if result.absorbed_rows else 0.0,
                "self_hit_rate": result.self_hit_rate,
                "calibration_mean": result.calibration_mean,
                "calibration_std": result.calibration_std,
                **metrics.as_dict(),
            },
        )
        print(f"[{object_name}] {spec.arm_id} done", flush=True)

    reference = rows[0]
    for row in rows:
        row["delta_patch_pauroc_005"] = row["patch_pauroc_005"] - reference["patch_pauroc_005"]
        row["delta_patch_best_f1"] = row["patch_best_f1"] - reference["patch_best_f1"]
    return {
        "object": object_name,
        "support_rows": int(support_latents.shape[0]),
        "patch_grid": list(records[0].latents.shape),
        "n_images": len(records),
        "n_bad": int(sum(1 for value in is_good if not value)),
        "n_good": int(sum(1 for value in is_good if value)),
        "density_threshold": density_threshold,
        "elapsed_seconds": time.time() - started,
        "arms": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    support_path = Path(args.support_json)
    support_sha = base.sha256_file(support_path)
    if support_sha != base.FROZEN_SUPPORT_MANIFEST_SHA256:
        raise RunnerError(f"support manifest SHA mismatch: {support_sha}")
    support = base.load_support_manifest(support_path, Path(args.data_root))
    config = frozen_config(args)
    backbone = make_backbone(args)
    summaries = []
    for object_name in args.objects:
        summary = object_phase(args, backbone, config, object_name, support[object_name])
        (output_root / f"{object_name}.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
        )
        summaries.append(summary)
    payload = {
        "config": config,
        "support_manifest_sha256": support_sha,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "objects": summaries,
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8",
    )
    print(f"wrote {output_root / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
