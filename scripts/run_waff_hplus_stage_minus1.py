#!/usr/bin/env python3
"""Run the WAFF-H+ L1 smoke on exact train/good support only.

This runner has no dataset object and no code path that enumerates test_public
or test_private. It loads only paths sealed by the SuperAD-16 support manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from dinov3_backbone import DINOv3Backbone  # noqa: E402
from flow_tte_mvtec_ad2_core import (  # noqa: E402
    FeatureExtractionConfig,
    extract_feature_map_from_rgb,
)
from run_cfntr_stage_minus1 import (  # noqa: E402
    AXES,
    RESOLUTIONS,
    deterministic_sign,
    load_support_manifest,
    make_synthetic_defect,
    mask_to_patch_grid,
    read_rgb,
)

from flow_tte.cfntr import apply_rgb_transform, make_axis_transform  # noqa: E402
from flow_tte.config import ExpansionConfig, FlowConfig, FlowTTEConfig, ScoreConfig  # noqa: E402
from flow_tte.denoising import PositionMeanArtifactDenoiser  # noqa: E402
from flow_tte.pipeline import FlowTTE  # noqa: E402
from flow_tte.waff import (  # noqa: E402
    WaffConfig,
    WaffFactorialCell,
    audit_weight_simplex_feasibility,
    score_waff_components,
    train_waff_scorer,
)

PROTOCOL_ID = "waff-hplus-stage-minus1-support-only-20260719-l1"
EXPECTED_SUPPORT_SHA256 = "9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5"
FEATURE_LAYERS = (7, 15, 23, 31)
DEFECT_FAMILIES = ("cutpaste", "chromatic_blotch", "gaussian_blur")
OBJECTS = ("can", "rice")
DEFECT_RETRIES = 32


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--support-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--objects", default="can,rice")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host-gpu-id", required=True)
    parser.add_argument("--container-cuda-slot", default="0")
    parser.add_argument("--experiment-seed", type=int, default=0)
    parser.add_argument("--stimulus-source-index", type=int, default=0)
    parser.add_argument("--waff-steps", type=int, default=300)
    args = parser.parse_args(list(argv))
    args.objects = tuple(part for part in args.objects.replace(",", " ").split() if part)
    if not args.objects or any(item not in OBJECTS for item in args.objects):
        parser.error(f"--objects must be a non-empty subset of {OBJECTS}")
    if len(set(args.objects)) != len(args.objects):
        parser.error("--objects must be unique")
    if not 0 <= args.stimulus_source_index < 16:
        parser.error("--stimulus-source-index must be in [0, 15]")
    if args.waff_steps <= 0:
        parser.error("--waff-steps must be positive")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract(
    backbone: DINOv3Backbone,
    image: np.ndarray,
    denoiser: PositionMeanArtifactDenoiser | None,
) -> np.ndarray:
    feature_map = extract_feature_map_from_rgb(
        backbone,
        image,
        FeatureExtractionConfig(feature_fusion="layer_norm_mean", context_source="none"),
    )
    values = feature_map.values
    if denoiser is not None:
        values = denoiser.transform(values)
    return np.asarray(values, dtype=np.float32)


def _pipeline(seed: int, device: str) -> FlowTTE:
    return FlowTTE(
        FlowTTEConfig(
            flow=FlowConfig(
                n_coupling_layers=2,
                hidden_multiplier=1,
                n_epochs=3,
                lr=2e-4,
                clamp=1.9,
                batch_size=512,
                seed=seed,
                tail_weight=0.3,
                tail_top_k_ratio=0.05,
                lambda_logdet=1e-3,
            ),
            expansion=ExpansionConfig(budget=1.0, density_quantile=0.9, random_seed=seed),
            score=ScoreConfig(
                distance_weight=1.0,
                density_weight=0.25,
                top_percent=0.01,
                query_chunk_size=512,
                calibration_sample_size=0,
                loo_standardize=True,
            ),
            device=device,
        ),
    )


def _interior_background(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(mask, dtype=np.uint8)
    interior = cv2.erode(value, np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
    background = ~value.astype(bool)
    return interior.reshape(-1), background.reshape(-1)


def _margin(scores: np.ndarray, interior: np.ndarray, background: np.ndarray) -> float:
    background_scores = scores[background]
    k = max(1, int(np.ceil(background_scores.size * 0.05)))
    top_background = np.partition(background_scores, -k)[-k:].mean()
    return float(scores[interior].mean() - top_background)


def _build_defect(
    image: np.ndarray,
    *,
    object_name: str,
    relative_path: str,
    family: str,
    grid: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    for retry in range(DEFECT_RETRIES):
        try:
            defect, native_mask, metadata = make_synthetic_defect(
                image,
                object_name=object_name,
                relative_path=relative_path,
                kind=family,
                feature_retry=retry,
            )
        except RuntimeError:
            continue
        patch_mask = np.asarray(mask_to_patch_grid(native_mask, grid), dtype=bool).reshape(grid)
        interior, background = _interior_background(patch_mask)
        if bool(interior.any()) and bool(background.any()):
            return defect, interior, background, metadata
    message = f"{object_name}/{family}: no deterministic interior-capable defect"
    raise RuntimeError(message)


def _run_object(  # noqa: PLR0915
    args: argparse.Namespace,
    backbone: DINOv3Backbone,
    entries: Sequence[Any],
    object_name: str,
) -> dict[str, Any]:
    backbone.set_resolution(RESOLUTIONS[object_name])
    images = [read_rgb(entry.path) for entry in entries]
    raw_clean = [_extract(backbone, image, None) for image in images]
    denoiser = PositionMeanArtifactDenoiser.fit(raw_clean, alpha=1.0)
    clean_maps = [denoiser.transform(feature_map) for feature_map in raw_clean]
    grid = (int(clean_maps[0].shape[0]), int(clean_maps[0].shape[1]))
    if any(tuple(feature_map.shape[:2]) != grid for feature_map in clean_maps):
        message = f"{object_name}: clean support grids are not aligned"
        raise RuntimeError(message)
    flat_clean = np.concatenate(
        [feature_map.reshape(-1, feature_map.shape[-1]) for feature_map in clean_maps],
        axis=0,
    ).astype(np.float32, copy=False)
    patches_per_source = int(np.prod(grid))
    bank_source_ids = np.repeat(np.arange(len(entries), dtype=np.int64), patches_per_source)
    pipeline = _pipeline(args.experiment_seed, args.device)
    stats = pipeline.fit(flat_clean)
    memory_before = pipeline.memory.bank.features.detach().clone()  # type: ignore[union-attr]

    source_index = int(args.stimulus_source_index)
    entry = entries[source_index]
    image = images[source_index]
    query_source_ids = np.full(patches_per_source, source_index, dtype=np.int64)

    def source_excluded_components(feature_map: np.ndarray) -> np.ndarray:
        return pipeline.factorial_score_components(
            feature_map,
            query_source_ids=query_source_ids,
            bank_source_ids=bank_source_ids,
        )

    clean_components = source_excluded_components(clean_maps[source_index])
    shifted_clean: dict[str, np.ndarray] = {}
    shifted_images: dict[str, np.ndarray] = {}
    axis_specs: dict[str, Any] = {}
    for axis in AXES:
        sign = deterministic_sign(object_name, entry.relative_path, axis, PROTOCOL_ID)
        spec = make_axis_transform(axis, sign=sign, amplitude_scale=2.0)
        shifted_image = apply_rgb_transform(image, spec)
        shifted_images[axis] = shifted_image
        axis_specs[axis] = spec
        shifted_clean[axis] = source_excluded_components(
            _extract(backbone, shifted_image, denoiser),
        )

    cells: list[WaffFactorialCell] = []
    defect_audit: list[dict[str, Any]] = []
    for family in DEFECT_FAMILIES:
        defect_image, interior, background, metadata = _build_defect(
            image,
            object_name=object_name,
            relative_path=entry.relative_path,
            family=family,
            grid=grid,
        )
        defect_components = source_excluded_components(_extract(backbone, defect_image, denoiser))
        for axis in AXES:
            shifted_defect_image = apply_rgb_transform(defect_image, axis_specs[axis])
            shifted_defect = source_excluded_components(
                _extract(backbone, shifted_defect_image, denoiser),
            )
            cells.append(
                WaffFactorialCell(
                    clean=clean_components,
                    shifted_clean=shifted_clean[axis],
                    defect=defect_components,
                    shifted_defect=shifted_defect,
                    defect_mask=interior,
                    background_mask=background,
                    axis=axis,
                    family=family,
                ),
            )
        defect_audit.append(
            {
                "family": family,
                "interior_patches": int(interior.sum()),
                "background_patches": int(background.sum()),
                "generator": metadata,
            },
        )

    scorer = train_waff_scorer(
        cells,
        config=WaffConfig(steps=args.waff_steps),
        initial_weight=(1.0, 0.25),
        device=args.device,
    )
    feasibility = audit_weight_simplex_feasibility(
        cells,
        config=WaffConfig(steps=args.waff_steps),
        grid_points=101,
        device=args.device,
    )
    default_weight = np.asarray([1.0, 0.25], dtype=np.float32)
    default_margins: dict[str, list[float]] = {axis: [] for axis in AXES}
    active_margins: dict[str, list[float]] = {axis: [] for axis in AXES}
    default_scores: list[np.ndarray] = []
    active_scores: list[np.ndarray] = []
    for cell in cells:
        default_defect = cell.defect @ default_weight
        default_shifted = cell.shifted_defect @ default_weight
        active_defect = score_waff_components(scorer, cell.defect)
        active_shifted = score_waff_components(scorer, cell.shifted_defect)
        default_margins[cell.axis].append(
            _margin(default_shifted, cell.defect_mask, cell.background_mask)
            - _margin(default_defect, cell.defect_mask, cell.background_mask),
        )
        active_margins[cell.axis].append(
            _margin(active_shifted, cell.defect_mask, cell.background_mask)
            - _margin(active_defect, cell.defect_mask, cell.background_mask),
        )
        default_scores.extend((default_defect, default_shifted))
        active_scores.extend((active_defect, active_shifted))

    pipeline.set_waff_scorer(scorer)
    integration_map = _extract(backbone, defect_image, denoiser)
    inference_components = pipeline.factorial_score_components(integration_map)
    integrated = pipeline.score_static(integration_map)
    expected = score_waff_components(scorer, inference_components)
    integration_error = float(np.max(np.abs(integrated.patch_scores.reshape(-1) - expected)))
    memory_unchanged = bool(
        torch.equal(pipeline.memory.bank.features, memory_before),  # type: ignore[union-attr]
    )
    default_flat = np.concatenate(default_scores)
    active_flat = np.concatenate(active_scores)
    noncollapse = bool(
        np.all(np.isfinite(active_flat))
        and float(np.std(active_flat)) > 1e-6
        and abs(float(scorer.weight.sum()) - 1.25) <= 1e-6
        and not np.array_equal(active_flat, default_flat)
        and integration_error <= 1e-6
        and memory_unchanged,
    )
    return {
        "object": object_name,
        "support_count": len(entries),
        "support_paths": [entry.relative_path for entry in entries],
        "stimulus_source": entry.relative_path,
        "grid": list(grid),
        "feature_dim": int(clean_maps[0].shape[-1]),
        "flow_stats": asdict(stats),
        "waff_scorer": {
            **asdict(scorer),
            "weight": scorer.weight.tolist(),
        },
        "simplex_feasibility": {
            **asdict(feasibility),
            "best_joint_weight": feasibility.best_joint_weight.tolist(),
        },
        "factorial_cell_count": len(cells),
        "defect_audit": defect_audit,
        "default_worst_axis_margin_drop": float(
            max(-np.mean(values) for values in default_margins.values()),
        ),
        "waff_worst_axis_margin_drop": float(
            max(-np.mean(values) for values in active_margins.values()),
        ),
        "active_score_std": float(np.std(active_flat)),
        "active_score_changed_fraction": float(np.mean(active_flat != default_flat)),
        "integration_max_abs_error": integration_error,
        "static_memory_unchanged": memory_unchanged,
        "noncollapse": noncollapse,
    }


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    support_json = Path(args.support_json).resolve(strict=True)
    if _sha256(support_json) != EXPECTED_SUPPORT_SHA256:
        raise RuntimeError("support manifest checksum does not match frozen SuperAD-16")
    manifests = load_support_manifest(support_json, Path(args.data_root))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    backbone = DINOv3Backbone(
        model_name="dinov3_vith16plus",
        device=args.device,
        smaller_edge_size=672,
        feature_layers=FEATURE_LAYERS,
    )
    results = [
        _run_object(args, backbone, manifests[object_name], object_name)
        for object_name in args.objects
    ]
    payload = {
        "protocol_id": PROTOCOL_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "L1_support_only_runtime_noncollapse_smoke",
        "target": "MVTec AD2 single-image",
        "population": "exact SuperAD-16 train/good only",
        "objects": list(args.objects),
        "seed": args.experiment_seed,
        "host_gpu_id": args.host_gpu_id,
        "container_cuda_slot": args.container_cuda_slot,
        "backbone": "dinov3_vith16plus",
        "feature_layers": list(FEATURE_LAYERS),
        "feature_fusion": "layer_norm_mean",
        "dvt_denoise": "position_mean_alpha1",
        "flow": "default_hplus_e3_c2_h1_static",
        "defect_families": list(DEFECT_FAMILIES),
        "forbidden_family_absent": "high_frequency_ribbon" not in DEFECT_FAMILIES,
        "query_or_test_access": False,
        "test_access_attempts": 0,
        "public_metrics_computed": False,
        "results": results,
        "gate_pass": bool(all(result["noncollapse"] for result in results)),
        "claim_authorized": False,
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if payload["gate_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
