#!/usr/bin/env python3
# ruff: noqa: C901, PERF203, PLR0912, PLR0913, PLR0915, TRY300
"""Run the preregistered FLRS support-only Stage -1 assay.

The runner is structurally unable to enumerate MVTec AD2 test splits.  It
loads only the exact support-16 manifest and writes compact aggregate JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_functional
from sklearn.metrics import precision_recall_curve, roc_auc_score

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "src", Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_cfntr_stage_minus1 import (  # noqa: E402
    DEFECT_KINDS,
    LAYERS,
    OBJECTS,
    RESOLUTIONS,
    SupportEntry,
    assigned_defect_shift,
    extract_layers,
    load_support_manifest,
    make_synthetic_defect,
    mask_to_patch_grid,
    read_rgb,
    sha256_file,
    validation_specs,
)
from run_cfntr_stage_minus1 import (  # noqa: E402
    stable_seed as stimulus_seed,
)

from flow_tte.cfntr import apply_rgb_transform  # noqa: E402
from flow_tte.flrs import (  # noqa: E402
    RADIUS1_OFFSETS,
    RADIUS2_OFFSETS,
    RANDOM_NEIGHBOR_OFFSETS,
    RobustEnvelope,
    combine_layer_scores,
    invert_permuted_map,
    permute_feature_grid,
)
from fmad.backbones.dinov2 import DINOv2Backbone  # noqa: E402

PROTOCOL_ID = "flrs-stage-minus1-support-only-20260718"
VARIANTS = (
    "flrs",
    "raw_state",
    "random_neighbor",
    "spatial_permutation",
    "rgb_gradient",
    "radius1_only",
)
SCALE_FLOOR = 1e-3
EXPECTED_SUPPORT_SHA256 = "9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5"
DEFECT_GENERATION_RETRIES = 32


def _seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in (PROTOCOL_ID, *parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _split_words(value: str) -> tuple[str, ...]:
    return tuple(part for part in value.replace(",", " ").split() if part)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--support-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone-model", default="dinov2_vitl14")
    parser.add_argument("--host-gpu-id", required=True)
    parser.add_argument("--container-cuda-slot", default="0")
    args = parser.parse_args(list(argv))
    args.objects = _split_words(args.objects)
    if not args.objects or len(set(args.objects)) != len(args.objects):
        parser.error("--objects must contain unique object names")
    if any(name not in OBJECTS for name in args.objects):
        parser.error(f"--objects must be a subset of {OBJECTS}")
    if args.backbone_model != "dinov2_vitl14":
        parser.error("Stage -1 freezes --backbone-model=dinov2_vitl14")
    return args


def _feature_grid(features: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    return np.asarray(features, dtype=np.float32).reshape(grid[0], grid[1], -1)


def _rgb_gradient(image: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(dx, dy)
    resized = cv2.resize(magnitude, (grid[1], grid[0]), interpolation=cv2.INTER_AREA)
    return np.asarray(resized[..., None], dtype=np.float32)


def _torch_offset_cosines(
    normalized: torch.Tensor,
    offsets: Sequence[tuple[int, int]],
) -> torch.Tensor:
    max_y = max(abs(offset[0]) for offset in offsets)
    max_x = max(abs(offset[1]) for offset in offsets)
    channels_first = normalized.permute(2, 0, 1).unsqueeze(0)
    padded = (
        torch_functional.pad(
            channels_first,
            (max_x, max_x, max_y, max_y),
            mode="replicate",
        )
        .squeeze(0)
        .permute(1, 2, 0)
    )
    height, width = normalized.shape[:2]
    shifted = torch.stack(
        [
            padded[
                max_y + y_offset : max_y + y_offset + height,
                max_x + x_offset : max_x + x_offset + width,
            ]
            for y_offset, x_offset in offsets
        ],
        dim=-2,
    )
    return torch.clamp(
        torch.sum(normalized.unsqueeze(-2) * shifted, dim=-1),
        -1.0,
        1.0,
    )


def _relation_signature_torch(
    features: np.ndarray,
    *,
    device: str,
    kind: str,
) -> np.ndarray:
    value = torch.as_tensor(features, dtype=torch.float32, device=device)
    value = torch_functional.normalize(value, p=2.0, dim=-1)
    if kind == "random_neighbor":
        distant = _torch_offset_cosines(value, RANDOM_NEIGHBOR_OFFSETS)
        first, second = distant[..., :4], distant[..., 4:]
        signature = torch.stack(
            (
                first.mean(dim=-1),
                first.std(dim=-1, correction=0),
                second.mean(dim=-1),
                torch.abs(first.mean(dim=-1) - second.mean(dim=-1)),
            ),
            dim=-1,
        )
    else:
        radius1 = _torch_offset_cosines(value, RADIUS1_OFFSETS)
        horizontal = _torch_offset_cosines(value, ((0, -1), (0, 1))).mean(dim=-1)
        vertical = _torch_offset_cosines(value, ((-1, 0), (1, 0))).mean(dim=-1)
        if kind == "radius1_only":
            signature = torch.stack(
                (
                    radius1.mean(dim=-1),
                    radius1.std(dim=-1, correction=0),
                    torch.abs(horizontal - vertical),
                ),
                dim=-1,
            )
        elif kind == "flrs":
            radius2 = _torch_offset_cosines(value, RADIUS2_OFFSETS)
            signature = torch.stack(
                (
                    radius1.mean(dim=-1),
                    radius1.std(dim=-1, correction=0),
                    radius2.mean(dim=-1),
                    torch.abs(horizontal - vertical),
                ),
                dim=-1,
            )
        else:
            raise ValueError("unknown relation signature kind")
    return signature.detach().cpu().numpy().astype(np.float32, copy=False)


def _fit_envelope_torch(values: Sequence[np.ndarray], *, device: str) -> RobustEnvelope:
    matrix = torch.cat(
        [torch.as_tensor(value.reshape(-1, value.shape[-1]), device=device) for value in values],
        dim=0,
    ).to(dtype=torch.float32)
    center = torch.quantile(matrix, 0.5, dim=0, interpolation="linear")
    mad = torch.quantile(torch.abs(matrix - center), 0.5, dim=0, interpolation="linear")
    scale = torch.clamp(1.4826 * mad, min=SCALE_FLOOR)
    return RobustEnvelope(
        center=center.detach().cpu().numpy().astype(np.float32, copy=False),
        scale=scale.detach().cpu().numpy().astype(np.float32, copy=False),
    )


def _score_envelope_torch(
    values: np.ndarray,
    envelope: RobustEnvelope,
    *,
    device: str,
) -> np.ndarray:
    value = torch.as_tensor(values, dtype=torch.float32, device=device)
    center = torch.as_tensor(envelope.center, device=device)
    scale = torch.as_tensor(envelope.scale, device=device)
    scores = torch.mean(torch.square((value - center) / scale), dim=-1)
    return scores.detach().cpu().numpy().astype(np.float32, copy=False)


def _precompute_clean_signatures(
    entries: Sequence[SupportEntry],
    images: Mapping[str, np.ndarray],
    features: Mapping[tuple[str, int], np.ndarray],
    grids: Mapping[str, tuple[int, int]],
    *,
    device: str,
) -> dict[tuple[str, int | str, str], np.ndarray]:
    result: dict[tuple[str, int | str, str], np.ndarray] = {}
    for entry in entries:
        path = entry.relative_path
        result[(path, "rgb", "rgb_gradient")] = _rgb_gradient(images[path], grids[path])
        for layer in LAYERS:
            value = _feature_grid(features[(path, layer)], grids[path])
            result[(path, layer, "raw_state")] = value
            for variant in ("flrs", "radius1_only", "random_neighbor"):
                result[(path, layer, variant)] = _relation_signature_torch(
                    value,
                    device=device,
                    kind=variant,
                )
            shuffled, _ = permute_feature_grid(
                value,
                seed=_seed(path, layer, "spatial-permutation"),
            )
            result[(path, layer, "spatial_permutation")] = _relation_signature_torch(
                shuffled,
                device=device,
                kind="flrs",
            )
    return result


def _fit_fold(
    entries: Sequence[SupportEntry],
    fold: int,
    clean_signatures: Mapping[tuple[str, int | str, str], np.ndarray],
    *,
    device: str,
) -> dict[str, Any]:
    train = [entry for entry in entries if entry.fold != fold]
    models: dict[str, Any] = {variant: {} for variant in VARIANTS if variant != "rgb_gradient"}
    for layer in LAYERS:
        for variant in (
            "flrs",
            "radius1_only",
            "random_neighbor",
            "spatial_permutation",
            "raw_state",
        ):
            models[variant][layer] = _fit_envelope_torch(
                [clean_signatures[(entry.relative_path, layer, variant)] for entry in train],
                device=device,
            )
    models["rgb_gradient"] = _fit_envelope_torch(
        [clean_signatures[(entry.relative_path, "rgb", "rgb_gradient")] for entry in train],
        device=device,
    )
    return models


def _score_variants(
    image: np.ndarray,
    layer_features: Mapping[int, np.ndarray],
    grid: tuple[int, int],
    source_path: str,
    models: Mapping[str, Any],
    *,
    device: str,
) -> dict[str, np.ndarray]:
    layer_maps: dict[str, list[np.ndarray]] = defaultdict(list)
    for layer in LAYERS:
        value = _feature_grid(layer_features[layer], grid)
        flrs_signature = _relation_signature_torch(value, device=device, kind="flrs")
        layer_maps["flrs"].append(
            _score_envelope_torch(flrs_signature, models["flrs"][layer], device=device),
        )
        layer_maps["radius1_only"].append(
            _score_envelope_torch(
                flrs_signature[..., (0, 1, 3)],
                models["radius1_only"][layer],
                device=device,
            ),
        )
        layer_maps["random_neighbor"].append(
            _score_envelope_torch(
                _relation_signature_torch(value, device=device, kind="random_neighbor"),
                models["random_neighbor"][layer],
                device=device,
            ),
        )
        layer_maps["raw_state"].append(
            _score_envelope_torch(value, models["raw_state"][layer], device=device),
        )
        shuffled, inverse = permute_feature_grid(
            value,
            seed=_seed(source_path, layer, "spatial-permutation"),
        )
        permuted_score = _score_envelope_torch(
            _relation_signature_torch(shuffled, device=device, kind="flrs"),
            models["spatial_permutation"][layer],
            device=device,
        )
        layer_maps["spatial_permutation"].append(
            invert_permuted_map(permuted_score, inverse),
        )
    scores = {name: combine_layer_scores(values) for name, values in layer_maps.items()}
    scores["rgb_gradient"] = _score_envelope_torch(
        _rgb_gradient(image, grid),
        models["rgb_gradient"],
        device=device,
    )
    return scores


def _best_f1(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    index = int(np.argmax(f1))
    threshold = (
        float(thresholds[min(index, max(0, thresholds.size - 1))]) if thresholds.size else 0.0
    )
    return float(f1[index]), threshold


def _pixel_summary(labels: Sequence[np.ndarray], scores: Sequence[np.ndarray]) -> dict[str, float]:
    y = np.concatenate([np.asarray(value, dtype=np.uint8).reshape(-1) for value in labels])
    s = np.concatenate([np.asarray(value, dtype=np.float64).reshape(-1) for value in scores])
    if np.unique(y).size != 2 or not np.all(np.isfinite(s)):
        raise RuntimeError("defect metric population is invalid")
    f1, threshold = _best_f1(y, s)
    return {
        "pAUROC_0.05": float(roc_auc_score(y, s, max_fpr=0.05)),
        "F1": f1,
        "best_threshold": threshold,
        "pixels": int(y.size),
        "positive_pixels": int(y.sum()),
    }


def _coverage_update(target: dict[str, dict[str, int]], scores: Mapping[str, np.ndarray]) -> None:
    for variant, score_map in scores.items():
        target[variant]["total"] += 1
        target[variant]["finite"] += int(bool(np.all(np.isfinite(score_map))))
        target[variant]["nonconstant"] += int(float(np.ptp(score_map)) > 0.0)


def _make_valid_defect(
    image: np.ndarray,
    *,
    object_name: str,
    relative_path: str,
    kind: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], int]:
    last_error: RuntimeError | None = None
    for generation_retry in range(DEFECT_GENERATION_RETRIES):
        try:
            defect_image, mask, metadata = make_synthetic_defect(
                image,
                object_name=object_name,
                relative_path=relative_path,
                kind=kind,
                feature_retry=generation_retry,
            )
            return defect_image, mask, metadata, generation_retry
        except RuntimeError as error:
            last_error = error
    message = f"{object_name}/{relative_path}/{kind}: no acquisition-visible deterministic defect"
    raise RuntimeError(message) from last_error


def _defect_metrics(
    populations: Mapping[tuple[str, str, str], dict[str, list[np.ndarray]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in VARIANTS:
        result[variant] = {}
        for acquisition in ("clean", "shifted"):
            key = (variant, acquisition, "all")
            result[variant][acquisition] = _pixel_summary(
                populations[key]["labels"],
                populations[key]["scores"],
            )
            result[variant][acquisition]["by_family"] = {
                family: _pixel_summary(
                    populations[(variant, acquisition, family)]["labels"],
                    populations[(variant, acquisition, family)]["scores"],
                )
                for family in DEFECT_KINDS
            }
    return result


def run_object(
    args: argparse.Namespace,
    backbone: DINOv2Backbone,
    entries: Sequence[SupportEntry],
    object_name: str,
) -> dict[str, Any]:
    backbone.set_resolution(RESOLUTIONS[object_name])
    images: dict[str, np.ndarray] = {}
    features: dict[tuple[str, int], np.ndarray] = {}
    grids: dict[str, tuple[int, int]] = {}
    for entry in entries:
        image = read_rgb(entry.path)
        extracted, grid = extract_layers(backbone, image)
        images[entry.relative_path] = image
        grids[entry.relative_path] = grid
        for layer in LAYERS:
            features[(entry.relative_path, layer)] = extracted[layer]

    clean_signatures = _precompute_clean_signatures(
        entries,
        images,
        features,
        grids,
        device=args.device,
    )

    coverage = {variant: defaultdict(int) for variant in VARIANTS}
    clean_p95: dict[tuple[str, str], float] = {}
    shift_inflations: dict[str, list[float]] = defaultdict(list)
    shift_pairs: list[dict[str, Any]] = []
    defect_populations: dict[tuple[str, str, str], dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {"labels": [], "scores": []},
    )
    stimulus_audit: list[dict[str, Any]] = []
    for fold in range(4):
        models = _fit_fold(
            entries,
            fold,
            clean_signatures,
            device=args.device,
        )
        for entry in (item for item in entries if item.fold == fold):
            image = images[entry.relative_path]
            grid = grids[entry.relative_path]
            clean_layers = {layer: features[(entry.relative_path, layer)] for layer in LAYERS}
            clean_scores = _score_variants(
                image,
                clean_layers,
                grid,
                entry.relative_path,
                models,
                device=args.device,
            )
            _coverage_update(coverage, clean_scores)
            for variant in VARIANTS:
                clean_p95[(entry.relative_path, variant)] = float(
                    np.percentile(clean_scores[variant], 95),
                )

            validation_cache: dict[str, tuple[np.ndarray, dict[int, np.ndarray]]] = {}
            for transform_name, spec in validation_specs(object_name, entry.relative_path):
                shifted_image = apply_rgb_transform(image, spec)
                shifted_layers, shifted_grid = extract_layers(backbone, shifted_image)
                if shifted_grid != grid:
                    raise RuntimeError("photometric shift changed the feature grid")
                validation_cache[transform_name] = (shifted_image, shifted_layers)
                shifted_scores = _score_variants(
                    shifted_image,
                    shifted_layers,
                    grid,
                    entry.relative_path,
                    models,
                    device=args.device,
                )
                _coverage_update(coverage, shifted_scores)
                for variant in VARIANTS:
                    shifted_p95 = float(np.percentile(shifted_scores[variant], 95))
                    clean_value = clean_p95[(entry.relative_path, variant)]
                    inflation = max(0.0, shifted_p95 - clean_value)
                    shift_inflations[variant].append(inflation)
                    shift_pairs.append(
                        {
                            "source": entry.relative_path,
                            "fold": fold,
                            "transform": transform_name,
                            "variant": variant,
                            "clean_p95": clean_value,
                            "shifted_p95": shifted_p95,
                            "positive_inflation": inflation,
                        },
                    )
                stimulus_audit.append(
                    {"source": entry.relative_path, "kind": "shift", "name": transform_name},
                )

            for defect_kind in DEFECT_KINDS:
                shift_name, shift_spec = assigned_defect_shift(
                    object_name,
                    entry.relative_path,
                    defect_kind,
                )
                defect_image, mask, metadata, generation_retry = _make_valid_defect(
                    image,
                    object_name=object_name,
                    relative_path=entry.relative_path,
                    kind=defect_kind,
                )
                candidates = (
                    ("clean", defect_image),
                    ("shifted", apply_rgb_transform(defect_image, shift_spec)),
                )
                for acquisition, evaluated_image in candidates:
                    defect_layers, defect_grid = extract_layers(backbone, evaluated_image)
                    if defect_grid != grid:
                        raise RuntimeError("synthetic defect changed the feature grid")
                    scores = _score_variants(
                        evaluated_image,
                        defect_layers,
                        grid,
                        entry.relative_path,
                        models,
                        device=args.device,
                    )
                    _coverage_update(coverage, scores)
                    patch_mask = mask_to_patch_grid(mask, grid)
                    for variant in VARIANTS:
                        for family in ("all", defect_kind):
                            population = defect_populations[(variant, acquisition, family)]
                            population["labels"].append(patch_mask)
                            population["scores"].append(scores[variant])
                    stimulus_audit.append(
                        {
                            "source": entry.relative_path,
                            "kind": "defect",
                            "name": defect_kind,
                            "acquisition": acquisition,
                            "assigned_shift": shift_name if acquisition == "shifted" else "none",
                            "seed": metadata["seed"],
                            "generation_retry": generation_retry,
                            "stimulus_protocol_seed": stimulus_seed(
                                object_name,
                                entry.relative_path,
                                defect_kind,
                            ),
                        },
                    )

    coverage_json = {variant: dict(values) for variant, values in coverage.items()}
    expected_maps = len(entries) * (1 + 6 + 2 * len(DEFECT_KINDS))
    for values in coverage_json.values():
        values["expected"] = expected_maps
        values["complete"] = (
            values.get("total") == expected_maps
            and values.get("finite") == expected_maps
            and values.get("nonconstant") == expected_maps
        )
    summary = {
        "schema": "flrs-stage-minus1-object-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "stimulus_protocol_id": "cfntr-stage-minus1-support-only-20260718",
        "evaluation_role": "support_only_synthetic_identifiability_preflight",
        "maximum_verdict": "PROMISING_DIAGNOSTIC",
        "object": object_name,
        "population": "train/good exact support-16 only",
        "test_public_accessed": False,
        "test_private_accessed": False,
        "tta_used": False,
        "access_audit": {
            "test_public_accessed": False,
            "test_private_accessed": False,
            "ground_truth_accessed": False,
            "query_update": False,
            "query_transform": False,
            "tta": False,
            "forbidden_path_hits": [],
        },
        "support_count": len(entries),
        "fold_counts": np.bincount([entry.fold for entry in entries], minlength=4).tolist(),
        "resolution": RESOLUTIONS[object_name],
        "backbone": args.backbone_model,
        "layers": list(LAYERS),
        "scale_floor": SCALE_FLOOR,
        "variants": list(VARIANTS),
        "shift": {
            variant: {
                "positive_p95_inflation_mean": float(np.mean(values)),
                "positive_p95_inflation_median": float(np.median(values)),
                "pairs": len(values),
            }
            for variant, values in shift_inflations.items()
        },
        "shift_pairs": shift_pairs,
        "defect": _defect_metrics(defect_populations),
        "coverage": coverage_json,
        "stimulus_audit": stimulus_audit,
        "support": [{**asdict(entry), "path": str(entry.path)} for entry in entries],
        "host_gpu_id": str(args.host_gpu_id),
        "container_cuda_slot": str(args.container_cuda_slot),
    }
    output = Path(args.output_root) / f"{object_name}_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for remote Stage -1 execution")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    support_path = Path(args.support_json).resolve(strict=True)
    if sha256_file(support_path) != EXPECTED_SUPPORT_SHA256:
        raise RuntimeError("support manifest SHA-256 does not match preregistration")
    support = load_support_manifest(support_path, Path(args.data_root))
    backbone = DINOv2Backbone(
        model_name=args.backbone_model,
        device=args.device,
        smaller_edge_size=672,
        feature_layers=LAYERS,
    )
    summaries = [
        run_object(args, backbone, support[object_name], object_name)
        for object_name in args.objects
    ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "objects": list(args.objects),
        "test_public_accessed": False,
        "test_private_accessed": False,
        "tta_used": False,
        "runner_sha256": sha256_file(Path(__file__)),
        "core_sha256": sha256_file(_ROOT / "src" / "flow_tte" / "flrs.py"),
        "support_manifest_sha256": sha256_file(support_path),
        "summary_sha256": {
            summary["object"]: sha256_file(output_root / f"{summary['object']}_summary.json")
            for summary in summaries
        },
        "host_gpu_id": str(args.host_gpu_id),
        "container_cuda_slot": str(args.container_cuda_slot),
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
