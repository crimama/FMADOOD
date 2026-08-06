#!/usr/bin/env python3
# ruff: noqa: C901, PLR0912, PLR0913, PLR0915
"""Run the frozen FSDD-LODF support-only Stage -1 assay.

Only exact manifest train/good paths are loadable.  The runner has no dataset
enumeration code and therefore cannot access either MVTec AD2 test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from run_cfntr_stage_minus1 import (  # noqa: E402
    AXES,
    LAYERS,
    OBJECTS,
    RESOLUTIONS,
    SupportEntry,
    deterministic_sign,
    extract_layers,
    load_support_manifest,
    make_synthetic_defect,
    mask_to_patch_grid,
    read_rgb,
    sha256_file,
    validation_specs,
)

from flow_tte.cfntr import (  # noqa: E402
    apply_rgb_transform,
    make_axis_transform,
)
from flow_tte.fsdd import (  # noqa: E402
    FactorialEffects,
    LinearHead,
    LinearHeadConfig,
    balanced_map_sample,
    project_layer_features,
    score_linear_head,
    stable_seed,
    toroidal_mask_shuffle,
    train_linear_head,
)
from fmad.backbones.dinov2 import DINOv2Backbone  # noqa: E402

PROTOCOL_ID = "fsdd-lodf-stage-minus1-support-only-20260719-r1"
EXPECTED_SUPPORT_SHA256 = "9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5"
FAMILIES = ("cutpaste", "chromatic_blotch", "gaussian_blur", "high_frequency_ribbon")
GENERATOR_KIND = {
    "cutpaste": "cutpaste",
    "chromatic_blotch": "chromatic_blotch",
    "gaussian_blur": "gaussian_blur",
    "high_frequency_ribbon": "scratch",
}
LEARNED_VARIANTS = (
    "treatment",
    "nonfactorial",
    "defect_only",
    "label_shuffle",
    "mask_shuffle",
    "boundary_only",
)
VARIANTS = (*LEARNED_VARIANTS, "rgb_seam", "rgb_texture", "raw_state")
PROJECTED_DIM_PER_LAYER = 32
SAMPLES_PER_CLASS = 8
DEFECT_RETRIES = 32
HEAD_CONFIG = LinearHeadConfig(steps=400, learning_rate=0.05, factorial_weight=0.25)


@dataclass(frozen=True)
class StimulusPool:
    clean: Mapping[str, np.ndarray]
    shifts: Mapping[tuple[str, str], np.ndarray]
    defects: Mapping[tuple[str, str, str, int], np.ndarray]
    masks: Mapping[tuple[str, str], np.ndarray]
    primary_shift_names: Mapping[tuple[str, str, int], str]
    compound_shift_names: Mapping[str, tuple[str, ...]]
    grids: Mapping[str, tuple[int, int]]
    rgb_seams: Mapping[tuple[str, ...], np.ndarray]
    rgb_textures: Mapping[tuple[str, ...], np.ndarray]


def _words(value: str) -> tuple[str, ...]:
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
    parser.add_argument("--experiment-seed", type=int, choices=(0, 1), default=0)
    args = parser.parse_args(list(argv))
    args.objects = _words(args.objects)
    if not args.objects or len(set(args.objects)) != len(args.objects):
        parser.error("--objects must contain unique object names")
    if any(item not in OBJECTS for item in args.objects):
        parser.error(f"--objects must be a subset of {OBJECTS}")
    if args.backbone_model != "dinov2_vitl14":
        parser.error("FSDD Stage -1 freezes dinov2_vitl14")
    return args


def _project(features: Mapping[int, np.ndarray]) -> np.ndarray:
    return project_layer_features(
        features,
        LAYERS,
        output_dim_per_layer=PROJECTED_DIM_PER_LAYER,
        seed_prefix=PROTOCOL_ID,
    )


def _interior_boundary(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(mask, dtype=np.uint8)
    interior = cv2.erode(value, np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
    boundary = np.asarray(value.astype(bool) & ~interior, dtype=bool)
    return interior, boundary


def _rgb_seam_map(image: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    gray = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    return cv2.resize(magnitude, (grid[1], grid[0]), interpolation=cv2.INTER_AREA).reshape(-1)


def _rgb_texture_map(image: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Fixed same-image high-frequency null using absolute 4-neighbour Laplacian energy."""
    gray = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    energy = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=1))
    return cv2.resize(energy, (grid[1], grid[0]), interpolation=cv2.INTER_AREA).reshape(-1)


def _visibility_audit(
    reference_image: np.ndarray,
    stimulus_image: np.ndarray,
    reference_features: np.ndarray,
    stimulus_features: np.ndarray,
    *,
    selection: np.ndarray | None = None,
) -> dict[str, Any]:
    changed_pixels = np.any(stimulus_image != reference_image, axis=2)
    delta = np.linalg.norm(stimulus_features - reference_features, axis=1)
    selected = (
        np.ones(delta.size, dtype=bool) if selection is None else np.asarray(selection, dtype=bool)
    )
    values = delta[selected]
    changed_patches = int(np.sum(values > 1e-6))
    valid = bool(
        np.any(changed_pixels)
        and values.size >= 2
        and np.all(np.isfinite(values))
        and changed_patches >= 2
        and float(np.ptp(values)) > 1e-7,
    )
    return {
        "valid": valid,
        "changed_pixels": int(changed_pixels.sum()),
        "changed_patches": changed_patches,
        "delta_min": float(values.min(initial=0.0)),
        "delta_max": float(values.max(initial=0.0)),
        "delta_ptp": float(np.ptp(values)),
    }


def _primary_specs(
    object_name: str,
    source: str,
) -> tuple[tuple[str, str, int, Any], ...]:
    result: list[tuple[str, str, int, Any]] = []
    for axis in AXES:
        sign = deterministic_sign(object_name, source, axis, "fsdd-primary")
        for scale in (1, 2):
            name = f"{axis}_{sign:+d}_x{scale}"
            result.append(
                (
                    name,
                    axis,
                    scale,
                    make_axis_transform(axis, sign=sign, amplitude_scale=float(scale)),
                ),
            )
    return tuple(result)


def _interior_capable_high_frequency_ribbon(
    image: np.ndarray,
    grid: tuple[int, int],
    metadata: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Rebuild sealed endpoints as the preregistered high-frequency ribbon construct."""
    endpoints = metadata.get("endpoints")
    if not isinstance(endpoints, list) or len(endpoints) != 2:
        raise RuntimeError("high-frequency ribbon metadata must contain two endpoints")
    (y1, x1), (y2, x2) = endpoints
    height, width = image.shape[:2]
    patch_span = max(height / grid[0], width / grid[1])
    thickness = max(3, int(np.ceil(4.0 * patch_span)))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 1, thickness)
    parity = np.indices((height, width)).sum(axis=0) % 2
    values = np.repeat(np.where(parity == 0, 255, 0).astype(np.uint8)[:, :, None], 3, axis=2)
    output = np.asarray(image, dtype=np.uint8).copy()
    output[mask.astype(bool)] = values[mask.astype(bool)]
    revised = dict(metadata)
    revised.update(
        {
            "fsdd_construct": "high_frequency_ribbon",
            "fsdd_ribbon_fill": "one_pixel_checkerboard_0_255",
            "fsdd_ribbon_thickness_rule": "ceil(4*max(H/grid_h,W/grid_w))",
            "fsdd_ribbon_thickness": thickness,
            "mask_pixels": int(mask.sum()),
            "image_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
            "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        },
    )
    return output, mask.astype(bool), revised


def build_pool(
    backbone: DINOv2Backbone,
    entries: Sequence[SupportEntry],
    object_name: str,
) -> tuple[StimulusPool, list[dict[str, Any]]]:
    clean: dict[str, np.ndarray] = {}
    shifts: dict[tuple[str, str], np.ndarray] = {}
    defects: dict[tuple[str, str, str, int], np.ndarray] = {}
    masks: dict[tuple[str, str], np.ndarray] = {}
    primary_names: dict[tuple[str, str, int], str] = {}
    compound_names: dict[str, tuple[str, ...]] = {}
    grids: dict[str, tuple[int, int]] = {}
    rgb_seams: dict[tuple[str, ...], np.ndarray] = {}
    rgb_textures: dict[tuple[str, ...], np.ndarray] = {}
    audit: list[dict[str, Any]] = []
    for entry in entries:
        source = entry.relative_path
        image = read_rgb(entry.path)
        extracted, grid = extract_layers(backbone, image)
        grids[source] = grid
        clean[source] = _project(extracted)
        rgb_seams[("clean", source)] = _rgb_seam_map(image, grid)
        rgb_textures[("clean", source)] = _rgb_texture_map(image, grid)
        primary_specs = _primary_specs(object_name, source)
        compound_specs = tuple(validation_specs(object_name, source)[-2:])
        compound_names[source] = tuple(name for name, _ in compound_specs)
        for name, axis, scale, spec in primary_specs:
            shifted_image = apply_rgb_transform(image, spec)
            shifted_layers, shifted_grid = extract_layers(
                backbone,
                shifted_image,
            )
            if shifted_grid != grid:
                raise RuntimeError("shift changed the DINO patch grid")
            shifts[(source, name)] = _project(shifted_layers)
            primary_names[(source, axis, scale)] = name
            visibility = _visibility_audit(
                image,
                shifted_image,
                clean[source],
                shifts[(source, name)],
            )
            if not visibility["valid"]:
                message = f"{source}/{name}: pixel/feature no-op shift"
                raise RuntimeError(message)
            rgb_seams[("shift", source, name)] = _rgb_seam_map(shifted_image, grid)
            rgb_textures[("shift", source, name)] = _rgb_texture_map(shifted_image, grid)
            audit.append(
                {
                    "source": source,
                    "kind": "primary_shift",
                    "name": name,
                    "axis": axis,
                    "scale": scale,
                    **visibility,
                },
            )
        for name, spec in compound_specs:
            shifted_image = apply_rgb_transform(image, spec)
            shifted_layers, shifted_grid = extract_layers(
                backbone,
                shifted_image,
            )
            if shifted_grid != grid:
                raise RuntimeError("compound shift changed the DINO patch grid")
            shifts[(source, name)] = _project(shifted_layers)
            visibility = _visibility_audit(
                image,
                shifted_image,
                clean[source],
                shifts[(source, name)],
            )
            if not visibility["valid"]:
                message = f"{source}/{name}: pixel/feature no-op compound shift"
                raise RuntimeError(message)
            rgb_seams[("shift", source, name)] = _rgb_seam_map(shifted_image, grid)
            rgb_textures[("shift", source, name)] = _rgb_texture_map(shifted_image, grid)
            audit.append({"source": source, "kind": "compound_shift", "name": name, **visibility})
        for family in FAMILIES:
            selected: (
                tuple[
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    dict[str, Any],
                    dict[tuple[str, int], tuple[np.ndarray, np.ndarray, dict[str, Any]]],
                ]
                | None
            ) = None
            for retry in range(DEFECT_RETRIES):
                try:
                    defect, native_mask, metadata = make_synthetic_defect(
                        image,
                        object_name=object_name,
                        relative_path=source,
                        kind=GENERATOR_KIND[family],
                        feature_retry=retry,
                    )
                except RuntimeError:
                    continue
                if family == "high_frequency_ribbon":
                    defect, native_mask, metadata = _interior_capable_high_frequency_ribbon(
                        image,
                        grid,
                        metadata,
                    )
                patch_mask = np.asarray(mask_to_patch_grid(native_mask, grid), dtype=bool).reshape(
                    grid,
                )
                interior, boundary = _interior_boundary(patch_mask)
                if not bool(np.any(interior)) or not bool(np.any(boundary)):
                    continue
                defect_layers, defect_grid = extract_layers(backbone, defect)
                if defect_grid != grid:
                    raise RuntimeError("defect changed the DINO patch grid")
                clean_defect = _project(defect_layers)
                affected = patch_mask.reshape(-1)
                clean_visibility = _visibility_audit(
                    image,
                    defect,
                    clean[source],
                    clean_defect,
                    selection=affected,
                )
                if not clean_visibility["valid"]:
                    continue
                shifted_features: dict[
                    tuple[str, int],
                    tuple[np.ndarray, np.ndarray, dict[str, Any]],
                ] = {}
                visible = True
                for name, axis, scale, spec in primary_specs:
                    shifted_clean_image = apply_rgb_transform(image, spec)
                    shifted_defect_image = apply_rgb_transform(defect, spec)
                    shifted_layers, shifted_grid = extract_layers(
                        backbone,
                        shifted_defect_image,
                    )
                    if shifted_grid != grid:
                        raise RuntimeError("shifted defect changed the DINO patch grid")
                    value = _project(shifted_layers)
                    shifted_clean = shifts[(source, name)]
                    shifted_visibility = _visibility_audit(
                        shifted_clean_image,
                        shifted_defect_image,
                        shifted_clean,
                        value,
                        selection=affected,
                    )
                    if not shifted_visibility["valid"]:
                        visible = False
                        break
                    shifted_features[(axis, scale)] = (
                        value,
                        shifted_defect_image,
                        shifted_visibility,
                    )
                if visible:
                    selected = defect, clean_defect, patch_mask, metadata, shifted_features
                    break
            if selected is None:
                message = f"{source}/{family}: no deterministic feature-visible interior defect"
                raise RuntimeError(message)
            defect_image, clean_defect, patch_mask, metadata, shifted_features = selected
            masks[(source, family)] = patch_mask
            defects[(source, family, "clean", 0)] = clean_defect
            rgb_seams[("defect", source, family, "clean", "0")] = _rgb_seam_map(
                defect_image,
                grid,
            )
            rgb_textures[("defect", source, family, "clean", "0")] = _rgb_texture_map(
                defect_image,
                grid,
            )
            shifted_audit: dict[str, Any] = {}
            for (axis, scale), (value, shifted_image, visibility) in shifted_features.items():
                defects[(source, family, axis, scale)] = value
                rgb_seams[("defect", source, family, axis, str(scale))] = _rgb_seam_map(
                    shifted_image,
                    grid,
                )
                rgb_textures[("defect", source, family, axis, str(scale))] = _rgb_texture_map(
                    shifted_image,
                    grid,
                )
                shifted_audit[f"{axis}_x{scale}"] = visibility
            audit.append(
                {
                    "source": source,
                    "kind": "feature_visible_defect",
                    "family": family,
                    "seed": metadata["seed"],
                    "feature_retry": metadata["feature_retry"],
                    "interior_pixels": int(_interior_boundary(patch_mask)[0].sum()),
                    "boundary_pixels": int(_interior_boundary(patch_mask)[1].sum()),
                    "clean": _visibility_audit(
                        image,
                        defect_image,
                        clean[source],
                        clean_defect,
                        selection=patch_mask.reshape(-1),
                    ),
                    "shifted": shifted_audit,
                },
            )
    return StimulusPool(
        clean,
        shifts,
        defects,
        masks,
        primary_names,
        compound_names,
        grids,
        rgb_seams,
        rgb_textures,
    ), audit


def _append_sample(
    rows: list[np.ndarray],
    labels: list[np.ndarray],
    features: np.ndarray,
    mask: np.ndarray,
    *,
    seed: int,
) -> None:
    sampled, target, _ = balanced_map_sample(
        features,
        mask,
        samples_per_class=SAMPLES_PER_CLASS,
        seed=seed,
        replace_shortfall=True,
    )
    rows.append(sampled)
    labels.append(target)


def training_data(
    pool: StimulusPool,
    entries: Sequence[SupportEntry],
    *,
    source_fold: int,
    heldout_family: str | None,
    heldout_axis: str | None,
    variant: str,
    experiment_seed: int,
) -> tuple[np.ndarray, np.ndarray, FactorialEffects | None]:
    if variant not in LEARNED_VARIANTS:
        raise ValueError("unknown FSDD arm")
    rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    effect_parts: dict[str, list[np.ndarray]] = defaultdict(list)
    sampling_arm = "factorial_cells" if variant in {"treatment", "nonfactorial"} else variant
    for entry in (item for item in entries if source_fold < 0 or item.fold != source_fold):
        source = entry.relative_path
        zeros = np.zeros(pool.clean[source].shape[0], dtype=bool)
        _append_sample(
            rows,
            labels,
            pool.clean[source],
            zeros,
            seed=stable_seed(experiment_seed, source, sampling_arm, "clean"),
        )
        if variant != "defect_only":
            for axis in (item for item in AXES if item != heldout_axis):
                for scale in (1, 2):
                    shift_name = pool.primary_shift_names[(source, axis, scale)]
                    _append_sample(
                        rows,
                        labels,
                        pool.shifts[(source, shift_name)],
                        zeros,
                        seed=stable_seed(experiment_seed, source, shift_name, sampling_arm),
                    )
        for family in (item for item in FAMILIES if item != heldout_family):
            interior, boundary = _interior_boundary(pool.masks[(source, family)])
            base_mask = boundary if variant == "boundary_only" else interior
            used_mask = (
                toroidal_mask_shuffle(
                    base_mask,
                    seed=stable_seed(experiment_seed, source, family, "mask-null"),
                )
                if variant == "mask_shuffle"
                else base_mask
            )
            _append_sample(
                rows,
                labels,
                pool.defects[(source, family, "clean", 0)],
                used_mask,
                seed=stable_seed(experiment_seed, source, family, "clean", sampling_arm),
            )
            if variant != "defect_only":
                for axis in (item for item in AXES if item != heldout_axis):
                    for scale in (1, 2):
                        _append_sample(
                            rows,
                            labels,
                            pool.defects[(source, family, axis, scale)],
                            used_mask,
                            seed=stable_seed(
                                experiment_seed,
                                source,
                                family,
                                axis,
                                scale,
                                sampling_arm,
                            ),
                        )
            if variant == "treatment":
                for axis in (item for item in AXES if item != heldout_axis):
                    for scale in (1, 2):
                        _, _, indices = balanced_map_sample(
                            pool.defects[(source, family, "clean", 0)],
                            interior,
                            samples_per_class=SAMPLES_PER_CLASS,
                            seed=stable_seed(
                                experiment_seed,
                                source,
                                family,
                                axis,
                                scale,
                                "effect",
                            ),
                            replace_shortfall=True,
                        )
                        shift_name = pool.primary_shift_names[(source, axis, scale)]
                        effect_parts["clean"].append(pool.clean[source][indices])
                        effect_parts["shifted_clean"].append(
                            pool.shifts[(source, shift_name)][indices],
                        )
                        effect_parts["defect"].append(
                            pool.defects[(source, family, "clean", 0)][indices],
                        )
                        effect_parts["shifted_defect"].append(
                            pool.defects[(source, family, axis, scale)][indices],
                        )
    x = np.concatenate(rows, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(labels).astype(np.float32, copy=False)
    if variant == "label_shuffle":
        y = y[
            np.random.default_rng(
                stable_seed(experiment_seed, source_fold, heldout_axis, heldout_family, variant),
            ).permutation(y.size)
        ]
    effects = None
    if variant == "treatment":
        effects = FactorialEffects(
            **{name: np.concatenate(values, axis=0) for name, values in effect_parts.items()},
        )
    return x, y, effects


def _best_f1(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(np.max(2.0 * precision * recall / np.maximum(precision + recall, 1e-12)))


def _defect_metric(labels: np.ndarray, scores: np.ndarray, *, threshold: float) -> dict[str, float]:
    y = np.asarray(labels).reshape(-1).astype(np.uint8)
    s = np.asarray(scores).reshape(-1).astype(np.float64)
    if np.unique(y).size != 2 or not bool(np.all(np.isfinite(s))):
        raise RuntimeError("invalid held-out defect metric population")
    prediction = s >= threshold
    tp = float(np.sum(prediction & (y == 1)))
    fp = float(np.sum(prediction & (y == 0)))
    fn = float(np.sum(~prediction & (y == 1)))
    return {
        "pAUROC_0.05": float(roc_auc_score(y, s, max_fpr=0.05)),
        "fixed_F1": 2.0 * tp / max(2.0 * tp + fp + fn, 1.0),
        "oracle_F1": _best_f1(y, s),
        "pixels": int(y.size),
        "positive_pixels": int(y.sum()),
    }


def _metric_population(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    full = np.asarray(mask, dtype=bool)
    interior, _boundary = _interior_boundary(full)
    keep = np.asarray(interior | ~full, dtype=bool)
    labels = np.asarray(interior[keep], dtype=bool)
    if not np.any(labels) or np.all(labels):
        raise RuntimeError("defect interior metric population is degenerate")
    return keep.reshape(-1), labels.reshape(-1)


def _fit_raw_state(
    pool: StimulusPool,
    train_entries: Sequence[SupportEntry],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.concatenate([pool.clean[item.relative_path] for item in train_entries], axis=0)
    mean = matrix.astype(np.float64).mean(axis=0)
    scale = np.maximum(matrix.astype(np.float64).std(axis=0), 1e-4)
    return np.asarray(mean, dtype=np.float32), np.asarray(scale, dtype=np.float32)


def _raw_state_score(model: tuple[np.ndarray, np.ndarray], features: np.ndarray) -> np.ndarray:
    mean, scale = model
    standardized = (features.astype(np.float64) - mean) / scale
    return np.asarray(np.mean(np.square(standardized), axis=1), dtype=np.float32)


def _fpr(scores: np.ndarray, threshold: float) -> float:
    return float(np.mean(np.asarray(scores, dtype=np.float64) >= threshold))


def _coverage(maps: Sequence[np.ndarray]) -> dict[str, bool]:
    finite = all(bool(np.all(np.isfinite(value))) for value in maps)
    nonconstant = all(float(np.ptp(value)) > 0.0 for value in maps)
    aligned = len({np.asarray(value).shape for value in maps}) == 1
    return {
        "complete": bool(finite and nonconstant and aligned),
        "finite": bool(finite),
        "nonconstant": bool(nonconstant),
        "aligned": bool(aligned),
    }


def _head_audit(head: LinearHead) -> dict[str, Any]:
    return {
        "initial_loss": head.initial_loss,
        "final_loss": head.final_loss,
        "training_rows": head.training_rows,
        "positive_rows": head.positive_rows,
        "weight_l2": float(np.linalg.norm(head.weight)),
        "weight_sha256": hashlib.sha256(head.weight.tobytes()).hexdigest(),
    }


def run_object(
    args: argparse.Namespace,
    backbone: DINOv2Backbone,
    entries: Sequence[SupportEntry],
    object_name: str,
) -> dict[str, Any]:
    backbone.set_resolution(RESOLUTIONS[object_name])
    pool, stimulus_audit = build_pool(backbone, entries, object_name)
    head_records: list[dict[str, Any]] = []
    head_audit: list[dict[str, Any]] = []
    for source_fold in range(4):
        train_entries = [entry for entry in entries if entry.fold != source_fold]
        heldout_entries = [entry for entry in entries if entry.fold == source_fold]
        raw_model = _fit_raw_state(pool, train_entries)
        raw_train = np.concatenate(
            [_raw_state_score(raw_model, pool.clean[item.relative_path]) for item in train_entries],
        )
        raw_threshold = float(np.quantile(raw_train, 0.995))
        rgb_train = np.concatenate(
            [pool.rgb_seams[("clean", item.relative_path)] for item in train_entries],
        )
        rgb_threshold = float(np.quantile(rgb_train, 0.995))
        texture_train = np.concatenate(
            [pool.rgb_textures[("clean", item.relative_path)] for item in train_entries],
        )
        texture_threshold = float(np.quantile(texture_train, 0.995))
        for heldout_axis in AXES:
            for family in FAMILIES:
                heads: dict[str, LinearHead] = {}
                thresholds: dict[str, float] = {
                    "raw_state": raw_threshold,
                    "rgb_seam": rgb_threshold,
                    "rgb_texture": texture_threshold,
                }
                for variant in LEARNED_VARIANTS:
                    x, y, effects = training_data(
                        pool,
                        entries,
                        source_fold=source_fold,
                        heldout_family=family,
                        heldout_axis=heldout_axis,
                        variant=variant,
                        experiment_seed=args.experiment_seed,
                    )
                    head = train_linear_head(
                        x,
                        y,
                        config=HEAD_CONFIG,
                        effects=effects,
                        device=args.device,
                    )
                    heads[variant] = head
                    train_clean_scores = np.concatenate(
                        [
                            score_linear_head(head, pool.clean[item.relative_path])
                            for item in train_entries
                        ],
                    )
                    thresholds[variant] = float(np.quantile(train_clean_scores, 0.995))
                    head_audit.append(
                        {
                            "source_fold": source_fold,
                            "heldout_axis": heldout_axis,
                            "heldout_family": family,
                            "variant": variant,
                            **_head_audit(head),
                        },
                    )
                for entry in heldout_entries:
                    source = entry.relative_path
                    mask = pool.masks[(source, family)]
                    keep, metric_labels = _metric_population(mask)
                    variant_payloads: dict[str, Any] = {}
                    for variant in VARIANTS:
                        if variant in heads:
                            head = heads[variant]

                            def scorer(
                                value: np.ndarray,
                                fitted_head: LinearHead = head,
                            ) -> np.ndarray:
                                return score_linear_head(fitted_head, value)

                            clean_score = scorer(pool.clean[source])
                            shifted_scores = {
                                scale: scorer(
                                    pool.shifts[
                                        (
                                            source,
                                            pool.primary_shift_names[(source, heldout_axis, scale)],
                                        )
                                    ],
                                )
                                for scale in (1, 2)
                            }
                            clean_defect = scorer(pool.defects[(source, family, "clean", 0)])
                            shifted_defects = {
                                scale: scorer(
                                    pool.defects[(source, family, heldout_axis, scale)],
                                )
                                for scale in (1, 2)
                            }
                            compound_scores = [
                                scorer(pool.shifts[(source, name)])
                                for name in pool.compound_shift_names[source]
                            ]
                        elif variant == "raw_state":
                            clean_score = _raw_state_score(raw_model, pool.clean[source])
                            shifted_scores = {
                                scale: _raw_state_score(
                                    raw_model,
                                    pool.shifts[
                                        (
                                            source,
                                            pool.primary_shift_names[(source, heldout_axis, scale)],
                                        )
                                    ],
                                )
                                for scale in (1, 2)
                            }
                            clean_defect = _raw_state_score(
                                raw_model,
                                pool.defects[(source, family, "clean", 0)],
                            )
                            shifted_defects = {
                                scale: _raw_state_score(
                                    raw_model,
                                    pool.defects[(source, family, heldout_axis, scale)],
                                )
                                for scale in (1, 2)
                            }
                            compound_scores = [
                                _raw_state_score(raw_model, pool.shifts[(source, name)])
                                for name in pool.compound_shift_names[source]
                            ]
                        elif variant == "rgb_seam":
                            clean_score = pool.rgb_seams[("clean", source)]
                            shifted_scores = {
                                scale: pool.rgb_seams[
                                    (
                                        "shift",
                                        source,
                                        pool.primary_shift_names[(source, heldout_axis, scale)],
                                    )
                                ]
                                for scale in (1, 2)
                            }
                            clean_defect = pool.rgb_seams[("defect", source, family, "clean", "0")]
                            shifted_defects = {
                                scale: pool.rgb_seams[
                                    ("defect", source, family, heldout_axis, str(scale))
                                ]
                                for scale in (1, 2)
                            }
                            compound_scores = [
                                pool.rgb_seams[("shift", source, name)]
                                for name in pool.compound_shift_names[source]
                            ]
                        else:
                            clean_score = pool.rgb_textures[("clean", source)]
                            shifted_scores = {
                                scale: pool.rgb_textures[
                                    (
                                        "shift",
                                        source,
                                        pool.primary_shift_names[(source, heldout_axis, scale)],
                                    )
                                ]
                                for scale in (1, 2)
                            }
                            clean_defect = pool.rgb_textures[
                                ("defect", source, family, "clean", "0")
                            ]
                            shifted_defects = {
                                scale: pool.rgb_textures[
                                    ("defect", source, family, heldout_axis, str(scale))
                                ]
                                for scale in (1, 2)
                            }
                            compound_scores = [
                                pool.rgb_textures[("shift", source, name)]
                                for name in pool.compound_shift_names[source]
                            ]
                        threshold = thresholds[variant]
                        clean_metric = _defect_metric(
                            metric_labels,
                            clean_defect[keep],
                            threshold=threshold,
                        )
                        shifted_metric = _defect_metric(
                            np.tile(metric_labels, 2),
                            np.concatenate([shifted_defects[1][keep], shifted_defects[2][keep]]),
                            threshold=threshold,
                        )
                        maps = [
                            clean_score,
                            shifted_scores[1],
                            shifted_scores[2],
                            clean_defect,
                            shifted_defects[1],
                            shifted_defects[2],
                        ]
                        variant_payloads[variant] = {
                            "threshold": threshold,
                            "threshold_quantile": 0.995,
                            "clean_fpr": _fpr(clean_score, threshold),
                            "primary_shift_fpr": {
                                "x1": _fpr(shifted_scores[1], threshold),
                                "x2": _fpr(shifted_scores[2], threshold),
                            },
                            "secondary_compound_fpr": float(
                                np.mean([_fpr(value, threshold) for value in compound_scores]),
                            ),
                            "clean": clean_metric,
                            "shifted": shifted_metric,
                            "interior": {
                                "clean": {
                                    key: clean_metric[key] for key in ("pAUROC_0.05", "fixed_F1")
                                },
                                "shifted": {
                                    key: shifted_metric[key] for key in ("pAUROC_0.05", "fixed_F1")
                                },
                            },
                            "coverage": _coverage(maps),
                        }
                    head_records.append(
                        {
                            "source": source,
                            "source_fold": source_fold,
                            "heldout_family": family,
                            "heldout_axis": heldout_axis,
                            "variants": variant_payloads,
                        },
                    )
    summary = {
        "schema": "fsdd-lodf-stage-minus1-object-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "evaluation_role": "support_only_synthetic_identifiability_preflight",
        "maximum_verdict": "PROMISING_DIAGNOSTIC",
        "object": object_name,
        "experiment_seed": args.experiment_seed,
        "support_count": len(entries),
        "fold_counts": np.bincount([entry.fold for entry in entries], minlength=4).tolist(),
        "support": [
            {
                "object_name": entry.object_name,
                "path": str(entry.path),
                "relative_path": entry.relative_path,
                "path_sha256": entry.path_sha256,
                "content_sha256": entry.content_sha256,
                "fold": entry.fold,
            }
            for entry in entries
        ],
        "support_manifest_sha256": EXPECTED_SUPPORT_SHA256,
        "resolution": RESOLUTIONS[object_name],
        "layers": list(LAYERS),
        "projected_dim_per_layer": PROJECTED_DIM_PER_LAYER,
        "variants": list(VARIANTS),
        "head_records": head_records,
        "post_gate_deployment_rule": {
            "authorized_only_if_strict_gate_passes": True,
            "seed": 0,
            "steps": HEAD_CONFIG.steps,
            "threshold": "q=0.995 over all-16 clean training maps",
            "ensemble": False,
        },
        "stimulus_audit": stimulus_audit,
        "head_audit": head_audit,
        "access_audit": {
            "test_public_accessed": False,
            "test_private_accessed": False,
            "ground_truth_accessed": False,
            "query_update": False,
            "query_transform": False,
            "tta": False,
            "forbidden_path_hits": [],
        },
        "host_gpu_id": str(args.host_gpu_id),
        "container_cuda_slot": str(args.container_cuda_slot),
    }
    output = Path(args.output_root) / f"{object_name}_seed{args.experiment_seed}_summary.json"
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
        raise RuntimeError("support manifest SHA-256 mismatch")
    support = load_support_manifest(support_path, Path(args.data_root))
    backbone = DINOv2Backbone(
        model_name=args.backbone_model,
        device=args.device,
        smaller_edge_size=672,
        feature_layers=LAYERS,
    )
    summaries = [run_object(args, backbone, support[name], name) for name in args.objects]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "objects": list(args.objects),
        "experiment_seed": args.experiment_seed,
        "support_manifest_sha256": EXPECTED_SUPPORT_SHA256,
        "runner_sha256": sha256_file(Path(__file__)),
        "core_sha256": sha256_file(ROOT / "src" / "flow_tte" / "fsdd.py"),
        "summary_sha256": {
            item["object"]: sha256_file(
                output_root / f"{item['object']}_seed{args.experiment_seed}_summary.json",
            )
            for item in summaries
        },
        "test_public_accessed": False,
        "test_private_accessed": False,
        "tta_used": False,
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
