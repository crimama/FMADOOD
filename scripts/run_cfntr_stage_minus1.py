#!/usr/bin/env python3
"""Run the support-only CF-NTR Stage -1 identifiability preflight.

This runner is deliberately unable to enumerate MVTec AD2 TESTpublic.  Its
entire image population is the exact sixteen train/good paths supplied for
each object by the frozen SuperAD support manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from sklearn.metrics import precision_recall_curve, roc_auc_score

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "src", Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from flow_tte.cfntr import (  # noqa: E402
    TRAIN_AXIS_AMPLITUDES,
    VALIDATION_AMPLITUDE_SCALE,
    RGBTransformSpec,
    apply_rgb_transform,
    central_difference_observation,
    deterministic_random_basis,
    fit_image_level_basis,
    image_level_mean_observation,
    make_axis_transform,
    make_compound_transform,
    normalize_rows,
    score_patchwise_residuals,
    score_raw_residuals,
    subspace_energy_coverage,
    subspace_overlap_stability,
    summarize_map_coverage,
)
from fmad.backbones.dinov2 import DINOv2Backbone  # noqa: E402

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
LAYERS = (5, 11, 17, 23)
AXES = ("exposure", "gamma", "contrast", "white_balance")
COMPOUNDS = ("exposure_white_balance", "gamma_contrast")
DEFECT_KINDS = ("cutpaste", "chromatic_blotch", "gaussian_blur", "scratch")
RANK = 4
FOLD_COUNT = 4
SUPPORT_COUNT = 16
RANDOM_BASIS_COUNT = 16
BASIS_BOOTSTRAPS = 200
MACRO_BOOTSTRAPS = 2000
DEFECT_GENERATION_ATTEMPTS = 32
RESOLUTIONS = {name: (448 if name == "sheet_metal" else 672) for name in OBJECTS}
PROTOCOL_ID = "cfntr-stage-minus1-support-only-20260718"


@dataclass(frozen=True)
class SupportEntry:
    object_name: str
    path: Path
    relative_path: str
    path_sha256: str
    content_sha256: str
    fold: int


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in (PROTOCOL_ID, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def deterministic_sign(*parts: object) -> int:
    return 1 if stable_seed(*parts) & 1 else -1


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
    parser.add_argument("--basis-bootstraps", type=int, default=BASIS_BOOTSTRAPS)
    args = parser.parse_args(list(argv))
    args.objects = _split_words(args.objects)
    if not args.objects or len(set(args.objects)) != len(args.objects):
        parser.error("--objects must contain unique object names")
    if any(name not in OBJECTS for name in args.objects):
        parser.error(f"--objects must be a subset of {OBJECTS}")
    if args.backbone_model != "dinov2_vitl14":
        parser.error("Stage -1 freezes --backbone-model=dinov2_vitl14")
    if args.basis_bootstraps != BASIS_BOOTSTRAPS:
        parser.error(f"Stage -1 freezes --basis-bootstraps={BASIS_BOOTSTRAPS}")
    return args


def _manifest_tail(object_name: str, raw_path: str) -> tuple[str, str, str, str]:
    parts = PurePosixPath(raw_path.replace("\\", "/")).parts
    if len(parts) < 4:
        msg = f"{object_name}: malformed support path {raw_path!r}"
        raise ValueError(msg)
    return tuple(parts[-4:])  # type: ignore[return-value]


def load_support_manifest(  # noqa: C901
    path: Path,
    data_root: Path,
) -> dict[str, tuple[SupportEntry, ...]]:
    """Load exactly 16 train/good images per object and assign frozen SHA folds."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("support manifest must be a JSON object")
    if set(payload) != set(OBJECTS):
        raise ValueError("support manifest must contain exactly the eight frozen objects")
    root = data_root.resolve(strict=True)
    result: dict[str, tuple[SupportEntry, ...]] = {}
    for object_name in OBJECTS:
        raw_paths = payload[object_name]
        if (
            not isinstance(raw_paths, list)
            or len(raw_paths) != SUPPORT_COUNT
            or not all(isinstance(item, str) for item in raw_paths)
        ):
            msg = f"{object_name}: expected exactly {SUPPORT_COUNT} string paths"
            raise ValueError(msg)
        resolved: list[tuple[Path, str, str]] = []
        for raw_path in raw_paths:
            tail = _manifest_tail(object_name, raw_path)
            if tail[:3] != (object_name, "train", "good"):
                msg = f"{object_name}: support path must be inside {object_name}/train/good"
                raise ValueError(
                    msg,
                )
            candidate = (root / object_name / "train" / "good" / tail[3]).resolve(strict=True)
            train_root = (root / object_name / "train" / "good").resolve(strict=True)
            if candidate.parent != train_root or not candidate.is_file():
                msg = f"{object_name}: support path escapes train/good"
                raise ValueError(msg)
            relative = candidate.relative_to(root).as_posix()
            if any(
                segment in {"test_public", "test_private", "ground_truth"}
                for segment in relative.split("/")
            ):
                raise ValueError("public data paths are forbidden in Stage -1")
            resolved.append((candidate, relative, sha256_bytes(relative.encode("utf-8"))))
        if len({item[0] for item in resolved}) != SUPPORT_COUNT:
            msg = f"{object_name}: support paths must be unique"
            raise ValueError(msg)
        ordered = sorted(resolved, key=lambda item: (item[2], item[1]))
        fold_by_path = {item[0]: index % FOLD_COUNT for index, item in enumerate(ordered)}
        entries = tuple(
            SupportEntry(
                object_name=object_name,
                path=candidate,
                relative_path=relative,
                path_sha256=path_hash,
                content_sha256=sha256_file(candidate),
                fold=fold_by_path[candidate],
            )
            for candidate, relative, path_hash in resolved
        )
        counts = np.bincount([item.fold for item in entries], minlength=FOLD_COUNT)
        if counts.tolist() != [4, 4, 4, 4]:
            msg = f"{object_name}: invalid frozen fold allocation {counts.tolist()}"
            raise RuntimeError(msg)
        result[object_name] = entries
    return result


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        msg = f"failed to read support image: {path}"
        raise RuntimeError(msg)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def extract_layers(
    backbone: DINOv2Backbone,
    image_rgb: np.ndarray,
) -> tuple[dict[int, np.ndarray], tuple[int, int]]:
    tensor, grid = backbone.prepare_image(image_rgb)
    raw = backbone.extract_features(tensor)
    if len(raw) != len(LAYERS):
        msg = f"expected {len(LAYERS)} feature layers, got {len(raw)}"
        raise RuntimeError(msg)
    expected = int(grid[0]) * int(grid[1])
    result: dict[int, np.ndarray] = {}
    for layer, value in zip(LAYERS, raw):
        features = np.asarray(value, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != expected:
            msg = f"layer {layer}: feature/grid mismatch {features.shape} vs {grid}"
            raise RuntimeError(msg)
        result[layer] = normalize_rows(features)
    return result, (int(grid[0]), int(grid[1]))


def validation_specs(object_name: str, relative_path: str) -> tuple[tuple[str, Any], ...]:
    specs: list[tuple[str, Any]] = []
    for axis in AXES:
        sign = deterministic_sign(object_name, relative_path, axis, "validation")
        specs.append(
            (
                f"{axis}_{sign:+d}_x2",
                make_axis_transform(
                    axis,
                    sign=sign,
                    amplitude_scale=2.0,
                ),
            ),
        )
    for compound in COMPOUNDS:
        first = deterministic_sign(object_name, relative_path, compound, "first")
        second = deterministic_sign(object_name, relative_path, compound, "second")
        specs.append(
            (
                f"{compound}_{first:+d}_{second:+d}",
                make_compound_transform(
                    compound,
                    signs=(first, second),
                    amplitude_scale=2.0,
                ),
            ),
        )
    return tuple(specs)


def assigned_defect_shift(
    object_name: str,
    relative_path: str,
    kind: str,
) -> tuple[str, RGBTransformSpec]:
    specs = validation_specs(object_name, relative_path)
    return specs[stable_seed(object_name, relative_path, kind, "assigned-shift") % len(specs)]


def _bounded_rectangle(
    height: int,
    width: int,
    rng: np.random.Generator,
) -> tuple[int, int, int, int]:
    box_h = max(5, min(height // 3, round(height * rng.uniform(0.10, 0.20))))
    box_w = max(5, min(width // 3, round(width * rng.uniform(0.10, 0.20))))
    y = int(rng.integers(0, max(1, height - box_h + 1)))
    x = int(rng.integers(0, max(1, width - box_w + 1)))
    return y, x, box_h, box_w


def _defect_survives_acquisition(
    clean: np.ndarray,
    defective: np.ndarray,
    mask: np.ndarray,
    *,
    shifted_clean: np.ndarray,
    shift_spec: RGBTransformSpec,
) -> bool:
    selection = np.asarray(mask, dtype=bool)
    if not np.any(defective[selection] != clean[selection]):
        return False
    shifted_defective = apply_rgb_transform(defective, shift_spec)
    return bool(np.any(shifted_defective[selection] != shifted_clean[selection]))


def make_synthetic_defect(  # noqa: C901, PLR0912, PLR0915
    image_rgb: np.ndarray,
    *,
    object_name: str,
    relative_path: str,
    kind: str,
    feature_retry: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Create one deterministic, localized, sealed support-side stress defect."""
    if kind not in DEFECT_KINDS:
        msg = f"unknown defect kind {kind!r}"
        raise ValueError(msg)
    image = np.asarray(image_rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must be uint8 HxWx3")
    height, width = image.shape[:2]
    if min(height, width) < 32:
        raise ValueError("synthetic defects require both image dimensions >= 32")
    if feature_retry < 0:
        raise ValueError("feature_retry must be non-negative")
    seed = (
        stable_seed(object_name, relative_path, kind, "defect")
        if feature_retry == 0
        else stable_seed(object_name, relative_path, kind, "defect-feature-retry", feature_retry)
    )
    rng = np.random.default_rng(seed)
    output = image.copy()
    mask = np.zeros((height, width), dtype=np.uint8)
    _, shift_spec = assigned_defect_shift(object_name, relative_path, kind)
    shifted_clean = apply_rgb_transform(image, shift_spec)
    metadata: dict[str, Any] = {
        "kind": kind,
        "seed": seed,
        "feature_retry": feature_retry,
        "generation_attempt": 0,
    }
    if kind == "cutpaste":
        accepted = False
        for attempt in range(DEFECT_GENERATION_ATTEMPTS):
            dy, dx, box_h, box_w = _bounded_rectangle(height, width, rng)
            candidates = [
                (0, 0),
                (0, width - box_w),
                (height - box_h, 0),
                (height - box_h, width - box_w),
            ]
            candidates = [
                (sy, sx)
                for sy, sx in candidates
                if sy + box_h <= dy or dy + box_h <= sy or sx + box_w <= dx or dx + box_w <= sx
            ]
            candidates.sort(
                key=lambda point: abs(point[0] - dy) + abs(point[1] - dx),
                reverse=True,
            )
            for sy, sx in candidates:
                candidate = image.copy()
                candidate_mask = np.zeros_like(mask)
                patch = image[sy : sy + box_h, sx : sx + box_w]
                candidate[dy : dy + box_h, dx : dx + box_w] = np.flip(patch, axis=1)
                candidate_mask[dy : dy + box_h, dx : dx + box_w] = 1
                if _defect_survives_acquisition(
                    image,
                    candidate,
                    candidate_mask,
                    shifted_clean=shifted_clean,
                    shift_spec=shift_spec,
                ):
                    output, mask, accepted = candidate, candidate_mask, True
                    metadata.update(
                        {
                            "destination": [dy, dx, box_h, box_w],
                            "source": [sy, sx],
                            "generation_attempt": attempt,
                        },
                    )
                    break
            if accepted:
                break
        if not accepted:
            raise RuntimeError("cutpaste could not construct an effective defect")
    elif kind == "chromatic_blotch":
        cy = int(rng.integers(height // 5, max(height // 5 + 1, 4 * height // 5)))
        cx = int(rng.integers(width // 5, max(width // 5 + 1, 4 * width // 5)))
        axes = (
            max(4, int(width * rng.uniform(0.04, 0.10))),
            max(4, int(height * rng.uniform(0.04, 0.10))),
        )
        cv2.ellipse(mask, (cx, cy), axes, float(rng.uniform(0, 180)), 0, 360, 1, -1)
        color = rng.integers(0, 256, size=3, dtype=np.uint8)
        alpha = float(rng.uniform(0.35, 0.60))
        blended = np.rint((1.0 - alpha) * output + alpha * color.reshape(1, 1, 3)).astype(np.uint8)
        output[mask.astype(bool)] = blended[mask.astype(bool)]
        metadata.update(
            {"center": [cy, cx], "axes": list(axes), "alpha": alpha, "color": color.tolist()},
        )
    elif kind == "gaussian_blur":
        accepted = False
        for attempt in range(DEFECT_GENERATION_ATTEMPTS):
            y, x, box_h, box_w = _bounded_rectangle(height, width, rng)
            kernel = int(max(5, 2 * (min(box_h, box_w) // 8) + 1))
            if kernel % 2 == 0:
                kernel += 1
            # Use a deterministic CPU filter.  Some OpenCV builds vary their SIMD
            # accumulation order even for repeated calls in one process.
            region = image[y : y + box_h, x : x + box_w]
            sigma = max(0.8, (kernel - 1) / 6.0)
            blurred = (
                np.rint(
                    gaussian_filter(
                        region.astype(np.float32),
                        sigma=(sigma, sigma, 0.0),
                        mode="reflect",
                    ),
                )
                .clip(0, 255)
                .astype(np.uint8)
            )
            candidate = image.copy()
            candidate_mask = np.zeros_like(mask)
            candidate[y : y + box_h, x : x + box_w] = blurred
            candidate_mask[y : y + box_h, x : x + box_w] = 1
            if _defect_survives_acquisition(
                image,
                candidate,
                candidate_mask,
                shifted_clean=shifted_clean,
                shift_spec=shift_spec,
            ):
                output, mask, accepted = candidate, candidate_mask, True
                metadata.update(
                    {
                        "rectangle": [y, x, box_h, box_w],
                        "kernel": kernel,
                        "generation_attempt": attempt,
                    },
                )
                break
        if not accepted:
            raise RuntimeError("gaussian_blur could not construct an effective defect")
    else:
        x1 = int(rng.integers(width // 10, max(width // 10 + 1, 9 * width // 10)))
        y1 = int(rng.integers(height // 10, max(height // 10 + 1, 9 * height // 10)))
        angle = float(rng.uniform(0, 2 * np.pi))
        length = int(rng.uniform(0.20, 0.45) * min(height, width))
        x2 = int(np.clip(round(x1 + length * np.cos(angle)), 0, width - 1))
        y2 = int(np.clip(round(y1 + length * np.sin(angle)), 0, height - 1))
        thickness = max(1, round(min(height, width) * 0.004))
        cv2.line(mask, (x1, y1), (x2, y2), 1, thickness)
        high = np.where(np.indices((height, width)).sum(axis=0) % 2 == 0, 255, 0).astype(np.uint8)
        values = np.repeat(high[:, :, None], 3, axis=2)
        output[mask.astype(bool)] = values[mask.astype(bool)]
        metadata.update({"endpoints": [[y1, x1], [y2, x2]], "thickness": thickness})
    if int(mask.sum()) == 0 or int(mask.sum()) == mask.size:
        msg = f"{kind}: generated invalid synthetic mask"
        raise RuntimeError(msg)
    if not _defect_survives_acquisition(
        image,
        output,
        mask,
        shifted_clean=shifted_clean,
        shift_spec=shift_spec,
    ):
        message = f"{kind}: generated a no-op defect under a sealed acquisition"
        raise RuntimeError(message)
    metadata["mask_pixels"] = int(mask.sum())
    metadata["image_sha256"] = sha256_bytes(output.tobytes())
    metadata["mask_sha256"] = sha256_bytes(mask.tobytes())
    return output, mask.astype(bool), metadata


def _best_f1(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    index = int(np.nanargmax(f1))
    threshold = float(thresholds[index]) if index < thresholds.size else float("inf")
    return float(f1[index]), threshold


def pixel_metrics(mask: np.ndarray, score_map: np.ndarray) -> dict[str, float]:
    labels = np.asarray(mask, dtype=np.uint8).reshape(-1)
    scores = np.asarray(score_map, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(scores)) or np.unique(labels).tolist() != [0, 1]:
        raise ValueError("pixel metrics require finite scores and both mask classes")
    f1, threshold = _best_f1(labels, scores)
    return {
        "seg_AUROC_0.05": float(roc_auc_score(labels, scores, max_fpr=0.05)),
        "seg_F1": f1,
        "best_threshold": threshold,
    }


def random_null_columns(
    captures: Sequence[float],
    drift_reductions: Sequence[float],
) -> dict[str, float]:
    """Retain every seeded null so bootstrap envelopes can be recomputed."""
    if len(captures) != RANDOM_BASIS_COUNT or len(drift_reductions) != RANDOM_BASIS_COUNT:
        msg = f"expected exactly {RANDOM_BASIS_COUNT} random-null values"
        raise ValueError(msg)
    columns: dict[str, float] = {}
    for index, (capture, reduction) in enumerate(zip(captures, drift_reductions)):
        columns[f"random_energy_capture_{index:02d}"] = float(capture)
        columns[f"random_drift_reduction_{index:02d}"] = float(reduction)
    return columns


def defect_feature_residual(
    defect_features: np.ndarray,
    *,
    acquisition: str,
    clean_features: np.ndarray,
    shifted_clean_features: np.ndarray,
) -> np.ndarray:
    """Subtract the clean feature under the identical acquisition condition."""
    if acquisition == "clean":
        baseline = clean_features
    elif acquisition == "shifted":
        baseline = shifted_clean_features
    else:
        raise ValueError("acquisition must be clean or shifted")
    query = np.asarray(defect_features, dtype=np.float32)
    reference = np.asarray(baseline, dtype=np.float32)
    if query.shape != reference.shape:
        raise ValueError("defect and matched-clean features must have identical shape")
    return query - reference


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = tuple(rows[0])
    if any(tuple(row) != keys for row in rows):
        raise ValueError("TSV rows must expose identical ordered fields")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[float]) -> float:
    materialized = np.asarray(tuple(values), dtype=np.float64)
    if materialized.size == 0 or not np.all(np.isfinite(materialized)):
        raise ValueError("cannot summarize empty or non-finite values")
    return float(materialized.mean())


def mask_to_patch_grid(mask: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Return a flat patch-selection mask aligned with flat patch scores."""
    # INTER_AREA returns an occupancy fraction only for floating-point input.
    # uint8 {0, 1} input quantizes sub-patch defects back to zero and violates
    # the preregistered any-native-pixel-overlap rule.
    binary = np.asarray(mask, dtype=np.float32)
    if binary.ndim != 2 or len(grid) != 2 or min(grid) <= 0:
        raise ValueError("mask and feature grid must be two-dimensional and non-empty")
    if not np.all(np.isin(binary, (0.0, 1.0))):
        raise ValueError("mask must be binary")
    occupancy = cv2.resize(
        binary,
        (grid[1], grid[0]),
        interpolation=cv2.INTER_AREA,
    ).reshape(-1)
    return occupancy > 0


def _basis_bootstrap_stability(
    observations_by_image: Sequence[np.ndarray],
    full_basis: np.ndarray,
    *,
    object_name: str,
    layer: int,
    replicates: int,
) -> list[float]:
    values: list[float] = []
    image_count = len(observations_by_image)
    for replicate in range(replicates):
        rng = np.random.default_rng(stable_seed(object_name, layer, "basis-bootstrap", replicate))
        indices = rng.integers(0, image_count, size=image_count)
        observations = np.concatenate([observations_by_image[int(i)] for i in indices], axis=0)
        try:
            basis = fit_image_level_basis(observations, rank=RANK, center=False)
            values.append(float(subspace_overlap_stability(full_basis, basis)))
        except ValueError:
            values.append(0.0)
    return values


def _fit_fold_bases(
    central: Mapping[tuple[str, int], np.ndarray],
    clean: Mapping[tuple[str, int], np.ndarray],
    entries: Sequence[SupportEntry],
    fold: int,
    layer: int,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    fit = [entry for entry in entries if entry.fold != fold]
    physical_obs = np.concatenate([central[(entry.relative_path, layer)] for entry in fit], axis=0)
    clean_obs = np.stack(
        [image_level_mean_observation(clean[(entry.relative_path, layer)]) for entry in fit],
    )
    physical = fit_image_level_basis(physical_obs, rank=RANK, center=False)
    pca = fit_image_level_basis(clean_obs, rank=RANK, center=True)
    feature_dim = int(physical.shape[0])
    random = tuple(
        deterministic_random_basis(
            feature_dim=feature_dim,
            rank=RANK,
            seed=stable_seed(entries[0].object_name, fold, layer, "random", index),
        )
        for index in range(RANDOM_BASIS_COUNT)
    )
    return physical, pca, random


def _frozen_config() -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "target_dataset": "MVTec AD2 single-image",
        "population": "exact SuperAD-16 train/good support only",
        "objects": list(OBJECTS),
        "layers": list(LAYERS),
        "rank": RANK,
        "fold_count": FOLD_COUNT,
        "fold_assignment": "SHA256(data-root-relative-path), sorted, modulo 4",
        "axes": list(AXES),
        "training_axis_amplitudes": dict(TRAIN_AXIS_AMPLITUDES),
        "validation_amplitude_scale": VALIDATION_AMPLITUDE_SCALE,
        "compounds": list(COMPOUNDS),
        "defects": list(DEFECT_KINDS),
        "random_basis_count": RANDOM_BASIS_COUNT,
        "basis_bootstraps": BASIS_BOOTSTRAPS,
        "macro_bootstraps": MACRO_BOOTSTRAPS,
        "resolutions": dict(RESOLUTIONS),
        "determinism": "SHA256 protocol/object/data-relative-path/component",
        "query_or_test_access": False,
    }


def expected_audit_counts() -> dict[str, int]:
    arms = 3 + RANDOM_BASIS_COUNT  # raw, physical, clean PCA, random seeds
    shift_rows = SUPPORT_COUNT * len(LAYERS) * (len(AXES) + len(COMPOUNDS))
    defect_rows = SUPPORT_COUNT * len(LAYERS) * len(DEFECT_KINDS) * 2
    return {
        "arms_per_case": arms,
        "shift_rows": shift_rows,
        "defect_rows": defect_rows,
        "shift_maps": shift_rows * arms,
        "defect_maps": defect_rows * arms,
        "total_maps": (shift_rows + defect_rows) * arms,
    }


def run_object(  # noqa: C901, PLR0912, PLR0915
    args: argparse.Namespace,
    backbone: DINOv2Backbone,
    entries: Sequence[SupportEntry],
    object_name: str,
) -> dict[str, Any]:
    backbone.set_resolution(RESOLUTIONS[object_name])
    clean_features: dict[tuple[str, int], np.ndarray] = {}
    clean_images: dict[str, np.ndarray] = {}
    grids: dict[str, tuple[int, int]] = {}
    central: dict[tuple[str, int], np.ndarray] = {}

    for entry in entries:
        image = read_rgb(entry.path)
        clean_images[entry.relative_path] = image
        layers, grid = extract_layers(backbone, image)
        grids[entry.relative_path] = grid
        for layer, features in layers.items():
            clean_features[(entry.relative_path, layer)] = features
        observations_by_layer: dict[int, list[np.ndarray]] = {layer: [] for layer in LAYERS}
        for axis in AXES:
            negative, _ = extract_layers(
                backbone,
                apply_rgb_transform(
                    image,
                    make_axis_transform(axis, sign=-1, amplitude_scale=1.0),
                ),
            )
            positive, _ = extract_layers(
                backbone,
                apply_rgb_transform(
                    image,
                    make_axis_transform(axis, sign=+1, amplitude_scale=1.0),
                ),
            )
            for layer in LAYERS:
                observations_by_layer[layer].append(
                    central_difference_observation(negative[layer], positive[layer]),
                )
        for layer in LAYERS:
            central[(entry.relative_path, layer)] = np.stack(observations_by_layer[layer])

    full_bases: dict[int, np.ndarray] = {}
    stability_rows: list[dict[str, Any]] = []
    for layer in LAYERS:
        observations_by_image = [central[(entry.relative_path, layer)] for entry in entries]
        full_bases[layer] = fit_image_level_basis(
            np.concatenate(observations_by_image, axis=0),
            rank=RANK,
            center=False,
        )
        values = _basis_bootstrap_stability(
            observations_by_image,
            full_bases[layer],
            object_name=object_name,
            layer=layer,
            replicates=args.basis_bootstraps,
        )
        stability_rows.extend(
            {
                "object": object_name,
                "layer": layer,
                "replicate": replicate,
                "stability": value,
            }
            for replicate, value in enumerate(values)
        )

    shift_rows: list[dict[str, Any]] = []
    defect_rows: list[dict[str, Any]] = []
    packed_masks: dict[str, np.ndarray] = {}
    maps_for_coverage: list[np.ndarray] = []
    basis_hash_rows: list[dict[str, Any]] = []
    for fold in range(FOLD_COUNT):
        fold_bases: dict[int, tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]] = {}
        for layer in LAYERS:
            fold_bases[layer] = _fit_fold_bases(
                central,
                clean_features,
                entries,
                fold,
                layer,
            )
            physical, pca, random = fold_bases[layer]
            basis_hash_rows.append(
                {
                    "object": object_name,
                    "fold": fold,
                    "layer": layer,
                    "physical_sha256": sha256_bytes(physical.tobytes()),
                    "clean_pca_sha256": sha256_bytes(pca.tobytes()),
                    "random_sha256": sha256_bytes(b"".join(item.tobytes() for item in random)),
                },
            )
        for entry in (item for item in entries if item.fold == fold):
            image = clean_images[entry.relative_path]
            grid = grids[entry.relative_path]
            validation_feature_cache: dict[str, dict[int, np.ndarray]] = {}
            for transform_name, spec in validation_specs(object_name, entry.relative_path):
                shifted, shifted_grid = extract_layers(backbone, apply_rgb_transform(image, spec))
                if shifted_grid != grid:
                    raise RuntimeError("photometric validation changed the feature grid")
                validation_feature_cache[transform_name] = shifted
                for layer in LAYERS:
                    base = clean_features[(entry.relative_path, layer)]
                    residual = shifted[layer] - base
                    delta = residual.astype(np.float64).mean(axis=0)
                    physical, pca, random_bases = fold_bases[layer]
                    raw_scores = score_raw_residuals(residual)
                    physical_scores = score_patchwise_residuals(residual, physical)
                    pca_scores = score_patchwise_residuals(residual, pca)
                    random_scores = [
                        score_patchwise_residuals(residual, basis) for basis in random_bases
                    ]
                    raw_p95 = float(np.percentile(raw_scores, 95))
                    if raw_p95 <= 0.0:
                        raise RuntimeError("held-out shift produced zero raw p95")
                    random_capture = [
                        subspace_energy_coverage(delta, basis) for basis in random_bases
                    ]
                    random_drift = [
                        1.0 - float(np.percentile(scores, 95)) / raw_p95 for scores in random_scores
                    ]
                    row = {
                        "object": object_name,
                        "relative_path": entry.relative_path,
                        "fold": fold,
                        "layer": layer,
                        "transform": transform_name,
                        "physical_energy_capture": subspace_energy_coverage(delta, physical),
                        "clean_pca_energy_capture": subspace_energy_coverage(delta, pca),
                        "random_energy_capture_q95": float(np.percentile(random_capture, 95)),
                        "raw_p95": raw_p95,
                        "physical_p95": float(np.percentile(physical_scores, 95)),
                        "clean_pca_p95": float(np.percentile(pca_scores, 95)),
                        "random_p95_q05": float(
                            np.percentile(
                                [np.percentile(scores, 95) for scores in random_scores],
                                5,
                            ),
                        ),
                        "physical_drift_reduction": 1.0
                        - float(np.percentile(physical_scores, 95)) / raw_p95,
                        "clean_pca_drift_reduction": 1.0
                        - float(np.percentile(pca_scores, 95)) / raw_p95,
                        "random_drift_reduction_q95": float(np.percentile(random_drift, 95)),
                    }
                    row.update(random_null_columns(random_capture, random_drift))
                    shift_rows.append(row)
                    maps_for_coverage.extend(
                        [
                            raw_scores.reshape(grid),
                            physical_scores.reshape(grid),
                            pca_scores.reshape(grid),
                            *(scores.reshape(grid) for scores in random_scores),
                        ],
                    )

            for kind in DEFECT_KINDS:
                shift_name, shift_spec = assigned_defect_shift(
                    object_name,
                    entry.relative_path,
                    kind,
                )
                shifted_clean_features = validation_feature_cache[shift_name]
                selected_variants: list[tuple[str, np.ndarray, dict[int, np.ndarray]]] = []
                for feature_retry in range(DEFECT_GENERATION_ATTEMPTS):
                    defective, mask, defect_metadata = make_synthetic_defect(
                        image,
                        object_name=object_name,
                        relative_path=entry.relative_path,
                        kind=kind,
                        feature_retry=feature_retry,
                    )
                    candidate_variants = (
                        ("clean", defective),
                        ("shifted", apply_rgb_transform(defective, shift_spec)),
                    )
                    extracted_variants: list[tuple[str, np.ndarray, dict[int, np.ndarray]]] = []
                    raw_nonconstant = True
                    for acquisition, defect_image in candidate_variants:
                        defect_features, defect_grid = extract_layers(backbone, defect_image)
                        if defect_grid != grid:
                            raise RuntimeError("synthetic defect changed the feature grid")
                        extracted_variants.append(
                            (acquisition, defect_image, defect_features),
                        )
                        for layer in LAYERS:
                            residual = defect_feature_residual(
                                defect_features[layer],
                                acquisition=acquisition,
                                clean_features=clean_features[(entry.relative_path, layer)],
                                shifted_clean_features=shifted_clean_features[layer],
                            )
                            raw_scores = score_raw_residuals(residual)
                            if float(np.ptp(raw_scores)) <= 0.0:
                                raw_nonconstant = False
                    if raw_nonconstant:
                        selected_variants = extracted_variants
                        break
                if not selected_variants:
                    message = (
                        f"{object_name}/{entry.relative_path}/{kind}: "
                        "could not construct a feature-visible defect"
                    )
                    raise RuntimeError(message)
                mask_key = f"{entry.path_sha256[:16]}__{kind}"
                packed_masks[mask_key] = np.packbits(mask.reshape(-1))
                for acquisition, defect_image, defect_features in selected_variants:
                    for layer in LAYERS:
                        residual = defect_feature_residual(
                            defect_features[layer],
                            acquisition=acquisition,
                            clean_features=clean_features[(entry.relative_path, layer)],
                            shifted_clean_features=shifted_clean_features[layer],
                        )
                        physical, pca, random_bases = fold_bases[layer]
                        raw_scores = score_raw_residuals(residual)
                        physical_scores = score_patchwise_residuals(residual, physical)
                        pca_scores = score_patchwise_residuals(residual, pca)
                        random_scores = [
                            score_patchwise_residuals(residual, basis) for basis in random_bases
                        ]
                        patch_mask = mask_to_patch_grid(mask, grid)
                        raw_metrics = pixel_metrics(patch_mask, raw_scores)
                        physical_metrics = pixel_metrics(patch_mask, physical_scores)
                        raw_energy = float(np.sum(raw_scores[patch_mask]))
                        physical_energy = float(np.sum(physical_scores[patch_mask]))
                        retention = physical_energy / max(raw_energy, 1e-12)
                        defect_rows.append(
                            {
                                "object": object_name,
                                "relative_path": entry.relative_path,
                                "fold": fold,
                                "layer": layer,
                                "defect": kind,
                                "acquisition": acquisition,
                                "assigned_shift": shift_name
                                if acquisition == "shifted"
                                else "none",
                                "defect_seed": defect_metadata["seed"],
                                "feature_retry": defect_metadata["feature_retry"],
                                "generation_attempt": defect_metadata["generation_attempt"],
                                "defect_image_sha256": defect_metadata["image_sha256"],
                                "evaluated_image_sha256": sha256_bytes(defect_image.tobytes()),
                                "mask_key": mask_key,
                                "mask_height": mask.shape[0],
                                "mask_width": mask.shape[1],
                                "mask_sha256": defect_metadata["mask_sha256"],
                                "raw_seg_AUROC_0.05": raw_metrics["seg_AUROC_0.05"],
                                "physical_seg_AUROC_0.05": physical_metrics["seg_AUROC_0.05"],
                                "delta_seg_AUROC_0.05": physical_metrics["seg_AUROC_0.05"]
                                - raw_metrics["seg_AUROC_0.05"],
                                "raw_seg_F1": raw_metrics["seg_F1"],
                                "physical_seg_F1": physical_metrics["seg_F1"],
                                "delta_seg_F1": physical_metrics["seg_F1"] - raw_metrics["seg_F1"],
                                "raw_anomaly_energy": raw_energy,
                                "physical_anomaly_energy": physical_energy,
                                "anomaly_energy_retention": retention,
                            },
                        )
                        # G4 and G5 are feature-grid diagnostics.  Keep only
                        # patch arrays; no dense score map is materialized.
                        maps_for_coverage.extend(
                            [
                                raw_scores,
                                physical_scores,
                                pca_scores,
                                *random_scores,
                            ],
                        )

    expected = expected_audit_counts()
    expected_arms = expected["arms_per_case"]
    expected_shift_rows = expected["shift_rows"]
    expected_defect_rows = expected["defect_rows"]
    expected_shift_maps = expected["shift_maps"]
    expected_defect_maps = expected["defect_maps"]
    expected_total_maps = expected["total_maps"]
    if len(shift_rows) != expected_shift_rows or len(defect_rows) != expected_defect_rows:
        raise RuntimeError("support/fold/transform/defect row parity failed")
    if len(maps_for_coverage) != expected_total_maps:
        raise RuntimeError("physical/null score-map arm parity failed")
    if len(packed_masks) != SUPPORT_COUNT * len(DEFECT_KINDS):
        raise RuntimeError("sealed synthetic-mask retention parity failed")
    coverage = summarize_map_coverage(maps_for_coverage)
    output_root = Path(args.output_root)
    paths = {
        "shifts": output_root / f"{object_name}_shifts.tsv",
        "stability": output_root / f"{object_name}_stability.tsv",
        "defects": output_root / f"{object_name}_defects.tsv",
        "bases": output_root / f"{object_name}_basis_hashes.tsv",
        "support": output_root / f"{object_name}_support.json",
        "masks": output_root / f"{object_name}_synthetic_masks.npz",
    }
    _write_tsv(paths["shifts"], shift_rows)
    _write_tsv(paths["stability"], stability_rows)
    _write_tsv(paths["defects"], defect_rows)
    _write_tsv(paths["bases"], basis_hash_rows)
    np.savez_compressed(paths["masks"], **packed_masks)
    paths["support"].write_text(
        json.dumps(
            [
                {
                    **asdict(entry),
                    "path": str(entry.path),
                }
                for entry in entries
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    physical_capture = _mean(row["physical_energy_capture"] for row in shift_rows)
    pca_capture = _mean(row["clean_pca_energy_capture"] for row in shift_rows)
    random_capture = _mean(row["random_energy_capture_q95"] for row in shift_rows)
    physical_drift = _mean(row["physical_drift_reduction"] for row in shift_rows)
    pca_drift = _mean(row["clean_pca_drift_reduction"] for row in shift_rows)
    random_drift = _mean(row["random_drift_reduction_q95"] for row in shift_rows)
    stability_values = [row["stability"] for row in stability_rows]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "evaluation_role": "support_only_synthetic_identifiability_preflight",
        "maximum_verdict": "PROMISING_DIAGNOSTIC",
        "object": object_name,
        "data_root": str(Path(args.data_root)),
        "population": "train/good exact support-16 only",
        "test_public_accessed": False,
        "test_private_accessed": False,
        "support_count": len(entries),
        "fold_counts": np.bincount([entry.fold for entry in entries], minlength=4).tolist(),
        "resolution": RESOLUTIONS[object_name],
        "backbone": args.backbone_model,
        "layers": list(LAYERS),
        "rank": RANK,
        "g1_ingredients": {
            "physical_energy_capture_mean": physical_capture,
            "clean_pca_energy_capture_mean": pca_capture,
            "random_energy_capture_q95_mean": random_capture,
            "physical_margin_over_hard_null": physical_capture - max(pca_capture, random_capture),
        },
        "g2_ingredients": {
            "bootstrap_count_per_layer": args.basis_bootstraps,
            "stability_median": float(np.median(stability_values)),
            "stability_p05": float(np.percentile(stability_values, 5)),
        },
        "g3_ingredients": {
            "physical_drift_reduction_mean": physical_drift,
            "clean_pca_drift_reduction_mean": pca_drift,
            "random_drift_reduction_q95_mean": random_drift,
            "physical_margin_over_hard_null": physical_drift - max(pca_drift, random_drift),
        },
        "g4_ingredients": {
            "delta_seg_AUROC_0.05_mean": _mean(row["delta_seg_AUROC_0.05"] for row in defect_rows),
            "delta_seg_F1_mean": _mean(row["delta_seg_F1"] for row in defect_rows),
            "delta_seg_AUROC_0.05_row_min": float(
                min(row["delta_seg_AUROC_0.05"] for row in defect_rows),
            ),
            "delta_seg_F1_row_min": float(min(row["delta_seg_F1"] for row in defect_rows)),
            "anomaly_energy_retention_row_mean_diagnostic": _mean(
                row["anomaly_energy_retention"] for row in defect_rows
            ),
            "anomaly_energy_retention_ratio_of_sums": (
                float(sum(row["physical_anomaly_energy"] for row in defect_rows))
                / max(float(sum(row["raw_anomaly_energy"] for row in defect_rows)), 1e-12)
            ),
            "clean_row_count": sum(row["acquisition"] == "clean" for row in defect_rows),
            "shifted_row_count": sum(row["acquisition"] == "shifted" for row in defect_rows),
            "metric_grid": "feature_grid_with_any_native_mask_overlap",
        },
        "g5_ingredients": {
            **asdict(coverage),
            "complete": coverage.complete,
            "finite_value_fraction": coverage.finite_value_fraction,
            "expected_arms_per_case": expected_arms,
            "observed_arms_per_case": (coverage.total_maps // (len(shift_rows) + len(defect_rows))),
            "expected_shift_row_count": expected_shift_rows,
            "shift_row_count": len(shift_rows),
            "expected_shift_map_count": expected_shift_maps,
            "expected_defect_row_count": expected_defect_rows,
            "defect_row_count": len(defect_rows),
            "expected_defect_map_count": expected_defect_maps,
            "expected_retained_mask_count": SUPPORT_COUNT * len(DEFECT_KINDS),
            "retained_mask_count": len(packed_masks),
            "expected_total_maps": expected_total_maps,
            "parity_complete": coverage.total_maps == expected_total_maps,
        },
        "artifacts": {name: str(path) for name, path in paths.items()},
        "artifact_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "host_gpu_id": str(args.host_gpu_id),
        "container_cuda_slot": str(args.container_cuda_slot),
    }
    summary_path = output_root / f"{object_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for remote Stage -1 execution")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    support_path = Path(args.support_json).resolve(strict=True)
    support = load_support_manifest(support_path, Path(args.data_root))
    config = _frozen_config()
    config_path = output_root / "frozen_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
        "runner_sha256": sha256_file(Path(__file__)),
        "core_sha256": sha256_file(_ROOT / "src" / "flow_tte" / "cfntr.py"),
        "config_sha256": sha256_file(config_path),
        "support_manifest_sha256": sha256_file(support_path),
        "summaries": [f"{summary['object']}_summary.json" for summary in summaries],
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
