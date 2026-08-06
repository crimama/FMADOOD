#!/usr/bin/env python3
"""Audit and aggregate the all-eight CF-NTR Stage -1 compact artifacts.

The aggregator is support-only: it discovers only explicitly named compact
artifacts and never enumerates a dataset directory.  Bootstrap sampling is at
the support-image level within each object; every sampled image carries all of
its layer, transform, and synthetic-defect child rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

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
DEFECTS = ("cutpaste", "chromatic_blotch", "gaussian_blur", "scratch")
ACQUISITIONS = ("clean", "shifted")
PROTOCOL_ID = "cfntr-stage-minus1-support-only-20260718"
SUPPORT_COUNT = 16
FOLD_COUNT = 4
RANK = 4
RANDOM_BASIS_COUNT = 16
BASIS_BOOTSTRAPS = 200
MACRO_BOOTSTRAPS = 2000
FROZEN_SUPPORT_MANIFEST_SHA256 = "7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3"

SHIFT_BASE_FLOAT_FIELDS = (
    "physical_energy_capture",
    "clean_pca_energy_capture",
    "random_energy_capture_q95",
    "raw_p95",
    "physical_p95",
    "clean_pca_p95",
    "random_p95_q05",
    "physical_drift_reduction",
    "clean_pca_drift_reduction",
    "random_drift_reduction_q95",
)
RANDOM_CAPTURE_FIELDS = tuple(
    f"random_energy_capture_{index:02d}" for index in range(RANDOM_BASIS_COUNT)
)
RANDOM_DRIFT_FIELDS = tuple(
    f"random_drift_reduction_{index:02d}" for index in range(RANDOM_BASIS_COUNT)
)
SHIFT_FLOAT_FIELDS = SHIFT_BASE_FLOAT_FIELDS + RANDOM_CAPTURE_FIELDS + RANDOM_DRIFT_FIELDS
DEFECT_FLOAT_FIELDS = (
    "raw_seg_AUROC_0.05",
    "physical_seg_AUROC_0.05",
    "delta_seg_AUROC_0.05",
    "raw_seg_F1",
    "physical_seg_F1",
    "delta_seg_F1",
    "raw_anomaly_energy",
    "physical_anomaly_energy",
    "anomaly_energy_retention",
)


class AuditError(RuntimeError):
    """Raised when compact artifacts cannot satisfy the frozen G5 audit."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in (PROTOCOL_ID, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def deterministic_sign(*parts: object) -> int:
    return 1 if stable_seed(*parts) & 1 else -1


def validation_names(object_name: str, relative_path: str) -> tuple[str, ...]:
    names = [
        f"{axis}_{deterministic_sign(object_name, relative_path, axis, 'validation'):+d}_x2"
        for axis in AXES
    ]
    for compound in COMPOUNDS:
        first = deterministic_sign(object_name, relative_path, compound, "first")
        second = deterministic_sign(object_name, relative_path, compound, "second")
        names.append(f"{compound}_{first:+d}_{second:+d}")
    return tuple(names)


def assigned_defect_shift(object_name: str, relative_path: str, defect: str) -> str:
    names = validation_names(object_name, relative_path)
    index = stable_seed(object_name, relative_path, defect, "assigned-shift") % len(names)
    return names[index]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"cannot read JSON artifact {path}: {error}"
        raise AuditError(msg) from error


def _load_tsv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))
    except (OSError, csv.Error) as error:
        msg = f"cannot read TSV artifact {path}: {error}"
        raise AuditError(msg) from error


def _float(row: Mapping[str, str], field: str, context: str) -> float:
    _require(field in row and row[field] != "", f"{context}: missing {field}")
    try:
        value = float(row[field])
    except ValueError as error:
        msg = f"{context}: nonnumeric {field}"
        raise AuditError(msg) from error
    _require(np.isfinite(value), f"{context}: non-finite {field}")
    return value


def _integer(row: Mapping[str, str], field: str, context: str) -> int:
    _require(field in row, f"{context}: missing {field}")
    try:
        value = int(row[field])
    except ValueError as error:
        msg = f"{context}: noninteger {field}"
        raise AuditError(msg) from error
    return value


def _artifact_path(summary_path: Path, recorded: object, expected_name: str) -> Path:
    _require(isinstance(recorded, str), f"{summary_path}: invalid artifact path")
    _require(Path(recorded).name == expected_name, f"{summary_path}: artifact basename mismatch")
    path = summary_path.parent / expected_name
    _require(path.is_file(), f"{summary_path}: missing artifact {expected_name}")
    return path


def _expected_config() -> dict[str, Any]:
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
        "training_axis_amplitudes": {
            "exposure": 0.25,
            "gamma": 0.10,
            "contrast": 0.10,
            "white_balance": 0.08,
        },
        "validation_amplitude_scale": 2.0,
        "compounds": list(COMPOUNDS),
        "defects": list(DEFECTS),
        "random_basis_count": RANDOM_BASIS_COUNT,
        "basis_bootstraps": BASIS_BOOTSTRAPS,
        "macro_bootstraps": MACRO_BOOTSTRAPS,
        "resolutions": {name: (448 if name == "sheet_metal" else 672) for name in OBJECTS},
        "determinism": "SHA256 protocol/object/data-relative-path/component",
        "query_or_test_access": False,
    }


def _audit_support(path: Path, object_name: str) -> tuple[list[str], dict[str, int]]:
    payload = _load_json(path)
    _require(
        isinstance(payload, list) and len(payload) == SUPPORT_COUNT,
        f"{object_name}: support must contain exactly 16 entries",
    )
    paths: list[str] = []
    folds: dict[str, int] = {}
    path_hashes: dict[str, str] = {}
    for index, item in enumerate(payload):
        context = f"{object_name}: support[{index}]"
        _require(isinstance(item, dict), f"{context}: entry must be an object")
        relative = item.get("relative_path")
        _require(isinstance(relative, str), f"{context}: missing relative_path")
        parts = Path(relative).parts
        _require(
            len(parts) == 4 and tuple(parts[:3]) == (object_name, "train", "good"),
            f"{context}: support path is not exact train/good",
        )
        lower_paths = f"{relative}/{item.get('path', '')}".lower()
        _require(
            "test_public" not in lower_paths and "ground_truth" not in lower_paths,
            f"{context}: forbidden public path token",
        )
        path_hash = hashlib.sha256(relative.encode("utf-8")).hexdigest()
        _require(item.get("path_sha256") == path_hash, f"{context}: path hash mismatch")
        _require(_is_sha256(item.get("content_sha256")), f"{context}: invalid content hash")
        fold = item.get("fold")
        _require(
            isinstance(fold, int) and not isinstance(fold, bool) and 0 <= fold < FOLD_COUNT,
            f"{context}: invalid fold",
        )
        _require(relative not in folds, f"{object_name}: duplicate support path")
        paths.append(relative)
        folds[relative] = fold
        path_hashes[relative] = path_hash
    ordered = sorted(paths, key=lambda item: (path_hashes[item], item))
    expected_folds = {relative: index % FOLD_COUNT for index, relative in enumerate(ordered)}
    _require(folds == expected_folds, f"{object_name}: SHA256 fold assignment mismatch")
    _require(
        [sum(value == fold for value in folds.values()) for fold in range(FOLD_COUNT)] == [4] * 4,
        f"{object_name}: fold counts must be 4/4/4/4",
    )
    return paths, folds


def _audit_shift_rows(
    rows: list[dict[str, str]],
    object_name: str,
    paths: Sequence[str],
    folds: Mapping[str, int],
) -> dict[str, np.ndarray]:
    expected_count = SUPPORT_COUNT * len(LAYERS) * (len(AXES) + len(COMPOUNDS))
    _require(len(rows) == expected_count, f"{object_name}: expected {expected_count} shift rows")
    keys: set[tuple[str, int, str]] = set()
    values_by_path: dict[str, list[list[float]]] = {path: [] for path in paths}
    for index, row in enumerate(rows):
        context = f"{object_name}: shift[{index}]"
        _require(row.get("object") == object_name, f"{context}: object mismatch")
        relative = row.get("relative_path", "")
        _require(relative in folds, f"{context}: unknown support image")
        fold = _integer(row, "fold", context)
        layer = _integer(row, "layer", context)
        transform = row.get("transform", "")
        _require(fold == folds[relative], f"{context}: fold mismatch")
        _require(layer in LAYERS, f"{context}: invalid layer")
        _require(
            transform in validation_names(object_name, relative),
            f"{context}: invalid sealed transform",
        )
        key = (relative, layer, transform)
        _require(key not in keys, f"{context}: duplicate parity key")
        keys.add(key)
        numeric = [_float(row, field, context) for field in SHIFT_FLOAT_FIELDS]
        by_field = dict(zip(SHIFT_FLOAT_FIELDS, numeric))
        for field in (
            "physical_energy_capture",
            "clean_pca_energy_capture",
            *RANDOM_CAPTURE_FIELDS,
        ):
            _require(0.0 <= by_field[field] <= 1.0, f"{context}: {field} outside [0,1]")
        _require(
            by_field["raw_p95"] > 0.0
            and by_field["physical_p95"] >= 0.0
            and by_field["clean_pca_p95"] >= 0.0
            and by_field["random_p95_q05"] >= 0.0,
            f"{context}: invalid residual p95",
        )
        _require(
            np.isclose(
                by_field["physical_drift_reduction"],
                1.0 - by_field["physical_p95"] / by_field["raw_p95"],
                rtol=1e-10,
                atol=1e-12,
            )
            and np.isclose(
                by_field["clean_pca_drift_reduction"],
                1.0 - by_field["clean_pca_p95"] / by_field["raw_p95"],
                rtol=1e-10,
                atol=1e-12,
            ),
            f"{context}: drift-reduction equation mismatch",
        )
        _require(
            all(by_field[field] <= 1.0 for field in RANDOM_DRIFT_FIELDS),
            f"{context}: random drift reduction exceeds one",
        )
        _require(
            np.isclose(
                by_field["random_energy_capture_q95"],
                np.percentile([by_field[field] for field in RANDOM_CAPTURE_FIELDS], 95),
                rtol=1e-10,
                atol=1e-12,
            )
            and np.isclose(
                by_field["random_drift_reduction_q95"],
                np.percentile([by_field[field] for field in RANDOM_DRIFT_FIELDS], 95),
                rtol=1e-10,
                atol=1e-12,
            )
            and np.isclose(
                by_field["random_drift_reduction_q95"],
                1.0 - by_field["random_p95_q05"] / by_field["raw_p95"],
                rtol=1e-10,
                atol=1e-12,
            ),
            f"{context}: random q95 envelope mismatch",
        )
        values_by_path[relative].append(numeric)
    for relative in paths:
        expected = {
            (relative, layer, transform)
            for layer in LAYERS
            for transform in validation_names(object_name, relative)
        }
        _require(expected.issubset(keys), f"{object_name}: incomplete shift parity for {relative}")
    return {
        relative: np.asarray(values_by_path[relative], dtype=np.float64).mean(axis=0)
        for relative in paths
    }


def _audit_defect_rows(  # noqa: PLR0915
    rows: list[dict[str, str]],
    object_name: str,
    paths: Sequence[str],
    folds: Mapping[str, int],
) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, tuple[str, int, int]]]:
    expected_count = SUPPORT_COUNT * len(LAYERS) * len(DEFECTS) * len(ACQUISITIONS)
    _require(len(rows) == expected_count, f"{object_name}: expected {expected_count} defect rows")
    keys: set[tuple[str, int, str, str]] = set()
    values_by_path: dict[str, list[list[float]]] = {path: [] for path in paths}
    acquisition_counts = dict.fromkeys(ACQUISITIONS, 0)
    mask_hashes: dict[tuple[str, str], str] = {}
    mask_specs: dict[str, tuple[str, int, int]] = {}
    for index, row in enumerate(rows):
        context = f"{object_name}: defect[{index}]"
        _require(row.get("object") == object_name, f"{context}: object mismatch")
        relative = row.get("relative_path", "")
        _require(relative in folds, f"{context}: unknown support image")
        fold = _integer(row, "fold", context)
        layer = _integer(row, "layer", context)
        defect = row.get("defect", "")
        acquisition = row.get("acquisition", "")
        _require(fold == folds[relative], f"{context}: fold mismatch")
        _require(
            layer in LAYERS and defect in DEFECTS and acquisition in ACQUISITIONS,
            f"{context}: invalid defect parity fields",
        )
        expected_shift = (
            "none"
            if acquisition == "clean"
            else assigned_defect_shift(
                object_name,
                relative,
                defect,
            )
        )
        _require(row.get("assigned_shift") == expected_shift, f"{context}: assigned shift mismatch")
        mask_hash = row.get("mask_sha256")
        _require(_is_sha256(mask_hash), f"{context}: invalid mask hash")
        expected_mask_key = f"{hashlib.sha256(relative.encode('utf-8')).hexdigest()[:16]}__{defect}"
        _require(row.get("mask_key") == expected_mask_key, f"{context}: mask key mismatch")
        height = _integer(row, "mask_height", context)
        width = _integer(row, "mask_width", context)
        _require(height > 0 and width > 0, f"{context}: invalid mask shape")
        _require(
            _integer(row, "defect_seed", context)
            == stable_seed(
                object_name,
                relative,
                defect,
                "defect",
            ),
            f"{context}: defect seed mismatch",
        )
        _require(
            _is_sha256(row.get("defect_image_sha256"))
            and _is_sha256(row.get("evaluated_image_sha256")),
            f"{context}: invalid defect image hash",
        )
        mask_key = (relative, defect)
        if mask_key in mask_hashes:
            _require(
                mask_hashes[mask_key] == mask_hash,
                f"{context}: mask hash changed across arms",
            )
        else:
            mask_hashes[mask_key] = str(mask_hash)
        packed_spec = (str(mask_hash), height, width)
        if expected_mask_key in mask_specs:
            _require(
                mask_specs[expected_mask_key] == packed_spec,
                f"{context}: mask metadata changed across arms",
            )
        else:
            mask_specs[expected_mask_key] = packed_spec
        key = (relative, layer, defect, acquisition)
        _require(key not in keys, f"{context}: duplicate defect parity key")
        keys.add(key)
        numeric = [_float(row, field, context) for field in DEFECT_FLOAT_FIELDS]
        numeric_by_field = dict(zip(DEFECT_FLOAT_FIELDS, numeric))
        for field in (
            "raw_seg_AUROC_0.05",
            "physical_seg_AUROC_0.05",
            "raw_seg_F1",
            "physical_seg_F1",
        ):
            _require(0.0 <= numeric_by_field[field] <= 1.0, f"{context}: {field} outside [0,1]")
        _require(
            np.isclose(
                numeric_by_field["delta_seg_AUROC_0.05"],
                numeric_by_field["physical_seg_AUROC_0.05"]
                - numeric_by_field["raw_seg_AUROC_0.05"],
                rtol=1e-10,
                atol=1e-12,
            )
            and np.isclose(
                numeric_by_field["delta_seg_F1"],
                numeric_by_field["physical_seg_F1"] - numeric_by_field["raw_seg_F1"],
                rtol=1e-10,
                atol=1e-12,
            ),
            f"{context}: defect delta equation mismatch",
        )
        _require(
            numeric_by_field["raw_anomaly_energy"] >= 0.0
            and numeric_by_field["physical_anomaly_energy"] >= 0.0,
            f"{context}: invalid anomaly energy",
        )
        expected_retention = numeric_by_field["physical_anomaly_energy"] / max(
            numeric_by_field["raw_anomaly_energy"],
            1e-12,
        )
        _require(
            np.isclose(
                numeric_by_field["anomaly_energy_retention"],
                expected_retention,
                rtol=1e-10,
                atol=1e-12,
            ),
            f"{context}: anomaly energy retention mismatch",
        )
        values_by_path[relative].append(numeric)
        acquisition_counts[acquisition] += 1
    per_acquisition = SUPPORT_COUNT * len(LAYERS) * len(DEFECTS)
    _require(
        all(count == per_acquisition for count in acquisition_counts.values()),
        f"{object_name}: incomplete clean/shifted defect coverage",
    )
    raw_energy_index = DEFECT_FLOAT_FIELDS.index("raw_anomaly_energy")
    physical_energy_index = DEFECT_FLOAT_FIELDS.index("physical_anomaly_energy")
    per_image: dict[str, np.ndarray] = {}
    for relative in paths:
        matrix = np.asarray(values_by_path[relative], dtype=np.float64)
        values = matrix.mean(axis=0, dtype=np.float64)
        values[raw_energy_index] = matrix[:, raw_energy_index].sum(dtype=np.float64)
        values[physical_energy_index] = matrix[:, physical_energy_index].sum(dtype=np.float64)
        per_image[relative] = values
    return per_image, acquisition_counts, mask_specs


def _audit_packed_masks(path: Path, specs: Mapping[str, tuple[str, int, int]]) -> None:
    _require(len(specs) == SUPPORT_COUNT * len(DEFECTS), "packed mask parity is incomplete")
    try:
        with np.load(path, allow_pickle=False) as payload:
            _require(set(payload.files) == set(specs), "packed mask keys do not match defect rows")
            for key, (expected_hash, height, width) in specs.items():
                packed = np.asarray(payload[key], dtype=np.uint8).reshape(-1)
                expected_bytes = (height * width + 7) // 8
                _require(packed.size == expected_bytes, f"{key}: packed mask size mismatch")
                mask = np.unpackbits(packed, count=height * width).astype(np.uint8)
                _require(
                    hashlib.sha256(mask.tobytes()).hexdigest() == expected_hash,
                    f"{key}: packed mask content hash mismatch",
                )
                _require(0 < int(mask.sum()) < mask.size, f"{key}: mask is empty or full")
    except (OSError, ValueError) as error:
        msg = f"cannot audit packed masks {path}: {error}"
        raise AuditError(msg) from error


def _audit_stability(rows: list[dict[str, str]], object_name: str) -> np.ndarray:
    expected_count = len(LAYERS) * BASIS_BOOTSTRAPS
    _require(
        len(rows) == expected_count,
        f"{object_name}: expected {expected_count} stability rows",
    )
    keys: set[tuple[int, int]] = set()
    values: list[float] = []
    for index, row in enumerate(rows):
        context = f"{object_name}: stability[{index}]"
        _require(row.get("object") == object_name, f"{context}: object mismatch")
        layer = _integer(row, "layer", context)
        replicate = _integer(row, "replicate", context)
        _require(
            layer in LAYERS and 0 <= replicate < BASIS_BOOTSTRAPS,
            f"{context}: invalid layer/replicate",
        )
        _require((layer, replicate) not in keys, f"{context}: duplicate stability key")
        keys.add((layer, replicate))
        value = _float(row, "stability", context)
        _require(0.0 <= value <= 1.0, f"{context}: stability outside [0,1]")
        values.append(value)
    return np.asarray(values, dtype=np.float64)


def _audit_basis_hashes(rows: list[dict[str, str]], object_name: str) -> None:
    _require(len(rows) == FOLD_COUNT * len(LAYERS), f"{object_name}: incomplete basis-hash rows")
    keys: set[tuple[int, int]] = set()
    for index, row in enumerate(rows):
        context = f"{object_name}: basis[{index}]"
        _require(row.get("object") == object_name, f"{context}: object mismatch")
        fold = _integer(row, "fold", context)
        layer = _integer(row, "layer", context)
        _require(0 <= fold < FOLD_COUNT and layer in LAYERS, f"{context}: invalid fold/layer")
        _require((fold, layer) not in keys, f"{context}: duplicate basis key")
        keys.add((fold, layer))
        for field in ("physical_sha256", "clean_pca_sha256", "random_sha256"):
            _require(_is_sha256(row.get(field)), f"{context}: invalid {field}")


def _audit_object(
    summary_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    dict[str, int],
]:
    summary = _load_json(summary_path)
    _require(isinstance(summary, dict), f"{summary_path}: summary must be an object")
    object_name = summary.get("object")
    _require(object_name in OBJECTS, f"{summary_path}: invalid object")
    object_name = str(object_name)
    _require(summary.get("protocol_id") == PROTOCOL_ID, f"{object_name}: protocol mismatch")
    _require(
        summary.get("maximum_verdict") == "PROMISING_DIAGNOSTIC",
        f"{object_name}: invalid maximum verdict",
    )
    _require(
        summary.get("test_public_accessed") is False,
        f"{object_name}: public access audit failed",
    )
    _require(
        summary.get("test_private_accessed") is False,
        f"{object_name}: private access audit failed",
    )
    _require(
        summary.get("population") == "train/good exact support-16 only",
        f"{object_name}: population mismatch",
    )
    _require(
        summary.get("support_count") == SUPPORT_COUNT,
        f"{object_name}: support count mismatch",
    )
    _require(summary.get("fold_counts") == [4, 4, 4, 4], f"{object_name}: fold count mismatch")
    _require(
        summary.get("resolution") == (448 if object_name == "sheet_metal" else 672),
        f"{object_name}: resolution mismatch",
    )
    _require(summary.get("backbone") == "dinov2_vitl14", f"{object_name}: backbone mismatch")
    _require(
        summary.get("layers") == list(LAYERS) and summary.get("rank") == RANK,
        f"{object_name}: layer/rank mismatch",
    )
    artifacts = summary.get("artifacts")
    hashes = summary.get("artifact_sha256")
    _require(
        isinstance(artifacts, dict) and isinstance(hashes, dict),
        f"{object_name}: missing artifact audit maps",
    )
    artifact_paths: dict[str, Path] = {}
    expected_names = {
        "shifts": f"{object_name}_shifts.tsv",
        "stability": f"{object_name}_stability.tsv",
        "defects": f"{object_name}_defects.tsv",
        "bases": f"{object_name}_basis_hashes.tsv",
        "support": f"{object_name}_support.json",
        "masks": f"{object_name}_synthetic_masks.npz",
    }
    for key, filename in expected_names.items():
        path = _artifact_path(summary_path, artifacts.get(key), filename)
        _require(
            _is_sha256(hashes.get(key)) and sha256_file(path) == hashes[key],
            f"{object_name}: {key} artifact hash mismatch",
        )
        artifact_paths[key] = path
    paths, folds = _audit_support(artifact_paths["support"], object_name)
    shifts = _audit_shift_rows(_load_tsv(artifact_paths["shifts"]), object_name, paths, folds)
    defects, acquisition_counts, mask_specs = _audit_defect_rows(
        _load_tsv(artifact_paths["defects"]),
        object_name,
        paths,
        folds,
    )
    _audit_packed_masks(artifact_paths["masks"], mask_specs)
    stability = _audit_stability(_load_tsv(artifact_paths["stability"]), object_name)
    _audit_basis_hashes(_load_tsv(artifact_paths["bases"]), object_name)
    coverage = summary.get("g5_ingredients")
    _require(isinstance(coverage, dict), f"{object_name}: missing map coverage")
    _require(
        coverage.get("complete") is True and coverage.get("finite_value_fraction") == 1.0,
        f"{object_name}: finite/nonconstant map coverage failed",
    )
    _require(
        coverage.get("expected_total_maps") == 17024
        and coverage.get("parity_complete") is True
        and coverage.get("shift_row_count") == 384
        and coverage.get("defect_row_count") == 512
        and coverage.get("expected_arms_per_case") == 19
        and coverage.get("observed_arms_per_case") == 19
        and coverage.get("expected_shift_row_count") == 384
        and coverage.get("expected_defect_row_count") == 512
        and coverage.get("expected_shift_map_count") == 7296
        and coverage.get("expected_defect_map_count") == 9728
        and coverage.get("expected_retained_mask_count") == 64
        and coverage.get("retained_mask_count") == 64,
        f"{object_name}: G5 expected-count parity failed",
    )
    for field in ("total_maps", "finite_maps", "nonconstant_maps", "valid_maps"):
        _require(isinstance(coverage.get(field), int), f"{object_name}: missing {field}")
    _require(
        coverage["total_maps"] == 17024
        and coverage["total_maps"]
        == coverage["finite_maps"]
        == coverage["nonconstant_maps"]
        == coverage["valid_maps"],
        f"{object_name}: incomplete map coverage counts",
    )
    return summary, shifts, defects, stability, acquisition_counts


def _audit_shard_manifests(
    summary_paths: Sequence[Path],
    summaries: Mapping[str, dict[str, Any]],
    expected_support_hash: str,
) -> dict[str, Any]:
    parents = sorted({path.parent for path in summary_paths})
    runner_hashes: set[str] = set()
    core_hashes: set[str] = set()
    config_hashes: set[str] = set()
    support_hashes: set[str] = set()
    manifest_hashes: dict[str, str] = {}
    seen_objects: set[str] = set()
    for parent in parents:
        manifest_path = parent / "run_manifest.json"
        config_path = parent / "frozen_config.json"
        _require(
            manifest_path.is_file() and config_path.is_file(),
            f"{parent}: shard manifest/config missing",
        )
        manifest = _load_json(manifest_path)
        config = _load_json(config_path)
        _require(config == _expected_config(), f"{parent}: frozen config mismatch")
        _require(
            isinstance(manifest, dict) and manifest.get("protocol_id") == PROTOCOL_ID,
            f"{parent}: run manifest protocol mismatch",
        )
        _require(
            manifest.get("test_public_accessed") is False,
            f"{parent}: run manifest records public access",
        )
        _require(
            manifest.get("test_private_accessed") is False,
            f"{parent}: run manifest records private access",
        )
        for field, target in (
            ("runner_sha256", runner_hashes),
            ("core_sha256", core_hashes),
            ("config_sha256", config_hashes),
            ("support_manifest_sha256", support_hashes),
        ):
            value = manifest.get(field)
            _require(_is_sha256(value), f"{parent}: invalid {field}")
            target.add(str(value))
        _require(
            manifest["config_sha256"] == sha256_file(config_path),
            f"{parent}: config hash mismatch",
        )
        objects = manifest.get("objects")
        summary_hashes = manifest.get("summary_sha256")
        _require(
            isinstance(objects, list) and isinstance(summary_hashes, dict),
            f"{parent}: invalid shard object/hash maps",
        )
        expected_files = manifest.get("summaries")
        _require(isinstance(expected_files, list), f"{parent}: missing summary list")
        for object_name in objects:
            _require(
                object_name in OBJECTS and object_name not in seen_objects,
                f"{parent}: duplicate or invalid shard object",
            )
            path = parent / f"{object_name}_summary.json"
            _require(
                path in summary_paths and path.name in expected_files,
                f"{parent}: object summary parity mismatch",
            )
            value = summary_hashes.get(object_name)
            _require(
                _is_sha256(value) and value == sha256_file(path),
                f"{parent}: object summary hash mismatch",
            )
            _require(
                summaries[object_name].get("host_gpu_id") == manifest.get("host_gpu_id")
                and summaries[object_name].get("container_cuda_slot")
                == manifest.get("container_cuda_slot"),
                f"{parent}: GPU audit mismatch",
            )
            seen_objects.add(str(object_name))
        manifest_hashes[str(parent)] = sha256_file(manifest_path)
    _require(seen_objects == set(OBJECTS), "shard manifests do not cover exact all-eight objects")
    _require(
        len(runner_hashes) == len(core_hashes) == len(config_hashes) == len(support_hashes) == 1,
        "runner/core/config/support hashes differ across shards",
    )
    _require(
        next(iter(support_hashes)) == expected_support_hash,
        "support manifest hash is not the frozen SuperAD-16 manifest",
    )
    return {
        "runner_sha256": next(iter(runner_hashes)),
        "core_sha256": next(iter(core_hashes)),
        "config_sha256": next(iter(config_hashes)),
        "support_manifest_sha256": next(iter(support_hashes)),
        "run_manifest_sha256": manifest_hashes,
    }


def _mean_metric(image_values: Mapping[str, np.ndarray], index: int) -> float:
    return float(np.mean([value[index] for value in image_values.values()], dtype=np.float64))


def _energy_retention(physical: float, raw: float) -> float:
    """Return a conservative ratio when a sampled defect has no raw mask energy."""
    if raw <= 1e-12:
        return 0.0
    return physical / raw


def _object_row(
    object_name: str,
    shifts: Mapping[str, np.ndarray],
    defects: Mapping[str, np.ndarray],
    stability: np.ndarray,
) -> dict[str, Any]:
    shift_index = {field: index for index, field in enumerate(SHIFT_FLOAT_FIELDS)}
    defect_index = {field: index for index, field in enumerate(DEFECT_FLOAT_FIELDS)}
    row: dict[str, Any] = {
        "object": object_name,
        "physical_energy_capture": _mean_metric(shifts, shift_index["physical_energy_capture"]),
        "clean_pca_energy_capture": _mean_metric(shifts, shift_index["clean_pca_energy_capture"]),
        "physical_drift_reduction": _mean_metric(shifts, shift_index["physical_drift_reduction"]),
        "clean_pca_drift_reduction": _mean_metric(shifts, shift_index["clean_pca_drift_reduction"]),
        "delta_seg_AUROC_0.05": _mean_metric(defects, defect_index["delta_seg_AUROC_0.05"]),
        "delta_seg_F1": _mean_metric(defects, defect_index["delta_seg_F1"]),
        "stability_median": float(np.median(stability)),
        "stability_p05": float(np.percentile(stability, 5)),
    }
    raw_energy = _mean_metric(defects, defect_index["raw_anomaly_energy"])
    physical_energy = _mean_metric(defects, defect_index["physical_anomaly_energy"])
    row["raw_anomaly_energy_sum"] = raw_energy * len(defects)
    row["physical_anomaly_energy_sum"] = physical_energy * len(defects)
    row["anomaly_energy_retention"] = _energy_retention(physical_energy, raw_energy)
    random_capture = np.asarray(
        [_mean_metric(shifts, shift_index[field]) for field in RANDOM_CAPTURE_FIELDS],
    )
    random_drift = np.asarray(
        [_mean_metric(shifts, shift_index[field]) for field in RANDOM_DRIFT_FIELDS],
    )
    row["random_energy_capture_seed_q95"] = float(np.percentile(random_capture, 95))
    row["random_drift_reduction_seed_q95"] = float(np.percentile(random_drift, 95))
    return row


def _bootstrap_rows(
    shifts_by_object: Mapping[str, Mapping[str, np.ndarray]],
    defects_by_object: Mapping[str, Mapping[str, np.ndarray]],
    replicates: int,
) -> list[dict[str, float | int]]:
    shift_index = {field: index for index, field in enumerate(SHIFT_FLOAT_FIELDS)}
    defect_index = {field: index for index, field in enumerate(DEFECT_FLOAT_FIELDS)}
    output: list[dict[str, float | int]] = []
    for replicate in range(replicates):
        object_shift_means: list[np.ndarray] = []
        object_defect_means: list[np.ndarray] = []
        object_energy_retentions: list[float] = []
        for object_name in OBJECTS:
            ordered_paths = sorted(shifts_by_object[object_name])
            shift_matrix = np.stack([shifts_by_object[object_name][path] for path in ordered_paths])
            defect_matrix = np.stack(
                [defects_by_object[object_name][path] for path in ordered_paths],
            )
            rng = np.random.default_rng(stable_seed("macro-bootstrap", object_name, replicate))
            indices = rng.integers(0, SUPPORT_COUNT, size=SUPPORT_COUNT)
            object_shift_means.append(shift_matrix[indices].mean(axis=0, dtype=np.float64))
            defect_mean = defect_matrix[indices].mean(axis=0, dtype=np.float64)
            object_defect_means.append(defect_mean)
            object_energy_retentions.append(
                float(
                    _energy_retention(
                        defect_mean[defect_index["physical_anomaly_energy"]],
                        defect_mean[defect_index["raw_anomaly_energy"]],
                    ),
                ),
            )
        shift_macro = np.stack(object_shift_means).mean(axis=0, dtype=np.float64)
        defect_macro = np.stack(object_defect_means).mean(axis=0, dtype=np.float64)
        random_capture = shift_macro[[shift_index[field] for field in RANDOM_CAPTURE_FIELDS]]
        random_drift = shift_macro[[shift_index[field] for field in RANDOM_DRIFT_FIELDS]]
        physical_capture = float(shift_macro[shift_index["physical_energy_capture"]])
        pca_capture = float(shift_macro[shift_index["clean_pca_energy_capture"]])
        physical_drift = float(shift_macro[shift_index["physical_drift_reduction"]])
        pca_drift = float(shift_macro[shift_index["clean_pca_drift_reduction"]])
        capture_null = max(pca_capture, float(np.percentile(random_capture, 95)))
        drift_null = max(pca_drift, float(np.percentile(random_drift, 95)))
        output.append(
            {
                "replicate": replicate,
                "physical_energy_capture": physical_capture,
                "clean_pca_energy_capture": pca_capture,
                "random_energy_capture_seed_q95": float(np.percentile(random_capture, 95)),
                "g1_margin_over_hard_null": physical_capture - capture_null,
                "physical_drift_reduction": physical_drift,
                "clean_pca_drift_reduction": pca_drift,
                "random_drift_reduction_seed_q95": float(np.percentile(random_drift, 95)),
                "g3_margin_over_hard_null": physical_drift - drift_null,
                "delta_seg_AUROC_0.05": float(defect_macro[defect_index["delta_seg_AUROC_0.05"]]),
                "delta_seg_F1": float(defect_macro[defect_index["delta_seg_F1"]]),
                "anomaly_energy_retention": float(
                    np.mean(object_energy_retentions, dtype=np.float64),
                ),
            },
        )
    return output


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(rows), f"cannot write empty TSV {path}")
    fields = tuple(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(
    run_root: Path,
    output_root: Path,
    *,
    bootstrap_replicates: int = MACRO_BOOTSTRAPS,
    expected_support_manifest_sha256: str = FROZEN_SUPPORT_MANIFEST_SHA256,
) -> dict[str, Any]:
    _require(bootstrap_replicates > 0, "bootstrap_replicates must be positive")
    summary_paths = sorted(
        path
        for path in run_root.rglob("*_summary.json")
        if path.stem.endswith("_summary") and path.stem[: -len("_summary")] in OBJECTS
    )
    _require(len(summary_paths) == len(OBJECTS), "exactly eight object summaries are required")
    summaries: dict[str, dict[str, Any]] = {}
    shifts_by_object: dict[str, dict[str, np.ndarray]] = {}
    defects_by_object: dict[str, dict[str, np.ndarray]] = {}
    stability_by_object: dict[str, np.ndarray] = {}
    acquisition_counts: dict[str, dict[str, int]] = {}
    for path in summary_paths:
        summary, shifts, defects, stability, counts = _audit_object(path)
        object_name = str(summary["object"])
        _require(object_name not in summaries, f"duplicate summary for {object_name}")
        summaries[object_name] = summary
        shifts_by_object[object_name] = shifts
        defects_by_object[object_name] = defects
        stability_by_object[object_name] = stability
        acquisition_counts[object_name] = counts
    _require(set(summaries) == set(OBJECTS), "object summaries do not cover exact all-eight")
    source_hashes = _audit_shard_manifests(
        summary_paths,
        summaries,
        expected_support_manifest_sha256,
    )

    per_object = [
        _object_row(
            object_name,
            shifts_by_object[object_name],
            defects_by_object[object_name],
            stability_by_object[object_name],
        )
        for object_name in OBJECTS
    ]
    bootstraps = _bootstrap_rows(
        shifts_by_object,
        defects_by_object,
        bootstrap_replicates,
    )
    macro = {
        field: float(np.mean([float(row[field]) for row in per_object], dtype=np.float64))
        for field in (
            "physical_energy_capture",
            "clean_pca_energy_capture",
            "physical_drift_reduction",
            "clean_pca_drift_reduction",
            "delta_seg_AUROC_0.05",
            "delta_seg_F1",
            "anomaly_energy_retention",
        )
    }
    shift_index = {field: index for index, field in enumerate(SHIFT_FLOAT_FIELDS)}
    random_capture_macro = np.asarray(
        [
            np.mean(
                [
                    _mean_metric(shifts_by_object[object_name], shift_index[field])
                    for object_name in OBJECTS
                ],
                dtype=np.float64,
            )
            for field in RANDOM_CAPTURE_FIELDS
        ],
    )
    random_drift_macro = np.asarray(
        [
            np.mean(
                [
                    _mean_metric(shifts_by_object[object_name], shift_index[field])
                    for object_name in OBJECTS
                ],
                dtype=np.float64,
            )
            for field in RANDOM_DRIFT_FIELDS
        ],
    )
    random_capture_q95 = float(np.percentile(random_capture_macro, 95))
    random_drift_q95 = float(np.percentile(random_drift_macro, 95))
    g1_margin = macro["physical_energy_capture"] - max(
        macro["clean_pca_energy_capture"],
        random_capture_q95,
    )
    g3_margin = macro["physical_drift_reduction"] - max(
        macro["clean_pca_drift_reduction"],
        random_drift_q95,
    )
    g1_lcb = float(
        np.percentile(
            [float(row["g1_margin_over_hard_null"]) for row in bootstraps],
            2.5,
        ),
    )
    g3_lcb = float(
        np.percentile(
            [float(row["physical_drift_reduction"]) for row in bootstraps],
            2.5,
        ),
    )
    g3_margin_lcb = float(
        np.percentile(
            [float(row["g3_margin_over_hard_null"]) for row in bootstraps],
            2.5,
        ),
    )
    stability_all = np.concatenate([stability_by_object[name] for name in OBJECTS])
    object_stability_p05 = {
        name: float(np.percentile(stability_by_object[name], 5)) for name in OBJECTS
    }
    object_g3 = {row["object"]: float(row["physical_drift_reduction"]) for row in per_object}
    object_g4_auroc = {row["object"]: float(row["delta_seg_AUROC_0.05"]) for row in per_object}
    object_g4_f1 = {row["object"]: float(row["delta_seg_F1"]) for row in per_object}

    gates = {
        "G1": {
            "pass": bool(macro["physical_energy_capture"] >= 0.60 and g1_lcb > 0.10),
            "physical_energy_capture_macro": macro["physical_energy_capture"],
            "clean_pca_energy_capture_macro": macro["clean_pca_energy_capture"],
            "random_seed_macro_q95": random_capture_q95,
            "margin_over_hard_null": g1_margin,
            "margin_lcb95": g1_lcb,
        },
        "G2": {
            "pass": bool(
                float(np.median(stability_all)) >= 0.90
                and min(object_stability_p05.values()) >= 0.75,
            ),
            "stability_median": float(np.median(stability_all)),
            "object_stability_p05": object_stability_p05,
            "minimum_object_stability_p05": min(object_stability_p05.values()),
        },
        "G3": {
            "pass": bool(
                macro["physical_drift_reduction"] >= 0.65
                and g3_lcb >= 0.50
                and min(object_g3.values()) >= 0.40
                and g3_margin >= 0.10,
            ),
            "physical_drift_reduction_macro": macro["physical_drift_reduction"],
            "clean_pca_drift_reduction_macro": macro["clean_pca_drift_reduction"],
            "random_seed_macro_q95": random_drift_q95,
            "physical_lcb95": g3_lcb,
            "margin_over_hard_null": g3_margin,
            "margin_lcb95_diagnostic": g3_margin_lcb,
            "per_object": object_g3,
            "minimum_object": min(object_g3.values()),
        },
        "G4": {
            "pass": bool(
                macro["delta_seg_AUROC_0.05"] >= -0.005
                and macro["delta_seg_F1"] >= -0.010
                and min(object_g4_auroc.values()) >= -0.020
                and min(object_g4_f1.values()) >= -0.030
                and macro["anomaly_energy_retention"] >= 0.90
                and all(
                    counts
                    == {
                        "clean": SUPPORT_COUNT * len(LAYERS) * len(DEFECTS),
                        "shifted": SUPPORT_COUNT * len(LAYERS) * len(DEFECTS),
                    }
                    for counts in acquisition_counts.values()
                ),
            ),
            "delta_seg_AUROC_0.05_macro": macro["delta_seg_AUROC_0.05"],
            "delta_seg_F1_macro": macro["delta_seg_F1"],
            "minimum_object_delta_seg_AUROC_0.05": min(object_g4_auroc.values()),
            "minimum_object_delta_seg_F1": min(object_g4_f1.values()),
            "anomaly_energy_retention_macro": macro["anomaly_energy_retention"],
            "clean_and_shifted_row_coverage": acquisition_counts,
        },
        "G5": {
            "pass": True,
            "finite_nonconstant_coverage": True,
            "exact_support_fold_transform_defect_parity": True,
            "no_public_access": True,
            "source_hashes": source_hashes,
        },
    }
    all_pass = all(bool(gates[name]["pass"]) for name in ("G1", "G2", "G3", "G4", "G5"))
    verdict = "PROMISING_DIAGNOSTIC" if all_pass else "KILL_FOR_CLAIM / NO_CONTINUE"
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "evaluation_role": "support_only_synthetic_identifiability_preflight",
        "maximum_verdict": "PROMISING_DIAGNOSTIC",
        "verdict": verdict,
        "objects": list(OBJECTS),
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "unit": "support image within object; all child rows preserved",
            "stream": "SHA256 protocol/object/replicate",
            "confidence_bound": "2.5th percentile (95% lower bound)",
        },
        "defect_metric_grid": "feature_grid_any_overlap",
        "gates": gates,
        "per_object": per_object,
        "claim_boundary": (
            "Synthetic support-only, feature-grid mechanism evidence; no TESTpublic/private image "
            "or label was read, and seg_* fields are not original-resolution pixel metrics."
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_tsv(output_root / "per_object.tsv", per_object)
    _write_tsv(output_root / "bootstrap.tsv", bootstraps)
    (output_root / "verdict.txt").write_text(verdict + "\n", encoding="utf-8")
    payload["output_sha256"] = {
        "per_object.tsv": sha256_file(output_root / "per_object.tsv"),
        "bootstrap.tsv": sha256_file(output_root / "bootstrap.tsv"),
        "verdict.txt": sha256_file(output_root / "verdict.txt"),
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _emit_invalid(output_root: Path, message: str) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "maximum_verdict": "PROMISING_DIAGNOSTIC",
        "verdict": "INVALID_RUN",
        "gates": {"G5": {"pass": False, "error": message}},
        "claim_boundary": "No scientific conclusion is valid from this run.",
    }
    (output_root / "verdict.txt").write_text("INVALID_RUN\n", encoding="utf-8")
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-root")
    parser.add_argument(
        "--require-all8",
        action="store_true",
        help="Compatibility flag; Stage -1 always requires the frozen all-eight population.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=MACRO_BOOTSTRAPS)
    args = parser.parse_args(list(argv))
    if args.bootstrap_replicates != MACRO_BOOTSTRAPS:
        parser.error(f"Stage -1 freezes --bootstrap-replicates={MACRO_BOOTSTRAPS}")
    return args


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    run_root = Path(args.run_root)
    output_root = Path(args.output_root) if args.output_root else run_root
    try:
        payload = aggregate(run_root, output_root, bootstrap_replicates=args.bootstrap_replicates)
    except AuditError as error:
        payload = _emit_invalid(output_root, str(error))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
