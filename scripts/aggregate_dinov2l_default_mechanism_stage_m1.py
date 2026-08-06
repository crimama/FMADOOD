#!/usr/bin/env python3
# ruff: noqa: ANN401, C901, EM102, PLR0912, PLR0915
"""Aggregate the frozen DINOv2-L Basic default Stage-M1 mechanism audit.

The runner emits one compact JSONL row for each arm/condition/stimulus cell.
This module deliberately keeps normalization and aggregation as public functions
so object-sharded runs can be audited before their artifacts are copied together.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

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
FOLDS = (0, 1, 2, 3)
ARMS = ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7")
LAYER_ARMS = ("L5", "L11", "L17", "L23")
ALL_ARMS = (*ARMS, *LAYER_ARMS)
CONDITIONS = ("C", "S", "D", "SD")
AXES = ("exposure", "gamma", "contrast", "white_balance")
SIGNS = (-1, 1)
MAGNITUDES = (1, 2)
FAMILIES = ("cutpaste", "chromatic_blotch", "gaussian_blur", "smooth_local_warp")
SUPPORT_SHA256 = "7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3"
PROTOCOL_ID = "dinov2l-default-mechanism-stage-m1-20260719"
ROW_SCHEMA = "dinov2l-default-mechanism-m1-row-v1"
OBJECT_SCHEMA = "dinov2l-default-mechanism-m1-summary-v1"
SUMMARY_SCHEMA = "dinov2l-default-mechanism-m1-aggregate-v1"
BOOTSTRAPS = 2000
CUDA_CDIST_PARITY_ATOL = 1e-4

NUMERIC_FIELDS = (
    "score_q95",
    "score_q99",
    "score_q995",
    "inside_median",
    "outside_median",
    "defect_contrast",
    "seg_auroc_0_05",
    "fixed_f1",
    "nn_identity_retention",
    "nn_margin_mean",
    "nn_source_entropy",
    "nn_position_displacement_mean",
    "memory_gini",
    "pair_distance_spearman",
    "effective_rank",
    "latent_rms",
    "logdet_q05",
    "logdet_q50",
    "logdet_q95",
    "distance_density_correlation",
    "dvt_field_energy",
    "dvt_variance_removed",
    "dvt_cosine_displacement_before",
    "dvt_cosine_displacement_after",
)

OUTCOMES = (
    "tail_inflation_abs",
    "tail_inflation_relative",
    "D_pauroc",
    "SD_pauroc",
    "D_fixed_f1",
    "SD_fixed_f1",
    "pauroc_retention",
    "f1_retention",
    "factorial_contrast",
    "nn_identity_retention",
    "nn_margin_mean",
    "pair_distance_spearman",
    "effective_rank",
    "latent_rms",
)

CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("dvt_total", "A0", "A1"),
    ("dvt_support_side", "A2", "A1"),
    ("dvt_query_side", "A3", "A1"),
    ("density", "A0", "A4"),
    ("flow_with_dvt", "A4", "A5"),
    ("flow_without_dvt", "A6", "A7"),
)

_FAMILY_ALIASES = {
    "cutpaste_texture": "cutpaste",
    "copy_paste_texture": "cutpaste",
    "low_frequency_cutpaste": "cutpaste",
    "chromatic_stain": "chromatic_blotch",
    "low_frequency_chromatic_stain": "chromatic_blotch",
    "local_blur": "gaussian_blur",
    "smooth_local_geometric_warp": "smooth_local_warp",
    "smooth_geometric_warp": "smooth_local_warp",
    "geometric_warp": "smooth_local_warp",
}


class AuditError(RuntimeError):
    """Raised when an input is scientifically incomplete or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _finite_or_none(value: object, field: str, context: str) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AuditError(f"{context}: {field} is not numeric or null") from error
    _require(np.isfinite(result), f"{context}: {field} is non-finite")
    return result


def _normalize_sign(value: object) -> int:
    aliases: dict[object, int] = {
        -1: -1,
        1: 1,
        "-1": -1,
        "+1": 1,
        "negative": -1,
        "positive": 1,
        "minus": -1,
        "plus": 1,
    }
    if value not in aliases:
        raise AuditError(f"unknown shift sign: {value!r}")
    return aliases[value]


def _normalize_magnitude(value: object) -> int:
    aliases: dict[object, int] = {1: 1, 2: 2, "1": 1, "2": 2, "x1": 1, "x2": 2}
    if value not in aliases:
        raise AuditError(f"unknown shift magnitude: {value!r}")
    return aliases[value]


def _normalize_family(value: object) -> str:
    family = str(value)
    return _FAMILY_ALIASES.get(family, family)


def _none_token(value: object) -> bool:
    return value is None or str(value).lower() in {"", "none", "null", "na", "n/a"}


def _hash_field(row: Mapping[str, Any], field: str, context: str) -> str:
    value = str(row.get(field, ""))
    _require(
        len(value) == 64 and all(character in "0123456789abcdef" for character in value),
        f"{context}: invalid {field}",
    )
    return value


def audit_stimulus_custody(
    rows: Sequence[Mapping[str, Any]],
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Validate raw stimulus identity before normalization drops custody fields."""
    arm_metadata: dict[tuple[Any, ...], set[tuple[Any, ...]]] = defaultdict(set)
    defect_metadata: dict[tuple[str, str, str], set[tuple[Any, ...]]] = defaultdict(set)
    clean_recorded: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        context = "/".join(
            str(row.get(key, "?")) for key in ("object", "relative_path", "condition", "arm")
        )
        condition = str(row.get("condition", "")).upper()
        _require(condition in CONDITIONS, f"{context}: unknown condition in custody audit")
        evaluated_hash = _hash_field(row, "evaluated_image_sha256", context)
        stimulus_key = (
            str(row.get("object")), str(row.get("relative_path")), condition,
            str(row.get("stimulus_id")), row.get("shift_axis"), row.get("shift_sign"),
            row.get("shift_magnitude"), str(row.get("defect")),
        )
        if condition in {"D", "SD"}:
            defect_hash = _hash_field(row, "defect_image_sha256", context)
            mask_hash = _hash_field(row, "mask_sha256", context)
            seed = int(row.get("defect_seed", -1))
            retry = int(row.get("defect_feature_retry", -1))
            _require(seed >= 0 and retry >= 0, f"{context}: invalid defect seed/retry")
            base_metadata = (defect_hash, mask_hash, seed, retry)
            defect_metadata[
                (str(row.get("object")), str(row.get("relative_path")), str(row.get("defect")))
            ].add(base_metadata)
        else:
            base_metadata = ()
        arm_metadata[stimulus_key].add((evaluated_hash, *base_metadata))
        if condition == "C":
            clean_recorded[str(row.get("relative_path"))].add(evaluated_hash)

    inconsistent_arm_cells = sum(len(values) != 1 for values in arm_metadata.values())
    inconsistent_defect_bases = sum(len(values) != 1 for values in defect_metadata.values())
    _require(inconsistent_arm_cells == 0, "stimulus metadata differs across arms")
    _require(inconsistent_defect_bases == 0, "D/SD defect metadata identity mismatch")
    _require(
        all(len(values) == 1 for values in clean_recorded.values()),
        "C hash differs across arms",
    )

    clean_hash_mismatches: list[str] = []
    if data_root is not None:
        root = data_root.resolve(strict=True)
        for relative_path, recorded_values in sorted(clean_recorded.items()):
            path = (root / relative_path).resolve(strict=True)
            try:
                path.relative_to(root)
            except ValueError as error:
                raise AuditError(f"clean source escapes data root: {relative_path}") from error
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            observed = hashlib.sha256(rgb.tobytes()).hexdigest()
            if observed != next(iter(recorded_values)):
                clean_hash_mismatches.append(relative_path)
        _require(not clean_hash_mismatches, "C evaluated-image custody hash mismatch")

    return {
        "arm_consistent_cells": len(arm_metadata),
        "defect_base_cells": len(defect_metadata),
        "clean_sources": len(clean_recorded),
        "clean_rgb_hash_verified": data_root is not None,
        "clean_hash_mismatch_count": len(clean_hash_mismatches),
        "arm_metadata_consistent": True,
        "d_sd_defect_identity_consistent": True,
    }


def normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one runner row and normalize harmless serialization aliases."""
    context = "/".join(
        str(row.get(key, "?")) for key in ("object", "relative_path", "arm", "condition")
    )
    _require(row.get("schema") == ROW_SCHEMA, f"{context}: row schema mismatch")
    _require(row.get("protocol_id") == PROTOCOL_ID, f"{context}: protocol mismatch")
    object_name = str(row.get("object"))
    arm = str(row.get("arm"))
    condition = str(row.get("condition")).upper()
    relative_path = str(row.get("relative_path"))
    fold = int(row.get("fold", -1))
    _require(object_name in OBJECTS, f"{context}: unknown object")
    _require(arm in ALL_ARMS, f"{context}: unknown arm")
    _require(condition in CONDITIONS, f"{context}: unknown condition")
    _require(fold in FOLDS, f"{context}: unknown fold")
    _require("/train/good/" in relative_path, f"{context}: source is not train/good")
    _require(
        not any(
            token in relative_path for token in ("test_public", "test_private", "ground_truth")
        ),
        f"{context}: forbidden path",
    )
    path_hash = str(row.get("path_sha256", ""))
    _require(
        len(path_hash) == 64 and all(character in "0123456789abcdef" for character in path_hash),
        f"{context}: invalid path hash",
    )
    stimulus_id = str(row.get("stimulus_id", ""))
    _require(bool(stimulus_id), f"{context}: empty stimulus id")

    has_shift = condition in {"S", "SD"}
    has_defect = condition in {"D", "SD"}
    if has_shift:
        axis = str(row.get("shift_axis"))
        sign = _normalize_sign(row.get("shift_sign"))
        magnitude = _normalize_magnitude(row.get("shift_magnitude"))
        _require(axis in AXES, f"{context}: unknown shift axis")
    else:
        _require(_none_token(row.get("shift_axis")), f"{context}: shift axis on {condition}")
        _require(_none_token(row.get("shift_sign")), f"{context}: shift sign on {condition}")
        _require(_none_token(row.get("shift_magnitude")), f"{context}: magnitude on {condition}")
        axis, sign, magnitude = "none", 0, 0
    if has_defect:
        family = _normalize_family(row.get("defect"))
        _require(family in FAMILIES, f"{context}: unknown defect family")
    else:
        _require(_none_token(row.get("defect")), f"{context}: defect on {condition}")
        family = "none"

    numeric = {
        field: _finite_or_none(row.get(field), field, context) for field in NUMERIC_FIELDS
    }
    required = {"score_q95", "score_q99", "score_q995"}
    if has_defect:
        required.update(
            {"inside_median", "outside_median", "defect_contrast", "seg_auroc_0_05", "fixed_f1"},
        )
    for field in required:
        _require(numeric[field] is not None, f"{context}: required metric {field} is null")
    if has_defect:
        _require(
            np.isclose(
                float(numeric["defect_contrast"]),
                float(numeric["inside_median"]) - float(numeric["outside_median"]),
                rtol=1e-7,
                atol=1e-9,
            ),
            f"{context}: defect contrast equation mismatch",
        )
        for field in ("seg_auroc_0_05", "fixed_f1"):
            _require(0.0 <= float(numeric[field]) <= 1.0, f"{context}: {field} outside [0,1]")
    for field in ("nn_identity_retention", "pair_distance_spearman"):
        if numeric[field] is not None:
            lower = -1.0 if field == "pair_distance_spearman" else 0.0
            _require(lower <= float(numeric[field]) <= 1.0, f"{context}: {field} out of range")

    return {
        "object": object_name,
        "fold": fold,
        "relative_path": relative_path,
        "path_sha256": path_hash,
        "stimulus_id": stimulus_id,
        "arm": arm,
        "condition": condition,
        "axis": axis,
        "sign": sign,
        "magnitude": magnitude,
        "family": family,
        **numeric,
    }


def validate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize rows and enforce the exact all-eight factorial cross-product."""
    normalized = [normalize_row(row) for row in rows]
    by_object_source: dict[str, dict[str, int]] = defaultdict(dict)
    source_hashes: dict[tuple[str, str], str] = {}
    keys: set[tuple[Any, ...]] = set()
    observed_by_prefix: dict[tuple[str, str, str], set[tuple[Any, ...]]] = defaultdict(set)
    stimulus_ids: dict[tuple[Any, ...], str] = {}
    for row in normalized:
        object_name = row["object"]
        source = row["relative_path"]
        fold = row["fold"]
        previous = by_object_source[object_name].setdefault(source, fold)
        _require(previous == fold, f"{object_name}/{source}: fold changed across rows")
        hash_key = (object_name, source)
        previous_hash = source_hashes.setdefault(hash_key, row["path_sha256"])
        _require(previous_hash == row["path_sha256"], f"{object_name}/{source}: path hash changed")
        identity = (
            object_name,
            source,
            row["arm"],
            row["condition"],
            row["axis"],
            row["sign"],
            row["magnitude"],
            row["family"],
        )
        _require(identity not in keys, f"duplicate row: {identity}")
        keys.add(identity)
        observed_by_prefix[(object_name, source, row["arm"])].add(identity)
        stimulus_key = (
            object_name,
            source,
            row["condition"],
            row["axis"],
            row["sign"],
            row["magnitude"],
            row["family"],
        )
        previous_stimulus = stimulus_ids.setdefault(stimulus_key, row["stimulus_id"])
        _require(
            previous_stimulus == row["stimulus_id"],
            f"{object_name}/{source}: stimulus changed across arms",
        )

    _require(set(by_object_source) == set(OBJECTS), "rows do not cover exact all-eight objects")
    for object_name in OBJECTS:
        sources = by_object_source[object_name]
        _require(len(sources) == 16, f"{object_name}: expected 16 held sources")
        counts = [sum(fold == expected for fold in sources.values()) for expected in FOLDS]
        _require(counts == [4, 4, 4, 4], f"{object_name}: expected four sources per fold")
        for source in sources:
            for arm in ALL_ARMS:
                expected = {(object_name, source, arm, "C", "none", 0, 0, "none")}
                expected.update(
                    (object_name, source, arm, "S", axis, sign, magnitude, "none")
                    for axis in AXES
                    for sign in SIGNS
                    for magnitude in MAGNITUDES
                )
                expected.update(
                    (object_name, source, arm, "D", "none", 0, 0, family)
                    for family in FAMILIES
                )
                expected.update(
                    (object_name, source, arm, "SD", axis, sign, magnitude, family)
                    for axis in AXES
                    for sign in SIGNS
                    for magnitude in MAGNITUDES
                    for family in FAMILIES
                )
                observed = observed_by_prefix[(object_name, source, arm)]
                _require(
                    observed == expected,
                    f"{object_name}/{source}/{arm}: factorial parity failed",
                )
    expected_count = len(OBJECTS) * 16 * len(ALL_ARMS) * (
        1 + len(AXES) * len(SIGNS) * len(MAGNITUDES) + len(FAMILIES)
        + len(AXES) * len(SIGNS) * len(MAGNITUDES) * len(FAMILIES)
    )
    _require(len(normalized) == expected_count, "unexpected total row count")
    return normalized


def _value(row: Mapping[str, Any], field: str, context: str) -> float:
    value = row[field]
    _require(value is not None, f"{context}: {field} is required for aggregation")
    return float(value)


def build_case_array(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Build the object/source/shift/family/arm/outcome paired tensor."""
    sources = {
        object_name: sorted(
            {str(row["relative_path"]) for row in rows if row["object"] == object_name},
        )
        for object_name in OBJECTS
    }
    source_index = {
        (object_name, source): index
        for object_name, object_sources in sources.items()
        for index, source in enumerate(object_sources)
    }
    lookup = {
        (
            row["object"], row["relative_path"], row["arm"], row["condition"],
            row["axis"], row["sign"], row["magnitude"], row["family"],
        ): row
        for row in rows
    }
    shape = (
        len(OBJECTS), 16, len(AXES), len(SIGNS), len(MAGNITUDES), len(FAMILIES),
        len(ALL_ARMS), len(OUTCOMES),
    )
    result = np.empty(shape, dtype=np.float64)
    for oi, object_name in enumerate(OBJECTS):
        for source in sources[object_name]:
            si = source_index[(object_name, source)]
            for ai, axis in enumerate(AXES):
                for sgi, sign in enumerate(SIGNS):
                    for mi, magnitude in enumerate(MAGNITUDES):
                        for fi, family in enumerate(FAMILIES):
                            for ri, arm in enumerate(ALL_ARMS):
                                prefix = (object_name, source, arm)
                                c = lookup[(*prefix, "C", "none", 0, 0, "none")]
                                s = lookup[(*prefix, "S", axis, sign, magnitude, "none")]
                                d = lookup[(*prefix, "D", "none", 0, 0, family)]
                                sd = lookup[(*prefix, "SD", axis, sign, magnitude, family)]
                                clean_tail = _value(c, "score_q995", f"{prefix}/C")
                                shift_tail = _value(s, "score_q995", f"{prefix}/S")
                                tail_abs = shift_tail - clean_tail
                                tail_rel = tail_abs / max(abs(clean_tail), 1e-12)
                                d_p = _value(d, "seg_auroc_0_05", f"{prefix}/D")
                                sd_p = _value(sd, "seg_auroc_0_05", f"{prefix}/SD")
                                d_f = _value(d, "fixed_f1", f"{prefix}/D")
                                sd_f = _value(sd, "fixed_f1", f"{prefix}/SD")
                                d_contrast = _value(d, "defect_contrast", f"{prefix}/D")
                                sd_contrast = _value(sd, "defect_contrast", f"{prefix}/SD")
                                diagnostics = (
                                    _value(s, "nn_identity_retention", f"{prefix}/S"),
                                    _value(s, "nn_margin_mean", f"{prefix}/S"),
                                    _value(s, "pair_distance_spearman", f"{prefix}/S"),
                                    _value(s, "effective_rank", f"{prefix}/S"),
                                    _value(s, "latent_rms", f"{prefix}/S"),
                                )
                                result[oi, si, ai, sgi, mi, fi, ri] = (
                                    tail_abs, tail_rel, d_p, sd_p, d_f, sd_f,
                                    sd_p - d_p, sd_f - d_f, sd_contrast - d_contrast,
                                    *diagnostics,
                                )
    _require(np.isfinite(result).all(), "derived case tensor contains non-finite metrics")
    return result


def _hierarchical_mean(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample object→source→axis→magnitude→family, retaining both signs."""
    # Signs are the frozen paired directions within each sampled shift cell.
    values = data.mean(axis=3)  # O,S,A,M,F,R,K
    o, s, a, m, f = values.shape[:5]
    f_index = rng.integers(0, f, size=(o, s, a, m, f))
    values = np.take_along_axis(values, f_index[..., None, None], axis=4).mean(axis=4)
    m_index = rng.integers(0, m, size=(o, s, a, m))
    values = np.take_along_axis(values, m_index[..., None, None], axis=3).mean(axis=3)
    a_index = rng.integers(0, a, size=(o, s, a))
    values = np.take_along_axis(values, a_index[..., None, None], axis=2).mean(axis=2)
    s_index = rng.integers(0, s, size=(o, s))
    values = np.take_along_axis(values, s_index[..., None, None], axis=1).mean(axis=1)
    o_index = rng.integers(0, o, size=o)
    return values[o_index].mean(axis=0)


def _contrast_values(arm_metrics: np.ndarray) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for name, candidate, baseline in CONTRASTS:
        delta = arm_metrics[ALL_ARMS.index(candidate)] - arm_metrics[ALL_ARMS.index(baseline)]
        result.update(
            {(name, outcome): float(delta[index]) for index, outcome in enumerate(OUTCOMES)},
        )
    interaction = (
        arm_metrics[ALL_ARMS.index("A4")] - arm_metrics[ALL_ARMS.index("A5")]
        - arm_metrics[ALL_ARMS.index("A6")] + arm_metrics[ALL_ARMS.index("A7")]
    )
    result.update(
        {
            ("dvt_flow_interaction", outcome): float(interaction[index])
            for index, outcome in enumerate(OUTCOMES)
        },
    )
    layer_values = arm_metrics[[ALL_ARMS.index(arm) for arm in LAYER_ARMS]]
    envelope = layer_values.max(axis=0)
    fusion = arm_metrics[ALL_ARMS.index("A5")] - envelope
    result.update(
        {
            ("fusion_layer_envelope", outcome): float(fusion[index])
            for index, outcome in enumerate(OUTCOMES)
        },
    )
    return result


def bootstrap_contrasts(data: np.ndarray) -> tuple[list[dict[str, Any]], int]:
    """Return deterministic paired hierarchical-bootstrap contrast intervals."""
    seed = int.from_bytes(
        hashlib.sha256(f"{PROTOCOL_ID}\x1fhierarchical-bootstrap".encode()).digest()[:8], "big",
    )
    rng = np.random.default_rng(seed)
    point_arm = data.mean(axis=(0, 1, 2, 3, 4, 5))
    point = _contrast_values(point_arm)
    samples = {key: np.empty(BOOTSTRAPS, dtype=np.float64) for key in point}
    for replicate in range(BOOTSTRAPS):
        values = _contrast_values(_hierarchical_mean(data, rng))
        for key, value in values.items():
            samples[key][replicate] = value

    object_arm = data.mean(axis=(1, 2, 3, 4, 5))
    object_effects = [_contrast_values(object_arm[index]) for index in range(len(OBJECTS))]
    rows: list[dict[str, Any]] = []
    for key in point:
        values = samples[key]
        effects = [item[key] for item in object_effects]
        rows.append(
            {
                "contrast": key[0],
                "outcome": key[1],
                "estimate": point[key],
                "ci_lower_2_5": float(np.percentile(values, 2.5)),
                "ci_upper_97_5": float(np.percentile(values, 97.5)),
                "positive_objects": sum(value > 0.0 for value in effects),
                "negative_objects": sum(value < 0.0 for value in effects),
                "zero_objects": sum(value == 0.0 for value in effects),
            },
        )
    return rows, seed


def _arm_summary(data: np.ndarray) -> dict[str, dict[str, float]]:
    means = data.mean(axis=(0, 1, 2, 3, 4, 5))
    return {
        arm: {outcome: float(means[ai, oi]) for oi, outcome in enumerate(OUTCOMES)}
        for ai, arm in enumerate(ALL_ARMS)
    }


def _contrast_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(str(row["contrast"]), str(row["outcome"])): row for row in rows}


def classify_mechanisms(
    data: np.ndarray,
    contrast_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply only the frozen mechanism labels and materiality/collapse gates."""
    arms = _arm_summary(data)
    contrasts = _contrast_lookup(contrast_rows)

    def lower(name: str, outcome: str) -> float:
        return float(contrasts[(name, outcome)]["ci_lower_2_5"])

    def upper(name: str, outcome: str) -> float:
        return float(contrasts[(name, outcome)]["ci_upper_97_5"])

    def positive_objects(name: str, outcome: str) -> int:
        return int(contrasts[(name, outcome)]["positive_objects"])

    def negative_objects(name: str, outcome: str) -> int:
        return int(contrasts[(name, outcome)]["negative_objects"])

    directional_object_count = min(6, len(OBJECTS))

    def relative_reduction(candidate: float, baseline: float) -> float:
        return 0.0 if baseline <= 1e-12 else (baseline - candidate) / baseline

    object_arm = data.mean(axis=(1, 2, 3, 4, 5))
    a0, a1 = ALL_ARMS.index("A0"), ALL_ARMS.index("A1")
    p_indices = [OUTCOMES.index("D_pauroc"), OUTCOMES.index("SD_pauroc")]
    f_indices = [OUTCOMES.index("D_fixed_f1"), OUTCOMES.index("SD_fixed_f1")]
    dvt_no_collapse = bool(
        np.all(object_arm[:, a0, p_indices] - object_arm[:, a1, p_indices] >= -0.02)
        and np.all(object_arm[:, a0, f_indices] - object_arm[:, a1, f_indices] >= -0.03),
    )
    dvt_tail_reduction = relative_reduction(
        arms["A0"]["tail_inflation_abs"], arms["A1"]["tail_inflation_abs"],
    )
    dvt_direction = (
        upper("dvt_total", "tail_inflation_abs") < 0.0
        and negative_objects("dvt_total", "tail_inflation_abs") >= directional_object_count
    )
    dvt_preserved = (
        contrasts[("dvt_total", "D_pauroc")]["estimate"] >= -0.005
        and contrasts[("dvt_total", "SD_pauroc")]["estimate"] >= -0.005
        and contrasts[("dvt_total", "D_fixed_f1")]["estimate"] >= -0.010
        and contrasts[("dvt_total", "SD_fixed_f1")]["estimate"] >= -0.010
    )
    shift_corrector = (
        dvt_tail_reduction >= 0.10 and dvt_direction and dvt_preserved and dvt_no_collapse
    )

    total_tail = arms["A1"]["tail_inflation_abs"] - arms["A0"]["tail_inflation_abs"]
    total_retention = arms["A0"]["pauroc_retention"] - arms["A1"]["pauroc_retention"]

    def recovery(arm: str) -> tuple[float, float]:
        tail = arms["A1"]["tail_inflation_abs"] - arms[arm]["tail_inflation_abs"]
        retention = arms[arm]["pauroc_retention"] - arms["A1"]["pauroc_retention"]
        return (
            tail / total_tail if total_tail > 1e-12 else 0.0,
            retention / total_retention if total_retention > 1e-12 else 0.0,
        )

    support_recovery = recovery("A2")
    query_recovery = recovery("A3")
    support_side_failed = (
        min(support_recovery) < 0.70
        or arms["A2"]["nn_identity_retention"]
        < arms["A0"]["nn_identity_retention"] - 0.10
    )
    query_side_failed = (
        min(query_recovery) < 0.70
        or arms["A3"]["nn_identity_retention"]
        < arms["A0"]["nn_identity_retention"] - 0.10
    )
    ranking_gain = (
        contrasts[("dvt_total", "SD_pauroc")]["estimate"] >= 0.005
        and lower("dvt_total", "SD_pauroc") > 0.0
        and positive_objects("dvt_total", "SD_pauroc") >= directional_object_count
    )
    if shift_corrector and min(query_recovery) >= 0.70 and max(support_recovery) <= 0.30:
        dvt_label = "QUERY_DRIVEN"
    elif shift_corrector and support_side_failed and query_side_failed:
        dvt_label = "COMMON_COORDINATE_ALIGNMENT"
    elif shift_corrector:
        dvt_label = "SHIFT_CORRECTOR"
    elif ranking_gain:
        dvt_label = "RANKING_ONLY_OR_MIXED"
    else:
        dvt_label = "NO_MECHANISM_CREDIT"

    flow_metric = all(
        contrasts[("flow_with_dvt", outcome)]["estimate"] >= threshold
        and lower("flow_with_dvt", outcome) > 0.0
        and positive_objects("flow_with_dvt", outcome) >= directional_object_count
        for outcome, threshold in (
            ("D_pauroc", 0.005), ("SD_pauroc", 0.005),
            ("D_fixed_f1", 0.010), ("SD_fixed_f1", 0.010),
        )
    )
    a4, a5 = arms["A4"], arms["A5"]
    flow_tail_safe = a4["tail_inflation_abs"] <= a5["tail_inflation_abs"] + max(
        0.10 * abs(a5["tail_inflation_abs"]), 1e-12,
    )
    geometry_safe = (
        a4["pair_distance_spearman"] - a5["pair_distance_spearman"] >= -0.10
        and a4["effective_rank"] >= 0.50 * max(a5["effective_rank"], 1e-12)
        and 0.50 * max(a5["latent_rms"], 1e-12) <= a4["latent_rms"]
        <= 2.0 * max(a5["latent_rms"], 1e-12)
        and a4["nn_identity_retention"] - a5["nn_identity_retention"] >= -0.10
    )
    flow_partial_signal = any(
        contrasts[("flow_with_dvt", outcome)]["estimate"] >= threshold
        and lower("flow_with_dvt", outcome) > 0.0
        and positive_objects("flow_with_dvt", outcome) >= directional_object_count
        for outcome, threshold in (
            ("D_pauroc", 0.005), ("SD_pauroc", 0.005),
            ("D_fixed_f1", 0.010), ("SD_fixed_f1", 0.010),
        )
    )
    if flow_metric and flow_tail_safe and geometry_safe:
        flow_label = "ROBUST_REPRESENTATION_REMAP"
    elif flow_metric:
        flow_label = "DISCRIMINATIVE_BUT_SHIFT_SENSITIVE"
    elif flow_partial_signal:
        flow_label = "MIXED_OR_INCONCLUSIVE"
    else:
        flow_label = "NOT_NECESSARY"

    density_separation = (
        (
            contrasts[("density", "SD_pauroc")]["estimate"] >= 0.005
            and lower("density", "SD_pauroc") > 0.0
            and positive_objects("density", "SD_pauroc") >= directional_object_count
        )
        or (
            contrasts[("density", "SD_fixed_f1")]["estimate"] >= 0.010
            and lower("density", "SD_fixed_f1") > 0.0
            and positive_objects("density", "SD_fixed_f1") >= directional_object_count
        )
    )
    density_tail_extra = (
        arms["A0"]["tail_inflation_abs"]
        > arms["A4"]["tail_inflation_abs"] + 0.10 * abs(arms["A4"]["tail_inflation_abs"])
        and lower("density", "tail_inflation_abs") > 0.0
        and positive_objects("density", "tail_inflation_abs") >= directional_object_count
    )
    density_correction = (
        contrasts[("density", "pauroc_retention")]["estimate"] > 0.0
        and lower("density", "pauroc_retention") > 0.0
        and positive_objects("density", "pauroc_retention") >= directional_object_count
        and upper("density", "tail_inflation_abs") < 0.0
        and negative_objects("density", "tail_inflation_abs") >= directional_object_count
    )
    if density_correction:
        density_label = "SHIFT_CORRECTOR"
    elif density_separation and density_tail_extra:
        density_label = "SEPARATION_AMPLIFIER_WITH_SHIFT_RISK"
    elif not density_separation and density_tail_extra:
        density_label = "HARMFUL_TAIL_EXPANSION"
    elif density_separation:
        density_label = "SEPARATION_AMPLIFIER_WITHOUT_OBSERVED_SHIFT_RISK"
    else:
        density_label = "NO_MATERIAL_CONTRIBUTION"

    fusion_synergy = all(
        contrasts[("fusion_layer_envelope", outcome)]["estimate"] >= threshold
        and lower("fusion_layer_envelope", outcome) > 0.0
        and positive_objects("fusion_layer_envelope", outcome) >= directional_object_count
        for outcome, threshold in (
            ("D_pauroc", 0.005), ("SD_pauroc", 0.005),
            ("D_fixed_f1", 0.010), ("SD_fixed_f1", 0.010),
        )
    )
    dominant = max(LAYER_ARMS, key=lambda arm: arms[arm]["SD_pauroc"])
    fusion_label = (
        "MULTI_LEVEL_SYNERGY" if fusion_synergy else "DOMINANT_LAYER_OR_DESTRUCTIVE_FUSION"
    )
    interaction_material = any(
        (
            abs(float(contrasts[("dvt_flow_interaction", outcome)]["estimate"])) >= threshold
            and (
                lower("dvt_flow_interaction", outcome) > 0.0
                or upper("dvt_flow_interaction", outcome) < 0.0
            )
        )
        for outcome, threshold in (("SD_pauroc", 0.005), ("SD_fixed_f1", 0.010))
    )
    return {
        "dvt": {
            "label": dvt_label,
            "tail_reduction_fraction": dvt_tail_reduction,
            "support_recovery_fraction": list(support_recovery),
            "query_recovery_fraction": list(query_recovery),
            "support_side_failed": bool(support_side_failed),
            "query_side_failed": bool(query_side_failed),
            "no_object_collapse": bool(dvt_no_collapse),
        },
        "flow": {
            "label": flow_label,
            "tail_safe": bool(flow_tail_safe),
            "geometry_safe": bool(geometry_safe),
        },
        "density": {"label": density_label, "tail_extra_10pct": bool(density_tail_extra)},
        "fusion": {"label": fusion_label, "dominant_single_layer": dominant},
        "dvt_flow_interaction": {
            "material": bool(interaction_material),
            "independent_credit_allowed": not interaction_material,
        },
        "postprocessing": {"label": "NOT_EVALUATED_MISSING_P1_P2"},
    }


def validate_object_summaries(
    summaries: Sequence[Mapping[str, Any]],
    row_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Validate M0/access/static-memory flags in independently produced shards."""
    by_object = {str(item.get("object")): item for item in summaries}
    _require(len(by_object) == len(summaries), "duplicate object summaries")
    _require(set(by_object) == set(OBJECTS), "object summaries do not cover exact all-eight")
    expected_per_object = 16 * len(ALL_ARMS) * (
        1 + len(AXES) * len(SIGNS) * len(MAGNITUDES) + len(FAMILIES)
        + len(AXES) * len(SIGNS) * len(MAGNITUDES) * len(FAMILIES)
    )
    audit: dict[str, Any] = {}
    for object_name in OBJECTS:
        item = by_object[object_name]
        context = f"{object_name} summary"
        _require(item.get("schema") == OBJECT_SCHEMA, f"{context}: schema mismatch")
        _require(item.get("protocol_id") == PROTOCOL_ID, f"{context}: protocol mismatch")
        _require(
            item.get("mode") in {"real", "support_only", "stage_m1_support_only"},
            f"{context}: mode",
        )
        _require(item.get("support_count") == 16, f"{context}: support count")
        _require(item.get("fold_counts") == [4, 4, 4, 4], f"{context}: fold counts")
        _require(set(item.get("arms", [])) == set(ALL_ARMS), f"{context}: arm parity")
        _require(set(item.get("conditions", [])) == set(CONDITIONS), f"{context}: condition parity")
        observed = int(item.get("observed_factorial_count", item.get("row_count", -1)))
        expected = int(item.get("expected_factorial_count", expected_per_object))
        _require(observed == expected == expected_per_object, f"{context}: factorial row count")
        _require(row_counts.get(object_name) == expected_per_object, f"{context}: JSONL row count")
        _require(
            float(item.get("m0_score_atol", np.inf)) == CUDA_CDIST_PARITY_ATOL,
            f"{context}: unregistered CUDA score tolerance",
        )
        _require(
            float(item.get("m0_score_max_abs_error", np.inf)) <= CUDA_CDIST_PARITY_ATOL,
            f"{context}: score parity",
        )
        _require(int(item.get("m0_nn_mismatch_count", -1)) == 0, f"{context}: NN parity")
        for field in ("parity_complete", "finite_complete", "static_scoring_only"):
            _require(item.get(field) is True, f"{context}: {field} failed")
        _require(item.get("test_public_accessed") is False, f"{context}: public access")
        _require(item.get("test_private_accessed") is False, f"{context}: private access")
        _require(int(item.get("tta_calls", -1)) == 0, f"{context}: TTA calls")
        _require(int(item.get("query_update_calls", -1)) == 0, f"{context}: query update calls")
        audit[object_name] = {
            "m0_score_max_abs_error": float(item["m0_score_max_abs_error"]),
            "m0_score_atol": CUDA_CDIST_PARITY_ATOL,
            "m0_nn_mismatch_count": int(item["m0_nn_mismatch_count"]),
            "parity_complete": True,
            "finite_complete": True,
            "static_scoring_only": True,
            "access_clean": True,
        }
    return audit


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    data_root: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    stimulus_custody = audit_stimulus_custody(rows, data_root=data_root)
    normalized = validate_rows(rows)
    counts = {name: sum(row["object"] == name for row in normalized) for name in OBJECTS}
    audit = validate_object_summaries(summaries, counts)
    data = build_case_array(normalized)
    contrast_rows, bootstrap_seed = bootstrap_contrasts(data)
    mechanism = classify_mechanisms(data, contrast_rows)
    arm_macro = _arm_summary(data)
    object_arm = data.mean(axis=(1, 2, 3, 4, 5))
    per_object = [
        {
            "object": object_name,
            "arm": arm,
            **{outcome: float(object_arm[oi, ai, mi]) for mi, outcome in enumerate(OUTCOMES)},
        }
        for oi, object_name in enumerate(OBJECTS)
        for ai, arm in enumerate(ALL_ARMS)
    ]
    summary = {
        "schema": SUMMARY_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "support_manifest_sha256": SUPPORT_SHA256,
        "population": "train/good exact support-16 only",
        "metric_scope": "DINO feature patch-grid pAUROC@0.05 and fixed-threshold F1",
        "objects": list(OBJECTS),
        "folds": list(FOLDS),
        "arms": list(ALL_ARMS),
        "record_count": len(normalized),
        "factorial_case_count": int(np.prod(data.shape[:6])),
        "validity": {
            "complete_all8": True,
            "complete_four_folds": True,
            "complete_factorial": True,
            "finite": True,
            "static_memory": True,
            "test_public_accessed": False,
            "test_private_accessed": False,
            "tta_used": False,
            "query_update_used": False,
            "object_audit": audit,
            "stimulus_custody": stimulus_custody,
        },
        "arm_macro": arm_macro,
        "mechanism": mechanism,
        "bootstrap": {
            "replicates": BOOTSTRAPS,
            "seed": bootstrap_seed,
            "unit": "object->held-source->shift-axis->magnitude->defect-family; signs paired",
            "layer_envelope_recomputed_per_replicate": True,
        },
        "claim_verdict": "KILL_FOR_CLAIM",
        "diagnostic_verdict": "MECHANISM_AUDIT_COMPLETE",
        "public_shadow_authorized": False,
        "claim_boundary": (
            "support-only synthetic patch-grid mechanism audit; "
            "no pixel-level, tuning, or superiority claim"
        ),
    }
    return summary, contrast_rows, per_object


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_records_path(run_root: Path, summary_path: Path, summary: Mapping[str, Any]) -> Path:
    artifacts = summary.get("artifacts", {})
    candidate = None
    if isinstance(artifacts, Mapping):
        for key in ("records", "compact_jsonl", "jsonl"):
            if artifacts.get(key):
                candidate = Path(str(artifacts[key]))
                break
    candidates = []
    if candidate is not None:
        candidates.extend((summary_path.parent / candidate, run_root / candidate))
    object_name = str(summary["object"])
    candidates.extend(
        (
            summary_path.parent / f"{object_name}.jsonl",
            run_root / "objects" / f"{object_name}.jsonl",
        ),
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise AuditError(f"{summary_path}: cannot resolve compact JSONL artifact")


def _validate_run_root_audit(run_root: Path) -> None:
    """Audit consolidated-run custody files when a run manifest is present."""
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.is_file():
        return  # independently copied per-object shards remain aggregatable
    manifest = _load_json(manifest_path)
    _require(
        isinstance(manifest, Mapping)
        and manifest.get("schema") == "dinov2l-default-mechanism-m1-run-v1",
        "run manifest schema mismatch",
    )
    _require(manifest.get("protocol_id") == PROTOCOL_ID, "run manifest protocol mismatch")
    support_hash = manifest.get("support_manifest_sha256", SUPPORT_SHA256)
    _require(support_hash == SUPPORT_SHA256, "run manifest support hash mismatch")
    if "objects" in manifest:
        _require(set(manifest["objects"]) == set(OBJECTS), "run manifest object parity failed")

    ledger_path = run_root / "access_ledger.json"
    _require(ledger_path.is_file(), "consolidated run is missing access_ledger.json")
    ledger = _load_json(ledger_path)
    _require(isinstance(ledger, Mapping), "access ledger must be an object")
    _require(int(ledger.get("support_reads", 0)) > 0, "access ledger has no support reads")
    for field in ("test_public_reads", "test_private_reads", "query_update_calls", "tta_calls"):
        _require(int(ledger.get(field, -1)) == 0, f"access ledger {field} must be zero")
    for field in ("static_scoring_only", "whitelist_complete"):
        _require(ledger.get(field) is True, f"access ledger {field} failed")


def _validate_chunk_audits(chunks_root: Path) -> None:
    """Validate the four independently launched GPU shards.

    The controller intentionally keeps custody files inside each isolated
    chunk, while its own root also contains the preflight smoke shard.  Audit
    only real chunks here so the smoke fixture can never enter aggregation.
    """
    manifests = sorted(chunks_root.glob("*/run_manifest.json"))
    ledgers = sorted(chunks_root.glob("*/access_ledger.json"))
    _require(len(manifests) == 4, "expected four real GPU chunk manifests")
    _require(len(ledgers) == 4, "expected four real GPU access ledgers")
    covered: set[str] = set()
    for manifest_path in manifests:
        manifest = _load_json(manifest_path)
        _require(
            isinstance(manifest, Mapping)
            and manifest.get("schema") == "dinov2l-default-mechanism-m1-run-v1",
            f"{manifest_path}: run manifest schema mismatch",
        )
        _require(
            manifest.get("protocol_id") == PROTOCOL_ID,
            f"{manifest_path}: protocol mismatch",
        )
        _require(
            manifest.get("support_manifest_sha256") == SUPPORT_SHA256,
            f"{manifest_path}: support hash mismatch",
        )
        _require(manifest.get("mode") == "real", f"{manifest_path}: non-real chunk")
        objects = {str(value) for value in manifest.get("objects", [])}
        _require(len(objects) == 2, f"{manifest_path}: expected two objects")
        _require(not covered.intersection(objects), f"{manifest_path}: duplicate object custody")
        covered.update(objects)
        for field in (
            "m0_complete",
            "factorial_complete",
            "static_scoring_only",
        ):
            _require(manifest.get(field) is True, f"{manifest_path}: {field} failed")
        for field in ("test_public_accessed", "test_private_accessed"):
            _require(manifest.get(field) is False, f"{manifest_path}: {field}")
    _require(covered == set(OBJECTS), "GPU chunks do not cover exact all-eight objects")
    for ledger_path in ledgers:
        ledger = _load_json(ledger_path)
        _require(isinstance(ledger, Mapping), f"{ledger_path}: ledger must be an object")
        _require(ledger.get("mode") == "real", f"{ledger_path}: non-real ledger")
        manifest = _load_json(ledger_path.parent / "run_manifest.json")
        shard_object_count = len(manifest.get("objects", []))
        expected_hash_reads = 16 * len(OBJECTS)
        expected_image_reads = 16 * shard_object_count
        expected_reads = expected_hash_reads + expected_image_reads
        _require(
            int(ledger.get("support_content_hash_reads", -1)) == expected_hash_reads,
            f"{ledger_path}: support content-hash reads",
        )
        _require(
            int(ledger.get("support_image_reads", -1)) == expected_image_reads,
            f"{ledger_path}: support image reads",
        )
        _require(
            int(ledger.get("support_reads", -1)) == expected_reads,
            f"{ledger_path}: total support reads",
        )
        for field in ("test_public_reads", "test_private_reads", "query_update_calls", "tta_calls"):
            _require(int(ledger.get(field, -1)) == 0, f"{ledger_path}: {field} must be zero")
        for field in ("static_scoring_only", "whitelist_complete"):
            _require(ledger.get(field) is True, f"{ledger_path}: {field} failed")


def load_run(run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load either a consolidated run root or independently copied object shards."""
    _validate_run_root_audit(run_root)
    chunks_root = run_root / "chunks"
    if chunks_root.is_dir():
        _validate_chunk_audits(chunks_root)
        paths = sorted(chunks_root.glob("*/objects/*_summary.json"))
    else:
        paths = sorted(run_root.glob("**/*_summary.json"))
    summaries_with_paths = []
    for path in paths:
        payload = _load_json(path)
        if isinstance(payload, Mapping) and payload.get("schema") == OBJECT_SCHEMA:
            summaries_with_paths.append((path, dict(payload)))
    _require(len(summaries_with_paths) == len(OBJECTS), "expected eight object summary shards")
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for summary_path, summary in summaries_with_paths:
        records_path = _resolve_records_path(run_root, summary_path, summary)
        artifact_hashes = summary.get("artifact_sha256", summary.get("sha256", {}))
        if isinstance(artifact_hashes, Mapping):
            expected_hash = artifact_hashes.get("records", artifact_hashes.get("compact_jsonl"))
            if expected_hash is not None:
                observed = hashlib.sha256(records_path.read_bytes()).hexdigest()
                _require(observed == expected_hash, f"{records_path}: artifact hash mismatch")
        with records_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as error:
                        raise AuditError(f"{records_path}:{line_number}: invalid JSON") from error
        summaries.append(summary)
    return rows, summaries


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(rows), f"cannot write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    run_root: Path,
    summary: Mapping[str, Any],
    contrast_rows: Sequence[Mapping[str, Any]],
    per_object_rows: Sequence[Mapping[str, Any]],
) -> None:
    (run_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
    )
    _write_tsv(run_root / "causal_contrasts.tsv", contrast_rows)
    _write_tsv(run_root / "per_object_metrics.tsv", per_object_rows)
    _write_tsv(
        run_root / "neighbor_stability.tsv",
        [
            {
                "object": row["object"],
                "arm": row["arm"],
                "nn_identity_retention": row["nn_identity_retention"],
                "nn_margin_mean": row["nn_margin_mean"],
            }
            for row in per_object_rows
        ],
    )
    _write_tsv(
        run_root / "geometry.tsv",
        [
            {
                "object": row["object"],
                "arm": row["arm"],
                "pair_distance_spearman": row["pair_distance_spearman"],
                "effective_rank": row["effective_rank"],
                "latent_rms": row["latent_rms"],
            }
            for row in per_object_rows
        ],
    )
    mechanism = summary["mechanism"]
    lines = [
        f"claim={summary['claim_verdict']}",
        f"diagnostic={summary['diagnostic_verdict']}",
        f"dvt={mechanism['dvt']['label']}",
        f"flow={mechanism['flow']['label']}",
        f"density={mechanism['density']['label']}",
        f"fusion={mechanism['fusion']['label']}",
        f"postprocessing={mechanism['postprocessing']['label']}",
        "public_shadow_authorized=false",
    ]
    (run_root / "verdict.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Optional local MVTec AD2 root for strict decoded-RGB custody verification.",
    )
    parser.add_argument(
        "--require-all8",
        action="store_true",
        help="Compatibility flag; all-eight coverage is always a strict Stage-M1 requirement.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_root = args.run_root.resolve(strict=True)
    rows, summaries = load_run(run_root)
    summary, contrast_rows, per_object_rows = aggregate_rows(
        rows, summaries, data_root=args.data_root,
    )
    write_outputs(run_root, summary, contrast_rows, per_object_rows)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
