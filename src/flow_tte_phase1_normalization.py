"""Label-free score normalization diagnostics for retained FlowTTE maps."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_auc_score

from src.flow_tte_gap_decomposition import (
    load_records,
    locate_map_dir,
    oracle_f1,
    parse_condition,
    safe_ratio,
)

FloatArray = npt.NDArray[np.float32]
Record = Dict[str, Any]
EPS = 1e-8
MAD_SCALE = 1.4826
LOWER_QUANTILES = (0.5, 0.6, 0.7, 0.8)
SHRINKAGE_LAMBDAS = (0.25, 0.5, 0.75)
QUANTILE_MATCH_KNOTS = 4096
SUPPLEMENTARY_VARIANTS = (
    "condition_group_quantile_match_to_regular_q4096",
    "condition_tail_affine_to_regular",
)
OBJECTS = (
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "vial",
    "wallplugs",
    "sheet_metal",
    "walnuts",
)
IDENTITY_F1_REFERENCE = {
    "can": 0.0007,
    "fabric": 0.6979,
    "fruit_jelly": 0.4768,
    "rice": 0.7125,
    "vial": 0.4396,
    "wallplugs": 0.6657,
    "sheet_metal": 0.5161,
    "walnuts": 0.7357,
}
IDENTITY_MEAN_F1_REFERENCE = 0.5306351899731461
CLAIM_SCOPE = (
    "MVTec AD2 public shadow (test_public), diagnostic only; oracle sweeps are "
    "development-only and are not deployable threshold estimates"
)
SUPPORT_DEVIATION = (
    "Deviation from plan section 6.2: support-side score statistics are unavailable without a "
    "GPU rerun. Shrinkage therefore uses the per-object median over all unlabeled test-image "
    "statistics (without GT or good/bad folder labels) as a deployment-plausible calibration batch."
)
CROSS_FIT_PROCEDURE = (
    "Sort good stems and assign fold=index modulo 4. For each fold, set one threshold to the "
    "p99.9 of float16 good-pixel scores from the other three folds. Sort bad stems separately and "
    "assign fold=index modulo 4. Score every held-out good image and every assigned bad image once "
    "with its fold threshold, then pool pixel TP/FP/FN. This is a transductive diagnostic."
)


def lower_region(score: npt.ArrayLike, quantile: float) -> FloatArray:
    """Return pixels at or below an image's requested quantile."""
    array = np.asarray(score, dtype=np.float32)
    cutoff = np.quantile(array, quantile)
    return np.asarray(array[array <= cutoff], dtype=np.float32)


def median_mad(values: npt.ArrayLike) -> tuple[float, float]:
    """Return median and robust 1.4826-scaled MAD."""
    array = np.asarray(values, dtype=np.float32).ravel()
    center = float(np.median(array))
    sigma = float(MAD_SCALE * np.median(np.abs(array - center)))
    return center, sigma


def image_statistics(
    scores: Sequence[npt.ArrayLike],
    mode: str,
    quantile: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-image centers and scales without labels."""
    centers, scales = [], []
    for score in scores:
        array = np.asarray(score, dtype=np.float32)
        if mode == "mean_std":
            center, sigma = float(np.mean(array)), float(np.std(array))
        elif mode == "median_mad":
            center, sigma = median_mad(array)
        elif mode == "lower_median_mad" and quantile is not None:
            center, sigma = median_mad(lower_region(array, quantile))
        else:
            raise ValueError(f"unsupported statistic mode: {mode}")
        centers.append(center)
        scales.append(sigma)
    return np.asarray(centers, dtype=np.float64), np.asarray(scales, dtype=np.float64)


def apply_sigma_floor(scales: npt.ArrayLike) -> tuple[np.ndarray, dict[str, float | int]]:
    """Clamp image scales at 25% of their population median."""
    raw = np.asarray(scales, dtype=np.float64)
    population_median = float(np.median(raw))
    floor = 0.25 * population_median
    clamp_mask = raw < floor
    clamped = np.maximum(raw, floor)
    return clamped, {
        "count": int(np.count_nonzero(clamp_mask)),
        "total": int(raw.size),
        "population_median_sigma": population_median,
        "sigma_floor": floor,
    }


def normalize_per_image(
    scores: Sequence[npt.ArrayLike],
    mode: str,
    quantile: float | None = None,
) -> tuple[list[FloatArray], dict[str, float | int]]:
    """Normalize each image with label-free per-image statistics."""
    centers, raw_scales = image_statistics(scores, mode, quantile)
    scales, clamp = apply_sigma_floor(raw_scales)
    transformed = [
        np.asarray(
            (np.asarray(score, dtype=np.float32) - centers[index]) / (scales[index] + EPS),
            dtype=np.float32,
        )
        for index, score in enumerate(scores)
    ]
    return transformed, clamp


def shrinkage_normalize(
    scores: Sequence[npt.ArrayLike],
    lam: float,
    quantile: float = 0.7,
) -> tuple[list[FloatArray], dict[str, float | int]]:
    """Shrink lower-tail per-image location/log-scale toward an unlabeled population median."""
    centers, raw_scales = image_statistics(scores, "lower_median_mad", quantile)
    scales, clamp = apply_sigma_floor(raw_scales)
    center_pop, sigma_pop = float(np.median(centers)), float(np.median(scales))
    log_scales = np.log(np.maximum(scales, EPS))
    log_sigma_pop = float(np.log(max(sigma_pop, EPS)))
    mixed_centers = (1.0 - lam) * center_pop + lam * centers
    mixed_scales = np.exp((1.0 - lam) * log_sigma_pop + lam * log_scales)
    transformed = [
        np.asarray(
            (np.asarray(score, dtype=np.float32) - mixed_centers[index])
            / (mixed_scales[index] + EPS),
            dtype=np.float32,
        )
        for index, score in enumerate(scores)
    ]
    return transformed, {
        **clamp,
        "population_center": center_pop,
        "population_sigma": sigma_pop,
    }


def condition_group_statistics(
    scores: Sequence[npt.ArrayLike],
    stems: Sequence[str],
    quantile: float = 0.7,
) -> tuple[dict[str, tuple[float, float]], dict[str, float | int]]:
    """Compute pooled lower-tail statistics for filename-derived condition groups."""
    grouped: dict[str, list[FloatArray]] = defaultdict(list)
    for score, stem in zip(scores, stems):
        grouped[parse_condition(stem)].append(np.asarray(score, dtype=np.float32).ravel())
    _, raw_image_scales = image_statistics(scores, "lower_median_mad", quantile)
    _, image_clamp = apply_sigma_floor(raw_image_scales)
    group_floor = float(image_clamp["sigma_floor"])
    output, group_clamps = {}, 0
    for condition, images in sorted(grouped.items()):
        pooled = np.concatenate(images)
        center, raw_sigma = median_mad(lower_region(pooled, quantile))
        group_clamps += int(raw_sigma < group_floor)
        output[condition] = (center, max(raw_sigma, group_floor))
    return output, {
        "count": group_clamps,
        "total": len(output),
        "population_median_sigma": image_clamp["population_median_sigma"],
        "sigma_floor": group_floor,
    }


def condition_group_normalize(
    scores: Sequence[npt.ArrayLike],
    stems: Sequence[str],
    quantile: float = 0.7,
) -> tuple[list[FloatArray], dict[str, float | int]]:
    """Normalize images with their condition group's pooled lower-tail statistics."""
    statistics, clamp = condition_group_statistics(scores, stems, quantile)
    transformed = []
    for score, stem in zip(scores, stems):
        center, sigma = statistics[parse_condition(stem)]
        transformed.append(np.asarray((np.asarray(score) - center) / (sigma + EPS), dtype=np.float32))
    return transformed, clamp


def condition_affine_to_regular(
    scores: Sequence[npt.ArrayLike],
    stems: Sequence[str],
    quantile: float = 0.7,
) -> tuple[list[FloatArray], dict[str, float | int]]:
    """Affine-match every condition's lower-tail statistics to regular, leaving regular untouched."""
    statistics, clamp = condition_group_statistics(scores, stems, quantile)
    if "regular" not in statistics:
        raise ValueError("condition_affine_to_regular requires a regular condition group")
    regular_center, regular_sigma = statistics["regular"]
    transformed = []
    for score, stem in zip(scores, stems):
        condition = parse_condition(stem)
        array = np.asarray(score, dtype=np.float32)
        if condition == "regular":
            transformed.append(array.copy())
            continue
        center, sigma = statistics[condition]
        matched = ((array - center) / (sigma + EPS)) * regular_sigma + regular_center
        transformed.append(np.asarray(matched, dtype=np.float32))
    return transformed, clamp


def _condition_pools(
    scores: Sequence[npt.ArrayLike],
    stems: Sequence[str],
) -> dict[str, FloatArray]:
    """Pool all pixels by filename-derived condition without consulting labels."""
    grouped: dict[str, list[FloatArray]] = defaultdict(list)
    for score, stem in zip(scores, stems):
        grouped[parse_condition(stem)].append(np.asarray(score, dtype=np.float32).ravel())
    return {
        condition: np.concatenate(images).astype(np.float32, copy=False)
        for condition, images in sorted(grouped.items())
    }


def condition_group_quantile_match_to_regular(
    scores: Sequence[npt.ArrayLike],
    stems: Sequence[str],
    knots: int = QUANTILE_MATCH_KNOTS,
) -> tuple[list[FloatArray], dict[str, Any]]:
    """Empirically quantile-match each condition's pooled scores to regular."""
    if knots < 2:
        raise ValueError("quantile matching requires at least two knots")
    pools = _condition_pools(scores, stems)
    if "regular" not in pools:
        raise ValueError("condition quantile matching requires a regular condition group")
    probabilities = np.linspace(0.0, 1.0, knots, dtype=np.float64)
    regular_knots = np.quantile(pools["regular"], probabilities).astype(np.float64)
    mappings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    monotone_by_condition: dict[str, bool] = {"regular": True}
    effective_knots: dict[str, int] = {"regular": knots}
    for condition, pool in pools.items():
        if condition == "regular":
            continue
        source_knots = np.quantile(pool, probabilities).astype(np.float64)
        unique_source, first_indices = np.unique(source_knots, return_index=True)
        target_knots = regular_knots[first_indices]
        mappings[condition] = (unique_source, target_knots)
        effective_knots[condition] = int(unique_source.size)
        monotone_by_condition[condition] = bool(
            np.all(np.diff(unique_source) >= 0.0) and np.all(np.diff(target_knots) >= 0.0)
        )
    if not all(monotone_by_condition.values()):
        raise RuntimeError("condition quantile mapping is not monotone")
    transformed = []
    for score, stem in zip(scores, stems):
        condition = parse_condition(stem)
        array = np.asarray(score, dtype=np.float32)
        if condition == "regular":
            transformed.append(array.copy())
            continue
        source_knots, target_knots = mappings[condition]
        matched = np.interp(
            array.ravel().astype(np.float64),
            source_knots,
            target_knots,
            left=float(target_knots[0]),
            right=float(target_knots[-1]),
        ).reshape(array.shape)
        transformed.append(np.asarray(matched, dtype=np.float32))
    return transformed, {
        "count": 0,
        "total": len(mappings),
        "requested_knots": knots,
        "effective_knots": effective_knots,
        "monotone_by_condition": monotone_by_condition,
    }


def condition_tail_affine_to_regular(
    scores: Sequence[npt.ArrayLike],
    stems: Sequence[str],
) -> tuple[list[FloatArray], dict[str, Any]]:
    """Map each condition's pooled p99/p99.9 anchors to regular by affine transform."""
    pools = _condition_pools(scores, stems)
    if "regular" not in pools:
        raise ValueError("condition tail affine matching requires a regular condition group")
    anchors = {
        condition: np.quantile(pool, (0.99, 0.999)).astype(np.float64)
        for condition, pool in pools.items()
    }
    regular_low, regular_high = anchors["regular"]
    transforms: dict[str, tuple[float, float]] = {}
    degenerate_conditions: list[str] = []
    for condition, (low, high) in anchors.items():
        if condition == "regular":
            continue
        denominator = float(high - low)
        if denominator <= 0.0:
            transforms[condition] = (1.0, 0.0)
            degenerate_conditions.append(condition)
            continue
        slope = float((regular_high - regular_low) / denominator)
        if not np.isfinite(slope) or slope <= 0.0:
            transforms[condition] = (1.0, 0.0)
            degenerate_conditions.append(condition)
            continue
        transforms[condition] = (slope, float(regular_low - slope * low))
    transformed = []
    for score, stem in zip(scores, stems):
        condition = parse_condition(stem)
        array = np.asarray(score, dtype=np.float32)
        if condition == "regular":
            transformed.append(array.copy())
            continue
        slope, offset = transforms[condition]
        transformed.append(np.asarray(slope * array + offset, dtype=np.float32))
    return transformed, {
        "count": len(degenerate_conditions),
        "total": len(transforms),
        "degenerate_conditions": degenerate_conditions,
        "anchors": {name: values.tolist() for name, values in anchors.items()},
        "affine": {
            name: {"slope": values[0], "offset": values[1]}
            for name, values in transforms.items()
        },
    }


def variant_names() -> list[str]:
    """Return the preregistered object-independent variant sweep."""
    return [
        "identity",
        "query_mean_std",
        "query_median_mad",
        *(f"lower_quantile_median_mad_q{q:g}" for q in LOWER_QUANTILES),
        *(f"shrinkage_lambda_{lam:g}" for lam in SHRINKAGE_LAMBDAS),
        "condition_group_median_mad_q0.7",
        "condition_affine_to_regular_q0.7",
    ]


def transform_variant(records: Sequence[Record], variant: str) -> tuple[list[FloatArray], dict[str, Any]]:
    """Create one transformed map per record without consulting labels or split names."""
    scores = [record["score"] for record in records]
    stems = [str(record["stem"]) for record in records]
    if variant == "identity":
        return scores, {"count": 0, "total": 0}
    if variant == "query_mean_std":
        return normalize_per_image(scores, "mean_std")
    if variant == "query_median_mad":
        return normalize_per_image(scores, "median_mad")
    if variant.startswith("lower_quantile_median_mad_q"):
        return normalize_per_image(scores, "lower_median_mad", float(variant.rsplit("q", 1)[1]))
    if variant.startswith("shrinkage_lambda_"):
        return shrinkage_normalize(scores, float(variant.rsplit("_", 1)[1]))
    if variant == "condition_group_median_mad_q0.7":
        return condition_group_normalize(scores, stems)
    if variant == "condition_affine_to_regular_q0.7":
        return condition_affine_to_regular(scores, stems)
    if variant == "condition_group_quantile_match_to_regular_q4096":
        return condition_group_quantile_match_to_regular(scores, stems)
    if variant == "condition_tail_affine_to_regular":
        return condition_tail_affine_to_regular(scores, stems)
    raise ValueError(f"unknown variant: {variant}")


def fixed_f1(labels: Sequence[npt.ArrayLike], predictions: Sequence[npt.ArrayLike]) -> float:
    """Compute pooled binary F1 using the Phase 0 safe-ratio helper."""
    tp = fp = fn = 0
    for label, prediction in zip(labels, predictions):
        truth = np.asarray(label, dtype=np.bool_)
        pred = np.asarray(prediction, dtype=np.bool_)
        tp += int(np.count_nonzero(pred & truth))
        fp += int(np.count_nonzero(pred & ~truth))
        fn += int(np.count_nonzero(~pred & truth))
    precision, recall = safe_ratio(tp, tp + fp), safe_ratio(tp, tp + fn)
    return safe_ratio(2.0 * precision * recall, precision + recall)


def normal_fpr(records: Sequence[Record], scores: Sequence[FloatArray], threshold: float) -> dict[str, Any]:
    """Evaluate false-positive rates on normal images at a pooled oracle threshold."""
    rows, positive, pixels = [], 0, 0
    for record, score in zip(records, scores):
        if record["split"] != "good":
            continue
        prediction = score.astype(np.float16) >= np.float16(threshold)
        count = int(np.count_nonzero(prediction))
        positive += count
        pixels += int(prediction.size)
        rows.append({"stem": record["stem"], "fpr": safe_ratio(count, prediction.size)})
    return {
        "mean_per_image": float(np.mean([row["fpr"] for row in rows])),
        "pooled_pixel": safe_ratio(positive, pixels),
        "images": rows,
    }


def cross_fit_fixed_f1(records: Sequence[Record], scores: Sequence[FloatArray]) -> dict[str, Any]:
    """Evaluate deterministic four-fold transductive fixed thresholds."""
    assignments: dict[int, int] = {}
    for split in ("good", "bad"):
        indices = sorted(
            (index for index, record in enumerate(records) if record["split"] == split),
            key=lambda index: str(records[index]["stem"]),
        )
        assignments.update({index: order % 4 for order, index in enumerate(indices)})
    thresholds = []
    for fold in range(4):
        training = [
            scores[index].astype(np.float16).ravel()
            for index, record in enumerate(records)
            if record["split"] == "good" and assignments[index] != fold
        ]
        if not training:
            raise ValueError(f"fold {fold} has no disjoint good-image calibration pixels")
        thresholds.append(float(np.quantile(np.concatenate(training), 0.999)))
    predictions = [
        score.astype(np.float16) >= np.float16(thresholds[assignments[index]])
        for index, score in enumerate(scores)
    ]
    return {
        "f1": fixed_f1([record["gt"] for record in records], predictions),
        "thresholds": thresholds,
        "procedure": CROSS_FIT_PROCEDURE,
        "label": "transductive_diagnostic",
    }


def per_bad_image_oracle(records: Sequence[Record], scores: Sequence[FloatArray]) -> dict[str, Any]:
    """Compute mean oracle F1 across anomalous images."""
    rows = []
    for record, score in zip(records, scores):
        if record["split"] != "bad":
            continue
        result = oracle_f1(record["gt"], score, cast_float16=True)
        rows.append({"stem": record["stem"], **result})
    return {"mean_f1": float(np.mean([row["f1"] for row in rows])), "images": rows}


def large_defect_f1(
    records: Sequence[Record],
    scores: Sequence[FloatArray],
    threshold: float,
) -> dict[str, Any]:
    """Evaluate pooled-threshold F1 on bad images at or above the GT-area p90."""
    bad = [index for index, record in enumerate(records) if record["split"] == "bad"]
    areas = np.asarray([np.count_nonzero(records[index]["gt"]) for index in bad], dtype=np.int64)
    cutoff = float(np.quantile(areas, 0.9))
    selected = [index for index, area in zip(bad, areas) if area >= cutoff]
    predictions = [scores[index].astype(np.float16) >= np.float16(threshold) for index in selected]
    return {
        "f1": fixed_f1([records[index]["gt"] for index in selected], predictions),
        "gt_area_p90": cutoff,
        "n_images": len(selected),
        "stems": [records[index]["stem"] for index in selected],
        "threshold_source": "variant pooled oracle threshold; subset contains bad images only",
    }


def evaluate_variant(records: Sequence[Record], scores: Sequence[FloatArray]) -> dict[str, Any]:
    """Compute all Phase 1 metrics for one object/variant."""
    labels = np.concatenate([np.asarray(record["gt"]).ravel() for record in records])
    pooled_scores = np.concatenate([score.ravel() for score in scores])
    pooled = oracle_f1(labels, pooled_scores, cast_float16=True)
    partial_auc = float(
        roc_auc_score(labels.astype(np.uint8), pooled_scores.astype(np.float16), max_fpr=0.05),
    )
    return {
        "pooled_oracle_f1": pooled["f1"],
        "pooled_oracle_threshold": pooled["threshold"],
        "seg_AUROC_0.05": partial_auc,
        "normal_image_fpr": normal_fpr(records, scores, pooled["threshold"]),
        "cross_fit_fixed_threshold": cross_fit_fixed_f1(records, scores),
        "per_bad_image_oracle": per_bad_image_oracle(records, scores),
        "large_defect_subset": large_defect_f1(records, scores, pooled["threshold"]),
    }


def aggregate_variant(objects: Mapping[str, dict[str, Any]]) -> dict[str, float]:
    """Compute unweighted eight-object metric means."""
    rows = list(objects.values())
    return {
        "pooled_oracle_f1": float(np.mean([row["pooled_oracle_f1"] for row in rows])),
        "seg_AUROC_0.05": float(np.mean([row["seg_AUROC_0.05"] for row in rows])),
        "normal_image_fpr_mean_per_image": float(
            np.mean([row["normal_image_fpr"]["mean_per_image"] for row in rows]),
        ),
        "normal_image_fpr_pooled_pixel": float(
            np.mean([row["normal_image_fpr"]["pooled_pixel"] for row in rows]),
        ),
        "cross_fit_fixed_f1": float(
            np.mean([row["cross_fit_fixed_threshold"]["f1"] for row in rows]),
        ),
        "per_bad_image_oracle_f1": float(
            np.mean([row["per_bad_image_oracle"]["mean_f1"] for row in rows]),
        ),
        "large_defect_subset_f1": float(
            np.mean([row["large_defect_subset"]["f1"] for row in rows]),
        ),
    }


def regular_condition_identity_check(
    records: Sequence[Record],
    transformed_scores: Sequence[FloatArray],
) -> dict[str, Any]:
    """Require exact float16-path oracle-F1 parity on the untouched regular group."""
    indices = [
        index
        for index, record in enumerate(records)
        if parse_condition(str(record["stem"])) == "regular"
    ]
    if not indices:
        raise ValueError("regular-condition parity check requires regular records")
    labels = np.concatenate([np.asarray(records[index]["gt"]).ravel() for index in indices])
    identity_scores = np.concatenate(
        [np.asarray(records[index]["score"], dtype=np.float32).ravel() for index in indices]
    )
    variant_scores = np.concatenate(
        [np.asarray(transformed_scores[index], dtype=np.float32).ravel() for index in indices]
    )
    identity = oracle_f1(labels, identity_scores, cast_float16=True)
    variant = oracle_f1(labels, variant_scores, cast_float16=True)
    identity_bits = int(np.asarray(identity["f1"], dtype=np.float64).view(np.uint64))
    variant_bits = int(np.asarray(variant["f1"], dtype=np.float64).view(np.uint64))
    score_arrays_equal = bool(
        np.array_equal(identity_scores.astype(np.float16), variant_scores.astype(np.float16))
    )
    passed = identity_bits == variant_bits
    if not passed:
        raise RuntimeError(
            "regular-condition float16 oracle F1 changed: "
            f"identity={identity['f1']} variant={variant['f1']}"
        )
    return {
        "pass": passed,
        "identity_f1": identity["f1"],
        "variant_f1": variant["f1"],
        "identity_f1_float64_bits": f"0x{identity_bits:016x}",
        "variant_f1_float64_bits": f"0x{variant_bits:016x}",
        "float16_score_arrays_equal": score_arrays_equal,
        "regular_images": len(indices),
        "regular_pixels": int(labels.size),
    }


def identity_parity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Compare identity F1 against the retained Phase 0 evaluator-parity anchor."""
    observed = {name: float(row["pooled_oracle_f1"]) for name, row in identity["objects"].items()}
    object_pass = {
        name: round(observed[name], 4) == reference
        for name, reference in IDENTITY_F1_REFERENCE.items()
    }
    mean_value = float(identity["mean"]["pooled_oracle_f1"])
    return {
        "observed": observed,
        "reference_rounded_4dp": IDENTITY_F1_REFERENCE,
        "object_pass_rounded_4dp": object_pass,
        "observed_mean": mean_value,
        "reference_mean": IDENTITY_MEAN_F1_REFERENCE,
        "mean_absolute_error": abs(mean_value - IDENTITY_MEAN_F1_REFERENCE),
        "pass": all(object_pass.values()) and abs(mean_value - IDENTITY_MEAN_F1_REFERENCE) <= 1e-12,
    }


def keep_gate(variant: Mapping[str, Any], identity: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate, but do not decide, the preregistered Phase 1 KEEP criteria."""
    mean_f1 = float(variant["mean"]["pooled_oracle_f1"])
    cross_fit = float(variant["mean"]["cross_fit_fixed_f1"])
    normal_fpr_value = float(variant["mean"]["normal_image_fpr_mean_per_image"])
    identity_cross_fit = float(identity["mean"]["cross_fit_fixed_f1"])
    identity_fpr = float(identity["mean"]["normal_image_fpr_mean_per_image"])
    fabric_delta = float(
        variant["objects"]["fabric"]["pooled_oracle_f1"]
        - identity["objects"]["fabric"]["pooled_oracle_f1"],
    )
    wallplugs_delta = float(
        variant["objects"]["wallplugs"]["pooled_oracle_f1"]
        - identity["objects"]["wallplugs"]["pooled_oracle_f1"],
    )
    checks = {
        "mean_pooled_oracle_f1_at_least_0.5456": mean_f1 >= 0.5306 + 0.015,
        "cross_fit_fixed_f1_improved_vs_identity": cross_fit > identity_cross_fit,
        "mean_normal_image_fpr_not_worse_than_identity": normal_fpr_value <= identity_fpr,
        "fabric_or_wallplugs_f1_gain_at_least_0.03": max(fabric_delta, wallplugs_delta) >= 0.03,
    }
    return {
        "all_checks_pass": all(checks.values()),
        "checks": checks,
        "values": {
            "mean_pooled_oracle_f1": mean_f1,
            "required_mean_pooled_oracle_f1": 0.5456,
            "cross_fit_fixed_f1": cross_fit,
            "identity_cross_fit_fixed_f1": identity_cross_fit,
            "mean_normal_image_fpr": normal_fpr_value,
            "identity_mean_normal_image_fpr": identity_fpr,
            "fabric_f1_delta": fabric_delta,
            "wallplugs_f1_delta": wallplugs_delta,
        },
        "decision_owner": "requester",
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_leaderboard(path: Path, variants: Mapping[str, dict[str, Any]]) -> None:
    fields = ["variant", "mean_f1", "mean_seg_AUROC_0.05"]
    for object_name in OBJECTS:
        fields.extend((f"{object_name}_f1", f"{object_name}_seg_AUROC_0.05"))
    rows = []
    for name, payload in variants.items():
        row: dict[str, Any] = {
            "variant": name,
            "mean_f1": payload["mean"]["pooled_oracle_f1"],
            "mean_seg_AUROC_0.05": payload["mean"]["seg_AUROC_0.05"],
        }
        for object_name in OBJECTS:
            metrics = payload["objects"][object_name]
            row[f"{object_name}_f1"] = metrics["pooled_oracle_f1"]
            row[f"{object_name}_seg_AUROC_0.05"] = metrics["seg_AUROC_0.05"]
        rows.append(row)
    rows.sort(key=lambda row: float(row["mean_f1"]), reverse=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def analyze_object_variants(
    result_root: Path,
    data_root: Path,
    object_name: str,
) -> dict[str, Any]:
    """Load one object's maps once and evaluate every preregistered variant."""
    print(f"loading object={object_name} map_dir={locate_map_dir(result_root, object_name)}", flush=True)
    records = load_records(result_root, data_root, object_name)
    print(f"loaded object={object_name} images={len(records)}", flush=True)
    metrics, clamps = {}, {}
    for name in variant_names():
        transformed, clamp = transform_variant(records, name)
        metrics[name] = evaluate_variant(records, transformed)
        clamps[name] = clamp
        print(
            f"evaluated object={object_name} variant={name} "
            f"f1={metrics[name]['pooled_oracle_f1']:.6f}",
            flush=True,
        )
        if name == "identity":
            observed = metrics[name]["pooled_oracle_f1"]
            reference = IDENTITY_F1_REFERENCE[object_name]
            if round(float(observed), 4) != reference:
                raise RuntimeError(
                    f"identity parity failed for {object_name}: "
                    f"observed={observed}, expected_rounded_4dp={reference}",
                )
    return {"object": object_name, "metrics": metrics, "clamps": clamps}


def analyze_object_supplementary(
    result_root: Path,
    data_root: Path,
    object_name: str,
) -> dict[str, Any]:
    """Evaluate only the two tail-anchored supplementary variants for one object."""
    print(f"loading object={object_name} map_dir={locate_map_dir(result_root, object_name)}", flush=True)
    records = load_records(result_root, data_root, object_name)
    print(f"loaded object={object_name} images={len(records)}", flush=True)
    metrics, diagnostics, regular_checks = {}, {}, {}
    for name in SUPPLEMENTARY_VARIANTS:
        transformed, diagnostic = transform_variant(records, name)
        metrics[name] = evaluate_variant(records, transformed)
        diagnostics[name] = diagnostic
        regular_checks[name] = regular_condition_identity_check(records, transformed)
        print(
            f"evaluated object={object_name} variant={name} "
            f"f1={metrics[name]['pooled_oracle_f1']:.6f} "
            f"regular_bitwise_identity={regular_checks[name]['pass']}",
            flush=True,
        )
    return {
        "object": object_name,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "regular_condition_identity_checks": regular_checks,
    }


def analyze_supplementary_run(
    result_root: Path,
    data_root: Path,
    objects: Sequence[str],
    output_dir: Path,
    workers: int = 1,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Evaluate and write only collision-safe supplementary Phase-1 artifacts."""
    if tuple(objects) != OBJECTS:
        raise ValueError(f"Phase 1 requires the fixed object order/set: {OBJECTS}")
    if workers < 1:
        raise ValueError("workers must be positive")
    targets = [
        *(output_dir / f"{name}.json" for name in SUPPLEMENTARY_VARIANTS),
        output_dir / "supplementary_leaderboard.tsv",
        output_dir / "supplementary_metadata.json",
    ]
    collisions = [str(path) for path in targets if path.exists()]
    if collisions:
        raise FileExistsError(
            "refusing to overwrite supplementary Phase-1 artifacts: " + ", ".join(collisions)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    variants: dict[str, dict[str, Any]] = {
        name: {
            "variant": name,
            "objects": {},
            "condition_diagnostics": {},
            "regular_condition_identity_checks": {},
        }
        for name in SUPPLEMENTARY_VARIANTS
    }
    if workers == 1:
        object_results = [
            analyze_object_supplementary(result_root, data_root, name) for name in objects
        ]
    else:
        object_results = []
        with ProcessPoolExecutor(max_workers=min(workers, len(objects))) as executor:
            futures = {
                executor.submit(analyze_object_supplementary, result_root, data_root, name): name
                for name in objects
            }
            for future in as_completed(futures):
                result = future.result()
                object_results.append(result)
                progress(f"completed object={result['object']}")
    for result in object_results:
        object_name = result["object"]
        for name in SUPPLEMENTARY_VARIANTS:
            variants[name]["objects"][object_name] = result["metrics"][name]
            variants[name]["condition_diagnostics"][object_name] = result["diagnostics"][name]
            variants[name]["regular_condition_identity_checks"][object_name] = result[
                "regular_condition_identity_checks"
            ][name]
    for payload in variants.values():
        payload["mean"] = aggregate_variant(payload["objects"])
        payload["metadata"] = {
            "claim_scope": CLAIM_SCOPE,
            "normalization_is_label_free": True,
            "transform_inputs": "score maps and filename-derived condition suffixes only",
            "threshold_source": "pooled test_public oracle float16, diagnostic only",
        }
    metadata = {
        "claim_scope": CLAIM_SCOPE,
        "result_root_read_only": str(result_root),
        "data_root": str(data_root),
        "objects": list(objects),
        "variants": list(SUPPLEMENTARY_VARIANTS),
        "evaluated_only_supplementary_variants": True,
        "quantile_match_knots": QUANTILE_MATCH_KNOTS,
        "cross_fit_fixed_threshold_procedure": CROSS_FIT_PROCEDURE,
        "regular_condition_identity_checks": {
            name: payload["regular_condition_identity_checks"]
            for name, payload in variants.items()
        },
    }
    for name, payload in variants.items():
        write_json(output_dir / f"{name}.json", payload)
    write_leaderboard(output_dir / "supplementary_leaderboard.tsv", variants)
    write_json(output_dir / "supplementary_metadata.json", metadata)
    progress("phase1_supplementary_complete")
    return {"metadata": metadata, "variants": variants}


def analyze_run(
    result_root: Path,
    data_root: Path,
    objects: Sequence[str],
    output_dir: Path,
    workers: int = 1,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Load each object's maps once, run all variants, and write compact artifacts."""
    if tuple(objects) != OBJECTS:
        raise ValueError(f"Phase 1 requires the fixed object order/set: {OBJECTS}")
    output_dir.mkdir(parents=True, exist_ok=True)
    variants: dict[str, dict[str, Any]] = {
        name: {"variant": name, "objects": {}, "clamp_counts": {}} for name in variant_names()
    }
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        object_results = [analyze_object_variants(result_root, data_root, name) for name in objects]
    else:
        object_results = []
        with ProcessPoolExecutor(max_workers=min(workers, len(objects))) as executor:
            futures = {
                executor.submit(analyze_object_variants, result_root, data_root, name): name
                for name in objects
            }
            for future in as_completed(futures):
                result = future.result()
                object_results.append(result)
                progress(f"completed object={result['object']}")
    for result in object_results:
        object_name = result["object"]
        for name in variant_names():
            variants[name]["objects"][object_name] = result["metrics"][name]
            variants[name]["clamp_counts"][object_name] = result["clamps"][name]
    for payload in variants.values():
        payload["mean"] = aggregate_variant(payload["objects"])
        payload["metadata"] = {
            "claim_scope": CLAIM_SCOPE,
            "normalization_is_label_free": True,
            "threshold_source": "pooled test_public oracle float16, diagnostic only",
        }
    parity = identity_parity(variants["identity"])
    if not parity["pass"]:
        failure_metadata = {
            "claim_scope": CLAIM_SCOPE,
            "status": "identity_parity_failed",
            "identity_parity": parity,
            "keep_gates": {},
            "interpretation_blocked": True,
        }
        write_json(output_dir / "metadata.json", failure_metadata)
        write_json(output_dir / "identity.json", variants["identity"])
        write_leaderboard(output_dir / "leaderboard.tsv", variants)
        progress("identity_parity_pass=False; KEEP gates and non-identity interpretation suppressed")
        raise RuntimeError("identity mean parity failed; stopping before KEEP-gate interpretation")
    gates = {name: keep_gate(payload, variants["identity"]) for name, payload in variants.items()}
    metadata = {
        "claim_scope": CLAIM_SCOPE,
        "result_root_read_only": str(result_root),
        "data_root": str(data_root),
        "objects": list(objects),
        "variants": variant_names(),
        "eps": EPS,
        "sigma_floor": "sigma >= 0.25 * per-object median raw image sigma",
        "support_stat_deviation": SUPPORT_DEVIATION,
        "shrinkage_lambda_zero_note": (
            "lambda=0 is a per-object global affine transform and therefore a no-op for pooled "
            "ranking/F1; it is noted rather than computed"
        ),
        "cross_fit_fixed_threshold_procedure": CROSS_FIT_PROCEDURE,
        "large_defect_subset_procedure": (
            "Bad images with GT area >= the per-object p90 (ties included), evaluated at the "
            "variant pooled oracle threshold; GT is used only for evaluation/subset definition."
        ),
        "identity_parity": parity,
        "keep_gates": gates,
        "clamp_counts": {name: payload["clamp_counts"] for name, payload in variants.items()},
    }
    for name, payload in variants.items():
        payload["keep_gate"] = gates[name]
        write_json(output_dir / f"{name}.json", payload)
    write_leaderboard(output_dir / "leaderboard.tsv", variants)
    write_json(output_dir / "metadata.json", metadata)
    progress(f"identity_parity_pass={parity['pass']}")
    for name in variant_names():
        checks = gates[name]["checks"]
        progress(
            f"KEEP_GATE variant={name} all={gates[name]['all_checks_pass']} "
            + " ".join(f"{key}={value}" for key, value in checks.items()),
        )
    progress("phase1_complete")
    return {"metadata": metadata, "variants": variants}
