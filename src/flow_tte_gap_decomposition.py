"""Class-agnostic diagnostics for retained FlowTTE anomaly maps."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np
import numpy.typing as npt
import tifffile
from PIL import Image
from scipy import ndimage
from sklearn.metrics import precision_recall_curve, roc_auc_score

if TYPE_CHECKING:
    from pathlib import Path

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float32]
CONDITION_RE = re.compile(r"_(regular|overexposed|underexposed|shift_\d+)$")
TOLERANCES = (0, 2, 4, 8, 12, 16, 32, 64)
EROSIONS = (4, 8, 16)
QUANTILES = (0.5, 0.6, 0.7, 0.8)
STRUCTURE_8 = np.ones((3, 3), dtype=np.uint8)
CLAIM_SCOPE = (
    "diagnostic_only_on_ad2_public_shadow_development_data; "
    "not confirmatory evidence and not a basis for per-object tuning"
)


def parse_condition(stem: str) -> str:
    """Parse the preregistered lighting suffix from an image stem."""
    match = CONDITION_RE.search(stem)
    if match is None:
        message = f"unrecognized lighting condition suffix: {stem}"
        raise ValueError(message)
    return match.group(1)


def oracle_f1(labels: npt.ArrayLike, scores: npt.ArrayLike, cast_float16: bool) -> dict[str, float]:
    """Match post_eval.py's pooled PR-curve maximum and threshold selection."""
    y_true = np.asarray(labels, dtype=np.uint8).ravel()
    dtype = np.float16 if cast_float16 else np.float32
    y_score = np.asarray(scores, dtype=dtype).ravel()
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2.0 * precision * recall / (precision + recall)
    finite = np.isfinite(f1)
    best_index = int(np.argmax(f1[finite]))
    return {"f1": float(np.max(f1[finite])), "threshold": float(thresholds[best_index])}


def boundary_tolerant_counts(
    prediction: BoolArray,
    gt: BoolArray,
    tolerance: int,
) -> tuple[int, int, int, int]:
    """Return tolerant precision/recall numerators and denominators."""
    dilated_gt = dilate(gt, tolerance)
    dilated_prediction = dilate(prediction, tolerance)
    return (
        int(np.count_nonzero(prediction & dilated_gt)),
        int(np.count_nonzero(prediction)),
        int(np.count_nonzero(gt & dilated_prediction)),
        int(np.count_nonzero(gt)),
    )


def boundary_tolerant_f1(prediction: BoolArray, gt: BoolArray, tolerance: int) -> float:
    """Compute symmetric tolerance F1 for one mask pair."""
    p_num, p_den, r_num, r_den = boundary_tolerant_counts(prediction, gt, tolerance)
    return harmonic(safe_ratio(p_num, p_den), safe_ratio(r_num, r_den))


def component_recall(prediction: BoolArray, gt: BoolArray) -> dict[str, int | float]:
    """Recall GT 8-connected components hit by at least one predicted pixel."""
    labels, count = ndimage.label(gt, structure=STRUCTURE_8)
    hits = sum(bool(np.any(prediction[labels == index])) for index in range(1, count + 1))
    return {"hit": int(hits), "total": int(count), "recall": safe_ratio(hits, count)}


def locate_map_dir(result_root: Path, object_name: str) -> Path:
    direct = result_root / "anomaly_maps" / object_name / "test"
    candidates = (
        [direct]
        if direct.is_dir()
        else sorted(result_root.glob(f"**/anomaly_maps/{object_name}/test"))
    )
    if len(candidates) != 1:
        message = f"expected one anomaly-map directory for {object_name}, found {len(candidates)}"
        raise RuntimeError(message)
    return candidates[0]


def load_records(result_root: Path, data_root: Path, object_name: str) -> list[dict[str, Any]]:
    """Load maps, masks, and verified RGB paths for one object."""
    map_dir = locate_map_dir(result_root, object_name)
    records = []
    for map_path in sorted(map_dir.glob("*/*.tiff")):
        split, stem = map_path.parent.name, map_path.stem
        if split not in {"good", "bad"}:
            continue
        score = np.asarray(tifffile.imread(map_path), dtype=np.float32)
        rgb_path = data_root / object_name / "test_public" / split / f"{stem}.png"
        if not rgb_path.is_file():
            message = f"missing RGB image corresponding to {map_path}: {rgb_path}"
            raise FileNotFoundError(message)
        gt = load_gt(data_root, object_name, split, stem, score.shape)
        records.append(
            {
                "split": split,
                "stem": stem,
                "condition": parse_condition(stem),
                "map_path": str(map_path),
                "rgb_path": str(rgb_path),
                "score": score,
                "gt": gt,
            },
        )
    if not records or not any(record["split"] == "bad" for record in records):
        message = f"no complete good/bad TIFF set found in {map_dir}"
        raise RuntimeError(message)
    verify_complete_map_set(records, data_root, object_name)
    return records


def verify_complete_map_set(
    records: Sequence[dict[str, Any]],
    data_root: Path,
    object_name: str,
) -> None:
    """Require one retained TIFF for every public good/bad RGB image."""
    for split in ("good", "bad"):
        image_dir = data_root / object_name / "test_public" / split
        expected = {path.stem for path in image_dir.glob("*.png")}
        observed = {record["stem"] for record in records if record["split"] == split}
        if expected != observed:
            missing = sorted(expected - observed)[:5]
            extra = sorted(observed - expected)[:5]
            message = f"incomplete {object_name}/{split} map set; missing={missing}, extra={extra}"
            raise RuntimeError(message)


def load_gt(
    data_root: Path,
    object_name: str,
    split: str,
    stem: str,
    shape: Sequence[int],
) -> BoolArray:
    if split == "good":
        return np.zeros(tuple(shape), dtype=np.bool_)
    gt_path = data_root / object_name / "test_public" / "ground_truth" / split / f"{stem}_mask.png"
    if not gt_path.is_file():
        message = f"missing anomalous-image GT mask: {gt_path}"
        raise FileNotFoundError(message)
    gt = np.asarray(Image.open(gt_path)) > 0
    if gt.shape != tuple(shape):
        message = f"map/GT shape mismatch for {stem}: {shape} versus {gt.shape}"
        raise ValueError(message)
    return np.asarray(gt, dtype=np.bool_)


def analyze_object(result_root: Path, data_root: Path, object_name: str) -> dict[str, Any]:
    records = load_records(result_root, data_root, object_name)
    labels = np.concatenate([record["gt"].ravel() for record in records])
    scores = np.concatenate([record["score"].ravel() for record in records])
    pooled16 = oracle_f1(labels, scores, cast_float16=True)
    pooled32 = oracle_f1(labels, scores, cast_float16=False)
    threshold = pooled16["threshold"]
    partial_auc = float(
        roc_auc_score(labels.astype(np.uint8), scores.astype(np.float16), max_fpr=0.05),
    )
    payload: dict[str, Any] = {
        "object": object_name,
        "n_images": len(records),
        "n_good_images": sum(r["split"] == "good" for r in records),
        "n_bad_images": sum(r["split"] == "bad" for r in records),
        "pooled_oracle_float16": pooled16,
        "pooled_oracle_float32": pooled32,
        "seg_AUROC_0.05_float16": partial_auc,
    }
    per_image = per_image_analysis(records)
    payload["per_anomalous_image_oracle"] = per_image
    payload["delta_scale"] = float(per_image["mean_f1"] - pooled16["f1"])
    payload["normal_image_fpr"] = normal_fpr(records, threshold)
    payload["image_score_stats"] = [image_score_stats(record) for record in records]
    payload["condition_analysis"] = condition_analysis(records)
    payload["boundary_analysis"] = boundary_analysis(records, threshold)
    payload["component_analysis"] = component_analysis(records, threshold)
    payload["bottleneck"] = classify_bottleneck(payload)
    return payload


def per_image_analysis(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for record in records:
        if record["split"] != "bad":
            continue
        result = oracle_f1(record["gt"], record["score"], cast_float16=True)
        result.update({"stem": record["stem"], "condition": record["condition"]})
        rows.append(result)
    return {"mean_f1": float(np.mean([row["f1"] for row in rows])), "images": rows}


def normal_fpr(records: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any]:
    rows = []
    total_positive = total_pixels = 0
    for record in records:
        if record["split"] != "good":
            continue
        prediction = record["score"].astype(np.float16) >= np.float16(threshold)
        positive, pixels = int(np.count_nonzero(prediction)), int(prediction.size)
        total_positive += positive
        total_pixels += pixels
        rows.append(
            {
                "stem": record["stem"],
                "condition": record["condition"],
                "fpr": safe_ratio(positive, pixels),
            },
        )
    return {
        "mean_image_fpr": float(np.mean([row["fpr"] for row in rows])),
        "pooled_pixel_fpr": safe_ratio(total_positive, total_pixels),
        "images": rows,
    }


def image_score_stats(record: dict[str, Any]) -> dict[str, Any]:
    score, gt = record["score"], record["gt"]
    row: dict[str, Any] = {
        "stem": record["stem"],
        "split": record["split"],
        "condition": record["condition"],
        "rgb_path": record["rgb_path"],
        "map_path": record["map_path"],
        "median": float(np.median(score)),
    }
    for q in QUANTILES:
        quantile = float(np.quantile(score, q))
        region = score[score <= quantile]
        median = float(np.median(region))
        row[f"lower_q{q:g}_median"] = median
        row[f"lower_q{q:g}_mad"] = float(np.median(np.abs(region - median)))
    for q, name in ((0.9, "p90"), (0.95, "p95"), (0.99, "p99"), (0.999, "p99.9")):
        row[name] = float(np.quantile(score, q))
    if record["split"] == "bad":
        anomaly_median, background_median = (
            float(np.median(score[gt])),
            float(np.median(score[~gt])),
        )
        row.update(
            {
                "anomaly_median": anomaly_median,
                "background_median": background_median,
                "separation": anomaly_median - background_median,
            },
        )
    return row


def condition_analysis(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["condition"]].append(record)
    conditions = {}
    for condition, subset in sorted(grouped.items()):
        good_scores = [r["score"].ravel() for r in subset if r["split"] == "good"]
        labels = np.concatenate([r["gt"].ravel() for r in subset])
        scores = np.concatenate([r["score"].ravel() for r in subset])
        good = np.concatenate(good_scores) if good_scores else np.asarray([], dtype=np.float32)
        conditions[condition] = condition_row(subset, good, labels, scores)
    regular = conditions.get("regular", {}).get("good_pixel_median")
    offsets = {
        name: None
        if regular is None or row["good_pixel_median"] is None
        else float(row["good_pixel_median"] - regular)
        for name, row in conditions.items()
    }
    return {"conditions": conditions, "good_median_offset_from_regular": offsets}


def condition_row(
    subset: Sequence[dict[str, Any]],
    good: FloatArray,
    labels: BoolArray,
    scores: FloatArray,
) -> dict[str, Any]:
    has_both = bool(np.any(labels)) and bool(np.any(~labels))
    return {
        "n_good_images": sum(r["split"] == "good" for r in subset),
        "n_bad_images": sum(r["split"] == "bad" for r in subset),
        "good_pixel_median": None if good.size == 0 else float(np.median(good)),
        "good_pixel_p99": None if good.size == 0 else float(np.quantile(good, 0.99)),
        "good_pixel_p99.9": None if good.size == 0 else float(np.quantile(good, 0.999)),
        "pooled_oracle_float16": (
            oracle_f1(labels, scores, cast_float16=True) if has_both else None
        ),
    }


def boundary_analysis(records: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any]:
    pairs = [(r["score"].astype(np.float16) >= np.float16(threshold), r["gt"]) for r in records]
    summaries = [boundary_pair_summary(prediction, gt) for prediction, gt in pairs]
    tolerant: dict[str, Any] = {}
    for tolerance in TOLERANCES:
        counts = np.sum([row["tolerant"][tolerance] for row in summaries], axis=0)
        precision, recall = safe_ratio(counts[0], counts[1]), safe_ratio(counts[2], counts[3])
        tolerant[str(tolerance)] = {
            "precision": precision,
            "recall": recall,
            "f1": harmonic(precision, recall),
        }
    return {
        "definition": (
            "precision uses pred AND Euclidean-dilate(GT,t); recall uses GT AND "
            "Euclidean-dilate(pred,t); counts are pooled over images"
        ),
        "tolerant_f1_native_px": tolerant,
        "interior_recall": pooled_interior(summaries),
        "exterior_boundary_band_fp_t16": pooled_exterior_band(summaries),
        "predicted_positive_distance_to_nearest_gt": pooled_localization(summaries),
    }


def boundary_pair_summary(prediction: BoolArray, gt: BoolArray) -> dict[str, Any]:
    has_gt, has_prediction = bool(np.any(gt)), bool(np.any(prediction))
    distance_to_gt = ndimage.distance_transform_edt(~gt) if has_gt else None
    distance_to_prediction = ndimage.distance_transform_edt(~prediction) if has_prediction else None
    gt_depth = ndimage.distance_transform_edt(gt) if has_gt else np.zeros(gt.shape)
    tolerant = {}
    for tolerance in TOLERANCES:
        pred_hit = (
            prediction & gt
            if tolerance == 0
            else distance_hit(prediction, distance_to_gt, tolerance)
        )
        gt_hit = (
            gt & prediction
            if tolerance == 0
            else distance_hit(gt, distance_to_prediction, tolerance)
        )
        tolerant[tolerance] = (
            int(np.count_nonzero(pred_hit)),
            int(np.count_nonzero(prediction)),
            int(np.count_nonzero(gt_hit)),
            int(np.count_nonzero(gt)),
        )
    outside_prediction = prediction & ~gt
    band = (
        outside_prediction & (distance_to_gt <= 16)
        if distance_to_gt is not None
        else np.zeros(gt.shape, bool)
    )
    localization = distance_to_gt[prediction] if distance_to_gt is not None else np.asarray([])
    interior = {
        t: (int(np.count_nonzero(prediction & (gt_depth > t))), int(np.count_nonzero(gt_depth > t)))
        for t in EROSIONS
    }
    return {
        "tolerant": tolerant,
        "interior": interior,
        "band_fp": int(np.count_nonzero(band)),
        "bad_fp": int(np.count_nonzero(outside_prediction)) if has_gt else 0,
        "localization_sum": float(np.sum(localization)),
        "localization_count": int(localization.size),
    }


def distance_hit(mask: BoolArray, distance: FloatArray | None, tolerance: int) -> BoolArray:
    return (
        np.zeros(mask.shape, dtype=np.bool_) if distance is None else mask & (distance <= tolerance)
    )


def pooled_interior(summaries: Sequence[dict[str, Any]]) -> dict[str, float]:
    output = {}
    for tolerance in EROSIONS:
        numerator = sum(row["interior"][tolerance][0] for row in summaries)
        denominator = sum(row["interior"][tolerance][1] for row in summaries)
        output[str(tolerance)] = safe_ratio(numerator, denominator)
    return output


def pooled_exterior_band(summaries: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    band_fp = sum(row["band_fp"] for row in summaries)
    all_fp = sum(row["bad_fp"] for row in summaries)
    return {"count": band_fp, "fraction_of_bad_image_false_positives": safe_ratio(band_fp, all_fp)}


def pooled_localization(summaries: Sequence[dict[str, Any]]) -> dict[str, float | int | None]:
    count = sum(row["localization_count"] for row in summaries)
    if count == 0:
        return {"mean_native_px": None, "n_predicted_positive_pixels_on_bad_images": 0}
    total = sum(row["localization_sum"] for row in summaries)
    return {
        "mean_native_px": float(total / count),
        "n_predicted_positive_pixels_on_bad_images": count,
    }


def component_analysis(records: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any]:
    gt_hit = gt_total = predicted_count = predicted_pixels = total_pixels = 0
    sizes = []
    for record in records:
        prediction = record["score"].astype(np.float16) >= np.float16(threshold)
        recall = component_recall(prediction, record["gt"])
        gt_hit += int(recall["hit"])
        gt_total += int(recall["total"])
        labels, count = ndimage.label(prediction, structure=STRUCTURE_8)
        component_sizes = np.bincount(labels.ravel())[1:]
        sizes.extend(int(value) for value in component_sizes)
        predicted_count += int(count)
        predicted_pixels += int(np.count_nonzero(prediction))
        total_pixels += int(prediction.size)
    return {
        "connectivity": 8,
        "gt_component_recall": safe_ratio(gt_hit, gt_total),
        "gt_components_hit": gt_hit,
        "gt_components_total": gt_total,
        "predicted_component_count": predicted_count,
        "predicted_positive_area_fraction": safe_ratio(predicted_pixels, total_pixels),
        "predicted_component_size_native_px": distribution(sizes),
    }


def classify_bottleneck(payload: dict[str, Any]) -> dict[str, Any]:
    pauc, pooled, delta = (
        payload["seg_AUROC_0.05_float16"],
        payload["pooled_oracle_float16"]["f1"],
        payload["delta_scale"],
    )
    boundary = payload["boundary_analysis"]
    interior = boundary["interior_recall"]["8"]
    component = payload["component_analysis"]["gt_component_recall"]
    tolerant_gain = (
        boundary["tolerant_f1_native_px"]["16"]["f1"] - boundary["tolerant_f1_native_px"]["0"]["f1"]
    )
    offsets = [
        abs(v)
        for v in payload["condition_analysis"]["good_median_offset_from_regular"].values()
        if v is not None
    ]
    max_offset = max(offsets, default=0.0)
    regular = payload["condition_analysis"]["conditions"].get("regular", {})
    regular_scale = max(
        float(regular.get("good_pixel_p99") or 0.0)
        - float(regular.get("good_pixel_median") or 0.0),
        1e-12,
    )
    condition_drift_ratio = max_offset / regular_scale
    evidence = {
        "seg_AUROC_0.05": pauc,
        "pooled_f1": pooled,
        "delta_scale": delta,
        "max_abs_condition_good_median_offset": max_offset,
        "condition_drift_ratio_to_regular_p99_span": condition_drift_ratio,
        "interior_recall_t8": interior,
        "gt_component_recall": component,
        "boundary_tolerant_f1_gain_t16": tolerant_gain,
    }
    if pauc < 0.6:
        category, reason = "Type D", "observability/ranking collapse (pAUROC < 0.60)"
    elif interior >= 0.7 and (component < 0.7 or tolerant_gain >= 0.1):
        category, reason = "Type C", "interior preserved with component or boundary degradation"
    elif pauc >= 0.8 and (pooled < 0.55 or delta >= 0.05 or condition_drift_ratio >= 0.1):
        category, reason = (
            "Type A",
            "ranking strong but pooled calibration/scale or condition drift limits F1",
        )
    else:
        category, reason = "Type B", "ranking and pooled localization are jointly limiting"
    return {
        "type": category,
        "reason": reason,
        "numeric_evidence": evidence,
        "rule_version": "class_agnostic_preregistered_phase0_v1",
    }


def analyze_run(
    result_root: Path,
    data_root: Path,
    objects: Sequence[str],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "claim_scope": CLAIM_SCOPE,
        "class_agnostic": True,
        "per_object_tuning": False,
        "result_root": str(result_root),
        "data_root": str(data_root),
        "objects": list(objects),
        "threshold_source": "pooled test_public oracle float16, diagnostic only",
    }
    analyses = [analyze_object(result_root, data_root, name) for name in objects]
    payload = {"metadata": metadata, "objects": {row["object"]: row for row in analyses}}
    write_json(output_dir / "gap_decomposition.json", payload)
    write_tsv(output_dir / "gap_decomposition.tsv", analyses)
    for analysis in analyses:
        write_json(output_dir / f"{analysis['object']}.json", {"metadata": metadata, **analysis})
        write_tsv(output_dir / f"{analysis['object']}.tsv", [analysis])
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, analyses: Sequence[dict[str, Any]]) -> None:
    rows = []
    for analysis in analyses:
        rows.extend(flatten_rows(analysis["object"], analysis))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("object", "table", "item", "metric", "value"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def flatten_rows(object_name: str, payload: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def visit(value: object, parts: list[str]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, [*parts, str(key)])
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, [*parts, str(index)])
        else:
            padded = [*parts, "", "", ""][:3]
            rows.append(
                {
                    "object": object_name,
                    "table": padded[0],
                    "item": padded[1],
                    "metric": ".".join(parts[2:]),
                    "value": "" if value is None else str(value),
                },
            )

    visit(payload, [])
    return rows


def dilate(mask: BoolArray, tolerance: int) -> BoolArray:
    if tolerance == 0 or not np.any(mask):
        return mask
    return np.asarray(ndimage.distance_transform_edt(~mask) <= tolerance, dtype=np.bool_)


def erode(mask: BoolArray, tolerance: int) -> BoolArray:
    if tolerance == 0 or not np.any(mask):
        return mask
    return np.asarray(ndimage.distance_transform_edt(mask) > tolerance, dtype=np.bool_)


def safe_ratio(numerator: int | np.integer, denominator: int | np.integer) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def harmonic(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))


def distribution(values: Iterable[int]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "min": 0, "median": 0.0, "mean": 0.0, "p90": 0.0, "max": 0}
    return {
        "count": int(array.size),
        "min": int(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": int(np.max(array)),
    }
