#!/usr/bin/env python3
"""Evaluate the frozen A0--A3 default FlowTTE shift-attribution arms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from flow_tte_postprocess_core import (  # noqa: E402
    binary_mask_metrics,
    variant_profile,
)

from flow_tte_gap_decomposition import load_records  # noqa: E402

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
CONTINUOUS_ARMS = ("a0_distance", "a1_density", "a2_guided")


def parse_objects(raw: str) -> tuple[str, ...]:
    values = tuple(value for value in raw.replace(",", " ").split() if value)
    if not values:
        raise ValueError("objects must not be empty")
    return values


def object_arm_root(root: Path, object_name: str, arm: str) -> Path:
    value = root / "objects" / object_name / arm
    if not value.is_dir():
        message = f"missing object arm root: {value}"
        raise FileNotFoundError(message)
    return value


def finite_nonconstant(records: Sequence[dict[str, Any]]) -> dict[str, int | bool]:
    valid = 0
    for record in records:
        score = np.asarray(record["score"], dtype=np.float32)
        if not bool(np.all(np.isfinite(score))):
            continue
        if float(np.max(score) - np.min(score)) <= 0.0:
            continue
        valid += 1
    return {"valid": valid, "total": len(records), "pass": valid == len(records)}


def float16_order_keys(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float16).reshape(-1)
    if not bool(np.all(np.isfinite(values))):
        raise ValueError("float16 metrics require finite scores")
    bits = values.view(np.uint16).copy()
    bits[(bits & np.uint16(0x7FFF)) == 0] = 0
    negative = (bits & np.uint16(0x8000)) != 0
    return np.where(negative, np.bitwise_not(bits), bits ^ np.uint16(0x8000)).astype(
        np.uint16,
    )


def key_to_float16(key: int) -> float:
    value = np.uint16(key)
    bits = value ^ np.uint16(0x8000) if value & np.uint16(0x8000) else ~value
    return float(np.asarray(bits, dtype=np.uint16).view(np.float16))


def standardized_partial_auc(
    positive_counts: np.ndarray,
    negative_counts: np.ndarray,
    max_fpr: float,
) -> float:
    occupied = np.flatnonzero((positive_counts + negative_counts) > 0)[::-1]
    positives = float(positive_counts.sum())
    negatives = float(negative_counts.sum())
    if positives <= 0 or negatives <= 0:
        raise ValueError("partial AUC requires both classes")
    tpr = np.concatenate(([0.0], np.cumsum(positive_counts[occupied]) / positives))
    fpr = np.concatenate(([0.0], np.cumsum(negative_counts[occupied]) / negatives))
    stop = int(np.searchsorted(fpr, max_fpr, side="right"))
    clipped_fpr = fpr[:stop]
    clipped_tpr = tpr[:stop]
    if clipped_fpr[-1] < max_fpr:
        right = min(stop, fpr.size - 1)
        left = max(0, right - 1)
        if fpr[right] == fpr[left]:
            interpolated = tpr[right]
        else:
            fraction = (max_fpr - fpr[left]) / (fpr[right] - fpr[left])
            interpolated = tpr[left] + fraction * (tpr[right] - tpr[left])
        clipped_fpr = np.append(clipped_fpr, max_fpr)
        clipped_tpr = np.append(clipped_tpr, interpolated)
    partial = float(
        np.sum(0.5 * (clipped_tpr[1:] + clipped_tpr[:-1]) * np.diff(clipped_fpr)),
    )
    minimum = 0.5 * max_fpr**2
    maximum = max_fpr
    return float(0.5 * (1.0 + (partial - minimum) / (maximum - minimum)))


def binary_float16_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    truth = np.asarray(labels, dtype=np.uint8).reshape(-1)
    keys = float16_order_keys(scores)
    if truth.size != keys.size or not set(np.unique(truth)).issubset({0, 1}):
        raise ValueError("labels and scores must be aligned binary vectors")
    positive = np.bincount(keys[truth == 1], minlength=65536).astype(np.int64)
    negative = np.bincount(keys[truth == 0], minlength=65536).astype(np.int64)
    occupied = np.flatnonzero((positive + negative) > 0)
    true_positive = np.cumsum(positive[::-1])[::-1][occupied].astype(np.float64)
    false_positive = np.cumsum(negative[::-1])[::-1][occupied].astype(np.float64)
    false_negative = float(positive.sum()) - true_positive
    denominator = 2.0 * true_positive + false_positive + false_negative
    f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    best = int(np.argmax(f1))
    return {
        "seg_AUROC_0.05": standardized_partial_auc(positive, negative, 0.05),
        "seg_F1": float(f1[best]),
        "best_threshold": key_to_float16(int(occupied[best])),
    }


def continuous_metrics(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    labels = np.concatenate([np.asarray(row["gt"], dtype=np.uint8).ravel() for row in records])
    scores = np.concatenate(
        [np.asarray(row["score"], dtype=np.float32).astype(np.float16).ravel() for row in records],
    )
    if np.unique(labels).size != 2:
        raise ValueError("continuous metrics require both normal and anomalous pixels")
    return binary_float16_metrics(labels, scores)


def morphology_metrics(
    records: Sequence[dict[str, Any]],
    threshold: float,
) -> dict[str, float]:
    result = binary_mask_metrics(
        [np.asarray(row["score"], dtype=np.float32) for row in records],
        [np.asarray(row["gt"], dtype=np.bool_) for row in records],
        threshold,
        variant_profile("closefill_erode"),
        17,
        16,
    )
    return {
        "seg_F1": result.f1,
        "precision": result.precision,
        "recall": result.recall,
        "positive_area": result.positive_area,
    }


def morphology_condition_rows(
    records: Sequence[dict[str, Any]],
    threshold: float,
) -> dict[str, dict[str, Any]]:
    rows = condition_rows(records)
    for condition, metrics in rows.items():
        condition_records = [row for row in records if row["condition"] == condition]
        morphology = morphology_metrics(condition_records, threshold)
        metrics.update({
            "seg_F1": morphology["seg_F1"],
            "morphology_precision": morphology["precision"],
            "morphology_recall": morphology["recall"],
            "morphology_positive_area": morphology["positive_area"],
        })
    return rows


def image_p95(record: dict[str, Any]) -> float:
    return float(np.quantile(np.asarray(record["score"], dtype=np.float32), 0.95))


def condition_rows(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    conditions = sorted({str(row["condition"]) for row in records})
    result: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        rows = [row for row in records if row["condition"] == condition]
        good = [image_p95(row) for row in rows if row["split"] == "good"]
        bad = [image_p95(row) for row in rows if row["split"] == "bad"]
        if not good or not bad:
            message = f"condition {condition} lacks good/bad coverage"
            raise ValueError(message)
        result[condition] = {
            "n_good": len(good),
            "n_bad": len(bad),
            "good_image_p95_mean": float(np.mean(good)),
            "bad_image_p95_mean": float(np.mean(bad)),
            "bad_good_p95_separation": float(np.mean(bad) - np.mean(good)),
            **continuous_metrics(rows),
        }
    regular = result.get("regular")
    if regular is None:
        raise ValueError("regular condition is required")
    regular_good = float(regular["good_image_p95_mean"])
    for row in result.values():
        row["good_p95_inflation_vs_regular"] = float(row["good_image_p95_mean"] - regular_good)
    return result


def verify_parity(arms: dict[str, Sequence[dict[str, Any]]]) -> None:
    expected: set[tuple[str, str]] | None = None
    for arm, records in arms.items():
        observed = {(str(row["split"]), str(row["stem"])) for row in records}
        if expected is None:
            expected = observed
        elif observed != expected:
            message = f"map identity mismatch for {arm}"
            raise ValueError(message)


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return zero-based average ranks, including exact-tie handling."""
    flat = np.asarray(values).ravel()
    order = np.argsort(flat, kind="mergesort")
    sorted_values = flat[order]
    ranks = np.empty(flat.size, dtype=np.float64)
    starts = np.concatenate(
        (np.array([0], dtype=np.int64), np.flatnonzero(np.diff(sorted_values) != 0) + 1),
    )
    stops = np.concatenate((starts[1:], np.array([flat.size], dtype=np.int64)))
    average = 0.5 * (starts + stops - 1)
    ranks[order] = np.repeat(average, stops - starts)
    return ranks


def spearman_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_ranks = average_ranks(first)
    second_ranks = average_ranks(second)
    first_centered = first_ranks - first_ranks.mean()
    second_centered = second_ranks - second_ranks.mean()
    denominator = float(
        np.linalg.norm(first_centered) * np.linalg.norm(second_centered),
    )
    if denominator <= 0.0:
        return float("nan")
    return float(np.dot(first_centered, second_centered) / denominator)


def guide_agreement(
    raw_records: Sequence[dict[str, Any]],
    guided_records: Sequence[dict[str, Any]],
) -> list[dict[str, float | str | bool]]:
    guided_by_id = {
        (str(row["split"]), str(row["stem"])): row for row in guided_records
    }
    output: list[dict[str, float | str | bool]] = []
    for raw in raw_records:
        key = (str(raw["split"]), str(raw["stem"]))
        guided = guided_by_id[key]
        raw_score = np.asarray(raw["score"], dtype=np.float32)
        guided_score = np.asarray(guided["score"], dtype=np.float32)
        correlation = spearman_correlation(raw_score, guided_score)
        top_count = max(1, round(raw_score.size * 0.01))
        raw_top = float(np.partition(raw_score.ravel(), -top_count)[-top_count:].mean())
        guided_top = float(np.partition(guided_score.ravel(), -top_count)[-top_count:].mean())
        raw_peak = np.asarray(np.unravel_index(int(np.argmax(raw_score)), raw_score.shape))
        guided_peak = np.asarray(np.unravel_index(int(np.argmax(guided_score)), guided_score.shape))
        diagonal = max(float(np.linalg.norm(np.asarray(raw_score.shape, dtype=np.float64))), 1.0)
        displacement = float(np.linalg.norm(raw_peak - guided_peak) / diagonal)
        ratio = guided_top / raw_top if abs(raw_top) > 1e-12 else 0.0
        output.append({
            "split": key[0],
            "stem": key[1],
            "condition": str(raw["condition"]),
            "spearman": correlation,
            "top1_mean_ratio": ratio,
            "peak_displacement_fraction": displacement,
            "frozen_gate_accept": bool(
                correlation >= 0.90 and ratio >= 0.90 and displacement <= 0.10,
            ),
        })
    return output


def macro_mean(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    if not values:
        message = f"cannot macro-average empty field {key}"
        raise ValueError(message)
    return float(np.mean(values))


def evaluate(root: Path, data_root: Path, objects: Sequence[str]) -> dict[str, Any]:
    per_object: dict[str, Any] = {}
    for object_name in objects:
        arms = {
            arm: load_records(object_arm_root(root, object_name, arm), data_root, object_name)
            for arm in CONTINUOUS_ARMS
        }
        verify_parity(arms)
        object_result: dict[str, Any] = {"arms": {}}
        for arm, records in arms.items():
            metrics = continuous_metrics(records)
            object_result["arms"][arm] = {
                **metrics,
                "coverage": finite_nonconstant(records),
                "conditions": condition_rows(records),
            }
        guided = arms["a2_guided"]
        guided_metrics = object_result["arms"]["a2_guided"]
        guided_threshold = float(guided_metrics["best_threshold"])
        object_result["arms"]["a3_morphology"] = {
            "seg_AUROC_0.05": guided_metrics["seg_AUROC_0.05"],
            **morphology_metrics(guided, guided_threshold),
            "threshold_source": "a2_guided object-global raw best threshold",
            "coverage": guided_metrics["coverage"],
            "conditions": morphology_condition_rows(guided, guided_threshold),
        }
        object_result["guide_agreement"] = guide_agreement(
            arms["a1_density"],
            arms["a2_guided"],
        )
        per_object[object_name] = object_result

    arm_names = (*CONTINUOUS_ARMS, "a3_morphology")
    macro: dict[str, Any] = {}
    for arm in arm_names:
        rows = [per_object[name]["arms"][arm] for name in objects]
        condition_inflations = []
        separations = []
        for row in rows:
            for condition, condition_row in row["conditions"].items():
                if condition != "regular":
                    condition_inflations.append(
                        float(condition_row["good_p95_inflation_vs_regular"]),
                    )
                separations.append(float(condition_row["bad_good_p95_separation"]))
        macro[arm] = {
            "seg_AUROC_0.05": macro_mean(rows, "seg_AUROC_0.05"),
            "seg_F1": macro_mean(rows, "seg_F1"),
            "shifted_good_p95_inflation": float(np.mean(condition_inflations)),
            "bad_good_p95_separation": float(np.mean(separations)),
            "valid_object_count": sum(bool(row["coverage"]["pass"]) for row in rows),
        }

    return {
        "target_dataset": "MVTec AD2 single-image",
        "evaluation_role": "test_public_shadow_default_chain_attribution",
        "claim_boundary": "diagnostic only; no TESTprivate and no TTA/query adaptation",
        "objects": list(objects),
        "arms": list(arm_names),
        "macro": macro,
        "per_object": per_object,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--objects", default=",".join(OBJECTS))
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    payload = evaluate(Path(args.root), Path(args.data_root), parse_objects(args.objects))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["macro"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
