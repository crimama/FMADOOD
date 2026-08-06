#!/usr/bin/env python3
"""Evaluate a predeclared compact DSCR density cell without full-population claims."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import tifffile as tiff

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from flow_tte_gap_decomposition import load_gt  # noqa: E402
from flow_tte_phase3_scorer_suite import NativeMapRecord, evaluate_native_records  # noqa: E402

ARMS = ("G0", "D0", "D0c")
FOLD_COUNT = 4
NORMAL_QUANTILE = 0.9999


def _higher_quantile(values: np.ndarray, quantile: float) -> float:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        raise ValueError("quantile requires at least one value")
    index = math.ceil(quantile * (flat.size - 1))
    return float(np.partition(flat, index)[index])


def crossfit_fixed_metrics(records: Sequence[NativeMapRecord]) -> Mapping[str, Any]:
    good = sorted((row for row in records if row.split == "good"), key=lambda row: row.stem)
    bad = sorted((row for row in records if row.split != "good"), key=lambda row: row.stem)
    if len(good) < FOLD_COUNT or not bad:
        raise ValueError("fixed metrics require at least four good and one bad record")
    thresholds = []
    for fold in range(FOLD_COUNT):
        calibration = np.concatenate(
            [row.score.reshape(-1) for index, row in enumerate(good) if index % FOLD_COUNT != fold],
        )
        thresholds.append(_higher_quantile(calibration, NORMAL_QUANTILE))
    tp = fp = fn = good_positive = good_total = 0
    ordered = good + bad
    for index, row in enumerate(ordered):
        population_index = index if row.split == "good" else index - len(good)
        threshold = thresholds[population_index % FOLD_COUNT]
        predicted = np.asarray(row.score > threshold, dtype=np.bool_)
        labels = np.asarray(row.gt, dtype=np.bool_)
        tp += int(np.count_nonzero(predicted & labels))
        fp += int(np.count_nonzero(predicted & ~labels))
        fn += int(np.count_nonzero(~predicted & labels))
        if row.split == "good":
            good_positive += int(np.count_nonzero(predicted))
            good_total += int(predicted.size)
    denominator = 2 * tp + fp + fn
    return {
        "fixed_raw_f1": 0.0 if denominator == 0 else (2.0 * tp) / denominator,
        "fixed_normal_fpr": 0.0 if good_total == 0 else good_positive / good_total,
        "fixed_thresholds": thresholds,
        "target_normal_quantile": NORMAL_QUANTILE,
    }


def _source_path(data_root: Path, object_name: str, population: str, stem: str) -> Path:
    matches = sorted((data_root / object_name / "test_public" / population).glob(f"{stem}.*"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one source for {population}/{stem}, found {len(matches)}")
    return matches[0]


def load_arm_records(
    data_root: Path,
    cell_root: Path,
    object_name: str,
    arm: str,
) -> tuple[NativeMapRecord, ...]:
    records = []
    for population in ("good", "bad"):
        directory = cell_root / "arms" / arm / "anomaly_maps" / object_name / "test" / population
        maps = sorted(directory.glob("*.tiff"))
        if not maps:
            raise RuntimeError(f"no {arm} maps found for {object_name}/{population}")
        for map_path in maps:
            score = np.asarray(tiff.imread(map_path), dtype=np.float32)
            if score.ndim != 2 or not np.all(np.isfinite(score)):
                raise RuntimeError(f"invalid map: {map_path}")
            if float(np.max(score)) <= float(np.min(score)):
                raise RuntimeError(f"constant map: {map_path}")
            source = _source_path(data_root, object_name, population, map_path.stem)
            gt = load_gt(data_root, object_name, population, source.stem, score.shape)
            records.append(
                NativeMapRecord(
                    split=population,
                    score=score.astype(np.float16),
                    gt=np.asarray(gt, dtype=np.bool_),
                    stem=source.stem,
                ),
            )
    return tuple(records)


def evaluate_cell(data_root: Path, cell_root: Path, object_name: str) -> Mapping[str, Any]:
    arm_records = {
        arm: load_arm_records(data_root, cell_root, object_name, arm) for arm in ARMS
    }
    ids = {
        arm: tuple((row.split, row.stem) for row in records)
        for arm, records in arm_records.items()
    }
    if any(value != ids["G0"] for value in ids.values()):
        raise RuntimeError("arms do not contain identical logical query IDs")
    metrics = {}
    for arm, records in arm_records.items():
        metrics[arm] = {
            **evaluate_native_records(records),
            **crossfit_fixed_metrics(records),
        }
    keys = (
        "pooled_pauroc_0.05_float16",
        "pooled_pixel_ap_float16",
        "pooled_oracle_f1_float16",
        "fixed_raw_f1",
        "fixed_normal_fpr",
    )
    deltas = {
        arm: {key: float(metrics[arm][key]) - float(metrics["G0"][key]) for key in keys}
        for arm in ("D0", "D0c")
    }
    return {
        "schema": "dscr-density-compact-eval-v1",
        "object": object_name,
        "query_count": len(ids["G0"]),
        "good_count": sum(split == "good" for split, _ in ids["G0"]),
        "bad_count": sum(split == "bad" for split, _ in ids["G0"]),
        "arms": metrics,
        "delta_vs_g0": deltas,
        "comparable": False,
        "scope": "compact-public-shadow-mechanism-diagnostic",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cell-root", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = evaluate_cell(args.data_root, args.cell_root, args.object)
    output = args.output or args.cell_root / "compact_metrics.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
