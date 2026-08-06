#!/usr/bin/env python3
"""Aggregate object-scoped SDIG-NODyR Stage A compact artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

OBJECTS = (
    "can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts",
)
VARIANTS = (
    "nodyr", "state_angle", "state_l2", "raw_delta_l2", "shuffle_nodyr",
)
NULLS = ("state_angle", "state_l2", "raw_delta_l2", "shuffle_nodyr")
METRICS = ("seg_AUROC", "seg_F1")


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--require-all8", action="store_true")
    return parser.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    run_root = Path(args.run_root)
    files = sorted((run_root / "chunks").glob("*/**/*_object_summary.json"))
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    by_object = {item["object"]: item for item in summaries}
    if len(by_object) != len(summaries):
        raise RuntimeError("duplicate object summaries")
    if args.require_all8 and set(by_object) != set(OBJECTS):
        raise RuntimeError(f"all-eight summaries required, found {sorted(by_object)}")
    if not summaries:
        raise RuntimeError("no object summaries found")

    full_population = all(bool(item["full_population"]) for item in summaries)
    rows: list[dict[str, Any]] = []
    macro: dict[str, dict[str, float]] = {}
    if full_population:
        for object_name in sorted(by_object):
            metrics = by_object[object_name]["variant_metrics"]
            envelope = {metric: max(float(metrics[null][metric]) for null in NULLS) for metric in METRICS}
            row: dict[str, Any] = {"object": object_name}
            for metric in METRICS:
                row[f"nodyr_{metric}"] = float(metrics["nodyr"][metric])
                row[f"null_envelope_{metric}"] = envelope[metric]
                row[f"margin_{metric}"] = float(metrics["nodyr"][metric]) - envelope[metric]
            rows.append(row)
        for variant in VARIANTS:
            macro[variant] = {
                metric: mean([
                    float(by_object[obj]["variant_metrics"][variant][metric]) for obj in by_object
                ])
                for metric in METRICS
            }
    macro_envelope = {
        metric: max(macro[null][metric] for null in NULLS) for metric in METRICS
    } if macro else {}
    macro_margin = {
        metric: macro["nodyr"][metric] - macro_envelope[metric] for metric in METRICS
    } if macro else {}
    all_equivalence_valid = all(float(item["max_equivalence_error"]) <= 2e-5 for item in summaries)
    all_coverage_valid = all(float(item["finite_map_coverage"]) == 1.0 for item in summaries)
    immediate_both_margin_pass = bool(macro_margin) and all(value >= 0.005 for value in macro_margin.values())

    payload = {
        "target_dataset": "MVTec AD2 single-image",
        "objects": sorted(by_object),
        "full_population": full_population,
        "same_superad_reference_count": all(int(item["support_count"]) == 16 for item in summaries),
        "host_gpu_ids": sorted({str(item["host_gpu_id"]) for item in summaries}),
        "container_cuda_slots": sorted({str(item["container_cuda_slot"]) for item in summaries}),
        "primary_metrics": ["seg_AUROC_0.05", "seg_F1"],
        "macro": macro,
        "macro_independent_null_envelope": macro_envelope,
        "macro_margin_vs_independent_envelope": macro_margin,
        "immediate_both_margin_pass": immediate_both_margin_pass,
        "all_finite_map_coverage": all_coverage_valid,
        "algebraic_equivalence_audit_pass": all_equivalence_valid,
        "source_orbit_bootstrap_available": False,
        "same_run_superad_available": False,
        "same_run_rn_fmlk_available": False,
        "strict_claim_comparable": False,
        "strict_gate_status": "not_run_testpublic_shadow_only",
        "per_object": rows,
    }
    if full_population and not immediate_both_margin_pass:
        payload["shadow_signal_status"] = "negative_or_mixed_vs_intrinsic_envelope"
    elif full_population:
        payload["shadow_signal_status"] = "positive_vs_intrinsic_envelope_unconfirmed"
    else:
        payload["shadow_signal_status"] = "smoke_valid" if all_coverage_valid and all_equivalence_valid else "smoke_invalid"
    payload["diagnostic_verdict"] = "TESTPUBLIC_SHADOW_ONLY"
    (run_root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if rows:
        with (run_root / "per_object.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
