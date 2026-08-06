#!/usr/bin/env python3
"""Aggregate preregistered PAIA object metrics and apply its fixed gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

OBJECTS = ("fabric", "rice", "vial")
METRICS = ("seg_AUROC_0.05", "seg_F1", "pixel_AP")
PRIMARY_METRICS = ("seg_AUROC_0.05", "seg_F1")


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "dscr-density-condition-audit-v1":
        raise ValueError(f"unexpected schema in {path}")
    return payload


def aggregate(paths: Sequence[Path]) -> Mapping[str, Any]:
    rows = [_load(path) for path in paths]
    by_object = {str(row["object"]): row for row in rows}
    if len(by_object) != len(rows):
        raise ValueError("duplicate object inputs")
    if set(by_object) != set(OBJECTS):
        raise ValueError(f"expected objects {OBJECTS}, observed {tuple(sorted(by_object))}")

    complete = all(
        row["query_coverage"]["complete"]
        and row["validity"]["finite"]
        and row["validity"]["nonconstant"]
        and row["validity"]["map_gt_shape_match"]
        and row["validity"]["float32_tiff_complete"]
        and "float32" in row["precisions_evaluated"]
        for row in rows
    )
    object_metrics: dict[str, Any] = {}
    for object_name in OBJECTS:
        row = by_object[object_name]
        regular = row["conditions"]["regular"]["delta_vs_g0"]["float32"]
        interaction = row["interaction"]["float32"]
        object_metrics[object_name] = {
            "regular_delta": {metric: float(regular[metric]) for metric in METRICS},
            "mean_shifted_delta": {
                metric: float(interaction[metric]["mean_shifted_delta"])
                for metric in METRICS
            },
            "interaction": {
                metric: float(interaction[metric]["value"]) for metric in METRICS
            },
        }

    macro = {
        field: {
            metric: float(
                np.mean([object_metrics[name][field][metric] for name in OBJECTS]),
            )
            for metric in METRICS
        }
        for field in ("regular_delta", "mean_shifted_delta", "interaction")
    }
    negative_pauroc_interactions = sum(
        object_metrics[name]["interaction"]["seg_AUROC_0.05"] < 0.0 for name in OBJECTS
    )
    gate_checks = {
        "macro_regular_primary_nonnegative": all(
            macro["regular_delta"][metric] >= 0.0 for metric in PRIMARY_METRICS
        ),
        "no_regular_object_harm_beyond_floor": all(
            object_metrics[name]["regular_delta"]["seg_AUROC_0.05"] >= -0.03
            and object_metrics[name]["regular_delta"]["seg_F1"] >= -0.05
            for name in OBJECTS
        ),
        "pauroc_interaction_shift_specific": (
            macro["interaction"]["seg_AUROC_0.05"] <= -0.03
            and negative_pauroc_interactions >= 2
        ),
        "f1_interaction_negative": macro["interaction"]["seg_F1"] < 0.0,
        "complete_finite_nonconstant_float32": complete,
    }
    promising = all(gate_checks.values())
    return {
        "schema": "dscr-density-condition-audit-summary-v1",
        "objects": list(OBJECTS),
        "object_metrics": object_metrics,
        "macro": macro,
        "negative_pauroc_interaction_object_count": negative_pauroc_interactions,
        "gate_checks": gate_checks,
        "gate_pass": promising,
        "claim_verdict": "KILL_FOR_CLAIM",
        "diagnostic_verdict": "PROMISING_DIAGNOSTIC" if promising else "NO_CONTINUE",
        "comparable": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = aggregate(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
