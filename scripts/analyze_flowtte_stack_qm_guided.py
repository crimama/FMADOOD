#!/usr/bin/env python3
"""Evaluate the retained FlowTTE quantile-match + guided-filter stack."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.flow_tte_gap_decomposition import load_records  # noqa: E402
from src.flow_tte_phase1_normalization import (  # noqa: E402
    OBJECTS,
    QUANTILE_MATCH_KNOTS,
    condition_group_quantile_match_to_regular,
)
from src.flow_tte_phase2_refinement import (  # noqa: E402
    evaluate_variant,
    load_half_guidance,
    transform_score,
)

ARMS = ("identity", "quantile_match_only", "guided_r8_only", "quantile_match_then_guided_r8")
REFERENCES = {
    "identity": 0.530635,
    "quantile_match_only": 0.539296,
    "guided_r8_only": 0.550676,
}
REFERENCE_TOLERANCE = 1e-6


def compose_quantile_match_then_guided(
    scores: Sequence[np.ndarray],
    stems: Sequence[str],
    guidances: Sequence[np.ndarray],
) -> list[np.ndarray]:
    """Apply the fixed stack order using the existing winning module functions."""
    matched, _metadata = condition_group_quantile_match_to_regular(
        scores,
        stems,
        knots=QUANTILE_MATCH_KNOTS,
    )
    return [
        transform_score(score, guidance, "guided_r8_eps1e-2")
        for score, guidance in zip(matched, guidances)
    ]


def evaluate_object(result_root: Path, data_root: Path, object_name: str) -> dict[str, Any]:
    records = load_records(result_root, data_root, object_name)
    identity = [np.asarray(record["score"], dtype=np.float32) for record in records]
    stems = [str(record["stem"]) for record in records]
    guidances = [
        load_half_guidance(record["rgb_path"], score.shape)
        for record, score in zip(records, identity)
    ]
    matched, qm_metadata = condition_group_quantile_match_to_regular(
        identity,
        stems,
        knots=QUANTILE_MATCH_KNOTS,
    )
    guided = [
        transform_score(score, guidance, "guided_r8_eps1e-2")
        for score, guidance in zip(identity, guidances)
    ]
    stacked = [
        transform_score(score, guidance, "guided_r8_eps1e-2")
        for score, guidance in zip(matched, guidances)
    ]
    score_sets = {
        "identity": identity,
        "quantile_match_only": matched,
        "guided_r8_only": guided,
        "quantile_match_then_guided_r8": stacked,
    }
    return {
        "object": object_name,
        "arms": {name: evaluate_variant(records, values) for name, values in score_sets.items()},
        "quantile_match_metadata": qm_metadata,
    }


def _write_tsv(path: Path, arms: Mapping[str, Mapping[str, Any]]) -> None:
    fields = ["arm", "mean_f1", "mean_pauroc_0.05", "min_object_f1_delta_vs_identity"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for name in ARMS:
            mean = arms[name]["mean"]
            writer.writerow(
                {
                    "arm": name,
                    "mean_f1": mean["pooled_oracle_f1"],
                    "mean_pauroc_0.05": mean["seg_AUROC_0.05"],
                    "min_object_f1_delta_vs_identity": arms[name][
                        "min_object_f1_delta_vs_identity"
                    ],
                }
            )


def analyze(
    result_root: Path,
    data_root: Path,
    output_dir: Path,
    objects: Sequence[str] = OBJECTS,
    workers: int = 1,
) -> dict[str, Any]:
    if tuple(objects) != OBJECTS:
        raise ValueError(f"stack evaluation requires the fixed object order/set: {OBJECTS}")
    if workers < 1:
        raise ValueError("workers must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    if workers == 1:
        object_rows = [evaluate_object(result_root, data_root, name) for name in objects]
    else:
        object_rows = []
        with ProcessPoolExecutor(max_workers=min(workers, len(objects))) as executor:
            futures = {
                executor.submit(evaluate_object, result_root, data_root, name): name
                for name in objects
            }
            for future in as_completed(futures):
                object_rows.append(future.result())
        object_rows.sort(key=lambda row: objects.index(row["object"]))

    arms: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        per_object = {row["object"]: row["arms"][arm] for row in object_rows}
        identity = {
            row["object"]: row["arms"]["identity"]["pooled_oracle_f1"]
            for row in object_rows
        }
        deltas = {
            name: float(metrics["pooled_oracle_f1"] - identity[name])
            for name, metrics in per_object.items()
        }
        arms[arm] = {
            "objects": per_object,
            "per_object_f1_delta_vs_identity": deltas,
            "min_object_f1_delta_vs_identity": min(deltas.values()),
            "mean": {
                "pooled_oracle_f1": float(
                    np.mean([row["pooled_oracle_f1"] for row in per_object.values()])
                ),
                "seg_AUROC_0.05": float(
                    np.mean([row["seg_AUROC_0.05"] for row in per_object.values()])
                ),
            },
        }
    checks = {
        arm: {
            "reference": reference,
            "observed": arms[arm]["mean"]["pooled_oracle_f1"],
            "absolute_error": abs(arms[arm]["mean"]["pooled_oracle_f1"] - reference),
            "tolerance": REFERENCE_TOLERANCE,
            "pass": abs(arms[arm]["mean"]["pooled_oracle_f1"] - reference)
            <= REFERENCE_TOLERANCE,
        }
        for arm, reference in REFERENCES.items()
    }
    payload = {
        "schema": "flowtte-stack-qm-guided-v1",
        "result_root": str(result_root),
        "data_root": str(data_root),
        "order": ["condition_group_quantile_match_to_regular_q4096", "guided_r8_eps1e-2"],
        "arms": arms,
        "reference_checks": checks,
        "reference_checks_pass": all(row["pass"] for row in checks.values()),
        "quantile_match_metadata": {
            row["object"]: row["quantile_match_metadata"] for row in object_rows
        },
    }
    (output_dir / "stack_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_tsv(output_dir / "stack_leaderboard.tsv", arms)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--objects", nargs="+", default=list(OBJECTS))
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    objects = [part for value in args.objects for part in value.replace(",", " ").split() if part]
    payload = analyze(args.result_root, args.data_root, args.output_dir, objects, args.workers)
    print(json.dumps(payload["reference_checks"], indent=2, sort_keys=True), flush=True)
    return 0 if payload["reference_checks_pass"] else 42


if __name__ == "__main__":
    raise SystemExit(main())
