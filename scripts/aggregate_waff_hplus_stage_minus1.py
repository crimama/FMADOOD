#!/usr/bin/env python3
"""Aggregate the four WAFF-H+ can/rice L1 support-only smoke shards."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

EXPECTED_CASES = (("can", 0), ("can", 1), ("rice", 0), ("rice", 1))


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    return parser.parse_args(list(argv))


def aggregate(run_root: Path) -> dict[str, Any]:
    cases: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted((run_root / "chunks").glob("*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("query_or_test_access") is not False:
            message = f"{path}: support-only access invariant failed"
            raise RuntimeError(message)
        if payload.get("public_metrics_computed") is not False:
            message = f"{path}: public metric invariant failed"
            raise RuntimeError(message)
        seed = int(payload["seed"])
        results = payload.get("results")
        if not isinstance(results, list) or len(results) != 1:
            message = f"{path}: each shard must contain exactly one object"
            raise RuntimeError(message)
        object_name = str(results[0]["object"])
        key = (object_name, seed)
        if key in cases:
            message = f"duplicate WAFF L1 shard {key}"
            raise RuntimeError(message)
        cases[key] = {"path": str(path.relative_to(run_root)), **results[0]}
    if set(cases) != set(EXPECTED_CASES):
        message = f"incomplete WAFF L1 cases: {sorted(cases)}"
        raise RuntimeError(message)

    rows: list[dict[str, Any]] = []
    for (object_name, seed), result in sorted(cases.items()):
        default_drop = float(result["default_worst_axis_margin_drop"])
        waff_drop = float(result["waff_worst_axis_margin_drop"])
        rows.append(
            {
                "object": object_name,
                "seed": seed,
                "noncollapse": bool(result["noncollapse"]),
                "static_memory_unchanged": bool(result["static_memory_unchanged"]),
                "integration_max_abs_error": float(result["integration_max_abs_error"]),
                "active_score_std": float(result["active_score_std"]),
                "default_worst_axis_margin_drop": default_drop,
                "waff_worst_axis_margin_drop": waff_drop,
                "margin_drop_delta": waff_drop - default_drop,
                "weight": result["waff_scorer"]["weight"],
                "weight_sum": float(sum(result["waff_scorer"]["weight"])),
                "joint_feasible": bool(result["simplex_feasibility"]["joint_feasible"]),
                "min_joint_violation": float(
                    result["simplex_feasibility"]["min_joint_violation"],
                ),
            },
        )
    signs_by_seed = {
        seed: [
            float(row["margin_drop_delta"]) <= 0.0
            for row in rows
            if int(row["seed"]) == seed
        ]
        for seed in (0, 1)
    }
    gate_pass = bool(
        all(row["noncollapse"] for row in rows)
        and all(row["static_memory_unchanged"] for row in rows)
        and all(row["integration_max_abs_error"] <= 1e-6 for row in rows)
        and all(row["active_score_std"] > 1e-6 for row in rows)
        and all(abs(row["weight_sum"] - 1.25) <= 1e-6 for row in rows),
    )
    l2_ready = gate_pass and all(row["joint_feasible"] for row in rows)
    return {
        "protocol_id": "waff-hplus-stage-minus1-support-only-20260719-l1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "L1_support_only_runtime_noncollapse_smoke",
        "case_count": len(rows),
        "rows": rows,
        "all_noncollapse": all(row["noncollapse"] for row in rows),
        "directional_improvement_by_seed": signs_by_seed,
        "directional_consistency": signs_by_seed[0] == signs_by_seed[1],
        "gate_pass": gate_pass,
        "l2_ready": l2_ready,
        "selection_credit": False,
        "public_access_authorized": False,
        "next_gate": (
            "L2_all8_two_seed_nested_LODF_x_LOSF"
            if l2_ready
            else "REDESIGN_INFEASIBLE_CONSTRAINT_SURFACE"
        ),
    }


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    run_root = Path(args.run_root).resolve(strict=True)
    summary = aggregate(run_root)
    (run_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verdict = "L1_PASS_NO_SELECTION_CREDIT" if summary["gate_pass"] else "L1_FAIL"
    (run_root / "verdict.txt").write_text(verdict + "\n", encoding="utf-8")
    return 0 if summary["gate_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main(tuple(__import__("sys").argv[1:])))
