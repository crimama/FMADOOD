#!/usr/bin/env python3
"""Evaluate the frozen compact Stage 4 local-soft nuisance correction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from evaluate_flow_tte_default_shift_audit import (  # noqa: E402
    OBJECTS,
    condition_rows,
    continuous_metrics,
    finite_nonconstant,
    macro_mean,
    object_arm_root,
    parse_objects,
    verify_parity,
)

from flow_tte_gap_decomposition import load_gt, parse_condition  # noqa: E402

ARMS = ("identity", "global_hard_rank4", "local_soft_rank2")


def object_harm(delta_auroc: float, delta_f1: float) -> bool:
    return delta_auroc < -0.02 or delta_f1 < -0.03


def load_compact_records(
    result_root: Path,
    data_root: Path,
    object_name: str,
) -> list[dict[str, Any]]:
    map_root = result_root / "anomaly_maps" / object_name / "test"
    records: list[dict[str, Any]] = []
    for map_path in sorted(map_root.glob("*/*.tiff")):
        split = map_path.parent.name
        if split not in {"good", "bad"}:
            continue
        score = np.asarray(tifffile.imread(map_path), dtype=np.float32)
        records.append({
            "split": split,
            "stem": map_path.stem,
            "condition": parse_condition(map_path.stem),
            "score": score,
            "gt": load_gt(data_root, object_name, split, map_path.stem, score.shape),
        })
    if not records or not {"good", "bad"}.issubset({row["split"] for row in records}):
        raise RuntimeError("compact records require both good and bad maps")
    return records


def arm_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        **continuous_metrics(records),
        "coverage": finite_nonconstant(records),
        "conditions": condition_rows(records),
    }


def macro_metrics(per_object: dict[str, Any], objects: Sequence[str], arm: str) -> dict[str, Any]:
    rows = [per_object[name]["arms"][arm] for name in objects]
    inflation = [
        float(condition["good_p95_inflation_vs_regular"])
        for row in rows
        for condition_name, condition in row["conditions"].items()
        if condition_name != "regular"
    ]
    separation = [
        float(condition["bad_good_p95_separation"])
        for row in rows
        for condition in row["conditions"].values()
    ]
    return {
        "seg_AUROC_0.05": macro_mean(rows, "seg_AUROC_0.05"),
        "seg_F1": macro_mean(rows, "seg_F1"),
        "shifted_good_p95_inflation": float(np.mean(inflation)),
        "bad_good_p95_separation": float(np.mean(separation)),
        "valid_object_count": sum(bool(row["coverage"]["pass"]) for row in rows),
    }


def evaluate(root: Path, data_root: Path, objects: Sequence[str]) -> dict[str, Any]:
    per_object: dict[str, Any] = {}
    for object_name in objects:
        records = {
            arm: load_compact_records(
                object_arm_root(root, object_name, arm),
                data_root,
                object_name,
            )
            for arm in ARMS
        }
        verify_parity(records)
        arms = {arm: arm_metrics(rows) for arm, rows in records.items()}
        identity = arms["identity"]
        local = arms["local_soft_rank2"]
        delta_auroc = local["seg_AUROC_0.05"] - identity["seg_AUROC_0.05"]
        delta_f1 = local["seg_F1"] - identity["seg_F1"]
        manifest = json.loads(
            (root / "objects" / object_name / "manifest.json").read_text(encoding="utf-8"),
        )
        expected = int(manifest["orbit_count"])
        observed = len(records["local_soft_rank2"])
        per_object[object_name] = {
            "arms": arms,
            "local_vs_identity": {
                "seg_AUROC_0.05": delta_auroc,
                "seg_F1": delta_f1,
                "object_harm": object_harm(delta_auroc, delta_f1),
            },
            "orbit_expected": expected,
            "orbit_observed": observed,
            "orbit_complete": expected == observed,
            "tangent_diagnostics": manifest["tangent_diagnostics"],
        }

    macro = {arm: macro_metrics(per_object, objects, arm) for arm in ARMS}
    identity = macro["identity"]
    local = macro["local_soft_rank2"]
    delta_auroc = local["seg_AUROC_0.05"] - identity["seg_AUROC_0.05"]
    delta_f1 = local["seg_F1"] - identity["seg_F1"]
    tail_reduction = (
        identity["shifted_good_p95_inflation"] - local["shifted_good_p95_inflation"]
    ) / max(abs(identity["shifted_good_p95_inflation"]), 1e-12)
    harm_objects = [
        name for name in objects if per_object[name]["local_vs_identity"]["object_harm"]
    ]
    validity = (
        int(local["valid_object_count"]) == len(objects)
        and all(per_object[name]["orbit_complete"] for name in objects)
    )
    gates = {
        "tail_reduction": tail_reduction >= 0.25,
        "separation": local["bad_good_p95_separation"] >= identity["bad_good_p95_separation"],
        "macro_noninferiority": delta_auroc >= -0.005 and delta_f1 >= -0.010,
        "no_object_harm": not harm_objects,
        "validity": validity,
    }
    continuation = all(gates.values())
    return {
        "target_dataset": "MVTec AD2 single-image compact frozen orbit",
        "evaluation_role": "test_public_shadow_local_soft_nuisance",
        "claim_boundary": "diagnostic contrast only; no TTA or query adaptation",
        "arms": list(ARMS),
        "macro": macro,
        "per_object": per_object,
        "verdict": {
            "local_vs_identity_seg_AUROC_0.05": delta_auroc,
            "local_vs_identity_seg_F1": delta_f1,
            "shifted_good_inflation_relative_reduction": tail_reduction,
            "object_harm": harm_objects,
            "gates": gates,
            "continuation": continuation,
            "decision": "PROMISING_DIAGNOSTIC" if continuation else "NO_CONTINUE",
            "global_hard_rank4_promotable": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--objects", default=",".join(OBJECTS))
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    payload = evaluate(Path(args.root), Path(args.data_root), parse_objects(args.objects))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"macro": payload["macro"], "verdict": payload["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
