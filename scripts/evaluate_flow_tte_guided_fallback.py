#!/usr/bin/env python3
"""Evaluate the frozen label-free guided-refinement fallback gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from evaluate_flow_tte_default_shift_audit import (  # noqa: E402
    OBJECTS,
    condition_rows,
    continuous_metrics,
    finite_nonconstant,
    guide_agreement,
    macro_mean,
    object_arm_root,
    parse_objects,
    verify_parity,
)

from flow_tte_gap_decomposition import load_records  # noqa: E402


def fallback_records(
    raw_records: Sequence[dict[str, Any]],
    guided_records: Sequence[dict[str, Any]],
    frozen_agreements: Sequence[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    agreements = list(frozen_agreements or guide_agreement(raw_records, guided_records))
    guided_by_id = {
        (str(row["split"]), str(row["stem"])): row for row in guided_records
    }
    agreement_by_id = {
        (str(row["split"]), str(row["stem"])): row for row in agreements
    }
    selected: list[dict[str, Any]] = []
    for raw in raw_records:
        key = (str(raw["split"]), str(raw["stem"]))
        use_guided = bool(agreement_by_id[key]["frozen_gate_accept"])
        source = guided_by_id[key] if use_guided else raw
        selected.append({**raw, "score": np.asarray(source["score"], dtype=np.float32)})
    return selected, agreements


def arm_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        **continuous_metrics(records),
        "coverage": finite_nonconstant(records),
        "conditions": condition_rows(records),
    }


def is_object_harm(delta_auroc: float, delta_f1: float) -> bool:
    return delta_auroc < -0.02 or delta_f1 < -0.03


def evaluate(root: Path, data_root: Path, objects: Sequence[str]) -> dict[str, Any]:
    stage1_summary = json.loads(
        (root / "stage1_summary.json").read_text(encoding="utf-8"),
    )
    per_object: dict[str, Any] = {}
    for object_name in objects:
        raw = load_records(
            object_arm_root(root, object_name, "a1_density"),
            data_root,
            object_name,
        )
        guided = load_records(
            object_arm_root(root, object_name, "a2_guided"),
            data_root,
            object_name,
        )
        verify_parity({"always_raw": raw, "always_guided": guided})
        frozen_agreements = stage1_summary["per_object"][object_name]["guide_agreement"]
        fallback, agreements = fallback_records(raw, guided, frozen_agreements)
        arms = {
            "always_raw": arm_metrics(raw),
            "always_guided": arm_metrics(guided),
            "frozen_fallback": arm_metrics(fallback),
        }
        guided_delta_auroc = (
            arms["always_guided"]["seg_AUROC_0.05"]
            - arms["always_raw"]["seg_AUROC_0.05"]
        )
        guided_delta_f1 = (
            arms["always_guided"]["seg_F1"] - arms["always_raw"]["seg_F1"]
        )
        fallback_delta_auroc = (
            arms["frozen_fallback"]["seg_AUROC_0.05"]
            - arms["always_raw"]["seg_AUROC_0.05"]
        )
        fallback_delta_f1 = (
            arms["frozen_fallback"]["seg_F1"] - arms["always_raw"]["seg_F1"]
        )
        per_object[object_name] = {
            "arms": arms,
            "gate": {
                "accepted": sum(bool(row["frozen_gate_accept"]) for row in agreements),
                "total": len(agreements),
                "acceptance_rate": float(
                    np.mean([bool(row["frozen_gate_accept"]) for row in agreements]),
                ),
                "agreements": agreements,
            },
            "guided_vs_raw": {
                "seg_AUROC_0.05": guided_delta_auroc,
                "seg_F1": guided_delta_f1,
                "object_harm": is_object_harm(guided_delta_auroc, guided_delta_f1),
            },
            "fallback_vs_raw": {
                "seg_AUROC_0.05": fallback_delta_auroc,
                "seg_F1": fallback_delta_f1,
                "object_harm": is_object_harm(fallback_delta_auroc, fallback_delta_f1),
            },
        }

    macro: dict[str, Any] = {}
    for arm in ("always_raw", "always_guided", "frozen_fallback"):
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
        macro[arm] = {
            "seg_AUROC_0.05": macro_mean(rows, "seg_AUROC_0.05"),
            "seg_F1": macro_mean(rows, "seg_F1"),
            "shifted_good_p95_inflation": float(np.mean(inflation)),
            "bad_good_p95_separation": float(np.mean(separation)),
            "valid_object_count": sum(bool(row["coverage"]["pass"]) for row in rows),
        }

    guided = macro["always_guided"]
    fallback = macro["frozen_fallback"]
    delta_auroc = fallback["seg_AUROC_0.05"] - guided["seg_AUROC_0.05"]
    delta_f1 = fallback["seg_F1"] - guided["seg_F1"]
    guided_harm = {
        name for name in objects if per_object[name]["guided_vs_raw"]["object_harm"]
    }
    fallback_harm = {
        name for name in objects if per_object[name]["fallback_vs_raw"]["object_harm"]
    }
    direct_improvement = delta_auroc >= 0.002 and delta_f1 >= 0.005
    eliminates_guided_harm = bool(guided_harm) and not (guided_harm & fallback_harm)
    no_macro_loss = delta_auroc >= 0.0 and delta_f1 >= 0.0
    continuation = direct_improvement or (eliminates_guided_harm and no_macro_loss)
    return {
        "target_dataset": "MVTec AD2 single-image",
        "evaluation_role": "test_public_shadow_frozen_guided_fallback",
        "claim_boundary": "diagnostic only; gate uses no labels and performs no TTA",
        "gate": {
            "spearman_min": 0.90,
            "top1_mean_ratio_min": 0.90,
            "peak_displacement_fraction_max": 0.10,
        },
        "macro": macro,
        "per_object": per_object,
        "verdict": {
            "fallback_vs_always_guided_seg_AUROC_0.05": delta_auroc,
            "fallback_vs_always_guided_seg_F1": delta_f1,
            "guided_harm_objects": sorted(guided_harm),
            "fallback_harm_objects": sorted(fallback_harm),
            "direct_improvement_gate": direct_improvement,
            "guided_harm_elimination_gate": eliminates_guided_harm and no_macro_loss,
            "continuation": continuation,
            "decision": "PROMISING_DIAGNOSTIC" if continuation else "NO_CONTINUE",
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
