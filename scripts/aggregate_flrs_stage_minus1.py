#!/usr/bin/env python3
"""Aggregate all-eight FLRS Stage -1 object summaries and apply frozen gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

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
VARIANTS = (
    "flrs",
    "raw_state",
    "random_neighbor",
    "spatial_permutation",
    "rgb_gradient",
    "radius1_only",
)
HARD_NULLS = ("raw_state", "random_neighbor", "spatial_permutation", "rgb_gradient")
POPULATIONS = ("clean", "shifted")
METRICS = ("pAUROC_0.05", "F1")
SUPPORT_SHA256 = "9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5"
PROTOCOL_ID = "flrs-stage-minus1-support-only-20260718"
BOOTSTRAPS = 2000


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--require-all8", action="store_true")
    return parser.parse_args(list(argv))


def _finite(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        message = f"{name} must be finite"
        raise ValueError(message)
    return result


def _load_summaries(run_root: Path) -> list[dict[str, Any]]:
    paths = sorted(run_root.glob("chunks/*/*_summary.json"))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    objects = [str(payload.get("object")) for payload in payloads]
    if len(objects) != len(set(objects)):
        raise RuntimeError("duplicate object summaries")
    if set(objects) != set(OBJECTS):
        message = f"all-eight object parity failed: {sorted(objects)}"
        raise RuntimeError(message)
    return sorted(payloads, key=lambda payload: OBJECTS.index(payload["object"]))


def _metric(payload: Mapping[str, Any], variant: str, population: str, metric: str) -> float:
    return _finite(
        payload["defect"][variant][population][metric],
        f"{variant}/{population}/{metric}",
    )


def _object_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    object_name = str(payload["object"])
    if payload.get("schema") != "flrs-stage-minus1-object-v1":
        message = f"{object_name}: schema mismatch"
        raise RuntimeError(message)
    shift_flrs = _finite(payload["shift"]["flrs"]["positive_p95_inflation_mean"], "flrs shift")
    shift_raw = _finite(payload["shift"]["raw_state"]["positive_p95_inflation_mean"], "raw shift")
    shift_reduction = float("nan") if shift_raw <= 1e-12 else 1.0 - shift_flrs / shift_raw
    row: dict[str, Any] = {
        "object": object_name,
        "flrs_shift_inflation": shift_flrs,
        "raw_shift_inflation": shift_raw,
        "shift_reduction": shift_reduction,
    }
    for population in POPULATIONS:
        for metric in METRICS:
            flrs = _metric(payload, "flrs", population, metric)
            raw = _metric(payload, "raw_state", population, metric)
            envelope = max(_metric(payload, null, population, metric) for null in HARD_NULLS)
            prefix = f"{population}_{metric}"
            row[f"{prefix}_flrs"] = flrs
            row[f"{prefix}_raw"] = raw
            row[f"{prefix}_raw_delta"] = flrs - raw
            row[f"{prefix}_null_envelope"] = envelope
            row[f"{prefix}_null_margin"] = flrs - envelope
    row["worst_null_margin_pAUROC_0.05"] = min(
        row[f"{population}_pAUROC_0.05_null_margin"] for population in POPULATIONS
    )
    row["worst_null_margin_F1"] = min(
        row[f"{population}_F1_null_margin"] for population in POPULATIONS
    )
    pairs = payload.get("shift_pairs", [])
    shift_pair_ok = len(pairs) == 96 * len(VARIANTS)
    for variant in VARIANTS:
        selected = [item for item in pairs if item.get("variant") == variant]
        expected = _finite(payload["shift"][variant]["positive_p95_inflation_mean"], "shift mean")
        observed = (
            float(np.mean([_finite(item["positive_inflation"], "pair") for item in selected]))
            if selected
            else float("nan")
        )
        shift_pair_ok = bool(
            shift_pair_ok and len(selected) == 96 and np.isclose(observed, expected, atol=1e-10),
        )
    coverage_ok = all(
        payload["coverage"][variant].get("complete") is True
        and payload["coverage"][variant].get("expected") == 240
        for variant in VARIANTS
    )
    access = payload.get("access_audit", {})
    access_ok = (
        access.get("test_public_accessed") is False
        and access.get("test_private_accessed") is False
        and access.get("ground_truth_accessed") is False
        and access.get("query_update") is False
        and access.get("query_transform") is False
        and access.get("tta") is False
        and access.get("forbidden_path_hits") == []
    )
    support = payload.get("support", [])
    support_ok = (
        len(support) == 16
        and sorted(int(item["fold"]) for item in support) == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4
        and all("/train/good/" in str(item["relative_path"]) for item in support)
        and all(
            not any(
                part in str(item["relative_path"])
                for part in ("test_public", "test_private", "ground_truth")
            )
            for item in support
        )
    )
    row.update(
        {
            "coverage_ok": coverage_ok,
            "shift_pair_ok": shift_pair_ok,
            "access_ok": access_ok,
            "support_ok": support_ok,
            "validity_ok": coverage_ok and shift_pair_ok and access_ok and support_ok,
        },
    )
    return row


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([_finite(row[key], key) for row in rows]))


def _macro(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    raw = _mean(rows, "raw_shift_inflation")
    flrs = _mean(rows, "flrs_shift_inflation")
    result = {
        "flrs_shift_inflation": flrs,
        "raw_shift_inflation": raw,
        "shift_reduction": float("nan") if raw <= 1e-12 else 1.0 - flrs / raw,
    }
    for population in POPULATIONS:
        for metric in METRICS:
            for suffix in ("flrs", "raw", "raw_delta", "null_envelope", "null_margin"):
                key = f"{population}_{metric}_{suffix}"
                result[key] = _mean(rows, key)
    result["worst_null_margin_pAUROC_0.05"] = _mean(rows, "worst_null_margin_pAUROC_0.05")
    result["worst_null_margin_F1"] = _mean(rows, "worst_null_margin_F1")
    return result


def _bootstrap(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, float]], dict[str, float], int]:
    seed = int.from_bytes(
        hashlib.sha256(f"{PROTOCOL_ID}\x1fobject-bootstrap".encode()).digest()[:8],
        "big",
    )
    rng = np.random.default_rng(seed)
    replicates: list[dict[str, float]] = []
    for _ in range(BOOTSTRAPS):
        sample = [rows[index] for index in rng.integers(0, len(rows), size=len(rows))]
        values = _macro(sample)
        replicates.append(
            {
                "shift_reduction": values["shift_reduction"],
                "null_margin_pAUROC_0.05": values["worst_null_margin_pAUROC_0.05"],
                "null_margin_F1": values["worst_null_margin_F1"],
            },
        )
    lcbs = {
        key: float(np.percentile([item[key] for item in replicates], 2.5)) for key in replicates[0]
    }
    return replicates, lcbs, seed


def _gates(
    rows: Sequence[Mapping[str, Any]],
    macro: Mapping[str, float],
    lcbs: Mapping[str, float],
) -> dict[str, Any]:
    g1 = (
        all(row["raw_shift_inflation"] > 1e-12 for row in rows)
        and macro["shift_reduction"] >= 0.50
        and all(row["shift_reduction"] >= 0.25 for row in rows)
    )
    g2 = all(
        macro[f"{population}_pAUROC_0.05_raw_delta"] >= -0.005
        and macro[f"{population}_F1_raw_delta"] >= -0.010
        for population in POPULATIONS
    )
    g3 = all(
        row[f"{population}_pAUROC_0.05_null_margin"] >= 0.005
        and row[f"{population}_F1_null_margin"] >= 0.005
        for row in rows
        for population in POPULATIONS
    )
    g4 = all(value > 0.0 for value in lcbs.values())
    g5 = all(row["coverage_ok"] for row in rows)
    g6 = all(row["shift_pair_ok"] and row["access_ok"] and row["support_ok"] for row in rows)
    strict = all((g1, g2, g3, g4, g5, g6))
    positive_null_objects = int(
        sum(
            all(
                row[f"{population}_{metric}_null_margin"] > 0.0
                for population in POPULATIONS
                for metric in METRICS
            )
            for row in rows
        ),
    )
    no_collapse = all(
        row[f"{population}_pAUROC_0.05_raw_delta"] >= -0.020
        and row[f"{population}_F1_raw_delta"] >= -0.030
        for row in rows
        for population in POPULATIONS
    )
    soft = (
        g5
        and g6
        and macro["shift_reduction"] >= 0.25
        and g2
        and positive_null_objects >= 4
        and no_collapse
    )
    return {
        "G1_shift_reduction": g1,
        "G2_raw_noninferiority": g2,
        "G3_hard_null_superiority": g3,
        "G4_bootstrap_lcb": g4,
        "G5_coverage": g5,
        "G6_support_access_audit": g6,
        "strict": strict,
        "soft": soft,
        "positive_null_objects": positive_null_objects,
        "no_object_collapse": no_collapse,
    }


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    run_root = Path(args.run_root).resolve(strict=True)
    payloads = _load_summaries(run_root)
    rows = [_object_row(payload) for payload in payloads]
    macro = _macro(rows)
    bootstrap_rows, lcbs, seed = _bootstrap(rows)
    gates = _gates(rows, macro, lcbs)
    if gates["strict"]:
        diagnostic = "PROMISING_DIAGNOSTIC"
    elif gates["soft"]:
        diagnostic = "SOFT_CONTINUE_ONE_CYCLE"
    else:
        diagnostic = "NO_CONTINUE"
    source_hashes = {}
    for path in sorted(run_root.glob("chunks/*/run_manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("support_manifest_sha256") != SUPPORT_SHA256:
            raise RuntimeError("support manifest hash parity failed")
        source_hashes[str(path.relative_to(run_root))] = hashlib.sha256(
            path.read_bytes(),
        ).hexdigest()
    if len(source_hashes) != 8:
        raise RuntimeError("expected one run manifest per object")
    summary = {
        "schema": "flrs-stage-minus1-summary-v1",
        "protocol_id": PROTOCOL_ID,
        "objects": list(OBJECTS),
        "support_manifest_sha256": SUPPORT_SHA256,
        "test_public_accessed": False,
        "test_private_accessed": False,
        "tta_used": False,
        "per_object": rows,
        "macro": macro,
        "bootstrap": {"replicates": BOOTSTRAPS, "unit": "object", "seed": seed, "lcb_2.5": lcbs},
        "gates": gates,
        "claim_verdict": "KILL_FOR_CLAIM",
        "diagnostic_verdict": diagnostic,
        "public_shadow_authorized": bool(gates["strict"]),
        "revision_authorized": bool(not gates["strict"] and gates["soft"]),
        "claim_boundary": "support-only synthetic diagnostic; never a public method claim",
        "source_manifest_sha256": source_hashes,
    }
    (run_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_tsv(run_root / "per_object.tsv", rows)
    _write_tsv(run_root / "bootstrap.tsv", bootstrap_rows)
    (run_root / "verdict.txt").write_text(
        f"claim={summary['claim_verdict']}\ndiagnostic={diagnostic}\npublic_shadow_authorized={str(summary['public_shadow_authorized']).lower()}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
