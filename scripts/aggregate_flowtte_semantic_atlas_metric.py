# ruff: noqa: ANN401, COM812, EM102
"""Aggregate the support-only FlowTTE SemanticAtlasMetric diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np

OBJECTS = ("fruit_jelly", "walnuts", "vial", "fabric")
PROTOCOL_ID = "flowtte-semantic-atlas-metric-stage-minus1-20260805"
BASELINE_METHOD = "Baseline1-core"
BASELINE_ARM_ID = "A0_baseline1_core"
PROPOSED_ARM_ID = "A1_semantic_atlas_orth"
FROZEN_SUPPORT_MANIFEST_SHA256 = "9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5"
VARIANTS = (
    "A0_baseline1_core",
    "A1_semantic_atlas_orth",
    "A2_semantic_atlas_raw",
    "A3_global_rank4",
    "A4_cluster_shuffle",
    "A5_random_rank1",
)
CONTROL_ARMS = ("A4_cluster_shuffle", "A5_random_rank1")
FOLD_COUNT = 4


class AggregateError(RuntimeError):
    """Raised when artifacts are malformed or violate the frozen protocol."""


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("mechanism",), required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--summaries", nargs="+", required=True)
    return parser.parse_args(list(argv))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AggregateError(message)


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{context}: expected object")
    return cast("Mapping[str, Any]", value)


def _sequence(value: Any, context: str) -> Sequence[Any]:
    _require(isinstance(value, list | tuple), f"{context}: expected list")
    return cast("Sequence[Any]", value)


def _float(mapping: Mapping[str, Any], key: str, context: str) -> float:
    try:
        value = float(mapping[key])
    except (KeyError, TypeError, ValueError) as error:
        raise AggregateError(f"{context}: missing/nonnumeric {key}") from error
    _require(np.isfinite(value), f"{context}: non-finite {key}")
    return value


def _optional_float(mapping: Mapping[str, Any], key: str, context: str) -> float | None:
    if key not in mapping or mapping[key] is None:
        return None
    value = float(mapping[key])
    _require(np.isfinite(value), f"{context}: non-finite {key}")
    return value


def _bool(mapping: Mapping[str, Any], key: str, context: str) -> bool:
    _require(key in mapping, f"{context}: missing {key}")
    value = mapping[key]
    _require(isinstance(value, bool), f"{context}: non-boolean {key}")
    return value


def load_summary(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AggregateError(f"cannot read summary {path}: {error}") from error
    _require(isinstance(payload, dict), f"{path}: summary must be a JSON object")
    _require(payload.get("protocol_id") == PROTOCOL_ID, f"{path}: protocol mismatch")
    payload["_summary_sha256"] = sha256_file(path)
    return payload


def _support_count(summary: Mapping[str, Any], context: str) -> int:
    support = summary.get("support")
    if isinstance(support, dict) and "count" in support:
        return int(support["count"])
    if "support_count" in summary:
        return int(summary["support_count"])
    raise AggregateError(f"{context}: missing support count")


def audit_common(summaries: Sequence[Mapping[str, Any]]) -> None:
    objects = [str(summary.get("object")) for summary in summaries]
    _require(set(objects) == set(OBJECTS), f"expected exactly {OBJECTS}, got {objects}")
    _require(len(objects) == len(OBJECTS), "duplicate or missing object summary")
    for summary in summaries:
        context = str(summary.get("object"))
        _require(
            summary.get("support_manifest_sha256") == FROZEN_SUPPORT_MANIFEST_SHA256,
            f"{context}: support manifest SHA mismatch",
        )
        _require(_support_count(summary, context) == 16, f"{context}: support mismatch")
        ledger = _mapping(summary.get("access_ledger"), f"{context}/access_ledger")
        forbidden_flags = (
            "test_public_enumerated",
            "test_public_opened",
            "test_private_enumerated",
            "test_private_opened",
            "ground_truth_enumerated",
            "ground_truth_opened",
            "query_adaptation",
        )
        for flag in forbidden_flags:
            _require(not bool(ledger.get(flag)), f"{context}: forbidden access {flag}")
        _require(not bool(summary.get("query_adaptation")), f"{context}: query adaptation")
        support = summary.get("support")
        if isinstance(support, dict):
            _require(bool(support.get("source_disjoint_folds", True)), f"{context}: folds leak")
            fold_counts = support.get("fold_counts")
            if fold_counts is not None:
                _require(list(fold_counts) == [4, 4, 4, 4], f"{context}: fold counts mismatch")


def _variants(summary: Mapping[str, Any], context: str) -> Mapping[str, Mapping[str, Any]]:
    mechanism = _mapping(summary.get("mechanism"), f"{context}/mechanism")
    variants = _mapping(mechanism.get("variants"), f"{context}/mechanism/variants")
    for variant in VARIANTS:
        _require(isinstance(variants.get(variant), dict), f"{context}: missing {variant}")
    return cast("Mapping[str, Mapping[str, Any]]", variants)


def _validity_passed(validity: Mapping[str, Any], context: str) -> bool:
    min_eigenvalue = _optional_float(validity, "min_eigenvalue", context)
    orth_error = _optional_float(validity, "orthonormality_max_error", context)
    return bool(
        _bool(validity, "finite_outputs", context)
        and _bool(validity, "nonconstant_scores", context)
        and _bool(validity, "deterministic_repeat", context)
        and _bool(validity, "source_disjoint_folds", context)
        and _bool(validity, "support_only_custody", context)
        and (min_eigenvalue is None or min_eigenvalue >= 0.75)
        and (orth_error is None or orth_error <= 1e-4)
    )


def _fold_improved_count(item: Mapping[str, Any], context: str) -> int:
    folds = _sequence(item.get("folds"), f"{context}/folds")
    _require(len(folds) == FOLD_COUNT, f"{context}: expected {FOLD_COUNT} folds")
    improved = 0
    for fold_index, fold in enumerate(folds):
        fold_map = _mapping(fold, f"{context}/fold[{fold_index}]")
        _require(bool(fold_map.get("source_disjoint", True)), f"{context}: fold source leak")
        if _float(fold_map, "shift_reduction", f"{context}/fold[{fold_index}]") > 0.0:
            improved += 1
    return improved


def _quantile(rows: Sequence[Mapping[str, Any]], key: str, q: float) -> float:
    return float(np.quantile([_float(row, key, "A5 seed") for row in rows], q))


def _random_q95_row(item: Mapping[str, Any], base_row: Mapping[str, Any]) -> dict[str, Any] | None:
    seed_rows = item.get("random_seed_rows")
    if seed_rows is None:
        return None
    rows = [
        _mapping(row, "A5/random_seed_rows")
        for row in _sequence(seed_rows, "A5/random_seed_rows")
    ]
    _require(len(rows) >= 2, "A5/random_seed_rows: expected at least two random seeds")
    return {
        **base_row,
        "shift_reduction": _quantile(rows, "shift_reduction", 0.95),
        "shift_energy_capture": _quantile(rows, "shift_energy_capture", 0.95),
        "defect_retention": _quantile(rows, "defect_retention", 0.05),
        "clean_mean_delta_std": float(
            np.quantile(
                [abs(_float(row, "clean_mean_delta_std", "A5 seed")) for row in rows],
                0.95,
            ),
        ),
        "clean_std_ratio_low": _quantile(rows, "clean_std_ratio", 0.05),
        "clean_std_ratio_high": _quantile(rows, "clean_std_ratio", 0.95),
        "neighbor_retention": _quantile(rows, "neighbor_retention", 0.05),
        "pair_spearman": _quantile(rows, "pair_spearman", 0.05),
        "random_seed_count": len(rows),
    }


def _row_from_variant(object_name: str, variant: str, item: Mapping[str, Any]) -> dict[str, Any]:
    context = f"{object_name}/{variant}"
    validity = _mapping(item.get("validity"), f"{context}/validity")
    row: dict[str, Any] = {
        "object": object_name,
        "variant": variant,
        "shifted_normal_inflation": _float(item, "shifted_normal_inflation", context),
        "shift_reduction": _float(item, "shift_reduction", context),
        "defect_retention": _float(item, "defect_retention", context),
        "clean_mean_delta_std": _float(item, "clean_mean_delta_std", context),
        "clean_std_ratio": _float(item, "clean_std_ratio", context),
        "neighbor_retention": _float(item, "neighbor_retention", context),
        "pair_spearman": _float(item, "pair_spearman", context),
        "shift_energy_capture": _float(item, "shift_energy_capture", context),
        "disabled_patch_mass": _float(item, "disabled_patch_mass", context),
        "fold_improved_count": _fold_improved_count(item, context),
        "validity": dict(validity),
        "valid": _validity_passed(validity, f"{context}/validity"),
    }
    random_q95 = _random_q95_row(item, row)
    if random_q95 is not None:
        row["random_q95"] = random_q95
    return row


def _atlas_cluster_ari(summary: Mapping[str, Any], context: str) -> float:
    robustness = _mapping(summary.get("atlas_robustness"), f"{context}/atlas_robustness")
    return _float(robustness, "cluster_ari", f"{context}/atlas_robustness")


def _variant_rows(rows: Sequence[Mapping[str, Any]], variant: str) -> list[Mapping[str, Any]]:
    return [row for row in rows if row["variant"] == variant]


def _macro(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _variant_gate_payload(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    macro_shift = _macro(rows, "shift_reduction")
    macro_defect = _macro(rows, "defect_retention")
    shift_positive_objects = sum(float(row["shift_reduction"]) > 0.0 for row in rows)
    fold_positive_objects = sum(int(row["fold_improved_count"]) >= 3 for row in rows)
    defect_ok = macro_defect >= 0.97 and all(float(row["defect_retention"]) >= 0.90 for row in rows)
    high_shift_defect_ok = all(
        float(row["defect_retention"]) >= 0.95
        for row in rows
        if float(row["shift_reduction"]) >= 0.10
    )
    clean_ok = all(
        abs(float(row["clean_mean_delta_std"])) <= 0.05
        and 0.8 <= float(row["clean_std_ratio"]) <= 1.25
        for row in rows
    )
    geometry_ok = all(
        float(row["neighbor_retention"]) >= 0.95 and float(row["pair_spearman"]) >= 0.95
        for row in rows
    )
    atlas_ok = all(float(row["disabled_patch_mass"]) <= 0.20 for row in rows)
    validity_ok = all(bool(row["valid"]) for row in rows)
    shift_gate = macro_shift >= 0.10 and shift_positive_objects >= 3 and fold_positive_objects >= 3
    safety_gate = (
        defect_ok and high_shift_defect_ok and clean_ok and geometry_ok and atlas_ok and validity_ok
    )
    return {
        "macro_shift_reduction": macro_shift,
        "shift_positive_object_count": shift_positive_objects,
        "fold_positive_object_count": fold_positive_objects,
        "macro_defect_retention": macro_defect,
        "macro_shift_energy_capture": _macro(rows, "shift_energy_capture"),
        "max_abs_clean_mean_delta_std": max(
            abs(float(row["clean_mean_delta_std"])) for row in rows
        ),
        "min_clean_std_ratio": min(float(row["clean_std_ratio"]) for row in rows),
        "max_clean_std_ratio": max(float(row["clean_std_ratio"]) for row in rows),
        "min_neighbor_retention": min(float(row["neighbor_retention"]) for row in rows),
        "min_pair_spearman": min(float(row["pair_spearman"]) for row in rows),
        "max_disabled_patch_mass": max(float(row["disabled_patch_mass"]) for row in rows),
        "validity_ok": validity_ok,
        "shift_gate": shift_gate,
        "safety_gate": safety_gate,
        "same_shift_safety_gate": bool(shift_gate and safety_gate),
        "defect_retention_ok": defect_ok and high_shift_defect_ok,
        "clean_geometry_ok": clean_ok and geometry_ok,
        "atlas_coverage_ok": atlas_ok,
        "objects": {str(row["object"]): row for row in rows},
    }


def _random_q95_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    q95_rows = []
    for row in rows:
        q95 = row.get("random_q95")
        q95_rows.append(_mapping(q95, f"{row['object']}/A5/random_q95") if q95 is not None else row)
    return _variant_gate_payload(q95_rows)


def aggregate_mechanism(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    audit_common(summaries)
    rows: list[dict[str, Any]] = []
    cluster_ari_by_object: dict[str, float] = {}
    for summary in summaries:
        object_name = str(summary["object"])
        cluster_ari_by_object[object_name] = _atlas_cluster_ari(summary, object_name)
        variants = _variants(summary, object_name)
        rows.extend(
            _row_from_variant(object_name, variant, variants[variant]) for variant in VARIANTS
        )

    by_variant = {variant: _variant_rows(rows, variant) for variant in VARIANTS}
    variant_gates = {
        variant: _variant_gate_payload(items) for variant, items in by_variant.items()
    }
    variant_gates["A5_random_rank1_q95"] = _random_q95_gate(by_variant["A5_random_rank1"])

    a1 = variant_gates[PROPOSED_ARM_ID]
    a2 = variant_gates["A2_semantic_atlas_raw"]
    a3 = variant_gates["A3_global_rank4"]
    a4 = variant_gates["A4_cluster_shuffle"]
    a5q95 = variant_gates["A5_random_rank1_q95"]

    atlas_ari_macro = float(np.mean(list(cluster_ari_by_object.values())))
    atlas_robustness_ok = atlas_ari_macro >= 0.70
    identifiability_ok = bool(
        a1["macro_shift_energy_capture"] >= a4["macro_shift_energy_capture"] + 0.10
        and a1["macro_shift_energy_capture"] >= a5q95["macro_shift_energy_capture"] + 0.10
        and a1["macro_shift_energy_capture"] >= a3["macro_shift_energy_capture"] + 0.10
    )
    orthogonalization_value_ok = bool(
        a1["macro_defect_retention"] >= a2["macro_defect_retention"] + 0.03
        and a1["macro_shift_reduction"] >= a2["macro_shift_reduction"] - 0.02
    )
    non_retread_ok = bool(
        a1["macro_shift_reduction"] >= a3["macro_shift_reduction"] + 0.05
        or (
            a1["macro_shift_reduction"] >= a3["macro_shift_reduction"] - 0.02
            and a1["macro_defect_retention"] >= a3["macro_defect_retention"] + 0.03
        )
    )
    control_blocks = bool(a4["same_shift_safety_gate"] or a5q95["same_shift_safety_gate"])
    baseline_validity_ok = bool(variant_gates[BASELINE_ARM_ID]["validity_ok"])
    proposed_gate = bool(
        baseline_validity_ok
        and a1["shift_gate"]
        and a1["safety_gate"]
        and atlas_robustness_ok
        and identifiability_ok
        and orthogonalization_value_ok
        and non_retread_ok
        and not control_blocks
    )
    anomaly_absorption = bool(a1["shift_gate"] and not a1["defect_retention_ok"])
    continue_diagnostic = bool(
        baseline_validity_ok
        and a1["safety_gate"]
        and atlas_robustness_ok
        and 0.05 <= float(a1["macro_shift_reduction"]) < 0.10
        and not control_blocks
    )
    if proposed_gate:
        verdict = "MECHANISM_PASS"
    elif anomaly_absorption:
        verdict = "ANOMALY_ABSORPTION_KILL"
    elif continue_diagnostic:
        verdict = "CONTINUE_DIAGNOSTIC"
    else:
        verdict = "KILL_FOR_CLAIM"

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "baseline_method": BASELINE_METHOD,
        "baseline_arm_id": BASELINE_ARM_ID,
        "proposed_arm_id": PROPOSED_ARM_ID,
        "phase": "mechanism",
        "verdict": verdict,
        "public_access_authorized": False,
        "all8_support_authorized": verdict == "MECHANISM_PASS",
        "rows": rows,
        "variant_gates": variant_gates,
        "gates": {
            "baseline_validity_ok": baseline_validity_ok,
            "shift_correction_ok": bool(a1["shift_gate"]),
            "anti_absorption_ok": bool(a1["defect_retention_ok"]),
            "clean_geometry_ok": bool(a1["clean_geometry_ok"]),
            "atlas_robustness_ok": atlas_robustness_ok,
            "identifiability_ok": identifiability_ok,
            "orthogonalization_value_ok": orthogonalization_value_ok,
            "non_retread_ok": non_retread_ok,
            "causal_controls_ok": not control_blocks,
            "proposed_full_gate": proposed_gate,
        },
        "controls": {
            "cluster_shuffle_same_gate": bool(a4["same_shift_safety_gate"]),
            "random_q95_same_gate": bool(a5q95["same_shift_safety_gate"]),
            "generic_local_shrinkage_blocks_claim": control_blocks,
            "anomaly_absorption_indicated": anomaly_absorption,
        },
        "atlas_robustness": {
            "cluster_ari_macro": atlas_ari_macro,
            "cluster_ari_by_object": cluster_ari_by_object,
        },
        "summary_sha256": {
            str(summary["object"]): str(summary.get("_summary_sha256")) for summary in summaries
        },
        "claim_boundary": (
            "Support-only synthetic mechanism evidence. Public/private test images, ground truth, "
            "and query-adaptive fitting remain unauthorized."
        ),
    }


def write_outputs(output_root: Path, payload: Mapping[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "verdict.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if payload.get("verdict") == "MECHANISM_PASS":
        (output_root / "all8_support_authorized.txt").write_text(
            json.dumps(
                {
                    "verdict": "MECHANISM_PASS",
                    "protocol_id": PROTOCOL_ID,
                    "next_phase": "all8_support_only",
                    "public_access_authorized": False,
                    "proposed_arm_id": PROPOSED_ARM_ID,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def aggregate(phase: str, summary_paths: Sequence[Path], output_root: Path) -> dict[str, Any]:
    _require(phase == "mechanism", "public phase is forbidden for this diagnostic")
    summaries = [load_summary(path) for path in summary_paths]
    payload = aggregate_mechanism(summaries)
    write_outputs(output_root, payload)
    return payload


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    payload = aggregate(args.phase, [Path(path) for path in args.summaries], Path(args.output_root))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
