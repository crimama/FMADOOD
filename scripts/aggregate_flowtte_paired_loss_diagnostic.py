# ruff: noqa: ANN401, COM812, EM102
"""Aggregate the FlowTTE paired-loss support-only diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np

OBJECTS = ("fruit_jelly", "walnuts", "vial", "fabric")
PROTOCOL_ID = "flowtte-paired-loss-diagnostic-20260804"
BASELINE_METHOD = "Baseline1"
BASELINE_ARM_ID = "V0_default"
FROZEN_SUPPORT_MANIFEST_SHA256 = "9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5"
VARIANTS = (
    "V0_default",
    "V1_aug_nll",
    "V2_raw_pair_nll1",
    "V3_raw_pair_nll01",
    "C1_normalized_pair",
    "C2_random_pair",
)
PROMOTION_CANDIDATES = ("V2_raw_pair_nll1", "V3_raw_pair_nll01")


class AggregateError(RuntimeError):
    """Raised when artifacts are malformed or violate custody."""


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


def load_summary(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AggregateError(f"cannot read summary {path}: {error}") from error
    _require(isinstance(payload, dict), f"{path}: summary must be a JSON object")
    _require(payload.get("protocol_id") == PROTOCOL_ID, f"{path}: protocol mismatch")
    payload["_summary_sha256"] = sha256_file(path)
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AggregateError(message)


def _float(mapping: Mapping[str, Any], key: str, context: str) -> float:
    try:
        value = float(mapping[key])
    except (KeyError, TypeError, ValueError) as error:
        raise AggregateError(f"{context}: missing/nonnumeric {key}") from error
    _require(np.isfinite(value), f"{context}: non-finite {key}")
    return value


def audit_common(summaries: Sequence[Mapping[str, Any]]) -> None:
    objects = [str(summary.get("object")) for summary in summaries]
    _require(set(objects) == set(OBJECTS), f"expected exactly {OBJECTS}, got {objects}")
    _require(len(objects) == len(OBJECTS), "duplicate or missing object summary")
    for summary in summaries:
        context = str(summary.get("object"))
        _require(summary.get("phase") == "mechanism", f"{context}: phase mismatch")
        _require(
            summary.get("support_manifest_sha256") == FROZEN_SUPPORT_MANIFEST_SHA256,
            f"{context}: support manifest SHA mismatch",
        )
        _require(int(summary.get("support_count", -1)) == 16, f"{context}: support mismatch")
        ledger = summary.get("access_ledger")
        _require(isinstance(ledger, dict), f"{context}: missing access ledger")
        _require(not bool(summary.get("test_public_accessed")), f"{context}: public accessed")
        _require(not bool(ledger.get("test_public_enumerated")), f"{context}: public enumerated")
        _require(not bool(ledger.get("test_public_opened")), f"{context}: public opened")
        _require(not bool(ledger.get("test_private_opened")), f"{context}: private opened")
        _require(not bool(ledger.get("ground_truth_opened")), f"{context}: GT opened")
        _require(not bool(summary.get("query_adaptation")), f"{context}: query adaptation")


def aggregate_mechanism(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    audit_common(summaries)
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        object_name = str(summary["object"])
        variants = _variants(summary, object_name)
        for variant in VARIANTS:
            item = variants[variant]
            validity = _mapping(item.get("validity"), f"{object_name}/{variant}/validity")
            row = {
                "object": object_name,
                "variant": variant,
                "shifted_normal_inflation": _float(
                    item, "shifted_normal_inflation", f"{object_name}/{variant}"
                ),
                "shift_reduction": _float(item, "shift_reduction", f"{object_name}/{variant}"),
                "loo_delta": abs(_float(item, "loo_delta", f"{object_name}/{variant}")),
                "std_ratio": _float(item, "std_ratio", f"{object_name}/{variant}"),
                "effective_rank_ratio": _float(
                    item, "effective_rank_ratio", f"{object_name}/{variant}"
                ),
                "neighbor_retention": _float(
                    item, "neighbor_retention", f"{object_name}/{variant}"
                ),
                "defect_retention": _float(item, "defect_retention", f"{object_name}/{variant}"),
                "validity": validity,
                "valid": _validity_passed(validity),
                "runner_gate": bool(item.get("gate_passed")),
            }
            row["object_safety_gate"] = _object_safety_gate(row)
            rows.append(row)
    by_variant = {
        variant: [row for row in rows if row["variant"] == variant] for variant in VARIANTS
    }
    variant_gates = {
        variant: _variant_gate_payload(variant, items) for variant, items in by_variant.items()
    }
    candidate_passes = {
        variant: bool(variant_gates[variant]["full_gate"]) for variant in PROMOTION_CANDIDATES
    }
    c2_pass = bool(variant_gates["C2_random_pair"]["full_gate"])
    v1_pass = bool(variant_gates["V1_aug_nll"]["full_gate"])
    # Only the frozen baseline is a protocol-wide validity sentinel.  Every
    # candidate/control still has to pass its own validity and safety gates,
    # but a deliberately destructive negative control must not invalidate
    # otherwise interpretable candidate measurements.
    baseline_validity_ok = all(row["valid"] for row in by_variant[BASELINE_ARM_ID])
    all_variant_validity_ok = all(row["valid"] for row in rows)
    absorption = any(
        float(variant_gates[variant]["macro_shift_reduction"]) > 0.0
        and not bool(variant_gates[variant]["defect_retention_ok"])
        for variant in PROMOTION_CANDIDATES
    )
    if not baseline_validity_ok:
        verdict = "EARLY_VALIDITY_KILL"
    elif any(candidate_passes.values()) and not c2_pass:
        verdict = "MECHANISM_PASS"
    elif v1_pass and not any(candidate_passes.values()) and not c2_pass:
        verdict = "CONTINUE_DIAGNOSTIC"
    else:
        verdict = "KILL_FOR_CLAIM"
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "baseline_method": BASELINE_METHOD,
        "baseline_arm_id": BASELINE_ARM_ID,
        "phase": "mechanism",
        "verdict": verdict,
        "public_access_authorized": False,
        "all8_support_authorized": verdict == "MECHANISM_PASS",
        "objects": rows,
        "variant_gates": variant_gates,
        "candidate_passes": candidate_passes,
        "baseline_validity_ok": baseline_validity_ok,
        "all_variant_validity_ok": all_variant_validity_ok,
        "controls": {
            "v1_aug_nll_full_gate": v1_pass,
            "c2_random_pair_pass": c2_pass,
            "random_null_blocks_claim": c2_pass,
            "anomaly_absorption_indicated": absorption,
        },
        "checkpoints": {
            str(summary["object"]): str(summary.get("checkpoint", "")) for summary in summaries
        },
        "summary_sha256": {
            str(summary["object"]): str(summary.get("_summary_sha256")) for summary in summaries
        },
    }


def _variants(summary: Mapping[str, Any], context: str) -> Mapping[str, Mapping[str, Any]]:
    mechanism = _mapping(summary.get("mechanism"), f"{context}/mechanism")
    variants = _mapping(mechanism.get("variants"), f"{context}/mechanism/variants")
    for variant in VARIANTS:
        _require(isinstance(variants.get(variant), dict), f"{context}: missing {variant}")
    return cast("Mapping[str, Mapping[str, Any]]", variants)


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{context}: expected object")
    return cast("Mapping[str, Any]", value)


def _validity_passed(validity: Mapping[str, Any]) -> bool:
    return bool(
        _float(validity, "inverse_reconstruction_max_error", "validity") <= 1e-5
        and _float(validity, "logdet_cancellation_max_error", "validity") <= 2e-4
        and bool(validity.get("finite_gradients"))
        and bool(validity.get("deterministic_repeat"))
        and bool(validity.get("finite_outputs"))
        and bool(validity.get("raw_euclidean_memory_scoring"))
    )


def _object_safety_gate(row: Mapping[str, Any]) -> bool:
    return bool(
        row["valid"]
        and row["loo_delta"] <= 0.05
        and 0.8 <= row["std_ratio"] <= 1.25
        and row["effective_rank_ratio"] >= 0.90
        and row["neighbor_retention"] >= 0.90
        and row["defect_retention"] >= 0.90
    )


def _variant_gate_payload(variant: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    macro_shift = float(np.mean([float(row["shift_reduction"]) for row in rows]))
    improved = sum(1 for row in rows if float(row["shift_reduction"]) > 0.0)
    macro_defect = float(np.mean([float(row["defect_retention"]) for row in rows]))
    safety_ok = all(bool(row["object_safety_gate"]) for row in rows)
    defect_retention_ok = macro_defect >= 0.97 and all(
        float(row["defect_retention"]) >= 0.90 for row in rows
    )
    full_gate = (
        variant in {"V1_aug_nll", "V2_raw_pair_nll1", "V3_raw_pair_nll01", "C2_random_pair"}
        and macro_shift >= 0.10
        and improved >= 3
        and safety_ok
        and defect_retention_ok
    )
    return {
        "macro_shift_reduction": macro_shift,
        "improved_object_count": improved,
        "macro_defect_retention": macro_defect,
        "validity_ok": all(bool(row["valid"]) for row in rows),
        "safety_ok": safety_ok,
        "defect_retention_ok": defect_retention_ok,
        "full_gate": full_gate,
        "objects": {str(row["object"]): row for row in rows},
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
                    "checkpoints": payload.get("checkpoints", {}),
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
