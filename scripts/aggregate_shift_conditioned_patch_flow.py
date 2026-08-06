#!/usr/bin/env python3
# ruff: noqa: COM812, EM102
"""Aggregate the shift-conditioned patch-local FlowTTE phase artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

OBJECTS = ("fruit_jelly", "walnuts", "vial", "fabric")
PROTOCOL_ID = "shift-conditioned-patch-local-flow-20260804"
FROZEN_SUPPORT_MANIFEST_SHA256 = "9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5"


class AggregateError(RuntimeError):
    """Raised when artifacts are malformed or violate custody."""


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("mechanism", "public"), required=True)
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
    if not isinstance(payload, dict):
        raise AggregateError(f"{path}: summary must be a JSON object")
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise AggregateError(f"{path}: protocol mismatch")
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


def audit_common(summaries: Sequence[Mapping[str, Any]], phase: str) -> None:
    objects = [str(item.get("object")) for item in summaries]
    _require(set(objects) == set(OBJECTS), f"{phase}: expected exactly {OBJECTS}, got {objects}")
    for summary in summaries:
        context = str(summary.get("object"))
        _require(summary.get("phase") == phase, f"{context}: phase mismatch")
        _require(
            summary.get("support_manifest_sha256") in {FROZEN_SUPPORT_MANIFEST_SHA256, None},
            f"{context}: support manifest SHA mismatch",
        )
        ledger = summary.get("access_ledger")
        _require(isinstance(ledger, dict), f"{context}: missing access ledger")
        _require(not bool(ledger.get("test_private_opened")), f"{context}: private data accessed")
        _require(
            not bool(summary.get("query_adaptation")), f"{context}: query adaptation is forbidden"
        )


def mechanism_verdict(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    audit_common(summaries, "mechanism")
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        object_name = str(summary["object"])
        _require(
            not bool(summary.get("test_public_accessed")),
            f"{object_name}: public accessed in mechanism",
        )
        ledger = summary["access_ledger"]
        _require(
            not bool(ledger.get("test_public_enumerated")),
            f"{object_name}: public enumerated in mechanism",
        )
        mechanism = summary.get("mechanism")
        _require(isinstance(mechanism, dict), f"{object_name}: missing mechanism block")
        validity = mechanism.get("validity", {})
        held = mechanism.get("held_shift", {})
        identity = mechanism.get("identity", {})
        defect = mechanism.get("defect_retention", {})
        shift = mechanism.get("shift_prediction", {})
        row = {
            "object": object_name,
            "valid": (
                _float(validity, "inverse_reconstruction_max_error", object_name) <= 1e-5
                and _float(validity, "logdet_cancellation_max_error", object_name) <= 1e-5
                and _float(validity, "e0_e1_latent_max_error", object_name) < 1e-6
                and _float(validity, "e0_e1_logdet_max_error", object_name) < 1e-6
                and bool(validity.get("gradient_finite_nonzero"))
                and bool(validity.get("deterministic_repeat"))
                and bool(validity.get("descriptor_witness"))
                and bool(validity.get("raw_euclidean_memory_scoring"))
            ),
            "reduction": _float(held, "correct_reduction_fraction", object_name),
            "correct_beats_zero_shuffled": bool(held.get("correct_beats_zero_shuffled")),
            "loo_delta": abs(_float(identity, "loo_delta_e0_std", object_name)),
            "latent_std_ratio": _float(identity, "latent_std_ratio", object_name),
            "gap_retention": _float(defect, "gap_retention", object_name),
            "shift_beats_zero": bool(shift.get("beats_zero")),
            "shift_loss": _float(shift, "loss", object_name),
            "zero_loss": _float(shift, "zero_loss", object_name),
        }
        rows.append(row)
    validity_ok = all(row["valid"] for row in rows)
    mean_reduction = float(np.mean([row["reduction"] for row in rows]))
    beat_count = sum(1 for row in rows if row["correct_beats_zero_shuffled"])
    identity_ok = all(
        row["loo_delta"] <= 0.05 and 0.8 <= row["latent_std_ratio"] <= 1.25 for row in rows
    )
    retention_macro = float(np.mean([row["gap_retention"] for row in rows]))
    retention_ok = retention_macro >= 0.95 and all(row["gap_retention"] >= 0.90 for row in rows)
    shift_ok = all(row["shift_beats_zero"] for row in rows)
    full_pass = (
        validity_ok
        and mean_reduction >= 0.05
        and beat_count >= 3
        and identity_ok
        and retention_ok
        and shift_ok
    )
    partial = validity_ok and beat_count >= 2 and retention_ok and shift_ok
    if full_pass:
        verdict = "MECHANISM_PASS"
    elif not validity_ok:
        verdict = "EARLY_VALIDITY_KILL"
    elif partial:
        verdict = "CONTINUE_DIAGNOSTIC"
    else:
        verdict = "KILL_FOR_CLAIM"
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "phase": "mechanism",
        "verdict": verdict,
        "public_access_authorized": verdict == "MECHANISM_PASS",
        "objects": rows,
        "gates": {
            "validity_ok": validity_ok,
            "mean_correct_reduction": mean_reduction,
            "correct_beats_zero_shuffled_count": beat_count,
            "identity_ok": identity_ok,
            "retention_macro": retention_macro,
            "retention_ok": retention_ok,
            "shift_prediction_ok": shift_ok,
        },
        "checkpoints": {
            str(summary["object"]): str(summary.get("checkpoint", "")) for summary in summaries
        },
        "summary_sha256": {
            str(summary["object"]): summary.get("_summary_sha256") for summary in summaries
        },
    }


def public_verdict(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    audit_common(summaries, "public")
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        object_name = str(summary["object"])
        _require(bool(summary.get("test_public_accessed")), f"{object_name}: public not accessed")
        diagnostics = summary.get("diagnostics", {})
        metrics = summary.get("metrics", {})
        _require(
            isinstance(diagnostics, dict) and isinstance(metrics, dict),
            f"{object_name}: missing public metrics",
        )
        e0 = metrics.get("E0", {})
        e1 = metrics.get("E1", {})
        e5 = metrics.get("E5", {})
        row = {
            "object": object_name,
            "e0_pauroc": _float(e0, "seg_AUROC_0.05", object_name),
            "e1_pauroc": _float(e1, "seg_AUROC_0.05", object_name),
            "e5_pauroc": _float(e5, "seg_AUROC_0.05", object_name),
            "e0_f1": _float(e0, "seg_F1", object_name),
            "e5_f1": _float(e5, "seg_F1", object_name),
            "gap_delta": _float(diagnostics, "bad_image_gap_z_delta", object_name),
            "shifted_good_reduction": _float(
                diagnostics, "shifted_good_p95_inflation_reduction", object_name
            ),
            "e0_e1_maps_equal": bool(diagnostics.get("e0_e1_maps_equal")),
            "finite_nonconstant_complete": bool(diagnostics.get("finite_nonconstant_complete")),
        }
        rows.append(row)
    mean_e0 = float(np.mean([row["e0_pauroc"] for row in rows]))
    mean_e5 = float(np.mean([row["e5_pauroc"] for row in rows]))
    e0_e1_ok = all(
        row["e0_e1_maps_equal"] and abs(row["e0_pauroc"] - row["e1_pauroc"]) <= 1e-6 for row in rows
    )
    pauroc_ok = mean_e5 >= mean_e0 - 0.003
    gap_ok = float(np.mean([row["gap_delta"] for row in rows])) >= 0.0 and all(
        row["gap_delta"] >= -0.05 for row in rows
    )
    shift_ok = float(np.mean([row["shifted_good_reduction"] for row in rows])) >= 0.05
    fruit = next(row for row in rows if row["object"] == "fruit_jelly")
    fruit_ok = fruit["e5_f1"] >= fruit["e0_f1"] - 0.05
    maps_ok = all(row["finite_nonconstant_complete"] for row in rows)
    keep = e0_e1_ok and pauroc_ok and gap_ok and shift_ok and fruit_ok and maps_ok
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "phase": "public",
        "verdict": "PROMISING_DIAGNOSTIC" if keep else "PUBLIC_KILL",
        "objects": rows,
        "gates": {
            "e0_e1_parity_ok": e0_e1_ok,
            "mean_e0_pauroc": mean_e0,
            "mean_e5_pauroc": mean_e5,
            "pauroc_no_harm_ok": pauroc_ok,
            "gap_ok": gap_ok,
            "shifted_good_reduction_ok": shift_ok,
            "fruit_jelly_f1_ok": fruit_ok,
            "maps_ok": maps_ok,
        },
    }


def write_outputs(output_root: Path, payload: Mapping[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "verdict.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if payload.get("verdict") == "MECHANISM_PASS":
        (output_root / "mechanism_passed.txt").write_text(
            json.dumps(
                {
                    "verdict": "MECHANISM_PASS",
                    "protocol_id": PROTOCOL_ID,
                    "checkpoints": payload.get("checkpoints", {}),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def aggregate(phase: str, summary_paths: Sequence[Path], output_root: Path) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for path in summary_paths:
        summary = load_summary(path)
        summary["_summary_sha256"] = sha256_file(path)
        summaries.append(summary)
    payload = mechanism_verdict(summaries) if phase == "mechanism" else public_verdict(summaries)
    write_outputs(output_root, payload)
    return payload


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    payload = aggregate(args.phase, [Path(path) for path in args.summaries], Path(args.output_root))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
