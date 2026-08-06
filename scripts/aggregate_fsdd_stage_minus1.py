#!/usr/bin/env python3
# ruff: noqa: ANN401, EM102, PLR0911
"""Aggregate FSDD-LODF Stage -1 summaries and apply the frozen support-only gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
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
SEEDS = (0, 1)
FAMILIES = ("cutpaste", "chromatic_blotch", "gaussian_blur", "high_frequency_ribbon")
AXES = ("exposure", "gamma", "contrast", "white_balance")
VARIANTS = (
    "treatment",
    "nonfactorial",
    "defect_only",
    "label_shuffle",
    "mask_shuffle",
    "boundary_only",
    "rgb_seam",
    "rgb_texture",
    "raw_state",
)
OPERATIONAL_BASELINES = ("raw_state", "nonfactorial")
HARD_NULLS = ("label_shuffle", "mask_shuffle", "boundary_only", "rgb_seam", "rgb_texture")
ACQUISITIONS = ("clean", "shifted")
METRICS = ("pAUROC_0.05", "fixed_F1")
SUPPORT_SHA256 = "9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5"
PROTOCOL_ID = "fsdd-lodf-stage-minus1-support-only-20260719-r1"
BOOTSTRAPS = 2000
RIBBON_FAMILY = "high_frequency_ribbon"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--require-all8", action="store_true")
    return parser.parse_args(list(argv))


def _finite(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _json_safe(value: Any) -> Any:
    """Convert NumPy containers/scalars and reject non-finite JSON numbers."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("JSON output contains a non-finite number")
        return result
    return value


def _load_summaries(run_root: Path) -> list[dict[str, Any]]:
    paths = sorted(run_root.glob("chunks/*/*_seed*_summary.json"))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    identities = [
        (str(item.get("object")), int(item.get("experiment_seed", -1))) for item in payloads
    ]
    if len(identities) != len(set(identities)):
        raise RuntimeError("duplicate object/seed summaries")
    expected = {(object_name, seed) for object_name in OBJECTS for seed in SEEDS}
    if set(identities) != expected:
        raise RuntimeError(f"all-eight/two-seed parity failed: {sorted(identities)}")
    return sorted(
        payloads,
        key=lambda item: (OBJECTS.index(item["object"]), int(item["experiment_seed"])),
    )


def _variant_record(value: Mapping[str, Any], context: str) -> dict[str, Any]:
    clean_fpr = _finite(value["clean_fpr"], f"{context}/clean_fpr")
    primary = value["primary_shift_fpr"]
    primary_fpr = {
        scale: _finite(primary[scale], f"{context}/primary_shift_fpr/{scale}")
        for scale in ("x1", "x2")
    }
    shift_excess = float(np.mean([max(0.0, item - clean_fpr) for item in primary_fpr.values()]))
    threshold_quantile = _finite(value["threshold_quantile"], f"{context}/threshold_quantile")
    if not np.isclose(threshold_quantile, 0.995, atol=1e-12):
        raise RuntimeError(f"{context}: threshold quantile must be 0.995")
    metrics: dict[str, dict[str, float]] = {}
    for acquisition in ACQUISITIONS:
        metrics[acquisition] = {
            metric: _finite(value[acquisition][metric], f"{context}/{acquisition}/{metric}")
            for metric in (*METRICS, "oracle_F1")
        }
    coverage = value.get("coverage", {})
    coverage_ok = all(
        coverage.get(key) is True for key in ("complete", "finite", "nonconstant", "aligned")
    )
    interior = value.get("interior", {})
    interior_ok = all(
        acquisition in interior
        and all(
            np.isfinite(
                _finite(
                    interior[acquisition][metric],
                    f"{context}/interior/{acquisition}/{metric}",
                ),
            )
            for metric in METRICS
        )
        for acquisition in ACQUISITIONS
    )
    return {
        "threshold": _finite(value["threshold"], f"{context}/threshold"),
        "threshold_quantile": threshold_quantile,
        "clean_fpr": clean_fpr,
        "primary_shift_fpr": primary_fpr,
        "shift_fpr_excess": shift_excess,
        "clean": metrics["clean"],
        "shifted": metrics["shifted"],
        "coverage_ok": bool(coverage_ok),
        "interior_ok": bool(interior_ok),
    }


def _normalize_payload(payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    object_name = str(payload.get("object"))
    seed = int(payload.get("experiment_seed", -1))
    if payload.get("schema") != "fsdd-lodf-stage-minus1-object-v1":
        raise RuntimeError(f"{object_name}/seed{seed}: schema mismatch")
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError(f"{object_name}/seed{seed}: protocol mismatch")
    if object_name not in OBJECTS or seed not in SEEDS:
        raise RuntimeError("unknown object or experiment seed")
    if payload.get("support_manifest_sha256") != SUPPORT_SHA256:
        raise RuntimeError(f"{object_name}/seed{seed}: support manifest hash mismatch")
    support = payload.get("support", [])
    support_ok = (
        len(support) == 16
        and sorted(int(item["fold"]) for item in support) == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4
        and len({str(item["relative_path"]) for item in support}) == 16
        and all("/train/good/" in str(item["relative_path"]) for item in support)
        and all(
            not any(
                token in str(item["relative_path"])
                for token in ("test_public", "test_private", "ground_truth")
            )
            for item in support
        )
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
    feature_audit = [
        item
        for item in payload.get("stimulus_audit", [])
        if item.get("kind") == "feature_visible_defect"
    ]
    feature_ok = (
        len(feature_audit) == 16 * len(FAMILIES)
        and len({(item.get("source"), item.get("family")) for item in feature_audit})
        == 16 * len(FAMILIES)
        and all(
            int(item.get("interior_pixels", 0)) > 0
            and int(item.get("boundary_pixels", 0)) > 0
            and item.get("clean", {}).get("valid") is True
            and len(item.get("shifted", {})) == len(AXES) * 2
            and all(
                value.get("valid") is True
                for value in item.get("shifted", {}).values()
            )
            for item in feature_audit
        )
    )

    normalized: list[dict[str, Any]] = []
    identities: set[tuple[str, str, str]] = set()
    support_fold = {str(item["relative_path"]): int(item["fold"]) for item in support}
    for record in payload.get("head_records", []):
        source = str(record["source"])
        family = str(record["heldout_family"])
        axis = str(record["heldout_axis"])
        fold = int(record["source_fold"])
        identity = (source, family, axis)
        if identity in identities:
            raise RuntimeError(f"{object_name}/seed{seed}: duplicate head record {identity}")
        identities.add(identity)
        if source not in support_fold or support_fold[source] != fold:
            raise RuntimeError(f"{object_name}/seed{seed}: source/fold mismatch")
        if family not in FAMILIES or axis not in AXES:
            raise RuntimeError(f"{object_name}/seed{seed}: unknown LODF/LOSF value")
        variants = record.get("variants", {})
        if set(variants) != set(VARIANTS):
            raise RuntimeError(f"{object_name}/seed{seed}/{identity}: variant parity failed")
        normalized.append(
            {
                "object": object_name,
                "seed": seed,
                "source": source,
                "source_fold": fold,
                "family": family,
                "axis": axis,
                "variants": {
                    variant: _variant_record(
                        variants[variant],
                        f"{object_name}/seed{seed}/{source}/{family}/{axis}/{variant}",
                    )
                    for variant in VARIANTS
                },
            },
        )
    expected_identities = {
        (source, family, axis) for source in support_fold for family in FAMILIES for axis in AXES
    }
    lodf_losf_ok = identities == expected_identities and len(normalized) == 16 * len(
        FAMILIES,
    ) * len(AXES)
    maps_ok = bool(normalized) and all(
        value["coverage_ok"] and value["interior_ok"]
        for record in normalized
        for value in record["variants"].values()
    )
    return normalized, {
        "support_ok": bool(support_ok),
        "access_ok": bool(access_ok),
        "feature_visibility_ok": bool(feature_ok),
        "lodf_losf_ok": bool(lodf_losf_ok),
        "maps_ok": bool(maps_ok),
    }


def _mean(records: Sequence[Mapping[str, Any]], variant: str, key: str) -> float:
    return float(
        np.mean(
            [_finite(record["variants"][variant][key], f"{variant}/{key}") for record in records],
        ),
    )


def _metric_mean(
    records: Sequence[Mapping[str, Any]],
    variant: str,
    acquisition: str,
    metric: str,
) -> float:
    return float(
        np.mean(
            [
                _finite(
                    record["variants"][variant][acquisition][metric],
                    f"{variant}/{acquisition}/{metric}",
                )
                for record in records
            ],
        ),
    )


def _reduction(candidate: float, baseline: float) -> float:
    return 0.0 if baseline <= 1e-12 else 1.0 - candidate / baseline


def _statistics(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    treatment_shift = _mean(records, "treatment", "shift_fpr_excess")
    result = {
        "treatment_clean_fpr": _mean(records, "treatment", "clean_fpr"),
        "treatment_shift_fpr_excess": treatment_shift,
    }
    for baseline in OPERATIONAL_BASELINES:
        baseline_shift = _mean(records, baseline, "shift_fpr_excess")
        result[f"{baseline}_shift_fpr_excess"] = baseline_shift
        result[f"shift_gain_vs_{baseline}"] = baseline_shift - treatment_shift
        result[f"shift_reduction_vs_{baseline}"] = _reduction(treatment_shift, baseline_shift)
    for acquisition in ACQUISITIONS:
        for metric in METRICS:
            treatment = _metric_mean(records, "treatment", acquisition, metric)
            result[f"{acquisition}_{metric}_treatment"] = treatment
            null_values = {
                variant: _metric_mean(records, variant, acquisition, metric)
                for variant in HARD_NULLS
            }
            envelope = max(null_values.values())
            result[f"{acquisition}_{metric}_hard_null_envelope"] = envelope
            result[f"{acquisition}_{metric}_hard_null_margin"] = treatment - envelope
            for baseline in OPERATIONAL_BASELINES:
                value = _metric_mean(records, baseline, acquisition, metric)
                result[f"{acquisition}_{metric}_{baseline}"] = value
                result[f"{acquisition}_{metric}_delta_vs_{baseline}"] = treatment - value
    for variant in ("treatment", "nonfactorial"):
        p_drop = _metric_mean(records, variant, "clean", "pAUROC_0.05") - _metric_mean(
            records,
            variant,
            "shifted",
            "pAUROC_0.05",
        )
        f_drop = _metric_mean(records, variant, "clean", "fixed_F1") - _metric_mean(
            records,
            variant,
            "shifted",
            "fixed_F1",
        )
        result[f"{variant}_shifted_defect_drop_pAUROC_0.05"] = p_drop
        result[f"{variant}_shifted_defect_drop_fixed_F1"] = f_drop
    return result


def _group_statistics(
    records: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record[key])].append(record)
    return {name: _statistics(items) for name, items in sorted(grouped.items())}


def _hierarchical_sample(
    records: Sequence[Mapping[str, Any]],
    rng: np.random.Generator,
) -> list[Mapping[str, Any]]:
    """Resample object -> source -> LODF/LOSF head -> seed with replacement."""
    lookup = {
        (
            str(item["object"]),
            str(item["source"]),
            str(item["family"]),
            str(item["axis"]),
            int(item["seed"]),
        ): item
        for item in records
    }
    object_names = sorted({str(item["object"]) for item in records})
    sampled: list[Mapping[str, Any]] = []
    for object_index in rng.integers(0, len(object_names), size=len(object_names)):
        object_name = object_names[int(object_index)]
        sources = sorted({key[1] for key in lookup if key[0] == object_name})
        for source_index in rng.integers(0, len(sources), size=len(sources)):
            source = sources[int(source_index)]
            heads = sorted({(key[2], key[3]) for key in lookup if key[:2] == (object_name, source)})
            for head_index in rng.integers(0, len(heads), size=len(heads)):
                family, axis = heads[int(head_index)]
                seed = SEEDS[int(rng.integers(0, len(SEEDS)))]
                sampled.append(lookup[(object_name, source, family, axis, seed)])
    return sampled


def _bootstrap(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, float]], dict[str, float], int]:
    seed = int.from_bytes(
        hashlib.sha256(f"{PROTOCOL_ID}\x1fhierarchical-bootstrap".encode()).digest()[:8],
        "big",
    )
    rng = np.random.default_rng(seed)
    keys = (
        "shift_gain_vs_raw_state",
        "shift_gain_vs_nonfactorial",
        "clean_pAUROC_0.05_hard_null_margin",
        "clean_fixed_F1_hard_null_margin",
        "shifted_pAUROC_0.05_hard_null_margin",
        "shifted_fixed_F1_hard_null_margin",
    )
    replicates: list[dict[str, float]] = []
    for _ in range(BOOTSTRAPS):
        sampled_statistics = _statistics(_hierarchical_sample(records, rng))
        replicates.append({key: sampled_statistics[key] for key in keys})
    lcbs = {key: float(np.percentile([item[key] for item in replicates], 2.5)) for key in keys}
    return replicates, lcbs, seed


def _directional_seed_gate(stats: Mapping[str, float]) -> dict[str, bool]:
    shift = all(
        stats[f"shift_reduction_vs_{baseline}"] >= 0.30 for baseline in OPERATIONAL_BASELINES
    )
    retention = all(
        stats[f"{acquisition}_pAUROC_0.05_delta_vs_{baseline}"] >= -0.005
        and stats[f"{acquisition}_fixed_F1_delta_vs_{baseline}"] >= -0.010
        for acquisition in ACQUISITIONS
        for baseline in OPERATIONAL_BASELINES
    )
    nulls = all(
        stats[f"{acquisition}_pAUROC_0.05_hard_null_margin"] >= 0.005
        and stats[f"{acquisition}_fixed_F1_hard_null_margin"] >= 0.010
        for acquisition in ACQUISITIONS
    )
    drop = (
        stats["treatment_shifted_defect_drop_pAUROC_0.05"] <= 0.010
        and stats["treatment_shifted_defect_drop_fixed_F1"] <= 0.020
        and stats["treatment_shifted_defect_drop_pAUROC_0.05"]
        < stats["nonfactorial_shifted_defect_drop_pAUROC_0.05"]
        and stats["treatment_shifted_defect_drop_fixed_F1"]
        < stats["nonfactorial_shifted_defect_drop_fixed_F1"]
    )
    return {"shift": shift, "retention": retention, "hard_null": nulls, "shifted_drop": drop}


def _gates(
    records: Sequence[Mapping[str, Any]],
    audits: Sequence[Mapping[str, bool]],
    lcbs: Mapping[str, float],
) -> dict[str, Any]:
    stats = _statistics(records)
    by_seed = {
        seed: _statistics([item for item in records if item["seed"] == seed]) for seed in SEEDS
    }
    seed_direction = {seed: _directional_seed_gate(value) for seed, value in by_seed.items()}
    subset_direction = {
        seed: {
            "ribbon_only": _directional_seed_gate(
                _statistics(
                    [
                        item
                        for item in records
                        if item["seed"] == seed and item["family"] == RIBBON_FAMILY
                    ],
                ),
            ),
            "ribbon_excluded": _directional_seed_gate(
                _statistics(
                    [
                        item
                        for item in records
                        if item["seed"] == seed and item["family"] != RIBBON_FAMILY
                    ],
                ),
            ),
        }
        for seed in SEEDS
    }
    subset_gate = all(
        all(component.values())
        for seed_subsets in subset_direction.values()
        for component in seed_subsets.values()
    )
    by_object_seed = {
        (object_name, seed): _statistics(
            [item for item in records if item["object"] == object_name and item["seed"] == seed],
        )
        for object_name in OBJECTS
        for seed in SEEDS
    }
    by_family_seed = {
        (family, seed): _statistics(
            [item for item in records if item["family"] == family and item["seed"] == seed],
        )
        for family in FAMILIES
        for seed in SEEDS
    }
    validity = bool(audits) and all(all(audit.values()) for audit in audits)
    clean_fpr = all(item["treatment_clean_fpr"] <= 0.02 for item in by_object_seed.values())
    no_shift_harm = all(
        item["treatment_shift_fpr_excess"]
        - min(item[f"{baseline}_shift_fpr_excess"] for baseline in OPERATIONAL_BASELINES)
        <= 0.01
        for item in by_object_seed.values()
    )
    shift = (
        all(value["shift"] for value in seed_direction.values())
        and no_shift_harm
        and lcbs["shift_gain_vs_raw_state"] > 0.0
        and lcbs["shift_gain_vs_nonfactorial"] > 0.0
    )
    retention = all(value["retention"] for value in seed_direction.values())
    nulls = all(value["hard_null"] for value in seed_direction.values()) and all(
        lcbs[f"{acquisition}_{metric}_hard_null_margin"] > 0.0
        for acquisition in ACQUISITIONS
        for metric in METRICS
    )
    family_raw = all(
        item[f"{acquisition}_pAUROC_0.05_delta_vs_raw_state"] >= -0.005
        and item[f"{acquisition}_fixed_F1_delta_vs_raw_state"] >= -0.010
        for item in by_family_seed.values()
        for acquisition in ACQUISITIONS
    )
    positive_families = {
        seed: sum(
            all(
                by_family_seed[(family, seed)][f"{acquisition}_{metric}_hard_null_margin"] > 0.0
                for acquisition in ACQUISITIONS
                for metric in METRICS
            )
            for family in FAMILIES
        )
        for seed in SEEDS
    }
    family_drop = all(value["shifted_drop"] for value in seed_direction.values())
    g6 = family_raw and min(positive_families.values()) >= 3 and family_drop
    strict = all((validity, clean_fpr, shift, retention, nulls, g6, subset_gate))

    soft_shift = all(
        value[f"shift_reduction_vs_{baseline}"] >= 0.20
        for value in by_seed.values()
        for baseline in OPERATIONAL_BASELINES
    )
    soft_raw = all(
        value[f"{acquisition}_pAUROC_0.05_delta_vs_raw_state"] >= -0.005
        and value[f"{acquisition}_fixed_F1_delta_vs_raw_state"] >= -0.010
        for value in by_seed.values()
        for acquisition in ACQUISITIONS
    )
    positive_objects = {
        seed: sum(
            all(
                by_object_seed[(object_name, seed)][f"{acquisition}_{metric}_hard_null_margin"]
                > 0.0
                for acquisition in ACQUISITIONS
                for metric in METRICS
            )
            for object_name in OBJECTS
        )
        for seed in SEEDS
    }
    no_family_collapse = all(
        value[f"{acquisition}_pAUROC_0.05_delta_vs_raw_state"] >= -0.020
        and value[f"{acquisition}_fixed_F1_delta_vs_raw_state"] >= -0.030
        for value in by_family_seed.values()
        for acquisition in ACQUISITIONS
    )
    soft = (
        validity
        and clean_fpr
        and soft_shift
        and soft_raw
        and min(positive_families.values()) >= 2
        and min(positive_objects.values()) >= 4
        and no_family_collapse
    )
    return {
        "G1_validity_access_lodf_losf_coverage": validity,
        "G2_clean_fpr": clean_fpr,
        "G3_shift_excess": shift,
        "G4_defect_retention": retention,
        "G5_hard_null_superiority": nulls,
        "G6_family_and_shifted_drop": g6,
        "G7_ribbon_and_ribbon_excluded": subset_gate,
        "strict": strict,
        "soft": soft,
        "seed_directional_gates": seed_direction,
        "ribbon_subset_directional_gates": subset_direction,
        "positive_families_by_seed": positive_families,
        "positive_objects_by_seed": positive_objects,
        "no_object_shift_harm": no_shift_harm,
        "no_family_collapse": no_family_collapse,
        "macro": stats,
    }


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    run_root = Path(args.run_root).resolve(strict=True)
    payloads = _load_summaries(run_root)
    records: list[dict[str, Any]] = []
    audits: list[dict[str, bool]] = []
    for payload in payloads:
        normalized, audit = _normalize_payload(payload)
        records.extend(normalized)
        audits.append(audit)
    bootstrap_rows, lcbs, bootstrap_seed = _bootstrap(records)
    gates = _gates(records, audits, lcbs)
    if gates["strict"]:
        diagnostic = "PROMISING_DIAGNOSTIC"
    elif gates["soft"]:
        diagnostic = "SOFT_CONTINUE_ONE_FIXED_REPEAT"
    else:
        diagnostic = "NO_CONTINUE"

    manifest_hashes: dict[str, str] = {}
    manifest_coverage: set[tuple[str, int]] = set()
    for path in sorted(run_root.glob("chunks/*/run_manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("support_manifest_sha256") != SUPPORT_SHA256:
            raise RuntimeError("run-manifest support hash mismatch")
        seed = int(manifest["experiment_seed"])
        manifest_coverage.update((str(object_name), seed) for object_name in manifest["objects"])
        manifest_hashes[str(path.relative_to(run_root))] = hashlib.sha256(
            path.read_bytes(),
        ).hexdigest()
    expected_coverage = {(object_name, seed) for object_name in OBJECTS for seed in SEEDS}
    if manifest_coverage != expected_coverage:
        raise RuntimeError("run manifests do not cover every object/seed pair")

    object_rows = []
    for object_name in OBJECTS:
        for seed in SEEDS:
            subset = [
                item for item in records if item["object"] == object_name and item["seed"] == seed
            ]
            object_rows.append({"object": object_name, "seed": seed, **_statistics(subset)})
    summary = _json_safe(
        {
            "schema": "fsdd-lodf-stage-minus1-summary-v1",
            "protocol_id": PROTOCOL_ID,
            "objects": list(OBJECTS),
            "experiment_seeds": list(SEEDS),
            "support_manifest_sha256": SUPPORT_SHA256,
            "test_public_accessed": False,
            "test_private_accessed": False,
            "tta_used": False,
            "record_count": len(records),
            "per_object_seed": object_rows,
            "per_family_seed": {
                f"{family}/seed{seed}": _statistics(
                    [item for item in records if item["family"] == family and item["seed"] == seed],
                )
                for family in FAMILIES
                for seed in SEEDS
            },
            "macro": _statistics(records),
            "bootstrap": {
                "replicates": BOOTSTRAPS,
                "unit": "object->source->LODF/LOSF head->seed",
                "seed": bootstrap_seed,
                "lcb_2.5": lcbs,
                "hard_null_envelope_recomputed_per_replicate": True,
            },
            "gates": gates,
            "claim_verdict": "KILL_FOR_CLAIM",
            "diagnostic_verdict": diagnostic,
            "public_shadow_authorized": bool(gates["strict"]),
            "revision_authorized": bool(not gates["strict"] and gates["soft"]),
            "claim_boundary": "support-only synthetic diagnostic; never a public method claim",
            "source_manifest_sha256": manifest_hashes,
        },
    )
    (run_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_tsv(run_root / "per_object_seed.tsv", object_rows)
    _write_tsv(run_root / "bootstrap.tsv", bootstrap_rows)
    (run_root / "verdict.txt").write_text(
        f"claim={summary['claim_verdict']}\ndiagnostic={diagnostic}\npublic_shadow_authorized={str(summary['public_shadow_authorized']).lower()}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
