from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Dict, Final, Mapping, Sequence

from flow_tte.darc_gate2_provenance import (
    REGISTERED_MODEL_PROVENANCE,
    JsonValue,
    SeedProvenance,
    decode_json,
    file_sha256,
    sha256_bytes,
    sha256_json,
)
from flow_tte.darc_resources import P16_PROTOCOL_VERSION
from flow_tte.darc_synthetic import LINE_CUE_VERSION

if TYPE_CHECKING:
    from pathlib import Path

GATE2_METHOD_VERSION: Final = "darc-gate2-v1"
GATE2_COMPLETION_SCHEMA: Final = "darc-gate2-completion-v1"
GATE2_SOURCE_COUNT: Final = 16

_REQUIRED_ARTIFACTS: Final = ("source_rows.jsonl", "group_residual.json")
_DESIGN_PATH: Final = "skill_graph/analysis/2026-07-10_flowtte_darc_experiment_design.md"
_DESIGN_SHA256: Final = "26b736b10b32cc0ae9b1770f67d1e2aa861eb2b3281064d35e46643a5e9abf60"
_ADDENDUM_PATH: Final = (
    "skill_graph/analysis/2026-07-10_flowtte_darc_gate23_preregistration_addendum.md"
)
_ADDENDUM_SHA256: Final = "6576a547d1dba4a873b9ed347de636e4ad9ba7a16bfac0d615f265208b769a22"
_PREREG_PATH: Final = (
    "skill_graph/experiments/2026-07-10_flowtte_darc_resolution_correspondence/"
    "gate23_preregistration.json"
)
_PREREG_SHA256: Final = "e04a1d559744eccdb73365139168e0735ba6a4cfa6b81b858346e17d614de051"


# Python 3.8 dataclasses do not support slots=True.
@dataclass(frozen=True)
class CompletionExpectation:
    seed_root: Path
    selection_path: Path
    object_name: str
    seed: int
    smoke: bool
    source_count: int
    provenance: SeedProvenance

    def with_actual_selection_hash(self) -> CompletionExpectation:
        return replace(
            self,
            provenance=self.provenance.with_selection_sha256(file_sha256(self.selection_path)),
        )


def gate2_method_manifest() -> Dict[str, JsonValue]:
    return {
        "schema": "darc-gate2-method-v1",
        "method": GATE2_METHOD_VERSION,
        "registered_documents": [
            {"path": _DESIGN_PATH, "sha256": _DESIGN_SHA256},
            {"path": _ADDENDUM_PATH, "sha256": _ADDENDUM_SHA256},
            {"path": _PREREG_PATH, "sha256": _PREREG_SHA256},
        ],
        "resource": {
            "version": P16_PROTOCOL_VERSION,
            "seeds": [0, 1, 2],
            "objects": 15,
            "sources_per_object_seed": GATE2_SOURCE_COUNT,
            "expected_groups": 45,
            "expected_source_rows": 720,
        },
        "synthetic": {
            "version": LINE_CUE_VERSION,
            "signal_profiles": ["thin-w1-l32", "thin-w2-l48"],
            "diagnostic_profile": "broad-control-w16-l96",
            "component_connectivity": 8,
        },
        "feature_stream": {
            "crop_size": 512,
            "crop_stride": 384,
            "micro_input_size": 1024,
            "coarse_short_edge": 672,
            "patch_size": 16,
            "micro_layer": 7,
            "geometry_layer": 23,
            "dtype": "fp32-inference-fp16-cache-fp32-score",
        },
        "geometry": {
            "matcher": "mutual-nearest-neighbor",
            "transform": "orientation-preserving-4dof-similarity",
            "ransac_reprojection_threshold": 32.0,
            "minimum_inliers": 12,
            "minimum_inlier_ratio": 0.25,
            "scale_range": [0.8, 1.25],
            "maximum_median_error": 24.0,
        },
        "ladder": {
            "rungs": ["G0", "L0", "L1", "R1"],
            "local_neighborhood": "3x3",
            "top_k_supports": 5,
            "minimum_local_supports": 3,
            "fallback": "rung-independent-layer7-G0-residual-and-evidence",
        },
        "normal_residual": {
            "quantile": 0.999,
            "quantile_method": "higher",
            "maximum_l1_over_l0": 0.8,
        },
        "signal": {
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 20260710,
            "bootstrap_quantile": 0.025,
            "bootstrap_quantile_method": "linear",
            "require_ap_and_component_lower_bounds_strictly_positive": True,
        },
        "retention": {
            "minimum_r1_over_l1": 0.9,
            "require_l1_strictly_positive": True,
            "evidence_stage": "clean-subtracted-calibrated-micro-pre-confidence-pre-coarse",
        },
        "broad_control": {"metric": "R1-minus-L0-pAUROC-0.05", "diagnostic_only": True},
        "operational_freeze": {
            "signal_stage": "calibrated-micro-evidence-without-coarse-or-confidence-fusion",
            "component_threshold": {
                "reference": "fold-four-clean-calibration-maps",
                "quantile": 0.9999,
                "quantile_method": "higher",
                "comparison": "strict-greater-than",
            },
            "residual_population": (
                "every-emitted-high-crop-token-in-deterministic-crop-grid-order-"
                "including-overlap-duplicates"
            ),
            "token_support_population": "L0-L1-validity-intersection",
            "fallback": "rejected-or-fewer-than-3-alignments-use-common-G0",
            "g0_reference": "full-normal-LOO-token-population",
            "rung_reference": "nonfallback-normal-LOO-residuals",
            "query_condition_geometry": (
                "recompute-geometry-and-ranking-from-each-query-condition-detached-coarse-features"
            ),
            "ransac_seed": (
                "uint64-big-endian-first8(sha256(utf8(query-identity)\\0utf8(support-identity)))"
            ),
        },
        "model": REGISTERED_MODEL_PROVENANCE.to_manifest(),
    }


def gate2_method_hash() -> str:
    return sha256_json(gate2_method_manifest())


def json_document_sha256(payload: Mapping[str, JsonValue]) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    return sha256_bytes(encoded)


def valid_completion(expectation: CompletionExpectation) -> bool:
    complete_path = expectation.seed_root / "complete.json"
    if (
        expectation.smoke
        or expectation.source_count != GATE2_SOURCE_COUNT
        or expectation.provenance.method_sha256 != gate2_method_hash()
        or not complete_path.is_file()
        or not expectation.selection_path.is_file()
    ):
        return False
    try:
        payload = decode_json(complete_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    expected_identity = {
        "schema": GATE2_COMPLETION_SCHEMA,
        "object": expectation.object_name,
        "seed": expectation.seed,
        "smoke": False,
        "source_count": GATE2_SOURCE_COUNT,
        "method_sha256": expectation.provenance.method_sha256,
        "code_config_sha256": expectation.provenance.code_config_sha256,
        "model": expectation.provenance.model.to_manifest(),
        "dataset_inventory_sha256": expectation.provenance.dataset_inventory_sha256,
        "split_inventory_sha256": expectation.provenance.split_inventory_sha256,
        "selection_sha256": expectation.provenance.selection_sha256,
        "provenance": expectation.provenance.to_manifest(),
        "provenance_sha256": expectation.provenance.digest(),
    }
    if not isinstance(payload, dict) or any(
        payload.get(key) != value for key, value in expected_identity.items()
    ):
        return False
    artifacts = payload.get("artifact_sha256")
    if not isinstance(artifacts, dict) or set(artifacts) != set(_REQUIRED_ARTIFACTS):
        return False
    selection_valid = (
        file_sha256(expectation.selection_path) == expectation.provenance.selection_sha256
    )
    artifacts_valid = all(
        (expectation.seed_root / name).is_file()
        and artifacts.get(name) == file_sha256(expectation.seed_root / name)
        for name in _REQUIRED_ARTIFACTS
    )
    return (
        selection_valid
        and artifacts_valid
        and _has_exact_source_rows(expectation.seed_root / "source_rows.jsonl")
        and _is_json_object(expectation.seed_root / "group_residual.json")
    )


def write_completion(expectation: CompletionExpectation) -> None:
    artifacts: Dict[str, JsonValue] = {
        name: file_sha256(expectation.seed_root / name) for name in _REQUIRED_ARTIFACTS
    }
    provenance = expectation.provenance
    write_json(
        expectation.seed_root / "complete.json",
        {
            "schema": GATE2_COMPLETION_SCHEMA,
            "object": expectation.object_name,
            "seed": expectation.seed,
            "smoke": expectation.smoke,
            "source_count": expectation.source_count,
            "method_sha256": provenance.method_sha256,
            "code_config_sha256": provenance.code_config_sha256,
            "model": provenance.model.to_manifest(),
            "dataset_inventory_sha256": provenance.dataset_inventory_sha256,
            "split_inventory_sha256": provenance.split_inventory_sha256,
            "selection_sha256": provenance.selection_sha256,
            "provenance": provenance.to_manifest(),
            "provenance_sha256": provenance.digest(),
            "artifact_sha256": artifacts,
            "scratch_deleted": True,
            "raw_maps_deleted_after_compaction": True,
        },
    )


def write_json(path: Path, payload: Mapping[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, JsonValue]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _has_exact_source_rows(path: Path) -> bool:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = [decode_json(line) for line in lines if line]
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return (
        len(lines) == GATE2_SOURCE_COUNT
        and len(rows) == GATE2_SOURCE_COUNT
        and all(isinstance(row, dict) for row in rows)
    )


def _is_json_object(path: Path) -> bool:
    try:
        payload = decode_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict)
