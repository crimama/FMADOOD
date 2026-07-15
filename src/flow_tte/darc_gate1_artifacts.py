from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Dict, Final, List, Mapping, Sequence

from flow_tte.darc_gate1 import GATE1_METHOD_VERSION, Gate1Thresholds, SourceMetric
from flow_tte.darc_gate1_provenance import (
    REGISTERED_MODEL_PROVENANCE,
    JsonValue,
    SeedProvenance,
    file_sha256,
    sha256_bytes,
    sha256_json,
)
from flow_tte.darc_resources import P16_PROTOCOL_VERSION
from flow_tte.darc_synthetic import LINE_CUE_VERSION

if TYPE_CHECKING:
    from pathlib import Path

_SEED_ARTIFACTS: Final = (
    "source_metrics.json",
    "metrics.json",
    "bootstrap_inputs.json",
    "samples.jsonl",
)


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
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


def gate1_method_manifest() -> Dict[str, JsonValue]:
    thresholds = Gate1Thresholds()
    return {
        "method": GATE1_METHOD_VERSION,
        "resource": P16_PROTOCOL_VERSION,
        "synthetic": LINE_CUE_VERSION,
        "crop_size_stride": [512, 384],
        "input_sizes": [512, 1024, 672],
        "layers": [7, 23],
        "top_k": 5,
        "dtype": "fp32-inference-fp16-cache-fp32-score",
        "model": REGISTERED_MODEL_PROVENANCE.to_manifest(),
        "thresholds": {
            "ap_absolute_gain": thresholds.ap_absolute_gain,
            "ap_relative_gain": thresholds.ap_relative_gain,
            "component_gain": thresholds.component_gain,
            "maximum_control_pauroc_loss": thresholds.maximum_control_pauroc_loss,
            "normal_quantile": thresholds.normal_quantile,
            "bootstrap_replicates": thresholds.bootstrap_replicates,
            "bootstrap_seed": thresholds.bootstrap_seed,
            "maximum_stability_median_fpr": thresholds.maximum_stability_median_fpr,
            "maximum_stability_fpr": thresholds.maximum_stability_fpr,
            "maximum_stability_iqr_ratio": thresholds.maximum_stability_iqr_ratio,
        },
    }


def gate1_method_hash() -> str:
    return sha256_json(gate1_method_manifest())


def json_document_sha256(payload: Mapping[str, JsonValue]) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    return sha256_bytes(encoded)


def source_metrics_manifest(
    metrics: Sequence[SourceMetric],
    provenance_sha256: str,
) -> Dict[str, JsonValue]:
    rows: List[JsonValue] = [
        {
            "object": item.object_name,
            "source": item.source_id,
            "seed": item.seed,
            "fold": item.fold_index,
            "low": item.low.to_manifest(),
            "bilinear_null": item.bilinear_null.to_manifest(),
            "high": item.high.to_manifest(),
        }
        for item in metrics
    ]
    return {
        "method_hash": gate1_method_hash(),
        "provenance_sha256": provenance_sha256,
        "rows": rows,
    }


def bootstrap_inputs_manifest(
    metrics: Sequence[SourceMetric],
    provenance_sha256: str,
) -> Dict[str, JsonValue]:
    thresholds = Gate1Thresholds()
    rows: List[JsonValue] = [
        {
            "object": item.object_name,
            "source": item.source_id,
            "seed": item.seed,
            "fold": item.fold_index,
            "ap_low": item.low.ap,
            "ap_delta_high_minus_low": item.high.ap - item.low.ap,
            "component_delta_high_minus_low": (
                item.high.component_recall - item.low.component_recall
            ),
            "control_pauroc_delta_high_minus_low": (
                item.high.p_auroc_005 - item.low.p_auroc_005
            ),
            "ap_delta_high_minus_bilinear_null": item.high.ap - item.bilinear_null.ap,
        }
        for item in metrics
    ]
    return {
        "schema": "darc-stratified-source-bootstrap-v1",
        "algorithm": "resample sources with replacement within each object-seed stratum",
        "seed": thresholds.bootstrap_seed,
        "replicates": thresholds.bootstrap_replicates,
        "lower_quantile": 0.025,
        "provenance_sha256": provenance_sha256,
        "rows": rows,
    }


def valid_completion(expectation: CompletionExpectation) -> bool:
    complete_path = expectation.seed_root / "complete.json"
    if not complete_path.is_file() or not expectation.selection_path.is_file():
        return False
    try:
        payload = json.loads(complete_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected_identity = {
        "schema": "darc-gate1-completion-v2",
        "object": expectation.object_name,
        "seed": expectation.seed,
        "smoke": expectation.smoke,
        "source_count": expectation.source_count,
        "provenance": expectation.provenance.to_manifest(),
        "provenance_sha256": expectation.provenance.digest(),
        "selection_sha256": expectation.provenance.selection_sha256,
    }
    identity_valid = isinstance(payload, dict) and all(
        payload.get(key) == value for key, value in expected_identity.items()
    )
    if not identity_valid:
        return False
    artifacts = payload.get("artifact_sha256")
    selection_valid = (
        file_sha256(expectation.selection_path) == expectation.provenance.selection_sha256
    )
    if not isinstance(artifacts, dict) or set(artifacts) != set(_SEED_ARTIFACTS):
        return False
    artifact_hashes_valid = all(
        (expectation.seed_root / name).is_file()
        and artifacts.get(name) == file_sha256(expectation.seed_root / name)
        for name in _SEED_ARTIFACTS
    )
    return selection_valid and artifact_hashes_valid


def write_completion(expectation: CompletionExpectation) -> None:
    artifacts: Dict[str, JsonValue] = {
        name: file_sha256(expectation.seed_root / name) for name in _SEED_ARTIFACTS
    }
    write_json(
        expectation.seed_root / "complete.json",
        {
            "schema": "darc-gate1-completion-v2",
            "object": expectation.object_name,
            "seed": expectation.seed,
            "smoke": expectation.smoke,
            "source_count": expectation.source_count,
            "provenance": expectation.provenance.to_manifest(),
            "provenance_sha256": expectation.provenance.digest(),
            "selection_sha256": expectation.provenance.selection_sha256,
            "artifact_sha256": artifacts,
            "scratch_deleted": True,
            "raw_maps_deleted_after_evaluation": True,
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


def write_jsonl(path: Path, rows: Sequence[JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
