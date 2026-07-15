from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from flow_tte.darc_gate1_artifacts import (
    CompletionExpectation,
    SeedProvenance,
    valid_completion,
    write_completion,
    write_json,
    write_jsonl,
)
from flow_tte.darc_gate1_provenance import (
    MODEL_CONFIG_SHA256,
    MODEL_REVISION,
    MODEL_WEIGHTS_SHA256,
)


def _expectation(root: Path) -> CompletionExpectation:
    selection = root / "selections" / "bottle" / "seed=0.json"
    write_json(selection, {"split_inventory_sha256": "split", "support_inventory": []})
    seed_root = root / "objects" / "bottle" / "seed=0"
    write_json(seed_root / "source_metrics.json", {"rows": []})
    write_json(seed_root / "metrics.json", {"source_count": 16})
    write_json(seed_root / "bootstrap_inputs.json", {"rows": []})
    write_jsonl(seed_root / "samples.jsonl", [])
    provenance = SeedProvenance(
        method_sha256="method",
        code_config_sha256="code",
        dataset_inventory_sha256="dataset",
        split_inventory_sha256="split",
        selection_sha256=selection.read_bytes().hex()[:64],
    )
    return CompletionExpectation(
        seed_root=seed_root,
        selection_path=selection,
        object_name="bottle",
        seed=0,
        smoke=False,
        source_count=16,
        provenance=provenance,
    )


def test_completion_rejects_stale_selection_after_completion(tmp_path: Path) -> None:
    # Given
    expectation = _expectation(tmp_path)
    expectation = expectation.with_actual_selection_hash()
    write_completion(expectation)
    write_json(expectation.selection_path, {"split_inventory_sha256": "stale"})

    # When / Then
    assert not valid_completion(expectation)


def test_completion_rejects_missing_or_modified_artifact(tmp_path: Path) -> None:
    # Given
    expectation = _expectation(tmp_path).with_actual_selection_hash()
    write_completion(expectation)
    (expectation.seed_root / "samples.jsonl").unlink()

    # When / Then
    assert not valid_completion(expectation)


def test_completion_accepts_exact_selection_provenance_and_artifact_hashes(
    tmp_path: Path,
) -> None:
    # Given
    expectation = _expectation(tmp_path).with_actual_selection_hash()

    # When
    write_completion(expectation)

    # Then
    assert valid_completion(expectation)


def test_registered_hf_snapshot_and_digests_are_immutable() -> None:
    # Given / When / Then
    assert MODEL_REVISION == "c807c9eeea853df70aec4069e6f56b28ddc82acc"
    assert MODEL_CONFIG_SHA256 == "35770e98e425c9383534bcbfa7f3b7a3cb1d943f6d26a37d7d19fea0735eab57"
    assert MODEL_WEIGHTS_SHA256 == (
        "3e1d4d18b9bfa9f28fad8e9de6a783f1313532d3460efa4cd0b12521d81d1a4d"
    )
