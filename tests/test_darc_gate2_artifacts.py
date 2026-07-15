from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from flow_tte.darc_gate2_artifacts import (
    CompletionExpectation,
    gate2_method_hash,
    gate2_method_manifest,
    valid_completion,
    write_completion,
    write_json,
    write_jsonl,
)
from flow_tte.darc_gate2_provenance import (
    CODE_BUNDLE_FILES,
    MODEL_CONFIG_SHA256,
    MODEL_REVISION,
    MODEL_WEIGHTS_SHA256,
    SeedProvenance,
    code_bundle_hash,
    dataset_inventory_hash,
    file_sha256,
    selection_manifest,
    sha256_json,
)
from flow_tte.darc_resources import build_p16_split


def _expectation(
    root: Path,
    *,
    source_count: int = 16,
    smoke: bool = False,
) -> CompletionExpectation:
    selection = root / "selections" / "bottle" / "seed=0.json"
    write_json(selection, {"schema": "selection", "split_inventory_sha256": "split"})
    seed_root = root / "objects" / "bottle" / "seed=0"
    write_jsonl(
        seed_root / "source_rows.jsonl",
        [{"source": str(index)} for index in range(source_count)],
    )
    write_json(seed_root / "group_residual.json", {"source_count": source_count})
    provenance = SeedProvenance(
        method_sha256=gate2_method_hash(),
        code_config_sha256="code",
        dataset_inventory_sha256="dataset",
        split_inventory_sha256="split",
        selection_sha256="pending",
    )
    return CompletionExpectation(
        seed_root=seed_root,
        selection_path=selection,
        object_name="bottle",
        seed=0,
        smoke=smoke,
        source_count=source_count,
        provenance=provenance,
    ).with_actual_selection_hash()


def test_method_manifest_pins_registered_documents_and_thresholds() -> None:
    # Given / When
    manifest = gate2_method_manifest()

    # Then
    documents = manifest["registered_documents"]
    assert documents == [
        {
            "path": "skill_graph/analysis/2026-07-10_flowtte_darc_experiment_design.md",
            "sha256": "26b736b10b32cc0ae9b1770f67d1e2aa861eb2b3281064d35e46643a5e9abf60",
        },
        {
            "path": (
                "skill_graph/analysis/2026-07-10_flowtte_darc_gate23_preregistration_addendum.md"
            ),
            "sha256": "6576a547d1dba4a873b9ed347de636e4ad9ba7a16bfac0d615f265208b769a22",
        },
        {
            "path": (
                "skill_graph/experiments/"
                "2026-07-10_flowtte_darc_resolution_correspondence/"
                "gate23_preregistration.json"
            ),
            "sha256": "e04a1d559744eccdb73365139168e0735ba6a4cfa6b81b858346e17d614de051",
        },
    ]
    assert manifest["normal_residual"] == {
        "quantile": 0.999,
        "quantile_method": "higher",
        "maximum_l1_over_l0": 0.8,
    }
    assert manifest["operational_freeze"] == {
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
    }
    assert gate2_method_hash() == "3b93670c19f7a45b8623ce4f2f4bcd449e343d8f844e838062dbe25ed1f716fd"
    assert gate2_method_hash() == sha256_json(gate2_method_manifest())


def test_code_bundle_hash_changes_with_relevant_file_or_manifest(tmp_path: Path) -> None:
    # Given
    for relative in CODE_BUNDLE_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    method = gate2_method_manifest()
    original = code_bundle_hash(tmp_path, method)

    # When
    changed = tmp_path / CODE_BUNDLE_FILES[-1]
    changed.write_text("changed", encoding="utf-8")

    # Then
    assert code_bundle_hash(tmp_path, method) != original
    assert code_bundle_hash(tmp_path, {**method, "method": "changed"}) != original


def test_code_bundle_seals_extracted_calibration_logic() -> None:
    # Given: Gate 2 calibration is implemented in its own runtime module.
    calibration_path = "src/flow_tte/darc_gate2_calibration.py"

    # When/Then: provenance must hash that executable scientific logic.
    assert calibration_path in CODE_BUNDLE_FILES


def test_inventory_and_selection_provenance_are_order_independent(tmp_path: Path) -> None:
    # Given
    data_root = tmp_path / "data"
    paths = []
    for index in range(16):
        path = data_root / "bottle" / "train" / "good" / f"{index:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes([index]))
        paths.append(path)
    split = build_p16_split(paths, seed=0)

    # When
    digest = dataset_inventory_hash(data_root, tuple(reversed(paths)))
    manifest = selection_manifest(split, data_root, digest)

    # Then
    assert digest == dataset_inventory_hash(data_root, paths)
    assert manifest["schema"] == "darc-gate2-selection-v1"
    assert manifest["dataset_inventory_sha256"] == digest
    support_inventory = manifest["support_inventory"]
    assert isinstance(support_inventory, list)
    assert len(support_inventory) == 16
    assert manifest["split_inventory_sha256"] == sha256_json(
        {key: value for key, value in manifest.items() if key != "split_inventory_sha256"},
    )


def test_atomic_json_writers_replace_without_temporary_residue(tmp_path: Path) -> None:
    # Given
    json_path = tmp_path / "nested" / "value.json"
    jsonl_path = tmp_path / "nested" / "rows.jsonl"
    write_json(json_path, {"old": True})

    # When
    write_json(json_path, {"new": 1})
    write_jsonl(jsonl_path, [{"row": 1}, {"row": 2}])

    # Then
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"new": 1}
    assert jsonl_path.read_text(encoding="utf-8").splitlines() == [
        '{"row": 1}',
        '{"row": 2}',
    ]
    assert not json_path.with_suffix(".json.tmp").exists()
    assert not jsonl_path.with_suffix(".jsonl.tmp").exists()


def test_completion_accepts_exact_full_provenance_and_artifacts(tmp_path: Path) -> None:
    # Given
    expectation = _expectation(tmp_path)

    # When
    write_completion(expectation)

    # Then
    assert valid_completion(expectation)
    payload = json.loads((expectation.seed_root / "complete.json").read_text(encoding="utf-8"))
    assert payload["method_sha256"] == expectation.provenance.method_sha256
    assert payload["code_config_sha256"] == expectation.provenance.code_config_sha256
    assert payload["model"] == expectation.provenance.model.to_manifest()
    assert payload["dataset_inventory_sha256"] == expectation.provenance.dataset_inventory_sha256
    assert payload["split_inventory_sha256"] == expectation.provenance.split_inventory_sha256
    assert payload["selection_sha256"] == file_sha256(expectation.selection_path)
    assert payload["provenance_sha256"] == expectation.provenance.digest()


@pytest.mark.parametrize("artifact", ["source_rows.jsonl", "group_residual.json"])
def test_completion_rejects_tampered_scientific_artifact(tmp_path: Path, artifact: str) -> None:
    # Given
    expectation = _expectation(tmp_path)
    write_completion(expectation)

    # When
    (expectation.seed_root / artifact).write_text("{}\n", encoding="utf-8")

    # Then
    assert not valid_completion(expectation)


def test_completion_rejects_stale_selection(tmp_path: Path) -> None:
    # Given
    expectation = _expectation(tmp_path)
    write_completion(expectation)

    # When
    write_json(expectation.selection_path, {"schema": "stale"})

    # Then
    assert not valid_completion(expectation)


@pytest.mark.parametrize("field", ["method_sha256", "code_config_sha256"])
def test_completion_rejects_method_or_code_mismatch(tmp_path: Path, field: str) -> None:
    # Given
    expectation = _expectation(tmp_path)
    write_completion(expectation)
    stale_provenance = replace(expectation.provenance, **{field: "stale"})

    # When / Then
    assert not valid_completion(replace(expectation, provenance=stale_provenance))


@pytest.mark.parametrize(("source_count", "smoke"), [(15, False), (16, True), (17, False)])
def test_completion_rejects_smoke_or_non_exact_source_population(
    tmp_path: Path,
    source_count: int,
    smoke: bool,
) -> None:
    # Given
    expectation = _expectation(tmp_path, source_count=source_count, smoke=smoke)

    # When
    write_completion(expectation)

    # Then
    assert not valid_completion(expectation)


def test_registered_hf_snapshot_and_digests_are_immutable() -> None:
    # Given / When / Then
    assert MODEL_REVISION == "c807c9eeea853df70aec4069e6f56b28ddc82acc"
    assert MODEL_CONFIG_SHA256 == "35770e98e425c9383534bcbfa7f3b7a3cb1d943f6d26a37d7d19fea0735eab57"
    assert MODEL_WEIGHTS_SHA256 == (
        "3e1d4d18b9bfa9f28fad8e9de6a783f1313532d3460efa4cd0b12521d81d1a4d"
    )
