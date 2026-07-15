from __future__ import annotations

# pyright: reportMissingImports=false
import json
import shutil
from typing import TYPE_CHECKING, Dict, List, NamedTuple

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from flow_tte.darc_gate2_aggregate import (
    load_and_decide_gate2,
    load_full_gate2_inputs,
)
from flow_tte.darc_gate2_artifacts import (
    CompletionExpectation,
    gate2_method_hash,
    write_completion,
    write_json,
    write_jsonl,
)
from flow_tte.darc_gate2_metrics import (
    AD1_OBJECTS,
    Gate2Config,
    GroupResidual,
    InvalidGateInput,
    SourceAudit,
    SourceMetric,
    decide_gate2,
)
from flow_tte.darc_gate2_provenance import (
    JsonValue,
    SeedProvenance,
    file_sha256,
    sha256_bytes,
    sha256_json,
)
from flow_tte.darc_resources import P16_PROTOCOL_VERSION


class _CellConfig(NamedTuple):
    object_name: str
    seed: int
    method_sha256: str = ""
    code_sha256: str = "c" * 64


def _digest(text: str) -> str:
    return sha256_bytes(text.encode())


def _selection(object_name: str, seed: int) -> Dict[str, JsonValue]:
    sources = [f"{object_name}/train/good/{index:03}.png" for index in range(16)]
    folds: List[JsonValue] = []
    for fold_index in range(4):
        start = fold_index * 4
        folds.append(
            {
                "fold_index": fold_index,
                "memory_paths": sources[:start] + sources[start + 4 :],
                "calibration_paths": sources[start : start + 4],
            },
        )
    base: Dict[str, JsonValue] = {
        "schema": "darc-gate2-selection-v1",
        "version": P16_PROTOCOL_VERSION,
        "seed": seed,
        "source_pool_count": 16,
        "dataset_inventory_sha256": _digest(f"dataset/{object_name}"),
        "support_inventory": [
            {"path": source, "size": index + 1, "sha256": _digest(source)}
            for index, source in enumerate(sources)
        ],
        "folds": folds,
    }
    return {**base, "split_inventory_sha256": sha256_json(base)}


def _write_cell(
    root: Path,
    config: _CellConfig,
) -> None:
    object_name, seed = config.object_name, config.seed
    selection = _selection(object_name, seed)
    selection_path = root / "selections" / object_name / f"seed={seed}.json"
    write_json(selection_path, selection)
    split_sha256 = selection["split_inventory_sha256"]
    dataset_sha256 = selection["dataset_inventory_sha256"]
    assert isinstance(split_sha256, str)
    assert isinstance(dataset_sha256, str)
    provenance = SeedProvenance(
        method_sha256=config.method_sha256 or gate2_method_hash(),
        code_config_sha256=config.code_sha256,
        dataset_inventory_sha256=dataset_sha256,
        split_inventory_sha256=split_sha256,
        selection_sha256=file_sha256(selection_path),
    )
    provenance_sha256 = provenance.digest()
    source_ids = tuple(f"{object_name}/train/good/{index:03}.png" for index in range(16))
    rows: List[Dict[str, JsonValue]] = []
    for index, source_id in enumerate(source_ids):
        audit = SourceAudit(*(_digest(f"{name}/{source_id}") for name in ("p", "s", "f", "m")))
        metric = SourceMetric(
            object_name=object_name,
            seed=seed,
            fold_index=index // 4,
            source_id=source_id,
            d_ap=0.1,
            d_component_recall=0.1,
            l1_responses=(1.0, 1.0),
            r1_responses=(0.9, 0.9),
            broad_pauroc_delta=0.01,
            l0_audit=audit,
            l1_audit=audit,
            r1_audit=audit,
        )
        rows.append(
            {
                **metric.to_manifest(),
                "evaluation": {
                    "thresholds": {"l0": 1.0, "l1": 1.0},
                    "absolute_ap": {"l0": 0.5, "l1": 0.6},
                    "absolute_component_recall": {"l0": 0.5, "l1": 0.6},
                    "l1_profiles": [
                        {"clean_mean": 0.1, "cue_mean": 1.1, "response": 1.0},
                        {"clean_mean": 0.2, "cue_mean": 1.2, "response": 1.0},
                    ],
                    "r1_profiles": [
                        {"clean_mean": 0.1, "cue_mean": 1.0, "response": 0.9},
                        {"clean_mean": 0.2, "cue_mean": 1.1, "response": 0.9},
                    ],
                    "broad_pauroc_005": {"l0": 0.5, "r1": 0.51, "delta_r1_l0": 0.01},
                },
                "provenance_sha256": provenance_sha256,
            },
        )
    population_sha256 = _digest(f"tokens/{object_name}/{seed}")
    group = GroupResidual(
        object_name=object_name,
        seed=seed,
        source_ids=source_ids,
        fold_indices=tuple(index // 4 for index in range(16)),
        l0_p999=1.0,
        l1_p999=0.8,
        l0_population_sha256=population_sha256,
        l1_population_sha256=population_sha256,
        l0_residual_sha256=_digest(f"l0/{object_name}/{seed}"),
        l1_residual_sha256=_digest(f"l1/{object_name}/{seed}"),
    )
    seed_root = root / "objects" / object_name / f"seed={seed}"
    write_jsonl(seed_root / "source_rows.jsonl", rows)
    write_json(
        seed_root / "group_residual.json",
        {**group.to_manifest(), "provenance_sha256": provenance_sha256},
    )
    write_completion(
        CompletionExpectation(
            seed_root=seed_root,
            selection_path=selection_path,
            object_name=object_name,
            seed=seed,
            smoke=False,
            source_count=16,
            provenance=provenance,
        ),
    )


@pytest.fixture(scope="module")
def full_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("gate2-aggregate")
    for object_name in AD1_OBJECTS:
        for seed in (0, 1, 2):
            _write_cell(root, _CellConfig(object_name, seed))
    return root


def _copy_root(full_root: Path, tmp_path: Path) -> Path:
    return shutil.copytree(full_root, tmp_path / "run")


def _refresh_hash(root: Path, relative: str) -> None:
    artifact = root / relative
    complete_path = artifact.parent / "complete.json"
    payload = json.loads(complete_path.read_text(encoding="utf-8"))
    payload["artifact_sha256"][artifact.name] = file_sha256(artifact)
    write_json(complete_path, payload)


def test_loader_accepts_exact_45_by_16_population(full_root: Path) -> None:
    # Given / When
    aggregate = load_full_gate2_inputs(full_root)

    # Then
    assert len(aggregate.sources) == 720
    assert len(aggregate.groups) == 45
    assert aggregate.method_sha256 == gate2_method_hash()
    assert aggregate.code_config_sha256 == "c" * 64
    first = aggregate.sources[0]
    assert first.l0_audit.population_sha256 == _digest(f"p/{first.source_id}")
    assert first.l0_audit.support_sha256 == _digest(f"s/{first.source_id}")
    assert first.l0_audit.fallback_sha256 == _digest(f"f/{first.source_id}")
    assert first.l0_audit.mask_sha256 == _digest(f"m/{first.source_id}")


def test_loader_rejects_tampered_artifact_hash(full_root: Path, tmp_path: Path) -> None:
    # Given
    root = _copy_root(full_root, tmp_path)
    artifact = root / "objects" / "bottle" / "seed=0" / "group_residual.json"

    # When
    artifact.write_text("{}\n", encoding="utf-8")

    # Then
    with pytest.raises(InvalidGateInput, match="checksum"):
        load_full_gate2_inputs(root)


@pytest.mark.parametrize("case", ["method", "code", "embedded-provenance"])
def test_loader_rejects_mixed_method_code_or_provenance(
    full_root: Path,
    tmp_path: Path,
    case: str,
) -> None:
    # Given
    root = _copy_root(full_root, tmp_path)
    if case == "method":
        _write_cell(root, _CellConfig("bottle", 0, method_sha256="f" * 64))
    elif case == "code":
        _write_cell(root, _CellConfig("bottle", 0, code_sha256="d" * 64))
    else:
        relative = "objects/bottle/seed=0/source_rows.jsonl"
        path = root / relative
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["provenance_sha256"] = "0" * 64
        write_jsonl(path, rows)
        _refresh_hash(root, relative)

    # When / Then
    with pytest.raises(InvalidGateInput):
        load_full_gate2_inputs(root)


@pytest.mark.parametrize("case", ["wrong-type", "nonfinite", "extra-row", "reordered"])
def test_loader_rejects_malformed_or_non_exact_source_rows(
    full_root: Path,
    tmp_path: Path,
    case: str,
) -> None:
    # Given
    root = _copy_root(full_root, tmp_path)
    relative = "objects/bottle/seed=0/source_rows.jsonl"
    path = root / relative
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if case == "wrong-type":
        rows[0]["d_ap"] = "0.1"
    elif case == "nonfinite":
        rows[0]["d_ap"] = float("nan")
    elif case == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    else:
        rows.append(rows[0])
    write_jsonl(path, rows)
    _refresh_hash(root, relative)

    # When / Then
    with pytest.raises(InvalidGateInput):
        load_full_gate2_inputs(root)


@pytest.mark.parametrize("case", ["missing", "extra", "duplicate-identity"])
def test_loader_rejects_missing_or_extra_completion(
    full_root: Path,
    tmp_path: Path,
    case: str,
) -> None:
    # Given
    root = _copy_root(full_root, tmp_path)
    if case == "missing":
        (root / "objects" / "bottle" / "seed=0" / "complete.json").unlink()
    elif case == "extra":
        source = root / "objects" / "bottle" / "seed=0"
        shutil.copytree(source, root / "objects" / "unknown" / "seed=0")
    else:
        source = root / "objects" / "bottle" / "seed=0"
        shutil.copytree(source, root / "objects" / "bottle" / "seed=00")

    # When / Then
    with pytest.raises(InvalidGateInput, match="45 object-seed"):
        load_full_gate2_inputs(root)


def test_aggregate_decision_exactly_reproduces_direct_decision(full_root: Path) -> None:
    # Given
    config = Gate2Config(bootstrap_replicates=64)
    aggregate = load_full_gate2_inputs(full_root)
    expected = decide_gate2(aggregate.sources, aggregate.groups, config)

    # When
    result = load_and_decide_gate2(full_root, config)
    manifest = result.to_manifest()

    # Then
    assert result.decision == expected
    assert manifest["method_sha256"] == gate2_method_hash()
    assert manifest["code_config_sha256"] == "c" * 64
    assert manifest["model"] == aggregate.model.to_manifest()
    assert manifest["passed"] is True
