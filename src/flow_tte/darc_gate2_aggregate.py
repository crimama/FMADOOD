"""Checksum-strict loader for the complete DARC Gate 2 population."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, FrozenSet, List, NamedTuple, Optional, Set, Tuple

from flow_tte.darc_gate2_aggregate_types import (
    AggregateInput,
    AggregateResult,
    SeedIdentity,
    boolean,
    exact_object,
    fail,
    integer,
    json_list,
    parse_group,
    parse_provenance,
    parse_source,
    read_json,
    read_jsonl,
    sha256,
    string,
    strings,
)
from flow_tte.darc_gate2_artifacts import (
    GATE2_COMPLETION_SCHEMA,
    GATE2_SOURCE_COUNT,
    gate2_method_hash,
)
from flow_tte.darc_gate2_metrics import (
    AD1_OBJECTS,
    Gate2Config,
    GroupResidual,
    SourceMetric,
    decide_gate2,
)
from flow_tte.darc_gate2_provenance import (
    REGISTERED_MODEL_PROVENANCE,
    JsonValue,
    ModelProvenance,
    SeedProvenance,
    file_sha256,
    sha256_json,
)
from flow_tte.darc_resources import P16_PROTOCOL_VERSION

if TYPE_CHECKING:
    from pathlib import Path

__all__ = (
    "AggregateInput",
    "AggregateResult",
    "decide_gate2",
    "load_and_decide_gate2",
    "load_full_gate2_inputs",
)

_COMPLETION_KEYS: Final[FrozenSet[str]] = frozenset(
    {
        "schema",
        "object",
        "seed",
        "smoke",
        "source_count",
        "method_sha256",
        "code_config_sha256",
        "model",
        "dataset_inventory_sha256",
        "split_inventory_sha256",
        "selection_sha256",
        "provenance",
        "provenance_sha256",
        "artifact_sha256",
        "scratch_deleted",
        "raw_maps_deleted_after_compaction",
    },
)
_SELECTION_KEYS: Final[FrozenSet[str]] = frozenset(
    {
        "schema",
        "version",
        "seed",
        "source_pool_count",
        "dataset_inventory_sha256",
        "support_inventory",
        "folds",
        "split_inventory_sha256",
    },
)
_ARTIFACT_KEYS: Final[FrozenSet[str]] = frozenset(
    {"source_rows.jsonl", "group_residual.json"},
)


class _SelectionPopulation(NamedTuple):
    pairs: Tuple[Tuple[str, int], ...]


class _LoadedCell(NamedTuple):
    sources: Tuple[SourceMetric, ...]
    group: GroupResidual
    provenance: SeedProvenance


class _SelectionExpectation(NamedTuple):
    identity: SeedIdentity
    provenance: SeedProvenance
    path: Path


def load_full_gate2_inputs(root: Path) -> AggregateInput:
    """Load exactly 15 objects by three seeds after independent checksum validation."""
    expected = {(object_name, seed) for object_name in AD1_OBJECTS for seed in (0, 1, 2)}
    completion_paths = tuple(root.glob("objects/*/seed=*/complete.json"))
    identities = tuple(_completion_identity(path) for path in completion_paths)
    actual = set(identities)
    if len(completion_paths) != 45 or len(actual) != 45 or actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        prefix = "aggregate requires exactly 45 object-seed completions"
        fail(f"{prefix}; missing={missing} extra={extra}")
    sources: List[SourceMetric] = []
    groups: List[GroupResidual] = []
    methods: Set[str] = set()
    code_bundles: Set[str] = set()
    models: Set[ModelProvenance] = set()
    for object_name in AD1_OBJECTS:
        for seed in (0, 1, 2):
            cell = _load_cell(root, SeedIdentity(object_name, seed))
            sources.extend(cell.sources)
            groups.append(cell.group)
            methods.add(cell.provenance.method_sha256)
            code_bundles.add(cell.provenance.code_config_sha256)
            models.add(cell.provenance.model)
    if methods != {gate2_method_hash()} or len(code_bundles) != 1 or len(models) != 1:
        fail("method, code/config bundle, or model differs across completed cells")
    aggregate = AggregateInput(
        sources=tuple(sources),
        groups=tuple(groups),
        method_sha256=methods.pop(),
        code_config_sha256=code_bundles.pop(),
        model=models.pop(),
    )
    decide_gate2(aggregate.sources, aggregate.groups, Gate2Config(bootstrap_replicates=1))
    return aggregate


def load_and_decide_gate2(
    root: Path,
    config: Optional[Gate2Config] = None,
) -> AggregateResult:
    aggregate = load_full_gate2_inputs(root)
    decision = decide_gate2(aggregate.sources, aggregate.groups, config)
    return AggregateResult(aggregate=aggregate, decision=decision)


def _completion_identity(path: Path) -> Tuple[str, int]:
    parts = path.parent.name.split("=", 1)
    if len(parts) != 2 or parts[0] != "seed":
        fail(f"invalid completion path: {path}")
    try:
        seed = int(parts[1])
    except ValueError:
        fail(f"invalid completion seed path: {path}")
    return path.parents[1].name, seed


def _load_cell(root: Path, identity: SeedIdentity) -> _LoadedCell:
    seed_root = root / "objects" / identity.object_name / f"seed={identity.seed}"
    complete_path = seed_root / "complete.json"
    completion = exact_object(read_json(complete_path), complete_path, _COMPLETION_KEYS)
    provenance = _completion_provenance(completion, complete_path, identity)
    selection_path = root / "selections" / identity.object_name / f"seed={identity.seed}.json"
    if not selection_path.is_file() or file_sha256(selection_path) != provenance.selection_sha256:
        fail(f"selection checksum differs from completion: {selection_path}")
    selection = _parse_selection(
        read_json(selection_path),
        _SelectionExpectation(identity, provenance, selection_path),
    )
    artifacts = exact_object(completion["artifact_sha256"], complete_path, _ARTIFACT_KEYS)
    for name in _ARTIFACT_KEYS:
        artifact_path = seed_root / name
        if not artifact_path.is_file() or file_sha256(artifact_path) != sha256(
            artifacts,
            name,
            complete_path,
        ):
            fail(f"artifact checksum differs from completion: {artifact_path}")
    rows = read_jsonl(seed_root / "source_rows.jsonl")
    if len(rows) != GATE2_SOURCE_COUNT:
        fail(f"source artifact must contain exactly 16 rows: {seed_root}")
    provenance_sha256 = provenance.digest()
    parsed = tuple(parse_source(row, identity, provenance_sha256) for row in rows)
    actual_pairs = tuple((row.source_id, row.fold_index) for row in parsed)
    if actual_pairs != selection.pairs:
        fail(f"source rows differ from selected source/fold population: {seed_root}")
    group = parse_group(read_json(seed_root / "group_residual.json"), identity, provenance_sha256)
    if tuple(zip(group.source_ids, group.fold_indices)) != selection.pairs:
        fail(f"group residual differs from selected source/fold population: {seed_root}")
    return _LoadedCell(parsed, group, provenance)


def _completion_provenance(
    values: JsonValue,
    path: Path,
    identity: SeedIdentity,
) -> SeedProvenance:
    completion = exact_object(values, path, _COMPLETION_KEYS)
    provenance = parse_provenance(completion["provenance"], path)
    if (
        string(completion, "schema", path) != GATE2_COMPLETION_SCHEMA
        or string(completion, "object", path) != identity.object_name
        or integer(completion, "seed", path) != identity.seed
        or boolean(completion, "smoke", path)
        or integer(completion, "source_count", path) != GATE2_SOURCE_COUNT
        or not boolean(completion, "scratch_deleted", path)
        or not boolean(completion, "raw_maps_deleted_after_compaction", path)
        or provenance.model != REGISTERED_MODEL_PROVENANCE
        or provenance.method_sha256 != gate2_method_hash()
        or completion["model"] != provenance.model.to_manifest()
        or sha256(completion, "method_sha256", path) != provenance.method_sha256
        or sha256(completion, "code_config_sha256", path) != provenance.code_config_sha256
        or sha256(completion, "dataset_inventory_sha256", path)
        != provenance.dataset_inventory_sha256
        or sha256(completion, "split_inventory_sha256", path) != provenance.split_inventory_sha256
        or sha256(completion, "selection_sha256", path) != provenance.selection_sha256
        or sha256(completion, "provenance_sha256", path) != provenance.digest()
    ):
        fail(f"completion identity or full provenance digest mismatch: {path}")
    return provenance


def _parse_selection(
    payload: JsonValue,
    expectation: _SelectionExpectation,
) -> _SelectionPopulation:
    identity, provenance, path = expectation
    values = exact_object(payload, path, _SELECTION_KEYS)
    base = {key: value for key, value in values.items() if key != "split_inventory_sha256"}
    if (
        string(values, "schema", path) != "darc-gate2-selection-v1"
        or string(values, "version", path) != P16_PROTOCOL_VERSION
        or integer(values, "seed", path) != identity.seed
        or integer(values, "source_pool_count", path) < GATE2_SOURCE_COUNT
        or sha256(values, "dataset_inventory_sha256", path) != provenance.dataset_inventory_sha256
        or sha256(values, "split_inventory_sha256", path) != provenance.split_inventory_sha256
        or sha256_json(base) != provenance.split_inventory_sha256
    ):
        fail(f"selection header or split provenance mismatch: {path}")
    inventory = json_list(values["support_inventory"], path)
    sources: List[str] = []
    inventory_keys = frozenset({"path", "size", "sha256"})
    for row in inventory:
        item = exact_object(row, path, inventory_keys)
        if integer(item, "size", path) <= 0:
            fail(f"selection file size must be positive: {path}")
        sha256(item, "sha256", path)
        source = string(item, "path", path)
        if not source.startswith(f"{identity.object_name}/train/good/"):
            fail(f"selection source is outside the registered object train/good pool: {path}")
        sources.append(source)
    folds = json_list(values["folds"], path)
    pairs: List[Tuple[str, int]] = []
    fold_keys = frozenset({"fold_index", "memory_paths", "calibration_paths"})
    for expected_fold, row in enumerate(folds):
        item = exact_object(row, path, fold_keys)
        fold = integer(item, "fold_index", path)
        memory = strings(item["memory_paths"], path)
        calibration = strings(item["calibration_paths"], path)
        start = expected_fold * 4
        if fold != expected_fold or memory != sources[:start] + sources[start + 4 :]:
            fail(f"selection memory differs from consecutive P16 fold: {path}")
        if calibration != sources[start : start + 4]:
            fail(f"selection calibration differs from consecutive P16 fold: {path}")
        pairs.extend((source, fold) for source in calibration)
    if len(sources) != 16 or len(set(sources)) != 16 or len(folds) != 4:
        fail(f"selection must contain 16 unique sources and four folds: {path}")
    return _SelectionPopulation(tuple(pairs))
