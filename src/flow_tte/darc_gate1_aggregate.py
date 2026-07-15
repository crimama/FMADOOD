from __future__ import annotations

# pyright: reportMissingImports=false
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set, Tuple

from flow_tte.darc_gate1 import SourceMetric
from flow_tte.darc_gate1_aggregate_io import (
    DarcAggregateError,
)
from flow_tte.darc_gate1_aggregate_io import (
    fail as _fail,
)
from flow_tte.darc_gate1_aggregate_io import (
    json_dict as _dict,
)
from flow_tte.darc_gate1_aggregate_io import (
    json_integer as _integer,
)
from flow_tte.darc_gate1_aggregate_io import (
    json_list as _list,
)
from flow_tte.darc_gate1_aggregate_io import (
    json_string as _string,
)
from flow_tte.darc_gate1_aggregate_io import (
    parse_condition as _condition,
)
from flow_tte.darc_gate1_aggregate_io import (
    read_json as _read_json,
)
from flow_tte.darc_gate1_aggregate_io import (
    read_json_lines as _read_json_lines,
)
from flow_tte.darc_gate1_artifacts import CompletionExpectation, valid_completion
from flow_tte.darc_gate1_provenance import (
    REGISTERED_MODEL_PROVENANCE,
    JsonValue,
    SeedProvenance,
    sha256_json,
)

__all__ = ("AggregateInput", "DarcAggregateError", "load_full_gate1_metrics")


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class AggregateInput:
    metrics: Tuple[SourceMetric, ...]
    method_sha256: str
    code_config_sha256: str


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class _SeedIdentity:
    object_name: str
    seed: int


def load_full_gate1_metrics(
    root: Path,
    objects: Sequence[str],
    seeds: Sequence[int],
) -> AggregateInput:
    """Load only a checksum-valid exact object-by-seed Gate1 matrix."""
    expected = {(object_name, seed) for object_name in objects for seed in seeds}
    actual = {
        _completion_identity(path)
        for path in root.glob("objects/*/seed=*/complete.json")
    }
    if actual != expected:
        _fail(
            "aggregate requires exactly 15 objects x 3 seeds; "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}",
        )
    metrics: List[SourceMetric] = []
    methods: Set[str] = set()
    code_bundles: Set[str] = set()
    for object_name, seed in sorted(expected):
        loaded, provenance = _load_seed(root, _SeedIdentity(object_name, seed))
        metrics.extend(loaded)
        methods.add(provenance.method_sha256)
        code_bundles.add(provenance.code_config_sha256)
    if len(methods) != 1 or len(code_bundles) != 1:
        _fail("method or code/config bundle differs across completed seeds")
    return AggregateInput(tuple(metrics), methods.pop(), code_bundles.pop())


def _completion_identity(path: Path) -> Tuple[str, int]:
    parts = path.parent.name.split("=", 1)
    try:
        seed = int(parts[1]) if len(parts) == 2 and parts[0] == "seed" else -1
    except ValueError:
        seed = -1
    if seed < 0:
        _fail(f"invalid completion path: {path}")
    return path.parents[1].name, seed


def _load_seed(
    root: Path,
    identity: _SeedIdentity,
) -> Tuple[Tuple[SourceMetric, ...], SeedProvenance]:
    seed_root = root / "objects" / identity.object_name / f"seed={identity.seed}"
    completion = _read_json(seed_root / "complete.json")
    provenance = _parse_provenance(completion.get("provenance"))
    expectation = CompletionExpectation(
        seed_root=seed_root,
        selection_path=root / "selections" / identity.object_name / f"seed={identity.seed}.json",
        object_name=identity.object_name,
        seed=identity.seed,
        smoke=False,
        source_count=16,
        provenance=provenance,
    )
    if not valid_completion(expectation):
        _fail(f"stale, incomplete, or checksum-invalid seed: {seed_root}")
    selection = _read_json(expectation.selection_path)
    selected_sources, source_folds = _selection_sources(selection, provenance)
    source_payload = _read_json(seed_root / "source_metrics.json")
    _matching_provenance(source_payload, provenance, seed_root / "source_metrics.json")
    rows = _list(source_payload.get("rows"), seed_root / "source_metrics.json")
    metrics = tuple(_metric(row, identity) for row in rows)
    sources = {item.source_id for item in metrics}
    if len(metrics) != 16 or len(sources) != 16 or sources != selected_sources:
        _fail(f"expected 16 unique selected metric sources: {seed_root}")
    if any(source_folds.get(item.source_id) != item.fold_index for item in metrics):
        _fail(f"source/fold mapping differs from selection: {seed_root}")
    _validate_samples(seed_root / "samples.jsonl", identity, sources)
    _validate_bootstrap(seed_root / "bootstrap_inputs.json", provenance, sources)
    decision = _read_json(seed_root / "metrics.json")
    _matching_provenance(decision, provenance, seed_root / "metrics.json")
    if decision.get("source_count") != 16:
        _fail(f"seed decision source count is not 16: {seed_root}")
    return metrics, provenance


def _parse_provenance(payload: JsonValue | None) -> SeedProvenance:
    values = _dict(payload, Path("complete.json"))
    model = _dict(values.get("model"), Path("complete.json"))
    if model != REGISTERED_MODEL_PROVENANCE.to_manifest():
        _fail("completion model revision or digest is not registered")
    return SeedProvenance(
        method_sha256=_string(values, "method_sha256"),
        code_config_sha256=_string(values, "code_config_sha256"),
        dataset_inventory_sha256=_string(values, "dataset_inventory_sha256"),
        split_inventory_sha256=_string(values, "split_inventory_sha256"),
        selection_sha256=_string(values, "selection_sha256"),
    )


def _selection_sources(
    payload: Dict[str, JsonValue],
    provenance: SeedProvenance,
) -> Tuple[Set[str], Dict[str, int]]:
    split_sha256 = payload.get("split_inventory_sha256")
    base = {key: value for key, value in payload.items() if key != "split_inventory_sha256"}
    if (
        split_sha256 != provenance.split_inventory_sha256
        or payload.get("dataset_inventory_sha256") != provenance.dataset_inventory_sha256
        or sha256_json(base) != split_sha256
    ):
        _fail("selection inventory provenance mismatch")
    inventory = _list(payload.get("support_inventory"), Path("selection.json"))
    sources = {_string(_dict(row, Path("selection.json")), "path") for row in inventory}
    folds = _list(payload.get("folds"), Path("selection.json"))
    source_folds: Dict[str, int] = {}
    for row in folds:
        fold = _dict(row, Path("selection.json"))
        fold_index = _integer(fold, "fold_index")
        for source in _list(fold.get("calibration_paths"), Path("selection.json")):
            if not isinstance(source, str):
                _fail("calibration path must be a string")
            source_folds[source] = fold_index
    if len(inventory) != 16 or len(sources) != 16 or set(source_folds) != sources:
        _fail("selection must expose 16 unique sources across four folds")
    return sources, source_folds


def _metric(payload: JsonValue, identity: _SeedIdentity) -> SourceMetric:
    row = _dict(payload, Path("source_metrics.json"))
    if row.get("object") != identity.object_name or row.get("seed") != identity.seed:
        _fail("source metric identity differs from its seed directory")
    return SourceMetric(
        object_name=identity.object_name,
        source_id=_string(row, "source"),
        seed=identity.seed,
        fold_index=_integer(row, "fold"),
        low=_condition(row.get("low")),
        bilinear_null=_condition(row.get("bilinear_null")),
        high=_condition(row.get("high")),
    )


def _validate_samples(path: Path, identity: _SeedIdentity, sources: Set[str]) -> None:
    rows = _read_json_lines(path)
    sample_sources: Set[str] = set()
    for row in rows:
        if row.get("object") != identity.object_name or row.get("seed") != identity.seed:
            _fail(f"sample identity mismatch: {path}")
        sample_sources.add(_string(row, "source"))
        cues = _list(row.get("cues"), path)
        if len(cues) != 3:
            _fail(f"sample lacks three cue provenance rows: {path}")
        for cue in cues:
            cue_row = _dict(cue, path)
            metadata = _dict(cue_row.get("metadata"), path)
            _integer(metadata, "seed")
            for key in ("start_xy", "end_xy", "color_rgb"):
                _list(metadata.get(key), path)
            if len(_string(cue_row, "mask_sha256")) != 64:
                _fail(f"invalid cue mask hash: {path}")
    if len(rows) != 16 or sample_sources != sources:
        _fail(f"samples do not match 16 metric sources: {path}")


def _validate_bootstrap(path: Path, provenance: SeedProvenance, sources: Set[str]) -> None:
    payload = _read_json(path)
    _matching_provenance(payload, provenance, path, require_method_hash=False)
    rows = _list(payload.get("rows"), path)
    bootstrap_sources = {_string(_dict(row, path), "source") for row in rows}
    valid_header = (
        payload.get("schema") == "darc-stratified-source-bootstrap-v1"
        and payload.get("seed") == 20260710
        and payload.get("replicates") == 10000
    )
    if not valid_header or len(rows) != 16 or bootstrap_sources != sources:
        _fail(f"bootstrap inputs are incomplete or stale: {path}")


def _matching_provenance(
    payload: Mapping[str, JsonValue],
    provenance: SeedProvenance,
    path: Path,
    *,
    require_method_hash: bool = True,
) -> None:
    if (
        (require_method_hash and payload.get("method_hash") != provenance.method_sha256)
        or payload.get("provenance_sha256") != provenance.digest()
    ):
        _fail(f"artifact provenance mismatch: {path}")
