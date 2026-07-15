"""Path-frozen streaming runtime for the bounded DARC AD2 pilot."""

from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Final, List, NamedTuple, Protocol, Tuple

import numpy as np
from PIL import Image
from typing_extensions import final, override

from flow_tte.darc_ad2_pilot import (
    RawRungMaps,
    ladder_coverage,
    mean_rung_maps,
    raw_rung_maps,
)
from flow_tte.darc_ad2_pilot_io import (
    PilotMapTarget,
    PilotTestImage,
    TestLimits,
    discover_test_images,
    write_rung_maps,
)
from flow_tte.darc_feature_stream import ImageFeatures, RgbArray
from flow_tte.darc_gate2_pipeline import score_query_ladder
from flow_tte.darc_gate2_pipeline_types import QueryLadderInput, QueryPipelineConfig
from flow_tte.darc_gate2_provenance import file_sha256
from flow_tte.darc_knn import ChunkedKnnConfig
from flow_tte.darc_resources import P16Fold, P16Split, build_p16_split

_IMAGE_SUFFIXES: Final = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})


class PilotFeatureStream(Protocol):
    def extract(self, image: RgbArray) -> ImageFeatures: ...


@final
class PilotRuntimeError(ValueError):
    __slots__ = ("reason",)

    reason: str

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(str(self))

    @override
    def __str__(self) -> str:
        return f"Invalid DARC AD2 pilot runtime: {self.reason}"


# Python 3.8 dataclasses do not support slots=True.
@dataclass(frozen=True)
class PilotRuntimeConfig:
    data_root: Path
    output_root: Path
    object_name: str
    device: str
    seed: int
    fold_indices: Tuple[int, ...]
    test_limits: TestLimits = field(default_factory=TestLimits)
    query_chunk_size: int = 256
    memory_chunk_size: int = 16384
    shard_index: int = 0
    shard_count: int = 1

    def __post_init__(self) -> None:
        _validate_config(self)

class PreparedPilot(NamedTuple):
    split: P16Split
    folds: Tuple[P16Fold, ...]
    test_images: Tuple[PilotTestImage, ...]


class PilotRunReport(NamedTuple):
    object_name: str
    seed: int
    fold_count: int
    image_count: int
    coverage_row_count: int


def claim_fresh_output_root(output_root: Path) -> None:
    """Atomically reserve a new run root so stale rows/maps cannot be reused."""
    try:
        output_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        message = f"output root must not already exist: {output_root}"
        raise PilotRuntimeError(message) from error


def prepare_pilot(config: PilotRuntimeConfig) -> PreparedPilot:
    """Freeze the complete P16 identity before discovering bounded public queries."""
    _validate_config(config)
    train_directory = config.data_root / config.object_name / "train" / "good"
    if not train_directory.is_dir():
        message = f"missing normal training directory: {train_directory}"
        raise PilotRuntimeError(message)
    normal_paths = tuple(
        sorted(
            path
            for path in train_directory.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        ),
    )
    split = build_p16_split(normal_paths, config.seed)
    folds = tuple(split.folds[index] for index in config.fold_indices)
    all_test_images = discover_test_images(
        config.data_root,
        config.object_name,
        config.test_limits,
    )
    test_images = tuple(
        image
        for query_index, image in enumerate(all_test_images)
        if query_index % config.shard_count == config.shard_index
    )
    if not test_images:
        raise PilotRuntimeError("operational shard contains no public test queries")
    return PreparedPilot(split=split, folds=folds, test_images=test_images)


def pilot_query_id(object_name: str, path: Path) -> str:
    """Return a population-neutral, content-bound scorer identity."""
    if not object_name:
        raise PilotRuntimeError("query object name must be non-empty")
    return f"{object_name}/sha256={file_sha256(path)}"


def run_pilot(
    config: PilotRuntimeConfig,
    prepared: PreparedPilot,
    stream: PilotFeatureStream,
) -> PilotRunReport:
    """Cache P16 once, score selected folds, average raw maps, and persist each query."""
    _validate_config(config)
    if prepared.split.seed != config.seed:
        raise PilotRuntimeError("prepared split seed does not match the runtime config")
    expected_folds = tuple(config.fold_indices)
    if tuple(fold.fold_index for fold in prepared.folds) != expected_folds:
        raise PilotRuntimeError("prepared folds do not match the runtime config")

    cache: Dict[str, ImageFeatures] = {}
    for support_index, path_text in enumerate(prepared.split.support_paths, start=1):
        cache[path_text] = stream.extract(_read_rgb(Path(path_text)))
        _progress(
            "support_cached",
            {
                "index": support_index,
                "total": len(prepared.split.support_paths),
            },
        )
    if len(cache) != 16:
        raise PilotRuntimeError("prepared P16 cache must contain exactly 16 supports")

    knn = ChunkedKnnConfig(
        device=config.device,
        query_chunk_size=config.query_chunk_size,
        memory_chunk_size=config.memory_chunk_size,
        top_k=5,
    )
    target = PilotMapTarget(config.output_root, config.object_name)
    coverage_row_count = 0
    for image_index, image in enumerate(prepared.test_images, start=1):
        query = stream.extract(_read_rgb(image.path))
        fold_maps: List[RawRungMaps] = []
        coverage_rows: List[Dict[str, object]] = []
        for fold in prepared.folds:
            candidates = {
                Path(path).relative_to(config.data_root).as_posix(): cache[path]
                for path in fold.memory_paths
            }
            ladder = score_query_ladder(
                QueryLadderInput(
                    query_id=pilot_query_id(config.object_name, image.path),
                    query=query,
                    candidates=candidates,
                    knn_config=knn,
                ),
                QueryPipelineConfig(complete_g0=True),
            )
            fold_maps.append(raw_rung_maps(ladder))
            coverage = ladder_coverage(ladder)
            coverage_rows.append(
                {
                    "schema": "darc-ad2-raw-ladder-coverage-v1",
                    "object": config.object_name,
                    "seed": config.seed,
                    "fold_index": fold.fold_index,
                    "query_id": ladder.query_id,
                    "source_path": image.path.relative_to(config.data_root).as_posix(),
                    "population": image.population.value,
                    "selected_support_ids": list(ladder.selected_support_ids),
                    "token_count": coverage.token_count,
                    "nonfallback_count": coverage.nonfallback_count,
                    "fallback_fraction": coverage.fallback_fraction,
                    "registration_count": coverage.registration_count,
                    "accepted_registration_count": coverage.accepted_registration_count,
                    "registration_accepted": [
                        item.accepted for item in ladder.registration_audit
                    ],
                    "registration_pair_counts": [
                        item.pair_count for item in ladder.registration_audit
                    ],
                    "support_count_histograms": {
                        "l0": list(coverage.l0_support_histogram),
                        "l1": list(coverage.l1_support_histogram),
                        "shared": list(coverage.shared_support_histogram),
                        "r1": list(coverage.r1_support_histogram),
                    },
                    "population_sha256": ladder.audit.population_sha256,
                    "support_sha256": ladder.audit.support_sha256,
                    "fallback_sha256": ladder.audit.fallback_sha256,
                },
            )
        write_rung_maps(target, image, mean_rung_maps(tuple(fold_maps)))
        _append_jsonl(config.output_root / "coverage_rows.jsonl", coverage_rows)
        coverage_row_count += len(coverage_rows)
        _progress(
            "query_complete",
            {
                "index": image_index,
                "total": len(prepared.test_images),
                "population": image.population.value,
                "image": image.path.name,
            },
        )
    return PilotRunReport(
        object_name=config.object_name,
        seed=config.seed,
        fold_count=len(prepared.folds),
        image_count=len(prepared.test_images),
        coverage_row_count=coverage_row_count,
    )


def _read_rgb(path: Path) -> RgbArray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _validate_config(config: PilotRuntimeConfig) -> None:
    folds = config.fold_indices
    valid_folds = (
        bool(folds)
        and len(set(folds)) == len(folds)
        and all(0 <= value < 4 for value in folds)
    )
    if not config.object_name or not config.device:
        raise PilotRuntimeError("object_name and device must be non-empty")
    if not valid_folds:
        raise PilotRuntimeError("fold_indices must be unique values in [0, 4)")
    if config.query_chunk_size < 1 or config.memory_chunk_size < 1:
        raise PilotRuntimeError("k-NN chunk sizes must be positive")
    valid_shard = config.shard_count > 0 and 0 <= config.shard_index < config.shard_count
    if not valid_shard:
        raise PilotRuntimeError("shard_index must be in [0, shard_count)")


def _progress(event: str, details: Dict[str, object]) -> None:
    payload: Dict[str, object] = {"event": event}
    payload.update(details)
    print(json.dumps(payload, sort_keys=True), flush=True)


def _append_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
