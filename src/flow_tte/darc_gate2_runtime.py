"""Selection-first orchestration for the frozen DARC Gate 2 experiment."""

from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false
import gc
import json
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Dict, Final, List, Tuple

import numpy as np
from PIL import Image

from flow_tte.darc_feature_stream import DarcFeatureStream, ImageFeatures
from flow_tte.darc_gate2_artifacts import (
    CompletionExpectation,
    gate2_method_hash,
    json_document_sha256,
    valid_completion,
    write_completion,
    write_json,
    write_jsonl,
)
from flow_tte.darc_gate2_provenance import (
    JsonValue,
    SeedProvenance,
    dataset_inventory_hash,
    selection_manifest,
)
from flow_tte.darc_gate2_runtime_fold import FoldCompactResult, run_gate2_fold
from flow_tte.darc_gate2_runtime_support import (
    GroupResidualPopulation,
    build_group_residual,
)
from flow_tte.darc_gate2_runtime_types import (
    Gate2RuntimeConfig,
    PreparedGate2Run,
    PreparedSeedRun,
    SeedRunReport,
)
from flow_tte.darc_resources import build_p16_split

_IMAGE_SUFFIXES: Final = (".png", ".jpg", ".jpeg", ".bmp")
_PROFILE_ENV: Final = "FMAD_DARC_GATE2_PROFILE"
_LOGGER = logging.getLogger(__name__)


def prepare_gate2_run(config: Gate2RuntimeConfig) -> PreparedGate2Run:
    """Freeze all path-only P16 selections before model creation or image decode."""
    paths = train_good_paths(config.data_root, config.object_name)
    method_sha256 = gate2_method_hash()
    dataset_sha256 = dataset_inventory_hash(config.data_root, paths)
    prepared: List[PreparedSeedRun] = []
    pending: List[PreparedSeedRun] = []
    for seed in config.seeds:
        split = build_p16_split(paths, seed)
        manifest = selection_manifest(split, config.data_root, dataset_sha256)
        split_sha256 = manifest.get("split_inventory_sha256")
        if not isinstance(split_sha256, str):
            raise TypeError("Gate 2 selection manifest omitted its split digest")
        selection_path = (
            config.output_root / "selections" / config.object_name / f"seed={seed}.json"
        )
        provenance = SeedProvenance(
            method_sha256=method_sha256,
            code_config_sha256=config.code_config_sha256,
            dataset_inventory_sha256=dataset_sha256,
            split_inventory_sha256=split_sha256,
            selection_sha256=json_document_sha256(manifest),
        )
        expectation = CompletionExpectation(
            seed_root=config.output_root / "objects" / config.object_name / f"seed={seed}",
            selection_path=selection_path,
            object_name=config.object_name,
            seed=seed,
            smoke=config.smoke,
            source_count=1 if config.smoke else 16,
            provenance=provenance,
        )
        seed_run = PreparedSeedRun(split=split, expectation=expectation)
        prepared.append(seed_run)
        if not valid_completion(expectation):
            write_json(selection_path, manifest)
            pending.append(seed_run)
    return PreparedGate2Run(
        seeds=tuple(prepared),
        pending=tuple(pending),
        method_sha256=method_sha256,
    )


def run_gate2_seed(
    config: Gate2RuntimeConfig,
    prepared: PreparedSeedRun,
    stream: DarcFeatureStream,
) -> SeedRunReport:
    """Run one frozen P16 seed and persist only compact scientific artifacts."""
    split = prepared.split
    cache: Dict[str, ImageFeatures] = {}
    cache_started = perf_counter()
    for index, path in enumerate(split.support_paths, start=1):
        image_started = perf_counter()
        cache[path] = stream.extract(_read_rgb(Path(path)))
        _profile_event(
            "feature_extract",
            {
                "object": config.object_name,
                "seed": split.seed,
                "index": index,
                "total": len(split.support_paths),
                "elapsed_s": round(perf_counter() - image_started, 6),
            },
        )
    _profile_event(
        "feature_cache_total",
        {
            "object": config.object_name,
            "seed": split.seed,
            "image_count": len(cache),
            "elapsed_s": round(perf_counter() - cache_started, 6),
        },
    )
    provenance_sha256 = prepared.expectation.provenance.digest()
    folds = split.folds[:1] if config.smoke else split.folds
    compact: List[FoldCompactResult] = [
        run_gate2_fold(
            config,
            split.seed,
            fold,
            cache,
            stream,
            provenance_sha256,
        )
        for fold in folds
    ]
    rows: List[Dict[str, JsonValue]] = [row for result in compact for row in result.source_rows]
    source_ids = tuple(value for result in compact for value in result.source_ids)
    fold_indices = tuple(value for result in compact for value in result.fold_indices)
    population_rows = tuple(value for result in compact for value in result.population_rows)
    l0_residuals = tuple(value for result in compact for value in result.l0_residuals)
    l1_residuals = tuple(value for result in compact for value in result.l1_residuals)
    expected_sources = 1 if config.smoke else 16
    if len(rows) != expected_sources:
        message = f"Gate 2 seed produced {len(rows)} sources, expected {expected_sources}"
        raise RuntimeError(message)
    group = build_group_residual(
        GroupResidualPopulation(
            object_name=config.object_name,
            seed=split.seed,
            source_ids=source_ids,
            fold_indices=fold_indices,
            population_rows=population_rows,
        ),
        l0_residuals,
        l1_residuals,
    )
    group_manifest = group.to_manifest()
    group_manifest["provenance_sha256"] = provenance_sha256
    seed_root = prepared.expectation.seed_root
    write_jsonl(seed_root / "source_rows.jsonl", rows)
    write_json(seed_root / "group_residual.json", group_manifest)
    compact.clear()
    cache.clear()
    gc.collect()
    write_completion(prepared.expectation)
    return SeedRunReport(
        object_name=config.object_name,
        seed=split.seed,
        source_count=len(rows),
        smoke=config.smoke,
    )


def train_good_paths(data_root: Path, object_name: str) -> Tuple[Path, ...]:
    directory = data_root / object_name / "train" / "good"
    if not directory.is_dir():
        message = f"train/good directory not found: {directory}"
        raise FileNotFoundError(message)
    paths = tuple(
        sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        ),
    )
    if len(paths) < 16:
        raise ValueError("Gate 2 requires at least 16 train/good images")
    return paths


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _profile_event(event: str, details: Dict[str, object]) -> None:
    if os.environ.get(_PROFILE_ENV) != "1":
        return
    payload: Dict[str, object] = {"event": event}
    payload.update(details)
    _LOGGER.warning("DARC_GATE2_PROFILE %s", json.dumps(payload, sort_keys=True))
