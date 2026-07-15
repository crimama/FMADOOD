from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Final, List, Tuple

import numpy as np
from PIL import Image

from flow_tte.darc_feature_stream import DarcFeatureStream, ImageFeatures
from flow_tte.darc_gate1 import (
    DarcGate1Error,
    Gate1Thresholds,
    SourceConditions,
    SourceEvaluationInput,
    SourceMetric,
    decide_gate1,
    evaluate_source,
)
from flow_tte.darc_gate1_artifacts import (
    CompletionExpectation,
    JsonValue,
    bootstrap_inputs_manifest,
    gate1_method_hash,
    json_document_sha256,
    source_metrics_manifest,
    valid_completion,
    write_completion,
    write_json,
    write_jsonl,
)
from flow_tte.darc_gate1_provenance import (
    SeedProvenance,
    dataset_inventory_hash,
    selection_manifest,
)
from flow_tte.darc_gate1_samples import SourceSampleRecord, cue_record, sample_record
from flow_tte.darc_gate1_scoring import CandidateFeatures, ReferenceScores, leave_one_out_references
from flow_tte.darc_gate1_scoring import score_query as score_feature_query
from flow_tte.darc_knn import ChunkedKnnConfig
from flow_tte.darc_resources import P16Split, build_p16_split
from flow_tte.darc_synthetic import LINE_CUE_PROFILES, LINE_CUE_VERSION, insert_line_cue

_IMAGE_SUFFIXES: Final = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class Gate1RuntimeConfig:
    data_root: Path
    output_root: Path
    object_name: str
    device: str
    seeds: Tuple[int, ...]
    code_config_sha256: str
    smoke: bool = False


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class PreparedSeedRun:
    split: P16Split
    expectation: CompletionExpectation


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class PreparedGate1Run:
    seeds: Tuple[PreparedSeedRun, ...]
    pending: Tuple[PreparedSeedRun, ...]
    method_hash: str


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class SeedRunReport:
    metrics: Tuple[SourceMetric, ...]
    decision_passed: bool
    source_count: int


def prepare_gate1_run(config: Gate1RuntimeConfig) -> PreparedGate1Run:
    """Freeze path-only P16 selections before model creation or image decoding."""
    paths = _train_good_paths(config)
    splits = tuple(build_p16_split(paths, seed) for seed in config.seeds)
    method_hash = gate1_method_hash()
    dataset_sha256 = dataset_inventory_hash(config.data_root, paths)
    prepared = []
    pending = []
    for split in splits:
        manifest = selection_manifest(split, config.data_root, dataset_sha256)
        split_sha256 = manifest.get("split_inventory_sha256")
        if not isinstance(split_sha256, str):
            raise DarcGate1Error("selection manifest did not produce a split inventory hash")
        selection_path = (
            config.output_root / "selections" / config.object_name / f"seed={split.seed}.json"
        )
        provenance = SeedProvenance(
            method_sha256=method_hash,
            code_config_sha256=config.code_config_sha256,
            dataset_inventory_sha256=dataset_sha256,
            split_inventory_sha256=split_sha256,
            selection_sha256=json_document_sha256(manifest),
        )
        seed_root = config.output_root / "objects" / config.object_name / f"seed={split.seed}"
        expectation = CompletionExpectation(
            seed_root=seed_root,
            selection_path=selection_path,
            object_name=config.object_name,
            seed=split.seed,
            smoke=config.smoke,
            source_count=1 if config.smoke else 16,
            provenance=provenance,
        )
        seed_run = PreparedSeedRun(split=split, expectation=expectation)
        prepared.append(seed_run)
        if not valid_completion(expectation):
            write_json(selection_path, manifest)
            pending.append(seed_run)
    return PreparedGate1Run(seeds=tuple(prepared), pending=tuple(pending), method_hash=method_hash)


def run_gate1_seed(
    config: Gate1RuntimeConfig,
    prepared: PreparedSeedRun,
    stream: DarcFeatureStream,
) -> SeedRunReport:
    """Run one P16 seed in memory and persist compact evaluated artifacts only."""
    split = prepared.split
    cache: Dict[str, ImageFeatures] = {}
    for path_text in split.support_paths:
        cache[path_text] = stream.extract(_read_rgb(Path(path_text)))
    knn = ChunkedKnnConfig(device=config.device)
    metrics: List[SourceMetric] = []
    records: List[JsonValue] = []
    folds = split.folds[:1] if config.smoke else split.folds
    for fold in folds:
        memory = {path: cache[path] for path in fold.memory_paths}
        candidates = CandidateFeatures(features=memory, config=knn)
        raw_references = leave_one_out_references(candidates)
        references = ReferenceScores(
            low=np.sort(raw_references.low),
            bilinear_null=np.sort(raw_references.bilinear_null),
            high=np.sort(raw_references.high),
        )
        clean_scores = {
            path: score_feature_query(cache[path], candidates)
            for path in fold.calibration_paths
        }
        calibration_low = tuple(clean_scores[path].low for path in fold.calibration_paths)
        calibration_null = tuple(
            clean_scores[path].bilinear_null for path in fold.calibration_paths
        )
        calibration_high = tuple(clean_scores[path].high for path in fold.calibration_paths)
        source_paths = fold.calibration_paths[:1] if config.smoke else fold.calibration_paths
        for source_path in source_paths:
            source_image = _read_rgb(Path(source_path))
            clean_score = clean_scores[source_path]
            low_maps = [clean_score.low]
            null_maps = [clean_score.bilinear_null]
            high_maps = [clean_score.high]
            masks = [np.zeros(source_image.shape[:2], dtype=np.bool_)]
            cues = []
            for profile_index, profile in enumerate(LINE_CUE_PROFILES):
                cue_seed = _cue_seed(config, split.seed, source_path, profile_index)
                cue = insert_line_cue(source_image, profile, cue_seed)
                cue_score = score_feature_query(stream.extract(cue.image), candidates)
                low_maps.append(cue_score.low)
                null_maps.append(cue_score.bilinear_null)
                high_maps.append(cue_score.high)
                masks.append(np.asarray(cue.mask > 0, dtype=np.bool_))
                cues.append(cue_record(cue, cue_score.selected_support_ids))
            evaluated = evaluate_source(
                SourceEvaluationInput(
                    object_name=config.object_name,
                    source_id=Path(source_path).relative_to(config.data_root).as_posix(),
                    seed=split.seed,
                    fold_index=fold.fold_index,
                    masks=tuple(masks),
                    low=SourceConditions(tuple(low_maps), references.low, calibration_low),
                    bilinear_null=SourceConditions(
                        tuple(null_maps),
                        references.bilinear_null,
                        calibration_null,
                    ),
                    high=SourceConditions(tuple(high_maps), references.high, calibration_high),
                ),
            )
            metrics.append(evaluated)
            records.append(
                sample_record(
                    config.object_name,
                    config.data_root,
                    SourceSampleRecord(
                        seed=split.seed,
                        fold=fold.fold_index,
                        source_path=source_path,
                        clean_selection=clean_score.selected_support_ids,
                        cues=tuple(cues),
                    ),
                ),
            )
    thresholds = Gate1Thresholds()
    decision = decide_gate1(metrics, thresholds)
    provenance_sha256 = prepared.expectation.provenance.digest()
    seed_root = prepared.expectation.seed_root
    write_json(
        seed_root / "source_metrics.json",
        source_metrics_manifest(metrics, provenance_sha256),
    )
    write_json(
        seed_root / "metrics.json",
        {
            "method_hash": gate1_method_hash(),
            "provenance_sha256": provenance_sha256,
            **decision.to_manifest(),
        },
    )
    write_json(
        seed_root / "bootstrap_inputs.json",
        bootstrap_inputs_manifest(metrics, provenance_sha256),
    )
    write_jsonl(seed_root / "samples.jsonl", records)
    write_completion(prepared.expectation)
    return SeedRunReport(
        metrics=tuple(metrics),
        decision_passed=decision.passed,
        source_count=len(metrics),
    )


def _train_good_paths(config: Gate1RuntimeConfig) -> Tuple[Path, ...]:
    directory = config.data_root / config.object_name / "train" / "good"
    if not directory.is_dir():
        message = f"train/good directory not found: {directory}"
        raise FileNotFoundError(message)
    return tuple(
        sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        ),
    )


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _cue_seed(config: Gate1RuntimeConfig, seed: int, path: str, profile: int) -> int:
    relative = Path(path).relative_to(config.data_root).as_posix()
    payload = f"{LINE_CUE_VERSION}\0{config.object_name}\0{relative}\0{seed}\0{profile}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")
