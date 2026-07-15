"""One-fold streaming execution for the frozen DARC Gate 2 protocol."""

from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from pathlib import Path
from typing import Dict, List, Mapping, NamedTuple, Tuple

import numpy as np
import numpy.typing as npt
from PIL import Image

from flow_tte.darc_feature_stream import DarcFeatureStream, ImageFeatures
from flow_tte.darc_gate2_evaluation import evaluate_source_maps
from flow_tte.darc_gate2_evaluation_types import (
    FoldCleanCalibration,
    RungSourceMaps,
    SourceEvaluationInput,
    SourceMapBundle,
    SourceMasks,
)
from flow_tte.darc_gate2_pipeline import calibrate_query_ladder, score_query_ladder
from flow_tte.darc_gate2_pipeline_types import (
    QueryEvidenceMaps,
    QueryLadderInput,
    QueryLadderResult,
    QueryPipelineConfig,
)
from flow_tte.darc_gate2_provenance import JsonValue
from flow_tte.darc_gate2_runtime_support import (
    build_normal_references,
    build_source_audit,
    cue_seed,
    relative_id,
)
from flow_tte.darc_gate2_runtime_types import Gate2RuntimeConfig
from flow_tte.darc_gate2_scoring_types import RungNormalReferences
from flow_tte.darc_knn import ChunkedKnnConfig
from flow_tte.darc_resources import P16Fold
from flow_tte.darc_synthetic import LINE_CUE_PROFILES, insert_line_cue

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
RgbArray = npt.NDArray[np.uint8]


class FoldCompactResult(NamedTuple):
    source_rows: Tuple[Dict[str, JsonValue], ...]
    source_ids: Tuple[str, ...]
    fold_indices: Tuple[int, ...]
    population_rows: Tuple[Dict[str, JsonValue], ...]
    l0_residuals: Tuple[FloatArray, ...]
    l1_residuals: Tuple[FloatArray, ...]


class _ScoredQuery(NamedTuple):
    ladder: QueryLadderResult
    evidence: QueryEvidenceMaps


def run_gate2_fold(  # noqa: PLR0913 -- explicit frozen fold boundary
    config: Gate2RuntimeConfig,
    seed: int,
    fold: P16Fold,
    cache: Mapping[str, ImageFeatures],
    stream: DarcFeatureStream,
    provenance_sha256: str,
) -> FoldCompactResult:
    """Score one 12-memory/4-calibration fold and retain compact gate inputs."""
    knn = _knn_config(config)
    memory = _candidate_mapping(config.data_root, fold.memory_paths, cache)
    references = build_normal_references(
        tuple(
            score_query_ladder(
                QueryLadderInput(
                    query_id=relative_id(config.data_root, query_path),
                    query=cache[query_path],
                    candidates=_candidate_mapping(
                        config.data_root,
                        tuple(path for path in fold.memory_paths if path != query_path),
                        cache,
                    ),
                    knn_config=knn,
                ),
                QueryPipelineConfig(complete_g0=True),
            )
            for query_path in fold.memory_paths
        ),
    )
    calibration = {
        path: _score_query(
            relative_id(config.data_root, path),
            cache[path],
            memory,
            knn,
            references,
        )
        for path in fold.calibration_paths
    }
    calibration_maps = FoldCleanCalibration(
        l0=tuple(calibration[path].evidence.l0 for path in fold.calibration_paths),
        l1=tuple(calibration[path].evidence.l1 for path in fold.calibration_paths),
    )

    rows: List[Dict[str, JsonValue]] = []
    source_ids: List[str] = []
    fold_indices: List[int] = []
    populations: List[Dict[str, JsonValue]] = []
    l0_residuals: List[FloatArray] = []
    l1_residuals: List[FloatArray] = []
    active_sources = fold.calibration_paths[:1] if config.smoke else fold.calibration_paths
    for source_path in active_sources:
        source_id = relative_id(config.data_root, source_path)
        clean = calibration[source_path]
        image = _read_rgb(Path(source_path))
        cues: List[_ScoredQuery] = []
        masks: List[BoolArray] = []
        for profile_index, profile in enumerate(LINE_CUE_PROFILES):
            deterministic_seed = cue_seed(
                config.data_root,
                config.object_name,
                seed,
                source_path,
                profile_index,
            )
            cue = insert_line_cue(image, profile, deterministic_seed)
            query_id = f"{source_id}#cue={profile.name}#seed={deterministic_seed}"
            cues.append(
                _score_query(
                    query_id,
                    stream.extract(cue.image),
                    memory,
                    knn,
                    references,
                ),
            )
            masks.append(np.asarray(cue.mask > 0, dtype=np.bool_))
        source_masks = SourceMasks(
            thin_profiles=(masks[0], masks[1]),
            broad=masks[2],
        )
        source_maps = SourceMapBundle(
            l0=_rung_maps(
                clean.evidence.l0,
                tuple(item.evidence.l0 for item in cues),
            ),
            l1=_rung_maps(
                clean.evidence.l1,
                tuple(item.evidence.l1 for item in cues),
            ),
            r1=_rung_maps(
                clean.evidence.r1,
                tuple(item.evidence.r1 for item in cues),
            ),
        )
        audit = build_source_audit(
            (clean.ladder, *(item.ladder for item in cues)),
            tuple(masks),
        )
        evaluated = evaluate_source_maps(
            SourceEvaluationInput(
                object_name=config.object_name,
                seed=seed,
                fold_index=fold.fold_index,
                source_id=source_id,
                maps=source_maps,
                masks=source_masks,
                calibration=calibration_maps,
                audit=audit,
            ),
        )
        row = evaluated.metric.to_manifest()
        row["evaluation"] = evaluated.manifest.to_manifest()
        row["provenance_sha256"] = provenance_sha256
        rows.append(row)
        source_ids.append(source_id)
        fold_indices.append(fold.fold_index)
        clean_l0 = clean.ladder.concatenate_l0_residuals()
        clean_l1 = clean.ladder.concatenate_l1_residuals()
        populations.append(
            {
                "source": source_id,
                "fold": fold.fold_index,
                "population_sha256": clean.ladder.audit.population_sha256,
                "token_count": len(clean_l0),
            },
        )
        l0_residuals.append(clean_l0)
        l1_residuals.append(clean_l1)
    return FoldCompactResult(
        source_rows=tuple(rows),
        source_ids=tuple(source_ids),
        fold_indices=tuple(fold_indices),
        population_rows=tuple(populations),
        l0_residuals=tuple(l0_residuals),
        l1_residuals=tuple(l1_residuals),
    )


def _score_query(
    query_id: str,
    features: ImageFeatures,
    memory: Mapping[str, ImageFeatures],
    knn: ChunkedKnnConfig,
    references: RungNormalReferences,
) -> _ScoredQuery:
    ladder = score_query_ladder(
        QueryLadderInput(query_id, features, memory, knn),
        QueryPipelineConfig(complete_g0=True),
    )
    return _ScoredQuery(ladder, calibrate_query_ladder(ladder, references))


def _rung_maps(
    clean: FloatArray,
    values: Tuple[FloatArray, ...],
) -> RungSourceMaps:
    if len(values) != 3:
        raise ValueError("rung source maps require exactly two thin and one broad cue")
    return RungSourceMaps(clean=clean, thin_profiles=(values[0], values[1]), broad=values[2])


def _candidate_mapping(
    data_root: Path,
    paths: Tuple[str, ...],
    cache: Mapping[str, ImageFeatures],
) -> Dict[str, ImageFeatures]:
    return {relative_id(data_root, path): cache[path] for path in paths}


def _knn_config(config: Gate2RuntimeConfig) -> ChunkedKnnConfig:
    return ChunkedKnnConfig(
        device=config.device,
        query_chunk_size=min(config.query_chunk_size, config.candidate_chunk_size),
        memory_chunk_size=config.memory_chunk_size,
        top_k=5,
    )


def _read_rgb(path: Path) -> RgbArray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
