"""Pure compaction helpers for the DARC Gate 2 streaming runtime."""

from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
import hashlib
from pathlib import Path
from typing import Dict, Final, List, NamedTuple, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from flow_tte.darc_feature_stream import ImageFeatures
from flow_tte.darc_gate2_metrics import GroupResidual, SourceAudit, higher_p999
from flow_tte.darc_gate2_pipeline_types import QueryLadderResult
from flow_tte.darc_gate2_provenance import JsonValue, sha256_json
from flow_tte.darc_gate2_scoring_types import RungNormalReferences
from flow_tte.darc_synthetic import LINE_CUE_VERSION

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]

_EXPECTED_CONDITIONS: Final = 4


class GroupResidualPopulation(NamedTuple):
    object_name: str
    seed: int
    source_ids: Tuple[str, ...]
    fold_indices: Tuple[int, ...]
    population_rows: Tuple[Dict[str, JsonValue], ...]


def flatten_high_features(image: ImageFeatures) -> FloatArray:
    """Concatenate every emitted high-grid token in deterministic crop order."""
    if not image.high or len(image.high) != len(image.crops):
        raise ValueError("high feature grids must match the non-empty crop inventory")
    flattened: List[FloatArray] = []
    dimensions = set()
    for raw_grid in image.high:
        values: FloatArray = np.asarray(raw_grid, dtype=np.float32)
        if values.ndim != 3 or values.size == 0 or not bool(np.all(np.isfinite(values))):
            raise ValueError("high feature grids must be finite HxWxD arrays")
        dimension = int(np.size(values, axis=2))
        dimensions.add(dimension)
        flattened.append(np.asarray(values.reshape(-1, dimension), dtype=np.float32))
    if len(dimensions) != 1:
        raise ValueError("high feature grids must share one feature dimension")
    return np.asarray(np.concatenate(flattened, axis=0), dtype=np.float32)


def build_normal_references(results: Sequence[QueryLadderResult]) -> RungNormalReferences:
    """Compact complete normal LOO G0 and nonfallback local rung references."""
    if len(results) < 2:
        raise ValueError("normal references require at least two LOO query results")
    g0: List[FloatArray] = []
    l0: List[FloatArray] = []
    l1: List[FloatArray] = []
    r1: List[FloatArray] = []
    for result in results:
        if not result.crops:
            raise ValueError("normal LOO results require non-empty crop populations")
        for crop in result.crops:
            scores = crop.scores
            token_count = int(np.prod(tuple(crop.token_shape)))
            arrays = (scores.g0, scores.l0, scores.l1, scores.r1)
            if any(values.shape != (token_count,) for values in arrays):
                raise ValueError("normal LOO rung arrays must match their token population")
            if not bool(np.all(scores.g0_valid)):
                raise ValueError("normal LOO G0 references require every token")
            if any(not bool(np.all(np.isfinite(values))) for values in arrays):
                raise ValueError("normal LOO rung arrays must be finite")
            keep: BoolArray = np.asarray(~scores.common_fallback, dtype=np.bool_)
            g0.append(np.asarray(scores.g0, dtype=np.float32))
            l0.append(np.asarray(scores.l0[keep], dtype=np.float32))
            l1.append(np.asarray(scores.l1[keep], dtype=np.float32))
            r1.append(np.asarray(scores.r1[keep], dtype=np.float32))
    references = (
        _sorted_reference(g0, require_values=True),
        _sorted_reference(l0, require_values=False),
        _sorted_reference(l1, require_values=False),
        _sorted_reference(r1, require_values=False),
    )
    return RungNormalReferences(*references)


def build_source_audit(
    conditions: Sequence[QueryLadderResult],
    masks: Sequence[BoolArray],
) -> SourceAudit:
    """Seal the shared rung population for clean, two thin, and broad conditions."""
    if len(conditions) != _EXPECTED_CONDITIONS or len(masks) != 3:
        raise ValueError("source audit requires clean, two thin, broad and three masks")
    query_ids = [result.query_id for result in conditions]
    if len(set(query_ids)) != len(query_ids) or any(not value for value in query_ids):
        raise ValueError("source condition query identities must be unique and non-empty")
    population = sha256_json(
        {
            "schema": "darc-gate2-source-population-v1",
            "conditions": [
                {
                    "query_id": result.query_id,
                    "population_sha256": result.audit.population_sha256,
                    "crop_shapes": [list(shape) for shape in result.crop_shapes],
                }
                for result in conditions
            ],
        },
    )
    support = sha256_json(
        {
            "schema": "darc-gate2-source-support-v1",
            "conditions": [
                {
                    "query_id": result.query_id,
                    "support_sha256": result.audit.support_sha256,
                    "selected_support_ids": list(result.selected_support_ids),
                }
                for result in conditions
            ],
        },
    )
    fallback = sha256_json(
        {
            "schema": "darc-gate2-source-fallback-v1",
            "conditions": [
                {
                    "query_id": result.query_id,
                    "fallback_sha256": result.audit.fallback_sha256,
                }
                for result in conditions
            ],
        },
    )
    mask = sha256_json(
        {
            "schema": "darc-gate2-source-mask-v1",
            "masks": [_array_manifest(value) for value in masks],
        },
    )
    return SourceAudit(population, support, fallback, mask)


def build_group_residual(
    population: GroupResidualPopulation,
    l0_parts: Sequence[FloatArray],
    l1_parts: Sequence[FloatArray],
) -> GroupResidual:
    """Compact one object-seed's fixed clean-token residual population."""
    count = len(population.source_ids)
    if (
        count == 0
        or len(population.fold_indices) != count
        or len(population.population_rows) != count
    ):
        raise ValueError("group residual identities and populations must have equal length")
    if len(l0_parts) != count or len(l1_parts) != count:
        raise ValueError("group residuals require one paired array per source")
    l0 = _concatenate_finite(l0_parts)
    l1 = _concatenate_finite(l1_parts)
    if l0.shape != l1.shape:
        raise ValueError("L0 and L1 group residual populations must have equal shape")
    population_sha256 = sha256_json(
        {
            "schema": "darc-gate2-group-token-population-v1",
            "object": population.object_name,
            "seed": population.seed,
            "rows": list(population.population_rows),
        },
    )
    return GroupResidual(
        object_name=population.object_name,
        seed=population.seed,
        source_ids=population.source_ids,
        fold_indices=population.fold_indices,
        l0_p999=higher_p999(np.asarray(l0, dtype=np.float64)),
        l1_p999=higher_p999(np.asarray(l1, dtype=np.float64)),
        l0_population_sha256=population_sha256,
        l1_population_sha256=population_sha256,
        l0_residual_sha256=_raw_float_sha256(l0),
        l1_residual_sha256=_raw_float_sha256(l1),
    )


def cue_seed(data_root: Path, object_name: str, seed: int, path: str, profile: int) -> int:
    relative = Path(path).relative_to(data_root).as_posix()
    payload = f"{LINE_CUE_VERSION}\0{object_name}\0{relative}\0{seed}\0{profile}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def relative_id(data_root: Path, path: str) -> str:
    return Path(path).relative_to(data_root).as_posix()


def _sorted_reference(
    parts: Sequence[FloatArray],
    *,
    require_values: bool,
) -> FloatArray:
    values = _concatenate_finite(parts, require_values=require_values)
    return np.asarray(np.sort(values), dtype=np.float32)


def _concatenate_finite(
    parts: Sequence[FloatArray],
    *,
    require_values: bool = True,
) -> FloatArray:
    if not parts:
        raise ValueError("residual population must be non-empty")
    arrays: Tuple[FloatArray, ...] = tuple(
        np.asarray(part, dtype=np.float32).reshape(-1) for part in parts
    )
    values: FloatArray = np.asarray(np.concatenate(arrays), dtype=np.float32)
    if (require_values and values.size == 0) or not bool(np.all(np.isfinite(values))):
        raise ValueError("residual population must be non-empty and finite")
    return values


def _raw_float_sha256(values: FloatArray) -> str:
    contiguous: FloatArray = np.asarray(
        np.ascontiguousarray(values, dtype=np.float32),
        dtype=np.float32,
    )
    digest = hashlib.sha256()
    digest.update(str(tuple(contiguous.shape)).encode())
    digest.update(b"\0float32\0")
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _array_manifest(values: BoolArray) -> Dict[str, JsonValue]:
    array = np.asarray(values)
    if array.dtype != np.dtype(np.bool_) or array.ndim != 2 or not bool(np.any(array)):
        raise ValueError("cue masks must be non-empty 2D boolean arrays")
    contiguous: BoolArray = np.asarray(
        np.ascontiguousarray(array, dtype=np.bool_),
        dtype=np.bool_,
    )
    return {
        "shape": list(contiguous.shape),
        "positive_count": int(np.count_nonzero(contiguous)),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }
