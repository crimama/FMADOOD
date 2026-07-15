from __future__ import annotations

from hashlib import sha256
from typing import Tuple

import numpy as np

from flow_tte.darc_gate2_pipeline_types import (
    BoolArray,
    CropLadderResult,
    QueryLadderAudit,
)


def query_ladder_audit(
    query_id: str,
    selected_ids: Tuple[str, ...],
    crops: Tuple[CropLadderResult, ...],
) -> QueryLadderAudit:
    population = bytearray(b"darc-gate2-population-v1\0")
    _extend_text(population, query_id)
    support = bytearray(b"darc-gate2-support-v1\0")
    for support_id in selected_ids:
        _extend_text(support, support_id)
    fallback = bytearray(b"darc-gate2-fallback-v1\0")
    for crop in crops:
        _extend_ints(
            population,
            (
                crop.crop_index,
                crop.crop.y0,
                crop.crop.x0,
                crop.crop.height,
                crop.crop.width,
                crop.token_shape.height,
                crop.token_shape.width,
            ),
        )
        shared: BoolArray = np.asarray(crop.scores.support_validity.shared, dtype=np.bool_)
        shared_shape = (int(np.size(shared, axis=0)), int(np.size(shared, axis=1)))
        _extend_ints(support, shared_shape)
        support.extend(shared.astype(np.uint8, copy=False).tobytes())
        crop_fallback: BoolArray = np.asarray(crop.scores.common_fallback, dtype=np.bool_)
        _extend_ints(fallback, (crop.crop_index, len(crop_fallback)))
        fallback.extend(crop_fallback.astype(np.uint8, copy=False).tobytes())
    return QueryLadderAudit(
        population_sha256=sha256(population).hexdigest(),
        support_sha256=sha256(support).hexdigest(),
        fallback_sha256=sha256(fallback).hexdigest(),
    )


def _extend_text(payload: bytearray, value: str) -> None:
    encoded = value.encode("utf-8")
    payload.extend(len(encoded).to_bytes(8, byteorder="big", signed=False))
    payload.extend(encoded)


def _extend_ints(payload: bytearray, values: Tuple[int, ...]) -> None:
    for value in values:
        payload.extend(value.to_bytes(8, byteorder="big", signed=True))
