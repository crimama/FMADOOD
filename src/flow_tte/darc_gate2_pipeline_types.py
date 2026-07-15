from __future__ import annotations

from typing import Mapping, NamedTuple, Tuple

import numpy as np
import numpy.typing as npt

from flow_tte.darc_feature_stream import ImageFeatures
from flow_tte.darc_gate2_correspondence_types import CorrespondenceConfig, SupportRegistration
from flow_tte.darc_gate2_scoring_types import RungScores
from flow_tte.darc_geometry import ImageSize
from flow_tte.darc_knn import ChunkedKnnConfig
from flow_tte.darc_tiling import NativeCrop

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


class Gate2PipelineError(ValueError):
    reason: str

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Invalid DARC Gate 2 query pipeline input: {reason}")


class QueryLadderInput(NamedTuple):
    query_id: str
    query: ImageFeatures
    candidates: Mapping[str, ImageFeatures]
    knn_config: ChunkedKnnConfig


class QueryPipelineConfig(NamedTuple):
    correspondence: CorrespondenceConfig = CorrespondenceConfig()
    complete_g0: bool = True


class CropLadderResult(NamedTuple):
    crop_index: int
    crop: NativeCrop
    token_shape: ImageSize
    scores: RungScores


class QueryLadderAudit(NamedTuple):
    population_sha256: str
    support_sha256: str
    fallback_sha256: str


class QueryLadderResult(NamedTuple):
    query_id: str
    native_size: ImageSize
    selected_support_ids: Tuple[str, ...]
    crops: Tuple[CropLadderResult, ...]
    registration_audit: Tuple[SupportRegistration, ...]
    audit: QueryLadderAudit

    @property
    def crop_shapes(self) -> Tuple[ImageSize, ...]:
        return tuple(item.token_shape for item in self.crops)

    def concatenate_l0_residuals(self) -> FloatArray:
        return np.asarray(
            np.concatenate(tuple(item.scores.l0 for item in self.crops)),
            dtype=np.float32,
        )

    def concatenate_l1_residuals(self) -> FloatArray:
        return np.asarray(
            np.concatenate(tuple(item.scores.l1 for item in self.crops)),
            dtype=np.float32,
        )

    def concatenate_r1_residuals(self) -> FloatArray:
        return np.asarray(
            np.concatenate(tuple(item.scores.r1 for item in self.crops)),
            dtype=np.float32,
        )

    def concatenate_fallback_mask(self) -> BoolArray:
        return np.asarray(
            np.concatenate(tuple(item.scores.common_fallback for item in self.crops)),
            dtype=np.bool_,
        )


class QueryEvidenceMaps(NamedTuple):
    l0: FloatArray
    l1: FloatArray
    r1: FloatArray
