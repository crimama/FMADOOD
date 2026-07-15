from __future__ import annotations

from typing import Final, NamedTuple, Tuple

import numpy as np
import numpy.typing as npt

from flow_tte.darc_feature_stream import ImageFeatures
from flow_tte.darc_geometry import (
    FeatureGrid,
    Point2D,
    RealizedResize,
    SimilarityAlignment,
    SimilarityRansacConfig,
)
from flow_tte.darc_scoring import LocalCandidateSet

FloatArray = npt.NDArray[np.float32]
Float64Array = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]
FeatureArray = npt.NDArray[np.float16]

MAX_LOCAL_CANDIDATES: Final = 9
MATRIX_EPSILON: Final = 1e-12


class CorrespondenceError(ValueError):
    reason: str

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Invalid DARC Gate 2 correspondence input: {reason}")


class CorrespondenceConfig(NamedTuple):
    short_edge: int = 672
    patch_size: int = 16
    ransac: SimilarityRansacConfig = SimilarityRansacConfig()


class CandidateRequest(NamedTuple):
    crop_index: int
    start: int
    stop: int


class SelectedSupport(NamedTuple):
    support_id: str
    features: ImageFeatures


class CorrespondenceQuery(NamedTuple):
    query_id: str
    features: ImageFeatures


class CoarseGeometry(NamedTuple):
    resize: RealizedResize
    grid: FeatureGrid


class SupportRegistration(NamedTuple):
    support_id: str
    query_resize: RealizedResize
    support_resize: RealizedResize
    alignment: SimilarityAlignment
    query_to_support: FloatArray
    pair_count: int
    ransac_seed: int
    invertible: bool

    @property
    def accepted(self) -> bool:
        return self.alignment.accepted and self.invertible

    def map_query_native(self, point: Point2D) -> Point2D:
        query_resized = self.query_resize.native_to_resized(point)
        vector: FloatArray = np.asarray([query_resized.x, query_resized.y], dtype=np.float32)
        transformed: FloatArray = np.asarray(
            self.query_to_support[:, :2] @ vector + self.query_to_support[:, 2],
            dtype=np.float32,
        )
        support_resized = Point2D(
            x=float(np.sum(transformed[0:1])),
            y=float(np.sum(transformed[1:2])),
        )
        return self.support_resize.resized_to_native(support_resized)


class PreparedCorrespondence(NamedTuple):
    query_id: str
    query: ImageFeatures
    supports: Tuple[SelectedSupport, ...]
    registrations: Tuple[SupportRegistration, ...]
    feature_dimension: int


class CandidateLocation(NamedTuple):
    crop_index: int
    token_indices: IntArray


class TokenPointChunk(NamedTuple):
    request: CandidateRequest
    points: FloatArray


class CandidateAudit(NamedTuple):
    request: CandidateRequest
    ordered_support_ids: Tuple[str, ...]
    alignment_accepted: BoolArray
    mapped_support_points: FloatArray
    candidate_counts: IntArray
    support_valid: BoolArray


class CandidateChunk(NamedTuple):
    local: LocalCandidateSet
    audit: CandidateAudit


DEFAULT_CORRESPONDENCE_CONFIG: Final = CorrespondenceConfig()
