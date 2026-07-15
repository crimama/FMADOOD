"""High-resolution learned normal-density head for the DARC fine branch.

This module isolates the preregistered fine branch (see
``skill_graph/analysis/2026-07-11_flowtte_hres_density_fusion_preregistration.md``):
a normal-only density head over native high-resolution DINOv3 layer-7 tokens whose
per-token negative log-likelihood becomes the fine evidence ``e_micro``. It reuses
the existing ``FlowDensityEstimator`` and the DARC native-grid stitching so no
hard coordinate-local window, registration, or reconstruction is involved.

The head is trained on the fold's memory normal tokens only. Held-out (calibration)
normals are used exclusively to build the normal-only upper-tail reference used to
calibrate the density evidence. No anomalous pixel and no AD2 label participates in
fitting the head or the calibration reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from flow_tte.config import FlowConfig
from flow_tte.darc_feature_stream import FloatArray, ImageFeatures, stitch_score_grids
from flow_tte.darc_knn import ChunkedKnnConfig, cosine_1nn_scores
from flow_tte.darc_scoring import upper_tail_evidence
from flow_tte.trainer import FlowDensityEstimator


class HresDensityError(ValueError):
    """Raised when high-resolution density inputs violate a frozen invariant."""

    reason: str

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Invalid high-resolution density input: {reason}")


@dataclass(frozen=True)
class HresDensityConfig:
    """Frozen training and sampling controls for the fine density head."""

    flow: FlowConfig = field(default_factory=FlowConfig)
    density_quantile: float = 0.9
    train_sample_cap: int = 131072

    def __post_init__(self) -> None:
        if not 0.0 < self.density_quantile < 1.0:
            raise HresDensityError("density_quantile must be in (0, 1)")
        if self.train_sample_cap < 2:
            raise HresDensityError("train_sample_cap must be at least 2")


def high_token_matrix(features: ImageFeatures) -> FloatArray:
    """Concatenate every native high-resolution crop token into one (N, D) matrix."""
    grids = features.high
    if not grids:
        raise HresDensityError("image features contain no high-resolution crops")
    rows: List[FloatArray] = []
    dim = int(grids[0].shape[-1])
    for grid in grids:
        array = np.asarray(grid, dtype=np.float32)
        if array.ndim != 3 or array.shape[-1] != dim:
            raise HresDensityError("high crop grids must share one HxWxD layout")
        rows.append(array.reshape(-1, dim))
    return np.asarray(np.concatenate(rows, axis=0), dtype=np.float32)


def _deterministic_sample(matrix: FloatArray, cap: int) -> FloatArray:
    if matrix.shape[0] <= cap:
        return matrix
    indices = np.linspace(0, matrix.shape[0] - 1, num=cap).round().astype(np.int64)
    return np.asarray(matrix[indices], dtype=np.float32)


@dataclass(frozen=True)
class HresDensityHead:
    """A trained normal-only density head plus its normal upper-tail reference."""

    estimator: FlowDensityEstimator
    reference_nll: FloatArray

    def token_nll_grids(self, features: ImageFeatures) -> Tuple[FloatArray, ...]:
        """Return per-crop (H, W) negative-log-likelihood grids for a query image."""
        grids: List[FloatArray] = []
        for grid in features.high:
            array = np.asarray(grid, dtype=np.float32)
            height, width = int(array.shape[0]), int(array.shape[1])
            tokens = array.reshape(height * width, array.shape[-1])
            nll = self.estimator.nll(tokens).detach().cpu().numpy()
            grids.append(np.asarray(nll.reshape(height, width), dtype=np.float32))
        return tuple(grids)

    def density_native_map(self, features: ImageFeatures) -> FloatArray:
        """Stitch the raw per-token NLL onto the query's native image grid."""
        return stitch_score_grids(
            features.native_size,
            features.crops,
            self.token_nll_grids(features),
        )

    def calibrated_native_map(self, features: ImageFeatures) -> FloatArray:
        """Stitch the normal-only upper-tail calibrated evidence onto the native grid."""
        grids: List[FloatArray] = []
        for grid in self.token_nll_grids(features):
            height, width = int(grid.shape[0]), int(grid.shape[1])
            evidence = upper_tail_evidence(self.reference_nll, grid.reshape(-1))
            grids.append(np.asarray(evidence.reshape(height, width), dtype=np.float32))
        return stitch_score_grids(features.native_size, features.crops, tuple(grids))


def _flatten_nll(estimator: FlowDensityEstimator, features: ImageFeatures) -> FloatArray:
    matrix = high_token_matrix(features)
    nll = estimator.nll(matrix).detach().cpu().numpy()
    return np.asarray(nll, dtype=np.float32)


def train_density_head(
    memory: Sequence[ImageFeatures],
    heldout: Sequence[ImageFeatures],
    config: HresDensityConfig,
    device: str,
) -> HresDensityHead:
    """Fit the density head on memory normals and build the held-out normal reference.

    ``memory`` supplies the tokens the flow is trained on. ``heldout`` supplies the
    disjoint normal images whose token NLLs form the upper-tail calibration
    reference. The two populations must not overlap; that disjointness is the
    caller's responsibility (the P16 fold split guarantees it).
    """
    if not memory:
        raise HresDensityError("memory must contain at least one normal image")
    if not heldout:
        raise HresDensityError("held-out normals must contain at least one image")
    train_matrix = _deterministic_sample(
        np.asarray(
            np.concatenate([high_token_matrix(item) for item in memory], axis=0),
            dtype=np.float32,
        ),
        config.train_sample_cap,
    )
    dim = int(train_matrix.shape[1])
    estimator = FlowDensityEstimator(dim=dim, config=config.flow, device=device)
    _ = estimator.fit(train_matrix, density_quantile=config.density_quantile)
    reference = np.asarray(
        np.concatenate([_flatten_nll(estimator, item) for item in heldout], axis=0),
        dtype=np.float32,
    )
    return HresDensityHead(estimator=estimator, reference_nll=reference)


def g0_native_map(
    query: ImageFeatures,
    memory: Sequence[ImageFeatures],
    knn_config: ChunkedKnnConfig,
) -> FloatArray:
    """Same-memory global 1-NN cosine residual map for a raw-vs-learned comparison."""
    if not memory:
        raise HresDensityError("memory must contain at least one normal image")
    pool = np.asarray(
        np.concatenate([high_token_matrix(item) for item in memory], axis=0),
        dtype=np.float32,
    )
    grids: List[FloatArray] = []
    for grid in query.high:
        array = np.asarray(grid, dtype=np.float32)
        height, width = int(array.shape[0]), int(array.shape[1])
        tokens = array.reshape(height * width, array.shape[-1])
        residual = cosine_1nn_scores(tokens, pool, knn_config)
        grids.append(np.asarray(residual.reshape(height, width), dtype=np.float32))
    return stitch_score_grids(query.native_size, query.crops, tuple(grids))
