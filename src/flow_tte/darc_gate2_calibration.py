from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
import numpy as np
import numpy.typing as npt

from flow_tte.darc_gate2_scoring_types import (
    RungEvidence,
    RungNormalReferences,
    RungScores,
)
from flow_tte.darc_scoring import upper_tail_evidence
from flow_tte.metrics import MetricInputError as ScoringInputError

FloatArray = npt.NDArray[np.float32]


def calibrate_rung_evidence(
    scores: RungScores,
    references: RungNormalReferences,
) -> RungEvidence:
    """Apply local normal tails only where the shared G0 fallback is inactive."""
    score_arrays = (scores.g0_valid, scores.l0, scores.l1, scores.r1, scores.common_fallback)
    if scores.g0.ndim != 1 or any(values.shape != scores.g0.shape for values in score_arrays):
        raise ScoringInputError("rung score arrays must have identical shapes")
    if np.any(scores.common_fallback & ~scores.g0_valid):
        raise ScoringInputError("fallback tokens require valid G0 residuals")

    g0_reference: FloatArray = np.asarray(references.g0, dtype=np.float32).reshape(-1)
    if g0_reference.size == 0 or not np.all(np.isfinite(g0_reference)):
        raise ScoringInputError("G0 reference must be non-empty and finite")

    g0: FloatArray = np.zeros(scores.g0.shape, dtype=np.float32)
    if np.any(scores.g0_valid):
        g0[scores.g0_valid] = upper_tail_evidence(g0_reference, scores.g0[scores.g0_valid])

    fallback = scores.common_fallback
    local = np.asarray(~fallback, dtype=np.bool_)
    if np.any(local) and any(
        reference.size == 0 for reference in (references.l0, references.l1, references.r1)
    ):
        raise ScoringInputError("nonfallback tokens require local references")

    l0: FloatArray = np.asarray(g0.copy(), dtype=np.float32)
    l1: FloatArray = np.asarray(g0.copy(), dtype=np.float32)
    r1: FloatArray = np.asarray(g0.copy(), dtype=np.float32)
    if np.any(local):
        l0[local] = upper_tail_evidence(references.l0, scores.l0[local])
        l1[local] = upper_tail_evidence(references.l1, scores.l1[local])
        r1[local] = upper_tail_evidence(references.r1, scores.r1[local])

    return RungEvidence(
        g0=g0,
        g0_valid=np.asarray(scores.g0_valid, dtype=np.bool_),
        l0=l0,
        l1=l1,
        r1=r1,
    )
