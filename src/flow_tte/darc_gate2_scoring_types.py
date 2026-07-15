from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

from flow_tte.darc_scoring import LocalCandidateSet

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)  # noqa: SLOTS_OK -- Python 3.8 has no dataclass slots.
class RungScoringInput:
    query: FloatArray
    global_memory: FloatArray
    identity_candidates: LocalCandidateSet
    aligned_candidates: LocalCandidateSet
    precomputed_g0: Optional[FloatArray] = None


@dataclass(frozen=True)  # noqa: SLOTS_OK -- Python 3.8 has no dataclass slots.
class RungScoringConfig:
    g0_chunk_size: int
    r1_chunk_size: int
    complete_g0: bool = True
    device: str = "cpu"


@dataclass(frozen=True)  # noqa: SLOTS_OK -- Python 3.8 has no dataclass slots.
class SupportValidityAudit:
    l0: BoolArray
    l1: BoolArray
    shared: BoolArray
    r1: BoolArray


@dataclass(frozen=True)  # noqa: SLOTS_OK -- Python 3.8 has no dataclass slots.
class RungScores:
    g0: FloatArray
    g0_valid: BoolArray
    l0: FloatArray
    l1: FloatArray
    r1: FloatArray
    common_fallback: BoolArray
    support_validity: SupportValidityAudit


@dataclass(frozen=True)  # noqa: SLOTS_OK -- Python 3.8 has no dataclass slots.
class RungNormalReferences:
    g0: FloatArray
    l0: FloatArray
    l1: FloatArray
    r1: FloatArray


@dataclass(frozen=True)  # noqa: SLOTS_OK -- Python 3.8 has no dataclass slots.
class RungEvidence:
    g0: FloatArray
    g0_valid: BoolArray
    l0: FloatArray
    l1: FloatArray
    r1: FloatArray
