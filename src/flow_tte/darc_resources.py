"""Deterministic resource partitions for DARC experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, NamedTuple, Sequence, Tuple, TypedDict, cast

import numpy as np
from typing_extensions import final, override

if TYPE_CHECKING:
    from pathlib import Path

P16_PROTOCOL_VERSION: Final = "darc-p16-v1"
P16_SEEDS: Final = (0, 1, 2)
_SUPPORT_COUNT: Final = 16
_CALIBRATION_COUNT: Final = 4


class P16FoldManifest(TypedDict):
    fold_index: int
    memory_paths: Tuple[str, ...]
    calibration_paths: Tuple[str, ...]


class P16SplitManifest(TypedDict):
    version: str
    seed: int
    source_pool_count: int
    support_paths: Tuple[str, ...]
    folds: Tuple[P16FoldManifest, ...]


@final
class DarcResourceError(ValueError):
    __slots__ = ("field", "reason")

    field: str
    reason: str

    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(str(self))

    @override
    def __str__(self) -> str:
        return f"Invalid {self.field}: {self.reason}"


class P16Fold(NamedTuple):
    fold_index: int
    memory_paths: Tuple[str, ...]
    calibration_paths: Tuple[str, ...]

    def to_manifest(self) -> P16FoldManifest:
        return {
            "fold_index": self.fold_index,
            "memory_paths": self.memory_paths,
            "calibration_paths": self.calibration_paths,
        }


class P16Split(NamedTuple):
    version: str
    seed: int
    source_pool_count: int
    support_paths: Tuple[str, ...]
    folds: Tuple[P16Fold, ...]

    def to_manifest(self) -> P16SplitManifest:
        return {
            "version": self.version,
            "seed": self.seed,
            "source_pool_count": self.source_pool_count,
            "support_paths": self.support_paths,
            "folds": tuple(fold.to_manifest() for fold in self.folds),
        }


def build_p16_split(paths: Sequence[Path], seed: int) -> P16Split:
    """Expose exactly 16 uniformly selected normals before feature loading."""
    if seed not in P16_SEEDS:
        raise DarcResourceError("seed", f"must be one of {P16_SEEDS}")
    source_paths = tuple(sorted(str(path) for path in paths))
    if len(source_paths) < _SUPPORT_COUNT:
        raise DarcResourceError("paths", "must contain at least 16 normal images")
    if len(set(source_paths)) != len(source_paths):
        raise DarcResourceError("paths", "must contain unique image paths")

    permutation = np.random.Generator(np.random.PCG64(seed)).permutation(
        len(source_paths),
    )
    support_paths = tuple(
        source_paths[int(cast("np.int64", permutation[position]))]
        for position in range(_SUPPORT_COUNT)
    )
    folds = tuple(
        P16Fold(
            fold_index=fold_index,
            memory_paths=support_paths[:start] + support_paths[start + _CALIBRATION_COUNT :],
            calibration_paths=support_paths[start : start + _CALIBRATION_COUNT],
        )
        for fold_index in range(4)
        for start in (fold_index * _CALIBRATION_COUNT,)
    )
    return P16Split(
        version=P16_PROTOCOL_VERSION,
        seed=seed,
        source_pool_count=len(source_paths),
        support_paths=support_paths,
        folds=folds,
    )
