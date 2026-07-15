"""Runtime value objects for the frozen DARC Gate 2 experiment."""

from __future__ import annotations

# pyright: reportMissingImports=false
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

from flow_tte.darc_gate2_artifacts import CompletionExpectation
from flow_tte.darc_resources import P16Split

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class Gate2RuntimeConfig:
    data_root: Path
    output_root: Path
    object_name: str
    device: str
    seeds: Tuple[int, ...]
    code_config_sha256: str
    smoke: bool = False
    candidate_chunk_size: int = 1024
    query_chunk_size: int = 1024
    memory_chunk_size: int = 262144

    def __post_init__(self) -> None:
        if not self.object_name or not self.seeds:
            raise ValueError("Gate 2 object and seeds must be non-empty")
        if (
            min(
                self.candidate_chunk_size,
                self.query_chunk_size,
                self.memory_chunk_size,
            )
            <= 0
        ):
            raise ValueError("Gate 2 chunk sizes must be positive")


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class PreparedSeedRun:
    split: P16Split
    expectation: CompletionExpectation


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class PreparedGate2Run:
    seeds: Tuple[PreparedSeedRun, ...]
    pending: Tuple[PreparedSeedRun, ...]
    method_sha256: str


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class SeedRunReport:
    object_name: str
    seed: int
    source_count: int
    smoke: bool
