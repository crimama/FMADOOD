"""Filesystem boundaries for the DARC AD2 raw-ladder pilot."""

from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 -- NamedTuple fields expose Path at runtime
from typing import Final, List, NamedTuple, Optional, Tuple

import numpy as np
import tifffile as tiff

from flow_tte.darc_ad2_pilot import RawRungMaps
from flow_tte.darc_feature_stream import FloatArray
from flow_tte.darc_map_io import Population

_IMAGE_SUFFIXES: Final = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})


class PilotIoError(ValueError):
    reason: str

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Invalid DARC AD2 pilot I/O: {reason}")


# Python 3.8 dataclasses do not support slots=True.
@dataclass(frozen=True)
class TestLimits:
    good: Optional[int] = None
    bad: Optional[int] = None

    def __post_init__(self) -> None:
        if any(value is not None and value < 1 for value in (self.good, self.bad)):
            raise PilotIoError("test limits must be positive when provided")


class PilotTestImage(NamedTuple):
    population: Population
    path: Path


class PilotMapTarget(NamedTuple):
    output_root: Path
    object_name: str


_DEFAULT_TEST_LIMITS: Final = TestLimits()


def discover_test_images(
    data_root: Path,
    object_name: str,
    limits: TestLimits = _DEFAULT_TEST_LIMITS,
) -> Tuple[PilotTestImage, ...]:
    """Return deterministic good-then-bad public test images without ground truth access."""
    limits_by_population = {Population.GOOD: limits.good, Population.BAD: limits.bad}
    images: List[PilotTestImage] = []
    for population in Population:
        directory = data_root / object_name / "test_public" / population.value
        if not directory.is_dir():
            reason = f"missing test population directory: {directory}"
            raise PilotIoError(reason)
        paths = tuple(
            sorted(
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
            ),
        )
        limit = limits_by_population[population]
        selected = paths if limit is None else paths[:limit]
        images.extend(PilotTestImage(population, path) for path in selected)
    if not images:
        reason = f"no public test images found for {object_name}"
        raise PilotIoError(reason)
    return tuple(images)


def write_rung_maps(
    target: PilotMapTarget,
    image: PilotTestImage,
    maps: RawRungMaps,
) -> None:
    """Write one complete four-arm image bundle in common-evaluator layout."""
    arms: Tuple[Tuple[str, FloatArray], ...] = (
        ("G0", maps.g0),
        ("L0", maps.l0),
        ("L1", maps.l1),
        ("R1", maps.r1),
    )
    for arm, values in arms:
        directory = (
            target.output_root
            / "arms"
            / arm
            / "anomaly_maps"
            / target.object_name
            / "test"
            / image.population.value
        )
        directory.mkdir(parents=True, exist_ok=True)
        tiff.imwrite(
            str(directory / f"{image.path.stem}.tiff"),
            np.asarray(values, dtype=np.float32),
        )
