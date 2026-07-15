"""Strict loading and native-grid normalization for common DARC evaluation."""

from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Final, Mapping, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
from typing_extensions import override

if TYPE_CHECKING:
    from pathlib import Path

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
UInt8Array = npt.NDArray[np.uint8]
Shape = Tuple[int, int]
_IMAGE_SUFFIXES: Final = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})


class Population(str, Enum):
    GOOD = "good"
    BAD = "bad"


@dataclass(frozen=True)  # noqa: RUF100 -- Python 3.8; # noqa: SLOTS_OK
class MapInputError(ValueError):
    reason: str

    @override
    def __str__(self) -> str:
        return f"Invalid common-evaluator input: {self.reason}"


@dataclass(frozen=True)  # noqa: RUF100 -- Python 3.8; # noqa: SLOTS_OK
class ImageAudit:
    image_id: str
    population: Population
    map_path: Path
    source_path: Path
    gt_path: Optional[Path]
    map_sha256: str
    source_sha256: str
    gt_sha256: Optional[str]
    original_map_shape: Shape
    original_gt_shape: Optional[Shape]
    common_shape: Shape


@dataclass(frozen=True)  # noqa: RUF100 -- Python 3.8; # noqa: SLOTS_OK
class ImageRecord:
    audit: ImageAudit
    score_map: FloatArray
    gt_mask: BoolArray


@dataclass(frozen=True)  # noqa: RUF100 -- Python 3.8; # noqa: SLOTS_OK
class ObjectMapSet:
    object_name: str
    records: Tuple[ImageRecord, ...]


@dataclass(frozen=True)  # noqa: RUF100 -- Python 3.8; # noqa: SLOTS_OK
class ObjectAuditSet:
    object_name: str
    images: Tuple[ImageAudit, ...]


def audit_only(maps: ObjectMapSet) -> ObjectAuditSet:
    """Drop score and mask arrays before retaining a completed object's provenance."""
    return ObjectAuditSet(
        object_name=maps.object_name,
        images=tuple(record.audit for record in maps.records),
    )


def load_object_maps(
    data_root: Path,
    map_roots: Tuple[Path, ...],
    object_name: str,
) -> ObjectMapSet:
    """Load one complete object population and normalize scores to native RGB grids."""
    if not map_roots:
        raise MapInputError("at least one map root is required")
    source_paths = _discover_sources(data_root, object_name)
    map_paths = _discover_maps(map_roots, object_name)
    missing = sorted(set(source_paths) - set(map_paths))
    unexpected = sorted(set(map_paths) - set(source_paths))
    if missing:
        reason = f"missing map IDs: {', '.join(missing)}"
        raise MapInputError(reason)
    if unexpected:
        reason = f"unexpected map IDs: {', '.join(unexpected)}"
        raise MapInputError(reason)
    ordered_ids = sorted(source_paths, key=lambda item: (not item.startswith("good/"), item))
    records = tuple(
        _load_record(data_root, object_name, image_id, source_paths[image_id], map_paths[image_id])
        for image_id in ordered_ids
    )
    if not records:
        reason = f"no test images found for object {object_name}"
        raise MapInputError(reason)
    return ObjectMapSet(object_name=object_name, records=records)


def _discover_sources(data_root: Path, object_name: str) -> Mapping[str, Path]:
    found: dict[str, Path] = {}
    test_root = data_root / object_name / "test_public"
    for population in Population:
        directory = test_root / population.value
        if not directory.is_dir():
            reason = f"missing source directory: {directory}"
            raise MapInputError(reason)
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            image_id = f"{population.value}/{path.stem}"
            if image_id in found:
                reason = f"duplicate source ID: {image_id}"
                raise MapInputError(reason)
            found[image_id] = path
    return found


def _discover_maps(map_roots: Tuple[Path, ...], object_name: str) -> Mapping[str, Path]:
    found: dict[str, Path] = {}
    for root in map_roots:
        if not root.is_dir():
            reason = f"map root is not a directory: {root}"
            raise MapInputError(reason)
        anomaly_roots = ([root] if root.name == "anomaly_maps" else []) + list(
            root.rglob("anomaly_maps"),
        )
        for anomaly_root in anomaly_roots:
            for population in Population:
                directory = anomaly_root / object_name / "test" / population.value
                for path in sorted(directory.glob("*.tiff")):
                    image_id = f"{population.value}/{path.stem}"
                    if image_id in found:
                        reason = f"duplicate map ID {image_id}: {found[image_id]} and {path}"
                        raise MapInputError(reason)
                    found[image_id] = path
    return found


def _load_record(
    data_root: Path,
    object_name: str,
    image_id: str,
    source_path: Path,
    map_path: Path,
) -> ImageRecord:
    population = Population(image_id.split("/", maxsplit=1)[0])
    raw_source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    raw_score = cv2.imread(str(map_path), cv2.IMREAD_UNCHANGED)
    if raw_source is None or raw_score is None:
        reason = f"failed to decode source or score map for {image_id}"
        raise MapInputError(reason)
    source: UInt8Array = np.asarray(raw_source, dtype=np.uint8)
    score_array = np.asarray(raw_score)
    if score_array.ndim != 2 or not np.all(np.isfinite(score_array)):
        reason = f"score map must be finite and 2D for {image_id}"
        raise MapInputError(reason)
    score: FloatArray = np.asarray(score_array, dtype=np.float32)
    source_height = len(source)
    score_height = len(score)
    common_shape = (source_height, source.size // (source_height * 3))
    original_map_shape = (score_height, score.size // score_height)
    normalized: FloatArray = np.asarray(
        cv2.resize(score.astype(np.float32), common_shape[::-1], interpolation=cv2.INTER_LINEAR),
        dtype=np.float32,
    )
    gt_path: Optional[Path] = None
    gt_sha256: Optional[str] = None
    original_gt_shape: Optional[Shape] = None
    mask = np.zeros(common_shape, dtype=np.bool_)
    if population is Population.BAD:
        stem = image_id.split("/", maxsplit=1)[1]
        gt_path = (
            data_root / object_name / "test_public" / "ground_truth" / "bad" / f"{stem}_mask.png"
        )
        raw_gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if raw_gt is None:
            reason = f"failed to decode ground truth for {image_id}: {gt_path}"
            raise MapInputError(reason)
        gt_array: UInt8Array = np.asarray(raw_gt, dtype=np.uint8)
        gt_height = len(gt_array)
        original_gt_shape = (gt_height, gt_array.size // gt_height)
        resized_gt: UInt8Array = np.asarray(
            cv2.resize(gt_array, common_shape[::-1], interpolation=cv2.INTER_NEAREST),
            dtype=np.uint8,
        )
        mask = np.asarray(resized_gt > 0, dtype=np.bool_)
        if not np.any(mask):
            reason = f"bad image has an empty ground truth: {image_id}"
            raise MapInputError(reason)
        gt_sha256 = _sha256(gt_path)
    audit = ImageAudit(
        image_id=image_id,
        population=population,
        map_path=map_path,
        source_path=source_path,
        gt_path=gt_path,
        map_sha256=_sha256(map_path),
        source_sha256=_sha256(source_path),
        gt_sha256=gt_sha256,
        original_map_shape=original_map_shape,
        original_gt_shape=original_gt_shape,
        common_shape=common_shape,
    )
    return ImageRecord(audit=audit, score_map=normalized, gt_mask=mask)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
