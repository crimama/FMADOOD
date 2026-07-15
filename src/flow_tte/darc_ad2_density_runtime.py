"""Path-frozen runtime for the DARC AD2 fine-branch density cell.

This mirrors the raw-ladder pilot runtime but replaces the hard-local/registration
ladder with the preregistered fine branch: a normal-only high-resolution density
head. It emits three same-memory arms per query so the learned head can be
compared against the raw cosine residual on identical folds and tokens:

- ``G0``  raw global 1-NN cosine residual (the strongest raw arm from the pilot);
- ``D0``  raw learned density negative log-likelihood;
- ``D0c`` normal-only upper-tail calibrated density evidence.

The head is trained once per fold on that fold's 12 memory normals; the 4 held-out
normals build the calibration reference only. No AD2 label is used at any stage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, NamedTuple, Protocol, Tuple

import numpy as np
import tifffile as tiff
from PIL import Image
from typing_extensions import final, override

from flow_tte.darc_ad2_pilot_io import (
    PilotMapTarget,
    PilotTestImage,
    TestLimits,
    discover_test_images,
)
from flow_tte.darc_feature_stream import FloatArray, ImageFeatures, RgbArray
from flow_tte.darc_knn import ChunkedKnnConfig
from flow_tte.darc_resources import P16Fold, P16Split, build_p16_split
from flow_tte.hres_density import (
    HresDensityConfig,
    HresDensityHead,
    g0_native_map,
    train_density_head,
)

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})


class DensityStream(Protocol):
    def extract(self, image: RgbArray) -> ImageFeatures: ...


@final
class DensityRuntimeError(ValueError):
    reason: str

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(str(self))

    @override
    def __str__(self) -> str:
        return f"Invalid DARC AD2 density runtime: {self.reason}"


@dataclass(frozen=True)
class DensityRuntimeConfig:
    data_root: Path
    output_root: Path
    object_name: str
    device: str
    seed: int
    fold_indices: Tuple[int, ...]
    density: HresDensityConfig = field(default_factory=HresDensityConfig)
    test_limits: TestLimits = field(default_factory=TestLimits)
    query_chunk_size: int = 256
    memory_chunk_size: int = 16384
    shard_index: int = 0
    shard_count: int = 1

    def __post_init__(self) -> None:
        _validate_config(self)


class PreparedDensity(NamedTuple):
    split: P16Split
    folds: Tuple[P16Fold, ...]
    test_images: Tuple[PilotTestImage, ...]


class DensityArmMaps(NamedTuple):
    g0: FloatArray
    d0: FloatArray
    d0c: FloatArray


class DensityRunReport(NamedTuple):
    object_name: str
    seed: int
    fold_count: int
    image_count: int


def claim_fresh_output_root(output_root: Path) -> None:
    try:
        output_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        message = f"output root must not already exist: {output_root}"
        raise DensityRuntimeError(message) from error


def prepare_density(config: DensityRuntimeConfig) -> PreparedDensity:
    _validate_config(config)
    train_directory = config.data_root / config.object_name / "train" / "good"
    if not train_directory.is_dir():
        message = f"missing normal training directory: {train_directory}"
        raise DensityRuntimeError(message)
    normal_paths = tuple(
        sorted(
            path
            for path in train_directory.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        ),
    )
    split = build_p16_split(normal_paths, config.seed)
    folds = tuple(split.folds[index] for index in config.fold_indices)
    all_test_images = discover_test_images(
        config.data_root,
        config.object_name,
        config.test_limits,
    )
    test_images = tuple(
        image
        for query_index, image in enumerate(all_test_images)
        if query_index % config.shard_count == config.shard_index
    )
    if not test_images:
        raise DensityRuntimeError("operational shard contains no public test queries")
    return PreparedDensity(split=split, folds=folds, test_images=test_images)


def _read_rgb(path: Path) -> RgbArray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _mean_arm(values: Tuple[FloatArray, ...]) -> FloatArray:
    shape = values[0].shape
    if any(value.shape != shape for value in values):
        raise DensityRuntimeError("fold maps must have identical shapes")
    stacked = np.asarray(np.stack(values, axis=0), dtype=np.float64)
    return np.asarray(np.mean(stacked, axis=0, dtype=np.float64), dtype=np.float32)


def _write_arms(target: PilotMapTarget, image: PilotTestImage, maps: DensityArmMaps) -> None:
    arms: Tuple[Tuple[str, FloatArray], ...] = (
        ("G0", maps.g0),
        ("D0", maps.d0),
        ("D0c", maps.d0c),
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


def run_density_cell(
    config: DensityRuntimeConfig,
    prepared: PreparedDensity,
    stream: DensityStream,
) -> DensityRunReport:
    _validate_config(config)
    if prepared.split.seed != config.seed:
        raise DensityRuntimeError("prepared split seed does not match the runtime config")
    if tuple(fold.fold_index for fold in prepared.folds) != tuple(config.fold_indices):
        raise DensityRuntimeError("prepared folds do not match the runtime config")

    cache: Dict[str, ImageFeatures] = {}
    for support_index, path_text in enumerate(prepared.split.support_paths, start=1):
        cache[path_text] = stream.extract(_read_rgb(Path(path_text)))
        _progress("support_cached", {"index": support_index, "total": 16})
    if len(cache) != 16:
        raise DensityRuntimeError("prepared P16 cache must contain exactly 16 supports")

    knn = ChunkedKnnConfig(
        device=config.device,
        query_chunk_size=config.query_chunk_size,
        memory_chunk_size=config.memory_chunk_size,
    )
    heads: List[HresDensityHead] = []
    memories: List[Tuple[ImageFeatures, ...]] = []
    for fold in prepared.folds:
        memory = tuple(cache[path] for path in fold.memory_paths)
        heldout = tuple(cache[path] for path in fold.calibration_paths)
        heads.append(train_density_head(memory, heldout, config.density, config.device))
        memories.append(memory)
        _progress("fold_trained", {"fold_index": fold.fold_index})

    target = PilotMapTarget(config.output_root, config.object_name)
    for image_index, image in enumerate(prepared.test_images, start=1):
        query = stream.extract(_read_rgb(image.path))
        g0_maps: List[FloatArray] = []
        d0_maps: List[FloatArray] = []
        d0c_maps: List[FloatArray] = []
        for head, memory in zip(heads, memories):
            g0_maps.append(g0_native_map(query, memory, knn))
            d0_maps.append(head.density_native_map(query))
            d0c_maps.append(head.calibrated_native_map(query))
        _write_arms(
            target,
            image,
            DensityArmMaps(
                g0=_mean_arm(tuple(g0_maps)),
                d0=_mean_arm(tuple(d0_maps)),
                d0c=_mean_arm(tuple(d0c_maps)),
            ),
        )
        _progress(
            "query_complete",
            {
                "index": image_index,
                "total": len(prepared.test_images),
                "population": image.population.value,
                "image": image.path.name,
            },
        )
    return DensityRunReport(
        object_name=config.object_name,
        seed=config.seed,
        fold_count=len(prepared.folds),
        image_count=len(prepared.test_images),
    )


def _validate_config(config: DensityRuntimeConfig) -> None:
    folds = config.fold_indices
    valid_folds = (
        bool(folds) and len(set(folds)) == len(folds) and all(0 <= v < 4 for v in folds)
    )
    if not config.object_name or not config.device:
        raise DensityRuntimeError("object_name and device must be non-empty")
    if not valid_folds:
        raise DensityRuntimeError("fold_indices must be unique values in [0, 4)")
    if config.query_chunk_size < 1 or config.memory_chunk_size < 1:
        raise DensityRuntimeError("k-NN chunk sizes must be positive")
    valid_shard = config.shard_count > 0 and 0 <= config.shard_index < config.shard_count
    if not valid_shard:
        raise DensityRuntimeError("shard_index must be in [0, shard_count)")


def _progress(event: str, details: Dict[str, object]) -> None:
    payload: Dict[str, object] = {"event": event}
    payload.update(details)
    print(json.dumps(payload, sort_keys=True), flush=True)
