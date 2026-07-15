"""Image loading and score-map inference for the SuperADD HF adaptation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import BinaryIO, Final, Generator, Mapping, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np
import torch
from PIL import Image

from flow_tte.superadd_parity import layerwise_anomaly_map, nearest_distance_map
from flow_tte.superadd_patching import (
    PatchConfig,
    PreprocessConfig,
    extract_patched_features,
    preprocess_image,
)

FROZEN_MODEL_ID: Final = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
FROZEN_MODEL_REVISION: Final = "c807c9eeea853df70aec4069e6f56b28ddc82acc"
FROZEN_CONFIG_SHA256: Final = "35770e98e425c9383534bcbfa7f3b7a3cb1d943f6d26a37d7d19fea0735eab57"
FROZEN_CONFIG_SIZE: Final = 744
FROZEN_RESOLVED_CONFIG_SHA256: Final = (
    "b67733bf704e9ac92a55cf3f7594c16033439f4d739058283888f4f6b690fd15"
)
FROZEN_WEIGHT_SHA256: Final = "3e1d4d18b9bfa9f28fad8e9de6a783f1313532d3460efa4cd0b12521d81d1a4d"
FROZEN_WEIGHT_SIZE: Final = 3_362_432_800
FROZEN_TRANSFORMERS_VERSION: Final = "4.56.2"


class FrozenModelContractError(RuntimeError):
    """Raised when the loaded DINOv3 H+ artifact violates the frozen contract."""


@runtime_checkable
class _FrozenConfig(Protocol):
    patch_size: int
    num_hidden_layers: int
    num_register_tokens: int

    def to_dict(self) -> Mapping[str, object]: ...


@runtime_checkable
class _FrozenModel(Protocol):
    config: _FrozenConfig
    layer: Sequence[object]


@dataclass(frozen=True)
class FrozenModelFiles:
    config: Path
    weights: Path


@dataclass(frozen=True)
class FrozenModelVerification:
    files: FrozenModelFiles
    load_directory: Path
    config_sha256: str
    weight_sha256: str
    config_identity: Tuple[int, int, int, int, int]
    weight_identity: Tuple[int, int, int, int, int]
    config_stream: BinaryIO
    weight_stream: BinaryIO


@dataclass(frozen=True)
class ModelAudit:
    model_id: str
    revision: str
    model_class: str
    patch_size: int
    depth: int
    register_count: int
    config_sha256: str
    resolved_config_sha256: str
    weight_sha256: str
    transformers_version: str


class PatchFeatureResult(Protocol):
    @property
    def grids(self) -> Tuple[torch.Tensor, ...]: ...

    @property
    def used_early_exit(self) -> bool: ...


class PatchAdapter(Protocol):
    def extract(self, pixel_values: torch.Tensor) -> PatchFeatureResult: ...


def resolve_frozen_model_files() -> FrozenModelFiles:
    """Resolve the two pinned Hub artifacts from the cache used by Transformers."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    config = hf_hub_download(
        FROZEN_MODEL_ID,
        "config.json",
        revision=FROZEN_MODEL_REVISION,
        local_files_only=True,
    )
    weights = hf_hub_download(
        FROZEN_MODEL_ID,
        "model.safetensors",
        revision=FROZEN_MODEL_REVISION,
        local_files_only=True,
    )
    return FrozenModelFiles(Path(config), Path(weights))


def verify_frozen_runtime() -> None:
    """Fail first-run execution with an explicit dependency/version contract."""
    try:
        version = metadata.version("transformers")
        metadata.version("huggingface-hub")
    except metadata.PackageNotFoundError as error:
        raise FrozenModelContractError(
            "first-run SuperADD requires Transformers 4.56.2 and huggingface-hub",
        ) from error
    if version != FROZEN_TRANSFORMERS_VERSION:
        raise FrozenModelContractError("transformers runtime differs from audited 4.56.2")


@contextmanager
def verify_frozen_model_files(
    files: FrozenModelFiles,
) -> Generator[FrozenModelVerification, None, None]:
    """Yield loader paths bound to the exact file descriptors that were hashed."""
    with ExitStack() as stack:
        config_stream = stack.enter_context(files.config.open("rb"))
        weight_stream = stack.enter_context(files.weights.open("rb"))
        temporary = stack.enter_context(
            tempfile.TemporaryDirectory(prefix="superadd-frozen-model-"),
        )
        config_sha, config_identity = _verified_artifact(
            config_stream,
            files.config,
            FROZEN_CONFIG_SHA256,
            FROZEN_CONFIG_SIZE,
        )
        weight_sha, weight_identity = _verified_artifact(
            weight_stream,
            files.weights,
            FROZEN_WEIGHT_SHA256,
            FROZEN_WEIGHT_SIZE,
        )
        load_directory = Path(temporary)
        (load_directory / "config.json").symlink_to(
            f"/proc/self/fd/{config_stream.fileno()}",
        )
        (load_directory / "model.safetensors").symlink_to(
            f"/proc/self/fd/{weight_stream.fileno()}",
        )
        verification = FrozenModelVerification(
            files,
            load_directory,
            config_sha,
            weight_sha,
            config_identity,
            weight_identity,
            config_stream,
            weight_stream,
        )
        if not _verification_is_current(verification):
            raise FrozenModelContractError("verified model artifacts changed before load")
        yield verification


def audit_frozen_model(
    model: object,
    files: FrozenModelFiles,
    verification: FrozenModelVerification | None = None,
) -> ModelAudit:
    """Verify architecture and immutable Hub artifacts, then capture provenance."""
    if type(model).__name__ != "DINOv3ViTModel" or not isinstance(model, _FrozenModel):
        raise FrozenModelContractError("expected a DINOv3ViTModel with config and layer")
    config = model.config
    structure = (
        int(config.patch_size),
        int(config.num_hidden_layers),
        int(config.num_register_tokens),
        len(model.layer),
    )
    if structure != (16, 32, 4, 32):
        raise FrozenModelContractError("expected patch_size=16, depth=32, register_count=4")
    if getattr(config, "_commit_hash", None) != FROZEN_MODEL_REVISION:
        raise FrozenModelContractError("resolved model revision differs from frozen revision")
    version = metadata.version("transformers")
    if version != FROZEN_TRANSFORMERS_VERSION:
        raise FrozenModelContractError("transformers runtime differs from audited 4.56.2")
    if verification is None:
        with verify_frozen_model_files(files) as verified:
            return _finish_model_audit(model, files, verified, structure, version)
    return _finish_model_audit(model, files, verification, structure, version)


def _finish_model_audit(
    model: _FrozenModel,
    files: FrozenModelFiles,
    verified: FrozenModelVerification,
    structure: Tuple[int, int, int, int],
    version: str,
) -> ModelAudit:
    if verified.files != files or not _verification_is_current(verified):
        raise FrozenModelContractError("verified model artifacts changed before audit")
    serialized = json.dumps(
        model.config.to_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    resolved_config_sha = hashlib.sha256(serialized).hexdigest()
    if resolved_config_sha != FROZEN_RESOLVED_CONFIG_SHA256:
        raise FrozenModelContractError("resolved Transformers config digest mismatch")
    return ModelAudit(
        FROZEN_MODEL_ID,
        FROZEN_MODEL_REVISION,
        type(model).__name__,
        structure[0],
        structure[1],
        structure[2],
        verified.config_sha256,
        resolved_config_sha,
        verified.weight_sha256,
        version,
    )


def _verified_artifact(
    stream: BinaryIO,
    path: Path,
    expected: str,
    expected_size: int,
) -> Tuple[str, Tuple[int, int, int, int, int]]:
    digest = hashlib.sha256()
    before = _stat_identity(os.fstat(stream.fileno()))
    if before[2] != expected_size:
        message = f"artifact size mismatch: {path.name}"
        raise FrozenModelContractError(message)
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    actual = digest.hexdigest()
    after = _stat_identity(os.fstat(stream.fileno()))
    if before != after or after != _file_identity(path):
        raise FrozenModelContractError("model artifact changed during verification")
    if actual != expected:
        message = f"artifact digest mismatch: {path.name}"
        raise FrozenModelContractError(message)
    return actual, after


def _verification_is_current(verification: FrozenModelVerification) -> bool:
    try:
        return (
            verification.config_identity
            == _stat_identity(os.fstat(verification.config_stream.fileno()))
            == _file_identity(verification.load_directory / "config.json")
            and verification.weight_identity
            == _stat_identity(os.fstat(verification.weight_stream.fileno()))
            == _file_identity(verification.load_directory / "model.safetensors")
        )
    except (OSError, ValueError):
        return False


def _file_identity(path: Path) -> Tuple[int, int, int, int, int]:
    return _stat_identity(path.stat())


def _stat_identity(stat: os.stat_result) -> Tuple[int, int, int, int, int]:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@dataclass(frozen=True)
class ImageGridExtractor:
    adapter: PatchAdapter
    device: torch.device
    preprocess: PreprocessConfig
    patching: PatchConfig

    def extract(
        self,
        path: Path,
        brightness: float,
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[int, int], bool]:
        """Load, preprocess, patch-extract, and restore one image's grids."""
        image = read_rgb_tensor(path)
        native_size = (int(image.shape[-2]), int(image.shape[-1]))
        processed = preprocess_image(image.to(self.device), self.preprocess, brightness)
        early_exit_flags = []

        def extract_batch(batch: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            result = self.adapter.extract(batch)
            early_exit_flags.append(result.used_early_exit)
            return result.grids

        grids = extract_patched_features(processed, extract_batch, self.patching)
        return grids, native_size, all(early_exit_flags)


def read_rgb_tensor(path: Path) -> torch.Tensor:
    """Read an RGB image into the official float tensor range [0,1]."""
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))


def score_image(
    path: Path,
    extractor: ImageGridExtractor,
    banks: Sequence[torch.Tensor],
) -> Tuple[np.ndarray, bool]:
    """Score one image with layer-wise official 1-NN fusion."""
    grids, native_size, used_early_exit = extractor.extract(path, 1.0)
    maps = tuple(nearest_distance_map(grid, bank) for grid, bank in zip(grids, banks))
    result = layerwise_anomaly_map(maps, native_size)
    return result.numpy().astype(np.float32, copy=False), used_early_exit
