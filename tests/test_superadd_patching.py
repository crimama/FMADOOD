from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, NamedTuple

import pytest
import torch

import flow_tte.superadd_inference as superadd_inference
from flow_tte.superadd_inference import (
    FrozenModelContractError,
    FrozenModelFiles,
    ImageGridExtractor,
    audit_frozen_model,
    verify_frozen_model_files,
    verify_frozen_runtime,
)
from flow_tte.superadd_patching import (
    PatchConfig,
    PreprocessConfig,
    axis_patch_split,
    extract_patched_features,
    preprocess_image,
)


def test_axis_patch_split_matches_official_three_patch_ownership() -> None:
    # Given: the official 640/128 patch setup on a 1280-pixel axis.
    config = PatchConfig(patch_size=640, overlap=128, model_patch_size=16)

    # When: token-aligned input and ownership ROIs are constructed.
    split = axis_patch_split(1280, config)

    # Then: overlapping inputs are evenly spaced and ownership is gap-free.
    assert split.input_rois == ((0, 640), (320, 960), (640, 1280))
    assert split.prediction_rois == ((0, 30), (10, 30), (10, 40))
    assert split.result_rois == ((0, 30), (30, 50), (50, 80))


def test_extract_patched_features_stitches_each_token_once() -> None:
    # Given: a coordinate image split into two overlapping 4x4 patches.
    image = torch.arange(24, dtype=torch.float32).reshape(1, 1, 6, 4)
    config = PatchConfig(patch_size=4, overlap=1, model_patch_size=1)

    def fake_extract(batch: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (batch.permute(0, 2, 3, 1),)

    # When: patch predictions are stitched through official ownership ROIs.
    (grid,) = extract_patched_features(image, fake_extract, config)

    # Then: every source token is restored exactly once in row-major geometry.
    torch.testing.assert_close(grid[0, ..., 0], image[0, 0])


def test_preprocess_image_matches_official_brightness_resize_and_norm() -> None:
    # Given: a uniform RGB image and a fixed brightness realization.
    image = torch.ones((3, 4, 6), dtype=torch.float32)
    config = PreprocessConfig(resize_factor=0.5)

    # When: the official tensor preprocessing is applied.
    result = preprocess_image(image, config, brightness_factor=0.5)

    # Then: shape, clipping, bicubic resize, and ImageNet normalization agree.
    assert result.shape == (1, 3, 2, 3)
    expected = (torch.full((3,), 0.5) - torch.tensor(config.mean)) / torch.tensor(
        config.std,
    )
    torch.testing.assert_close(result[0, :, 0, 0], expected)


class _GridResult(NamedTuple):
    grids: tuple[torch.Tensor, ...]
    used_early_exit: bool


class _NumericAdapter:
    def extract(self, pixel_values: torch.Tensor) -> _GridResult:
        grid = pixel_values[:, :1].permute(0, 2, 3, 1)
        return _GridResult((grid,), used_early_exit=True)


def test_extractor_moves_decoded_tensor_before_numeric_preprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a decoded CPU tensor and an observable target device boundary.
    decoded = torch.zeros((3, 4, 4), dtype=torch.float32)
    observed_devices: list[torch.device] = []

    def fake_read(_path: Path) -> torch.Tensor:
        return decoded

    def fake_preprocess(
        image: torch.Tensor,
        _config: PreprocessConfig,
        _brightness: float,
    ) -> torch.Tensor:
        observed_devices.append(image.device)
        return torch.full((1, 3, 4, 4), 2.5)

    monkeypatch.setattr(superadd_inference, "read_rgb_tensor", fake_read)
    monkeypatch.setattr(superadd_inference, "preprocess_image", fake_preprocess)
    extractor = ImageGridExtractor(
        _NumericAdapter(),
        torch.device("meta"),
        PreprocessConfig(resize_factor=1.0),
        PatchConfig(patch_size=4, overlap=1, model_patch_size=1),
    )

    # When: extraction follows the official move-then-preprocess order.
    grids, native_size, used_early_exit = extractor.extract(Path("unused.png"), 1.0)

    # Then: preprocessing sees the target device and numeric output is preserved.
    assert observed_devices == [torch.device("meta")]
    assert native_size == (4, 4)
    assert used_early_exit is True
    torch.testing.assert_close(grids[0], torch.full((1, 4, 4, 1), 2.5))


class _FrozenConfig:
    patch_size = 16
    num_hidden_layers = 32
    num_register_tokens = 4
    _commit_hash = superadd_inference.FROZEN_MODEL_REVISION

    def to_dict(self) -> Mapping[str, object]:
        return {
            "num_hidden_layers": self.num_hidden_layers,
            "num_register_tokens": self.num_register_tokens,
            "patch_size": self.patch_size,
        }


class DINOv3ViTModel:
    def __init__(self) -> None:
        self.config = _FrozenConfig()
        self.layer = tuple(object() for _ in range(32))


def test_frozen_model_audit_verifies_structure_and_artifact_digests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: pinned config/weight artifacts and the exact H+ architecture.
    config_bytes = b'{"patch_size":16}'
    weight_bytes = b"frozen-weight"
    config_path = tmp_path / "config.json"
    weight_path = tmp_path / "model.safetensors"
    config_path.write_bytes(config_bytes)
    weight_path.write_bytes(weight_bytes)
    monkeypatch.setattr(
        superadd_inference,
        "FROZEN_CONFIG_SHA256",
        hashlib.sha256(config_bytes).hexdigest(),
    )
    monkeypatch.setattr(superadd_inference, "FROZEN_CONFIG_SIZE", len(config_bytes))
    monkeypatch.setattr(
        superadd_inference,
        "FROZEN_WEIGHT_SHA256",
        hashlib.sha256(weight_bytes).hexdigest(),
    )
    monkeypatch.setattr(superadd_inference, "FROZEN_WEIGHT_SIZE", len(weight_bytes))
    monkeypatch.setattr(superadd_inference.metadata, "version", lambda _name: "4.56.2")
    resolved = json.dumps(
        DINOv3ViTModel().config.to_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    monkeypatch.setattr(
        superadd_inference,
        "FROZEN_RESOLVED_CONFIG_SHA256",
        hashlib.sha256(resolved).hexdigest(),
    )

    # When: runtime provenance is audited.
    audit = audit_frozen_model(
        DINOv3ViTModel(),
        FrozenModelFiles(config_path, weight_path),
    )

    # Then: revision, structure, raw artifacts, and resolved config are recorded.
    assert (audit.patch_size, audit.depth, audit.register_count) == (16, 32, 4)
    assert audit.config_sha256 == hashlib.sha256(config_bytes).hexdigest()
    assert audit.weight_sha256 == hashlib.sha256(weight_bytes).hexdigest()
    assert len(audit.resolved_config_sha256) == 64


def test_frozen_model_audit_hashes_weight_bytes_even_for_cache_named_blob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a same-sized tampered blob whose filename falsely claims the expected digest.
    config_bytes = b"{}"
    expected_weight = b"trusted-weight"
    tampered_weight = b"tampered-data!"
    expected_digest = hashlib.sha256(expected_weight).hexdigest()
    config_path = tmp_path / "config.json"
    weight_path = tmp_path / expected_digest
    config_path.write_bytes(config_bytes)
    weight_path.write_bytes(tampered_weight)
    monkeypatch.setattr(
        superadd_inference,
        "FROZEN_CONFIG_SHA256",
        hashlib.sha256(config_bytes).hexdigest(),
    )
    monkeypatch.setattr(superadd_inference, "FROZEN_CONFIG_SIZE", len(config_bytes))
    monkeypatch.setattr(superadd_inference, "FROZEN_WEIGHT_SHA256", expected_digest)
    monkeypatch.setattr(superadd_inference, "FROZEN_WEIGHT_SIZE", len(expected_weight))
    monkeypatch.setattr(superadd_inference.metadata, "version", lambda _name: "4.56.2")

    # When/Then: the cache filename cannot substitute for hashing the bytes.
    with pytest.raises(FrozenModelContractError, match="artifact digest mismatch"):
        audit_frozen_model(
            DINOv3ViTModel(),
            FrozenModelFiles(config_path, weight_path),
        )


def test_frozen_model_audit_rejects_resolved_config_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_bytes = b"{}"
    weight_bytes = b"weight"
    config_path = tmp_path / "config.json"
    weight_path = tmp_path / "model.safetensors"
    config_path.write_bytes(config_bytes)
    weight_path.write_bytes(weight_bytes)
    monkeypatch.setattr(
        superadd_inference,
        "FROZEN_CONFIG_SHA256",
        hashlib.sha256(config_bytes).hexdigest(),
    )
    monkeypatch.setattr(superadd_inference, "FROZEN_CONFIG_SIZE", len(config_bytes))
    monkeypatch.setattr(
        superadd_inference,
        "FROZEN_WEIGHT_SHA256",
        hashlib.sha256(weight_bytes).hexdigest(),
    )
    monkeypatch.setattr(superadd_inference, "FROZEN_WEIGHT_SIZE", len(weight_bytes))
    monkeypatch.setattr(superadd_inference.metadata, "version", lambda _name: "4.56.2")

    with pytest.raises(FrozenModelContractError, match="resolved Transformers config"):
        audit_frozen_model(
            DINOv3ViTModel(),
            FrozenModelFiles(config_path, weight_path),
        )


def test_preload_verification_rejects_artifact_changed_after_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_bytes = b"config"
    weight_bytes = b"trusted"
    tampered_bytes = b"changed"
    config_path = tmp_path / "config.json"
    weight_path = tmp_path / "model.safetensors"
    replacement_path = tmp_path / "replacement.safetensors"
    config_path.write_bytes(config_bytes)
    weight_path.write_bytes(weight_bytes)
    replacement_path.write_bytes(tampered_bytes)
    monkeypatch.setattr(
        superadd_inference,
        "FROZEN_CONFIG_SHA256",
        hashlib.sha256(config_bytes).hexdigest(),
    )
    monkeypatch.setattr(superadd_inference, "FROZEN_CONFIG_SIZE", len(config_bytes))
    monkeypatch.setattr(
        superadd_inference,
        "FROZEN_WEIGHT_SHA256",
        hashlib.sha256(weight_bytes).hexdigest(),
    )
    monkeypatch.setattr(superadd_inference, "FROZEN_WEIGHT_SIZE", len(weight_bytes))
    real_sha256 = hashlib.sha256
    digest_count = 0

    class SwappingDigest:
        def __init__(self) -> None:
            nonlocal digest_count
            digest_count += 1
            self._index = digest_count
            self._digest = real_sha256()

        def update(self, value: bytes) -> None:
            self._digest.update(value)

        def hexdigest(self) -> str:
            result = self._digest.hexdigest()
            if self._index == 2:
                replacement_path.replace(weight_path)
            return result

    monkeypatch.setattr(superadd_inference.hashlib, "sha256", SwappingDigest)

    failure = pytest.raises(FrozenModelContractError, match="changed during verification")
    verification = verify_frozen_model_files(FrozenModelFiles(config_path, weight_path))
    with failure, verification:
        pass


def test_verified_loader_directory_stays_bound_to_hashed_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: mutable Hub-style symlinks pointing at trusted immutable blobs.
    trusted_config = b'{"model_type":"dinov3_vit"}'
    trusted_weight = b"trusted-weight"
    malicious_weight = b"malicious-data"
    blobs = tmp_path / "blobs"
    snapshot = tmp_path / "snapshot"
    blobs.mkdir()
    snapshot.mkdir()
    config_blob = blobs / "config"
    weight_blob = blobs / "weight"
    malicious_blob = blobs / "malicious"
    config_blob.write_bytes(trusted_config)
    weight_blob.write_bytes(trusted_weight)
    malicious_blob.write_bytes(malicious_weight)
    config_path = snapshot / "config.json"
    weight_path = snapshot / "model.safetensors"
    config_path.symlink_to(config_blob)
    weight_path.symlink_to(weight_blob)
    monkeypatch.setattr(
        superadd_inference,
        "FROZEN_CONFIG_SHA256",
        hashlib.sha256(trusted_config).hexdigest(),
    )
    monkeypatch.setattr(superadd_inference, "FROZEN_CONFIG_SIZE", len(trusted_config))
    monkeypatch.setattr(
        superadd_inference,
        "FROZEN_WEIGHT_SHA256",
        hashlib.sha256(trusted_weight).hexdigest(),
    )
    monkeypatch.setattr(superadd_inference, "FROZEN_WEIGHT_SIZE", len(trusted_weight))

    # When: the mutable cache symlink is repointed after byte verification.
    with verify_frozen_model_files(
        FrozenModelFiles(config_path, weight_path),
    ) as verification:
        weight_path.unlink()
        weight_path.symlink_to(malicious_blob)

        # Then: the loader-facing path still opens the exact verified inode.
        assert (verification.load_directory / "config.json").read_bytes() == trusted_config
        assert (
            verification.load_directory / "model.safetensors"
        ).read_bytes() == trusted_weight


def test_frozen_model_audit_rejects_register_contract_drift(
    tmp_path: Path,
) -> None:
    model = DINOv3ViTModel()
    model.config.num_register_tokens = 0

    with pytest.raises(FrozenModelContractError, match="register_count=4"):
        audit_frozen_model(
            model,
            FrozenModelFiles(tmp_path / "unused-config", tmp_path / "unused-weight"),
        )


def test_frozen_runtime_rejects_unreviewed_transformers_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        superadd_inference.metadata,
        "version",
        lambda name: "4.57.0" if name == "transformers" else "0.34.0",
    )

    with pytest.raises(FrozenModelContractError, match=r"audited 4\.56\.2"):
        verify_frozen_runtime()
