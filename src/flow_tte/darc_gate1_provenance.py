from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Final, List, Sequence, Union

from flow_tte.darc_resources import P16Split

JsonScalar = Union[str, int, float, bool, None]
JsonValue = Union[JsonScalar, Dict[str, "JsonValue"], List["JsonValue"]]

MODEL_ID: Final = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
MODEL_REVISION: Final = "c807c9eeea853df70aec4069e6f56b28ddc82acc"
MODEL_CONFIG_SHA256: Final = "35770e98e425c9383534bcbfa7f3b7a3cb1d943f6d26a37d7d19fea0735eab57"
MODEL_WEIGHTS_SHA256: Final = "3e1d4d18b9bfa9f28fad8e9de6a783f1313532d3460efa4cd0b12521d81d1a4d"

CODE_BUNDLE_FILES: Final = (
    "pyproject.toml",
    "scripts/run_flow_tte_darc_gate1.py",
    "scripts/run_flow_tte_darc_gate1_remote.sh",
    "src/flow_tte/darc_backbone.py",
    "src/flow_tte/darc_feature_stream.py",
    "src/flow_tte/darc_gate1.py",
    "src/flow_tte/darc_gate1_aggregate.py",
    "src/flow_tte/darc_gate1_aggregate_io.py",
    "src/flow_tte/darc_gate1_artifacts.py",
    "src/flow_tte/darc_gate1_metrics.py",
    "src/flow_tte/darc_gate1_provenance.py",
    "src/flow_tte/darc_gate1_runtime.py",
    "src/flow_tte/darc_gate1_samples.py",
    "src/flow_tte/darc_gate1_scoring.py",
    "src/flow_tte/darc_gate1_stability.py",
    "src/flow_tte/darc_geometry.py",
    "src/flow_tte/darc_knn.py",
    "src/flow_tte/darc_resources.py",
    "src/flow_tte/darc_synthetic.py",
    "src/flow_tte/darc_tiling.py",
)


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class ModelProvenance:
    model_id: str = MODEL_ID
    revision: str = MODEL_REVISION
    config_sha256: str = MODEL_CONFIG_SHA256
    weights_sha256: str = MODEL_WEIGHTS_SHA256

    def to_manifest(self) -> Dict[str, JsonValue]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "config_sha256": self.config_sha256,
            "weights_sha256": self.weights_sha256,
        }


REGISTERED_MODEL_PROVENANCE: Final = ModelProvenance()


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class SeedProvenance:
    method_sha256: str
    code_config_sha256: str
    dataset_inventory_sha256: str
    split_inventory_sha256: str
    selection_sha256: str
    model: ModelProvenance = REGISTERED_MODEL_PROVENANCE

    def to_manifest(self) -> Dict[str, JsonValue]:
        return {
            "method_sha256": self.method_sha256,
            "code_config_sha256": self.code_config_sha256,
            "dataset_inventory_sha256": self.dataset_inventory_sha256,
            "split_inventory_sha256": self.split_inventory_sha256,
            "selection_sha256": self.selection_sha256,
            "model": self.model.to_manifest(),
        }

    def digest(self) -> str:
        return sha256_json(self.to_manifest())

    def with_selection_sha256(self, digest: str) -> SeedProvenance:
        return replace(self, selection_sha256=digest)


def sha256_bytes(values: bytes) -> str:
    return hashlib.sha256(values).hexdigest()


def sha256_json(payload: Dict[str, JsonValue]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(encoded)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_bundle_hash(root: Path, method_manifest: Dict[str, JsonValue]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(method_manifest, sort_keys=True, separators=(",", ":")).encode())
    for relative in CODE_BUNDLE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(relative.encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def dataset_inventory_hash(data_root: Path, paths: Sequence[Path]) -> str:
    rows: List[JsonValue] = [
        {
            "path": path.relative_to(data_root).as_posix(),
            "size": path.stat().st_size,
        }
        for path in paths
    ]
    return sha256_json({"normal_path_inventory": rows})


def selection_manifest(
    split: P16Split,
    data_root: Path,
    dataset_sha256: str,
) -> Dict[str, JsonValue]:
    def relative(path_text: str) -> str:
        return Path(path_text).relative_to(data_root).as_posix()

    support_inventory: List[JsonValue] = []
    for path_text in split.support_paths:
        path = Path(path_text)
        support_inventory.append(
            {
                "path": relative(path_text),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            },
        )
    folds: List[JsonValue] = [
        {
            "fold_index": fold.fold_index,
            "memory_paths": [relative(path) for path in fold.memory_paths],
            "calibration_paths": [relative(path) for path in fold.calibration_paths],
        }
        for fold in split.folds
    ]
    base: Dict[str, JsonValue] = {
        "schema": "darc-gate1-selection-v2",
        "version": split.version,
        "seed": split.seed,
        "source_pool_count": split.source_pool_count,
        "dataset_inventory_sha256": dataset_sha256,
        "support_inventory": support_inventory,
        "folds": folds,
    }
    return {**base, "split_inventory_sha256": sha256_json(base)}
