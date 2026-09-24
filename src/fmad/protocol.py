"""Shared protocol objects and deterministic support selection."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


def validate_protocol(protocol: dict) -> None:
    """Validate controls before constructing models or creating run artifacts."""
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a mapping")
    for name, default, minimum in (("shots", 16, 1), ("warmup_iters", 25, 0)):
        value = protocol.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"protocol.{name} must be an integer >= {minimum}")


def seed_generators(seed: int) -> None:
    """Seed common generators before constructing or fitting a method."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
    else:
        torch.manual_seed(seed)


@dataclass(frozen=True)
class MethodContext:
    object_name: str
    seed: int
    resolution: int
    masking: bool
    rotation: bool


def load_fixed_reference_paths(
    manifest_path: Path, object_name: str, train_image_paths: Sequence[str], expected_count: int
) -> tuple[list[str], str]:
    raw = manifest_path.read_bytes()
    payload = json.loads(raw)
    requested = payload.get(object_name) if isinstance(payload, dict) else None
    if not isinstance(requested, list) or len(requested) != expected_count:
        raise ValueError(
            f"{object_name}: expected {expected_count} fixed references, "
            f"found {len(requested) if isinstance(requested, list) else 'invalid'}"
        )

    def identity(value: str) -> tuple[str, str, str, str]:
        parts = Path(value).parts
        suffix = (object_name, "train", "good")
        if len(parts) < 4 or tuple(parts[-4:-1]) != suffix:
            raise ValueError(
                f"{object_name}: fixed reference is outside {object_name}/train/good: {value}"
            )
        return (*suffix, parts[-1])

    active = {identity(path): path for path in train_image_paths}
    if len(active) != len(train_image_paths):
        raise ValueError(f"{object_name}: duplicate train/good reference identities")
    selected = []
    for value in requested:
        key = identity(str(value))
        if key not in active:
            raise ValueError(
                f"{object_name}: fixed reference {'/'.join(key)} "
                "not present in train/good active pool"
            )
        selected.append(active[key])
    if len(set(selected)) != len(selected):
        raise ValueError(f"{object_name}: fixed reference manifest contains duplicates")
    return selected, hashlib.sha256(raw).hexdigest()


def select_support_paths(
    train_image_paths: Sequence[str],
    shots: int,
    seed: int,
    *,
    manifest_path: Path | None = None,
    object_name: str | None = None,
) -> tuple[list[str], str | None]:
    if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)):
        raise ValueError("support shots must be an integer")
    paths = sorted(str(path) for path in train_image_paths)
    if len(set(paths)) != len(paths):
        raise ValueError("training image paths contain duplicates")
    if shots <= 0 or shots > len(paths):
        raise ValueError(f"support shots must be in [1, {len(paths)}], got {shots}")
    if manifest_path is not None:
        if object_name is None:
            raise ValueError("object_name is required with fixed reference manifest")
        return load_fixed_reference_paths(manifest_path, object_name, paths, shots)
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(paths), size=shots, replace=False)
    return [paths[int(index)] for index in selected], None
