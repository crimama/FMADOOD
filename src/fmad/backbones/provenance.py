"""Deterministic DINOv2 model provenance helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch


def backbone_state_sha256(model: Any) -> str:
    """Hash model state in state-dict order, including dtype, shape, and bytes."""
    if not hasattr(model, "state_dict"):
        raise TypeError("backbone hash requires a state_dict-bearing model")
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(str(name).encode() + b"\0")
        digest.update(str(value.dtype).encode() + b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode() + b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
