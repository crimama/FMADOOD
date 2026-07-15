"""DINOv2 torch.hub loader with an offline-cache-first policy."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import torch

_REPO_WITH_REF = "facebookresearch/dinov2:main"
_CACHE_DIRS = ("facebookresearch_dinov2_main", "facebookresearch_dinov2_master")
_TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def torch_home() -> Path:
    if os.environ.get("TORCH_HOME"):
        return Path(os.environ["TORCH_HOME"]).expanduser()
    if os.environ.get("XDG_CACHE_HOME"):
        return Path(os.environ["XDG_CACHE_HOME"]).expanduser() / "torch"
    return Path.home() / ".cache" / "torch"


def _candidate_repositories() -> Iterable[Path]:
    for name in ("DINOV2_LOCAL_REPO", "DINOv2_LOCAL_REPO", "DINO_V2_LOCAL_REPO"):
        value = os.environ.get(name)
        if value:
            yield Path(value).expanduser()
    for name in _CACHE_DIRS:
        yield torch_home() / "hub" / name


def find_local_dinov2_repo() -> Optional[Path]:
    for path in _candidate_repositories():
        if (path / "hubconf.py").is_file():
            return path
    return None


def load_dinov2_model(
    model_name: str,
    *,
    strict_offline: Optional[bool] = None,
) -> torch.nn.Module:
    """Load a DINOv2 model locally when cached, otherwise use the pinned hub ref."""
    local_repo = find_local_dinov2_repo()
    if local_repo is not None:
        try:
            return torch.hub.load(
                str(local_repo),
                model_name,
                source="local",
                trust_repo=True,
            )
        except TypeError:
            return torch.hub.load(str(local_repo), model_name, source="local")

    offline = (
        os.environ.get("FMAD_DINOV2_OFFLINE", "").strip().lower() in _TRUE_VALUES
        if strict_offline is None
        else strict_offline
    )
    if offline:
        searched = ", ".join(str(path) for path in _candidate_repositories())
        message = f"No cached DINOv2 hub repository found; searched: {searched}"
        raise FileNotFoundError(message)
    try:
        return torch.hub.load(
            _REPO_WITH_REF,
            model_name,
            trust_repo=True,
            skip_validation=True,
        )
    except TypeError:
        return torch.hub.load(_REPO_WITH_REF, model_name, trust_repo=True)
