"""Offline-cache-first loader for the DINOv2 torch hub repository."""

from __future__ import annotations

import hashlib
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


def _cached_checkpoint(model_name: str) -> Optional[Path]:
    checkpoint_dir = torch_home() / "hub" / "checkpoints"
    weight_name = f"{model_name}4" if model_name.endswith("_reg") else model_name
    checkpoint = checkpoint_dir / f"{weight_name}_pretrain.pth"
    return checkpoint if checkpoint.is_file() else None


def dinov2_repo_tree_sha256(repo: Path) -> str:
    """Hash repository file names and bytes deterministically."""
    repo = Path(repo).expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"DINOv2 repository is not a directory: {repo}")
    digest = hashlib.sha256()
    skipped = {".git", "__pycache__", ".cache", "cache"}
    files = []
    for path in repo.rglob("*"):
        relative = path.relative_to(repo)
        if any(part in skipped for part in relative.parts):
            continue
        if path.is_symlink():
            raise RuntimeError(f"DINOv2 repository contains unsupported symlink: {relative}")
        if path.is_file() and path.suffix != ".pyc":
            files.append(relative)
    for relative in sorted(files, key=lambda value: value.as_posix()):
        payload = relative.as_posix().encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        data = (repo / relative).read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def load_dinov2_model(model_name: str, *, strict_offline: Optional[bool] = None) -> torch.nn.Module:
    """Load cached weights, or the mutable upstream hub ref when allowed."""
    offline = (
        os.environ.get("FMAD_DINOV2_OFFLINE", "").strip().lower() in _TRUE_VALUES
        if strict_offline is None
        else strict_offline
    )
    local_repo = find_local_dinov2_repo()
    if offline:
        checkpoint = _cached_checkpoint(model_name)
        if local_repo is None or checkpoint is None:
            searched = ", ".join(str(path) for path in _candidate_repositories())
            raise FileNotFoundError(
                "Strict offline DINOv2 loading requires a cached hub repository and "
                f"checkpoint for {model_name}; searched repositories: {searched}"
            )
        model = torch.hub.load(
            str(local_repo), model_name, source="local", trust_repo=True, pretrained=False
        )
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        return model
    if local_repo is not None:
        return torch.hub.load(str(local_repo), model_name, source="local", trust_repo=True)
    return torch.hub.load(_REPO_WITH_REF, model_name, trust_repo=True, skip_validation=True)
