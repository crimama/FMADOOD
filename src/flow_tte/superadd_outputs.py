"""Atomic category-scoped artifacts for the SuperADD HF adaptation."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Generator, Sequence, Tuple

import cv2
import numpy as np


class SuperADDOutputError(IOError):
    """Raised when a parity artifact or completion contract is invalid."""


class CategoryRunActiveError(SuperADDOutputError):
    """Raised when another process owns the same category lease."""


@dataclass(frozen=True)
class MapIdentity:
    output_root: Path
    category: str
    split: str
    stem: str


@dataclass(frozen=True)
class CanonicalMapPaths:
    raw: Path
    binary: Path


@dataclass(frozen=True)
class CategoryRunPlan:
    output_root: Path
    category: str
    expected_maps: Tuple[CanonicalMapPaths, ...]
    spec_sha256: str

    def __post_init__(self) -> None:
        _validate_plan(self)


@dataclass(frozen=True)
class CategoryRun:
    plan: CategoryRunPlan
    resumed: bool

    @property
    def manifest_path(self) -> Path:
        return _category_dir(self.plan) / "manifest.json"


def canonical_map_paths(identity: MapIdentity) -> CanonicalMapPaths:
    """Place raw TIFFs where the common evaluator discovers anomaly maps."""
    relative = Path(identity.category) / "test" / identity.split
    return CanonicalMapPaths(
        raw=identity.output_root / "anomaly_maps" / relative / f"{identity.stem}.tiff",
        binary=(identity.output_root / "official_binary_maps" / relative / f"{identity.stem}.png"),
    )


def write_map_artifacts(
    paths: CanonicalMapPaths,
    raw_map: np.ndarray,
    binary_map: np.ndarray,
) -> None:
    """Atomically persist raw float32 TIFF and separate uint8 binary output."""
    raw_temp = _write_cv2_temp(paths.raw, np.asarray(raw_map, dtype=np.float32))
    binary_temp: Path | None = None
    try:
        binary_temp = _write_cv2_temp(paths.binary, np.asarray(binary_map, dtype=np.uint8))
        raw_temp.replace(paths.raw)
        binary_temp.replace(paths.binary)
        _fsync_dir(paths.raw.parent)
        _fsync_dir(paths.binary.parent)
    except BaseException:
        raw_temp.unlink(missing_ok=True)
        if binary_temp is not None:
            binary_temp.unlink(missing_ok=True)
        raise


@contextmanager
def category_run(plan: CategoryRunPlan) -> Generator[CategoryRun, None, None]:
    """Lease one category and finalize only an exact, checksummed inventory."""
    _prepare_lease_directory(plan)
    category_dir = _category_dir(plan)
    try:
        lock_descriptor = os.open(
            str(category_dir / ".run.lock"),
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
            0o600,
        )
    except OSError as error:
        raise SuperADDOutputError("category lock must not be a symlink") from error
    lock = os.fdopen(lock_descriptor, "a+b")
    try:
        _acquire_lock(lock, plan.category)
        _prepare_map_directories(plan)
        resumed = _valid_completion(plan)
        if not resumed:
            _clean_stale_category(plan)
            _prepare_expected_map_directories(plan)
            _write_state(plan, "running")
        run = CategoryRun(plan, resumed)
        try:
            yield run
            if not resumed:
                _finalize(run)
        except BaseException:
            if not resumed:
                _clean_stale_category(plan)
                _write_state(plan, "incomplete")
            raise
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def write_category_manifest(run: CategoryRun, serialized_json: str) -> None:
    """Atomically write the category manifest before completion is allowed."""
    if run.resumed:
        raise SuperADDOutputError("cannot rewrite a completed category manifest")
    parsed: object = json.loads(serialized_json)
    if not isinstance(parsed, dict):
        raise SuperADDOutputError("category manifest must be a JSON object")
    _atomic_text(run.manifest_path, serialized_json.rstrip() + "\n")


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def files_sha256(paths: Sequence[Path]) -> str:
    """Hash ordered file labels and bytes into a portable implementation digest."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8") + b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _acquire_lock(lock: BinaryIO, category: str) -> None:
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        message = f"category run is already active: {category}"
        raise CategoryRunActiveError(message) from error


def _finalize(run: CategoryRun) -> None:
    payload = _completion_payload(run.plan)
    _atomic_text(
        _category_dir(run.plan) / "completion.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    (_category_dir(run.plan) / "state.json").unlink(missing_ok=True)
    _fsync_dir(_category_dir(run.plan))


def _valid_completion(plan: CategoryRunPlan) -> bool:
    path = _category_dir(plan) / "completion.json"
    if not path.is_file():
        return False
    try:
        existing: object = json.loads(path.read_text(encoding="utf-8"))
        return existing == _completion_payload(plan)
    except (json.JSONDecodeError, OSError, SuperADDOutputError, UnicodeDecodeError):
        return False


def _completion_payload(plan: CategoryRunPlan) -> object:
    _assert_controlled_directories(plan)
    manifest = _category_dir(plan) / "manifest.json"
    if not manifest.is_file():
        raise SuperADDOutputError("category manifest is missing")
    expected = {
        path: kind
        for pair in plan.expected_maps
        for path, kind in ((pair.raw, "raw"), (pair.binary, "binary"))
    }
    discovered = tuple(
        path
        for root in _map_roots(plan)
        if root.exists()
        for path in root.rglob("*")
    )
    if any(path.is_symlink() for path in discovered):
        raise SuperADDOutputError("map inventory must not contain symlinks")
    actual = {path for path in discovered if path.is_file()}
    if actual != set(expected):
        raise SuperADDOutputError("actual map inventory differs from the frozen expectation")
    artifacts = [
        {
            "kind": expected[path],
            "path": str(path.relative_to(plan.output_root)),
            "sha256": _file_sha256(path),
        }
        for path in sorted(actual, key=lambda item: str(item.relative_to(plan.output_root)))
    ]
    return {
        "artifacts": artifacts,
        "category": plan.category,
        "manifest": {
            "path": str(manifest.relative_to(plan.output_root)),
            "sha256": _file_sha256(manifest),
        },
        "schema_version": 1,
        "spec_sha256": plan.spec_sha256,
        "state": "complete",
    }


def _clean_stale_category(plan: CategoryRunPlan) -> None:
    _assert_controlled_directories(plan)
    for root in _map_roots(plan):
        _remove_path(root)
    category_dir = _category_dir(plan)
    for path in category_dir.iterdir():
        if path.name != ".run.lock":
            _remove_path(path)


def _write_state(plan: CategoryRunPlan, state: str) -> None:
    payload = {"category": plan.category, "spec_sha256": plan.spec_sha256, "state": state}
    _atomic_text(
        _category_dir(plan) / "state.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def _validate_plan(plan: CategoryRunPlan) -> None:
    if plan.category in {"", ".", ".."} or Path(plan.category).name != plan.category:
        raise SuperADDOutputError("category must be one safe path segment")
    if len(plan.spec_sha256) != 64 or not plan.expected_maps:
        raise SuperADDOutputError("run plan requires a SHA256 spec and expected maps")
    flattened = tuple(path for pair in plan.expected_maps for path in (pair.raw, pair.binary))
    if len(set(flattened)) != len(flattened):
        raise SuperADDOutputError("expected map paths must be unique")
    roots = _map_roots(plan)
    for pair in plan.expected_maps:
        for path, root in zip((pair.raw, pair.binary), roots):
            if not _is_relative_to(path, root):
                raise SuperADDOutputError("expected map escapes its category root")


def _write_cv2_temp(destination: Path, values: np.ndarray) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded_ok, encoded = cv2.imencode(destination.suffix, values)
    if not encoded_ok:
        message = f"OpenCV failed to write {destination.name}"
        raise SuperADDOutputError(message)
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=str(destination.parent),
            prefix=f".{destination.stem}.",
            suffix=destination.suffix,
            delete=False,
        ) as stream:
            temp = Path(stream.name)
            stream.write(encoded.tobytes())
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if temp is not None:
            temp.unlink(missing_ok=True)
        raise
    else:
        return temp


def _atomic_text(destination: Path, value: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=str(destination.parent),
            prefix=f".{destination.name}.",
            delete=False,
            encoding="utf-8",
        ) as stream:
            temp = Path(stream.name)
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(destination)
        _fsync_dir(destination.parent)
    except BaseException:
        if temp is not None:
            temp.unlink(missing_ok=True)
        raise


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _category_dir(plan: CategoryRunPlan) -> Path:
    return plan.output_root / "categories" / plan.category


def _map_roots(plan: CategoryRunPlan) -> Tuple[Path, Path]:
    return (
        plan.output_root / "anomaly_maps" / plan.category,
        plan.output_root / "official_binary_maps" / plan.category,
    )


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def _is_relative_to(path: Path, root: Path) -> bool:
    normalized_path = Path(os.path.normpath(str(path.absolute())))
    normalized_root = Path(os.path.normpath(str(root.absolute())))
    try:
        normalized_path.relative_to(normalized_root)
    except ValueError:
        return False
    return True


def _prepare_lease_directory(plan: CategoryRunPlan) -> None:
    plan.output_root.mkdir(parents=True, exist_ok=True)
    _open_directory_no_follow(plan.output_root)
    for relative in (
        Path("categories"),
        Path("categories") / plan.category,
    ):
        _mkdir_chain_no_follow(plan.output_root, relative)


def _prepare_map_directories(plan: CategoryRunPlan) -> None:
    for relative in (
        Path("anomaly_maps"),
        Path("anomaly_maps") / plan.category,
        Path("official_binary_maps"),
        Path("official_binary_maps") / plan.category,
    ):
        _mkdir_chain_no_follow(plan.output_root, relative)


def _assert_controlled_directories(plan: CategoryRunPlan) -> None:
    for path in (
        plan.output_root,
        plan.output_root / "categories",
        _category_dir(plan),
        plan.output_root / "anomaly_maps",
        plan.output_root / "official_binary_maps",
        *_map_roots(plan),
    ):
        if path.exists() or path.is_symlink():
            _open_directory_no_follow(path)


def _mkdir_chain_no_follow(root: Path, relative: Path) -> None:
    current = root
    for segment in relative.parts:
        if segment in {"", ".", ".."}:
            raise SuperADDOutputError("controlled output path contains an unsafe segment")
        current = current / segment
        with suppress(FileExistsError):
            current.mkdir()
        _open_directory_no_follow(current)


def _prepare_expected_map_directories(plan: CategoryRunPlan) -> None:
    for pair in plan.expected_maps:
        for destination in (pair.raw, pair.binary):
            relative_parent = Path(os.path.relpath(destination.parent, plan.output_root))
            _mkdir_chain_no_follow(plan.output_root, relative_parent)


def _open_directory_no_follow(path: Path) -> None:
    try:
        descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        message = f"controlled output directory is a symlink or non-directory: {path}"
        raise SuperADDOutputError(message) from error
    os.close(descriptor)


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
