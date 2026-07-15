"""Disk-backed prototype-bank construction for bounded Pfull host memory."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, List, Protocol, Sequence, Tuple

import numpy as np
import torch

from flow_tte.superadd_parity import CoresetConfig, subsample_distance_based


class SuperADDBankError(RuntimeError):
    """Raised when extracted layer features violate the bank contract."""


class BankExtractor(Protocol):
    def extract(
        self,
        path: Path,
        brightness: float,
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[int, int], bool]: ...


@dataclass
class _LayerStore:
    path: Path
    rows: int = 0
    channels: int = 0


def fit_disk_backed_banks(
    paths: Sequence[Path],
    extractors: Sequence[BankExtractor],
    device: torch.device,
    rng: np.random.RandomState,
    scratch_root: Path,
) -> Tuple[Tuple[torch.Tensor, ...], bool]:
    """Spool, coreset, and delete one early-exit layer at a time."""
    if len(extractors) != 4 or not paths:
        raise SuperADDBankError("four layer extractors and non-empty prototype paths are required")
    brightness = tuple(float(rng.uniform(0.8, 1.2)) for _ in paths)
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".superadd-bank-", dir=str(scratch_root)) as name:
        directory = Path(name)
        banks = []
        early_exit_flags = []
        for layer_index, extractor in enumerate(extractors):
            store = _LayerStore(directory / f"layer-{layer_index}.float32")
            early_exit_flags.extend(
                _spool_layer(paths, brightness, extractor, store, layer_index),
            )
            banks.append(_fit_one_layer(store, device, rng, layer_index))
    return tuple(banks), all(early_exit_flags)


def _spool_layer(
    paths: Sequence[Path],
    brightness: Sequence[float],
    extractor: BankExtractor,
    store: _LayerStore,
    layer_index: int,
) -> List[bool]:
    early_exit_flags = []
    with store.path.open("wb") as stream:
        for index, (path, brightness_factor) in enumerate(zip(paths, brightness), start=1):
            grids, _, used_early_exit = extractor.extract(path, brightness_factor)
            if len(grids) != 1:
                raise SuperADDBankError("each early-exit extractor must return exactly one layer")
            _append_grid(stream, grids[0], store)
            early_exit_flags.append(used_early_exit)
            print(
                f"layer {layer_index} prototype {index}/{len(paths)} {path.name}",
                flush=True,
            )
    return early_exit_flags


def _append_grid(
    stream: BinaryIO,
    grid: torch.Tensor,
    store: _LayerStore,
) -> None:
    channel_count = int(grid.shape[-1])
    if store.channels not in {0, channel_count}:
        raise SuperADDBankError("feature channel count changed between prototype images")
    host_grid = (
        grid.detach()
        .reshape(-1, channel_count)
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
    )
    values = np.ascontiguousarray(
        host_grid.numpy(),
        dtype=np.float32,
    )
    values.tofile(stream)
    store.rows += len(values)
    store.channels = channel_count


def _fit_one_layer(
    store: _LayerStore,
    device: torch.device,
    rng: np.random.RandomState,
    layer_index: int,
) -> torch.Tensor:
    if store.rows < 1 or store.channels < 1:
        raise SuperADDBankError("feature layer store is empty")
    features = np.memmap(
        store.path,
        dtype=np.float32,
        mode="r",
        shape=(store.rows, store.channels),
    )
    sampled = subsample_distance_based(features, device, rng, CoresetConfig())
    owned = np.array(sampled, dtype=np.float32, copy=True, order="C")
    del sampled, features
    store.path.unlink()
    bank = torch.from_numpy(owned).to(device)
    print(f"layer {layer_index}: {store.rows} -> {len(bank)}", flush=True)
    return bank
