from __future__ import annotations

# pyright: reportMissingImports=false
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

from flow_tte.darc_map_io import MapInputError, Population, load_object_maps

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    run_root = tmp_path / "runs"
    _write(
        data_root / "can" / "test_public" / "good" / "000.png",
        np.zeros((4, 6, 3), dtype=np.uint8),
    )
    _write(
        data_root / "can" / "test_public" / "bad" / "001.png",
        np.zeros((4, 6, 3), dtype=np.uint8),
    )
    mask = np.zeros((2, 3), dtype=np.uint8)
    mask[0, 0] = 255
    _write(
        data_root / "can" / "test_public" / "ground_truth" / "bad" / "001_mask.png",
        mask,
    )
    _write(
        run_root / "chunk-a" / "anomaly_maps" / "can" / "test" / "good" / "000.tiff",
        np.full((2, 3), 0.25, dtype=np.float32),
    )
    _write(
        run_root / "chunk-b" / "anomaly_maps" / "can" / "test" / "bad" / "001.tiff",
        np.full((2, 3), 0.75, dtype=np.float32),
    )
    return data_root, run_root


def test_load_object_maps_collects_chunks_and_normalizes_to_native_shape(
    tmp_path: Path,
) -> None:
    # Given: maps split across chunk roots at half the native RGB resolution.
    data_root, run_root = _fixture(tmp_path)

    # When: the canonical object population is loaded.
    loaded = load_object_maps(data_root, (run_root,), "can")

    # Then: continuous maps and GT use one explicit native comparison grid.
    assert tuple(record.audit.image_id for record in loaded.records) == (
        "good/000",
        "bad/001",
    )
    assert loaded.records[0].audit.population is Population.GOOD
    assert loaded.records[0].audit.original_map_shape == (2, 3)
    assert loaded.records[0].audit.common_shape == (4, 6)
    assert loaded.records[0].score_map.shape == (4, 6)
    assert loaded.records[1].gt_mask.shape == (4, 6)
    assert loaded.records[1].audit.original_gt_shape == (2, 3)
    assert loaded.records[0].audit.map_sha256
    assert loaded.records[1].audit.gt_sha256


def test_load_object_maps_rejects_overlapping_duplicate_roots(tmp_path: Path) -> None:
    # Given: one physical map population is supplied twice.
    data_root, run_root = _fixture(tmp_path)

    # When / Then: duplicate canonical image IDs are rejected explicitly.
    with pytest.raises(MapInputError, match="duplicate map ID"):
        load_object_maps(data_root, (run_root, run_root / "chunk-a"), "can")


def test_load_object_maps_rejects_missing_expected_map(tmp_path: Path) -> None:
    # Given: the dataset has a bad image but its anomaly map is absent.
    data_root, run_root = _fixture(tmp_path)
    (run_root / "chunk-b" / "anomaly_maps" / "can" / "test" / "bad" / "001.tiff").unlink()

    # When / Then: population completeness is checked before metric evaluation.
    with pytest.raises(MapInputError, match=r"missing map IDs.*bad/001"):
        load_object_maps(data_root, (run_root,), "can")
