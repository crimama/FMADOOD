from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile as tiff
from PIL import Image

from fmad.datasets.mvtec_ad2 import MVTecAD2Dataset
from scripts.run_flow_tte_mvtecad2_guided_refinement import refine_maps


def _write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)).save(path)


def test_guided_refinement_preserves_ad2_map_layout(tmp_path: Path) -> None:
    data = tmp_path / "data"
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_rgb(data / "can" / "test_public" / "good" / "001.png", 20)
    _write_rgb(data / "can" / "test_public" / "bad" / "002.png", 200)
    for anomaly_type, stem in (("good", "001"), ("bad", "002")):
        path = source / "anomaly_maps" / "can" / "test" / anomaly_type / f"{stem}.tiff"
        path.parent.mkdir(parents=True, exist_ok=True)
        tiff.imwrite(path, np.arange(64, dtype=np.float32).reshape(8, 8))

    dataset = MVTecAD2Dataset(
        str(data),
        {"objects": ["can"], "preprocess": "no_mask_no_rotation"},
    )
    written = refine_maps(dataset, source, output)

    assert written == 2
    for anomaly_type, stem in (("good", "001"), ("bad", "002")):
        refined = tiff.imread(
            output / "anomaly_maps" / "can" / "test" / anomaly_type / f"{stem}.tiff",
        )
        assert refined.shape == (8, 8)
        assert refined.dtype == np.float32
        assert np.all(np.isfinite(refined))


def test_guided_refinement_supports_in_place_default_pipeline(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "output"
    image_path = data / "can" / "test_public" / "good" / "001.png"
    map_path = output / "anomaly_maps" / "can" / "test" / "good" / "001.tiff"
    _write_rgb(image_path, 80)
    map_path.parent.mkdir(parents=True)
    original = np.arange(64, dtype=np.float32).reshape(8, 8)
    tiff.imwrite(map_path, original)
    dataset = MVTecAD2Dataset(
        str(data),
        {"objects": ["can"], "preprocess": "no_mask_no_rotation"},
    )

    written = refine_maps(dataset, output, output)
    refined = tiff.imread(map_path)

    assert written == 1
    assert refined.shape == original.shape
    assert refined.dtype == np.float32
    assert np.all(np.isfinite(refined))
