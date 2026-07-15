from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile as tiff
from PIL import Image

from scripts.aggregate_mvtecad1_metric_chunks import aggregate
from scripts.flow_tte_mvtec_classic import ClassicMVTecDataset
from scripts.run_flow_tte_mvtecad1_guided_refinement import refine_maps


def _write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)).save(path)


def test_guided_refinement_preserves_classic_map_layout(tmp_path: Path) -> None:
    data = tmp_path / "data"
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_rgb(data / "bottle" / "train" / "good" / "000.png", 0)
    _write_rgb(data / "bottle" / "test" / "good" / "001.png", 20)
    _write_rgb(data / "bottle" / "test" / "broken" / "002.png", 200)
    for anomaly_type, stem in (("good", "001"), ("broken", "002")):
        path = source / "anomaly_maps" / "bottle" / "test" / anomaly_type / f"{stem}.tiff"
        path.parent.mkdir(parents=True, exist_ok=True)
        tiff.imwrite(path, np.arange(64, dtype=np.float32).reshape(8, 8))

    dataset = ClassicMVTecDataset(str(data), ("bottle",))
    written = refine_maps(dataset, source, output)

    assert written == 2
    for anomaly_type, stem in (("good", "001"), ("broken", "002")):
        refined = tiff.imread(
            output / "anomaly_maps" / "bottle" / "test" / anomaly_type / f"{stem}.tiff",
        )
        assert refined.shape == (8, 8)
        assert refined.dtype == np.float32
        assert np.all(np.isfinite(refined))


def test_metric_chunk_aggregation_uses_per_object_rows(tmp_path: Path) -> None:
    paths = []
    for index, object_name in enumerate(("bottle", "cable"), start=1):
        row = {metric: float(index) for metric in (
            "i_AUROC", "i_AUPRC", "p_AUROC", "p_AUPRC", "p_AUPRO"
        )}
        path = tmp_path / f"chunk{index}.json"
        path.write_text(json.dumps({"per_object": {object_name: row}}), encoding="utf-8")
        paths.append(path)

    metrics = aggregate(paths, ("bottle", "cable"))

    assert metrics["i_AUROC"] == 1.5
    assert metrics["p_AUPRO"] == 1.5
    assert tuple(metrics["per_object"]) == ("bottle", "cable")
