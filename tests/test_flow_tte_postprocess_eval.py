from __future__ import annotations

# pyright: reportMissingImports=false
import json
from typing import TYPE_CHECKING

import numpy as np
import tifffile as tiff
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

from scripts.flow_tte_postprocess_core import EvalConfig, collect_object_rows, postprocess_mask


def test_closing_fill_morphology_connects_fragmented_line() -> None:
    mask = np.zeros((5, 5), dtype=np.bool_)
    mask[2, 1] = True
    mask[2, 3] = True

    processed = postprocess_mask(
        mask,
        line_length=3,
        angle_count=1,
        erosion_size=0,
    )

    assert processed[2, 1]
    assert processed[2, 2]
    assert processed[2, 3]


def test_collect_object_result_reports_morphology_f1_gain(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    run_root = tmp_path / "run"
    map_dir = run_root / "anomaly_maps" / "can" / "test" / "scratch"
    gt_dir = data_root / "can" / "test_public" / "ground_truth" / "scratch"
    map_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    prediction = np.zeros((5, 5), dtype=np.float32)
    prediction[2, 1] = 1.0
    prediction[2, 3] = 1.0
    tiff.imwrite(map_dir / "000.tiff", prediction)

    gt_mask = np.zeros((5, 5), dtype=np.uint8)
    gt_mask[2, 1:4] = 255
    Image.fromarray(gt_mask).save(gt_dir / "000_mask.png")

    (run_root / "metrics.json").write_text(
        json.dumps(
            {
                "can": {
                    "best_thre": 0.5,
                    "seg_AUROC": 0.9,
                    "seg_F1": 0.8,
                },
            },
        )
        + "\n",
        encoding="utf-8",
    )

    rows = collect_object_rows(
        EvalConfig(
            data_root=data_root,
            run_roots=(run_root,),
            threshold_count=8,
            line_length=3,
            angle_count=1,
        ),
        "can",
    )

    f1_by_variant = {row.variant: row.f1 for row in rows}
    assert f1_by_variant["raw_at_metrics_best"] == 0.8
    assert "closefill_at_metrics_best" in f1_by_variant
    assert "closefill_oracle_grid" in f1_by_variant
