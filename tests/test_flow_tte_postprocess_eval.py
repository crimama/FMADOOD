from __future__ import annotations

# pyright: reportMissingImports=false
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile as tiff
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

from scripts.flow_tte_postprocess_core import (
    EvalConfig,
    binary_mask_metrics,
    collect_object_rows,
    postprocess_mask,
    variant_profile,
)
from scripts.run_flow_tte_mvtec_ad2 import evaluate_threshold_protocols, parse_args


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


def test_closefill_erode_binary_metrics_preserve_explicit_contract() -> None:
    score = np.zeros((7, 7), dtype=np.float32)
    score[3, 1] = 1.0
    score[3, 5] = 1.0
    gt = np.zeros((7, 7), dtype=np.bool_)
    gt[3, 2:5] = True

    metrics = binary_mask_metrics(
        (score,),
        (gt,),
        threshold=0.5,
        profile=variant_profile("closefill_erode"),
        line_length=5,
        angle_count=1,
    )

    assert 0.0 <= metrics.f1 <= 1.0
    assert 0.0 <= metrics.positive_area <= 1.0


def test_ad2_runner_defaults_to_rgb_guide_and_closefill_erode() -> None:
    config = parse_args(["--data-root", "/tmp/data", "--output-root", "/tmp/output"])

    assert config.rgb_guide == "guided_r8"
    assert config.binary_postprocess == "closefill_erode"
    assert config.morphology_line_length == 17
    assert config.morphology_angle_count == 16


def test_ad2_runner_can_restore_raw_map_ablation() -> None:
    config = parse_args([
        "--data-root",
        "/tmp/data",
        "--output-root",
        "/tmp/output",
        "--rgb-guide",
        "none",
        "--binary-postprocess",
        "none",
    ])

    assert config.rgb_guide == "none"
    assert config.binary_postprocess == "none"


def test_ad2_runner_accepts_official_superadd_full_normal_contract() -> None:
    config = parse_args([
        "--data-root", "/tmp/data",
        "--output-root", "/tmp/output",
        "--shots", "0",
        "--support-selection", "superadd_full_7of8",
        "--threshold-calibration-mode", "superadd_train95",
        "--expansion-budget", "1.0",
        "--latent-bank-subsample", "superadd_knn_score",
    ])

    assert config.shots == 0
    assert config.threshold_fraction == 8
    assert config.threshold_percentile == 95.0
    assert config.threshold_factor == 1.421


def test_superadd_and_oracle_threshold_metrics_share_one_test_map_set(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "output"
    calibration_dir = output_root / "calibration_maps" / "can" / "good"
    good_dir = output_root / "anomaly_maps" / "can" / "test" / "good"
    bad_dir = output_root / "anomaly_maps" / "can" / "test" / "bad"
    gt_dir = data_root / "can" / "test_public" / "ground_truth" / "bad"
    for path in (calibration_dir, good_dir, bad_dir, gt_dir):
        path.mkdir(parents=True)
    tiff.imwrite(calibration_dir / "000.tiff", np.full((4, 4), 0.1, dtype=np.float32))
    tiff.imwrite(good_dir / "001.tiff", np.full((4, 4), 0.05, dtype=np.float32))
    bad_score = np.zeros((4, 4), dtype=np.float32)
    bad_score[1:3, 1:3] = 1.0
    tiff.imwrite(bad_dir / "002.tiff", bad_score)
    gt = np.zeros((4, 4), dtype=np.uint8)
    gt[1:3, 1:3] = 255
    Image.fromarray(gt).save(gt_dir / "002_mask.png")
    config = parse_args([
        "--data-root", str(data_root),
        "--output-root", str(output_root),
        "--objects", "can",
        "--shots", "0",
        "--support-selection", "superadd_full_7of8",
        "--threshold-calibration-mode", "superadd_train95",
        "--expansion-budget", "1.0",
        "--latent-bank-subsample", "superadd_knn_score",
    ])
    oracle = {
        "can": {
            "seg_AUROC": 0.9,
            "seg_F1": 0.8,
            "seg_F1_raw": 1.0,
            "best_thre": 0.5,
        },
        "mean_segmentation_au_roc": 0.9,
        "mean_segmentation_f1": 0.8,
    }

    result = evaluate_threshold_protocols(config, oracle)
    per_object = result["per_object"]["can"]

    assert per_object["calibration_map_count"] == 1
    assert per_object["superadd_threshold"] == pytest.approx(0.1421)
    assert per_object["superadd_raw_f1"] == 1.0
    assert per_object["oracle_raw_f1"] == 1.0
    assert result["mean_seg_AUROC_0.05"] == 0.9


def test_ad2_evaluator_records_raw_and_default_postprocessed_f1(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from fmad.evaluation import metrics as metrics_module

    map_dir = tmp_path / "maps" / "can" / "test"
    map_dir.mkdir(parents=True)
    (tmp_path / "output").mkdir()
    gt_path = tmp_path / "gt.png"
    gt = np.zeros((7, 7), dtype=np.uint8)
    gt[3, 2:5] = 255
    Image.fromarray(gt).save(gt_path)
    score = np.zeros((7, 7), dtype=np.float32)
    score[3, 1] = 1.0
    score[3, 5] = 1.0

    monkeypatch.setattr(
        metrics_module,
        "parse_dataset_files",
        lambda **_kwargs: ([str(gt_path)], ["prediction"]),
    )
    monkeypatch.setattr(
        metrics_module,
        "eval_segmentation",
        lambda *_args, **_kwargs: (0.9, 0.4, 0.5),
    )
    monkeypatch.setattr(metrics_module, "read_tiff", lambda _path: score)

    evaluator = metrics_module.Evaluator({
        "binary_postprocess": "closefill_erode",
        "morphology_line_length": 5,
        "morphology_angle_count": 1,
    })
    result = evaluator.evaluate_run(
        dataset_name="MVTec_AD_2",
        data_root=str(tmp_path),
        anomaly_maps_dir=str(tmp_path / "maps"),
        output_dir=str(tmp_path / "output"),
        objects=["can"],
    )

    assert result["can"]["seg_AUROC"] == 0.9
    assert result["can"]["seg_F1_raw"] == 0.4
    assert result["can"]["binary_postprocess"] == "closefill_erode"
    assert result["can"]["binary_postprocess_threshold_source"] == "raw_best_thre"
    assert result["mean_segmentation_f1"] == result["can"]["seg_F1"]


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
