from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src import post_eval


def test_parse_dataset_files_ignores_ground_truth_directory(tmp_path: Path) -> None:
    object_root = tmp_path / "can" / "test_public"
    for relative in ("good", "bad", "ground_truth/bad"):
        (object_root / relative).mkdir(parents=True)
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(object_root / "good" / "001.png")
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(object_root / "bad" / "002.png")
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(
        object_root / "ground_truth" / "bad" / "002_mask.png",
    )

    ground_truth, predictions = post_eval.parse_dataset_files(
        "can",
        tmp_path,
        tmp_path / "maps",
        dataset="MVTec_AD_2",
    )

    assert len(ground_truth) == len(predictions) == 2
    assert not any("/ground_truth/" in value for value in predictions)
    assert sum(value is None for value in ground_truth) == 1


def test_eval_segmentation_preserves_scores_above_float16_range(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gt_path = tmp_path / "mask.png"
    Image.fromarray(np.asarray([[0, 0], [255, 255]], dtype=np.uint8)).save(gt_path)
    score = np.asarray([[70_000.0, 80_000.0], [90_000.0, 100_000.0]], dtype=np.float32)
    monkeypatch.setattr(post_eval, "read_tiff", lambda _path: score)

    auroc, f1, threshold = post_eval.eval_segmentation(
        [str(gt_path)],
        [str(tmp_path / "prediction")],
        delete_tiff_files=False,
    )

    assert np.isfinite(auroc)
    assert f1 == 1.0
    assert threshold == 90_000.0
    assert np.isfinite(threshold)


def test_eval_segmentation_rejects_non_finite_input(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gt_path = tmp_path / "mask.png"
    Image.fromarray(np.zeros((1, 2), dtype=np.uint8)).save(gt_path)
    monkeypatch.setattr(
        post_eval,
        "read_tiff",
        lambda _path: np.asarray([[0.0, np.inf]], dtype=np.float32),
    )

    with np.testing.assert_raises_regex(ValueError, "non-finite"):
        post_eval.eval_segmentation(
            [str(gt_path)],
            [str(tmp_path / "prediction")],
            delete_tiff_files=False,
        )


def test_eval_segmentation_metrics_reports_perfect_aupro_at_point_zero_five(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gt_path = tmp_path / "mask.png"
    Image.fromarray(
        np.asarray([[255, 255], [0, 0]], dtype=np.uint8),
    ).save(gt_path)
    predictions = {
        "good": np.zeros((2, 2), dtype=np.float32),
        "bad": np.asarray([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32),
    }
    monkeypatch.setattr(post_eval, "read_tiff", lambda name: predictions[name])

    metrics = post_eval.eval_segmentation_metrics(
        [None, str(gt_path)],
        ["good", "bad"],
        pro_integration_limit=0.05,
        delete_tiff_files=False,
    )

    assert metrics.seg_auroc_0_05 == 1.0
    assert metrics.seg_f1_raw == 1.0
    assert metrics.seg_aupro_0_05 == 1.0
    assert metrics.prediction_map_count == 2
    assert metrics.ground_truth_map_count == 2
    assert metrics.anomalous_region_count == 1
    assert metrics.non_finite_prediction_count == 0


def test_eval_segmentation_preserves_legacy_three_tuple(monkeypatch) -> None:
    monkeypatch.setattr(
        post_eval,
        "eval_segmentation_metrics",
        lambda *_args, **_kwargs: post_eval.SegmentationMetrics(
            seg_auroc_0_05=0.8,
            seg_f1_raw=0.6,
            best_threshold=0.4,
            seg_aupro_0_05=0.7,
            prediction_map_count=2,
            ground_truth_map_count=2,
            anomalous_region_count=1,
            non_finite_prediction_count=0,
        ),
    )

    assert post_eval.eval_segmentation([], []) == (0.8, 0.6, 0.4)
