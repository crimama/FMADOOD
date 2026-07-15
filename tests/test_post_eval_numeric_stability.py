from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src import post_eval


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
