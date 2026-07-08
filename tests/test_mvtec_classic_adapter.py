from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pytest

from scripts import flow_tte_map_metrics
from scripts.flow_tte_map_metrics import (
    MapMetricSet,
    compute_map_metric_set,
    histogram_binary_metrics,
)
from scripts.flow_tte_mvtec_classic import ClassicMVTecDataset, build_evaluation_file_lists
from scripts.flow_tte_superadd_preprocess import (
    BrightnessRange,
    apply_brightness,
    parse_feature_layers,
    tile_starts,
)
from scripts.flow_tte_support import (
    greedy_coreset_indices,
    is_cls_coreset_policy,
    is_fixed_support_policy,
    merge_layer_features,
    select_support_paths,
    transform_rgb,
)


def write_placeholder(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("placeholder\n", encoding="utf-8")


def test_classic_mvtec_adapter_maps_defect_gt_and_prediction_paths(tmp_path: Path) -> None:
    root = tmp_path / "MVTecAD"
    write_placeholder(root / "bottle" / "train" / "good" / "000.png")
    write_placeholder(root / "bottle" / "test" / "good" / "001.png")
    write_placeholder(root / "bottle" / "test" / "broken" / "002.png")
    write_placeholder(root / "bottle" / "ground_truth" / "broken" / "002_mask.png")
    dataset = ClassicMVTecDataset(data_root=str(root), objects=("bottle",))

    files = build_evaluation_file_lists(dataset, tmp_path / "run", "bottle")

    assert dataset.get_train_images("bottle") == [
        str(root / "bottle" / "train" / "good" / "000.png"),
    ]
    assert files.gt_filenames == (
        str(root / "bottle" / "ground_truth" / "broken" / "002_mask.png"),
        None,
    )
    assert files.prediction_filenames == (
        str(tmp_path / "run" / "anomaly_maps" / "bottle" / "test" / "broken" / "002"),
        str(tmp_path / "run" / "anomaly_maps" / "bottle" / "test" / "good" / "001"),
    )


def test_histogram_metrics_report_perfect_pixel_auroc_ap() -> None:
    scores: npt.NDArray[np.float16] = np.array([0.0, 0.1, 2.0, 2.5], dtype=np.float16)
    labels: npt.NDArray[np.float64] = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    codes: npt.NDArray[np.uint16] = scores.view(np.uint16)
    total_counts: npt.NDArray[np.float64] = np.bincount(
        codes,
        minlength=65536,
    ).astype(np.float64)
    positive_counts: npt.NDArray[np.float64] = np.bincount(
        codes,
        weights=labels,
        minlength=65536,
    ).astype(np.float64)

    metrics = histogram_binary_metrics(total_counts, positive_counts)

    assert metrics.pixel_auroc == 1.0
    assert metrics.pixel_ap == 1.0


def test_map_metric_set_serializes_pixel_pro() -> None:
    metrics = MapMetricSet(
        image_auroc=0.1,
        pixel_auroc=0.2,
        image_ap=0.3,
        pixel_ap=0.4,
        pixel_pro=0.5,
        image_score_aggregation="mean_top_1_percent_full_resolution_map",
        pixel_score_quantization="float16_histogram",
        pixel_pro_max_fpr=0.3,
    )

    payload = metrics.as_dict()

    assert payload["pixel_PRO"] == 0.5
    assert payload["pixel_PRO_max_fpr"] == 0.3


def test_map_metric_set_uses_configurable_image_top_fraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions: dict[str, npt.NDArray[np.float32]] = {
        "normal": np.array([[0.0, 0.0], [0.0, 4.0]], dtype=np.float32),
        "anomaly": np.array([[0.0, 0.0], [3.0, 3.0]], dtype=np.float32),
    }

    def fake_read_prediction(prediction_name: str) -> npt.NDArray[np.float32]:
        return predictions[prediction_name]

    def fake_read_mask(
        gt_filename: Optional[str],
        shape: Sequence[int],
    ) -> npt.NDArray[np.bool_]:
        if gt_filename is None:
            return np.zeros(tuple(shape), dtype=np.bool_)
        return np.array([[False, False], [True, True]], dtype=np.bool_)

    def fake_pixel_pro(
        predictions: Sequence[npt.NDArray[np.float32]],
        masks: Sequence[npt.NDArray[np.bool_]],
        max_fpr: float,
    ) -> float:
        _ = predictions, masks, max_fpr
        return 0.5

    monkeypatch.setattr(flow_tte_map_metrics, "_read_tiff_without_ext", fake_read_prediction)
    monkeypatch.setattr(flow_tte_map_metrics, "_read_gt_mask", fake_read_mask)
    monkeypatch.setattr(flow_tte_map_metrics, "pixel_pro_score", fake_pixel_pro)

    top_quarter = compute_map_metric_set(
        (None, "mask"),
        ("normal", "anomaly"),
        image_top_fraction=0.25,
    )
    full_map = compute_map_metric_set(
        (None, "mask"),
        ("normal", "anomaly"),
        image_top_fraction=1.0,
    )

    assert top_quarter.image_auroc == 0.0
    assert top_quarter.image_score_aggregation == "mean_top_25_percent_full_resolution_map"
    assert full_map.image_auroc == 1.0
    assert full_map.image_score_aggregation == "mean_top_100_percent_full_resolution_map"


def test_visionad_support_selection_uses_seeded_without_replacement() -> None:
    paths = tuple(Path(f"{index:03d}.png") for index in range(10))

    selected = select_support_paths(
        paths,
        shots=4,
        policy="visionad_seeded_random",
        seed=1,
    )

    expected_indices = (6, 4, 9, 3)
    assert selected == tuple(paths[int(index)] for index in expected_indices)


def test_greedy_coreset_indices_start_near_mean_then_cover_extremes() -> None:
    features: npt.NDArray[np.float32] = np.array(
        [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [5.0, 5.0]],
        dtype=np.float32,
    )

    selected = greedy_coreset_indices(features, shots=3)

    assert selected == (3, 0, 1)


def test_cls_coreset_policy_accepts_backbone_neutral_aliases() -> None:
    assert is_cls_coreset_policy("cls_greedy_coreset")
    assert is_cls_coreset_policy("dinov2_cls_greedy_coreset")
    assert is_cls_coreset_policy("dinov3_cls_greedy_coreset")
    assert not is_cls_coreset_policy("first")


def test_fixed_json_support_selection_preserves_manifest_order(tmp_path: Path) -> None:
    paths = tuple(
        tmp_path / "can" / "train" / "good" / f"{index:03d}.png"
        for index in range(4)
    )
    manifest = tmp_path / "support.json"
    manifest.write_text(
        '{"can": ["' + str(paths[2]) + '", "' + str(paths[0]) + '"]}\n',
        encoding="utf-8",
    )

    selected = select_support_paths(
        paths,
        shots=2,
        policy=f"fixed_json={manifest}",
        seed=0,
    )

    assert is_fixed_support_policy(f"fixed_json={manifest}")
    assert selected == (paths[2], paths[0])


def test_fixed_json_support_selection_rejects_paths_outside_train(
    tmp_path: Path,
) -> None:
    paths = tuple(
        tmp_path / "can" / "train" / "good" / f"{index:03d}.png"
        for index in range(2)
    )
    manifest = tmp_path / "support.json"
    manifest.write_text(
        '{"can": ["' + str(tmp_path / "other.png") + '"]}\n',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="not in train/good"):
        select_support_paths(
            paths,
            shots=1,
            policy=f"fixed_json={manifest}",
            seed=0,
        )


def test_visionad_support_transforms_match_rotations_and_flips() -> None:
    image: npt.NDArray[np.uint8] = np.zeros((2, 2, 3), dtype=np.uint8)
    image[0, 0] = np.array([0, 1, 2], dtype=np.uint8)
    image[0, 1] = np.array([3, 4, 5], dtype=np.uint8)
    image[1, 0] = np.array([6, 7, 8], dtype=np.uint8)
    image[1, 1] = np.array([9, 10, 11], dtype=np.uint8)

    assert np.array_equal(transform_rgb(image, "rot90"), np.rot90(image, k=1))
    assert np.array_equal(transform_rgb(image, "rot180"), np.rot90(image, k=2))
    assert np.array_equal(transform_rgb(image, "rot270"), np.rot90(image, k=3))
    assert np.array_equal(transform_rgb(image, "flip_vertical"), np.flip(image, axis=0))
    assert np.array_equal(transform_rgb(image, "flip_horizontal"), np.flip(image, axis=1))


def test_visionad_feature_fusion_means_then_normalizes() -> None:
    layers: Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]] = (
        np.array([[3.0, 0.0], [0.0, 4.0]], dtype=np.float32),
        np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32),
    )

    fused = merge_layer_features(layers, "visionad_mean_l2")

    assert np.allclose(fused, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))


def test_superadd_feature_layers_parse_comma_or_space_list() -> None:
    assert parse_feature_layers("7,15,23,31") == (7, 15, 23, 31)
    assert parse_feature_layers("7 15 23 31") == (7, 15, 23, 31)


def test_superadd_brightness_range_is_seed_deterministic() -> None:
    brightness = BrightnessRange(0.8, 1.2)

    assert brightness.factor_for(index=3, seed=7) == brightness.factor_for(index=3, seed=7)


def test_superadd_brightness_clips_rgb_values() -> None:
    image: npt.NDArray[np.uint8] = np.array([[[100, 200, 250]]], dtype=np.uint8)

    adjusted = apply_brightness(image, 1.2)

    assert adjusted.tolist() == [[[120, 240, 255]]]


def test_superadd_tile_starts_cover_tail_with_overlap() -> None:
    assert tile_starts(length=1024, patch_size=640, overlap=128) == (0, 384)
    assert tile_starts(length=1280, patch_size=640, overlap=128) == (0, 512, 640)
