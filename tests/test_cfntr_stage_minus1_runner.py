from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from flow_tte.cfntr import apply_rgb_transform
from scripts.run_cfntr_stage_minus1 import (
    AXES,
    BASIS_BOOTSTRAPS,
    DEFECT_KINDS,
    LAYERS,
    OBJECTS,
    RANDOM_BASIS_COUNT,
    SUPPORT_COUNT,
    assigned_defect_shift,
    defect_feature_residual,
    deterministic_sign,
    expected_audit_counts,
    load_support_manifest,
    make_synthetic_defect,
    mask_to_patch_grid,
    parse_args,
    pixel_metrics,
    random_null_columns,
    validation_specs,
)

if TYPE_CHECKING:
    from pathlib import Path


def _support_tree(tmp_path: Path) -> tuple[Path, Path, dict[str, list[str]]]:
    data_root = tmp_path / "mvtec_ad_2"
    payload: dict[str, list[str]] = {}
    for object_name in OBJECTS:
        directory = data_root / object_name / "train" / "good"
        directory.mkdir(parents=True)
        payload[object_name] = []
        for index in range(SUPPORT_COUNT):
            path = directory / f"{index:03d}_regular.png"
            path.write_bytes(f"{object_name}:{index}".encode())
            payload[object_name].append(
                f"/old/data/{object_name}/train/good/{path.name}",
            )
    manifest = tmp_path / "support.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return data_root, manifest, payload


def test_parse_args_freezes_protocol_capacity_and_layers() -> None:
    args = parse_args(
        [
            "--data-root",
            "/data",
            "--support-json",
            "/support.json",
            "--output-root",
            "/out",
            "--objects",
            "can,fabric",
            "--host-gpu-id",
            "0",
        ],
    )
    assert args.objects == ("can", "fabric")
    assert args.basis_bootstraps == BASIS_BOOTSTRAPS == 200
    assert LAYERS == (5, 11, 17, 23)
    assert RANDOM_BASIS_COUNT == 16

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--data-root",
                "/data",
                "--support-json",
                "/support.json",
                "--output-root",
                "/out",
                "--objects",
                "can",
                "--host-gpu-id",
                "0",
                "--basis-bootstraps",
                "10",
            ],
        )


def test_support_manifest_is_exact_train_good_and_four_by_four_sha_folded(
    tmp_path: Path,
) -> None:
    data_root, manifest, _ = _support_tree(tmp_path)

    loaded = load_support_manifest(manifest, data_root)

    assert set(loaded) == set(OBJECTS)
    assert all(len(entries) == SUPPORT_COUNT for entries in loaded.values())
    assert sorted(entry.fold for entry in loaded["can"]) == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4
    assert all("/train/good/" in f"/{entry.relative_path}" for entry in loaded["can"])
    assert all("test_public" not in entry.relative_path for entry in loaded["can"])


def test_support_manifest_rejects_public_or_private_path_even_when_name_exists(
    tmp_path: Path,
) -> None:
    data_root, manifest, payload = _support_tree(tmp_path)
    payload["can"][0] = "/old/data/can/test_public/good/000_regular.png"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="train/good"):
        load_support_manifest(manifest, data_root)

    payload["can"][0] = "/old/data/can/test_private/good/000_regular.png"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="train/good"):
        load_support_manifest(manifest, data_root)


def test_fold_assignment_is_invariant_to_data_root_rebase(tmp_path: Path) -> None:
    first_root, first_manifest, payload = _support_tree(tmp_path / "first")
    first = load_support_manifest(first_manifest, first_root)
    second_root = tmp_path / "second" / "mvtec_ad_2"
    for object_name in OBJECTS:
        directory = second_root / object_name / "train" / "good"
        directory.mkdir(parents=True)
        for index in range(SUPPORT_COUNT):
            (directory / f"{index:03d}_regular.png").write_bytes(
                f"{object_name}:{index}".encode(),
            )
    second_manifest = tmp_path / "second_support.json"
    second_manifest.write_text(json.dumps(payload), encoding="utf-8")

    second = load_support_manifest(second_manifest, second_root)

    assert {entry.relative_path: entry.fold for entry in first["wallplugs"]} == {
        entry.relative_path: entry.fold for entry in second["wallplugs"]
    }


def test_validation_transforms_are_six_sealed_double_amplitude_specs() -> None:
    first = validation_specs("can", "can/train/good/000_regular.png")
    second = validation_specs("can", "can/train/good/000_regular.png")

    assert len(first) == len(AXES) + 2 == 6
    assert first == second
    assert all(
        "x2" in name or name.startswith(("exposure_", "gamma_", "contrast_", "white_balance_"))
        for name, _ in first
    )
    assert deterministic_sign("can", "path", "axis") in (-1, 1)


@pytest.mark.parametrize("kind", DEFECT_KINDS)
def test_sealed_synthetic_defects_are_deterministic_local_and_nonempty(kind: str) -> None:
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)

    first_image, first_mask, first_meta = make_synthetic_defect(
        image,
        object_name="can",
        relative_path="can/train/good/000_regular.png",
        kind=kind,
    )
    second_image, second_mask, second_meta = make_synthetic_defect(
        image,
        object_name="can",
        relative_path="can/train/good/000_regular.png",
        kind=kind,
    )

    assert np.array_equal(first_image, second_image)
    assert np.array_equal(first_mask, second_mask)
    assert first_meta == second_meta
    assert 0 < int(first_mask.sum()) < first_mask.size
    assert first_image.shape == image.shape
    assert first_mask.shape == image.shape[:2]
    assert np.any(first_image[first_mask] != image[first_mask])
    if kind == "cutpaste":
        dy, dx, height, width = first_meta["destination"]
        sy, sx = first_meta["source"]
        assert sy + height <= dy or dy + height <= sy or sx + width <= dx or dx + width <= sx


def test_feature_retry_is_deterministic_and_uses_a_distinct_seed() -> None:
    rng = np.random.default_rng(19)
    image = rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)
    kwargs = {
        "object_name": "vial",
        "relative_path": "vial/train/good/043_regular.png",
        "kind": "cutpaste",
    }

    first = make_synthetic_defect(image, feature_retry=1, **kwargs)
    repeated = make_synthetic_defect(image, feature_retry=1, **kwargs)
    second = make_synthetic_defect(image, feature_retry=2, **kwargs)

    assert np.array_equal(first[0], repeated[0])
    assert np.array_equal(first[1], repeated[1])
    assert first[2] == repeated[2]
    assert first[2]["feature_retry"] == 1
    assert first[2]["seed"] != second[2]["seed"]
    assert not np.array_equal(first[0], second[0])


def test_shifted_defect_has_a_same_acquisition_clean_baseline() -> None:
    rng = np.random.default_rng(11)
    clean = rng.integers(0, 256, size=(64, 80, 3), dtype=np.uint8)
    defective, _, _ = make_synthetic_defect(
        clean,
        object_name="rice",
        relative_path="rice/train/good/001_regular.png",
        kind="scratch",
    )
    _, spec = assigned_defect_shift(
        "rice",
        "rice/train/good/001_regular.png",
        "scratch",
    )

    shifted_clean = apply_rgb_transform(clean, spec)
    shifted_defect = apply_rgb_transform(defective, spec)

    assert shifted_clean.shape == shifted_defect.shape
    assert not np.array_equal(shifted_clean, clean)
    assert np.any(shifted_defect != shifted_clean)

    clean_features = np.zeros((4, 6), dtype=np.float32)
    shifted_clean_features = np.full((4, 6), 3.0, dtype=np.float32)
    shifted_defect_features = np.full((4, 6), 4.0, dtype=np.float32)
    residual = defect_feature_residual(
        shifted_defect_features,
        acquisition="shifted",
        clean_features=clean_features,
        shifted_clean_features=shifted_clean_features,
    )
    assert np.array_equal(residual, np.ones((4, 6), dtype=np.float32))


def test_all_sixteen_random_nulls_are_retained_before_envelope() -> None:
    columns = random_null_columns(
        list(np.linspace(0.0, 1.0, RANDOM_BASIS_COUNT)),
        list(np.linspace(-1.0, 0.0, RANDOM_BASIS_COUNT)),
    )

    assert len(columns) == 2 * RANDOM_BASIS_COUNT
    assert "random_energy_capture_00" in columns
    assert "random_energy_capture_15" in columns
    assert "random_drift_reduction_00" in columns
    assert "random_drift_reduction_15" in columns


def test_pixel_metrics_match_repository_partial_auroc_convention() -> None:
    mask = np.zeros((12, 12), dtype=bool)
    mask[4:8, 4:8] = True
    score = np.zeros((12, 12), dtype=np.float32)
    score[mask] = 1.0

    metrics = pixel_metrics(mask, score)

    assert metrics["seg_AUROC_0.05"] == pytest.approx(1.0)
    assert metrics["seg_F1"] == pytest.approx(1.0)


def test_native_mask_is_flattened_to_match_flat_patch_scores() -> None:
    mask = np.zeros((16, 20), dtype=bool)
    mask[2:9, 3:12] = True

    patch_mask = mask_to_patch_grid(mask, (4, 5))

    assert patch_mask.shape == (20,)
    assert patch_mask.dtype == np.bool_
    assert np.any(patch_mask)


def test_native_subpatch_overlap_is_not_quantized_away() -> None:
    mask = np.zeros((16, 16), dtype=bool)
    mask[1, 1] = True

    patch_mask = mask_to_patch_grid(mask, (4, 4))

    assert int(patch_mask.sum()) == 1
    assert not np.all(patch_mask)


def test_g5_exact_per_object_arm_counts_are_frozen() -> None:
    assert expected_audit_counts() == {
        "arms_per_case": 19,
        "shift_rows": 384,
        "defect_rows": 512,
        "shift_maps": 7296,
        "defect_maps": 9728,
        "total_maps": 17024,
    }
