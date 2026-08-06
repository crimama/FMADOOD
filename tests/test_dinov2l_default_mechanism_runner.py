from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_dinov2l_default_mechanism_stage_m1.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("dinov2l_mechanism_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load_runner()


def test_frozen_arm_factorization_and_row_budget(runner) -> None:
    arms = runner.arm_specs()
    assert tuple(arms) == (
        "A0",
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
        "A7",
        "L5",
        "L11",
        "L17",
        "L23",
    )
    assert arms["A0"].state == arms["A4"].state == "dvt_flow"
    assert arms["A1"].state == arms["A6"].state == "raw_flow"
    assert arms["A2"].state == "dvt_flow" and not arms["A2"].query_dvt
    assert arms["A3"].state == "raw_flow" and arms["A3"].query_dvt
    assert arms["A5"].state == "dvt_identity"
    assert arms["A7"].state == "raw_identity"
    assert runner.expected_row_counts() == {
        "C": 192,
        "S": 3072,
        "D": 768,
        "SD": 12288,
        "total": 16320,
    }


def test_shift_grid_is_complete_and_unique(runner) -> None:
    specs = runner.shift_specs()
    keys = {(axis, sign, magnitude) for axis, sign, magnitude, _ in specs}
    assert len(specs) == len(keys) == 16
    assert {axis for axis, _, _ in keys} == set(runner.AXES)
    assert {sign for _, sign, _ in keys} == {-1, 1}
    assert {magnitude for _, _, magnitude in keys} == {1, 2}


def test_clean_custody_hash_is_bound_to_selected_source(runner) -> None:
    first = np.zeros((4, 5, 3), dtype=np.uint8)
    second = np.full((4, 5, 3), 17, dtype=np.uint8)
    image, metadata = runner.clean_source_with_metadata(
        {"first.png": first, "second.png": second}, "first.png",
    )

    np.testing.assert_array_equal(image, first)
    assert metadata == {"evaluated_image_sha256": runner.sha256_bytes(first.tobytes())}
    assert metadata["evaluated_image_sha256"] != runner.sha256_bytes(second.tobytes())


def test_basic_default_config_is_frozen(runner) -> None:
    config = runner.frozen_config("real")
    assert config["backbone"] == "dinov2_vitl14"
    assert config["layers"] == [5, 11, 17, 23]
    assert config["feature_fusion"] == "layer_norm_mean"
    assert config["dvt"] == {"mode": "position_mean", "alpha": 1.0}
    assert config["flow"]["n_coupling_layers"] == 2
    assert config["flow"]["hidden_multiplier"] == 1
    assert config["flow"]["n_epochs"] == 3
    assert config["flow"]["lr"] == pytest.approx(2e-4)
    assert config["score"]["density_weight"] == pytest.approx(0.25)
    assert config["score"]["loo_standardization"] is False
    assert config["parity"]["cpu_fixture_atol"] == pytest.approx(1e-6)
    assert config["parity"]["cuda_chunked_vs_full_cdist_atol"] == pytest.approx(1e-4)
    assert config["parity"]["nearest_neighbor_index_mismatches_allowed"] == 0
    assert config["tta"] is False and config["query_update"] is False
    assert config["test_public_access"] is False
    assert config["test_private_access"] is False


def test_smooth_warp_is_deterministic_visible_and_local(runner) -> None:
    yy, xx = np.mgrid[:96, :112]
    image = np.stack((xx, yy, (xx + yy) % 255), axis=-1).astype(np.uint8)
    first, first_mask, first_meta = runner.make_smooth_local_warp(
        image,
        object_name="can",
        relative_path="can/train/good/fixture.png",
    )
    second, second_mask, second_meta = runner.make_smooth_local_warp(
        image,
        object_name="can",
        relative_path="can/train/good/fixture.png",
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_mask, second_mask)
    assert first_meta == second_meta
    assert first_mask.any() and not first_mask.all()
    assert np.any(first != image)
    assert first_meta["mask_pixels"] == int(first_mask.sum())


def test_smoke_writes_atomic_compact_contract(runner, tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    assert runner.main(
        [
            "--mode",
            "smoke",
            "--output-root",
            str(output),
            "--objects",
            "can",
            "--device",
            "cpu",
            "--host-gpu-id",
            "smoke",
            "--container-cuda-slot",
            "cpu",
        ]
    ) == 0
    assert (output / "frozen_config.json").is_file()
    assert (output / "access_ledger.json").is_file()
    assert (output / "run_manifest.json").is_file()
    assert (output / "objects" / "can.jsonl").is_file()
    assert (output / "objects" / "can_summary.json").is_file()
    leftovers = [path for path in output.rglob(".*") if path.is_file()]
    assert leftovers == []
