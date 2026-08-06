from __future__ import annotations

# ruff: noqa: COM812, SLF001, TC003
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from scripts import run_shift_conditioned_patch_flow as runner


def _support_tree(tmp_path: Path) -> tuple[Path, Path, dict[str, list[str]]]:
    data_root = tmp_path / "mvtec_ad_2"
    payload: dict[str, list[str]] = {}
    for object_name in runner.OBJECTS:
        directory = data_root / object_name / "train" / "good"
        directory.mkdir(parents=True)
        payload[object_name] = []
        for index in range(runner.SUPPORT_COUNT):
            path = directory / f"{index:03d}.png"
            path.write_bytes(f"{object_name}:{index}".encode())
            payload[object_name].append(f"/old/{object_name}/train/good/{path.name}")
    manifest = tmp_path / "support.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return data_root, manifest, payload


def test_frozen_config_matches_preregistration() -> None:
    config = runner.frozen_config("cpu")
    assert config["backbone"] == "dinov3_vith16plus"
    assert config["layers"] == [7, 15, 23, 31]
    assert config["feature_fusion"] == "layer_norm_mean"
    assert config["dvt"] == {"mode": "position_mean", "alpha": 1.0}
    assert config["stage_a_flow"]["n_coupling_layers"] == 2
    assert config["stage_a_flow"]["hidden_multiplier"] == 1
    assert config["stage_a_flow"]["n_epochs"] == 3
    assert config["stage_a_score"]["density_weight"] == 0.0
    assert config["stage_a_score"]["loo_standardize"] is True
    assert config["stage_b"]["epochs"] == 10
    assert config["score"]["latent_metric"] == "raw_euclidean"
    assert config["query_adaptation"] is False


def test_support_manifest_requires_exact_train_good_and_rejects_public(tmp_path: Path) -> None:
    data_root, manifest, payload = _support_tree(tmp_path)
    loaded = runner.load_support_manifest(manifest, data_root)
    assert set(loaded) == set(runner.OBJECTS)
    assert len(loaded["fabric"]) == 16
    assert all(entry.relative_path.startswith("fabric/train/good/") for entry in loaded["fabric"])

    payload["fabric"][0] = "/old/fabric/test_public/good/000.png"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="train/good"):
        runner.load_support_manifest(manifest, data_root)


def test_deterministic_photometric_specs_are_mixed_and_targeted() -> None:
    specs = runner.deterministic_transform_specs("vial", "vial/train/good/000.png", 3, "train")
    assert len(specs) == 10
    counts = {count: sum(len(spec.active_axes) == count for spec in specs) for count in (0, 1, 2)}
    assert counts == runner.TRANSFORM_MIX_COUNTS
    assert specs == runner.deterministic_transform_specs(
        "vial", "vial/train/good/000.png", 3, "train"
    )
    compound = next(spec for spec in specs if len(spec.active_axes) == 2)
    targets = compound.targets()
    assert set(targets["multi_label"]) == set(runner.PHOTOMETRIC_AXES)
    assert sum(targets["multi_label"].values()) == 2.0
    assert all(0.0 <= value <= 1.0 for value in targets["continuous"].values())


def test_photometric_transform_is_deterministic_and_identity_safe() -> None:
    image = np.arange(6 * 7 * 3, dtype=np.uint8).reshape(6, 7, 3)
    identity = runner.PhotometricTransformSpec("id", ())
    np.testing.assert_array_equal(runner.apply_photometric_transform(image, identity), image)
    spec = runner.PhotometricTransformSpec(
        "shift",
        ("brightness", "noise"),
        brightness=1.2,
        noise_std=0.01,
        seed=5,
    )
    first = runner.apply_photometric_transform(image, spec)
    second = runner.apply_photometric_transform(image, spec)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, image)


def test_public_phase_requires_machine_readable_pass_sentinel(tmp_path: Path) -> None:
    with pytest.raises(runner.RunnerError, match="sentinel"):
        runner.public_phase(
            runner.parse_args(
                [
                    "--phase",
                    "public",
                    "--data-root",
                    str(tmp_path),
                    "--support-json",
                    str(tmp_path / "support.json"),
                    "--output-root",
                    str(tmp_path / "out"),
                    "--objects",
                    "fruit_jelly",
                    "--mechanism-pass-sentinel",
                    str(tmp_path / "missing.txt"),
                ],
            ),
        )


def test_smoke_mode_writes_import_safe_contract(tmp_path: Path) -> None:
    output = tmp_path / "out"
    assert (
        runner.main(
            [
                "--phase",
                "smoke",
                "--output-root",
                str(output),
                "--objects",
                "fabric",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert (output / "frozen_config.json").is_file()
    assert (output / "access_ledger.json").is_file()
    assert (output / "run_manifest.json").is_file()
    summary = json.loads((output / "fabric" / "summary.json").read_text())
    assert summary["access_ledger"]["test_public_enumerated"] is False
    assert summary["mechanism"]["held_shift"]["correct_beats_zero_shuffled"] is False


def test_concrete_backend_trains_scores_and_roundtrips_checkpoint(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    support = tuple(
        runner.FeatureMap(
            values=rng.normal(size=(2, 2, 4)).astype(np.float32),
            grid=(2, 2),
            image_shape=(32, 32),
            relative_path=f"fabric/train/good/{index:03d}.png",
        )
        for index in range(3)
    )
    shifted = runner.FeatureMap(
        values=(support[0].values + np.float32(0.05)).astype(np.float32),
        grid=support[0].grid,
        image_shape=support[0].image_shape,
        relative_path=support[0].relative_path,
        transform="brightness",
    )
    pair = runner.StageBPair(
        source=support[0],
        transformed=shifted,
        transform=runner.PhotometricTransformSpec(
            name="brightness",
            active_axes=("brightness",),
            brightness=1.1,
        ),
    )
    config = deepcopy(runner.frozen_config("cpu"))
    config["stage_a_flow"]["n_epochs"] = 1
    config["stage_a_flow"]["batch_size"] = 32
    config["stage_b"]["epochs"] = 1

    backend = runner.TorchShiftConditionedBackend()
    backend.set_dvt_artifact(None, alpha=1.0)
    backend.fit_stage_a(support, config)
    log = backend.train_stage_b((pair,), config)
    before = backend.score_maps(support, arm="E5", descriptor_mode="correct")
    checkpoint = tmp_path / "checkpoint.pt"
    backend.save_checkpoint(checkpoint)

    restored = runner.TorchShiftConditionedBackend()
    restored.load_checkpoint(checkpoint, config)
    after = restored.score_maps(support, arm="E5", descriptor_mode="correct")

    assert log["pair_count"] == 1
    assert len(log["stage_b_epoch_losses"]) == 1
    assert [item.shape for item in before] == [(2, 2)] * 3
    for left, right in zip(before, after):
        np.testing.assert_allclose(left, right, atol=0.0, rtol=0.0)


def test_public_map_diagnostics_use_real_conditions_and_pixel_gt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    object_name = "fabric"
    rows = []
    maps = {
        ("good", "000_regular"): np.full((4, 4), 1.0, dtype=np.float32),
        ("good", "001_overexposed"): np.full((4, 4), 1.5, dtype=np.float32),
        ("good", "002_underexposed"): np.full((4, 4), 1.3, dtype=np.float32),
        ("bad", "003_shift_1"): np.asarray(
            [[4.0, 4.0, 1.0, 1.0]] * 4,
            dtype=np.float32,
        ),
    }
    for split, stem in maps:
        path = tmp_path / "maps" / split / f"{stem}.tiff"
        rows.append({"arm": "E0", "split": split, "stem": stem, "map": str(path)})
    monkeypatch.setattr(
        runner,
        "_read_tiff",
        lambda path: maps[(path.parent.name, path.stem)],
    )
    mask = np.zeros((4, 4), dtype=np.bool_)
    mask[:, :2] = True
    monkeypatch.setattr(runner, "_load_public_gt_mask", lambda **_kwargs: mask)

    result = runner._public_map_diagnostics(
        data_root=tmp_path,
        object_name=object_name,
        rows=rows,
    )

    assert result["shifted_good_condition_count"] == 2
    assert result["shifted_good_p95_inflation"] == pytest.approx(0.4)
    assert result["bad_image_gap_z"] > 1.9
    assert result["bad_image_top1_gt_share"] == 1.0


def test_shifted_good_reduction_is_relative() -> None:
    assert runner._relative_reduction(0.4, 0.3) == pytest.approx(0.25)
