from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false
from pathlib import Path  # noqa: TC003 -- pytest injects concrete Path fixtures

import numpy as np
import pytest
import tifffile as tiff

from flow_tte.darc_ad2_pilot import (
    RawRungMaps,
    ladder_coverage,
    mean_rung_maps,
    raw_rung_maps,
)
from flow_tte.darc_ad2_pilot_io import (
    PilotMapTarget,
    PilotTestImage,
    discover_test_images,
    write_rung_maps,
)
from flow_tte.darc_ad2_pilot_io import (
    TestLimits as PilotTestLimits,
)
from flow_tte.darc_ad2_pilot_runtime import (
    PilotRuntimeConfig,
    claim_fresh_output_root,
    pilot_query_id,
    prepare_pilot,
)
from flow_tte.darc_gate2_pipeline_types import (
    CropLadderResult,
    QueryLadderAudit,
    QueryLadderResult,
)
from flow_tte.darc_gate2_scoring_types import RungScores, SupportValidityAudit
from flow_tte.darc_geometry import ImageSize
from flow_tte.darc_map_io import Population
from flow_tte.darc_tiling import NativeCrop


def _ladder_result() -> QueryLadderResult:
    values = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    valid = np.ones(4, dtype=np.bool_)
    support_valid = np.ones((4, 5), dtype=np.bool_)
    scores = RungScores(
        g0=values,
        g0_valid=valid,
        l0=np.asarray(values + 10.0, dtype=np.float32),
        l1=np.asarray(values + 20.0, dtype=np.float32),
        r1=np.asarray(values + 30.0, dtype=np.float32),
        common_fallback=np.zeros(4, dtype=np.bool_),
        support_validity=SupportValidityAudit(
            l0=support_valid,
            l1=support_valid,
            shared=support_valid,
            r1=support_valid,
        ),
    )
    return QueryLadderResult(
        query_id="can/test_public/good/000.png",
        native_size=ImageSize(4, 4),
        selected_support_ids=("a", "b", "c", "d", "e"),
        crops=(
            CropLadderResult(
                crop_index=0,
                crop=NativeCrop(0, 0, 4, 4),
                token_shape=ImageSize(2, 2),
                scores=scores,
            ),
        ),
        registration_audit=(),
        audit=QueryLadderAudit("population", "support", "fallback"),
    )


def test_raw_rung_maps_stitches_every_ladder_arm_to_native_grid() -> None:
    # Given: one raw ladder crop with distinct values for every rung.
    # When: the pilot converts token residuals to native maps.
    maps = raw_rung_maps(_ladder_result())

    # Then: all arms preserve their offsets on the complete native grid.
    assert maps.g0.shape == (4, 4)
    np.testing.assert_allclose(maps.l0 - maps.g0, 10.0)
    np.testing.assert_allclose(maps.l1 - maps.g0, 20.0)
    np.testing.assert_allclose(maps.r1 - maps.g0, 30.0)


def test_mean_rung_maps_averages_folds_in_float64_and_casts_once() -> None:
    # Given: two fold map bundles with float32 inputs.
    low = np.asarray([[1.0, 3.0]], dtype=np.float32)
    high = np.asarray([[3.0, 7.0]], dtype=np.float32)
    first = RawRungMaps(low, low, low, low)
    second = RawRungMaps(high, high, high, high)

    # When: the pilot performs its registered fold aggregation.
    averaged = mean_rung_maps((first, second))

    # Then: every rung has the arithmetic mean and one float32 output cast.
    expected = np.asarray([[2.0, 5.0]], dtype=np.float32)
    assert averaged.g0.dtype == np.float32
    np.testing.assert_array_equal(averaged.g0, expected)
    np.testing.assert_array_equal(averaged.l0, expected)
    np.testing.assert_array_equal(averaged.l1, expected)
    np.testing.assert_array_equal(averaged.r1, expected)


def test_ladder_coverage_records_fallback_and_support_validity_population() -> None:
    # Given: four nonfallback tokens with all five supports valid at every rung.
    # When: the label-free structural audit is compacted.
    coverage = ladder_coverage(_ladder_result())

    # Then: the complete scorer population and support-count histograms remain auditable.
    assert coverage.token_count == 4
    assert coverage.nonfallback_count == 4
    assert coverage.fallback_fraction == 0.0
    assert coverage.registration_count == 0
    assert coverage.accepted_registration_count == 0
    assert coverage.l0_support_histogram == (0, 0, 0, 0, 0, 4)
    assert coverage.l1_support_histogram == (0, 0, 0, 0, 0, 4)
    assert coverage.shared_support_histogram == (0, 0, 0, 0, 0, 4)
    assert coverage.r1_support_histogram == (0, 0, 0, 0, 0, 4)


def test_discover_test_images_applies_deterministic_per_population_limits(
    tmp_path: Path,
) -> None:
    # Given: unsorted AD2 good and bad image files.
    populations = (
        ("good", ("002.png", "000.png", "001.png")),
        ("bad", ("b.png", "a.png")),
    )
    for population, names in populations:
        directory = tmp_path / "can" / "test_public" / population
        directory.mkdir(parents=True)
        for name in names:
            (directory / name).touch()

    # When: the pilot requests a bounded smoke population.
    images = discover_test_images(tmp_path, "can", PilotTestLimits(good=2, bad=1))

    # Then: sorting and limits are independent for good and bad images.
    assert tuple((item.population.value, item.path.name) for item in images) == (
        ("good", "000.png"),
        ("good", "001.png"),
        ("bad", "a.png"),
    )


def test_write_rung_maps_emits_common_evaluator_layout(tmp_path: Path) -> None:
    # Given: one scored bad image and four distinct raw rung maps.
    source = tmp_path / "source" / "bad" / "sample.png"
    source.parent.mkdir(parents=True)
    source.touch()
    image = PilotTestImage(Population.BAD, source)
    maps = RawRungMaps(
        *(np.full((2, 3), value, dtype=np.float32) for value in (1.0, 2.0, 3.0, 4.0)),
    )

    # When: the pilot writes the completed image bundle.
    write_rung_maps(PilotMapTarget(tmp_path / "run", "can"), image, maps)

    # Then: every arm is readable from the common evaluator's TIFF layout.
    for arm, expected in (("G0", 1.0), ("L0", 2.0), ("L1", 3.0), ("R1", 4.0)):
        path = (
            tmp_path
            / "run"
            / "arms"
            / arm
            / "anomaly_maps"
            / "can"
            / "test"
            / "bad"
            / "sample.tiff"
        )
        np.testing.assert_array_equal(
            tiff.imread(path),
            np.full((2, 3), expected, dtype=np.float32),
        )


def test_prepare_pilot_freezes_supports_before_returning_bounded_queries(tmp_path: Path) -> None:
    # Given: one AD2 object with exactly 16 normals and two public populations.
    train = tmp_path / "can" / "train" / "good"
    train.mkdir(parents=True)
    for index in range(16):
        (train / f"{index:03d}.png").touch()
    for population in ("good", "bad"):
        directory = tmp_path / "can" / "test_public" / population
        directory.mkdir(parents=True)
        (directory / "000.png").touch()

    # When: seed 0 and fold 0 are prepared without feature extraction.
    prepared = prepare_pilot(
        PilotRuntimeConfig(
            data_root=tmp_path,
            output_root=tmp_path / "output",
            object_name="can",
            device="cpu",
            seed=0,
            fold_indices=(0,),
            test_limits=PilotTestLimits(good=1, bad=1),
        ),
    )

    # Then: the full P16 identity and exact 12/4 fold are fixed before scoring.
    assert len(prepared.split.support_paths) == 16
    assert len(prepared.folds) == 1
    assert len(prepared.folds[0].memory_paths) == 12
    assert len(prepared.folds[0].calibration_paths) == 4
    assert tuple(item.population.value for item in prepared.test_images) == ("good", "bad")


def test_prepare_pilot_shards_queries_without_changing_the_p16_split(tmp_path: Path) -> None:
    # Given: one P16 pool and eight ordered public queries.
    train = tmp_path / "can" / "train" / "good"
    train.mkdir(parents=True)
    for index in range(16):
        (train / f"{index:03d}.png").touch()
    for population in ("good", "bad"):
        directory = tmp_path / "can" / "test_public" / population
        directory.mkdir(parents=True)
        for index in range(4):
            (directory / f"{index:03d}.png").touch()

    # When: the same frozen cell is partitioned into two operational shards.
    first = prepare_pilot(
        PilotRuntimeConfig(
            data_root=tmp_path,
            output_root=tmp_path / "output",
            object_name="can",
            device="cpu",
            seed=0,
            fold_indices=(0,),
            shard_index=0,
            shard_count=2,
        ),
    )
    second = prepare_pilot(
        PilotRuntimeConfig(
            data_root=tmp_path,
            output_root=tmp_path / "output",
            object_name="can",
            device="cpu",
            seed=0,
            fold_indices=(0,),
            shard_index=1,
            shard_count=2,
        ),
    )

    # Then: supports are identical and query identities are disjoint and exhaustive.
    assert first.split.support_paths == second.split.support_paths
    first_ids = tuple(item.path.relative_to(tmp_path).as_posix() for item in first.test_images)
    second_ids = tuple(item.path.relative_to(tmp_path).as_posix() for item in second.test_images)
    assert first_ids == (
        "can/test_public/good/000.png",
        "can/test_public/good/002.png",
        "can/test_public/bad/000.png",
        "can/test_public/bad/002.png",
    )
    assert second_ids == (
        "can/test_public/good/001.png",
        "can/test_public/good/003.png",
        "can/test_public/bad/001.png",
        "can/test_public/bad/003.png",
    )
    assert set(first_ids).isdisjoint(second_ids)


def test_claim_fresh_output_root_rejects_rerun_before_coverage_can_append(
    tmp_path: Path,
) -> None:
    # Given: one shard has already claimed its run root and persisted coverage.
    output_root = tmp_path / "shard=0"
    claim_fresh_output_root(output_root)
    coverage_path = output_root / "coverage_rows.jsonl"
    coverage_path.write_text("{}\n", encoding="utf-8")

    # When/Then: a rerun cannot append rows or overwrite provenance in that root.
    with pytest.raises(ValueError, match="must not already exist"):
        claim_fresh_output_root(output_root)
    assert coverage_path.read_text(encoding="utf-8") == "{}\n"


def test_pilot_query_id_is_population_neutral_and_content_bound(tmp_path: Path) -> None:
    # Given: identical bytes under public good and bad directory names.
    good = tmp_path / "can" / "test_public" / "good" / "same.png"
    bad = tmp_path / "can" / "test_public" / "bad" / "same.png"
    good.parent.mkdir(parents=True)
    bad.parent.mkdir(parents=True)
    good.write_bytes(b"identical-pixels-placeholder")
    bad.write_bytes(b"identical-pixels-placeholder")

    # When: scorer identities are derived without the population path.
    good_id = pilot_query_id("can", good)
    bad_id = pilot_query_id("can", bad)

    # Then: directory labels cannot perturb RANSAC while changed content can.
    assert good_id == bad_id
    assert "/good/" not in good_id
    assert "/bad/" not in good_id
    bad.write_bytes(b"different-pixels-placeholder")
    assert pilot_query_id("can", bad) != good_id
