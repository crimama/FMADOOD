from __future__ import annotations

import json
from importlib.util import find_spec
from pathlib import Path
from typing import Tuple

import pytest

from flow_tte.darc_resources import DarcResourceError, build_p16_split


def test_darc_resources_module_exists_when_protocol_is_available() -> None:
    # Given: the FlowTTE package is importable.

    # When: the DARC resource module is resolved.
    module = find_spec("flow_tte.darc_resources")

    # Then: the protocol resource surface exists.
    assert module is not None


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_build_p16_split_is_order_independent_for_registered_seed(seed: int) -> None:
    # Given: the same source pool in opposite enumeration orders.
    paths = tuple(Path(f"/normal/{index:03d}.png") for index in range(25))

    # When: the P16 split is built before any features are loaded.
    forward = build_p16_split(paths, seed)
    reversed_order = build_p16_split(tuple(reversed(paths)), seed)

    # Then: sorting at the boundary makes the complete split deterministic.
    assert forward == reversed_order


def test_build_p16_split_matches_the_pcg64_seed_zero_selection() -> None:
    # Given: a stable lexical pool of normal-image paths.
    paths = tuple(Path(f"/normal/{index:03d}.png") for index in range(25))
    expected_names = (
        "019.png", "004.png", "010.png", "011.png", "024.png", "002.png",
        "023.png", "006.png", "016.png", "022.png", "003.png", "021.png",
        "008.png", "000.png", "020.png", "012.png",
    )

    # When: seed zero uniformly selects the exposed P16 resources.
    split = build_p16_split(paths, 0)
    selected_names = tuple(Path(path).name for path in split.support_paths)

    # Then: the versioned random ordering remains reproducible across runtimes.
    assert selected_names == expected_names


def test_build_p16_split_exposes_exactly_sixteen_from_the_source_pool() -> None:
    # Given: more normal images than the protocol consumes.
    paths = tuple(Path(f"/normal/{index:03d}.png") for index in range(25))

    # When: the P16 protocol resources are selected.
    split = build_p16_split(paths, 1)

    # Then: only P16 paths are exposed while the full pool size remains auditable.
    assert (len(split.support_paths), split.source_pool_count) == (16, 25)


def test_build_p16_split_makes_four_exact_disjoint_folds() -> None:
    # Given: one selected P16 support set.
    paths = tuple(Path(f"/normal/{index:03d}.png") for index in range(25))
    split = build_p16_split(paths, 2)

    # When: calibration membership is counted across all folds.
    calibration_paths: Tuple[str, ...] = tuple(
        path for fold in split.folds for path in fold.calibration_paths
    )

    # Then: every fold is 12/4 and each support calibrates exactly once.
    assert len(split.folds) == 4
    assert all(len(fold.memory_paths) == 12 for fold in split.folds)
    assert all(len(fold.calibration_paths) == 4 for fold in split.folds)
    assert all(set(fold.memory_paths).isdisjoint(fold.calibration_paths) for fold in split.folds)
    assert sorted(calibration_paths) == sorted(split.support_paths)


def test_p16_split_manifest_is_json_serializable_with_named_fold_fields() -> None:
    # Given: an immutable P16 split value.
    paths = tuple(Path(f"/normal/{index:03d}.png") for index in range(16))
    split = build_p16_split(paths, 0)

    # When: its manifest representation crosses the JSON boundary.
    restored = json.loads(json.dumps(split.to_manifest()))

    # Then: resource provenance and fold membership retain named fields.
    assert restored["seed"] == 0
    assert restored["source_pool_count"] == 16
    assert len(restored["support_paths"]) == 16
    assert len(restored["folds"][0]["memory_paths"]) == 12


@pytest.mark.parametrize(
    ("paths", "seed", "field"),
    [
        (tuple(Path(f"/normal/{index:03d}.png") for index in range(15)), 0, "paths"),
        (tuple(Path(f"/normal/{index:03d}.png") for index in range(16)), 3, "seed"),
        ((Path("/normal/000.png"),) * 16, 0, "paths"),
    ],
)
def test_build_p16_split_rejects_invalid_protocol_resources(
    paths: Tuple[Path, ...],
    seed: int,
    field: str,
) -> None:
    # Given: a resource pool or seed outside the registered protocol.

    # When: the P16 split is requested, Then: a typed boundary error names the field.
    with pytest.raises(DarcResourceError) as captured:
        build_p16_split(paths, seed)
    assert captured.value.field == field
