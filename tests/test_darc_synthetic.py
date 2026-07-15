from __future__ import annotations

from importlib.util import find_spec
from typing import Tuple, cast

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from flow_tte.darc_synthetic import (
    BROAD_CONTROL_W16_L96,
    LINE_CUE_VERSION,
    THIN_W1_L32,
    THIN_W2_L48,
    CueProfile,
    DarcSyntheticError,
    insert_line_cue,
)


def test_darc_synthetic_module_exists_when_cue_generation_is_available() -> None:
    # Given: the FlowTTE package is importable.

    # When: the DARC synthetic module is resolved.
    module = find_spec("flow_tte.darc_synthetic")

    # Then: the cue-generation surface exists.
    assert module is not None


def test_line_cue_profiles_are_frozen_at_registered_widths_and_lengths() -> None:
    # Given: the versioned, class-agnostic cue protocol.

    # When: its registered profiles are read.
    values = (
        (THIN_W1_L32.name, THIN_W1_L32.width, THIN_W1_L32.length),
        (THIN_W2_L48.name, THIN_W2_L48.width, THIN_W2_L48.length),
        (
            BROAD_CONTROL_W16_L96.name,
            BROAD_CONTROL_W16_L96.width,
            BROAD_CONTROL_W16_L96.length,
        ),
    )

    # Then: no dataset-specific profile can silently replace the preregistered controls.
    assert values == (
        ("thin-w1-l32", 1, 32),
        ("thin-w2-l48", 2, 48),
        ("broad-control-w16-l96", 16, 96),
    )


@pytest.mark.parametrize("profile", [THIN_W1_L32, THIN_W2_L48, BROAD_CONTROL_W16_L96])
def test_insert_line_cue_is_deterministic_with_an_exact_binary_mask(
    profile: CueProfile,
) -> None:
    # Given: one uniform RGB normal image and a fixed seed.
    image = np.full((160, 176, 3), 64, dtype=np.uint8)

    # When: the same registered cue is inserted twice.
    first = insert_line_cue(image, profile, seed=17)
    second = insert_line_cue(image, profile, seed=17)
    changed = np.any(first.image != image, axis=2)

    # Then: pixels, metadata, and the exact {0,1} mask all agree.
    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.mask, second.mask)
    assert first.metadata == second.metadata
    assert first.mask.dtype == np.uint8
    assert set(np.unique(first.mask).tolist()) == {0, 1}
    assert np.array_equal(changed, first.mask.astype(bool))
    assert first.metadata.version == LINE_CUE_VERSION


@pytest.mark.parametrize(
    ("level", "expected"),
    [(20, (255, 255, 255)), (235, (0, 0, 0))],
)
def test_insert_line_cue_chooses_the_opposite_black_or_white_contrast(
    level: int,
    expected: Tuple[int, int, int],
) -> None:
    # Given: a locally dark or bright RGB normal image.
    image = np.full((128, 128, 3), level, dtype=np.uint8)

    # When: a thin cue is inserted.
    cue = insert_line_cue(image, THIN_W1_L32, seed=5)

    # Then: its binary polarity maximizes contrast with the underlying pixels.
    assert cue.metadata.color_rgb == expected
    assert np.all(cue.image[cue.mask.astype(bool)] == np.asarray(expected, dtype=np.uint8))


def test_insert_line_cue_preserves_the_input_and_returns_read_only_outputs() -> None:
    # Given: a writable normal image and its independent snapshot.
    image = np.full((128, 128, 3), 80, dtype=np.uint8)
    snapshot = image.copy()

    # When: a cue is inserted.
    cue = insert_line_cue(image, THIN_W2_L48, seed=9)

    # Then: input ownership is preserved and returned evidence cannot drift.
    assert np.array_equal(image, snapshot)
    assert cue.image.flags.writeable is False
    assert cue.mask.flags.writeable is False


def test_insert_line_cue_changes_location_for_a_different_seed() -> None:
    # Given: one normal image and one profile.
    image = np.full((128, 128, 3), 90, dtype=np.uint8)

    # When: two different seeds generate cues.
    first = insert_line_cue(image, THIN_W1_L32, seed=0)
    second = insert_line_cue(image, THIN_W1_L32, seed=1)

    # Then: their exact spatial masks differ.
    assert not np.array_equal(first.mask, second.mask)


def test_thin_width_one_line_has_the_registered_centerline_length() -> None:
    # Given: the width-one, length-32 preregistered profile.
    image = np.full((128, 144, 3), 90, dtype=np.uint8)

    # When: its deterministic LINE_8 cue is rasterized.
    cue = insert_line_cue(image, THIN_W1_L32, seed=2)

    # Then: exactly 32 centerline pixels constitute the binary mask.
    assert (int(cue.mask.sum()), cue.metadata.line_type) == (32, "LINE_8")


def test_line_cue_mask_matches_opencv_line8_rasterization() -> None:
    # Given: a broad-control cue with recorded endpoints and thickness.
    image = np.full((160, 176, 3), 90, dtype=np.uint8)
    cue = insert_line_cue(image, BROAD_CONTROL_W16_L96, seed=17)
    expected = np.zeros_like(cue.mask)

    # When: OpenCV independently rasterizes the registered line parameters.
    cv2.line(
        expected,
        cue.metadata.start_xy,
        cue.metadata.end_xy,
        1,
        cue.metadata.width,
        cv2.LINE_8,
    )

    # Then: the emitted binary mask is the exact LINE_8 raster.
    assert np.array_equal(cue.mask, expected)


@pytest.mark.parametrize(
    ("image", "seed", "field"),
    [
        (np.zeros((64, 64), dtype=np.uint8), 0, "image"),
        (np.zeros((64, 64, 3), dtype=np.float32), 0, "image"),
        (np.zeros((64, 64, 3), dtype=np.uint8), -1, "seed"),
        (np.zeros((80, 80, 3), dtype=np.uint8), 0, "image"),
    ],
)
def test_insert_line_cue_rejects_invalid_boundary_values(
    image: npt.NDArray[np.generic],
    seed: int,
    field: str,
) -> None:
    # Given: an invalid RGB array, seed, or image/profile size pairing.
    typed_image = cast("npt.NDArray[np.uint8]", image)

    # When: the broad cue is requested, Then: a typed boundary error names the field.
    with pytest.raises(DarcSyntheticError) as captured:
        insert_line_cue(typed_image, BROAD_CONTROL_W16_L96, seed)
    assert captured.value.field == field
