from __future__ import annotations

import numpy as np
import pytest
import torch

from src.flow_tte_gridshift_2view import (
    align_shifted_native_map,
    combine_aligned_maps,
    max_aligned_maps,
    mean_aligned_maps,
    native_offset_yx,
    shift_resized_tensor_right_down,
)


def test_shift_and_native_back_alignment_restore_interior() -> None:
    original = np.arange(9 * 11, dtype=np.float32).reshape(9, 11)
    tensor = torch.from_numpy(original).unsqueeze(0)

    shifted = shift_resized_tensor_right_down(tensor, (2, 3)).squeeze(0).numpy()
    aligned = align_shifted_native_map(shifted, original.shape, (2, 3))

    assert shifted.shape == original.shape
    assert np.array_equal(shifted[:2, :3], np.full((2, 3), original[0, 0]))
    assert np.array_equal(shifted[2:, 3:], original[:-2, :-3])
    assert aligned[:-2, :-3] == pytest.approx(original[:-2, :-3])
    # Replicated borders are the only pixels that cannot be recovered after crop.
    assert np.all(aligned[-2:, :] == aligned[-3:-2, :])
    assert np.all(aligned[:, -3:] == aligned[:, -4:-3])


def test_shift_replicates_top_and_left_edges_and_preserves_tensor_properties() -> None:
    image = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4)

    shifted = shift_resized_tensor_right_down(image, (1, 2))

    assert shifted.shape == image.shape
    assert shifted.dtype == image.dtype
    assert torch.equal(shifted[..., 0, :], shifted[..., 1, :])
    assert torch.equal(shifted[..., :, :2], shifted[..., :, :1].expand(-1, -1, 2))
    assert torch.equal(shifted[..., 1:, 2:], image[..., :-1, :-2])


def test_native_offset_uses_independent_realized_resize_factors() -> None:
    assert native_offset_yx((100, 300), (50, 100), (8, 8)) == pytest.approx((16.0, 24.0))


def test_mean_and_max_combiners_are_elementwise_float32() -> None:
    first = np.array([[0.0, 4.0], [2.0, 8.0]], dtype=np.float32)
    second = np.array([[2.0, 2.0], [6.0, 0.0]], dtype=np.float64)

    expected_mean = np.array([[1.0, 3.0], [4.0, 4.0]], dtype=np.float32)
    expected_max = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    assert np.array_equal(mean_aligned_maps([first, second]), expected_mean)
    assert np.array_equal(max_aligned_maps([first, second]), expected_max)
    assert np.array_equal(combine_aligned_maps([first, second], "mean"), expected_mean)
    assert np.array_equal(combine_aligned_maps([first, second], "max"), expected_max)
    assert mean_aligned_maps([first, second]).dtype == np.float32


def test_combiners_reject_mismatched_shapes_and_unknown_method() -> None:
    with pytest.raises(ValueError, match="same shape"):
        mean_aligned_maps([np.zeros((2, 2)), np.zeros((2, 3))])
    with pytest.raises(ValueError, match="unknown"):
        combine_aligned_maps([np.zeros((2, 2))], "median")
