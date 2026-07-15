import numpy as np

from flow_tte.shift_projection import fit_shift_projection, remove_shift_component


def test_shift_projection_recovers_shared_direction_and_preserves_sparse_axis() -> None:
    rng = np.random.default_rng(7)
    shared = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    residuals = rng.normal(scale=0.02, size=(200, 3)).astype(np.float32)
    residuals += rng.normal(loc=2.0, scale=0.1, size=(200, 1)).astype(np.float32) * shared
    residuals[:20, 1] += 8.0

    projection = fit_shift_projection(
        [residuals], rank=1, trim_fraction=0.2, max_samples=500, seed=0
    )
    aligned = remove_shift_component(residuals, projection.basis)

    assert abs(float(projection.basis[:, 0] @ shared)) > 0.99
    assert np.mean(np.abs(aligned[:, 0])) < 0.1
    assert np.mean(np.abs(aligned[:20, 1])) > 7.0


def test_shift_projection_is_deterministic_when_sampling() -> None:
    residuals = np.random.default_rng(3).normal(size=(1000, 8)).astype(np.float32)
    first = fit_shift_projection(
        [residuals], rank=2, trim_fraction=0.2, max_samples=100, seed=11
    )
    second = fit_shift_projection(
        [residuals], rank=2, trim_fraction=0.2, max_samples=100, seed=11
    )
    np.testing.assert_allclose(np.abs(first.basis), np.abs(second.basis), atol=1e-6)
