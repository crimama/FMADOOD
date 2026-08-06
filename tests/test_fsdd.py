from __future__ import annotations

import numpy as np
import pytest

from flow_tte.fsdd import (
    FactorialEffects,
    LinearHeadConfig,
    balanced_map_sample,
    project_layer_features,
    score_linear_head,
    signed_channel_pool,
    stable_seed,
    toroidal_mask_shuffle,
    train_linear_head,
)


def test_stable_seed_is_repeatable_and_part_sensitive() -> None:
    assert stable_seed("can", 0, "clean") == stable_seed("can", 0, "clean")
    assert stable_seed("can", 0, "clean") != stable_seed("can", 1, "clean")


def test_signed_pool_is_fixed_and_finite() -> None:
    features = np.arange(60, dtype=np.float32).reshape(10, 6)
    first = signed_channel_pool(features, output_dim=3, seed=7)
    second = signed_channel_pool(features, output_dim=3, seed=7)
    assert first.shape == (10, 3)
    assert np.all(np.isfinite(first))
    np.testing.assert_array_equal(first, second)


def test_project_layers_requires_exact_layer_parity() -> None:
    layers = (5, 11)
    features = {5: np.ones((4, 6), dtype=np.float32), 11: np.full((4, 6), 2.0, dtype=np.float32)}
    projected = project_layer_features(
        features,
        layers,
        output_dim_per_layer=3,
        seed_prefix="fixture",
    )
    assert projected.shape == (4, 6)
    with pytest.raises(ValueError, match="exactly"):
        project_layer_features(
            {5: features[5]},
            layers,
            output_dim_per_layer=3,
            seed_prefix="fixture",
        )


def test_toroidal_shuffle_preserves_mask_mass() -> None:
    mask = np.zeros((5, 7), dtype=bool)
    mask[0, 0] = True
    shuffled = toroidal_mask_shuffle(mask, seed=11)
    assert shuffled.shape == mask.shape
    assert int(shuffled.sum()) == 1
    assert not np.array_equal(shuffled, mask)


def test_balanced_map_sample_is_reproducible() -> None:
    features = np.arange(80, dtype=np.float32).reshape(20, 4)
    labels = np.zeros(20, dtype=bool)
    labels[-5:] = True
    first = balanced_map_sample(features, labels, samples_per_class=4, seed=3)
    second = balanced_map_sample(features, labels, samples_per_class=4, seed=3)
    np.testing.assert_array_equal(first[2], second[2])
    assert first[0].shape == (8, 4)
    assert int(first[1].sum()) == 4


def test_linear_head_is_deterministic_finite_and_hidden_free() -> None:
    rng = np.random.default_rng(9)
    negative = rng.normal(-1.0, 0.2, size=(24, 5)).astype(np.float32)
    positive = rng.normal(1.0, 0.2, size=(24, 5)).astype(np.float32)
    x = np.concatenate((negative, positive))
    y = np.concatenate((np.zeros(24, dtype=bool), np.ones(24, dtype=bool)))
    config = LinearHeadConfig(steps=40, learning_rate=0.05, factorial_weight=0.0)
    first = train_linear_head(x, y, config=config)
    second = train_linear_head(x, y, config=config)
    np.testing.assert_allclose(first.weight, second.weight, rtol=0.0, atol=0.0)
    scores = score_linear_head(first, x)
    assert first.weight.shape == (5,)
    assert first.final_loss < first.initial_loss
    assert np.all(np.isfinite(scores))
    assert float(scores[24:].mean()) > float(scores[:24].mean())


def test_factorial_effect_shape_mismatch_is_rejected() -> None:
    x = np.arange(24, dtype=np.float32).reshape(6, 4)
    y = np.asarray([0, 0, 0, 1, 1, 1], dtype=bool)
    effects = FactorialEffects(
        clean=np.zeros((2, 4), dtype=np.float32),
        shifted_clean=np.zeros((3, 4), dtype=np.float32),
        defect=np.zeros((2, 4), dtype=np.float32),
        shifted_defect=np.zeros((2, 4), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="identical shape"):
        train_linear_head(x, y, effects=effects, config=LinearHeadConfig(steps=1))
