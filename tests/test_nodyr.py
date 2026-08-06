from __future__ import annotations

from typing import Callable, Tuple

import pytest
import torch

from flow_tte.nodyr import (
    fit_centered_ridge,
    fit_nodyr_models,
    score_nodyr_models,
    token_layer_norm,
    unit_l2_cosine_audit,
)


def _features(rows: int = 80, dimensions: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(19)
    x = torch.randn(rows, dimensions, generator=generator)
    mixing = torch.randn(dimensions, dimensions, generator=generator) / dimensions**0.5
    y = 0.65 * x + 0.35 * (x @ mixing) + 0.1
    return x, y


def test_centered_dense_ridge_recovers_affine_map() -> None:
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(100, 5, generator=generator, dtype=torch.float64) + 4.0
    weight = torch.randn(5, 4, generator=generator, dtype=torch.float64)
    bias = torch.tensor([2.0, -3.0, 0.5, 8.0], dtype=torch.float64)
    target = x @ weight + bias

    model = fit_centered_ridge(x, target, relative_ridge=1e-10)

    torch.testing.assert_close(model.predict(x), target, rtol=1e-8, atol=1e-8)
    expected_penalty = 1e-10 * torch.trace((x - x.mean(0)).mT @ (x - x.mean(0))) / x.shape[1]
    assert model.absolute_ridge == pytest.approx(float(expected_penalty), rel=1e-12)


def test_fit_and_score_exposes_candidate_controls_and_transition_norm() -> None:
    x, y = _features()
    bundle = fit_nodyr_models(x, y, ridge=0.01, seed=41)

    scores = score_nodyr_models(bundle, x[:17], y[:17])

    assert set(scores) == {
        "nodyr",
        "state_angle",
        "state_l2",
        "raw_delta_l2",
        "shuffle_nodyr",
        "unit_delta_l2_equiv",
        "raw_transition_norm",
    }
    assert all(score.shape == (17,) for score in scores.values())
    assert all(bool(torch.isfinite(score).all()) for score in scores.values())
    torch.testing.assert_close(
        scores["unit_delta_l2_equiv"], 2.0 * scores["nodyr"], rtol=1e-5, atol=1e-6,
    )


def test_state_angle_scores_predicted_transition_not_absolute_state() -> None:
    x, y = _features()
    bundle = fit_nodyr_models(x, y, ridge=0.01, seed=4)
    query_x = x[:13]
    query_y = y[:13]

    scores = score_nodyr_models(bundle, query_x, query_y)
    x_state = token_layer_norm(query_x)
    y_state = token_layer_norm(query_y)
    target_direction = torch.nn.functional.normalize(y_state - x_state, dim=1)
    predicted_direction = torch.nn.functional.normalize(
        bundle.state.predict(x_state) - x_state,
        dim=1,
    )
    expected = 1.0 - (predicted_direction * target_direction).sum(dim=1)

    torch.testing.assert_close(scores["state_angle"], expected)


def test_shuffle_target_permutation_is_cyclic_and_deterministic() -> None:
    x, y = _features(rows=101)

    first = fit_nodyr_models(x, y, ridge=0.01, seed=7)
    repeated = fit_nodyr_models(x, y, ridge=0.01, seed=7)
    different = fit_nodyr_models(x, y, ridge=0.01, seed=8)

    torch.testing.assert_close(first.shuffle_permutation, repeated.shuffle_permutation)
    assert not torch.equal(first.shuffle_permutation, different.shuffle_permutation)
    torch.testing.assert_close(
        torch.sort(first.shuffle_permutation).values, torch.arange(x.shape[0]),
    )
    torch.testing.assert_close(
        first.shuffle_nodyr.weight, repeated.shuffle_nodyr.weight, rtol=0.0, atol=0.0,
    )


def test_runner_can_supply_a_structured_shuffle_permutation() -> None:
    x, y = _features(rows=80)
    structured = torch.arange(80).reshape(8, 10).roll(shifts=(1, 3), dims=(0, 1)).reshape(-1)

    bundle = fit_nodyr_models(x, y, ridge=0.01, seed=0, shuffle_permutation=structured)

    torch.testing.assert_close(bundle.shuffle_permutation, structured)


def test_shuffle_permutation_must_be_a_bijection() -> None:
    x, y = _features()
    invalid = torch.zeros(x.shape[0], dtype=torch.long)

    with pytest.raises(ValueError, match="bijection"):
        fit_nodyr_models(x, y, ridge=0.01, seed=0, shuffle_permutation=invalid)


def test_unit_l2_cosine_audit_is_numerically_zero() -> None:
    generator = torch.Generator().manual_seed(2)
    predicted = torch.nn.functional.normalize(torch.randn(32, 11, generator=generator), dim=1)
    target = torch.nn.functional.normalize(torch.randn(32, 11, generator=generator), dim=1)

    error = unit_l2_cosine_audit(predicted, target)

    torch.testing.assert_close(error, torch.zeros_like(error), rtol=0.0, atol=1e-6)


def test_token_layer_norm_is_parameter_free_per_patch() -> None:
    x, _ = _features(rows=9)

    normalized = token_layer_norm(x)

    torch.testing.assert_close(normalized.mean(dim=1), torch.zeros(9), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(normalized.square().mean(dim=1), torch.ones(9), atol=1e-6, rtol=0.0)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda x, y: (x[:, :-1], y), "same patch-by-feature shape"),
        (lambda x, y: (x.reshape(4, 20, 8), y), "2D patch-by-feature"),
        (lambda x, y: (x.to(torch.float16), y.to(torch.float16)), "float32 or float64"),
        (
            lambda x, y: (
                x.index_put((torch.tensor([0]), torch.tensor([0])), torch.tensor(float("nan"))),
                y,
            ),
            "finite",
        ),
    ],
)
def test_fit_rejects_invalid_inputs(
    mutator: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    message: str,
) -> None:
    x, y = _features()
    invalid_x, invalid_y = mutator(x.clone(), y.clone())

    with pytest.raises((TypeError, ValueError), match=message):
        fit_nodyr_models(invalid_x, invalid_y, ridge=0.01, seed=0)


def test_fit_rejects_invalid_hyperparameters_and_undefined_transition() -> None:
    x, y = _features()

    with pytest.raises(ValueError, match="relative_ridge"):
        fit_nodyr_models(x, y, ridge=0.0, seed=0)
    with pytest.raises(ValueError, match="seed"):
        fit_nodyr_models(x, y, ridge=0.01, seed=-1)
    with pytest.raises(ValueError, match="undefined direction"):
        fit_nodyr_models(x, x, ridge=0.01, seed=0)


def test_score_rejects_nonfinite_and_feature_mismatch() -> None:
    x, y = _features()
    bundle = fit_nodyr_models(x, y, ridge=0.01, seed=0)

    with pytest.raises(ValueError, match="same patch-by-feature shape"):
        score_nodyr_models(bundle, x[:4], y[:4, :-1])
    invalid_y = y[:4].clone()
    invalid_y[0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        score_nodyr_models(bundle, x[:4], invalid_y)
