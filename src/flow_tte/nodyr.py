from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, cast

import torch

_SUPPORTED_DTYPES = (torch.float32, torch.float64)
_NORMALIZE_EPS = 1e-12


def _validate_matrix(name: str, value: object) -> None:
    if not isinstance(value, torch.Tensor):
        message = f"{name} must be a torch.Tensor"
        raise TypeError(message)
    if value.ndim != 2:
        message = f"{name} must be a 2D patch-by-feature matrix"
        raise ValueError(message)
    if value.shape[0] < 2 or value.shape[1] < 2:
        message = f"{name} must contain at least two patches and two features"
        raise ValueError(message)
    if value.dtype not in _SUPPORTED_DTYPES:
        message = f"{name} must have dtype float32 or float64"
        raise TypeError(message)
    if not bool(torch.isfinite(value).all().item()):
        message = f"{name} must contain only finite values"
        raise ValueError(message)


def _validate_pair(x: torch.Tensor, y: torch.Tensor) -> None:
    _validate_matrix("x", x)
    _validate_matrix("y", y)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same patch-by-feature shape")
    if x.device != y.device:
        raise ValueError("x and y must be on the same device")
    if x.dtype != y.dtype:
        raise TypeError("x and y must have the same dtype")


def token_layer_norm(value: torch.Tensor) -> torch.Tensor:
    """Apply parameter-free layer normalization independently to every patch."""
    _validate_matrix("value", value)
    centered = value - value.mean(dim=1, keepdim=True)
    variance = centered.square().mean(dim=1, keepdim=True)
    normalized = centered * torch.rsqrt(variance + _NORMALIZE_EPS)
    if not bool(torch.isfinite(normalized).all().item()):
        raise RuntimeError("token layer normalization produced non-finite values")
    return normalized


def _unit_rows(value: torch.Tensor, *, name: str) -> torch.Tensor:
    norms = cast("torch.Tensor", torch.linalg.vector_norm(value, dim=1, keepdim=True))
    if not bool(torch.isfinite(norms).all().item()):
        message = f"{name} norms must be finite"
        raise ValueError(message)
    if bool((norms <= _NORMALIZE_EPS).any().item()):
        message = f"{name} contains a row with undefined direction"
        raise ValueError(message)
    return value / norms


@dataclass(frozen=True)
class CenteredRidgeModel:
    """Dense affine map fitted by centered isotropic ridge regression."""

    weight: torch.Tensor
    bias: torch.Tensor
    relative_ridge: float
    absolute_ridge: float

    @property
    def input_dim(self) -> int:
        return int(self.weight.shape[0])

    @property
    def output_dim(self) -> int:
        return int(self.weight.shape[1])

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        _validate_matrix("x", x)
        if x.shape[1] != self.input_dim:
            raise ValueError("x feature dimension does not match the ridge model")
        if x.device != self.weight.device:
            raise ValueError("x and the ridge model must be on the same device")
        if x.dtype != self.weight.dtype:
            raise TypeError("x and the ridge model must have the same dtype")
        prediction = x @ self.weight + self.bias
        if not bool(torch.isfinite(prediction).all().item()):
            raise RuntimeError("ridge prediction produced non-finite values")
        return prediction


def _validate_relative_ridge(relative_ridge: float) -> float:
    value = float(relative_ridge)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("relative_ridge must be finite and positive")
    return value


@dataclass(frozen=True)
class _RidgeSystem:
    centered_x: torch.Tensor
    x_mean: torch.Tensor
    factor: torch.Tensor
    relative_ridge: float
    absolute_ridge: float


def _ridge_system(x: torch.Tensor, relative_ridge: float) -> _RidgeSystem:
    x_mean = x.mean(dim=0)
    centered_x = x - x_mean
    gram = centered_x.mT @ centered_x
    feature_scale = torch.trace(gram) / float(x.shape[1])
    if not bool(torch.isfinite(feature_scale).item()) or float(feature_scale.item()) <= 0.0:
        raise ValueError("x must have positive finite centered variance")
    absolute_ridge_tensor = feature_scale * relative_ridge
    regularized = gram + absolute_ridge_tensor * torch.eye(
        x.shape[1], dtype=x.dtype, device=x.device,
    )
    factor = cast("torch.Tensor", torch.linalg.cholesky(regularized))
    return _RidgeSystem(
        centered_x=centered_x,
        x_mean=x_mean,
        factor=factor,
        relative_ridge=relative_ridge,
        absolute_ridge=float(absolute_ridge_tensor.item()),
    )


def _fit_from_system(
    system: _RidgeSystem,
    target: torch.Tensor,
) -> CenteredRidgeModel:
    target_mean = target.mean(dim=0)
    centered_target = target - target_mean
    cross_covariance = system.centered_x.mT @ centered_target
    weight = torch.cholesky_solve(cross_covariance, system.factor)
    bias = target_mean - system.x_mean @ weight
    if not bool(torch.isfinite(weight).all().item()) or not bool(torch.isfinite(bias).all().item()):
        raise RuntimeError("ridge fitting produced non-finite parameters")
    return CenteredRidgeModel(
        weight=weight,
        bias=bias,
        relative_ridge=system.relative_ridge,
        absolute_ridge=system.absolute_ridge,
    )


def fit_centered_ridge(
    x: torch.Tensor,
    target: torch.Tensor,
    relative_ridge: float,
) -> CenteredRidgeModel:
    """Fit ``target = x @ weight + bias`` with centered relative ridge.

    The absolute penalty is ``relative_ridge * trace(Xc.T @ Xc) / D``. This
    makes the preregistered penalty invariant to patch count and a common
    rescaling of all input features.
    """
    _validate_matrix("x", x)
    _validate_matrix("target", target)
    if x.shape[0] != target.shape[0]:
        raise ValueError("x and target must have the same number of patches")
    if x.device != target.device:
        raise ValueError("x and target must be on the same device")
    if x.dtype != target.dtype:
        raise TypeError("x and target must have the same dtype")
    ridge = _validate_relative_ridge(relative_ridge)
    system = _ridge_system(x, ridge)
    return _fit_from_system(system, target)


@dataclass(frozen=True)
class NODyRModelBundle:
    nodyr: CenteredRidgeModel
    state: CenteredRidgeModel
    raw_delta: CenteredRidgeModel
    shuffle_nodyr: CenteredRidgeModel
    shuffle_permutation: torch.Tensor
    relative_ridge: float
    seed: int

    @property
    def feature_dim(self) -> int:
        return self.nodyr.input_dim


def _cyclic_permutation(length: int, *, seed: int, device: torch.device) -> torch.Tensor:
    offset = 1 + seed % (length - 1)
    return (torch.arange(length, device=device) + offset) % length


def _validate_permutation(
    permutation: object, *, length: int, device: torch.device,
) -> torch.Tensor:
    if not isinstance(permutation, torch.Tensor):
        raise TypeError("shuffle_permutation must be a torch.Tensor")
    if permutation.ndim != 1 or permutation.shape[0] != length:
        raise ValueError("shuffle_permutation must have one index per support patch")
    if permutation.dtype != torch.long:
        raise TypeError("shuffle_permutation must have dtype torch.long")
    local = permutation.to(device=device)
    expected = torch.arange(length, device=device)
    if not torch.equal(torch.sort(local).values, expected):
        raise ValueError("shuffle_permutation must be a bijection over support patches")
    return local


def fit_nodyr_models(
    x: torch.Tensor,
    y: torch.Tensor,
    ridge: float = 0.01,
    seed: int = 0,
    shuffle_permutation: Optional[torch.Tensor] = None,
) -> NODyRModelBundle:
    """Fit the frozen candidate and four intrinsic controls on support patches."""
    _validate_pair(x, y)
    relative_ridge = _validate_relative_ridge(ridge)
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    x_state = token_layer_norm(x)
    y_state = token_layer_norm(y)
    raw_delta = y_state - x_state
    direction = _unit_rows(raw_delta, name="support transition")
    if shuffle_permutation is None:
        permutation = _cyclic_permutation(x.shape[0], seed=seed, device=x.device)
    else:
        permutation = _validate_permutation(shuffle_permutation, length=x.shape[0], device=x.device)

    system = _ridge_system(x_state, relative_ridge)

    def fit(target: torch.Tensor) -> CenteredRidgeModel:
        return _fit_from_system(system, target)

    return NODyRModelBundle(
        nodyr=fit(direction),
        state=fit(y_state),
        raw_delta=fit(raw_delta),
        shuffle_nodyr=fit(direction[permutation]),
        shuffle_permutation=permutation,
        relative_ridge=relative_ridge,
        seed=seed,
    )


def unit_l2_cosine_audit(
    predicted_unit: torch.Tensor,
    target_unit: torch.Tensor,
) -> torch.Tensor:
    """Return the row-wise numerical error in ``L2 == 2 * (1 - cosine)``."""
    _validate_pair(predicted_unit, target_unit)
    predicted_norm = cast("torch.Tensor", torch.linalg.vector_norm(predicted_unit, dim=1))
    target_norm = cast("torch.Tensor", torch.linalg.vector_norm(target_unit, dim=1))
    tolerance = max(1e-12, 64.0 * torch.finfo(predicted_unit.dtype).eps)
    ones = torch.ones_like(predicted_norm)
    if not bool(torch.allclose(predicted_norm, ones, rtol=tolerance, atol=tolerance)):
        raise ValueError("predicted_unit rows must have unit L2 norm")
    if not bool(torch.allclose(target_norm, ones, rtol=tolerance, atol=tolerance)):
        raise ValueError("target_unit rows must have unit L2 norm")
    unit_l2 = (predicted_unit - target_unit).square().sum(dim=1)
    twice_cosine = 2.0 * (1.0 - (predicted_unit * target_unit).sum(dim=1))
    return unit_l2 - twice_cosine


def score_nodyr_models(
    bundle: NODyRModelBundle,
    x: torch.Tensor,
    y: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Score query patches with the candidate and preregistered controls."""
    _validate_pair(x, y)
    if x.shape[1] != bundle.feature_dim:
        raise ValueError("query feature dimension does not match the model bundle")

    x_state = token_layer_norm(x)
    y_state = token_layer_norm(y)
    raw_delta = y_state - x_state
    direction = _unit_rows(raw_delta, name="query transition")

    predicted_direction = _unit_rows(
        bundle.nodyr.predict(x_state), name="predicted query transition",
    )
    predicted_shuffle = _unit_rows(
        bundle.shuffle_nodyr.predict(x_state), name="shuffled predicted transition",
    )
    predicted_state = bundle.state.predict(x_state)
    predicted_state_direction = _unit_rows(
        predicted_state - x_state, name="predicted state transition",
    )
    predicted_raw_delta = bundle.raw_delta.predict(x_state)

    nodyr = 1.0 - (predicted_direction * direction).sum(dim=1)
    unit_delta_l2 = (predicted_direction - direction).square().sum(dim=1)
    audit_error = unit_l2_cosine_audit(predicted_direction, direction).abs().max()
    audit_tolerance = max(1e-12, 256.0 * torch.finfo(x.dtype).eps)
    if float(audit_error.item()) > audit_tolerance:
        raise RuntimeError("unit-L2/cosine equivalence audit failed")

    scores: Dict[str, torch.Tensor] = {
        "nodyr": nodyr,
        "state_angle": 1.0 - (predicted_state_direction * direction).sum(dim=1),
        "state_l2": (predicted_state - y_state).square().mean(dim=1),
        "raw_delta_l2": (predicted_raw_delta - raw_delta).square().mean(dim=1),
        "shuffle_nodyr": 1.0 - (predicted_shuffle * direction).sum(dim=1),
        "unit_delta_l2_equiv": unit_delta_l2,
        "raw_transition_norm": cast(
            "torch.Tensor", torch.linalg.vector_norm(raw_delta, dim=1),
        ),
    }
    if any(not bool(torch.isfinite(score).all().item()) for score in scores.values()):
        raise RuntimeError("NODyR scoring produced non-finite values")
    return scores
