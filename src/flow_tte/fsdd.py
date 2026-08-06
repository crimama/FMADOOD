from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as torch_functional

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class LinearHeadConfig:
    steps: int = 400
    learning_rate: float = 0.05
    l2_weight: float = 1e-4
    invariance_weight: float = 0.25
    factorial_weight: float = 0.25
    scale_floor: float = 1e-4


@dataclass(frozen=True)
class FactorialEffects:
    clean: FloatArray
    shifted_clean: FloatArray
    defect: FloatArray
    shifted_defect: FloatArray


@dataclass(frozen=True)
class LinearHead:
    mean: FloatArray
    scale: FloatArray
    weight: FloatArray
    bias: float
    initial_loss: float
    final_loss: float
    training_rows: int
    positive_rows: int


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in ("fsdd-stage-minus1-v1", *parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _finite_matrix(name: str, value: npt.ArrayLike) -> FloatArray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or not all(size > 0 for size in result.shape):
        message = f"{name} must be a non-empty two-dimensional matrix"
        raise ValueError(message)
    if not bool(np.all(np.isfinite(result))):
        message = f"{name} must contain only finite values"
        raise ValueError(message)
    return result


def signed_channel_pool(
    features: npt.ArrayLike,
    *,
    output_dim: int,
    seed: int,
) -> FloatArray:
    """Apply a fixed sparse signed projection without fitting evaluated data."""
    matrix = _finite_matrix("features", features)
    if output_dim <= 0 or output_dim > matrix.shape[1]:
        raise ValueError("output_dim must be positive and no larger than the channel count")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(matrix.shape[1])
    signs = rng.choice(np.asarray((-1.0, 1.0), dtype=np.float32), size=matrix.shape[1])
    signed = matrix[:, permutation] * signs[None, :]
    groups = np.array_split(np.arange(matrix.shape[1]), output_dim)
    pooled = np.stack(
        [signed[:, group].sum(axis=1) / np.sqrt(float(group.size)) for group in groups],
        axis=1,
    )
    return np.asarray(pooled, dtype=np.float32)


def project_layer_features(
    layer_features: Mapping[int, npt.ArrayLike],
    layers: Sequence[int],
    *,
    output_dim_per_layer: int,
    seed_prefix: object,
) -> FloatArray:
    if not layers or set(layer_features) != set(layers):
        raise ValueError("layer_features must contain exactly the requested layers")
    projected = [
        signed_channel_pool(
            layer_features[layer],
            output_dim=output_dim_per_layer,
            seed=stable_seed(seed_prefix, layer, "projection"),
        )
        for layer in layers
    ]
    row_counts = {value.shape[0] for value in projected}
    if len(row_counts) != 1:
        raise ValueError("all layer feature matrices must have the same patch count")
    return np.asarray(np.concatenate(projected, axis=1), dtype=np.float32)


def toroidal_mask_shuffle(mask: npt.ArrayLike, *, seed: int) -> BoolArray:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2 or min(value.shape) < 2:
        raise ValueError("mask must be a two-dimensional grid with both dimensions >= 2")
    rng = np.random.default_rng(seed)
    y_shift = int(rng.integers(1, value.shape[0]))
    x_shift = int(rng.integers(1, value.shape[1]))
    return np.asarray(np.roll(value, shift=(y_shift, x_shift), axis=(0, 1)), dtype=bool)


def balanced_map_sample(
    features: npt.ArrayLike,
    labels: npt.ArrayLike,
    *,
    samples_per_class: int,
    seed: int,
    replace_shortfall: bool = False,
) -> tuple[FloatArray, BoolArray, IntArray]:
    matrix = _finite_matrix("features", features)
    target = np.asarray(labels, dtype=bool).reshape(-1)
    if target.size != matrix.shape[0]:
        raise ValueError("labels must have one value per feature row")
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")
    rng = np.random.default_rng(seed)
    selected: list[IntArray] = []
    for label in (False, True):
        candidates = np.flatnonzero(target == label).astype(np.int64)
        if candidates.size == 0:
            continue
        replace = bool(replace_shortfall and candidates.size < samples_per_class)
        count = samples_per_class if replace else min(samples_per_class, int(candidates.size))
        selected.append(np.sort(rng.choice(candidates, size=count, replace=replace)))
    if not selected:
        raise ValueError("labels produced no sampleable rows")
    indices = np.sort(np.concatenate(selected)).astype(np.int64, copy=False)
    return matrix[indices], target[indices], indices


def _validate_effects(effects: FactorialEffects | None, dimension: int) -> FactorialEffects | None:
    if effects is None:
        return None
    arrays = tuple(
        _finite_matrix(name, value)
        for name, value in (
            ("effects.clean", effects.clean),
            ("effects.shifted_clean", effects.shifted_clean),
            ("effects.defect", effects.defect),
            ("effects.shifted_defect", effects.shifted_defect),
        )
    )
    if len({array.shape for array in arrays}) != 1 or arrays[0].shape[1] != dimension:
        raise ValueError("all factorial effect matrices must have one identical shape")
    return FactorialEffects(*arrays)


def train_linear_head(
    features: npt.ArrayLike,
    labels: npt.ArrayLike,
    *,
    config: LinearHeadConfig | None = None,
    effects: FactorialEffects | None = None,
    device: str = "cpu",
) -> LinearHead:
    matrix = _finite_matrix("features", features)
    resolved = LinearHeadConfig() if config is None else config
    target = np.asarray(labels, dtype=np.float32).reshape(-1)
    if target.size != matrix.shape[0] or not bool(np.all(np.isin(target, (0.0, 1.0)))):
        raise ValueError("labels must be binary with one value per feature row")
    positives = int(target.sum())
    negatives = int(target.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("training requires both positive and negative rows")
    if (
        resolved.steps <= 0
        or resolved.learning_rate <= 0.0
        or resolved.l2_weight < 0.0
        or resolved.invariance_weight < 0.0
        or resolved.factorial_weight < 0.0
        or resolved.scale_floor <= 0.0
    ):
        raise ValueError("invalid linear-head training configuration")
    validated_effects = _validate_effects(effects, matrix.shape[1])
    mean = matrix.astype(np.float64).mean(axis=0)
    scale = matrix.astype(np.float64).std(axis=0)
    scale = np.maximum(scale, resolved.scale_floor)
    standardized = np.asarray((matrix - mean) / scale, dtype=np.float32)

    x = torch.as_tensor(standardized, dtype=torch.float32, device=device)
    y = torch.as_tensor(target, dtype=torch.float32, device=device)
    weight = torch.zeros(matrix.shape[1], dtype=torch.float32, device=device, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam((weight, bias), lr=resolved.learning_rate)
    pos_weight = torch.as_tensor(negatives / positives, dtype=torch.float32, device=device)

    effect_tensors: tuple[torch.Tensor, ...] | None = None
    if validated_effects is not None:
        effect_tensors = tuple(
            torch.as_tensor((value - mean) / scale, dtype=torch.float32, device=device)
            for value in (
                validated_effects.clean,
                validated_effects.shifted_clean,
                validated_effects.defect,
                validated_effects.shifted_defect,
            )
        )

    losses: list[float] = []
    for _ in range(resolved.steps):
        logits = x @ weight + bias
        loss = torch_functional.binary_cross_entropy_with_logits(
            logits,
            y,
            pos_weight=pos_weight,
        )
        if effect_tensors is not None:
            clean, shifted_clean, defect, shifted_defect = effect_tensors
            clean_logits = clean @ weight + bias
            shifted_clean_logits = shifted_clean @ weight + bias
            defect_logits = defect @ weight + bias
            shifted_defect_logits = shifted_defect @ weight + bias
            invariance = torch.mean(torch.square(clean_logits - shifted_clean_logits))
            invariance = invariance + torch.mean(
                torch.square(defect_logits - shifted_defect_logits),
            )
            treatment_delta = defect_logits - clean_logits
            shifted_delta = shifted_defect_logits - shifted_clean_logits
            loss = loss + resolved.invariance_weight * invariance
            loss = loss + resolved.factorial_weight * torch.mean(
                torch.square(treatment_delta - shifted_delta),
            )
        loss = loss + resolved.l2_weight * torch.mean(torch.square(weight))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_((weight, bias), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if not losses or not np.all(np.isfinite(losses)):
        raise RuntimeError("linear-head optimization produced a non-finite loss")
    return LinearHead(
        mean=np.asarray(mean, dtype=np.float32),
        scale=np.asarray(scale, dtype=np.float32),
        weight=weight.detach().cpu().numpy().astype(np.float32, copy=False),
        bias=float(bias.detach().cpu()),
        initial_loss=losses[0],
        final_loss=losses[-1],
        training_rows=int(target.size),
        positive_rows=positives,
    )


def score_linear_head(head: LinearHead, features: npt.ArrayLike) -> FloatArray:
    matrix = _finite_matrix("features", features)
    if matrix.shape[1] != head.weight.size:
        raise ValueError("feature dimension does not match the fitted head")
    standardized = (matrix.astype(np.float64) - head.mean) / head.scale
    logits = standardized @ head.weight.astype(np.float64) + head.bias
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    return np.asarray(scores, dtype=np.float32)
