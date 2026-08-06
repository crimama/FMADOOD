from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from typing import Literal, Mapping, Protocol, Sequence

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as functional

from flow_tte.losses import patch_nll
from flow_tte.scoring import ScoreCalibration
from flow_tte.trainer import FlowDensityEstimator

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class GBCMFConfig:
    steps: int = 300
    learning_rate: float = 2e-5
    softmin_tau: float = 0.05
    ranking_margin: float = 0.5
    background_top_fraction: float = 0.05
    background_count: int = 384
    interior_cap: int = 128
    memory_rows: int = 4096
    proximal_weight: float = 1e-3
    constraint_rho: float = 2.0
    dual_update_interval: int = 10
    gradient_clip: float = 0.5


@dataclass(frozen=True)
class GBCMFCell:
    source_id: int
    clean: FloatArray
    shifted_clean: FloatArray
    defect: FloatArray
    shifted_defect: FloatArray
    defect_mask: BoolArray
    background_mask: BoolArray
    axis: str
    family: str


@dataclass(frozen=True)
class GBCMFTrainingResult:
    mode: str
    losses: tuple[float, ...]
    final_tail_constraints: Mapping[str, float]
    final_drop_constraints: Mapping[str, float]
    final_tail_duals: Mapping[str, float]
    final_drop_duals: Mapping[str, float]
    max_parameter_delta: float
    finite: bool


@dataclass(frozen=True)
class ScoreNull:
    kind: str
    state_dict: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class GeometryAudit:
    latent_rms_ratio: float
    pair_distance_spearman: float
    effective_rank_ratio: float
    base_density_quantile_drift_std: tuple[float, float, float]
    negative_logdet_quantile_drift_std: tuple[float, float, float]
    negative_logdet_share_ratio: float
    coupling_logdet_q99_ratio: float
    inverse_max_abs_error: float
    passed: bool


@dataclass(frozen=True)
class ProxyFidelity:
    spearman: float
    top_fraction_overlap: float
    q995_difference: float
    passed: bool


class ComponentCell(Protocol):
    clean: npt.ArrayLike
    shifted_clean: npt.ArrayLike
    defect: npt.ArrayLike
    shifted_defect: npt.ArrayLike
    defect_mask: npt.ArrayLike
    background_mask: npt.ArrayLike


class _AffineNull(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        initial = torch.tensor([1.0, 0.25], dtype=torch.float32)
        self.raw_weight = nn.Parameter(torch.log(initial / initial.sum()))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, components: torch.Tensor) -> torch.Tensor:
        weight = 1.25 * torch.softmax(self.raw_weight, dim=0)
        return components @ weight + self.bias


class _NonlinearNull(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(2, 8)
        self.output = nn.Linear(8, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, components: torch.Tensor) -> torch.Tensor:
        residual = 0.1 * self.output(functional.gelu(self.hidden(components))).squeeze(1)
        return components @ components.new_tensor([1.0, 0.25]) + residual


def clone_estimator(estimator: FlowDensityEstimator) -> FlowDensityEstimator:
    """Clone a fitted estimator so every Stage -1A arm shares the same theta_0."""
    return copy.deepcopy(estimator)


def refine_flow(  # noqa: C901, PLR0913, PLR0915
    estimator: FlowDensityEstimator,
    clean_by_source: Mapping[int, npt.ArrayLike],
    cells: Sequence[GBCMFCell],
    calibration: ScoreCalibration,
    *,
    mode: Literal["nonfactorial", "gb_cmf"],
    config: GBCMFConfig | None = None,
    seed: int = 0,
) -> GBCMFTrainingResult:
    resolved = GBCMFConfig() if config is None else config
    _validate_config(resolved)
    validated_clean = _validate_clean_sources(clean_by_source)
    validated_cells = tuple(_validate_cell(cell) for cell in cells)
    if not validated_cells:
        raise ValueError("GB-CMF requires factorial cells")
    if {cell.source_id for cell in validated_cells} != set(validated_clean):
        raise ValueError("every clean source must have factorial cells")
    if len({cell.axis for cell in validated_cells}) < 2:
        raise ValueError("GB-CMF requires at least two train axes")
    if len({cell.family for cell in validated_cells}) < 2:
        raise ValueError("GB-CMF requires at least two train defect families")
    if estimator.standardizer is None:
        raise RuntimeError("GB-CMF requires a fitted standardized flow")

    device = estimator.device
    torch.use_deterministic_algorithms(mode=True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    parameters = tuple(estimator.flow.parameters())
    initial = tuple(parameter.detach().clone() for parameter in parameters)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=resolved.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    lambda_tail = {axis: torch.zeros((), device=device) for axis in {c.axis for c in cells}}
    lambda_drop = {
        (cell.axis, cell.family): torch.zeros((), device=device)
        for cell in validated_cells
    }
    initial_clean_mean = abs(float(estimator.train_nll_mean)) + 1e-6
    losses: list[float] = []
    last_tail: dict[str, float] = {}
    last_drop: dict[str, float] = {}
    latest_tail: dict[str, torch.Tensor] = {}
    latest_drop: dict[tuple[str, str], torch.Tensor] = {}
    _ = estimator.flow.train()

    for step in range(resolved.steps):
        cell = validated_cells[step % len(validated_cells)]
        sampled = _sample_cell(cell, resolved, seed=seed, step=step)
        bank = _sample_other_source_memory(
            validated_clean,
            excluded_source=cell.source_id,
            rows=resolved.memory_rows,
            seed=seed,
            step=step,
        )
        with torch.no_grad():
            bank_z, _ = _forward_features(estimator, bank)
        component_by_name: dict[str, torch.Tensor] = {}
        nll_by_name: dict[str, torch.Tensor] = {}
        for name in ("clean", "shifted_clean", "defect", "shifted_defect"):
            feature_rows = sampled[name]
            z, logdet = _forward_features(estimator, feature_rows)
            nll = patch_nll(z, logdet)
            component_by_name[name] = _differentiable_components(
                z,
                nll,
                bank_z,
                estimator,
                calibration,
                resolved.softmin_tau,
            )
            nll_by_name[name] = nll

        default_weight = component_by_name["clean"].new_tensor([1.0, 0.25])
        scores = {name: value @ default_weight for name, value in component_by_name.items()}
        interior_count = sampled["interior_count"]
        clean_nll = nll_by_name["clean"].mean() / initial_clean_mean
        defect_rank = _sampled_ranking_loss(
            scores["defect"],
            interior_count,
            resolved.ranking_margin,
            resolved.background_top_fraction,
        )
        shifted_rank = _sampled_ranking_loss(
            scores["shifted_defect"],
            interior_count,
            resolved.ranking_margin,
            resolved.background_top_fraction,
        )
        proximal = torch.stack(
            [
                (parameter - anchor).square().mean()
                for parameter, anchor in zip(parameters, initial)
            ],
        ).mean()
        tail_constraint = torch.quantile(scores["shifted_clean"], 0.995) - torch.quantile(
            scores["clean"], 0.995,
        )
        drop_constraint = shifted_rank - defect_rank
        loss = clean_nll + defect_rank + shifted_rank + resolved.proximal_weight * proximal
        if mode == "gb_cmf":
            positive_tail = torch.relu(tail_constraint)
            positive_drop = torch.relu(drop_constraint)
            loss = loss + lambda_tail[cell.axis] * tail_constraint
            loss = loss + 0.5 * resolved.constraint_rho * positive_tail.square()
            loss = loss + lambda_drop[(cell.axis, cell.family)] * drop_constraint
            loss = loss + 0.5 * resolved.constraint_rho * positive_drop.square()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=resolved.gradient_clip)
        optimizer.step()
        latest_tail[cell.axis] = tail_constraint.detach()
        latest_drop[(cell.axis, cell.family)] = drop_constraint.detach()
        if mode == "gb_cmf" and (step + 1) % resolved.dual_update_interval == 0:
            for axis, value in latest_tail.items():
                lambda_tail[axis] = torch.clamp(
                    lambda_tail[axis] + resolved.constraint_rho * value,
                    min=0.0,
                )
            for key, value in latest_drop.items():
                lambda_drop[key] = torch.clamp(
                    lambda_drop[key] + resolved.constraint_rho * value,
                    min=0.0,
                )
        losses.append(float(loss.detach().cpu()))
        last_tail[cell.axis] = float(tail_constraint.detach().cpu())
        last_drop[f"{cell.axis}:{cell.family}"] = float(drop_constraint.detach().cpu())

    max_delta = max(
        float((parameter.detach() - anchor).abs().max().cpu())
        for parameter, anchor in zip(parameters, initial)
    )
    finite = bool(np.all(np.isfinite(losses))) and all(
        bool(torch.isfinite(parameter).all()) for parameter in parameters
    )
    return GBCMFTrainingResult(
        mode=mode,
        losses=tuple(losses),
        final_tail_constraints=last_tail,
        final_drop_constraints=last_drop,
        final_tail_duals={key: float(value.cpu()) for key, value in lambda_tail.items()},
        final_drop_duals={
            f"{axis}:{family}": float(value.cpu())
            for (axis, family), value in lambda_drop.items()
        },
        max_parameter_delta=max_delta,
        finite=finite,
    )


def train_score_null(  # noqa: PLR0913
    cells: Sequence[ComponentCell],
    *,
    kind: Literal["affine", "nonlinear"],
    steps: int = 300,
    learning_rate: float = 2e-5,
    seed: int = 0,
    device: str = "cpu",
    config: GBCMFConfig | None = None,
) -> ScoreNull:
    """Train a fixed-flow score-capacity null on WAFF-compatible component cells."""
    if steps <= 0 or learning_rate <= 0.0:
        raise ValueError("score-null optimizer values must be positive")
    torch.manual_seed(seed)
    resolved = GBCMFConfig() if config is None else config
    _validate_config(resolved)
    module: nn.Module = _AffineNull() if kind == "affine" else _NonlinearNull()
    module = module.to(device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate, weight_decay=0.0)
    if not cells:
        raise ValueError("score null requires component cells")
    for step in range(steps):
        cell_value = cells[step % len(cells)]
        cell = _sample_score_cell_tensors(cell_value, device, resolved, seed=seed, step=step)
        clean, shifted, defect, shifted_defect, mask, background = cell
        clean_score = module(clean)
        shifted_score = module(shifted)
        defect_score = module(defect)
        shifted_defect_score = module(shifted_defect)
        rank = _masked_ranking_loss(defect_score, mask, background, 0.5)
        shifted_rank = _masked_ranking_loss(shifted_defect_score, mask, background, 0.5)
        tail = torch.quantile(shifted_score, 0.995) - torch.quantile(clean_score, 0.995)
        drop = shifted_rank - rank
        loss = rank + shifted_rank + 2.0 * torch.relu(tail).square()
        loss = loss + 2.0 * torch.relu(drop).square()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=0.5)
        optimizer.step()
    state = {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
    return ScoreNull(kind=kind, state_dict=state)


def score_null(null: ScoreNull, components: npt.ArrayLike) -> FloatArray:
    matrix = np.asarray(components, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != 2:
        raise ValueError("components must have shape [rows, 2]")
    module: nn.Module = _AffineNull() if null.kind == "affine" else _NonlinearNull()
    module.load_state_dict(null.state_dict)
    module.eval()
    with torch.no_grad():
        values = module(torch.as_tensor(matrix)).cpu().numpy()
    return np.asarray(values, dtype=np.float32)


def audit_flow_geometry(
    baseline: FlowDensityEstimator,
    candidate: FlowDensityEstimator,
    held_clean: npt.ArrayLike,
    *,
    sample_rows: int = 256,
    seed: int = 0,
) -> GeometryAudit:
    features = np.asarray(held_clean, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("geometry audit requires a nontrivial feature matrix")
    indices = _deterministic_choice(features.shape[0], min(sample_rows, features.shape[0]), seed)
    sampled = features[indices]
    with torch.no_grad():
        base_z, base_logdet = _forward_features(baseline, sampled)
        cand_z, cand_logdet = _forward_features(candidate, sampled)
        reconstructed, _ = candidate.flow.forward(cand_z, reverse=True)
        standardized = candidate.standardizer.transform(  # type: ignore[union-attr]
            torch.as_tensor(sampled, device=candidate.device),
        )
    base_density = 0.5 * base_z.square().sum(dim=1)
    cand_density = 0.5 * cand_z.square().sum(dim=1)
    base_neg_logdet = -base_logdet
    cand_neg_logdet = -cand_logdet
    latent_rms_ratio = _safe_ratio(_rms(cand_z), _rms(base_z))
    pair_spearman = _spearman(
        torch.pdist(base_z).cpu().numpy(),
        torch.pdist(cand_z).cpu().numpy(),
    )
    effective_rank_ratio = _safe_ratio(_effective_rank(cand_z), _effective_rank(base_z))
    density_drift = _quantile_drift_std(base_density, cand_density)
    logdet_drift = _quantile_drift_std(base_neg_logdet, cand_neg_logdet)
    share_ratio = _safe_ratio(
        _logdet_share(cand_density, cand_neg_logdet),
        _logdet_share(base_density, base_neg_logdet),
    )
    coupling_ratio = _safe_ratio(
        _coupling_logdet_q99(candidate, sampled),
        _coupling_logdet_q99(baseline, sampled),
    )
    inverse_error = float((reconstructed - standardized).abs().max().cpu())
    passed = bool(
        0.8 <= latent_rms_ratio <= 1.25
        and pair_spearman >= 0.95
        and 0.8 <= effective_rank_ratio <= 1.25
        and all(abs(value) <= 1.0 for value in (*density_drift, *logdet_drift))
        and share_ratio <= 1.25
        and coupling_ratio <= 1.25
        and inverse_error <= 1e-5,
    )
    return GeometryAudit(
        latent_rms_ratio=latent_rms_ratio,
        pair_distance_spearman=pair_spearman,
        effective_rank_ratio=effective_rank_ratio,
        base_density_quantile_drift_std=density_drift,
        negative_logdet_quantile_drift_std=logdet_drift,
        negative_logdet_share_ratio=share_ratio,
        coupling_logdet_q99_ratio=coupling_ratio,
        inverse_max_abs_error=inverse_error,
        passed=passed,
    )


def audit_proxy_fidelity(  # noqa: PLR0913
    estimator: FlowDensityEstimator,
    calibration: ScoreCalibration,
    query_features: npt.ArrayLike,
    bank_features: npt.ArrayLike,
    *,
    tau: float = 0.05,
    top_fraction: float = 0.005,
) -> ProxyFidelity:
    if tau <= 0.0 or not 0.0 < top_fraction <= 1.0:
        raise ValueError("invalid proxy audit configuration")
    with torch.no_grad():
        query_z, query_logdet = _forward_features(estimator, query_features)
        bank_z, _ = _forward_features(estimator, bank_features)
        nll = patch_nll(query_z, query_logdet)
        distances = torch.cdist(query_z, bank_z)
        soft_distance = -tau * torch.logsumexp(-distances / tau, dim=1)
        hard_distance = distances.min(dim=1).values
        density = torch.relu((nll - estimator.train_nll_mean) / estimator.train_nll_std)
        soft_score = calibration.normalize_distance(soft_distance) + 0.25 * density
        hard_score = calibration.normalize_distance(hard_distance) + 0.25 * density
    soft = soft_score.cpu().numpy()
    hard = hard_score.cpu().numpy()
    spearman = _spearman(soft, hard)
    k = max(1, math.ceil(soft.size * top_fraction))
    soft_top = set(np.argpartition(soft, -k)[-k:].tolist())
    hard_top = set(np.argpartition(hard, -k)[-k:].tolist())
    overlap = len(soft_top & hard_top) / k
    q995_difference = abs(float(np.quantile(soft, 0.995) - np.quantile(hard, 0.995)))
    return ProxyFidelity(
        spearman=spearman,
        top_fraction_overlap=overlap,
        q995_difference=q995_difference,
        passed=bool(spearman >= 0.98 and overlap >= 0.90 and q995_difference <= 0.01),
    )


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _forward_features(
    estimator: FlowDensityEstimator,
    features: npt.ArrayLike,
) -> tuple[torch.Tensor, torch.Tensor]:
    if estimator.standardizer is None:
        raise RuntimeError("estimator must be fitted")
    tensor = torch.as_tensor(features, dtype=torch.float32, device=estimator.device)
    standardized = estimator.standardizer.transform(tensor)
    return estimator.flow.forward(standardized)


def _differentiable_components(  # noqa: PLR0913
    z: torch.Tensor,
    nll: torch.Tensor,
    bank_z: torch.Tensor,
    estimator: FlowDensityEstimator,
    calibration: ScoreCalibration,
    tau: float,
) -> torch.Tensor:
    distances = torch.cdist(z, bank_z)
    soft_distance = -tau * torch.logsumexp(-distances / tau, dim=1)
    normalized = calibration.normalize_distance(soft_distance)
    density = torch.relu((nll - estimator.train_nll_mean) / estimator.train_nll_std)
    return torch.stack((normalized, density), dim=1)


def _sample_cell(
    cell: GBCMFCell,
    config: GBCMFConfig,
    *,
    seed: int,
    step: int,
) -> dict[str, object]:
    interior = np.flatnonzero(cell.defect_mask)
    background = np.flatnonzero(cell.background_mask)
    interior_indices = interior[
        _deterministic_choice(
            interior.size,
            min(config.interior_cap, interior.size),
            stable_seed(seed, step, cell.source_id, cell.axis, cell.family, "interior"),
        )
    ]
    background_indices = background[
        _deterministic_choice(
            background.size,
            min(config.background_count, background.size),
            stable_seed(seed, step, cell.source_id, cell.axis, cell.family, "background"),
        )
    ]
    indices = np.concatenate((interior_indices, background_indices))
    return {
        "clean": cell.clean[indices],
        "shifted_clean": cell.shifted_clean[indices],
        "defect": cell.defect[indices],
        "shifted_defect": cell.shifted_defect[indices],
        "interior_count": int(interior_indices.size),
    }


def _sample_other_source_memory(
    clean_by_source: Mapping[int, FloatArray],
    *,
    excluded_source: int,
    rows: int,
    seed: int,
    step: int,
) -> FloatArray:
    sources = [source for source in sorted(clean_by_source) if source != excluded_source]
    if not sources:
        raise ValueError("source exclusion requires another clean source")
    per_source = max(1, math.ceil(rows / len(sources)))
    sampled: list[FloatArray] = []
    for source in sources:
        values = clean_by_source[source]
        indices = _deterministic_choice(
            values.shape[0],
            min(per_source, values.shape[0]),
            stable_seed(seed, step, excluded_source, source, "memory"),
        )
        sampled.append(values[indices])
    return np.concatenate(sampled, axis=0)[:rows].astype(np.float32, copy=False)


def _sampled_ranking_loss(
    scores: torch.Tensor,
    interior_count: object,
    margin: float,
    background_top_fraction: float,
) -> torch.Tensor:
    count = int(interior_count)
    if count <= 0 or count >= scores.numel():
        raise ValueError("sampled ranking requires interior and background rows")
    background = scores[count:]
    k = max(1, math.ceil(background.numel() * background_top_fraction))
    return functional.softplus(
        margin + torch.topk(background, k=k).values.mean() - scores[:count].mean(),
    )


def _masked_ranking_loss(
    scores: torch.Tensor,
    mask: torch.Tensor,
    background: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    background_scores = scores[background]
    k = max(1, math.ceil(background_scores.numel() * 0.05))
    return functional.softplus(
        margin + torch.topk(background_scores, k=k).values.mean() - scores[mask].mean(),
    )


def _score_cell_tensors(cell: ComponentCell, device: str) -> tuple[torch.Tensor, ...]:
    values = []
    for name in ("clean", "shifted_clean", "defect", "shifted_defect"):
        matrix = np.asarray(getattr(cell, name), dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != 2:
            raise ValueError("score-null cells require [rows,2] components")
        values.append(torch.as_tensor(matrix, device=device))
    mask = torch.as_tensor(np.asarray(cell.defect_mask, dtype=bool), device=device)
    background = torch.as_tensor(
        np.asarray(cell.background_mask, dtype=bool),
        device=device,
    )
    return (*values, mask, background)


def _sample_score_cell_tensors(
    cell: ComponentCell,
    device: str,
    config: GBCMFConfig,
    *,
    seed: int,
    step: int,
) -> tuple[torch.Tensor, ...]:
    tensors = _score_cell_tensors(cell, device)
    mask = np.asarray(cell.defect_mask, dtype=bool).reshape(-1)
    background = np.asarray(cell.background_mask, dtype=bool).reshape(-1)
    source_id = getattr(cell, "source_id", step % 1_000_003)
    axis = getattr(cell, "axis", "unlabeled-axis")
    family = getattr(cell, "family", "unlabeled-family")
    interior_rows = np.flatnonzero(mask)
    background_rows = np.flatnonzero(background)
    interior_indices = interior_rows[
        _deterministic_choice(
            interior_rows.size,
            min(config.interior_cap, interior_rows.size),
            stable_seed(seed, step, source_id, axis, family, "interior"),
        )
    ]
    background_indices = background_rows[
        _deterministic_choice(
            background_rows.size,
            min(config.background_count, background_rows.size),
            stable_seed(seed, step, source_id, axis, family, "background"),
        )
    ]
    indices = torch.as_tensor(
        np.concatenate((interior_indices, background_indices)),
        device=device,
    )
    local_mask = torch.zeros(indices.numel(), dtype=torch.bool, device=device)
    local_mask[: interior_indices.size] = True
    return (
        *(value[indices] for value in tensors[:4]),
        local_mask,
        ~local_mask,
    )


def _validate_clean_sources(value: Mapping[int, npt.ArrayLike]) -> dict[int, FloatArray]:
    result: dict[int, FloatArray] = {}
    for source, rows in value.items():
        matrix = np.asarray(rows, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or not np.all(np.isfinite(matrix)):
            raise ValueError("clean source matrices must be finite and nonempty")
        result[int(source)] = matrix
    if len(result) < 2 or len({rows.shape[1] for rows in result.values()}) != 1:
        raise ValueError("GB-CMF requires at least two same-dimensional clean sources")
    return result


def _validate_cell(cell: GBCMFCell) -> GBCMFCell:
    matrices = tuple(
        np.asarray(value, dtype=np.float32)
        for value in (cell.clean, cell.shifted_clean, cell.defect, cell.shifted_defect)
    )
    if len({matrix.shape for matrix in matrices}) != 1 or matrices[0].ndim != 2:
        raise ValueError("GB-CMF cells require aligned 2D feature matrices")
    if not all(np.all(np.isfinite(matrix)) for matrix in matrices):
        raise ValueError("GB-CMF cells must be finite")
    mask = np.asarray(cell.defect_mask, dtype=bool).reshape(-1)
    background = np.asarray(cell.background_mask, dtype=bool).reshape(-1)
    if mask.size != matrices[0].shape[0] or not mask.any():
        raise ValueError("GB-CMF cell needs defect-interior rows")
    if background.size != matrices[0].shape[0] or not background.any() or np.any(mask & background):
        raise ValueError("GB-CMF cell needs disjoint background rows")
    if not cell.axis or not cell.family:
        raise ValueError("GB-CMF cell labels cannot be empty")
    return GBCMFCell(cell.source_id, *matrices, mask, background, cell.axis, cell.family)


def _validate_config(config: GBCMFConfig) -> None:
    positive = (
        config.steps,
        config.learning_rate,
        config.softmin_tau,
        config.background_count,
        config.interior_cap,
        config.memory_rows,
        config.constraint_rho,
        config.dual_update_interval,
        config.gradient_clip,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("GB-CMF positive configuration fields must be positive")
    if config.ranking_margin < 0.0 or config.proximal_weight < 0.0:
        raise ValueError("GB-CMF margin/proximal values are invalid")
    if not 0.0 < config.background_top_fraction <= 1.0:
        raise ValueError("GB-CMF background top fraction must be in (0,1]")


def _deterministic_choice(size: int, count: int, seed: int) -> npt.NDArray[np.int64]:
    if count >= size:
        return np.arange(size, dtype=np.int64)
    return np.random.default_rng(seed).choice(size, size=count, replace=False).astype(np.int64)


def _rms(value: torch.Tensor) -> float:
    return float(value.square().mean().sqrt().cpu())


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-6:
        return 1.0 if abs(numerator) <= 1e-6 else float("inf")
    return float(numerator / denominator)


def _effective_rank(value: torch.Tensor) -> float:
    centered = value - value.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    eigenvalues = singular.square().clamp_min(0.0)
    probabilities = eigenvalues / (eigenvalues.sum() + 1e-12)
    return float(torch.exp(-(probabilities * torch.log(probabilities + 1e-12)).sum()).cpu())


def _average_ranks(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(left: npt.ArrayLike, right: npt.ArrayLike) -> float:
    a = _average_ranks(np.asarray(left, dtype=np.float64).reshape(-1))
    b = _average_ranks(np.asarray(right, dtype=np.float64).reshape(-1))
    if a.size != b.size or a.size < 2:
        raise ValueError("Spearman inputs must have matching nontrivial sizes")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _quantile_drift_std(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
) -> tuple[float, float, float]:
    std = float(baseline.std(unbiased=False).clamp_min(1e-6).cpu())
    quantiles = baseline.new_tensor([0.01, 0.5, 0.99])
    drift = (torch.quantile(candidate, quantiles) - torch.quantile(baseline, quantiles)) / std
    return tuple(float(value) for value in drift.cpu())  # type: ignore[return-value]


def _logdet_share(base_density: torch.Tensor, negative_logdet: torch.Tensor) -> float:
    numerator = float(negative_logdet.std(unbiased=False).cpu())
    denominator = numerator + float(base_density.std(unbiased=False).cpu()) + 1e-12
    return numerator / denominator


def _coupling_logdet_q99(estimator: FlowDensityEstimator, features: npt.ArrayLike) -> float:
    if estimator.standardizer is None:
        raise RuntimeError("estimator must be fitted")
    tensor = torch.as_tensor(features, dtype=torch.float32, device=estimator.device)
    out = estimator.standardizer.transform(tensor)
    values: list[torch.Tensor] = []
    with torch.no_grad():
        for block in estimator.flow.blocks:
            out, logdet = block.forward(out)
            active_dim = float((1.0 - block.mask).sum().item())
            values.append((logdet.abs() / max(active_dim, 1.0)).reshape(-1))
    return float(torch.quantile(torch.cat(values), 0.99).cpu())
