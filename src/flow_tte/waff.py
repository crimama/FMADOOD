from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as torch_functional

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class WaffConfig:
    objective_mode: str = "waff"
    steps: int = 300
    learning_rate: float = 0.02
    ranking_margin: float = 0.5
    background_fraction: float = 0.05
    factorial_weight: float = 0.25
    anchor_weight: float = 1e-3
    constraint_rho: float = 2.0
    constraint_update_interval: int = 10
    tail_quantile: float = 0.995
    gradient_clip: float = 5.0
    weight_sum: float = 1.25


@dataclass(frozen=True)
class WaffFactorialCell:
    clean: FloatArray
    shifted_clean: FloatArray
    defect: FloatArray
    shifted_defect: FloatArray
    defect_mask: BoolArray
    background_mask: BoolArray
    axis: str
    family: str


@dataclass(frozen=True)
class WaffScorer:
    weight: FloatArray
    initial_loss: float
    final_loss: float
    defect_constraint: float
    normal_tail_constraint: float
    training_cells: int


@dataclass(frozen=True)
class WaffFeasibilityAudit:
    grid_points: int
    min_defect_constraint: float
    min_normal_tail_constraint: float
    min_joint_violation: float
    best_joint_weight: FloatArray
    joint_feasible: bool


def source_excluded_nearest_distance(  # noqa: PLR0913
    query_features: torch.Tensor,
    bank_features: torch.Tensor,
    query_source_ids: torch.Tensor,
    bank_source_ids: torch.Tensor,
    *,
    chunk_size: int = 8192,
    squared: bool = False,
) -> torch.Tensor:
    """Return exact 1-NN distances while excluding every same-source bank row."""
    if query_features.ndim != 2 or bank_features.ndim != 2:
        raise ValueError("query_features and bank_features must be two-dimensional")
    if query_features.shape[1] != bank_features.shape[1] or bank_features.shape[0] == 0:
        raise ValueError("query and bank feature dimensions must match and bank must be non-empty")
    if query_source_ids.ndim != 1 or query_source_ids.shape[0] != query_features.shape[0]:
        raise ValueError("query_source_ids must have one value per query row")
    if bank_source_ids.ndim != 1 or bank_source_ids.shape[0] != bank_features.shape[0]:
        raise ValueError("bank_source_ids must have one value per bank row")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    values: list[torch.Tensor] = []
    for start in range(0, query_features.shape[0], chunk_size):
        stop = min(start + chunk_size, query_features.shape[0])
        distances = torch.cdist(query_features[start:stop], bank_features, p=2.0)
        excluded = query_source_ids[start:stop, None] == bank_source_ids[None, :]
        distances = distances.masked_fill(excluded, torch.inf)
        nearest = distances.min(dim=1).values
        if not bool(torch.isfinite(nearest).all()):
            raise ValueError("each query source needs at least one different-source bank row")
        values.append(nearest.square() if squared else nearest)
    return torch.cat(values, dim=0)


def score_waff_components(scorer: WaffScorer, components: npt.ArrayLike) -> FloatArray:
    matrix = _component_matrix("components", components)
    if scorer.weight.shape != (2,) or not bool(np.all(np.isfinite(scorer.weight))):
        raise ValueError("WAFF scorer weight must contain two finite values")
    return np.asarray(matrix @ scorer.weight, dtype=np.float32)


def score_waff_components_tensor(
    scorer: WaffScorer,
    components: torch.Tensor,
) -> torch.Tensor:
    if components.ndim != 2 or components.shape[1] != 2:
        raise ValueError("components must have shape [rows, 2]")
    weight = torch.as_tensor(scorer.weight, device=components.device, dtype=components.dtype)
    return components @ weight


def train_waff_scorer(
    cells: Sequence[WaffFactorialCell],
    *,
    config: WaffConfig | None = None,
    initial_weight: tuple[float, float] = (1.0, 0.25),
    device: str = "cpu",
) -> WaffScorer:
    resolved = WaffConfig() if config is None else config
    _validate_config(resolved)
    validated = tuple(_validate_cell(cell) for cell in cells)
    if not validated:
        raise ValueError("WAFF training requires at least one factorial cell")
    if len({cell.axis for cell in validated}) < 2:
        raise ValueError("WAFF training requires at least two held shift axes")
    if len({cell.family for cell in validated}) < 2:
        raise ValueError("WAFF training requires at least two defect families")

    initial = torch.as_tensor(initial_weight, dtype=torch.float32, device=device)
    if (
        initial.shape != (2,)
        or not bool(torch.isfinite(initial).all())
        or bool((initial <= 0).any())
    ):
        raise ValueError("initial_weight must contain two positive finite values")
    if not bool(torch.isclose(initial.sum(), torch.tensor(resolved.weight_sum, device=device))):
        raise ValueError("initial_weight sum must equal the frozen WAFF weight_sum")
    raw_weight = torch.log(initial / initial.sum()).detach().clone().requires_grad_()
    optimizer = torch.optim.Adam((raw_weight,), lr=resolved.learning_rate)
    lambda_defect = torch.zeros((), dtype=torch.float32, device=device)
    lambda_tail = torch.zeros((), dtype=torch.float32, device=device)
    tensor_cells = tuple(_cell_to_tensors(cell, device) for cell in validated)
    losses: list[float] = []

    for step in range(resolved.steps):
        weight = resolved.weight_sum * torch.softmax(raw_weight, dim=0)
        objective, defect_constraint, tail_constraint = _waff_objective(
            tensor_cells,
            weight,
            resolved,
        )
        loss = objective
        if resolved.objective_mode == "waff":
            positive_defect = torch.relu(defect_constraint)
            positive_tail = torch.relu(tail_constraint)
            loss = loss + lambda_defect * defect_constraint + 0.5 * resolved.constraint_rho * (
                positive_defect.square()
            )
            loss = loss + lambda_tail * tail_constraint + 0.5 * resolved.constraint_rho * (
                positive_tail.square()
            )
        loss = loss + resolved.anchor_weight * torch.mean((weight - initial).square())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_((raw_weight,), max_norm=resolved.gradient_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (
            resolved.objective_mode == "waff"
            and (step + 1) % resolved.constraint_update_interval == 0
        ):
            lambda_defect = torch.clamp(
                lambda_defect + resolved.constraint_rho * defect_constraint.detach(),
                min=0.0,
            )
            lambda_tail = torch.clamp(
                lambda_tail + resolved.constraint_rho * tail_constraint.detach(),
                min=0.0,
            )

    if not losses or not bool(np.all(np.isfinite(losses))):
        raise RuntimeError("WAFF optimization produced a non-finite loss")
    final_weight = resolved.weight_sum * torch.softmax(raw_weight, dim=0)
    with torch.no_grad():
        _, final_defect_constraint, final_tail_constraint = _waff_objective(
            tensor_cells,
            final_weight,
            resolved,
        )
    learned_weight = final_weight.detach().cpu().numpy()
    return WaffScorer(
        weight=np.asarray(learned_weight, dtype=np.float32),
        initial_loss=losses[0],
        final_loss=losses[-1],
        defect_constraint=float(final_defect_constraint.cpu()),
        normal_tail_constraint=float(final_tail_constraint.cpu()),
        training_cells=len(validated),
    )


def waff_constraint_values(
    cells: Sequence[WaffFactorialCell],
    weight: torch.Tensor,
    *,
    config: WaffConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expose differentiable constraints for the L0 gradient audit."""
    resolved = WaffConfig() if config is None else config
    _validate_config(resolved)
    validated = tuple(_validate_cell(cell) for cell in cells)
    if weight.shape != (2,):
        raise ValueError("weight must have shape [2]")
    tensor_cells = tuple(_cell_to_tensors(cell, str(weight.device)) for cell in validated)
    _, defect_constraint, tail_constraint = _waff_objective(
        tensor_cells,
        weight,
        resolved,
    )
    return defect_constraint, tail_constraint


def audit_weight_simplex_feasibility(
    cells: Sequence[WaffFactorialCell],
    *,
    config: WaffConfig | None = None,
    grid_points: int = 101,
    device: str = "cpu",
) -> WaffFeasibilityAudit:
    """Audit whether any fixed-scale positive two-component mixture satisfies both constraints."""
    resolved = WaffConfig() if config is None else config
    _validate_config(resolved)
    if grid_points < 2:
        raise ValueError("grid_points must be at least two")
    validated = tuple(_validate_cell(cell) for cell in cells)
    if not validated:
        raise ValueError("feasibility audit requires at least one factorial cell")
    tensor_cells = tuple(_cell_to_tensors(cell, device) for cell in validated)
    records: list[tuple[float, float, float, np.ndarray]] = []
    with torch.no_grad():
        for alpha in torch.linspace(0.0, 1.0, steps=grid_points, device=device):
            weight = resolved.weight_sum * torch.stack((alpha, 1.0 - alpha))
            _, defect, tail = _waff_objective(tensor_cells, weight, resolved)
            defect_value = float(defect.cpu())
            tail_value = float(tail.cpu())
            records.append(
                (
                    defect_value,
                    tail_value,
                    max(defect_value, tail_value),
                    weight.cpu().numpy().astype(np.float32, copy=False),
                ),
            )
    best = min(records, key=lambda item: item[2])
    return WaffFeasibilityAudit(
        grid_points=grid_points,
        min_defect_constraint=min(item[0] for item in records),
        min_normal_tail_constraint=min(item[1] for item in records),
        min_joint_violation=best[2],
        best_joint_weight=np.asarray(best[3], dtype=np.float32),
        joint_feasible=best[2] <= 0.0,
    )


@dataclass(frozen=True)
class _TensorCell:
    clean: torch.Tensor
    shifted_clean: torch.Tensor
    defect: torch.Tensor
    shifted_defect: torch.Tensor
    defect_mask: torch.Tensor
    background_mask: torch.Tensor
    axis: str
    family: str


def _waff_objective(
    cells: Sequence[_TensorCell],
    weight: torch.Tensor,
    config: WaffConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ranks: list[torch.Tensor] = []
    factorial: list[torch.Tensor] = []
    axis_defect: dict[str, list[torch.Tensor]] = {}
    axis_tail: dict[str, list[torch.Tensor]] = {}
    for cell in cells:
        clean_score = cell.clean @ weight
        shifted_clean_score = cell.shifted_clean @ weight
        defect_score = cell.defect @ weight
        shifted_defect_score = cell.shifted_defect @ weight
        defect_rank = _ranking_loss(
            defect_score,
            cell.defect_mask,
            cell.background_mask,
            config,
        )
        shifted_rank = _ranking_loss(
            shifted_defect_score,
            cell.defect_mask,
            cell.background_mask,
            config,
        )
        ranks.extend((defect_rank, shifted_rank))
        factorial.append(
            torch.mean(
                torch.square(
                    (defect_score - clean_score)
                    - (shifted_defect_score - shifted_clean_score),
                ),
            ),
        )
        axis_defect.setdefault(cell.axis, []).append(shifted_rank - defect_rank)
        axis_tail.setdefault(cell.axis, []).append(
            torch.quantile(shifted_clean_score, config.tail_quantile)
            - torch.quantile(clean_score, config.tail_quantile),
        )
    defect_constraint = torch.stack(
        [torch.stack(values).mean() for values in axis_defect.values()],
    ).max()
    tail_constraint = torch.stack(
        [torch.stack(values).mean() for values in axis_tail.values()],
    ).max()
    objective = torch.stack(ranks).mean()
    if config.objective_mode != "nonfactorial":
        objective = objective + config.factorial_weight * torch.stack(factorial).mean()
    return objective, defect_constraint, tail_constraint


def _ranking_loss(
    scores: torch.Tensor,
    defect_mask: torch.Tensor,
    background_mask: torch.Tensor,
    config: WaffConfig,
) -> torch.Tensor:
    interior = scores[defect_mask]
    background = scores[background_mask]
    if interior.numel() == 0 or background.numel() == 0:
        raise ValueError("every WAFF cell needs defect-interior and background patches")
    k = max(1, int(np.ceil(background.numel() * config.background_fraction)))
    top_background = torch.topk(background, k=k).values.mean()
    return torch_functional.softplus(config.ranking_margin + top_background - interior.mean())


def _component_matrix(name: str, value: npt.ArrayLike) -> FloatArray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != 2:
        message = f"{name} must have shape [nonzero rows, 2]"
        raise ValueError(message)
    if not bool(np.all(np.isfinite(matrix))):
        message = f"{name} must contain only finite values"
        raise ValueError(message)
    return matrix


def _validate_cell(cell: WaffFactorialCell) -> WaffFactorialCell:
    arrays = tuple(
        _component_matrix(name, value)
        for name, value in (
            ("clean", cell.clean),
            ("shifted_clean", cell.shifted_clean),
            ("defect", cell.defect),
            ("shifted_defect", cell.shifted_defect),
        )
    )
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("WAFF factorial cells must have exactly aligned component shapes")
    mask = np.asarray(cell.defect_mask, dtype=bool).reshape(-1)
    background = np.asarray(cell.background_mask, dtype=bool).reshape(-1)
    if mask.size != arrays[0].shape[0] or not bool(mask.any()):
        raise ValueError("defect_mask must contain aligned interior rows")
    if background.size != arrays[0].shape[0] or not bool(background.any()):
        raise ValueError("background_mask must contain aligned background rows")
    if bool(np.any(mask & background)):
        raise ValueError("defect and background masks must be disjoint")
    if not cell.axis or not cell.family:
        raise ValueError("WAFF cells require non-empty axis and family labels")
    return WaffFactorialCell(*arrays, mask, background, cell.axis, cell.family)


def _cell_to_tensors(cell: WaffFactorialCell, device: str) -> _TensorCell:
    return _TensorCell(
        clean=torch.as_tensor(cell.clean, dtype=torch.float32, device=device),
        shifted_clean=torch.as_tensor(cell.shifted_clean, dtype=torch.float32, device=device),
        defect=torch.as_tensor(cell.defect, dtype=torch.float32, device=device),
        shifted_defect=torch.as_tensor(cell.shifted_defect, dtype=torch.float32, device=device),
        defect_mask=torch.as_tensor(cell.defect_mask, dtype=torch.bool, device=device),
        background_mask=torch.as_tensor(cell.background_mask, dtype=torch.bool, device=device),
        axis=cell.axis,
        family=cell.family,
    )


def _validate_config(config: WaffConfig) -> None:
    if config.objective_mode not in ("nonfactorial", "mean_factorial", "waff"):
        raise ValueError("invalid WAFF objective_mode")
    if config.steps <= 0 or config.learning_rate <= 0.0:
        raise ValueError("WAFF steps and learning_rate must be positive")
    if config.ranking_margin < 0.0 or not 0.0 < config.background_fraction <= 1.0:
        raise ValueError("invalid WAFF ranking configuration")
    if config.factorial_weight < 0.0 or config.anchor_weight < 0.0:
        raise ValueError("WAFF loss weights must be non-negative")
    if config.constraint_rho <= 0.0 or config.constraint_update_interval <= 0:
        raise ValueError("invalid WAFF augmented-Lagrangian configuration")
    if (
        not 0.0 < config.tail_quantile < 1.0
        or config.gradient_clip <= 0.0
        or config.weight_sum <= 0.0
    ):
        raise ValueError("invalid WAFF tail or gradient configuration")
