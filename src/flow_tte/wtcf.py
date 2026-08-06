"""Witness-Tangent Cross-Fit refinement for a static Flow 1-NN detector.

The module is deliberately support-only.  It never consumes anomaly examples
and it does not own an inference-time state: callers rebuild one static latent
bank after refinement and continue to use exact hard 1-NN scoring.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Literal, Mapping

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as functional

from flow_tte.losses import patch_nll
from flow_tte.trainer import FlowDensityEstimator

FloatArray = npt.NDArray[np.float32]
WTCFMode = Literal[
    "crossfit_only",
    "global_isometry",
    "random_witness",
    "witness_only",
    "wtcf",
]


@dataclass(frozen=True)
class WTCFConfig:
    steps: int = 300
    learning_rate: float = 2e-5
    fold_count: int = 4
    query_rows: int = 128
    memory_rows: int = 512
    softmin_tau: float = 0.05
    cvar_fraction: float = 0.05
    nll_weight: float = 0.05
    directional_weight: float = 2.0
    gradient_clip: float = 0.5
    witness_rows_per_source: int = 48
    witness_candidates_per_pair: int = 12
    witness_cosine: float = 0.85
    witness_min_source_pairs: int = 3
    witness_min_fold_pairs: int = 3
    max_witnesses: int = 8
    unsupported_probes: int = 8
    witness_floor: float = 0.55
    witness_target: float = 0.85
    witness_ceiling: float = 1.10
    unsupported_floor: float = 0.90
    unsupported_ceiling: float = 1.10
    rollback_interval: int = 5
    audit_rows: int = 128
    geometry_ratio_floor: float = 0.80
    geometry_ratio_ceiling: float = 1.25
    pair_spearman_floor: float = 0.95
    neighbor_retention_floor: float = 0.80
    logdet_drift_std_ceiling: float = 1.0
    inverse_error_ceiling: float = 1e-5


@dataclass(frozen=True)
class WitnessSet:
    directions: FloatArray
    anchors: FloatArray
    candidate_count: int
    accepted_count: int
    source_pair_diversity: tuple[int, ...]
    fold_pair_diversity: tuple[int, ...]
    angular_stability: tuple[float, ...]


@dataclass(frozen=True)
class FunctionSpaceAudit:
    latent_rms_ratio: float
    effective_rank_ratio: float
    pair_distance_spearman: float
    score_std_ratio: float
    neighbor_retention: float
    logdet_quantile_drift_std: tuple[float, float, float]
    coupling_logdet_q99_ratio: float
    inverse_max_abs_error: float
    passed: bool


@dataclass(frozen=True)
class DirectionalAudit:
    witness_ratios: tuple[float, ...]
    unsupported_ratios: tuple[float, ...]
    passed: bool


@dataclass(frozen=True)
class WTCFTrainingResult:
    mode: str
    losses: tuple[float, ...]
    score_losses: tuple[float, ...]
    directional_losses: tuple[float, ...]
    accepted_updates: int
    rejected_update_blocks: int
    max_parameter_delta: float
    witness: WitnessSet
    final_geometry: FunctionSpaceAudit
    final_directional: DirectionalAudit
    fold_by_source: Mapping[int, int]
    finite: bool


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def make_image_disjoint_folds(
    source_ids: npt.ArrayLike,
    *,
    fold_count: int = 4,
    seed: int = 0,
) -> dict[int, int]:
    """Assign every source image to exactly one deterministic held fold."""
    sources = sorted({int(value) for value in np.asarray(source_ids).reshape(-1)})
    if fold_count < 2 or len(sources) < fold_count:
        raise ValueError("image-disjoint cross-fit needs at least one source per fold")
    order = np.asarray(sources, dtype=np.int64)
    np.random.default_rng(seed).shuffle(order)
    return {int(source): index % fold_count for index, source in enumerate(order)}


def soft_nearest_distance(
    query: torch.Tensor,
    bank: torch.Tensor,
    *,
    tau: float,
) -> torch.Tensor:
    """Differentiable soft minimum of deployed Euclidean memory distance."""
    if tau <= 0.0:
        raise ValueError("soft-1NN temperature must be positive")
    if query.ndim != 2 or bank.ndim != 2 or query.shape[1] != bank.shape[1]:
        raise ValueError("soft-1NN expects compatible [rows, dim] matrices")
    if query.shape[0] == 0 or bank.shape[0] == 0:
        raise ValueError("soft-1NN query and bank cannot be empty")
    distances = torch.cdist(query, bank, p=2.0)
    return -tau * torch.logsumexp(-distances / tau, dim=1)


def upper_cvar(values: torch.Tensor, fraction: float) -> torch.Tensor:
    """Mean of the largest fixed fraction, with at least one retained row."""
    if values.ndim != 1 or values.numel() == 0 or not 0.0 < fraction <= 1.0:
        raise ValueError("CVaR expects nonempty 1D values and fraction in (0,1]")
    count = max(1, math.ceil(values.numel() * fraction))
    return torch.topk(values, k=count).values.mean()


def discover_normal_witnesses(  # noqa: C901, PLR0912, PLR0915
    clean_by_source: Mapping[int, npt.ArrayLike],
    fold_by_source: Mapping[int, int],
    *,
    config: WTCFConfig | None = None,
    seed: int = 0,
) -> WitnessSet:
    """Mine repeated mutual-NN residual directions across image/fold pairs."""
    resolved = WTCFConfig() if config is None else config
    clean = _validate_sources(clean_by_source)
    if set(clean) != set(fold_by_source):
        raise ValueError("fold assignment must cover exactly the clean sources")
    sampled: dict[int, FloatArray] = {}
    for source, rows in clean.items():
        count = min(resolved.witness_rows_per_source, rows.shape[0])
        indices = _choice(rows.shape[0], count, stable_seed(seed, source, "witness-rows"))
        sampled[source] = rows[indices]

    candidates: list[tuple[FloatArray, FloatArray, tuple[int, int], tuple[int, int]]] = []
    for left_index, left_source in enumerate(sorted(sampled)):
        for right_source in sorted(sampled)[left_index + 1 :]:
            left_fold = int(fold_by_source[left_source])
            right_fold = int(fold_by_source[right_source])
            if left_fold == right_fold:
                continue
            left = sampled[left_source]
            right = sampled[right_source]
            distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
            right_for_left = distances.argmin(axis=1)
            left_for_right = distances.argmin(axis=0)
            mutual = [
                (left_row, int(right_row), float(distances[left_row, right_row]))
                for left_row, right_row in enumerate(right_for_left)
                if int(left_for_right[int(right_row)]) == left_row
            ]
            mutual.sort(key=lambda item: item[2])
            for left_row, right_row, _ in mutual[: resolved.witness_candidates_per_pair]:
                residual = left[left_row] - right[right_row]
                norm = float(np.linalg.norm(residual))
                if norm <= 1e-8:
                    continue
                direction = _canonical_direction(residual / norm)
                anchor = 0.5 * (left[left_row] + right[right_row])
                candidates.append(
                    (
                        direction,
                        anchor.astype(np.float32, copy=False),
                        (left_source, right_source),
                        tuple(sorted((left_fold, right_fold))),
                    ),
                )

    clusters: list[dict[str, object]] = []
    for direction, anchor, source_pair, fold_pair in candidates:
        best_index = -1
        best_cosine = -1.0
        for index, cluster in enumerate(clusters):
            center = np.asarray(cluster["direction_sum"], dtype=np.float32)
            center /= max(float(np.linalg.norm(center)), 1e-12)
            cosine = float(np.dot(direction, center))
            if cosine >= resolved.witness_cosine and cosine > best_cosine:
                best_index, best_cosine = index, cosine
        if best_index < 0:
            clusters.append(
                {
                    "direction_sum": direction.copy(),
                    "anchor_sum": anchor.copy(),
                    "count": 1,
                    "source_pairs": {source_pair},
                    "fold_pairs": {fold_pair},
                    "cosines": [1.0],
                },
            )
            continue
        cluster = clusters[best_index]
        cluster["direction_sum"] = np.asarray(cluster["direction_sum"]) + direction
        cluster["anchor_sum"] = np.asarray(cluster["anchor_sum"]) + anchor
        cluster["count"] = int(cluster["count"]) + 1
        source_pairs = cluster["source_pairs"]
        fold_pairs = cluster["fold_pairs"]
        cosines = cluster["cosines"]
        if not isinstance(source_pairs, set) or not isinstance(fold_pairs, set):
            raise TypeError("witness cluster pair state is malformed")
        if not isinstance(cosines, list):
            raise TypeError("witness cluster cosine state is malformed")
        source_pairs.add(source_pair)
        fold_pairs.add(fold_pair)
        cosines.append(best_cosine)

    accepted = [
        cluster
        for cluster in clusters
        if len(cluster["source_pairs"]) >= resolved.witness_min_source_pairs  # type: ignore[arg-type]
        and len(cluster["fold_pairs"]) >= resolved.witness_min_fold_pairs  # type: ignore[arg-type]
    ]
    accepted.sort(
        key=lambda cluster: (
            -len(cluster["source_pairs"]),  # type: ignore[arg-type]
            -len(cluster["fold_pairs"]),  # type: ignore[arg-type]
            -int(cluster["count"]),
        ),
    )
    accepted = accepted[: resolved.max_witnesses]
    dimension = next(iter(clean.values())).shape[1]
    if not accepted:
        empty = np.empty((0, dimension), dtype=np.float32)
        return WitnessSet(empty, empty.copy(), len(candidates), 0, (), (), ())
    directions: list[FloatArray] = []
    anchors: list[FloatArray] = []
    pair_counts: list[int] = []
    fold_counts: list[int] = []
    stability: list[float] = []
    for cluster in accepted:
        count = int(cluster["count"])
        direction = np.asarray(cluster["direction_sum"], dtype=np.float32)
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        directions.append(direction)
        anchors.append(np.asarray(cluster["anchor_sum"], dtype=np.float32) / count)
        pair_counts.append(len(cluster["source_pairs"]))  # type: ignore[arg-type]
        fold_counts.append(len(cluster["fold_pairs"]))  # type: ignore[arg-type]
        stability.append(float(min(cluster["cosines"])))  # type: ignore[arg-type]
    return WitnessSet(
        np.stack(directions).astype(np.float32, copy=False),
        np.stack(anchors).astype(np.float32, copy=False),
        len(candidates),
        len(directions),
        tuple(pair_counts),
        tuple(fold_counts),
        tuple(stability),
    )


def jvp_sensitivity(
    flow: torch.nn.Module,
    anchor: torch.Tensor,
    direction: torch.Tensor,
    *,
    create_graph: bool,
) -> torch.Tensor:
    """Return ||J_f(anchor) direction|| / ||direction||."""
    anchor_row = anchor.reshape(1, -1)
    direction_row = direction.reshape(1, -1)

    def transform(value: torch.Tensor) -> torch.Tensor:
        output, _ = flow(value)
        return output

    _, tangent = torch.autograd.functional.jvp(
        transform,
        anchor_row,
        direction_row,
        create_graph=create_graph,
        strict=False,
    )
    return tangent.norm() / direction_row.norm().clamp_min(1e-12)


def bounded_ratio_penalty(
    ratio: torch.Tensor,
    *,
    floor: float,
    ceiling: float,
    target_ceiling: float | None = None,
) -> torch.Tensor:
    if not 0.0 < floor <= ceiling:
        raise ValueError("directional bounds must be positive and ordered")
    penalty = functional.relu(ratio.new_tensor(floor) - ratio).square()
    penalty = penalty + functional.relu(ratio - ratio.new_tensor(ceiling)).square()
    if target_ceiling is not None:
        if not floor <= target_ceiling <= ceiling:
            raise ValueError("target ceiling must lie inside directional bounds")
        penalty = penalty + functional.relu(ratio - ratio.new_tensor(target_ceiling)).square()
    return penalty


def refine_wtcf(  # noqa: C901, PLR0912, PLR0915
    estimator: FlowDensityEstimator,
    clean_by_source: Mapping[int, npt.ArrayLike],
    *,
    mode: WTCFMode,
    config: WTCFConfig | None = None,
    seed: int = 0,
) -> WTCFTrainingResult:
    """Refine only a fitted Flow, using normal support and frozen rollbacks."""
    resolved = WTCFConfig() if config is None else config
    _validate_config(resolved)
    clean = _validate_sources(clean_by_source)
    if estimator.standardizer is None or estimator.config.transform_mode != "flow":
        raise RuntimeError("WTCF requires a fitted, standardized non-identity Flow")
    if estimator.config.condition_mode != "none" or estimator.config.spatial_context:
        raise RuntimeError("WTCF Stage -1 supports only an unconditioned patch Flow")

    device = estimator.device
    torch.use_deterministic_algorithms(mode=True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    standardized = {
        source: estimator.standardizer.transform(
            torch.as_tensor(rows, dtype=torch.float32, device=device),
        )
        for source, rows in clean.items()
    }
    folds = make_image_disjoint_folds(
        np.asarray(list(clean), dtype=np.int64),
        fold_count=resolved.fold_count,
        seed=stable_seed(seed, "folds"),
    )
    witness = discover_normal_witnesses(
        {source: tensor.detach().cpu().numpy() for source, tensor in standardized.items()},
        folds,
        config=resolved,
        seed=stable_seed(seed, "witness"),
    )
    witness_directions = torch.as_tensor(witness.directions, device=device)
    witness_anchors = torch.as_tensor(witness.anchors, device=device)
    unsupported = _unsupported_directions(
        witness_directions,
        dimension=next(iter(standardized.values())).shape[1],
        count=resolved.unsupported_probes,
        seed=stable_seed(seed, "unsupported"),
        device=device,
    )
    random_witness = _unsupported_directions(
        torch.empty((0, unsupported.shape[1]), device=device),
        dimension=unsupported.shape[1],
        count=max(witness.accepted_count, 1),
        seed=stable_seed(seed, "random-witness"),
        device=device,
    )[: witness.accepted_count]

    baseline_flow = copy.deepcopy(estimator.flow).eval()
    for parameter in baseline_flow.parameters():
        parameter.requires_grad_(requires_grad=False)
    probe, probe_sources = _audit_probe(standardized, resolved.audit_rows, seed)
    initial_parameters = tuple(
        parameter.detach().clone() for parameter in estimator.flow.parameters()
    )
    accepted_state = copy.deepcopy(estimator.flow.state_dict())
    optimizer = torch.optim.AdamW(
        estimator.flow.parameters(),
        lr=resolved.learning_rate,
        weight_decay=0.0,
    )
    accepted_optimizer_state = copy.deepcopy(optimizer.state_dict())
    losses: list[float] = []
    score_losses: list[float] = []
    direction_losses: list[float] = []
    accepted_updates = 0
    rejected_blocks = 0
    _ = estimator.flow.train()

    for step in range(resolved.steps):
        held_fold = step % resolved.fold_count
        query = _sample_fold_rows(
            standardized,
            folds,
            held_fold=held_fold,
            held=True,
            rows=resolved.query_rows,
            seed=stable_seed(seed, step, "query"),
        )
        bank = _sample_fold_rows(
            standardized,
            folds,
            held_fold=held_fold,
            held=False,
            rows=resolved.memory_rows,
            seed=stable_seed(seed, step, "bank"),
        )
        query_z, query_logdet = estimator.flow(query)
        bank_z, _ = estimator.flow(bank)
        soft = soft_nearest_distance(query_z, bank_z, tau=resolved.softmin_tau)
        with torch.no_grad():
            base_query, _ = baseline_flow(query)
            base_bank, _ = baseline_flow(bank)
            scale = torch.cdist(base_query, base_bank).min(dim=1).values.median().clamp_min(1e-6)
        score_loss = upper_cvar(soft / scale, resolved.cvar_fraction)
        nll = patch_nll(query_z, query_logdet).mean()
        nll = nll / (abs(float(estimator.train_nll_mean)) + 1e-6)
        directional = query_z.new_zeros(())

        active_directions = witness_directions
        active_anchors = witness_anchors
        if mode == "random_witness":
            active_directions = random_witness
            active_anchors = probe[: random_witness.shape[0]]
        if active_directions.shape[0] > 0 and mode in {
            "global_isometry", "random_witness", "witness_only", "wtcf",
        }:
            index = step % active_directions.shape[0]
            ratio = _relative_sensitivity(
                estimator.flow,
                baseline_flow,
                active_anchors[index],
                active_directions[index],
            )
            if mode == "global_isometry":
                directional = directional + bounded_ratio_penalty(
                    ratio,
                    floor=resolved.unsupported_floor,
                    ceiling=resolved.unsupported_ceiling,
                )
            else:
                directional = directional + bounded_ratio_penalty(
                    ratio,
                    floor=resolved.witness_floor,
                    ceiling=resolved.witness_ceiling,
                    target_ceiling=resolved.witness_target,
                )
        if mode in {"global_isometry", "random_witness", "witness_only", "wtcf"}:
            probe_index = step % unsupported.shape[0]
            anchor = probe[(step * 7) % probe.shape[0]]
            ratio = _relative_sensitivity(
                estimator.flow,
                baseline_flow,
                anchor,
                unsupported[probe_index],
            )
            directional = directional + bounded_ratio_penalty(
                ratio,
                floor=resolved.unsupported_floor,
                ceiling=resolved.unsupported_ceiling,
            )

        objective_score = query_z.new_zeros(()) if mode == "witness_only" else score_loss
        loss = objective_score + resolved.nll_weight * nll
        loss = loss + resolved.directional_weight * directional
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(estimator.flow.parameters(), resolved.gradient_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        score_losses.append(float(score_loss.detach().cpu()))
        direction_losses.append(float(directional.detach().cpu()))

        audit_due = (step + 1) % resolved.rollback_interval == 0 or step + 1 == resolved.steps
        if audit_due:
            audit = _geometry_values(
                baseline_flow, estimator.flow, probe, probe_sources, resolved,
            )
            # ``crossfit_only`` is the deliberately unguarded mechanistic arm;
            # all other refinement arms enforce the frozen rollback envelope.
            if audit.passed or mode == "crossfit_only":
                accepted_state = copy.deepcopy(estimator.flow.state_dict())
                accepted_optimizer_state = copy.deepcopy(optimizer.state_dict())
                accepted_updates = step + 1
            else:
                estimator.flow.load_state_dict(accepted_state)
                optimizer.load_state_dict(accepted_optimizer_state)
                rejected_blocks += 1

    final_geometry = _geometry_values(
        baseline_flow, estimator.flow, probe, probe_sources, resolved,
    )
    final_directional = _directional_audit(
        baseline_flow,
        estimator.flow,
        witness_anchors,
        witness_directions,
        probe,
        unsupported,
        resolved,
        global_isometry=mode == "global_isometry",
        require_witness=mode in {"random_witness", "witness_only", "wtcf"},
    )
    max_delta = max(
        float((parameter.detach() - anchor).abs().max().cpu())
        for parameter, anchor in zip(estimator.flow.parameters(), initial_parameters)
    )
    finite = bool(np.all(np.isfinite(losses))) and all(
        bool(torch.isfinite(parameter).all()) for parameter in estimator.flow.parameters()
    )
    return WTCFTrainingResult(
        mode=mode,
        losses=tuple(losses),
        score_losses=tuple(score_losses),
        directional_losses=tuple(direction_losses),
        accepted_updates=accepted_updates,
        rejected_update_blocks=rejected_blocks,
        max_parameter_delta=max_delta,
        witness=witness,
        final_geometry=final_geometry,
        final_directional=final_directional,
        fold_by_source=folds,
        finite=finite,
    )


def result_to_dict(result: WTCFTrainingResult) -> dict[str, object]:
    """JSON-friendly serialization kept here so runners cannot omit audits."""
    def convert(value: object) -> object:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    converted = convert(asdict(result))
    if not isinstance(converted, dict):
        raise TypeError("WTCF result serialization did not produce a mapping")
    return converted


def _relative_sensitivity(
    flow: torch.nn.Module,
    baseline_flow: torch.nn.Module,
    anchor: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    current = jvp_sensitivity(flow, anchor, direction, create_graph=True)
    baseline = jvp_sensitivity(
        baseline_flow, anchor.detach(), direction.detach(), create_graph=False,
    ).detach()
    return current / baseline.clamp_min(1e-8)


def _directional_audit(  # noqa: PLR0913
    baseline_flow: torch.nn.Module,
    candidate_flow: torch.nn.Module,
    witness_anchors: torch.Tensor,
    witness_directions: torch.Tensor,
    probe: torch.Tensor,
    unsupported: torch.Tensor,
    config: WTCFConfig,
    *,
    global_isometry: bool,
    require_witness: bool,
) -> DirectionalAudit:
    witness_ratios: list[float] = []
    unsupported_ratios: list[float] = []
    with torch.enable_grad():
        for anchor, direction in zip(witness_anchors, witness_directions):
            ratio = _relative_sensitivity(candidate_flow, baseline_flow, anchor, direction)
            witness_ratios.append(float(ratio.detach().cpu()))
        for index, direction in enumerate(unsupported):
            ratio = _relative_sensitivity(
                candidate_flow, baseline_flow, probe[index % probe.shape[0]], direction,
            )
            unsupported_ratios.append(float(ratio.detach().cpu()))
    witness_floor = config.unsupported_floor if global_isometry else config.witness_floor
    witness_ceiling = config.unsupported_ceiling if global_isometry else config.witness_target
    passed = (not require_witness or bool(witness_ratios)) and all(
        witness_floor <= value <= witness_ceiling for value in witness_ratios
    )
    passed = passed and all(
        config.unsupported_floor <= value <= config.unsupported_ceiling
        for value in unsupported_ratios
    )
    return DirectionalAudit(tuple(witness_ratios), tuple(unsupported_ratios), passed)


def _geometry_values(
    baseline_flow: torch.nn.Module,
    candidate_flow: torch.nn.Module,
    probe: torch.Tensor,
    sources: torch.Tensor,
    config: WTCFConfig,
) -> FunctionSpaceAudit:
    with torch.no_grad():
        base_z, base_logdet = baseline_flow(probe)
        cand_z, cand_logdet = candidate_flow(probe)
        reconstructed, _ = candidate_flow(cand_z, reverse=True)
        base_scores, base_neighbors = _source_excluded_scores(base_z, sources)
        cand_scores, cand_neighbors = _source_excluded_scores(cand_z, sources)
        rms_ratio = _ratio(_rms(cand_z), _rms(base_z))
        rank_ratio = _ratio(_effective_rank(cand_z), _effective_rank(base_z))
        score_std_ratio = _ratio(
            float(cand_scores.std(unbiased=False).cpu()),
            float(base_scores.std(unbiased=False).cpu()),
        )
        neighbor_retention = float((cand_neighbors == base_neighbors).float().mean().cpu())
        pair_spearman = _spearman_tensor(
            torch.pdist(base_z[: min(64, base_z.shape[0])]),
            torch.pdist(cand_z[: min(64, cand_z.shape[0])]),
        )
        logdet_std = float(base_logdet.std(unbiased=False).clamp_min(1e-6).cpu())
        quantiles = base_logdet.new_tensor([0.01, 0.5, 0.99])
        drift = (torch.quantile(cand_logdet, quantiles) - torch.quantile(base_logdet, quantiles))
        drift_values = tuple(float(value) for value in (drift / logdet_std).cpu())
        coupling_ratio = _ratio(
            _coupling_q99(candidate_flow, probe),
            _coupling_q99(baseline_flow, probe),
        )
        inverse_error = float((reconstructed - probe).abs().max().cpu())
    passed = config.geometry_ratio_floor <= rms_ratio <= config.geometry_ratio_ceiling
    passed = passed and config.geometry_ratio_floor <= rank_ratio <= config.geometry_ratio_ceiling
    passed = passed and (
        config.geometry_ratio_floor <= score_std_ratio <= config.geometry_ratio_ceiling
    )
    passed = passed and pair_spearman >= config.pair_spearman_floor
    passed = passed and neighbor_retention >= config.neighbor_retention_floor
    passed = passed and max(abs(value) for value in drift_values) <= config.logdet_drift_std_ceiling
    passed = passed and coupling_ratio <= config.geometry_ratio_ceiling
    passed = passed and inverse_error <= config.inverse_error_ceiling
    return FunctionSpaceAudit(
        rms_ratio,
        rank_ratio,
        pair_spearman,
        score_std_ratio,
        neighbor_retention,
        drift_values,
        coupling_ratio,
        inverse_error,
        bool(passed),
    )


def _source_excluded_scores(
    z: torch.Tensor,
    sources: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    distances = torch.cdist(z, z)
    invalid = sources[:, None] == sources[None, :]
    distances = distances.masked_fill(invalid, float("inf"))
    return distances.min(dim=1)


def _audit_probe(
    standardized: Mapping[int, torch.Tensor], rows: int, seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    per_source = max(2, math.ceil(rows / len(standardized)))
    values: list[torch.Tensor] = []
    sources: list[torch.Tensor] = []
    for source in sorted(standardized):
        matrix = standardized[source]
        count = min(per_source, matrix.shape[0])
        indices = _choice(matrix.shape[0], count, stable_seed(seed, source, "audit"))
        values.append(matrix[torch.as_tensor(indices, device=matrix.device)])
        sources.append(torch.full((count,), source, dtype=torch.long, device=matrix.device))
    return torch.cat(values)[:rows], torch.cat(sources)[:rows]


def _sample_fold_rows(  # noqa: PLR0913
    standardized: Mapping[int, torch.Tensor],
    folds: Mapping[int, int],
    *,
    held_fold: int,
    held: bool,
    rows: int,
    seed: int,
) -> torch.Tensor:
    sources = [
        source for source in sorted(standardized)
        if (folds[source] == held_fold) is held
    ]
    if not sources:
        raise ValueError("cross-fit fold produced an empty query or memory split")
    per_source = max(1, math.ceil(rows / len(sources)))
    parts: list[torch.Tensor] = []
    for source in sources:
        matrix = standardized[source]
        count = min(per_source, matrix.shape[0])
        indices = _choice(matrix.shape[0], count, stable_seed(seed, source))
        parts.append(matrix[torch.as_tensor(indices, device=matrix.device)])
    return torch.cat(parts)[:rows]


def _unsupported_directions(
    witnesses: torch.Tensor,
    *,
    dimension: int,
    count: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    directions: list[torch.Tensor] = []
    for _ in range(count * 8):
        value = torch.randn(dimension, generator=generator, device=device)
        for witness in witnesses:
            value = value - torch.dot(value, witness) * witness
        for previous in directions:
            value = value - torch.dot(value, previous) * previous
        norm = value.norm()
        if float(norm.cpu()) > 1e-6:
            directions.append(value / norm)
        if len(directions) == count:
            break
    if len(directions) != count:
        raise RuntimeError("could not construct deterministic unsupported probes")
    return torch.stack(directions)


def _coupling_q99(flow: torch.nn.Module, features: torch.Tensor) -> float:
    out = features
    parts: list[torch.Tensor] = []
    for block in flow.blocks:  # type: ignore[attr-defined]
        out, logdet = block(out)
        active = float((1.0 - block.mask).sum().item())
        parts.append(logdet.abs() / max(active, 1.0))
    return float(torch.quantile(torch.cat(parts), 0.99).cpu())


def _effective_rank(value: torch.Tensor) -> float:
    centered = value - value.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    probability = energy / energy.sum().clamp_min(1e-12)
    entropy = -(probability * torch.log(probability.clamp_min(1e-12))).sum()
    return float(torch.exp(entropy).cpu())


def _spearman_tensor(left: torch.Tensor, right: torch.Tensor) -> float:
    a = left.detach().cpu().numpy()
    b = right.detach().cpu().numpy()
    if a.size < 2 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 1.0 if np.array_equal(a, b) else 0.0
    ar = np.empty_like(a, dtype=np.float64)
    br = np.empty_like(b, dtype=np.float64)
    ar[np.argsort(a, kind="mergesort")] = np.arange(a.size, dtype=np.float64)
    br[np.argsort(b, kind="mergesort")] = np.arange(b.size, dtype=np.float64)
    return float(np.corrcoef(ar, br)[0, 1])


def _canonical_direction(value: np.ndarray) -> FloatArray:
    result = np.asarray(value, dtype=np.float32)
    pivot = int(np.argmax(np.abs(result)))
    if result[pivot] < 0:
        result = -result
    return result.astype(np.float32, copy=False)


def _rms(value: torch.Tensor) -> float:
    return float(value.square().mean().sqrt().cpu())


def _ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 1.0 if abs(numerator) < 1e-12 else float("inf")
    return float(numerator / denominator)


def _choice(size: int, count: int, seed: int) -> npt.NDArray[np.int64]:
    if count >= size:
        return np.arange(size, dtype=np.int64)
    return np.random.default_rng(seed).choice(size, size=count, replace=False).astype(np.int64)


def _validate_sources(value: Mapping[int, npt.ArrayLike]) -> dict[int, FloatArray]:
    clean: dict[int, FloatArray] = {}
    for source, rows in value.items():
        matrix = np.asarray(rows, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] < 2 or not bool(np.isfinite(matrix).all()):
            raise ValueError("clean source matrices must be finite, 2D, and nontrivial")
        clean[int(source)] = matrix
    if len(clean) < 4 or len({matrix.shape[1] for matrix in clean.values()}) != 1:
        raise ValueError("WTCF requires at least four same-dimensional source images")
    return clean


def _validate_config(config: WTCFConfig) -> None:
    positive = (
        config.steps,
        config.learning_rate,
        config.fold_count,
        config.query_rows,
        config.memory_rows,
        config.softmin_tau,
        config.cvar_fraction,
        config.gradient_clip,
        config.witness_rows_per_source,
        config.witness_candidates_per_pair,
        config.max_witnesses,
        config.unsupported_probes,
        config.rollback_interval,
        config.audit_rows,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("WTCF positive configuration fields must be positive")
    if not 0.0 < config.cvar_fraction <= 1.0 or not 0.0 < config.witness_cosine <= 1.0:
        raise ValueError("WTCF fractions/cosines are outside their domain")
    if not config.witness_floor <= config.witness_target <= config.witness_ceiling:
        raise ValueError("WTCF witness bounds are inconsistent")
    if not config.unsupported_floor <= 1.0 <= config.unsupported_ceiling:
        raise ValueError("WTCF unsupported bounds must contain baseline ratio one")
