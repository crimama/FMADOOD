"""Shared-latent scoring variants and metrics for the FlowTTE Phase-3 suite.

The suite deliberately owns only scoring.  Feature extraction, DVT denoising,
flow fitting, and flow evaluation remain in the existing FlowTTE pipeline.  In
particular, ``raw_1nn`` delegates to :func:`score_flow_memory` so its numerical
path remains the anchor implementation rather than a reimplementation of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import torch

from flow_tte.config import ScoreConfig
from flow_tte.memory import TorchMemoryBank
from flow_tte.scoring import (
    ScoreCalibration,
    ScoreInputs,
    ScoreResult,
    image_top_percent_mean,
    score_flow_memory,
)

VARIANT_NAMES = (
    "raw_1nn",
    "density_normalized_1nn_k3",
    "density_normalized_1nn_k5",
    "density_normalized_1nn_k10",
    "density_normalized_1nn_k20",
    "knn_mean_k5",
    "knn_quantile_k10_q0.5",
    "shrinkage_mahalanobis",
    "global_pca_residual",
)
DENSITY_K_VALUES = (3, 5, 10, 20)
_EPSILON = 1e-8
_RHO_QUANTILE = 0.01
_PCA_EXPLAINED_VARIANCE = 0.95
_PCA_MAX_COMPONENTS = 256
_STRUCTURE_8 = np.ones((3, 3), dtype=np.uint8)
VARIANT_DEFINITIONS = {
    "raw_1nn": "existing score_flow_memory raw Euclidean 1-NN baseline",
    **{
        f"density_normalized_1nn_k{k}": (
            f"d(q,nn1)/(median distance from nn1 to its {k} nearest other support latents+eps)"
        )
        for k in DENSITY_K_VALUES
    },
    "knn_mean_k5": "mean Euclidean distance to five nearest support latents",
    "knn_quantile_k10_q0.5": "median Euclidean distance to ten nearest support latents",
    "shrinkage_mahalanobis": "distance to one support Gaussian with inferred shrinkage covariance",
    "global_pca_residual": "orthogonal residual to 95%-energy global PCA, capped at 256 components",
}


@dataclass(frozen=True)
class DensityProfile:
    """Per-bank-entry local scale after the preregistered p1 floor."""

    rho: torch.Tensor
    floor: float
    clamp_count: int
    bank_size: int
    k: int
    effective_k: int


@dataclass(frozen=True)
class MahalanobisState:
    mean: torch.Tensor
    cholesky: torch.Tensor
    shrinkage: float
    jitter: float


@dataclass(frozen=True)
class PCAState:
    mean: torch.Tensor
    components: torch.Tensor
    rank: int
    explained_variance: float


@dataclass(frozen=True)
class ScorerSuiteState:
    bank: TorchMemoryBank
    config: ScoreConfig
    calibrations: Mapping[str, ScoreCalibration]
    score_caps: Mapping[str, float]
    density_profiles: Mapping[int, DensityProfile]
    mahalanobis: MahalanobisState
    pca: PCAState
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class NativeMapRecord:
    """One native-resolution map and its evaluation labels."""

    split: str
    score: npt.NDArray[np.float16]
    gt: npt.NDArray[np.bool_]
    stem: str = ""


def calibration_indices(n_rows: int, sample_size: int, device: torch.device) -> torch.Tensor:
    """Return exactly the deterministic subset used by ``ScoreCalibration.fit``."""

    if 1 < sample_size < n_rows:
        return torch.linspace(0, n_rows - 1, steps=sample_size, device=device).round().long()
    return torch.arange(n_rows, device=device)


def calibration_from_distances(distances: torch.Tensor) -> ScoreCalibration:
    if distances.numel() <= 1:
        return ScoreCalibration(distance_mean=0.0, distance_std=1.0)
    std = distances.std(unbiased=False).clamp_min(1e-6)
    return ScoreCalibration(
        distance_mean=float(distances.mean().detach().cpu()),
        distance_std=float(std.detach().cpu()),
    )


def _median_sorted(values: torch.Tensor, k: int) -> torch.Tensor:
    """Conventional median (average of middle values for even k)."""

    middle = k // 2
    if k % 2:
        return values[:, middle]
    return 0.5 * (values[:, middle - 1] + values[:, middle])


def leave_one_out_knn(
    features: torch.Tensor,
    k: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact Euclidean k-NN with the row's own bank entry excluded.

    Explicit diagonal masking is required: merely discarding the first result
    is incorrect when the bank contains duplicate vectors.
    """

    if features.ndim != 2:
        raise ValueError("features must be a 2D tensor")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    n_rows = int(features.shape[0])
    if n_rows <= 1:
        raise ValueError("leave-one-out k-NN requires at least two rows")
    effective_k = min(k, n_rows - 1)
    values: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        distances = torch.cdist(features[start:stop], features, p=2.0)
        local_rows = torch.arange(stop - start, device=features.device)
        global_rows = torch.arange(start, stop, device=features.device)
        distances[local_rows, global_rows] = torch.inf
        nearest = torch.topk(distances, k=effective_k, largest=False, sorted=True, dim=1)
        values.append(nearest.values)
        indices.append(nearest.indices)
    return torch.cat(values), torch.cat(indices)


def fit_density_profiles(
    bank_features: torch.Tensor,
    k_values: Sequence[int] = DENSITY_K_VALUES,
    chunk_size: int = 512,
) -> dict[int, DensityProfile]:
    """Fit all local-density profiles with one leave-one-out distance pass."""

    requested = tuple(sorted({int(k) for k in k_values}))
    if not requested or requested[0] <= 0:
        raise ValueError("k values must be positive")
    distances, _ = leave_one_out_knn(bank_features, requested[-1], chunk_size)
    profiles: dict[int, DensityProfile] = {}
    for k in requested:
        effective_k = min(k, int(distances.shape[1]))
        raw_rho = _median_sorted(distances[:, :effective_k], effective_k)
        floor_tensor = torch.quantile(raw_rho, _RHO_QUANTILE)
        clamped = raw_rho.clamp_min(floor_tensor)
        profiles[k] = DensityProfile(
            rho=clamped,
            floor=float(floor_tensor.detach().cpu()),
            clamp_count=int((raw_rho < floor_tensor).sum().detach().cpu()),
            bank_size=int(bank_features.shape[0]),
            k=k,
            effective_k=effective_k,
        )
    return profiles


def pca_rank_for_explained_variance(
    singular_values: torch.Tensor,
    target: float = _PCA_EXPLAINED_VARIANCE,
) -> tuple[int, float]:
    """Choose the smallest rank whose cumulative explained variance is >= target."""

    if not 0.0 < target <= 1.0:
        raise ValueError("target must be in (0, 1]")
    variance = singular_values.square()
    total = variance.sum()
    if not bool(torch.isfinite(total)) or float(total.detach().cpu()) <= 0.0:
        return 0, 0.0
    cumulative = torch.cumsum(variance, dim=0) / total
    rank = int(torch.searchsorted(cumulative, cumulative.new_tensor(target)).item()) + 1
    rank = min(rank, int(singular_values.numel()))
    return rank, float(cumulative[rank - 1].detach().cpu())


def fit_mahalanobis(
    bank_features: torch.Tensor,
    shrinkage: float | None = None,
) -> MahalanobisState:
    if bank_features.ndim != 2 or bank_features.shape[0] <= 1:
        raise ValueError("Mahalanobis fitting requires at least two 2D rows")
    x = bank_features.float()
    mean = x.mean(dim=0)
    centered = x - mean
    covariance = centered.T @ centered / x.shape[0]
    target_scale = torch.trace(covariance) / covariance.shape[0]
    if not bool(torch.isfinite(target_scale)) or float(target_scale.detach().cpu()) <= 0.0:
        raise ValueError("Mahalanobis covariance has non-positive total variance")
    if shrinkage is None:
        # Oracle Approximating Shrinkage is a Ledoit-Wolf-family estimator:
        # one class-agnostic coefficient is inferred from the support bank.
        dimension = covariance.shape[0]
        alpha = covariance.square().mean()
        denominator = (x.shape[0] + 1.0) * (alpha - target_scale.square() / dimension)
        if float(denominator.detach().cpu()) <= 0.0:
            shrinkage = 1.0
        else:
            numerator = alpha + target_scale.square()
            shrinkage = float((numerator / denominator).clamp(0.0, 1.0).detach().cpu())
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1]")
    identity = torch.eye(covariance.shape[0], device=x.device, dtype=torch.float32)
    shrunk = (1.0 - shrinkage) * covariance + shrinkage * target_scale * identity
    jitter = 0.0
    try:
        cholesky = torch.linalg.cholesky(shrunk)
    except torch.linalg.LinAlgError:
        # A fixed relative retry protects against float32 roundoff.  It is
        # surfaced in metadata rather than silently changing the estimator.
        jitter = float(target_scale.detach().cpu()) * 1e-6
        cholesky = torch.linalg.cholesky(shrunk + jitter * identity)
    return MahalanobisState(mean=mean, cholesky=cholesky, shrinkage=shrinkage, jitter=jitter)


def mahalanobis_distances(
    queries: torch.Tensor,
    state: MahalanobisState,
    chunk_size: int,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for start in range(0, queries.shape[0], chunk_size):
        centered = queries[start : start + chunk_size].float() - state.mean
        solved = torch.linalg.solve_triangular(state.cholesky, centered.T, upper=False)
        parts.append(solved.square().sum(dim=0).clamp_min(0.0).sqrt())
    return torch.cat(parts)


def fit_pca(
    bank_features: torch.Tensor,
    target: float = _PCA_EXPLAINED_VARIANCE,
    max_components: int = _PCA_MAX_COMPONENTS,
) -> PCAState:
    if bank_features.ndim != 2 or bank_features.shape[0] <= 1:
        raise ValueError("PCA fitting requires at least two 2D rows")
    x = bank_features.float()
    mean = x.mean(dim=0)
    _u, singular_values, vh = torch.linalg.svd(x - mean, full_matrices=False)
    requested_rank, _ = pca_rank_for_explained_variance(singular_values, target)
    rank = min(requested_rank, max_components, int(bank_features.shape[1]))
    variance = singular_values.square()
    explained = float((variance[:rank].sum() / variance.sum()).detach().cpu()) if rank else 0.0
    components = vh[:rank].contiguous()
    return PCAState(mean=mean, components=components, rank=rank, explained_variance=explained)


def pca_residual_distances(
    queries: torch.Tensor,
    state: PCAState,
    chunk_size: int,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for start in range(0, queries.shape[0], chunk_size):
        centered = queries[start : start + chunk_size].float() - state.mean
        if state.rank:
            centered = centered - (centered @ state.components.T) @ state.components
        parts.append(torch.linalg.vector_norm(centered, dim=1))
    return torch.cat(parts)


def fit_scorer_suite(bank_features: torch.Tensor, config: ScoreConfig) -> ScorerSuiteState:
    """Fit all scorer state once for one object's fixed latent memory bank."""

    if bank_features.ndim != 2:
        raise ValueError("bank_features must be 2D")
    if config.use_squared_distance:
        raise ValueError("Phase-3 variants are defined for Euclidean, not squared, distance")
    if config.score_mode != "latent_distance":
        raise ValueError("Phase-3 scorer suite requires score_mode='latent_distance'")
    if config.context_mode != "none":
        raise ValueError("Phase-3 scorer suite currently requires context_mode='none'")
    bank_features = bank_features.detach().float()
    bank = TorchMemoryBank()
    bank.fit(bank_features)
    raw_calibration = ScoreCalibration.fit(bank_features, config)
    profiles = fit_density_profiles(bank_features, chunk_size=config.query_chunk_size)

    indices = calibration_indices(
        int(bank_features.shape[0]),
        config.calibration_sample_size,
        bank_features.device,
    )
    calibration_features = bank_features[indices]
    calibration_max_k = max(*DENSITY_K_VALUES, 10)
    loo_distances, loo_indices = leave_one_out_knn(
        calibration_features,
        calibration_max_k,
        config.query_chunk_size,
    )
    calibration_profiles = fit_density_profiles(
        calibration_features,
        chunk_size=config.query_chunk_size,
    )
    calibrations: dict[str, ScoreCalibration] = {"raw_1nn": raw_calibration}
    score_caps: dict[str, float] = {}
    for k in DENSITY_K_VALUES:
        distances = loo_distances[:, 0] / (
            calibration_profiles[k].rho[loo_indices[:, 0]] + _EPSILON
        )
        name = f"density_normalized_1nn_k{k}"
        score_caps[name] = float(torch.quantile(distances, 0.999).detach().cpu())
        calibrations[name] = calibration_from_distances(distances.clamp_max(score_caps[name]))
    mean_k = min(5, int(loo_distances.shape[1]))
    quantile_k = min(10, int(loo_distances.shape[1]))
    calibrations["knn_mean_k5"] = calibration_from_distances(
        loo_distances[:, :mean_k].mean(1),
    )
    calibrations["knn_quantile_k10_q0.5"] = calibration_from_distances(
        _median_sorted(loo_distances[:, :quantile_k], quantile_k),
    )

    mahalanobis = fit_mahalanobis(bank_features)
    mahal_support = mahalanobis_distances(
        calibration_features,
        mahalanobis,
        config.query_chunk_size,
    )
    calibrations["shrinkage_mahalanobis"] = calibration_from_distances(mahal_support)
    pca = fit_pca(bank_features)
    pca_support = pca_residual_distances(calibration_features, pca, config.query_chunk_size)
    calibrations["global_pca_residual"] = calibration_from_distances(pca_support)

    metadata: dict[str, Any] = {
        "variants": list(VARIANT_NAMES),
        "variant_definitions": VARIANT_DEFINITIONS,
        "class_agnostic": True,
        "per_object_hyperparameters": False,
        "bank_size": int(bank_features.shape[0]),
        "latent_dimension": int(bank_features.shape[1]),
        "calibration_sample_size": int(calibration_features.shape[0]),
        "calibration_sampling": "anchor_linspace_round",
        "nn_variant_calibration": (
            "leave-one-out within the anchor-style calibration subset; "
            "density rho and its p1 floor are recomputed on that subset"
        ),
        "density_rho_epsilon": _EPSILON,
        "density_rho_floor_quantile": _RHO_QUANTILE,
        "density_profiles": {
            f"k{k}": {
                "floor": profiles[k].floor,
                "clamp_count": profiles[k].clamp_count,
                "effective_k": profiles[k].effective_k,
                "support_loo_p99_9_cap": score_caps[f"density_normalized_1nn_k{k}"],
            }
            for k in DENSITY_K_VALUES
        },
        "mahalanobis": {
            "shrinkage": mahalanobis.shrinkage,
            "cholesky_jitter": mahalanobis.jitter,
            "calibration": "in_sample_support_distances",
            "optimism_bias": True,
        },
        "pca": {
            "target_explained_variance": _PCA_EXPLAINED_VARIANCE,
            "max_components": _PCA_MAX_COMPONENTS,
            "rank": pca.rank,
            "explained_variance": pca.explained_variance,
            "calibration": "in_sample_support_distances",
            "optimism_bias": True,
        },
    }
    return ScorerSuiteState(
        bank=bank,
        config=config,
        calibrations=calibrations,
        score_caps=score_caps,
        density_profiles=profiles,
        mahalanobis=mahalanobis,
        pca=pca,
        metadata=metadata,
    )


def _compose_result(
    raw_distances: torch.Tensor,
    inputs: ScoreInputs,
    config: ScoreConfig,
    calibration: ScoreCalibration,
) -> ScoreResult:
    distance_scores = calibration.normalize_distance(raw_distances)
    patch_scores = (
        config.distance_weight * distance_scores + config.density_weight * inputs.nll_penalty
    )
    image_scores = image_top_percent_mean(
        patch_scores,
        inputs.image_indices,
        inputs.n_images,
        config.top_percent,
    )
    return ScoreResult(
        patch_scores=patch_scores,
        image_scores=image_scores,
        image_score=float(image_scores.max().detach().cpu()),
        distances=raw_distances,
        distance_scores=distance_scores,
        density_penalty=inputs.nll_penalty,
    )


def score_scorer_suite(
    inputs: ScoreInputs,
    state: ScorerSuiteState,
) -> dict[str, ScoreResult]:
    """Score every variant from a single already-computed flow evaluation."""

    config = state.config
    results = {
        "raw_1nn": score_flow_memory(
            inputs,
            state.bank,
            config,
            state.calibrations["raw_1nn"],
        ),
    }
    nearest = state.bank.query(
        inputs.query_z,
        k=10,
        chunk_size=config.query_chunk_size,
        squared=False,
    )
    nn_distance = nearest.distances[:, 0]
    nn_index = nearest.indices[:, 0]
    variant_distances: dict[str, torch.Tensor] = {}
    for k in DENSITY_K_VALUES:
        name = f"density_normalized_1nn_k{k}"
        variant_distances[name] = (nn_distance / (
            state.density_profiles[k].rho[nn_index] + _EPSILON
        )).clamp_max(state.score_caps[name])
    mean_k = min(5, int(nearest.distances.shape[1]))
    quantile_k = min(10, int(nearest.distances.shape[1]))
    variant_distances["knn_mean_k5"] = nearest.distances[:, :mean_k].mean(dim=1)
    variant_distances["knn_quantile_k10_q0.5"] = _median_sorted(
        nearest.distances[:, :quantile_k], quantile_k,
    )
    variant_distances["shrinkage_mahalanobis"] = mahalanobis_distances(
        inputs.query_z,
        state.mahalanobis,
        config.query_chunk_size,
    )
    variant_distances["global_pca_residual"] = pca_residual_distances(
        inputs.query_z,
        state.pca,
        config.query_chunk_size,
    )
    for name, distances in variant_distances.items():
        results[name] = _compose_result(
            distances,
            inputs,
            config,
            state.calibrations[name],
        )
    return results


def evaluate_native_records(records: Sequence[NativeMapRecord]) -> dict[str, Any]:
    """Compute all preregistered pooled metrics for one object and variant."""

    # Keep heavyweight evaluation-only dependencies out of scorer fitting and
    # scoring workers that do not evaluate metrics themselves.
    from scipy import ndimage  # noqa: PLC0415
    from sklearn.metrics import (  # noqa: PLC0415
        average_precision_score,
        roc_auc_score,
        roc_curve,
    )

    from flow_tte_gap_decomposition import oracle_f1  # noqa: PLC0415

    if not records:
        raise ValueError("at least one native map record is required")
    for record in records:
        if record.score.shape != record.gt.shape:
            message = f"score/GT shape mismatch for {record.stem}"
            raise ValueError(message)
    labels = np.concatenate([np.asarray(row.gt, dtype=np.uint8).ravel() for row in records])
    scores = np.concatenate([np.asarray(row.score, dtype=np.float16).ravel() for row in records])
    if not bool(np.any(labels)) or not bool(np.any(labels == 0)):
        raise ValueError("pooled metrics require both anomalous and good pixels")
    oracle = oracle_f1(labels, scores, cast_float16=True)
    threshold = np.float16(oracle["threshold"])
    fpr, tpr, _ = roc_curve(labels, scores)
    admissible = tpr[fpr <= 1e-4]
    normal_scores = np.concatenate(
        [row.score.ravel() for row in records if row.split == "good"],
    ).astype(np.float16, copy=False)
    if normal_scores.size == 0:
        raise ValueError("normal-image pooled FPR requires split='good' records")
    good_pixel_scores = scores[labels == 0]
    good_p999 = float(np.quantile(good_pixel_scores.astype(np.float32), 0.999))
    component_hit = component_total = 0
    normal_image_fprs: list[float] = []
    bad_image_oracles: list[float] = []
    for row in records:
        prediction = np.asarray(row.score, dtype=np.float16) >= threshold
        if row.split == "good":
            normal_image_fprs.append(float(np.mean(prediction)))
        else:
            bad_image_oracles.append(
                float(oracle_f1(row.gt, row.score, cast_float16=True)["f1"]),
            )
        components, count = ndimage.label(row.gt, structure=_STRUCTURE_8)
        component_total += int(count)
        component_hit += sum(
            bool(np.any(prediction[components == component_id]))
            for component_id in range(1, count + 1)
        )
    return {
        "pooled_oracle_f1_float16": float(oracle["f1"]),
        "oracle_threshold_float16": float(threshold),
        "pooled_pixel_ap_float16": float(average_precision_score(labels, scores)),
        "pooled_pauroc_0.05_float16": float(roc_auc_score(labels, scores, max_fpr=0.05)),
        "tpr_at_fpr_1e-4": float(np.max(admissible)) if admissible.size else 0.0,
        "normal_image_pooled_fpr_at_oracle": float(np.mean(normal_scores >= threshold)),
        "normal_image_mean_fpr_at_oracle": float(np.mean(normal_image_fprs)),
        "bad_image_oracle_mean_float16": float(np.mean(bad_image_oracles)),
        "good_pixel_p99_9": good_p999,
        "good_pixel_p99_9_exceeds_oracle_threshold": bool(
            np.float16(good_p999) >= threshold,
        ),
        "good_pixel_pooled_exceedance_rate_at_oracle": float(
            np.mean(good_pixel_scores >= threshold),
        ),
        "gt_component_recall_at_oracle": (
            float(component_hit / component_total) if component_total else 1.0
        ),
        "gt_components_hit": component_hit,
        "gt_components_total": component_total,
        "n_images": len(records),
        "n_pixels": int(labels.size),
    }


_MEAN_METRIC_KEYS = (
    "pooled_oracle_f1_float16",
    "pooled_pixel_ap_float16",
    "pooled_pauroc_0.05_float16",
    "tpr_at_fpr_1e-4",
    "normal_image_pooled_fpr_at_oracle",
    "good_pixel_pooled_exceedance_rate_at_oracle",
    "gt_component_recall_at_oracle",
)


def aggregate_object_metrics(
    per_object: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute the object-balanced mean row for one variant."""

    if not per_object:
        raise ValueError("per_object metrics cannot be empty")
    result: dict[str, Any] = {"object_count": len(per_object)}
    for key in _MEAN_METRIC_KEYS:
        result[f"mean_{key}"] = float(np.mean([float(row[key]) for row in per_object.values()]))
    return result


def object_floor_flags(
    variant_objects: Mapping[str, Mapping[str, Any]],
    raw_objects: Mapping[str, Mapping[str, Any]],
    tolerance: float = 0.02,
) -> dict[str, dict[str, Any]]:
    """Flag per-object F1 or AP drops strictly greater than the tolerance."""

    flags: dict[str, dict[str, Any]] = {}
    for object_name, raw in raw_objects.items():
        row = variant_objects[object_name]
        f1_drop = float(raw["pooled_oracle_f1_float16"]) - float(
            row["pooled_oracle_f1_float16"],
        )
        ap_drop = float(raw["pooled_pixel_ap_float16"]) - float(row["pooled_pixel_ap_float16"])
        flags[object_name] = {
            "f1_drop": f1_drop,
            "ap_drop": ap_drop,
            "violation": bool(f1_drop > tolerance or ap_drop > tolerance),
        }
    return flags
