from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class ShiftProjection:
    basis: FloatArray
    sampled_count: int
    retained_count: int
    retained_energy_ratio: float


def fit_shift_projection(
    residual_batches: Sequence[npt.ArrayLike],
    *,
    rank: int,
    trim_fraction: float,
    max_samples: int,
    seed: int,
) -> ShiftProjection:
    """Fit an uncentered low-rank basis to robust support-to-query residuals."""
    if rank <= 0:
        raise ValueError("rank must be positive")
    if not 0.0 <= trim_fraction < 1.0:
        raise ValueError("trim_fraction must be in [0, 1)")
    if max_samples <= rank:
        raise ValueError("max_samples must exceed rank")
    matrices = [np.asarray(batch, dtype=np.float32) for batch in residual_batches]
    if not matrices or any(matrix.ndim != 2 for matrix in matrices):
        raise ValueError("residual_batches must contain 2D arrays")
    feature_dim = matrices[0].shape[1]
    if any(matrix.shape[1] != feature_dim for matrix in matrices):
        raise ValueError("residual feature dimensions must match")
    residuals = np.concatenate(matrices, axis=0)
    if residuals.shape[0] <= rank:
        raise ValueError("not enough residual rows for requested rank")
    if not np.all(np.isfinite(residuals)):
        raise ValueError("residuals must be finite")

    rng = np.random.default_rng(seed)
    if residuals.shape[0] > max_samples:
        indices = rng.choice(residuals.shape[0], size=max_samples, replace=False)
        residuals = residuals[np.sort(indices)]
    sampled_count = int(residuals.shape[0])
    norms = np.linalg.norm(residuals, axis=1)
    retained_count = max(rank + 1, int(round(sampled_count * (1.0 - trim_fraction))))
    retained_indices = np.argpartition(norms, retained_count - 1)[:retained_count]
    retained = residuals[retained_indices]

    covariance = retained.T @ retained / float(retained.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:rank]
    basis = np.asarray(eigenvectors[:, order], dtype=np.float32)
    selected_energy = float(np.maximum(eigenvalues[order], 0.0).sum())
    total_energy = float(np.maximum(eigenvalues, 0.0).sum())
    return ShiftProjection(
        basis=basis,
        sampled_count=sampled_count,
        retained_count=int(retained.shape[0]),
        retained_energy_ratio=selected_energy / max(total_energy, 1e-12),
    )


def remove_shift_component(
    residuals: npt.ArrayLike,
    basis: npt.ArrayLike,
    *,
    strength: float = 1.0,
) -> FloatArray:
    matrix = np.asarray(residuals, dtype=np.float32)
    directions = np.asarray(basis, dtype=np.float32)
    if matrix.ndim != 2 or directions.ndim != 2:
        raise ValueError("residuals and basis must be 2D")
    if matrix.shape[1] != directions.shape[0]:
        raise ValueError("residual and basis dimensions must match")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    return np.asarray(matrix - strength * (matrix @ directions) @ directions.T, dtype=np.float32)
