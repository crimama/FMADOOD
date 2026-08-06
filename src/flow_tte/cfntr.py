from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple, cast

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
UInt8Array = npt.NDArray[np.uint8]
IntArray = npt.NDArray[np.int64]

_NORMALIZE_EPS = 1e-12
_ORTHONORMAL_ATOL = 2e-5
_EXPECTED_QUADRANTS: IntArray = np.arange(4, dtype=np.int64)


PHYSICAL_AXES: Tuple[str, ...] = ("exposure", "gamma", "contrast", "white_balance")
PAIRWISE_COMPOUNDS: Tuple[str, ...] = (
    "exposure_white_balance",
    "gamma_contrast",
)
TRAIN_AXIS_AMPLITUDES = {
    "exposure": 0.25,
    "gamma": 0.10,
    "contrast": 0.10,
    "white_balance": 0.08,
}
VALIDATION_AMPLITUDE_SCALE = 2.0


@dataclass(frozen=True)
class RGBTransformSpec:
    """Signed log-coordinate transform amplitudes with one final RGB clip."""

    name: str
    exposure_ev: float = 0.0
    gamma_log_step: float = 0.0
    contrast_log_step: float = 0.0
    white_balance_log_step: float = 0.0

    @property
    def is_identity(self) -> bool:
        return all(
            value == 0.0
            for value in (
                self.exposure_ev,
                self.gamma_log_step,
                self.contrast_log_step,
                self.white_balance_log_step,
            )
        )


def _validated_sign(sign: int) -> int:
    if isinstance(sign, bool) or not isinstance(sign, (int, np.integer)) or sign not in (-1, 1):
        raise ValueError("sign must be exactly -1 or +1")
    return int(sign)


def _validated_amplitude_scale(amplitude_scale: float) -> float:
    scale = float(amplitude_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("amplitude_scale must be finite and positive")
    return scale


def make_axis_transform(
    axis: str,
    *,
    sign: int,
    amplitude_scale: float = 1.0,
) -> RGBTransformSpec:
    """Construct a frozen signed central-difference axis transform."""
    if axis not in PHYSICAL_AXES:
        message = f"axis must be one of {', '.join(PHYSICAL_AXES)}"
        raise ValueError(message)
    direction = _validated_sign(sign)
    scale = _validated_amplitude_scale(amplitude_scale)
    amplitude = direction * scale * TRAIN_AXIS_AMPLITUDES[axis]
    values = {
        "exposure_ev": 0.0,
        "gamma_log_step": 0.0,
        "contrast_log_step": 0.0,
        "white_balance_log_step": 0.0,
    }
    field_by_axis = {
        "exposure": "exposure_ev",
        "gamma": "gamma_log_step",
        "contrast": "contrast_log_step",
        "white_balance": "white_balance_log_step",
    }
    values[field_by_axis[axis]] = amplitude
    return RGBTransformSpec(
        name=f"{axis}_{'pos' if direction > 0 else 'neg'}_x{scale:g}",
        **values,
    )


def make_compound_transform(
    compound: str,
    *,
    signs: Tuple[int, int],
    amplitude_scale: float = VALIDATION_AMPLITUDE_SCALE,
) -> RGBTransformSpec:
    """Construct one of the two frozen pairwise held-out validation compounds."""
    if compound not in PAIRWISE_COMPOUNDS:
        message = f"compound must be one of {', '.join(PAIRWISE_COMPOUNDS)}"
        raise ValueError(message)
    if len(signs) != 2:
        raise ValueError("signs must contain exactly two signs")
    first_sign, second_sign = (_validated_sign(value) for value in signs)
    scale = _validated_amplitude_scale(amplitude_scale)
    first_axis, second_axis = {
        "exposure_white_balance": ("exposure", "white_balance"),
        "gamma_contrast": ("gamma", "contrast"),
    }[compound]
    first = make_axis_transform(first_axis, sign=first_sign, amplitude_scale=scale)
    second = make_axis_transform(second_axis, sign=second_sign, amplitude_scale=scale)
    return RGBTransformSpec(
        name=(
            f"{compound}_{'p' if first_sign > 0 else 'n'}"
            f"{'p' if second_sign > 0 else 'n'}_x{scale:g}"
        ),
        exposure_ev=first.exposure_ev + second.exposure_ev,
        gamma_log_step=first.gamma_log_step + second.gamma_log_step,
        contrast_log_step=first.contrast_log_step + second.contrast_log_step,
        white_balance_log_step=(first.white_balance_log_step + second.white_balance_log_step),
    )


def _finite_matrix(name: str, value: npt.ArrayLike) -> FloatArray:
    matrix = cast("FloatArray", np.asarray(value, dtype=np.float32))
    if matrix.ndim != 2:
        message = f"{name} must be a 2D row-by-feature matrix"
        raise ValueError(message)
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        message = f"{name} must be non-empty"
        raise ValueError(message)
    if not np.all(np.isfinite(matrix)):
        message = f"{name} must contain only finite values"
        raise ValueError(message)
    return matrix


def _rgb_float(image_rgb: npt.ArrayLike) -> FloatArray:
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must have H-by-W-by-3 RGB shape")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image_rgb must have positive spatial dimensions")
    if image.dtype == np.uint8:
        return np.asarray(
            np.asarray(image, dtype=np.float32) / np.float32(255.0),
            dtype=np.float32,
        )
    if not np.issubdtype(image.dtype, np.floating):
        raise TypeError("image_rgb must be uint8 or floating point")
    rgb = np.asarray(image, dtype=np.float32)
    if not np.all(np.isfinite(rgb)):
        raise ValueError("image_rgb must contain only finite values")
    if float(rgb.min()) < 0.0 or float(rgb.max()) > 1.0:
        raise ValueError("floating image_rgb values must lie in [0, 1]")
    return rgb


def _validated_transform_spec(spec: RGBTransformSpec) -> RGBTransformSpec:
    if not isinstance(spec, RGBTransformSpec):
        raise TypeError("spec must be an RGBTransformSpec")
    if not spec.name:
        raise ValueError("spec.name must be non-empty")
    amplitudes = (
        spec.exposure_ev,
        spec.gamma_log_step,
        spec.contrast_log_step,
        spec.white_balance_log_step,
    )
    if not all(np.isfinite(value) for value in amplitudes):
        raise ValueError("transform amplitudes must be finite")
    if spec.is_identity:
        raise ValueError("identity is not a valid intervention")
    return spec


def apply_rgb_transform(image_rgb: npt.ArrayLike, spec: RGBTransformSpec) -> UInt8Array:
    """Apply a transform spec in frozen order and clip exactly once at the end.

    Operations are exposure, gamma, contrast about the original image's RGB
    channel mean, and reciprocal red/blue white balance.
    """
    rgb = _rgb_float(image_rgb)
    resolved = _validated_transform_spec(spec)
    transformed = rgb.astype(np.float64)
    transformed *= 2.0**resolved.exposure_ev
    transformed = np.power(transformed, np.exp(resolved.gamma_log_step))
    rgb_mean = rgb.astype(np.float64).mean(axis=(0, 1), keepdims=True)
    contrast_multiplier = np.exp(resolved.contrast_log_step)
    transformed = rgb_mean + contrast_multiplier * (transformed - rgb_mean)
    white_balance_gains = np.array(
        [
            np.exp(resolved.white_balance_log_step),
            1.0,
            np.exp(-resolved.white_balance_log_step),
        ],
        dtype=np.float64,
    ).reshape(1, 1, 3)
    transformed *= white_balance_gains
    clipped = np.clip(transformed, 0.0, 1.0)
    return np.asarray(np.rint(clipped * 255.0), dtype=np.uint8)


def apply_rgb_intervention(image_rgb: npt.ArrayLike, spec: RGBTransformSpec) -> UInt8Array:
    """Backward-compatible naming alias for :func:`apply_rgb_transform`."""
    return apply_rgb_transform(image_rgb, spec)


def apply_training_axis_interventions(image_rgb: npt.ArrayLike) -> Dict[str, UInt8Array]:
    """Apply the eight positive/negative transforms used for physical basis fitting."""
    specs = (make_axis_transform(axis, sign=sign) for axis in PHYSICAL_AXES for sign in (-1, 1))
    return {spec.name: apply_rgb_transform(image_rgb, spec) for spec in specs}


def normalize_rows(features: npt.ArrayLike) -> FloatArray:
    """L2-normalize feature rows, rejecting rows whose direction is undefined."""
    matrix = _finite_matrix("features", features)
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1, keepdims=True)
    if np.any(norms <= _NORMALIZE_EPS):
        raise ValueError("features contains a row with undefined L2 direction")
    normalized = matrix.astype(np.float64) / norms
    return np.asarray(normalized, dtype=np.float32)


def image_level_delta_observation(
    original_features: npt.ArrayLike,
    transformed_features: npt.ArrayLike,
) -> FloatArray:
    """Return the spatial mean of normalized transformed-minus-original features."""
    original = normalize_rows(original_features)
    transformed = normalize_rows(transformed_features)
    if original.shape != transformed.shape:
        raise ValueError("original_features and transformed_features must have identical shape")
    observation = (transformed.astype(np.float64) - original.astype(np.float64)).mean(axis=0)
    if not np.all(np.isfinite(observation)):
        raise RuntimeError("image-level delta observation is non-finite")
    return np.asarray(observation, dtype=np.float32)


def central_difference_observation(
    negative_features: npt.ArrayLike,
    positive_features: npt.ArrayLike,
) -> FloatArray:
    """Spatially average normalized positive-minus-negative axis features."""
    negative = normalize_rows(negative_features)
    positive = normalize_rows(positive_features)
    if negative.shape != positive.shape:
        raise ValueError("negative_features and positive_features must have identical shape")
    observation = (positive.astype(np.float64) - negative.astype(np.float64)).mean(axis=0)
    if not np.all(np.isfinite(observation)):
        raise RuntimeError("central-difference observation is non-finite")
    return np.asarray(observation, dtype=np.float32)


def image_level_mean_observation(features: npt.ArrayLike) -> FloatArray:
    """Return one independent image-level observation from normalized patch features."""
    normalized = normalize_rows(features)
    return np.asarray(normalized.astype(np.float64).mean(axis=0), dtype=np.float32)


def _canonicalize_basis_signs(basis: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    canonical = basis.copy()
    for column in range(canonical.shape[1]):
        pivot = int(np.argmax(np.abs(canonical[:, column])))
        if canonical[pivot, column] < 0.0:
            canonical[:, column] *= -1.0
    return canonical


def fit_image_level_basis(
    observations: npt.ArrayLike,
    *,
    rank: int,
    center: bool,
) -> FloatArray:
    """Fit a deterministic rank-r orthonormal basis from image-level observations.

    Physical transformed-minus-original observations use ``center=False``.
    The matched original-support PCA arm uses ``center=True``.
    """
    matrix = _finite_matrix("observations", observations).astype(np.float64)
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or rank <= 0:
        raise ValueError("rank must be a positive integer")
    if not isinstance(center, (bool, np.bool_)):
        raise TypeError("center must be boolean")
    max_rank = min(matrix.shape)
    if rank > max_rank:
        raise ValueError("rank cannot exceed the observation matrix rank bound")
    working = matrix - matrix.mean(axis=0, keepdims=True) if center else matrix
    if float(np.linalg.norm(working)) <= _NORMALIZE_EPS:
        raise ValueError("observations have no finite variation for basis fitting")
    _, singular_values, right_vectors = np.linalg.svd(working, full_matrices=False)
    tolerance = np.finfo(np.float64).eps * max(working.shape) * float(singular_values[0])
    if int(np.count_nonzero(singular_values > tolerance)) < rank:
        raise ValueError("observations do not support the requested numerical rank")
    basis = _canonicalize_basis_signs(right_vectors[:rank].T)
    return np.asarray(basis, dtype=np.float32)


def deterministic_random_basis(*, feature_dim: int, rank: int, seed: int) -> FloatArray:
    """Draw a deterministic Haar-distributed rank-r basis using Gaussian QR."""
    values = (feature_dim, rank, seed)
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in values):
        raise TypeError("feature_dim, rank, and seed must be integers")
    if feature_dim <= 0 or rank <= 0 or rank > feature_dim:
        raise ValueError("require feature_dim >= rank > 0")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    gaussian = np.random.default_rng(int(seed)).standard_normal((feature_dim, rank))
    basis, _ = np.linalg.qr(gaussian, mode="reduced")
    return np.asarray(_canonicalize_basis_signs(basis), dtype=np.float32)


def _validated_basis(
    basis: npt.ArrayLike,
    *,
    feature_dim: Optional[int] = None,
) -> FloatArray:
    directions = _finite_matrix("basis", basis)
    if feature_dim is not None and directions.shape[0] != feature_dim:
        raise ValueError("basis and residual feature dimensions must match")
    gram = directions.astype(np.float64).T @ directions.astype(np.float64)
    identity = np.eye(directions.shape[1], dtype=np.float64)
    if not np.allclose(gram, identity, rtol=0.0, atol=_ORTHONORMAL_ATOL):
        raise ValueError("basis columns must be orthonormal")
    return directions


def minimum_subspace_cosine(first: npt.ArrayLike, second: npt.ArrayLike) -> float:
    """Return the minimum principal-angle cosine between equal-rank subspaces."""
    left = _validated_basis(first)
    right = _validated_basis(second)
    if left.shape != right.shape:
        raise ValueError("subspace bases must have the same feature dimension and rank")
    singular_values = np.linalg.svd(
        left.astype(np.float64).T @ right.astype(np.float64),
        compute_uv=False,
    )
    value = float(np.clip(singular_values.min(), 0.0, 1.0))
    if not np.isfinite(value):
        raise RuntimeError("subspace cosine is non-finite")
    return value


def subspace_overlap_stability(first: npt.ArrayLike, second: npt.ArrayLike) -> float:
    """Return preregistered ``||U.T @ V||_F^2 / rank`` subspace stability."""
    left = _validated_basis(first)
    right = _validated_basis(second)
    if left.shape != right.shape:
        raise ValueError("subspace bases must have the same feature dimension and rank")
    overlap = left.astype(np.float64).T @ right.astype(np.float64)
    value = float(np.square(overlap).sum() / left.shape[1])
    return float(np.clip(value, 0.0, 1.0))


@dataclass(frozen=True)
class BasisStability:
    per_fold: Tuple[float, ...]
    median: float


def fold_basis_stability(
    full_basis: npt.ArrayLike,
    fold_bases: Sequence[npt.ArrayLike],
) -> BasisStability:
    """Summarize preregistered squared canonical-correlation stability."""
    if len(fold_bases) == 0:
        raise ValueError("fold_bases must be non-empty")
    values = tuple(subspace_overlap_stability(full_basis, fold) for fold in fold_bases)
    return BasisStability(per_fold=values, median=float(np.median(values)))


def support_image_fold_ids(image_count: int, *, fold_count: int = 4) -> IntArray:
    """Assign ordered support-image indices modulo the frozen fold count."""
    if isinstance(image_count, bool) or not isinstance(image_count, (int, np.integer)):
        raise TypeError("image_count must be an integer")
    if isinstance(fold_count, bool) or not isinstance(fold_count, (int, np.integer)):
        raise TypeError("fold_count must be an integer")
    if image_count < fold_count or fold_count <= 1:
        raise ValueError("require image_count >= fold_count > 1")
    return np.arange(image_count, dtype=np.int64) % int(fold_count)


def quadrant_fold_ids(grid_shape: Tuple[int, int]) -> IntArray:
    """Return row-major fold IDs for four contiguous 2x2 grid quadrants."""
    if len(grid_shape) != 2:
        raise ValueError("grid_shape must contain height and width")
    height, width = grid_shape
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in grid_shape
    ):
        raise TypeError("grid dimensions must be integers")
    if height < 2 or width < 2:
        raise ValueError("grid dimensions must both be at least two")
    row_side = (np.arange(height) >= height // 2).astype(np.int64)
    column_side = (np.arange(width) >= width // 2).astype(np.int64)
    folds = 2 * row_side[:, None] + column_side[None, :]
    return np.asarray(folds.reshape(-1), dtype=np.int64)


def _validated_fold_ids(fold_ids: npt.ArrayLike, *, row_count: int) -> IntArray:
    raw = np.asarray(fold_ids)
    if raw.ndim != 1 or raw.shape[0] != row_count:
        raise ValueError("fold_ids must have one value per residual row")
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("fold_ids must contain integers")
    folds = np.asarray(raw, dtype=np.int64)
    if not np.array_equal(np.unique(folds), _EXPECTED_QUADRANTS):
        raise ValueError("fold_ids must contain each quadrant ID 0, 1, 2, and 3")
    return folds


@dataclass(frozen=True)
class CrossFitScores:
    scores: FloatArray
    coefficients: FloatArray


def score_raw_residuals(residuals: npt.ArrayLike) -> FloatArray:
    """Return the preregistered raw-NN score ``0.5 * ||r_p||^2``."""
    matrix = _finite_matrix("residuals", residuals).astype(np.float64)
    scores = 0.5 * np.einsum("ij,ij->i", matrix, matrix)
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("raw residual scoring produced non-finite values")
    return np.asarray(scores, dtype=np.float32)


def score_crossfit_residuals(
    residuals: npt.ArrayLike,
    basis: npt.ArrayLike,
    fold_ids: npt.ArrayLike,
) -> CrossFitScores:
    """Score residuals with one held-out-quadrant shared coordinate median."""
    matrix = _finite_matrix("residuals", residuals).astype(np.float64)
    directions = _validated_basis(basis, feature_dim=matrix.shape[1]).astype(np.float64)
    folds = _validated_fold_ids(fold_ids, row_count=matrix.shape[0])
    projected = matrix @ directions
    scores = np.empty(matrix.shape[0], dtype=np.float64)
    coefficients = np.empty((4, directions.shape[1]), dtype=np.float64)
    for fold in range(4):
        held_out = folds == fold
        train = ~held_out
        coefficient = np.median(projected[train], axis=0)
        coefficients[fold] = coefficient
        corrected = matrix[held_out] - coefficient @ directions.T
        scores[held_out] = 0.5 * np.einsum("ij,ij->i", corrected, corrected)
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(coefficients)):
        raise RuntimeError("cross-fitted scoring produced non-finite values")
    return CrossFitScores(
        scores=np.asarray(scores, dtype=np.float32),
        coefficients=np.asarray(coefficients, dtype=np.float32),
    )


def score_patchwise_residuals(
    residuals: npt.ArrayLike,
    basis: npt.ArrayLike,
) -> FloatArray:
    """Return the patch-wise nuisance projection absorption hard-null score."""
    matrix = _finite_matrix("residuals", residuals).astype(np.float64)
    directions = _validated_basis(basis, feature_dim=matrix.shape[1]).astype(np.float64)
    corrected = matrix - (matrix @ directions) @ directions.T
    scores = 0.5 * np.einsum("ij,ij->i", corrected, corrected)
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("patch-wise scoring produced non-finite values")
    return np.asarray(scores, dtype=np.float32)


def subspace_energy_coverage(delta: npt.ArrayLike, basis: npt.ArrayLike) -> float:
    """Return the fraction of a real-shift mean-delta energy captured by a basis."""
    vector = np.asarray(delta, dtype=np.float64)
    if vector.ndim != 1 or vector.shape[0] == 0:
        raise ValueError("delta must be a non-empty feature vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError("delta must contain only finite values")
    directions = _validated_basis(basis, feature_dim=vector.shape[0]).astype(np.float64)
    total = float(vector @ vector)
    if total <= _NORMALIZE_EPS:
        raise ValueError("delta must have positive energy")
    projected = vector @ directions
    captured = float(projected @ projected)
    return float(np.clip(captured / total, 0.0, 1.0))


@dataclass(frozen=True)
class MapCoverage:
    total_maps: int
    finite_maps: int
    nonconstant_maps: int
    valid_maps: int
    total_values: int
    finite_values: int

    @property
    def complete(self) -> bool:
        return self.valid_maps == self.total_maps

    @property
    def finite_value_fraction(self) -> float:
        return self.finite_values / self.total_values


def summarize_map_coverage(
    score_maps: Iterable[npt.ArrayLike],
    *,
    nonconstant_atol: float = 0.0,
) -> MapCoverage:
    """Count finite, nonconstant score maps without hiding invalid values."""
    if not np.isfinite(nonconstant_atol) or nonconstant_atol < 0.0:
        raise ValueError("nonconstant_atol must be finite and non-negative")
    total_maps = 0
    finite_maps = 0
    nonconstant_maps = 0
    valid_maps = 0
    total_values = 0
    finite_values = 0
    for value in score_maps:
        score_map = np.asarray(value)
        if score_map.size < 2:
            raise ValueError("every score map must contain at least two values")
        if not np.issubdtype(score_map.dtype, np.number):
            raise TypeError("score maps must be numeric")
        total_maps += 1
        total_values += int(score_map.size)
        finite_mask = np.isfinite(score_map)
        finite_count = int(finite_mask.sum())
        finite_values += finite_count
        is_finite = finite_count == score_map.size
        is_nonconstant = False
        if is_finite:
            finite_maps += 1
            spread = float(np.max(score_map) - np.min(score_map))
            is_nonconstant = spread > nonconstant_atol
            if is_nonconstant:
                nonconstant_maps += 1
                valid_maps += 1
    if total_maps == 0:
        raise ValueError("score_maps must be non-empty")
    return MapCoverage(
        total_maps=total_maps,
        finite_maps=finite_maps,
        nonconstant_maps=nonconstant_maps,
        valid_maps=valid_maps,
        total_values=total_values,
        finite_values=finite_values,
    )
