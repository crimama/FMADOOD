"""PSMA-MR: paired support-only adaptation with multi-resolution memory.

The method deliberately owns all of its adaptation state.  The common DINO
backbone is only used as a frozen feature provider; in particular, the
support-derived adapter and memory banks below must not leak query images or
query labels into fitting.

``PSMA_MR`` is kept in a standalone module because the native backbone has a
small, stable interface (``prepare_image`` and ``extract_features``), while
the method needs a few more pieces of bookkeeping than the simple kNN
baseline.  Resolution selection is always explicit and non-mutating through
``prepare_image(img, resolution=...)``.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:  # OpenCV is a project dependency, but keep import errors local-friendly.
    import cv2
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal envs
    cv2 = None

from PIL import Image

from fmad.methods.base import BaseMethod
from fmad.protocol import MethodContext
from fmad.registry import METHOD_REGISTRY


def _json_safe(value: Any) -> Any:
    """Convert configuration/provenance values to strict JSON-compatible data."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _positive_int(name: str, value: Any, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < (0 if allow_zero else 1):
        bound = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {bound}")
    return value


def _finite_float(name: str, value: Any, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        if minimum is None:
            raise ValueError(f"{name} must be finite")
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return value


def _parse_resolutions(params: Mapping[str, Any]) -> tuple[int, ...]:
    value = params.get("resolutions", (448, 672))
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        value = [int(value)]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("resolutions must be a non-empty sequence of positive integers")
    resolutions = tuple(_positive_int("resolution", item) for item in value)
    if len(set(resolutions)) != len(resolutions):
        raise ValueError("resolutions must not contain duplicates")
    return resolutions


class _Rank4Residual(nn.Module):
    """Low-rank residual with an optional relative-norm trust region.

    ``U`` is zero-initialized and ``V`` is a deterministic non-zero matrix, so
    the initial mapping is exactly the normalized identity while gradients can
    immediately flow into the residual branch.  The default rank is four;
    accepting another rank is useful for controlled ablations without
    changing the default PSMA-MR method.
    """

    def __init__(
        self,
        dimension: int,
        rank: int = 4,
        seed: int = 0,
        enabled: bool = True,
        residual_epsilon: float | None = None,
        eps: float | None = None,
    ):
        super().__init__()
        self.dimension = int(dimension)
        self.rank = int(rank)
        self.enabled = bool(enabled)
        if residual_epsilon is None:
            residual_epsilon = eps
        self.residual_epsilon = None if residual_epsilon is None else float(residual_epsilon)
        self.eps = self.residual_epsilon
        self.U = nn.Parameter(torch.zeros(self.dimension, self.rank))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) % (2**63 - 1))
        initial_v = torch.randn(self.rank, self.dimension, generator=generator) * 0.02
        self.V = nn.Parameter(initial_v)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = torch.as_tensor(features, dtype=torch.float32)
        if features.shape[-1] != self.dimension:
            raise ValueError(
                f"adapter expected feature dimension {self.dimension}, "
                f"got {features.shape[-1]}"
            )
        normalized = F.layer_norm(features, (self.dimension,), eps=1e-5)
        if self.enabled:
            latent = F.linear(normalized, self.V)
            residual = torch.tanh(latent) @ self.U.T
            if self.residual_epsilon is not None:
                # Clip ``h`` to eps*||z|| without dividing by zero.  At U=0
                # the h=0 branch uses scale one, preserving a non-zero U
                # gradient whenever tanh(V LN(z)) is non-zero.  For z=0 the
                # permitted residual is exactly zero.
                residual_norm = torch.linalg.vector_norm(residual, dim=-1, keepdim=True)
                feature_norm = torch.linalg.vector_norm(features, dim=-1, keepdim=True)
                safe_norm = residual_norm.clamp_min(torch.finfo(residual.dtype).tiny)
                permitted = feature_norm * float(self.residual_epsilon)
                scale = permitted / safe_norm
                scale = torch.where(residual_norm > 0, scale.clamp_max(1.0), torch.ones_like(scale))
                residual = residual * scale
            features = features + residual
        return F.normalize(features, dim=-1, eps=1e-8)


@dataclass
class _ResolutionViews:
    clean: torch.Tensor
    nuisance: torch.Tensor
    # Each anomaly keeps an identity feature view and the *same* photometric
    # transform applied to that view.  The mask is indexed with the exact
    # shared patch subset used by clean/nuisance, so margin terms remain local
    # and paired rather than comparing unrelated random patches.
    anomalies: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str], ...]
    grid_size: tuple[int, int]


@dataclass
class _V2ResolutionViews:
    """Support-only caches for one PSMA-MR v2 resolution.

    The branch tensors are already spatially selected, but ``coords`` keeps
    the descendant id and original patch row/column for every retained token.
    This is deliberately carried alongside features so a later bounded cache
    cannot accidentally turn a random row sample into a purported spatial
    coreset.  Synthetic views are final-layer-only by contract.
    """

    clean: tuple[torch.Tensor, ...]
    nuisance: tuple[torch.Tensor, ...]
    foreground: tuple[torch.Tensor, ...]
    coords: tuple[torch.Tensor, ...]
    anomalies: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str], ...]
    grid_size: tuple[int, int]


def _spatial_coordinates(grid_size: tuple[int, int], descendant_id: int) -> torch.Tensor:
    """Return ``[descendant, row, column]`` coordinates in row-major order."""

    rows, columns = (int(grid_size[0]), int(grid_size[1]))
    if rows <= 0 or columns <= 0:
        raise ValueError("feature grid dimensions must be positive")
    row, column = torch.meshgrid(
        torch.arange(rows, dtype=torch.long),
        torch.arange(columns, dtype=torch.long),
        indexing="ij",
    )
    descendant = torch.full_like(row, int(descendant_id))
    return torch.stack((descendant.reshape(-1), row.reshape(-1), column.reshape(-1)), dim=1)


def _spatial_stratified_select(
    coordinates: torch.Tensor,
    count: int,
    *,
    seed: int = 0,
    eligible: torch.Tensor | None = None,
    required_masks: Sequence[torch.Tensor] = (),
) -> tuple[torch.Tensor, int]:
    """Select deterministic spatially stratified rows and preserve positives.

    Coordinates are expected to have descendant id, row, and column columns.
    Selection takes one or more rows from spatial bins in round-robin order,
    rather than sampling arbitrary rows.  ``required_masks`` are aligned with
    the coordinates and reserve one visible token per synthetic family when
    the budget permits it.  The second return value counts families without a
    visible eligible token (or families that could not fit in the budget).
    """

    del seed  # The coordinate order is the deterministic tie-breaker.
    coordinates = torch.as_tensor(coordinates, dtype=torch.long).reshape(-1, 3)
    total = int(coordinates.shape[0])
    count = max(1, int(count))
    if total == 0:
        return torch.empty(0, dtype=torch.long), len(required_masks)
    if eligible is None:
        eligible_tensor = torch.ones(total, dtype=torch.bool)
    else:
        eligible_tensor = torch.as_tensor(eligible, dtype=torch.bool).reshape(-1)
        if eligible_tensor.numel() != total:
            raise ValueError("eligible coordinate mask has a different token count")
    eligible_indices = torch.where(eligible_tensor)[0]
    if eligible_indices.numel() == 0:
        # This mirrors the existing masking contract: no-mask contexts retain
        # all rows, while an explicit empty foreground mask fails earlier.
        eligible_indices = torch.arange(total, dtype=torch.long)
        eligible_tensor = torch.ones(total, dtype=torch.bool)
    budget = min(count, int(eligible_indices.numel()))

    def _coordinate_key(index: int) -> tuple[int, int, int, int]:
        row = coordinates[index]
        return int(row[0]), int(row[1]), int(row[2]), int(index)

    anchors: list[int] = []
    skipped = 0
    for mask in required_masks:
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool).reshape(-1)
        if mask_tensor.numel() != total:
            raise ValueError("required spatial mask has a different token count")
        visible = torch.where(mask_tensor & eligible_tensor)[0].tolist()
        if not visible:
            skipped += 1
            continue
        anchors.append(min(visible, key=_coordinate_key))
    # Keep one anchor per family where possible; duplicate anchors are only
    # counted once while a skipped family remains diagnosable.
    unique_anchors = sorted(set(anchors), key=_coordinate_key)
    if len(unique_anchors) > budget:
        skipped += len(unique_anchors) - budget
        unique_anchors = unique_anchors[:budget]

    selected = list(unique_anchors)
    remaining = [
        int(index)
        for index in eligible_indices.tolist()
        if int(index) not in set(unique_anchors)
    ]
    if len(selected) < budget and remaining:
        # Allocate bins independently per descendant.  A roughly square grid
        # gives broad coverage for both 1-D test fixtures and DINO grids.
        descendants = sorted({int(coordinates[index, 0]) for index in remaining})
        target_per_descendant = max(1, int(math.ceil(budget / max(1, len(descendants)))))
        bin_side = max(1, int(math.ceil(math.sqrt(target_per_descendant))))
        bins: dict[tuple[int, int, int], list[int]] = {}
        for index in remaining:
            descendant, row, column = (int(value) for value in coordinates[index])
            desc_indices = [
                candidate
                for candidate in remaining
                if int(coordinates[candidate, 0]) == descendant
            ]
            max_row = max(1, 1 + max(int(coordinates[candidate, 1]) for candidate in desc_indices))
            max_column = max(
                1, 1 + max(int(coordinates[candidate, 2]) for candidate in desc_indices)
            )
            row_bin = min(bin_side - 1, row * bin_side // max_row)
            column_bin = min(bin_side - 1, column * bin_side // max_column)
            bins.setdefault((descendant, row_bin, column_bin), []).append(index)
        ordered_bins = sorted(bins)
        for values in bins.values():
            values.sort(key=_coordinate_key)
        cursors = {bucket: 0 for bucket in ordered_bins}
        while len(selected) < budget and ordered_bins:
            made_progress = False
            for bucket in ordered_bins:
                values = bins[bucket]
                cursor = cursors[bucket]
                while cursor < len(values) and values[cursor] in selected:
                    cursor += 1
                cursors[bucket] = cursor
                if cursor >= len(values):
                    continue
                selected.append(values[cursor])
                cursors[bucket] = cursor + 1
                made_progress = True
                if len(selected) >= budget:
                    break
            if not made_progress:
                break
        if len(selected) < budget:
            for index in sorted(remaining, key=_coordinate_key):
                if index not in selected:
                    selected.append(index)
                    if len(selected) >= budget:
                        break
    return torch.tensor(sorted(selected[:budget]), dtype=torch.long), skipped


def _spatial_stratified_indices(
    coordinates: torch.Tensor,
    count: int,
    seed: int = 0,
    eligible: torch.Tensor | None = None,
    required_masks: Sequence[torch.Tensor] = (),
) -> torch.Tensor:
    """Public helper returning only deterministic spatially selected indices."""

    indices, _ = _spatial_stratified_select(
        coordinates,
        count,
        seed=seed,
        eligible=eligible,
        required_masks=required_masks,
    )
    return indices


def _mean_top1p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    count = max(1, int(values.size * 0.01))
    return float(np.sort(values)[-count:].mean())


def _read_rgb(path_or_image: str | Path | np.ndarray) -> np.ndarray:
    """Read a support/query image as a contiguous uint8 RGB array."""

    if isinstance(path_or_image, np.ndarray):
        image = np.asarray(path_or_image)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
            raise ValueError("image arrays must have shape HxWx{1,3,4}")
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]
        if np.issubdtype(image.dtype, np.floating):
            scale = 255.0 if float(np.nanmax(image)) <= 1.0 else 1.0
            image = np.clip(image * scale, 0.0, 255.0)
        return np.ascontiguousarray(image.astype(np.uint8))

    path = str(path_or_image)
    image = None
    if cv2 is not None:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image is None:
        try:
            with Image.open(path) as opened:
                image = np.asarray(opened.convert("RGB"))
        except Exception as error:  # pragma: no cover - error path is diagnostic
            raise FileNotFoundError(f"could not read image: {path}") from error
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"image is not RGB: {path}")
    return np.ascontiguousarray(image.astype(np.uint8))


def _sample_photometric_transform(
    rng: np.random.Generator, strength: float
) -> tuple[float, float, tuple[float, float, float]]:
    """Sample one deterministic nuisance transform shared by every pair.

    The parameters are sampled once per source image (not once per anomaly),
    then reused for clean ``x`` and every synthetic ``A(x)``.  Keeping the
    transform explicit makes accidental independent nuisance draws difficult
    and lets metadata record that identity and transformed pairs were used.
    """

    contrast = 1.0 + float(rng.uniform(-0.18, 0.18) * strength)
    brightness = float(rng.uniform(-24.0, 24.0) * strength)
    gains = tuple(
        float(value)
        for value in (1.0 + rng.uniform(-0.08, 0.08, size=3).astype(np.float32) * strength)
    )
    return contrast, brightness, gains


def _apply_photometric(
    image: np.ndarray,
    transform: tuple[float, float, tuple[float, float, float]],
) -> np.ndarray:
    """Apply one sampled transform without modifying the source array."""

    contrast, brightness, gains = transform
    source = image.astype(np.float32)
    nuisance = source * float(contrast) * np.asarray(gains, dtype=np.float32)[None, None, :]
    nuisance = nuisance + float(brightness)
    return np.ascontiguousarray(np.clip(nuisance, 0.0, 255.0).astype(np.uint8))


def _photometric_pair(
    image: np.ndarray, rng: np.random.Generator, strength: float
) -> tuple[np.ndarray, tuple[float, float, tuple[float, float, float]]]:
    """Create a nuisance view and return the exact transform used."""

    transform = _sample_photometric_transform(rng, strength)
    return _apply_photometric(image, transform), transform


def _augment_image(image: np.ndarray) -> list[np.ndarray]:
    """Match SuperAD's deterministic eight support rotations."""

    if cv2 is None:  # pragma: no cover - OpenCV is a project dependency
        raise RuntimeError("context.rotation=True requires OpenCV for support augmentation")
    center = tuple(np.array(image.shape[1::-1]) / 2)
    return [
        cv2.warpAffine(
            image,
            cv2.getRotationMatrix2D(center, angle, 1.0),
            image.shape[1::-1],
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_DEFAULT,
        )
        for angle in (0, 45, 90, 135, 180, 225, 270, 315)
    ]


def _rectangle_anomaly(
    image: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, str]:
    """Replace a local rectangle with a deterministic cut/paste texture."""

    height, width = image.shape[:2]
    patch_h = int(round(height * float(rng.uniform(0.12, 0.32))))
    patch_w = int(round(width * float(rng.uniform(0.12, 0.32))))
    patch_h = max(1, min(height, patch_h))
    patch_w = max(1, min(width, patch_w))
    y0 = int(rng.integers(0, max(1, height - patch_h + 1)))
    x0 = int(rng.integers(0, max(1, width - patch_w + 1)))
    y1, x1 = min(height, y0 + patch_h), min(width, x0 + patch_w)
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True

    # A rolled, channel-mixed patch retains realistic local structure while
    # guaranteeing a change even for flat fake-test images.
    rolled = np.roll(image, shift=(max(1, height // 7), max(1, width // 9)), axis=(0, 1))
    patch = rolled[y0:y1, x0:x1].astype(np.float32)
    original = image[y0:y1, x0:x1].astype(np.float32)
    anomalous = np.clip(0.25 * original + 0.75 * (255.0 - patch), 0.0, 255.0)
    output = image.copy()
    output[y0:y1, x0:x1] = anomalous.astype(np.uint8)
    return np.ascontiguousarray(output), mask, "rectangle_cutpaste"


def _scratch_anomaly(
    image: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, str]:
    """Draw a local line/scratch with deterministic high-frequency noise."""

    height, width = image.shape[:2]
    output = image.copy()
    mask = np.zeros((height, width), dtype=np.uint8)
    y0 = int(rng.integers(0, max(1, height)))
    x0 = int(rng.integers(0, max(1, width)))
    y1 = int(rng.integers(0, max(1, height)))
    x1 = int(rng.integers(0, max(1, width)))
    thickness = max(1, int(round(min(height, width) * 0.025)))
    if cv2 is not None:
        cv2.line(mask, (x0, y0), (x1, y1), 1, thickness=thickness)
    else:  # pragma: no cover - OpenCV is present in the project environment
        yy = np.linspace(y0, y1, num=max(height, width) + 1).round().astype(int)
        xx = np.linspace(x0, x1, num=max(height, width) + 1).round().astype(int)
        mask[np.clip(yy, 0, height - 1), np.clip(xx, 0, width - 1)] = 1
    if not mask.any():
        mask[min(y0, height - 1), min(x0, width - 1)] = 1
    noise = rng.normal(0.0, 55.0, size=output.shape).astype(np.float32)
    changed = np.clip(output.astype(np.float32) + noise + 80.0, 0.0, 255.0)
    output[mask.astype(bool)] = changed[mask.astype(bool)].astype(np.uint8)
    return np.ascontiguousarray(output), mask.astype(bool), "scratch_noise"


def _synthetic_anomalies(
    image: np.ndarray, rng: np.random.Generator, names: Sequence[str]
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Generate at least two local anomaly families and their pixel masks."""

    factories = {
        "rectangle_cutpaste": _rectangle_anomaly,
        "scratch_noise": _scratch_anomaly,
    }
    selected = list(names)
    for default in factories:
        if default not in selected:
            selected.append(default)
    # The contract asks for >=2 local synthetic families.  If a caller gives a
    # shorter list, deterministically append the canonical families.
    while len(selected) < 2:
        selected.append(("scratch_noise", "rectangle_cutpaste")[len(selected) % 2])

    result: list[tuple[np.ndarray, np.ndarray, str]] = []
    for name in selected:
        factory = factories.get(str(name))
        if factory is None:
            raise ValueError(
                "synthetic_anomaly_types entries must be rectangle_cutpaste or scratch_noise"
            )
        result.append(factory(image, rng))
    return result


def _infer_grid(grid_size: Any, patch_count: int) -> tuple[int, int]:
    if grid_size is not None:
        try:
            values = tuple(int(v) for v in grid_size)
        except (TypeError, ValueError):
            values = ()
        if len(values) == 2 and values[0] > 0 and values[1] > 0:
            if values[0] * values[1] == patch_count:
                return values
    side = int(math.sqrt(max(1, patch_count)))
    if side * side == patch_count:
        return side, side
    return 1, max(1, patch_count)


def _feature_matrix(features: Any, grid_size: Any) -> tuple[torch.Tensor, tuple[int, int]]:
    """Normalize common DINO/fake-backbone layouts to ``[patches, dim]``."""

    # ``extract_features`` is called under inference_mode.  Clone here after
    # leaving that context so adapter training receives ordinary tensors that
    # autograd can safely consume as constants.
    tensor = torch.as_tensor(features, dtype=torch.float32).detach().cpu().clone()
    if tensor.ndim == 0:
        raise ValueError("backbone feature output must have at least one dimension")
    while tensor.ndim > 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError("backbone feature output must contain [patches, dimension]")

    requested = None
    if grid_size is not None:
        try:
            values = tuple(int(v) for v in grid_size)
            if len(values) == 2 and values[0] > 0 and values[1] > 0:
                requested = values
        except (TypeError, ValueError):
            requested = None
    if requested is not None and tensor.shape[0] != requested[0] * requested[1]:
        # A few fake backbones use [dimension, patches].  Detect that layout
        # only when the declared grid disambiguates it.
        if tensor.shape[1] == requested[0] * requested[1]:
            tensor = tensor.T.contiguous()
    grid = _infer_grid(requested, int(tensor.shape[0]))
    return tensor.contiguous(), grid


def _token_mask(mask: np.ndarray, grid_size: tuple[int, int], patch_count: int) -> torch.Tensor:
    """Project a pixel mask to the feature grid with nearest-neighbour sampling."""

    height, width = grid_size
    if cv2 is not None:
        resized = cv2.resize(
            mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        )
    else:  # pragma: no cover - OpenCV is present in the project environment
        y = np.minimum(
            (np.arange(height) * mask.shape[0] / max(1, height)).astype(int), mask.shape[0] - 1
        )
        x = np.minimum(
            (np.arange(width) * mask.shape[1] / max(1, width)).astype(int), mask.shape[1] - 1
        )
        resized = mask[np.ix_(y, x)].astype(np.uint8)
    flat = torch.from_numpy(resized.astype(bool).reshape(-1))
    if flat.numel() != patch_count:
        raise ValueError("projected anomaly mask has a different patch count than features")
    return flat


def _sample_rows(features: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    indices = _sample_indices(features.shape[0], count, seed)
    return features[indices]


def _sample_indices(
    total: int,
    count: int,
    seed: int,
    required: torch.Tensor | None = None,
    eligible: torch.Tensor | None = None,
) -> torch.Tensor:
    """Choose bounded deterministic rows, retaining local-positive rows when possible."""

    total = int(total)
    count = max(1, int(count))
    if total <= 0:
        return torch.empty(0, dtype=torch.long)
    if eligible is None:
        eligible_indices = torch.arange(total, dtype=torch.long)
    else:
        eligible_tensor = torch.as_tensor(eligible, dtype=torch.bool).reshape(-1)
        if eligible_tensor.numel() < total:
            padded = torch.zeros(total, dtype=torch.bool)
            padded[: eligible_tensor.numel()] = eligible_tensor
            eligible_tensor = padded
        eligible_indices = torch.where(eligible_tensor[:total])[0]
        # A backbone mask should normally contain foreground tokens.  If it
        # does not, retain all tokens rather than silently dropping a source.
        if eligible_indices.numel() == 0:
            eligible_indices = torch.arange(total, dtype=torch.long)
    count = min(count, int(eligible_indices.numel()))
    if count <= 0:
        return eligible_indices[:0]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % (2**63 - 1))
    required_indices = torch.empty(0, dtype=torch.long)
    if required is not None:
        required = torch.as_tensor(required, dtype=torch.bool).reshape(-1)
        if required.numel() < total:
            padded = torch.zeros(total, dtype=torch.bool)
            padded[: required.numel()] = required
            required = padded
        required_indices = torch.where(required[:total])[0]
        eligible_required = required_indices[torch.isin(required_indices, eligible_indices)]
        required_indices = eligible_required
    required_indices = required_indices[:count]
    candidates = eligible_indices[~torch.isin(eligible_indices, required_indices)]
    order = torch.randperm(candidates.numel(), generator=generator)
    remaining = candidates[order[: max(0, count - required_indices.numel())]]
    return torch.cat((required_indices, remaining), dim=0).sort().values


def _chunked_cosine_distance(
    queries: torch.Tensor, bank: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    """Return ``1 - max cosine`` with bounded query and bank chunks."""

    queries = torch.as_tensor(queries, dtype=torch.float32)
    bank = torch.as_tensor(bank, dtype=torch.float32)
    if queries.ndim != 2 or bank.ndim != 2:
        raise ValueError("nearest-neighbour inputs must be two-dimensional")
    if queries.shape[1] != bank.shape[1] and bank.shape[0] > 0:
        raise ValueError("nearest-neighbour feature dimensions do not match")
    if queries.shape[0] == 0:
        return queries.new_empty((0,))
    if bank.shape[0] == 0:
        return queries.new_zeros((queries.shape[0],))
    chunk_size = max(1, int(chunk_size))
    normalized_queries = F.normalize(queries, dim=1, eps=1e-8)
    normalized_bank = F.normalize(bank, dim=1, eps=1e-8)
    output: list[torch.Tensor] = []
    for q_start in range(0, normalized_queries.shape[0], chunk_size):
        query_chunk = normalized_queries[q_start : q_start + chunk_size]
        best = torch.full(
            (query_chunk.shape[0],),
            -1.0,
            dtype=query_chunk.dtype,
            device=query_chunk.device,
        )
        for b_start in range(0, normalized_bank.shape[0], chunk_size):
            bank_chunk = normalized_bank[b_start : b_start + chunk_size]
            similarity = query_chunk @ bank_chunk.T
            best = torch.maximum(best, similarity.max(dim=1).values)
        output.append((1.0 - best).clamp_min(0.0))
    return torch.cat(output, dim=0)


def _sample_indices_preserving_masks(
    total: int,
    count: int,
    seed: int,
    masks: Sequence[torch.Tensor],
    eligible: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Bound a source cache while retaining one visible token per mask.

    ``masks`` are synthetic anomaly token masks.  Visibility is intersected
    with the framework foreground mask, so a synthetic family fully hidden by
    background masking is skipped during training rather than relabeling
    unrelated normal patches.  The returned second value is the number of
    families with no visible token; this is recorded in training diagnostics.
    """

    total = int(total)
    count = max(1, int(count))
    if total <= 0:
        return torch.empty(0, dtype=torch.long), len(masks)
    if eligible is None:
        eligible_tensor = torch.ones(total, dtype=torch.bool)
    else:
        eligible_tensor = torch.as_tensor(eligible, dtype=torch.bool).reshape(-1)
        if eligible_tensor.numel() != total:
            raise ValueError("eligible patch mask has a different patch count")
    eligible_indices = torch.where(eligible_tensor)[0]
    if eligible_indices.numel() == 0:
        eligible_indices = torch.arange(total, dtype=torch.long)
        eligible_tensor = torch.ones(total, dtype=torch.bool)
    effective_count = min(count, int(eligible_indices.numel()))

    anchors: list[int] = []
    skipped = 0
    for mask in masks:
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool).reshape(-1)
        if mask_tensor.numel() != total:
            raise ValueError("synthetic anomaly mask has a different patch count")
        visible = mask_tensor & eligible_tensor
        if not bool(visible.any()):
            skipped += 1
            continue
        anchors.append(int(torch.where(visible)[0][0].item()))
    if len(anchors) > effective_count:
        raise ValueError(
            "max_patches_per_source is too small to retain one visible synthetic "
            "mask positive per anomaly family"
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % (2**63 - 1))
    anchor_tensor = torch.tensor(sorted(set(anchors)), dtype=torch.long)
    candidates = eligible_indices[~torch.isin(eligible_indices, anchor_tensor)]
    order = torch.randperm(candidates.numel(), generator=generator)
    remaining = candidates[order[: max(0, effective_count - anchor_tensor.numel())]]
    selected = torch.cat((anchor_tensor, remaining), dim=0)
    return selected.sort().values, skipped


@METHOD_REGISTRY.register("PSMA_MR")
class PSMA_MR(BaseMethod):
    """Paired synthetic support adaptation with two-resolution cosine banks."""

    def __init__(self, backbone, config: dict):
        super().__init__(backbone, config)
        if not isinstance(config, Mapping):
            raise TypeError("PSMA_MR config must be a flat mapping")
        shared_model = getattr(backbone, "model", None)
        if isinstance(shared_model, nn.Module):
            shared_model.eval()
            for parameter in shared_model.parameters():
                parameter.requires_grad_(False)
        # The pipeline passes method_config directly.  Do not inspect nested
        # wrappers or speculative aliases: they hide precedence bugs and make
        # the recorded method provenance ambiguous.
        self.params = dict(config)
        revision = self.params.get("revision", "v1")
        if revision not in ("v1", "v2"):
            raise ValueError("revision must be 'v1' or 'v2'")
        self.revision = str(revision)
        self.v2_feature_layers = (5, 11, 17, 23)
        self.resolutions = _parse_resolutions(self.params)
        self.rank = _positive_int("adapter_rank", self.params.get("adapter_rank", 4))
        adapter_setting = self.params.get("adapter_enabled", True)
        if not isinstance(adapter_setting, bool):
            raise ValueError("adapter_enabled must be a boolean")
        self.adapter_enabled = adapter_setting
        default_steps = 32 if self.revision == "v2" else 24
        self.steps = _positive_int("steps", self.params.get("steps", default_steps), allow_zero=True)
        self.learning_rate = _finite_float(
            "learning_rate", self.params.get("learning_rate", 1e-3), minimum=0.0
        )
        self.weight_decay = _finite_float(
            "weight_decay", self.params.get("weight_decay", 0.0), minimum=0.0
        )
        self.margin = _finite_float("margin", self.params.get("margin", 0.10), minimum=0.0)
        self.consistency_weight = _finite_float(
            "normal_consistency_weight",
            self.params.get("normal_consistency_weight", 1.0),
            minimum=0.0,
        )
        self.margin_weight = _finite_float(
            "synthetic_margin_weight",
            self.params.get("synthetic_margin_weight", 1.0),
            minimum=0.0,
        )
        self.local_weight = _finite_float(
            "local_weight", self.params.get("local_weight", self.margin_weight), minimum=0.0
        )
        self.anchor_weight = _finite_float(
            "feature_anchor_weight",
            self.params.get("feature_anchor_weight", 0.10),
            minimum=0.0,
        )
        self.max_patches_per_source = _positive_int(
            "max_patches_per_source", self.params.get("max_patches_per_source", 256)
        )
        self.max_bank_patches = _positive_int(
            "max_bank_patches", self.params.get("max_bank_patches", self.max_patches_per_source)
        )
        self.max_train_patches = _positive_int(
            "max_train_patches", self.params.get("max_train_patches", 96)
        )
        self.nn_chunk_size = _positive_int(
            "nn_chunk_size", self.params.get("nn_chunk_size", 1024)
        )
        self.photometric_strength = _finite_float(
            "photometric_strength", self.params.get("photometric_strength", 1.0), minimum=0.0
        )
        names = self.params.get(
            "synthetic_anomaly_types", ("rectangle_cutpaste", "scratch_noise")
        )
        if isinstance(names, str):
            names = [names]
        if not isinstance(names, (list, tuple)):
            raise ValueError("synthetic_anomaly_types must be a sequence")
        selected_names = [str(name) for name in names]
        for default_name in ("rectangle_cutpaste", "scratch_noise"):
            if default_name not in selected_names:
                selected_names.append(default_name)
        self.synthetic_anomaly_types = tuple(selected_names)
        self.gradient_clip = _finite_float(
            "gradient_clip", self.params.get("gradient_clip", 5.0), minimum=0.0
        )
        self.residual_epsilon = _finite_float(
            "residual_epsilon", self.params.get("residual_epsilon", 0.10), minimum=0.0
        )
        # v2 defaults are the approved bounded-memory contract.  The legacy
        # names remain accepted for tiny local fixtures and controlled tests;
        # the canonical v2 config uses final_bank_cap/intermediate_bank_cap.
        self.final_bank_cap = _positive_int(
            "final_bank_cap",
            self.params.get("final_bank_cap", self.params.get("max_bank_patches", 256)),
        )
        self.intermediate_bank_cap = _positive_int(
            "intermediate_bank_cap", self.params.get("intermediate_bank_cap", 128)
        )
        self.v2_train_tokens = _positive_int(
            "train_tokens", self.params.get("train_tokens", self.params.get("max_train_patches", 96))
        )
        if self.revision == "v2" and self.rank != 4:
            raise ValueError("PSMA_MR v2 requires adapter_rank=4")
        self._exact_config = _json_safe(copy.deepcopy(config))

        self.adapter: _Rank4Residual | None = None
        self._support_paths: list[str] = []
        self._context: MethodContext | None = None
        self._seed: int | None = None
        self._masking = False
        self._rotation = False
        self._feature_dim: int | None = None
        self._selected_layer: int | None = None
        self._records: list[dict[int, _ResolutionViews]] = []
        self._loo_banks: dict[tuple[int, int], torch.Tensor] = {}
        self._loo_bank_sources: dict[tuple[int, int], tuple[int, ...]] = {}
        self._banks: dict[int, torch.Tensor] = {}
        self._grid_sizes: dict[int, tuple[int, int]] = {}
        self._fit_complete = False
        self._executed_steps = 0
        self._last_losses: dict[str, float] = {}
        self._skipped_synthetic_types = 0

        # v2 owns independent multi-branch banks and calibration.  Keeping
        # these separate from v1 state makes the revision switch explicit and
        # ensures an adapter-off control cannot reuse a historical bank.
        self._v2_records: list[dict[int, _V2ResolutionViews]] = []
        self._v2_banks: dict[tuple[int, int], torch.Tensor] = {}
        self._v2_bank_source_ids: dict[tuple[int, int], torch.Tensor] = {}
        self._v2_bank_descendant_ids: dict[tuple[int, int], torch.Tensor] = {}
        self._v2_loo_banks: dict[tuple[int, int, int], torch.Tensor] = {}
        self._v2_loo_bank_source_ids: dict[tuple[int, int, int], torch.Tensor] = {}
        self._v2_calibration: dict[tuple[int, int], dict[str, float | int]] = {}
        self._v2_schedule: list[dict[str, int]] = []
        self._v2_adapter_fit_complete = False
        self._v2_calibration_complete = False
        self._v2_branch_bank_caps: dict[int, int] = {}
        self._v2_final_layer = 23
        self._v2_final_branch_index = 3

    def _reset_state(self) -> None:
        """Clear every fitted object so repeated fit calls cannot retain state."""

        self.adapter = None
        self._support_paths = []
        self._context = None
        self._seed = None
        self._masking = False
        self._rotation = False
        self._feature_dim = None
        self._selected_layer = None
        self._records = []
        self._loo_banks = {}
        self._loo_bank_sources = {}
        self._banks = {}
        self._grid_sizes = {}
        self._fit_complete = False
        self._executed_steps = 0
        self._last_losses = {}
        self._skipped_synthetic_types = 0
        self._v2_records = []
        self._v2_banks = {}
        self._v2_bank_source_ids = {}
        self._v2_bank_descendant_ids = {}
        self._v2_loo_banks = {}
        self._v2_loo_bank_source_ids = {}
        self._v2_calibration = {}
        self._v2_schedule = []
        self._v2_adapter_fit_complete = False
        self._v2_calibration_complete = False
        self._v2_branch_bank_caps = {}

    def _prepare_image(self, image: np.ndarray, resolution: int):
        """Call the shared non-mutating resolution-aware backbone API."""

        # Do not catch TypeError here: a missing explicit resolution argument
        # is an integration error and would make multi-resolution fitting
        # order-dependent.
        result = self.backbone.prepare_image(image, resolution=resolution)
        if isinstance(result, (tuple, list)) and len(result) == 2:
            return result[0], result[1]
        return result, None

    def _extract_final(
        self, image: np.ndarray, resolution: int
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        image_tensor, declared_grid = self._prepare_image(image, resolution)
        with torch.inference_mode():
            outputs = self.backbone.extract_features(image_tensor)
        if isinstance(outputs, Mapping):
            outputs = list(outputs.values())
        if not isinstance(outputs, (tuple, list)) or not outputs:
            raise ValueError("backbone.extract_features must return a non-empty sequence")
        feature_layers = tuple(getattr(self.backbone, "feature_layers", ()))
        self._selected_layer = int(feature_layers[-1]) if feature_layers else len(outputs) - 1
        matrix, grid = _feature_matrix(outputs[-1], declared_grid)
        if not torch.isfinite(matrix).all():
            raise ValueError("backbone features must be finite")
        return matrix, grid

    def _background_mask(
        self, features: torch.Tensor, grid_size: tuple[int, int]
    ) -> torch.Tensor:
        """Return the framework-consistent foreground mask for one feature view."""

        compute_mask = getattr(self.backbone, "compute_background_mask", None)
        if not callable(compute_mask):
            if self._masking:
                raise RuntimeError(
                    "MethodContext.masking=True requires backbone.compute_background_mask; "
                    "use the neutral no-mask preprocessing configuration for this backbone"
                )
            return torch.ones(features.shape[0], dtype=torch.bool)
        raw_mask = compute_mask(
            features.detach().cpu().numpy(),
            grid_size,
            threshold=1,
            masking_type=bool(self._masking),
        )
        mask = torch.as_tensor(raw_mask, dtype=torch.bool).reshape(-1)
        if mask.numel() != features.shape[0]:
            raise ValueError(
                "backbone.compute_background_mask returned a mask with the wrong patch count"
            )
        if bool(mask.any()):
            return mask
        if self._masking:
            raise RuntimeError(
                "backbone foreground mask is empty under MethodContext.masking=True; "
                "use neutral no-mask preprocessing or fix the backbone mask"
            )
        return torch.ones_like(mask)

    # ------------------------------------------------------------------ v2
    # v2 uses one frozen multi-output backbone pass per image/resolution.  The
    # final branch alone owns the rank-4 adapter; intermediate branches remain
    # raw frozen features and are never fed through trainable parameters.

    def _v2_extract_outputs(
        self, image: np.ndarray, resolution: int
    ) -> tuple[tuple[torch.Tensor, ...], tuple[int, int]]:
        image_tensor, declared_grid = self._prepare_image(image, resolution)
        with torch.inference_mode():
            outputs = self.backbone.extract_features(image_tensor)
        if isinstance(outputs, Mapping):
            outputs = list(outputs.values())
        if not isinstance(outputs, (tuple, list)) or len(outputs) != len(self.v2_feature_layers):
            raise ValueError(
                "PSMA_MR v2 requires exactly four outputs for frozen layers "
                "[5, 11, 17, 23]"
            )
        available_layers = tuple(getattr(self.backbone, "feature_layers", self.v2_feature_layers))
        if available_layers and tuple(available_layers) != self.v2_feature_layers:
            raise ValueError(
                "PSMA_MR v2 requires backbone feature_layers=(5, 11, 17, 23); "
                f"got {available_layers}"
            )
        matrices: list[torch.Tensor] = []
        grid: tuple[int, int] | None = None
        for output in outputs:
            matrix, output_grid = _feature_matrix(output, declared_grid)
            if grid is None:
                grid = output_grid
            if output_grid != grid:
                raise ValueError("PSMA_MR v2 feature grids differ across frozen layers")
            if not torch.isfinite(matrix).all():
                raise ValueError("backbone features must be finite")
            matrices.append(matrix)
        if grid is None:  # pragma: no cover - four outputs are required above
            raise RuntimeError("PSMA_MR v2 did not receive feature outputs")
        self._selected_layer = self.v2_feature_layers[-1]
        return tuple(matrices), grid

    def _v2_branch_features(self, branch_index: int, features: torch.Tensor) -> torch.Tensor:
        """Apply the final-only adapter and normalize frozen branches."""

        if branch_index == self._v2_final_branch_index:
            if self.adapter is None:
                raise RuntimeError("v2 adapter has not been initialized")
            return self.adapter(features)
        return F.normalize(torch.as_tensor(features, dtype=torch.float32), dim=-1, eps=1e-8)

    def _v2_build_loo_banks(self) -> None:
        """Index raw leave-one-source-out banks for adapter fitting/calibration.

        The tensors are assembled on demand instead of retaining one 15-source
        copy per held-out source.  This keeps the persistent v2 bank near the
        all-support inference footprint (about 80 MiB at fp32/d=1024) while
        preserving exact source and descendant exclusion.
        """

        self._v2_loo_banks = {}
        self._v2_loo_bank_source_ids = {}
        for source_index in range(len(self._v2_records)):
            for branch_index, layer in enumerate(self.v2_feature_layers):
                for resolution in self.resolutions:
                    pieces: list[torch.Tensor] = []
                    source_ids: list[int] = []
                    for other_index, record in enumerate(self._v2_records):
                        if other_index == source_index:
                            continue
                        source_ids.extend(
                            [other_index]
                            * int(record[resolution].clean[branch_index].shape[0])
                        )
                    key = (source_index, layer, resolution)
                    self._v2_loo_bank_source_ids[key] = torch.tensor(
                        source_ids, dtype=torch.long
                    )

    def _v2_source_excluded_bank(
        self, source_index: int, branch_index: int, resolution: int
    ) -> torch.Tensor:
        """Assemble one raw bank while excluding a source and all descendants."""

        pieces = [
            record[resolution].clean[branch_index]
            for other_index, record in enumerate(self._v2_records)
            if other_index != source_index
        ]
        if pieces:
            return torch.cat(pieces, dim=0).contiguous()
        dimension = int(self._v2_records[source_index][resolution].clean[branch_index].shape[1])
        return torch.empty((0, dimension), dtype=torch.float32)

    def _v2_training_step(
        self, source_index: int, resolution: int, step: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run one final-branch support-only update with aligned local hinge terms."""

        if self.adapter is None:
            raise RuntimeError("v2 adapter has not been initialized")
        views = self._v2_records[source_index][resolution]
        final = self._v2_final_branch_index
        raw_bank = self._v2_source_excluded_bank(source_index, final, resolution)
        bank = self.adapter(raw_bank)
        train_indices, _ = _spatial_stratified_select(
            views.coords[final],
            self.v2_train_tokens,
            seed=(self._seed or 0) + int(step) * 17 + 1,
            eligible=views.foreground[final],
            required_masks=[mask for _, _, mask, _ in views.anomalies],
        )
        clean = views.clean[final][train_indices]
        nuisance = views.nuisance[final][train_indices]
        clean_adapted = self.adapter(clean)
        nuisance_adapted = self.adapter(nuisance)
        clean_distance = _chunked_cosine_distance(clean_adapted, bank, self.nn_chunk_size)
        nuisance_distance = _chunked_cosine_distance(nuisance_adapted, bank, self.nn_chunk_size)
        consistency = (clean_distance - nuisance_distance).square().mean()

        local_terms: list[torch.Tensor] = []
        skipped_types = 0
        for anomaly_identity, anomaly_nuisance, patch_mask, _name in views.anomalies:
            local_mask = patch_mask[train_indices]
            if not bool(local_mask.any()):
                skipped_types += 1
                continue
            identity_distance = _chunked_cosine_distance(
                self.adapter(anomaly_identity[train_indices][local_mask]),
                bank,
                self.nn_chunk_size,
            )
            nuisance_anomaly_distance = _chunked_cosine_distance(
                self.adapter(anomaly_nuisance[train_indices][local_mask]),
                bank,
                self.nn_chunk_size,
            )
            # Keep the aligned identity/T positions through the hinge.  Taking
            # means before applying relu would erase the intended local guard.
            local_terms.extend(
                [
                    F.relu(self.margin + clean_distance[local_mask] - identity_distance),
                    F.relu(self.margin + nuisance_distance[local_mask] - nuisance_anomaly_distance),
                ]
            )
        local = torch.cat(local_terms).mean() if local_terms else clean_distance.new_zeros(())
        anchor_target = F.normalize(clean.detach(), dim=-1, eps=1e-8)
        anchor = (1.0 - (clean_adapted * anchor_target).sum(dim=-1)).mean()
        total = (
            self.consistency_weight * consistency
            + self.local_weight * local
            + self.anchor_weight * anchor
        )
        return total, {
            "normal_consistency": consistency,
            "consistency": consistency,
            "synthetic_margin": local,
            "local": local,
            "feature_anchor": anchor,
            "anchor": anchor,
            "loss": total,
            "synthetic_skipped_types": clean_distance.new_tensor(float(skipped_types)),
        }

    def _v2_build_inference_banks(self) -> None:
        """Materialize independent inference banks from all support sources."""

        if self.adapter is None:
            raise RuntimeError("v2 adapter has not been initialized")
        self._v2_banks = {}
        self._v2_bank_source_ids = {}
        self._v2_bank_descendant_ids = {}
        with torch.inference_mode():
            for branch_index, layer in enumerate(self.v2_feature_layers):
                for resolution in self.resolutions:
                    pieces: list[torch.Tensor] = []
                    source_ids: list[int] = []
                    descendant_ids: list[int] = []
                    for source_index, record in enumerate(self._v2_records):
                        views = record[resolution]
                        raw = views.clean[branch_index]
                        pieces.append(self._v2_branch_features(branch_index, raw).contiguous())
                        source_ids.extend([source_index] * int(raw.shape[0]))
                        descendant_ids.extend(views.coords[branch_index][:, 0].tolist())
                    if pieces:
                        bank = torch.cat(pieces, dim=0).contiguous()
                    else:  # pragma: no cover - fit rejects empty support paths
                        bank = torch.empty((0, 0), dtype=torch.float32)
                    key = (layer, resolution)
                    self._v2_banks[key] = bank
                    self._v2_bank_source_ids[key] = torch.tensor(source_ids, dtype=torch.long)
                    self._v2_bank_descendant_ids[key] = torch.tensor(
                        descendant_ids, dtype=torch.long
                    )

    def _v2_fit_calibration(self) -> None:
        """Fit branch affine calibration on clean support tokens with LOO banks."""

        if self.adapter is None:
            raise RuntimeError("v2 adapter has not been initialized")
        values: dict[tuple[int, int], list[float]] = {
            (layer, resolution): []
            for layer in self.v2_feature_layers
            for resolution in self.resolutions
        }
        with torch.inference_mode():
            for source_index, record in enumerate(self._v2_records):
                for branch_index, layer in enumerate(self.v2_feature_layers):
                    for resolution in self.resolutions:
                        key = (source_index, layer, resolution)
                        raw_bank = self._v2_source_excluded_bank(
                            source_index, branch_index, resolution
                        )
                        bank = self._v2_branch_features(
                            branch_index, raw_bank
                        )
                        views = record[resolution]
                        held_out = self._v2_branch_features(
                            branch_index, views.clean[branch_index]
                        )
                        distances = _chunked_cosine_distance(
                            held_out, bank, self.nn_chunk_size
                        )
                        foreground = views.foreground[branch_index]
                        if distances.numel() and bool(foreground.any()):
                            values[(layer, resolution)].extend(
                                distances[foreground].detach().cpu().tolist()
                            )
        self._v2_calibration = {}
        for key, raw_values in values.items():
            array = np.asarray(raw_values, dtype=np.float64)
            if array.size:
                median = float(np.median(array))
                mad = float(np.median(np.abs(array - median)))
            else:
                median = 0.0
                mad = 0.0
            scale = max(1.4826 * mad, 0.001)
            self._v2_calibration[key] = {
                "median": median,
                "mad": mad,
                "scale": scale,
                "count": int(array.size),
            }
        self._v2_calibration_complete = True

    def _v2_calibrate(self, distances: torch.Tensor, layer: int, resolution: int) -> torch.Tensor:
        stats = self._v2_calibration[(layer, resolution)]
        # Intentionally no clamp: affine calibration is allowed to retain
        # negative values and values above one for ranking diagnostics.
        return (distances - float(stats["median"])) / float(stats["scale"])

    @staticmethod
    def _resize_score_map(score_map: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
        height, width = (int(image_shape[0]), int(image_shape[1]))
        if cv2 is not None:
            return np.asarray(
                cv2.resize(score_map, (width, height), interpolation=cv2.INTER_LINEAR),
                dtype=np.float32,
            )
        return np.asarray(
            Image.fromarray(np.asarray(score_map, dtype=np.float32)).resize(
                (width, height), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )

    def _fit_v2(self, paths: list[str], context: MethodContext) -> None:
        """Fit the approved v2 four-branch, final-only adapter variant."""

        if tuple(self.resolutions) != (448, 672):
            # Small explicit resolutions remain useful for CPU contract tests;
            # the canonical experiment config below is fixed at 448/672.
            self._v2_resolutions_are_canonical = False
        else:
            self._v2_resolutions_are_canonical = True
        self._v2_branch_bank_caps = {
            layer: (self.final_bank_cap if layer == self._v2_final_layer else self.intermediate_bank_cap)
            for layer in self.v2_feature_layers
        }
        for source_index, path in enumerate(paths):
            image = _read_rgb(path)
            generator = np.random.default_rng(self._seed + 1009 * (source_index + 1))
            source_views = _augment_image(image) if self._rotation else [image]
            _, nuisance_transform = _photometric_pair(
                image, generator, self.photometric_strength
            )
            descendant_anomalies = [
                _synthetic_anomalies(clean_image, generator, self.synthetic_anomaly_types)
                for clean_image in source_views
            ]
            anomaly_names = [name for _, _, name in descendant_anomalies[0]]
            if len(anomaly_names) < 2:
                raise RuntimeError("PSMA_MR requires at least two synthetic anomaly families")
            if any(
                [name for _, _, name in anomalies] != anomaly_names
                for anomalies in descendant_anomalies
            ):
                raise RuntimeError("synthetic anomaly descendants changed family ordering")

            by_resolution: dict[int, _V2ResolutionViews] = {}
            for resolution in self.resolutions:
                clean_parts: list[list[torch.Tensor]] = [[] for _ in self.v2_feature_layers]
                nuisance_parts: list[list[torch.Tensor]] = [
                    [] for _ in self.v2_feature_layers
                ]
                foreground_parts: list[torch.Tensor] = []
                coordinate_parts: list[torch.Tensor] = []
                anomaly_identity_parts: list[list[torch.Tensor]] = [
                    [] for _ in anomaly_names
                ]
                anomaly_nuisance_parts: list[list[torch.Tensor]] = [
                    [] for _ in anomaly_names
                ]
                anomaly_mask_parts: list[list[torch.Tensor]] = [
                    [] for _ in anomaly_names
                ]
                grid_size: tuple[int, int] | None = None
                for descendant_id, (clean_image, anomalies) in enumerate(
                    zip(source_views, descendant_anomalies)
                ):
                    clean_outputs, clean_grid = self._v2_extract_outputs(
                        clean_image, resolution
                    )
                    nuisance_image = _apply_photometric(clean_image, nuisance_transform)
                    nuisance_outputs, nuisance_grid = self._v2_extract_outputs(
                        nuisance_image, resolution
                    )
                    if grid_size is None:
                        grid_size = clean_grid
                    if nuisance_grid != grid_size:
                        raise ValueError("backbone feature grid changed across v2 paired views")
                    if any(
                        clean_outputs[index].shape[1]
                        != nuisance_outputs[index].shape[1]
                        for index in range(len(self.v2_feature_layers))
                    ):
                        raise ValueError("backbone feature dimension changed across v2 pairs")
                    foreground = self._background_mask(clean_outputs[-1], grid_size)
                    coordinates = _spatial_coordinates(grid_size, descendant_id)
                    foreground_parts.append(foreground)
                    coordinate_parts.append(coordinates)
                    for branch_index in range(len(self.v2_feature_layers)):
                        clean_parts[branch_index].append(clean_outputs[branch_index])
                        nuisance_parts[branch_index].append(nuisance_outputs[branch_index])
                    for anomaly_index, (anomaly_image, pixel_mask, _name) in enumerate(
                        anomalies
                    ):
                        # Synthetic views intentionally retain only final
                        # features; intermediate outputs are not cached.
                        anomaly_outputs, anomaly_grid = self._v2_extract_outputs(
                            anomaly_image, resolution
                        )
                        anomaly_nuisance_outputs, anomaly_nuisance_grid = (
                            self._v2_extract_outputs(
                                _apply_photometric(anomaly_image, nuisance_transform),
                                resolution,
                            )
                        )
                        if anomaly_grid != grid_size or anomaly_nuisance_grid != grid_size:
                            raise ValueError("backbone feature grid changed across v2 synthetic views")
                        anomaly_identity_parts[anomaly_index].append(anomaly_outputs[-1])
                        anomaly_nuisance_parts[anomaly_index].append(
                            anomaly_nuisance_outputs[-1]
                        )
                        anomaly_mask_parts[anomaly_index].append(
                            _token_mask(pixel_mask, grid_size, anomaly_outputs[-1].shape[0])
                        )
                if grid_size is None:  # pragma: no cover - source_views is never empty
                    raise RuntimeError("no support descendants were generated")
                full_clean = tuple(torch.cat(parts, dim=0) for parts in clean_parts)
                full_nuisance = tuple(torch.cat(parts, dim=0) for parts in nuisance_parts)
                full_foreground = torch.cat(foreground_parts, dim=0)
                full_coordinates = torch.cat(coordinate_parts, dim=0)
                anomaly_records_raw = tuple(
                    (
                        torch.cat(identity_parts, dim=0),
                        torch.cat(nuisance_feature_parts, dim=0),
                        torch.cat(mask_parts, dim=0),
                        name,
                    )
                    for identity_parts, nuisance_feature_parts, mask_parts, name in zip(
                        anomaly_identity_parts,
                        anomaly_nuisance_parts,
                        anomaly_mask_parts,
                        anomaly_names,
                    )
                )
                final_index, skipped_types = _spatial_stratified_select(
                    full_coordinates,
                    self.final_bank_cap,
                    seed=self._seed + 8191 * (source_index + 1) + resolution,
                    eligible=full_foreground,
                    required_masks=[mask for _, _, mask, _ in anomaly_records_raw],
                )
                self._skipped_synthetic_types += skipped_types
                clean_selected: list[torch.Tensor] = []
                nuisance_selected: list[torch.Tensor] = []
                foreground_selected: list[torch.Tensor] = []
                coordinates_selected: list[torch.Tensor] = []
                for branch_index, layer in enumerate(self.v2_feature_layers):
                    if branch_index == self._v2_final_branch_index:
                        branch_index_rows = final_index
                    else:
                        branch_index_rows = _spatial_stratified_indices(
                            full_coordinates,
                            self.intermediate_bank_cap,
                            seed=self._seed + 8191 * (source_index + 1) + resolution + branch_index,
                            eligible=full_foreground,
                        )
                    clean_selected.append(full_clean[branch_index][branch_index_rows])
                    nuisance_selected.append(full_nuisance[branch_index][branch_index_rows])
                    foreground_selected.append(full_foreground[branch_index_rows])
                    coordinates_selected.append(full_coordinates[branch_index_rows])
                anomalies = tuple(
                    (
                        identity_features[final_index],
                        nuisance_features[final_index],
                        mask[final_index],
                        name,
                    )
                    for identity_features, nuisance_features, mask, name in anomaly_records_raw
                )
                by_resolution[resolution] = _V2ResolutionViews(
                    clean=tuple(clean_selected),
                    nuisance=tuple(nuisance_selected),
                    foreground=tuple(foreground_selected),
                    coords=tuple(coordinates_selected),
                    anomalies=anomalies,
                    grid_size=grid_size,
                )
            self._v2_records.append(by_resolution)
            self.log_training(
                step=source_index + 1,
                stage="v2_support_features",
                source_index=source_index,
                support_total=len(paths),
                resolutions=list(self.resolutions),
                feature_layers=list(self.v2_feature_layers),
                synthetic_cache="final_only",
                spatial_stratified=True,
                descendant_count=len(source_views),
                final_bank_cap=self.final_bank_cap,
                intermediate_bank_cap=self.intermediate_bank_cap,
                train_tokens=self.v2_train_tokens,
                skipped_synthetic_types=self._skipped_synthetic_types,
            )

        self._feature_dim = int(self._v2_records[0][self.resolutions[0]].clean[-1].shape[1])
        self.adapter = _Rank4Residual(
            self._feature_dim,
            rank=4,
            seed=self._seed or 0,
            enabled=self.adapter_enabled,
            residual_epsilon=self.residual_epsilon,
        )
        if not self.adapter_enabled:
            for parameter in self.adapter.parameters():
                parameter.requires_grad_(False)
        self._v2_build_loo_banks()

        self._v2_schedule = []
        branch_count = len(self.resolutions)
        for step in range(self.steps):
            source_index = (step // branch_count) % len(self._v2_records)
            resolution = self.resolutions[step % branch_count]
            self._v2_schedule.append(
                {"t": step, "source_index": source_index, "resolution": resolution}
            )
        if self.adapter_enabled and self.steps > 0:
            optimizer = torch.optim.Adam(
                self.adapter.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            for step in range(self.steps):
                source_index = self._v2_schedule[step]["source_index"]
                resolution = self._v2_schedule[step]["resolution"]
                optimizer.zero_grad(set_to_none=True)
                loss, components = self._v2_training_step(source_index, resolution, step)
                loss.backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), self.gradient_clip)
                optimizer.step()
                self._executed_steps = step + 1
                self._last_losses = {
                    name: float(value.detach().cpu().item()) for name, value in components.items()
                }
                self.log_training(
                    step=step,
                    stage="v2_adapter_step",
                    adapter_enabled=True,
                    **self._v2_schedule[-1],
                    **self._last_losses,
                )
        else:
            self.log_training(
                step=0,
                stage="v2_adapter_disabled" if not self.adapter_enabled else "v2_adapter_no_steps",
                adapter_enabled=self.adapter_enabled,
                loss=0.0,
                consistency=0.0,
                local=0.0,
                anchor=0.0,
            )
        self.adapter.eval()
        self._v2_adapter_fit_complete = True
        # Calibration is deliberately a single post-adapter operation and
        # sees only clean support features through source-excluded banks.
        self._v2_fit_calibration()
        self._v2_build_inference_banks()
        self._fit_complete = True
        self.log_training(
            stage="v2_memory_bank_ready",
            branch_count=len(self.v2_feature_layers) * len(self.resolutions),
            bank_vectors={
                f"layer{layer}:resolution{resolution}": int(bank.shape[0])
                for (layer, resolution), bank in self._v2_banks.items()
            },
            calibration_fitted=True,
            calibration_source_exclusion="LOO",
            adapter_enabled=self.adapter_enabled,
        )

    def _predict_v2(self, test_image_path: str) -> tuple[np.ndarray, float]:
        """Score eight independently calibrated layer/resolution branches."""

        if (
            not self._fit_complete
            or self.adapter is None
            or not self._v2_calibration_complete
        ):
            raise RuntimeError("fit must complete before prediction")
        image = _read_rgb(test_image_path)
        branch_maps: list[np.ndarray] = []
        with torch.inference_mode():
            for resolution in self.resolutions:
                outputs, grid_size = self._v2_extract_outputs(image, resolution)
                foreground = self._background_mask(outputs[-1], grid_size)
                for branch_index, layer in enumerate(self.v2_feature_layers):
                    features = self._v2_branch_features(branch_index, outputs[branch_index])
                    raw_distance = _chunked_cosine_distance(
                        features,
                        self._v2_banks[(layer, resolution)],
                        self.nn_chunk_size,
                    )
                    calibrated = self._v2_calibrate(raw_distance, layer, resolution)
                    calibrated = torch.where(foreground, calibrated, torch.zeros_like(calibrated))
                    branch_map = calibrated.reshape(grid_size).detach().cpu().numpy().astype(
                        np.float32
                    )
                    branch_maps.append(self._resize_score_map(branch_map, image.shape[:2]))
        expected_branches = len(self.v2_feature_layers) * len(self.resolutions)
        if len(branch_maps) != expected_branches:
            raise RuntimeError(
                f"PSMA_MR v2 expected {expected_branches} branch maps, got {len(branch_maps)}"
            )
        anomaly_map = np.mean(np.stack(branch_maps, axis=0), axis=0).astype(np.float32)
        return anomaly_map, _mean_top1p(anomaly_map)

    def _build_loo_banks(self) -> None:
        if self._feature_dim is None:
            raise RuntimeError("feature dimension is unknown")
        for resolution in self.resolutions:
            for source_index in range(len(self._records)):
                pieces: list[torch.Tensor] = []
                sources: list[int] = []
                for other_index, record in enumerate(self._records):
                    if other_index == source_index:
                        continue
                    clean = record[resolution].clean
                    clean = _sample_rows(
                        clean,
                        self.max_bank_patches,
                        seed=(self._seed or 0) + 104729 * (source_index + 1) + other_index,
                    )
                    pieces.append(clean)
                    sources.append(other_index)
                if pieces:
                    bank = torch.cat(pieces, dim=0)
                else:
                    bank = torch.empty((0, self._feature_dim), dtype=torch.float32)
                # This is intentionally a clean-source bank.  Thus no
                # photometric or synthetic descendant of the held-out source
                # (or of any source) can enter its leave-one-source-out bank.
                self._loo_banks[(source_index, resolution)] = bank
                self._loo_bank_sources[(source_index, resolution)] = tuple(sources)

    def _training_step(
        self, source_index: int, resolution: int, step: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.adapter is None:
            raise RuntimeError("adapter has not been initialized")
        views = self._records[source_index][resolution]
        raw_bank = _sample_rows(
            self._loo_banks[(source_index, resolution)],
            self.max_bank_patches,
            (self._seed or 0) + step * 13 + 7,
        )
        # The bank is deliberately stored raw.  Adapt it on every step so
        # train-time nearest-neighbour geometry is exactly the same as the
        # final inference geometry, while retaining gradients through U/V.
        bank = self.adapter(raw_bank)

        patch_indices, _ = _sample_indices_preserving_masks(
            views.clean.shape[0],
            self.max_train_patches,
            (self._seed or 0) + step * 17 + 1,
            [mask for _, _, mask, _ in views.anomalies],
        )
        clean = views.clean[patch_indices]
        nuisance = views.nuisance[patch_indices]
        clean_adapted = self.adapter(clean)
        nuisance_adapted = self.adapter(nuisance)
        clean_distance = _chunked_cosine_distance(clean_adapted, bank, self.nn_chunk_size)
        nuisance_distance = _chunked_cosine_distance(nuisance_adapted, bank, self.nn_chunk_size)
        normal_consistency = (clean_distance - nuisance_distance).square().mean()

        margin_terms: list[torch.Tensor] = []
        skipped_types = 0
        for anomaly_identity, anomaly_nuisance, patch_mask, _name in views.anomalies:
            local_mask = patch_mask[patch_indices]
            if not bool(local_mask.any()):
                skipped_types += 1
                continue
            normal_identity_score = clean_distance[local_mask].mean()
            normal_nuisance_score = nuisance_distance[local_mask].mean()
            anomaly_identity_tokens = anomaly_identity[patch_indices][local_mask]
            anomaly_nuisance_tokens = anomaly_nuisance[patch_indices][local_mask]
            anomaly_identity_score = _chunked_cosine_distance(
                self.adapter(anomaly_identity_tokens), bank, self.nn_chunk_size
            ).mean()
            anomaly_nuisance_score = _chunked_cosine_distance(
                self.adapter(anomaly_nuisance_tokens), bank, self.nn_chunk_size
            ).mean()
            # Compare each synthetic score to its corresponding normal patch
            # score under the same identity/T pair, never to an unrelated
            # independently sampled clean subset.
            margin_terms.extend(
                [
                    F.relu(self.margin + normal_identity_score - anomaly_identity_score),
                    F.relu(self.margin + normal_nuisance_score - anomaly_nuisance_score),
                ]
            )
        if margin_terms:
            synthetic_margin = torch.stack(margin_terms).mean()
        else:  # The generator contract ensures this is not normally reached.
            synthetic_margin = clean_distance.new_zeros(())

        anchor_target = F.normalize(clean.detach(), dim=-1, eps=1e-8)
        feature_anchor = (1.0 - (clean_adapted * anchor_target).sum(dim=-1)).mean()
        total = (
            self.consistency_weight * normal_consistency
            + self.margin_weight * synthetic_margin
            + self.anchor_weight * feature_anchor
        )
        return total, {
            "normal_consistency": normal_consistency,
            "synthetic_margin": synthetic_margin,
            "feature_anchor": feature_anchor,
            "loss": total,
            "synthetic_skipped_types": clean_distance.new_tensor(float(skipped_types)),
        }

    def fit(self, support_paths: list[str], context: MethodContext) -> None:
        """Fit only on normal support paths and immutable object context."""

        self._reset_state()
        if not support_paths:
            raise ValueError("support paths must not be empty")
        if not isinstance(context, MethodContext):
            raise TypeError("context must be a MethodContext")
        paths = [str(path) for path in support_paths]
        if len(set(paths)) != len(paths):
            raise ValueError("support paths must be unique")
        self._support_paths = list(paths)
        self._context = context
        self._seed = int(context.seed)
        if not isinstance(context.masking, bool) or not isinstance(context.rotation, bool):
            raise TypeError("MethodContext.masking and rotation must be booleans")
        self._masking = context.masking
        self._rotation = context.rotation
        if self.revision == "v2":
            self._fit_v2(paths, context)
            return
        feature_layer_values = tuple(getattr(self.backbone, "feature_layers", ()))
        if feature_layer_values:
            self._selected_layer = int(feature_layer_values[-1])

        # Use per-source generators.  All views are descendants of their
        # source image and are made before feature extraction; no query path is
        # visible to this method.  Rotated descendants stay grouped under the
        # source so leave-one-source-out banks exclude all of them together.
        for source_index, path in enumerate(paths):
            image = _read_rgb(path)
            generator = np.random.default_rng(self._seed + 1009 * (source_index + 1))
            source_views = _augment_image(image) if self._rotation else [image]
            # One nuisance transform T is shared by x and every A(x), across
            # all rotated descendants of this source.  The identity pair is
            # represented by each clean/anomaly feature view itself.
            _, nuisance_transform = _photometric_pair(
                image, generator, self.photometric_strength
            )
            descendant_anomalies = [
                _synthetic_anomalies(clean_image, generator, self.synthetic_anomaly_types)
                for clean_image in source_views
            ]
            anomaly_names = [name for _, _, name in descendant_anomalies[0]]
            if len(anomaly_names) < 2:
                raise RuntimeError("PSMA_MR requires at least two synthetic anomaly families")
            if any(
                [name for _, _, name in anomalies] != anomaly_names
                for anomalies in descendant_anomalies
            ):
                raise RuntimeError("synthetic anomaly descendants changed family ordering")

            by_resolution: dict[int, _ResolutionViews] = {}
            for resolution in self.resolutions:
                clean_parts: list[torch.Tensor] = []
                nuisance_parts: list[torch.Tensor] = []
                foreground_parts: list[torch.Tensor] = []
                anomaly_identity_parts: list[list[torch.Tensor]] = [
                    [] for _ in anomaly_names
                ]
                anomaly_nuisance_parts: list[list[torch.Tensor]] = [
                    [] for _ in anomaly_names
                ]
                anomaly_mask_parts: list[list[torch.Tensor]] = [[] for _ in anomaly_names]
                grid_size: tuple[int, int] | None = None

                for clean_image, anomalies in zip(source_views, descendant_anomalies):
                    nuisance_image = _apply_photometric(clean_image, nuisance_transform)
                    clean_features, clean_grid = self._extract_final(clean_image, resolution)
                    nuisance_features, nuisance_grid = self._extract_final(
                        nuisance_image, resolution
                    )
                    if grid_size is None:
                        grid_size = clean_grid
                    if (
                        clean_grid != grid_size
                        or nuisance_grid != grid_size
                        or nuisance_features.shape[1] != clean_features.shape[1]
                    ):
                        raise ValueError("backbone feature grid/dimension changed across paired views")
                    clean_foreground = self._background_mask(clean_features, grid_size)
                    clean_parts.append(clean_features)
                    nuisance_parts.append(nuisance_features)
                    foreground_parts.append(clean_foreground)
                    for anomaly_index, (anomaly_image, pixel_mask, _name) in enumerate(anomalies):
                        anomaly_identity, anomaly_grid = self._extract_final(
                            anomaly_image, resolution
                        )
                        anomaly_nuisance, anomaly_nuisance_grid = self._extract_final(
                            _apply_photometric(anomaly_image, nuisance_transform), resolution
                        )
                        if (
                            anomaly_grid != grid_size
                            or anomaly_nuisance_grid != grid_size
                            or anomaly_identity.shape[1] != clean_features.shape[1]
                            or anomaly_nuisance.shape[1] != clean_features.shape[1]
                        ):
                            raise ValueError(
                                "backbone feature grid/dimension changed across synthetic views"
                            )
                        anomaly_identity_parts[anomaly_index].append(anomaly_identity)
                        anomaly_nuisance_parts[anomaly_index].append(anomaly_nuisance)
                        anomaly_mask_parts[anomaly_index].append(
                            _token_mask(pixel_mask, grid_size, anomaly_identity.shape[0])
                        )

                if grid_size is None:  # pragma: no cover - source_views is never empty
                    raise RuntimeError("no support descendants were generated")
                clean_features = torch.cat(clean_parts, dim=0)
                nuisance_features = torch.cat(nuisance_parts, dim=0)
                foreground = torch.cat(foreground_parts, dim=0)
                anomaly_records_raw = [
                    (
                        torch.cat(identity_parts, dim=0),
                        torch.cat(nuisance_feature_parts, dim=0),
                        torch.cat(mask_parts, dim=0),
                        name,
                    )
                    for identity_parts, nuisance_feature_parts, mask_parts, name in zip(
                        anomaly_identity_parts,
                        anomaly_nuisance_parts,
                        anomaly_mask_parts,
                        anomaly_names,
                    )
                ]
                # Keep support-side caches bounded *after* all descendants are
                # grouped.  One shared index set preserves clean/nuisance,
                # identity/T synthetic pairs, and one visible token per
                # anomaly family whenever the cache budget allows it.
                patch_indices, skipped_types = _sample_indices_preserving_masks(
                    clean_features.shape[0],
                    self.max_patches_per_source,
                    self._seed + 8191 * (source_index + 1) + resolution,
                    [mask for _, _, mask, _ in anomaly_records_raw],
                    eligible=foreground,
                )
                self._skipped_synthetic_types += skipped_types
                clean_features = clean_features[patch_indices]
                nuisance_features = nuisance_features[patch_indices]
                anomaly_records = tuple(
                    (
                        identity_features[patch_indices],
                        nuisance_feature[patch_indices],
                        mask[patch_indices],
                        name,
                    )
                    for identity_features, nuisance_feature, mask, name in anomaly_records_raw
                )
                by_resolution[resolution] = _ResolutionViews(
                    clean=clean_features,
                    nuisance=nuisance_features,
                    anomalies=anomaly_records,
                    grid_size=grid_size,
                )
                self._grid_sizes[resolution] = grid_size
            self._records.append(by_resolution)
            self.log_training(
                step=source_index + 1,
                stage="support_features",
                support_total=len(paths),
                source_index=source_index,
                resolutions=list(self.resolutions),
                synthetic_types=list(self.synthetic_anomaly_types),
                rotation=self._rotation,
                masking=self._masking,
                descendant_count=len(source_views),
                patch_cache_limit=self.max_patches_per_source,
                skipped_synthetic_types=skipped_types,
            )

        self._feature_dim = int(self._records[0][self.resolutions[0]].clean.shape[1])
        if any(
            int(record[resolution].clean.shape[1]) != self._feature_dim
            for record in self._records
            for resolution in self.resolutions
        ):
            raise ValueError("backbone feature dimension changed across support images/resolutions")
        self.adapter = _Rank4Residual(
            self._feature_dim,
            rank=self.rank,
            seed=self._seed,
            enabled=self.adapter_enabled,
        )
        if not self.adapter_enabled:
            for parameter in self.adapter.parameters():
                parameter.requires_grad_(False)
        self._build_loo_banks()

        if self.adapter_enabled and self.steps > 0:
            optimizer = torch.optim.Adam(
                self.adapter.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
            for step in range(1, self.steps + 1):
                source_index = (step - 1) % len(self._records)
                resolution = self.resolutions[(step - 1) % len(self.resolutions)]
                optimizer.zero_grad(set_to_none=True)
                loss, components = self._training_step(source_index, resolution, step)
                loss.backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), self.gradient_clip)
                optimizer.step()
                self._executed_steps = step
                self._last_losses = {
                    name: float(value.detach().cpu().item()) for name, value in components.items()
                }
                self.log_training(
                    step=step,
                    stage="adapter_step",
                    source_index=source_index,
                    resolution=resolution,
                    adapter_enabled=True,
                    **self._last_losses,
                )
        else:
            self.log_training(
                step=0,
                stage="adapter_disabled" if not self.adapter_enabled else "adapter_no_steps",
                adapter_enabled=self.adapter_enabled,
                loss=0.0,
                normal_consistency=0.0,
                synthetic_margin=0.0,
                feature_anchor=0.0,
            )

        if self.adapter is None:
            raise RuntimeError("adapter initialization failed")
        self.adapter.eval()
        # Final inference banks contain only clean normal support features.
        # Adapter outputs are materialized once, so query scoring is read-only.
        with torch.inference_mode():
            for resolution in self.resolutions:
                pieces = [
                    _sample_rows(
                        record[resolution].clean,
                        self.max_bank_patches,
                        (self._seed or 0) + 65537 * (index + 1) + resolution,
                    )
                    for index, record in enumerate(self._records)
                ]
                raw_bank = torch.cat(pieces, dim=0)
                self._banks[resolution] = self.adapter(raw_bank).contiguous()
        self._fit_complete = True
        self.log_training(
            stage="memory_bank_ready",
            bank_vectors={
                str(resolution): int(bank.shape[0]) for resolution, bank in self._banks.items()
            },
            leave_one_source_out=True,
            adapter_enabled=self.adapter_enabled,
        )

    def predict(self, test_image_path: str) -> tuple[np.ndarray, float]:
        """Score one query using per-resolution clean support banks."""

        if self.revision == "v2":
            return self._predict_v2(test_image_path)
        if not self._fit_complete or self.adapter is None:
            raise RuntimeError("fit must complete before prediction")
        image = _read_rgb(test_image_path)
        maps: list[np.ndarray] = []
        self.adapter.eval()
        with torch.inference_mode():
            for resolution in self.resolutions:
                features, grid_size = self._extract_final(image, resolution)
                if features.shape[1] != self._feature_dim:
                    raise ValueError(
                        "query feature dimension does not match fitted support features"
                    )
                foreground = self._background_mask(features, grid_size)
                distances = _chunked_cosine_distance(
                    self.adapter(features), self._banks[resolution], self.nn_chunk_size
                )
                # Match SuperAD's masking semantics: score foreground patches
                # and leave background map values at exactly zero.
                distances = torch.where(foreground, distances, torch.zeros_like(distances))
                if distances.numel() != grid_size[0] * grid_size[1]:
                    grid_size = _infer_grid(grid_size, int(distances.numel()))
                feature_map = distances.reshape(grid_size).detach().cpu().numpy().astype(np.float32)
                if cv2 is not None:
                    resized = cv2.resize(
                        feature_map,
                        (int(image.shape[1]), int(image.shape[0])),
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:  # pragma: no cover - OpenCV is present in the project environment
                    resized = np.asarray(
                        Image.fromarray(feature_map).resize(
                            (int(image.shape[1]), int(image.shape[0])), Image.Resampling.BILINEAR
                        )
                    )
                maps.append(np.asarray(resized, dtype=np.float32))
        anomaly_map = np.mean(np.stack(maps, axis=0), axis=0).astype(np.float32)
        return anomaly_map, _mean_top1p(anomaly_map)

    def _v2_metadata(self) -> dict:
        """Return compact v2 provenance without evaluator-owned fields."""

        branch_metadata: dict[str, dict[str, Any]] = {}
        loo_sources: dict[str, list[int]] = {}
        for branch_index, layer in enumerate(self.v2_feature_layers):
            cap = self._v2_branch_bank_caps.get(layer)
            for resolution in self.resolutions:
                key = (layer, resolution)
                bank = self._v2_banks.get(key)
                source_ids = self._v2_bank_source_ids.get(key, torch.empty(0, dtype=torch.long))
                descendant_ids = self._v2_bank_descendant_ids.get(
                    key, torch.empty(0, dtype=torch.long)
                )
                branch_metadata[f"layer{layer}:resolution{resolution}"] = {
                    "layer": layer,
                    "resolution": resolution,
                    "frozen": layer != self._v2_final_layer,
                    "adapter": layer == self._v2_final_layer,
                    "source_cap": cap,
                    "bank_vectors": int(bank.shape[0]) if bank is not None else 0,
                    "source_ids": sorted(set(int(value) for value in source_ids.tolist())),
                    "descendant_ids": sorted(
                        set(int(value) for value in descendant_ids.tolist())
                    ),
                    "score": "raw_cosine_nn_then_affine_calibration",
                }
                for source_index in range(len(self._v2_records)):
                    loo_key = (source_index, layer, resolution)
                    loo_ids = self._v2_loo_bank_source_ids.get(
                        loo_key, torch.empty(0, dtype=torch.long)
                    )
                    loo_sources[f"source{source_index}:layer{layer}:resolution{resolution}"] = sorted(
                        set(int(value) for value in loo_ids.tolist())
                    )
        return {
            "method_class": type(self).__name__,
            "method": "PSMA_MR",
            "revision": "v2",
            "selected_reference_count": len(self._support_paths),
            "selected_reference_paths": list(self._support_paths),
            "resolutions": list(self.resolutions),
            "feature_layers": list(self.v2_feature_layers),
            "selected_layer": self._v2_final_layer,
            "feature_dim": self._feature_dim,
            "backbone_frozen": True,
            "intermediate_features_frozen": True,
            "adapter_enabled": self.adapter_enabled,
            "adapter_rank": 4,
            "adapter": {
                "trainable_parameters": ["adapter.U", "adapter.V"]
                if self.adapter_enabled
                else [],
                "formula": "normalize(z + clip(U*tanh(V*LN(z)), 0.1*||z||))",
                "residual_epsilon": self.residual_epsilon,
                "U_initialization": "zero",
                "V_initialization": "seeded_random",
                "zero_safe": True,
            },
            "trainable_parameters": ["adapter.U", "adapter.V"]
            if self.adapter_enabled
            else [],
            "seed": self._seed,
            "context": _json_safe(self._context.__dict__) if self._context is not None else None,
            "masking": self._masking,
            "rotation": self._rotation,
            "steps": self.steps,
            "executed_steps": self._executed_steps,
            "config": self._exact_config,
            "method_params": _json_safe(self.params),
            "training": {
                "seed": self._seed,
                "steps": self.steps,
                "executed_steps": self._executed_steps,
                "schedule": list(self._v2_schedule),
                "schedule_rule": "t0..31: source=floor(t/2)%support_count, resolution=res[t%2]",
                "schedule_full_cartesian": (
                    len(self._v2_schedule) == 32
                    and len(self._v2_records) == 16
                    and {entry["source_index"] for entry in self._v2_schedule}
                    == set(range(16))
                    and {entry["resolution"] for entry in self._v2_schedule}
                    == set(self.resolutions)
                ),
                "config": self._exact_config,
                "synthetic_anomaly_types": list(self.synthetic_anomaly_types),
                "synthetic_cache": "final_only",
                "synthetic_pairing": {
                    "identity_included": True,
                    "shared_photometric_transform": True,
                    "corresponding_normal_patch_margin": True,
                    "aligned_positions_before_hinge": True,
                },
                "rotation_descendants_per_source": 8 if self._rotation else 1,
                "masking": self._masking,
                "rotation": self._rotation,
                "skipped_synthetic_types": self._skipped_synthetic_types,
                "spatial_stratified": True,
                "preserves_grid_coordinates": True,
                "train_tokens": self.v2_train_tokens,
                "last_losses": dict(self._last_losses),
                "adapter_fit_support_count": len(self._v2_records),
                "adapter_fit_holdout": "nonstrict_support16",
                "leave_one_source_out": True,
                "loo_bank_sources": loo_sources,
            },
            "banks": branch_metadata,
            "bank_vectors": {
                key: value["bank_vectors"] for key, value in branch_metadata.items()
            },
            "bank_caps": {
                "final": self.final_bank_cap,
                "intermediate": self.intermediate_bank_cap,
            },
            "calibration": {
                "formula": "(d-median)/max(1.4826*MAD,0.001)",
                "clipping": False,
                "fitted_once_after_adapter": self._v2_adapter_fit_complete,
                "clean_normal_only": True,
                "query_used": False,
                "ground_truth_used": False,
                "source_exclusion": "LOO15" if len(self._v2_records) == 16 else "LOO",
                "inference_bank": "all_support16"
                if len(self._v2_records) == 16
                else "all_support",
                "branches": {
                    f"layer{layer}:resolution{resolution}": dict(
                        self._v2_calibration.get((layer, resolution), {})
                    )
                    for layer in self.v2_feature_layers
                    for resolution in self.resolutions
                },
            },
            "fusion": {
                "branch_count": len(self.v2_feature_layers) * len(self.resolutions),
                "aggregation": "equal_mean",
                "maps": "resized_to_query_native_size",
                "image_score": "top1_percent_mean",
            },
        }

    def object_metadata(self) -> dict:
        """Return reproducible method provenance without evaluator-owned fields."""

        if self.revision == "v2":
            return self._v2_metadata()

        return {
            "method_class": type(self).__name__,
            "method": "PSMA_MR",
            "revision": "v1",
            "selected_reference_count": len(self._support_paths),
            "selected_reference_paths": list(self._support_paths),
            "resolutions": list(self.resolutions),
            "feature_layers": [self._selected_layer] if self._selected_layer is not None else [],
            "feature_dim": self._feature_dim,
            "backbone_frozen": True,
            "adapter_enabled": self.adapter_enabled,
            "adapter_rank": self.rank,
            "trainable_parameters": ["adapter.U", "adapter.V"]
            if self.adapter_enabled
            else [],
            "seed": self._seed,
            "context": _json_safe(self._context.__dict__) if self._context is not None else None,
            "masking": self._masking,
            "rotation": self._rotation,
            "steps": self.steps,
            "executed_steps": self._executed_steps,
            "config": self._exact_config,
            "method_params": _json_safe(self.params),
            "synthetic_pairing": {
                "identity_included": True,
                "shared_photometric_transform": True,
                "corresponding_normal_patch_margin": True,
            },
            "training": {
                "seed": self._seed,
                "steps": self.steps,
                "executed_steps": self._executed_steps,
                "config": self._exact_config,
                "synthetic_anomaly_types": list(self.synthetic_anomaly_types),
                "synthetic_pairing": {
                    "identity_included": True,
                    "shared_photometric_transform": True,
                    "corresponding_normal_patch_margin": True,
                },
                "rotation_descendants_per_source": 8 if self._rotation else 1,
                "masking": self._masking,
                "rotation": self._rotation,
                "skipped_synthetic_types": self._skipped_synthetic_types,
                "leave_one_source_out": True,
                "loo_bank_sources": {
                    f"source{source}:resolution{resolution}": list(sources)
                    for (source, resolution), sources in self._loo_bank_sources.items()
                },
                "last_losses": dict(self._last_losses),
            },
            "bank_vectors": {
                str(resolution): int(bank.shape[0]) for resolution, bank in self._banks.items()
            },
        }

    def save_checkpoint(self, directory: Path) -> None:
        """Persist the fitted adapter and clean support banks, not backbone weights."""

        if not self._fit_complete or self.adapter is None:
            raise RuntimeError("fit must complete before checkpoint saving")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 2 if self.revision == "v2" else 1,
                "method_class": type(self).__name__,
                "revision": self.revision,
                "adapter_state_dict": self.adapter.state_dict(),
            },
            directory / "adapter.pt",
        )
        bank_files: dict[str, str] = {}
        if self.revision == "v2":
            for (layer, resolution), bank in self._v2_banks.items():
                filename = f"bank_layer_{layer}_resolution_{resolution}.npy"
                np.save(directory / filename, bank.detach().cpu().numpy().astype(np.float32))
                bank_files[f"layer{layer}:resolution{resolution}"] = filename
        else:
            for resolution, bank in self._banks.items():
                filename = f"bank_resolution_{resolution}.npy"
                np.save(directory / filename, bank.detach().cpu().numpy().astype(np.float32))
                bank_files[str(resolution)] = filename
        metadata = dict(self.object_metadata())
        metadata["checkpoint"] = {
            "adapter": "adapter.pt",
            "banks": bank_files,
            "backbone_weights_included": False,
        }
        (directory / "metadata.json").write_text(
            json.dumps(_json_safe(metadata), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
