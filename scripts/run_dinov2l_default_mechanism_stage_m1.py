#!/usr/bin/env python3
"""Run the frozen DINOv2-L Basic-default Stage-M1 mechanism audit.

Only paths listed in the exact SuperAD-16 ``train/good`` manifest can be read.
The runner has no dataset enumeration code and implements static scoring only:
there is no TTA, query update, memory expansion, or public/private-test access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "src", Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_cfntr_stage_minus1 as synthetic  # noqa: E402
from flow_tte.cfntr import (  # noqa: E402
    RGBTransformSpec,
    apply_rgb_transform,
    make_axis_transform,
)
from flow_tte.config import FlowConfig  # noqa: E402
from flow_tte.denoising import PositionMeanArtifactDenoiser  # noqa: E402
from flow_tte.dinov3_early_exit import patch_torch_pytree_compat  # noqa: E402
from flow_tte.mechanism_diagnostics import (  # noqa: E402
    assert_static_memory,
    cosine_displacement,
    derive_logdet,
    dvt_variance_removed,
    effective_rank,
    neighbor_diagnostics,
)
from flow_tte.memory import TorchMemoryBank  # noqa: E402
from flow_tte.trainer import FlowDensityEstimator  # noqa: E402
from fmad.backbones.dinov2 import DINOv2Backbone  # noqa: E402
from flow_tte_support import merge_layer_features  # noqa: E402

# The fixed dsba3 container pairs PyTorch 2.1 with a newer transformers
# package.  AdamW imports torch.onnx, which reaches the transformer's public
# PyTree registration API; install the repository's audited forward-compatible
# alias before any Flow optimizer is constructed.
patch_torch_pytree_compat()

PROTOCOL_ID = "dinov2l-default-mechanism-stage-m1-20260719"
ROW_SCHEMA = "dinov2l-default-mechanism-m1-row-v1"
SUMMARY_SCHEMA = "dinov2l-default-mechanism-m1-summary-v1"
RUN_SCHEMA = "dinov2l-default-mechanism-m1-run-v1"
REQUIRED_SUPPORT_SHA256 = "7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3"
BACKBONE_MODEL = "dinov2_vitl14"
LAYERS = (5, 11, 17, 23)
OBJECTS = synthetic.OBJECTS
AXES = ("exposure", "gamma", "contrast", "white_balance")
SIGNS = (-1, 1)
MAGNITUDES = (1, 2)
DEFECTS = ("cutpaste", "chromatic_blotch", "gaussian_blur", "smooth_local_warp")
FOLD_COUNT = 4
SUPPORT_COUNT = 16
FIT_COUNT = 12
QUERY_CHUNK_SIZE = 512
GEOMETRY_SAMPLE_ROWS = 64
FIXTURE_PARITY_ATOL = 1e-6
CUDA_CDIST_PARITY_ATOL = 1e-4
RESOLUTIONS = {name: (448 if name == "sheet_metal" else 672) for name in OBJECTS}


@dataclass(frozen=True)
class ArmSpec:
    state: str
    query_dvt: bool
    density_weight: float
    layer: int | None = None


@dataclass
class PipelineState:
    name: str
    estimator: FlowDensityEstimator
    bank: TorchMemoryBank
    bank_snapshot: torch.Tensor
    source_count: int
    patches_per_source: int
    grid: tuple[int, int]


@dataclass(frozen=True)
class ScoreOutput:
    scores: np.ndarray
    indices: np.ndarray
    distances: np.ndarray
    distance_scores: np.ndarray
    density: np.ndarray
    raw_features: np.ndarray
    latents: np.ndarray
    nll: np.ndarray


def arm_specs() -> dict[str, ArmSpec]:
    """Return the frozen arms mapped onto four fitted fused states."""
    arms = {
        "A0": ArmSpec("dvt_flow", True, 0.25),
        "A1": ArmSpec("raw_flow", False, 0.25),
        "A2": ArmSpec("dvt_flow", False, 0.25),
        "A3": ArmSpec("raw_flow", True, 0.25),
        "A4": ArmSpec("dvt_flow", True, 0.0),
        "A5": ArmSpec("dvt_identity", True, 0.0),
        "A6": ArmSpec("raw_flow", False, 0.0),
        "A7": ArmSpec("raw_identity", False, 0.0),
    }
    arms.update({f"L{layer}": ArmSpec(f"layer_{layer}_identity", True, 0.0, layer) for layer in LAYERS})
    return arms


def shift_specs() -> tuple[tuple[str, int, int, RGBTransformSpec], ...]:
    """Return all 4 axes x 2 signs x 2 frozen nonzero magnitudes."""
    return tuple(
        (axis, sign, magnitude, make_axis_transform(axis, sign=sign, amplitude_scale=magnitude))
        for axis in AXES
        for sign in SIGNS
        for magnitude in MAGNITUDES
    )


def expected_row_counts(paths: int = SUPPORT_COUNT) -> dict[str, int]:
    arms = len(arm_specs())
    shifts = len(shift_specs())
    defects = len(DEFECTS)
    by_condition = {
        "C": paths * arms,
        "S": paths * shifts * arms,
        "D": paths * defects * arms,
        "SD": paths * shifts * defects * arms,
    }
    return {**by_condition, "total": sum(by_condition.values())}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def clean_source_with_metadata(
    clean_images: Mapping[str, np.ndarray],
    relative_path: str,
) -> tuple[np.ndarray, dict[str, str]]:
    """Select a clean source and bind its custody hash to the same array."""
    image = clean_images[relative_path]
    return image, {"evaluated_image_sha256": sha256_bytes(image.tobytes())}


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in (PROTOCOL_ID, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _split_words(value: str) -> tuple[str, ...]:
    return tuple(part for part in value.replace(",", " ").split() if part)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("real", "smoke"), default="real")
    parser.add_argument("--data-root")
    parser.add_argument("--support-json")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host-gpu-id", required=True)
    parser.add_argument("--container-cuda-slot", default="0")
    args = parser.parse_args(list(argv))
    args.objects = _split_words(args.objects)
    if not args.objects or len(set(args.objects)) != len(args.objects):
        parser.error("--objects must contain unique object names")
    if any(name not in OBJECTS for name in args.objects):
        parser.error(f"--objects must be a subset of {OBJECTS}")
    if args.mode == "real" and (not args.data_root or not args.support_json):
        parser.error("real mode requires --data-root and --support-json")
    if args.mode == "smoke" and args.device != "cpu":
        parser.error("smoke mode is synthetic-only and freezes --device=cpu")
    return args


def _prepare_output_root(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise FileExistsError(f"output root must be new or empty: {path}")
    else:
        path.mkdir(parents=True)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return count


def make_smooth_local_warp(
    image_rgb: np.ndarray,
    *,
    object_name: str,
    relative_path: str,
    feature_retry: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply a deterministic smooth displacement inside one elliptical region."""
    image = np.asarray(image_rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 32:
        raise ValueError("image_rgb must be uint8 HxWx3 with both dimensions >= 32")
    seed = stable_seed(object_name, relative_path, "smooth_local_warp", feature_retry)
    rng = np.random.default_rng(seed)
    height, width = image.shape[:2]
    center_y = int(rng.integers(height // 4, max(height // 4 + 1, 3 * height // 4)))
    center_x = int(rng.integers(width // 4, max(width // 4 + 1, 3 * width // 4)))
    radius_y = max(8, int(height * rng.uniform(0.08, 0.16)))
    radius_x = max(8, int(width * rng.uniform(0.08, 0.16)))
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    radial = ((yy - center_y) / radius_y) ** 2 + ((xx - center_x) / radius_x) ** 2
    support = radial <= 1.0
    taper = np.zeros((height, width), dtype=np.float32)
    taper[support] = np.cos(np.pi * np.sqrt(radial[support]) / 2.0) ** 2
    sigma = max(3.0, min(height, width) * 0.025)
    dy = gaussian_filter(rng.standard_normal((height, width)), sigma=sigma, mode="reflect")
    dx = gaussian_filter(rng.standard_normal((height, width)), sigma=sigma, mode="reflect")
    scale = max(2.0, min(height, width) * 0.025)
    dy = dy / max(float(np.std(dy)), 1e-6) * scale * taper
    dx = dx / max(float(np.std(dx)), 1e-6) * scale * taper
    warped = cv2.remap(
        image,
        (xx + dx).astype(np.float32),
        (yy + dy).astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    changed = np.any(warped != image, axis=2) & support
    if not bool(changed.any()):
        raise RuntimeError("smooth_local_warp generated a pixel-level no-op")
    mask = support & (taper > 0.05)
    metadata = {
        "kind": "smooth_local_warp",
        "seed": seed,
        "feature_retry": feature_retry,
        "generation_attempt": 0,
        "center": [center_y, center_x],
        "radii": [radius_y, radius_x],
        "displacement_scale": scale,
        "mask_pixels": int(mask.sum()),
        "image_sha256": sha256_bytes(warped.tobytes()),
        "mask_sha256": sha256_bytes(mask.astype(np.uint8).tobytes()),
    }
    return warped, mask, metadata


def make_defect(
    image_rgb: np.ndarray,
    *,
    object_name: str,
    relative_path: str,
    kind: str,
    feature_retry: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if kind == "smooth_local_warp":
        return make_smooth_local_warp(
            image_rgb,
            object_name=object_name,
            relative_path=relative_path,
            feature_retry=feature_retry,
        )
    if kind not in DEFECTS[:3]:
        raise ValueError(f"unknown defect kind: {kind}")
    return synthetic.make_synthetic_defect(
        image_rgb,
        object_name=object_name,
        relative_path=relative_path,
        kind=kind,
        feature_retry=feature_retry,
    )


def _flow_config(transform_mode: str, *, smoke: bool = False) -> FlowConfig:
    return FlowConfig(
        n_coupling_layers=2,
        hidden_multiplier=1,
        clamp=1.9,
        transform_mode=transform_mode,
        condition_mode="none",
        n_epochs=3,
        lr=2e-4,
        batch_size=64 if smoke else 512,
        seed=0,
        tail_weight=0.3,
        tail_top_k_ratio=0.05,
        lambda_logdet=1e-3,
        standardize=True,
    )


def _fit_state(
    name: str,
    maps: Sequence[np.ndarray],
    *,
    transform_mode: str,
    device: str,
    smoke: bool = False,
) -> PipelineState:
    if not maps:
        raise ValueError("state fitting requires support maps")
    shape = maps[0].shape
    if len(shape) != 3 or any(value.shape != shape for value in maps):
        raise ValueError("state support maps must share HxWxD shape")
    support = np.stack(maps).astype(np.float32, copy=False)
    estimator = FlowDensityEstimator(
        dim=shape[-1],
        config=_flow_config(transform_mode, smoke=smoke),
        device=device,
    )
    estimator.fit(support, density_quantile=0.9)
    evaluation = estimator.evaluate(support)
    bank = TorchMemoryBank()
    bank.fit(evaluation.z)
    snapshot = bank.features.detach().clone()
    return PipelineState(
        name=name,
        estimator=estimator,
        bank=bank,
        bank_snapshot=snapshot,
        source_count=len(maps),
        patches_per_source=shape[0] * shape[1],
        grid=(shape[0], shape[1]),
    )


def _score(
    state: PipelineState,
    feature_map: np.ndarray,
    *,
    density_weight: float,
    exclude_source: int | None = None,
) -> ScoreOutput:
    values = np.asarray(feature_map, dtype=np.float32)
    if values.ndim != 3 or values.shape[:2] != state.grid:
        raise ValueError("query feature map does not match fitted state grid")
    evaluation = state.estimator.evaluate(values[np.newaxis])
    bank = state.bank
    if exclude_source is not None:
        if not 0 <= exclude_source < state.source_count:
            raise ValueError("exclude_source is outside the fitted support bank")
        keep = torch.ones(bank.features.shape[0], dtype=torch.bool, device=bank.features.device)
        start = exclude_source * state.patches_per_source
        keep[start : start + state.patches_per_source] = False
        bank = TorchMemoryBank()
        bank.fit(state.bank.features[keep])
    nearest = bank.query(evaluation.z, k=2, chunk_size=QUERY_CHUNK_SIZE, squared=False)
    distance_scores = nearest.distances[:, 0]
    density = state.estimator.density_penalty(evaluation.nll)
    scores = distance_scores + density_weight * density
    assert_static_memory(state.bank_snapshot, state.bank.features)
    return ScoreOutput(
        scores=scores.detach().cpu().numpy().astype(np.float32, copy=False),
        indices=nearest.indices.detach().cpu().numpy().astype(np.int64, copy=False),
        distances=nearest.distances.detach().cpu().numpy().astype(np.float32, copy=False),
        distance_scores=distance_scores.detach().cpu().numpy().astype(np.float32, copy=False),
        density=density.detach().cpu().numpy().astype(np.float32, copy=False),
        raw_features=values.reshape(-1, values.shape[-1]),
        latents=evaluation.z.detach().cpu().numpy().astype(np.float32, copy=False),
        nll=evaluation.nll.detach().cpu().numpy().astype(np.float32, copy=False),
    )


def m0_parity_assertion(
    state: PipelineState,
    feature_map: np.ndarray,
    *,
    density_weight: float,
    atol: float = FIXTURE_PARITY_ATOL,
) -> tuple[float, int]:
    """Assert static scorer parity against an independent full-cdist reference."""
    result = _score(state, feature_map, density_weight=density_weight)
    # Eight rows are sufficient to prove the score/NN implementation contract
    # without allocating the full query-by-memory distance matrix at 672 px.
    sample_rows = min(8, result.latents.shape[0])
    query = torch.as_tensor(result.latents[:sample_rows], device=state.bank.features.device)
    distances = torch.cdist(query, state.bank.features, p=2.0)
    reference = torch.topk(distances, k=2, largest=False, sorted=True, dim=1)
    reference_scores = (
        reference.values[:, 0].detach().cpu().numpy()
        + density_weight * result.density[:sample_rows]
    )
    error = float(np.max(np.abs(reference_scores - result.scores[:sample_rows])))
    mismatch = int(
        np.sum(reference.indices.detach().cpu().numpy() != result.indices[:sample_rows])
    )
    if error > atol or mismatch != 0:
        raise RuntimeError(
            f"M0 score/NN parity failed: error={error}, atol={atol}, mismatch={mismatch}"
        )
    return error, mismatch


def _fixed_f1(mask: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    labels = np.asarray(mask, dtype=bool).reshape(-1)
    predicted = np.asarray(scores).reshape(-1) >= threshold
    tp = int(np.sum(predicted & labels))
    fp = int(np.sum(predicted & ~labels))
    fn = int(np.sum(~predicted & labels))
    return float((2 * tp) / max(2 * tp + fp + fn, 1))


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else 0.0


def _geometry(result: ScoreOutput, *, seed: int) -> dict[str, float]:
    count = min(GEOMETRY_SAMPLE_ROWS, result.raw_features.shape[0])
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(result.raw_features.shape[0], size=count, replace=False))
    raw = result.raw_features[indices]
    latent = result.latents[indices]
    upper = np.triu_indices(count, k=1)
    raw_dist = np.linalg.norm(raw[:, None] - raw[None, :], axis=2)[upper]
    latent_dist = np.linalg.norm(latent[:, None] - latent[None, :], axis=2)[upper]
    rho = float(spearmanr(raw_dist, latent_dist).statistic)
    if not np.isfinite(rho):
        rho = 1.0 if np.array_equal(raw_dist, latent_dist) else 0.0
    logdet = derive_logdet(result.latents[indices], result.nll[indices])
    return {
        "pair_distance_spearman": rho,
        "effective_rank": float(effective_rank(latent)),
        "latent_rms": float(np.sqrt(np.mean(np.square(latent)))),
        "logdet_q05": float(np.quantile(logdet, 0.05)),
        "logdet_q50": float(np.quantile(logdet, 0.50)),
        "logdet_q95": float(np.quantile(logdet, 0.95)),
        "distance_density_correlation": _corr(result.distance_scores, result.density),
    }


def _row_metrics(
    result: ScoreOutput,
    baseline: ScoreOutput,
    *,
    state: PipelineState,
    mask: np.ndarray | None,
    threshold: float,
    seed: int,
) -> dict[str, float | None]:
    neighbor = neighbor_diagnostics(
        baseline.indices,
        baseline.distances,
        result.indices,
        result.distances,
        patches_per_source=state.patches_per_source,
        grid=state.grid,
        source_count=state.source_count,
    )
    values: dict[str, float | None] = {
        "score_q95": float(np.quantile(result.scores, 0.95)),
        "score_q99": float(np.quantile(result.scores, 0.99)),
        "score_q995": float(np.quantile(result.scores, 0.995)),
        "inside_median": None,
        "outside_median": None,
        "defect_contrast": None,
        "seg_auroc_0_05": None,
        "fixed_f1": None,
        "nn_identity_retention": neighbor.identity_retention,
        "nn_margin_mean": neighbor.shifted_margin_mean,
        "nn_source_entropy": neighbor.source_entropy,
        "nn_position_displacement_mean": neighbor.normalized_position_displacement,
        "memory_gini": neighbor.source_gini,
    }
    values.update(_geometry(result, seed=seed))
    if mask is not None:
        patch_mask = synthetic.mask_to_patch_grid(mask, state.grid)
        if not bool(patch_mask.any()) or bool(patch_mask.all()):
            raise RuntimeError("synthetic mask is invalid on the feature grid")
        inside = result.scores[patch_mask]
        outside = result.scores[~patch_mask]
        values.update(
            {
                "inside_median": float(np.median(inside)),
                "outside_median": float(np.median(outside)),
                "defect_contrast": float(np.median(inside) - np.median(outside)),
                "seg_auroc_0_05": float(
                    roc_auc_score(patch_mask.astype(np.uint8), result.scores, max_fpr=0.05)
                ),
                "fixed_f1": _fixed_f1(patch_mask, result.scores, threshold),
            }
        )
    return values


def _extract_maps(
    backbone: DINOv2Backbone,
    image_rgb: np.ndarray,
) -> tuple[dict[int, np.ndarray], np.ndarray, tuple[int, int]]:
    layers, grid = synthetic.extract_layers(backbone, image_rgb)
    layer_maps = {layer: values.reshape(grid[0], grid[1], -1) for layer, values in layers.items()}
    fused = merge_layer_features([layers[layer] for layer in LAYERS], "layer_norm_mean")
    return layer_maps, fused.reshape(grid[0], grid[1], -1), grid


def _apply_dvt(
    fused: np.ndarray,
    layers: Mapping[int, np.ndarray],
    *,
    fused_dvt: PositionMeanArtifactDenoiser,
    layer_dvt: Mapping[int, PositionMeanArtifactDenoiser],
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    return fused_dvt.transform(fused), {layer: layer_dvt[layer].transform(layers[layer]) for layer in LAYERS}


def _fit_fold(
    clean: Mapping[str, tuple[dict[int, np.ndarray], np.ndarray]],
    entries: Sequence[synthetic.SupportEntry],
    fold: int,
    *,
    device: str,
    smoke: bool = False,
) -> tuple[
    dict[str, PipelineState],
    PositionMeanArtifactDenoiser,
    dict[int, PositionMeanArtifactDenoiser],
    dict[str, float],
]:
    fit_entries = [entry for entry in entries if entry.fold != fold]
    if len(fit_entries) != FIT_COUNT and not smoke:
        raise RuntimeError(f"fold {fold}: expected {FIT_COUNT} fit sources")
    raw_fused = [clean[entry.relative_path][1] for entry in fit_entries]
    fused_dvt = PositionMeanArtifactDenoiser.fit(raw_fused, alpha=1.0)
    dvt_fused = [fused_dvt.transform(value) for value in raw_fused]
    layer_dvt = {
        layer: PositionMeanArtifactDenoiser.fit(
            [clean[entry.relative_path][0][layer] for entry in fit_entries], alpha=1.0
        )
        for layer in LAYERS
    }
    states = {
        "dvt_flow": _fit_state("dvt_flow", dvt_fused, transform_mode="flow", device=device, smoke=smoke),
        "raw_flow": _fit_state("raw_flow", raw_fused, transform_mode="flow", device=device, smoke=smoke),
        "dvt_identity": _fit_state(
            "dvt_identity", dvt_fused, transform_mode="identity", device=device, smoke=smoke
        ),
        "raw_identity": _fit_state(
            "raw_identity", raw_fused, transform_mode="identity", device=device, smoke=smoke
        ),
    }
    for layer in LAYERS:
        values = [
            layer_dvt[layer].transform(clean[entry.relative_path][0][layer]) for entry in fit_entries
        ]
        states[f"layer_{layer}_identity"] = _fit_state(
            f"layer_{layer}_identity", values, transform_mode="identity", device=device, smoke=smoke
        )
    dvt_stats = {
        "fused_field_energy": float(np.mean(np.square(fused_dvt.artifact))),
        "fused_variance_removed": dvt_variance_removed(raw_fused, dvt_fused),
    }
    for layer in LAYERS:
        raw = [clean[entry.relative_path][0][layer] for entry in fit_entries]
        denoised = [layer_dvt[layer].transform(value) for value in raw]
        dvt_stats[f"layer_{layer}_field_energy"] = float(
            np.mean(np.square(layer_dvt[layer].artifact))
        )
        dvt_stats[f"layer_{layer}_variance_removed"] = dvt_variance_removed(raw, denoised)
    return states, fused_dvt, layer_dvt, dvt_stats


def _arm_map(
    spec: ArmSpec,
    raw_layers: Mapping[int, np.ndarray],
    raw_fused: np.ndarray,
    dvt_layers: Mapping[int, np.ndarray],
    dvt_fused: np.ndarray,
) -> np.ndarray:
    if spec.layer is not None:
        return dvt_layers[spec.layer] if spec.query_dvt else raw_layers[spec.layer]
    return dvt_fused if spec.query_dvt else raw_fused


def _fit_thresholds(
    states: Mapping[str, PipelineState],
    arms: Mapping[str, ArmSpec],
    clean: Mapping[str, tuple[dict[int, np.ndarray], np.ndarray]],
    fit_entries: Sequence[synthetic.SupportEntry],
    fused_dvt: PositionMeanArtifactDenoiser,
    layer_dvt: Mapping[int, PositionMeanArtifactDenoiser],
) -> dict[str, float]:
    values: dict[str, list[np.ndarray]] = {arm: [] for arm in arms}
    for source_index, entry in enumerate(fit_entries):
        raw_layers, raw_fused = clean[entry.relative_path]
        dvt_fused, dvt_layers = _apply_dvt(
            raw_fused, raw_layers, fused_dvt=fused_dvt, layer_dvt=layer_dvt
        )
        for arm, spec in arms.items():
            feature_map = _arm_map(spec, raw_layers, raw_fused, dvt_layers, dvt_fused)
            values[arm].append(
                _score(
                    states[spec.state],
                    feature_map,
                    density_weight=spec.density_weight,
                    exclude_source=source_index,
                ).scores
            )
    return {arm: float(np.quantile(np.concatenate(parts), 0.995)) for arm, parts in values.items()}


def _base_row(
    *,
    object_name: str,
    entry: synthetic.SupportEntry,
    arm: str,
    condition: str,
    stimulus_id: str,
    shift_axis: str | None,
    shift_sign: int | None,
    shift_magnitude: int | None,
    defect: str,
) -> dict[str, Any]:
    return {
        "schema": ROW_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "object": object_name,
        "fold": entry.fold,
        "relative_path": entry.relative_path,
        "path_sha256": entry.path_sha256,
        "stimulus_id": stimulus_id,
        "arm": arm,
        "condition": condition,
        "shift_axis": shift_axis,
        "shift_sign": shift_sign,
        "shift_magnitude": shift_magnitude,
        "defect": defect,
    }


def _diagnostic_fields(
    *,
    spec: ArmSpec,
    raw_base: np.ndarray,
    raw_current: np.ndarray,
    dvt_base: np.ndarray,
    dvt_current: np.ndarray,
    dvt_stats: Mapping[str, float],
) -> dict[str, float]:
    prefix = "fused" if spec.layer is None else f"layer_{spec.layer}"
    return {
        "dvt_field_energy": dvt_stats[f"{prefix}_field_energy"],
        "dvt_variance_removed": dvt_stats[f"{prefix}_variance_removed"],
        "dvt_cosine_displacement_before": cosine_displacement(
            raw_base.reshape(-1, raw_base.shape[-1]), raw_current.reshape(-1, raw_current.shape[-1])
        ),
        "dvt_cosine_displacement_after": cosine_displacement(
            dvt_base.reshape(-1, dvt_base.shape[-1]), dvt_current.reshape(-1, dvt_current.shape[-1])
        ),
    }


def _emit_condition_rows(
    *,
    object_name: str,
    entry: synthetic.SupportEntry,
    condition: str,
    stimulus_id: str,
    shift_axis: str | None,
    shift_sign: int | None,
    shift_magnitude: int | None,
    defect: str,
    raw_layers: Mapping[int, np.ndarray],
    raw_fused: np.ndarray,
    base_raw_layers: Mapping[int, np.ndarray],
    base_raw_fused: np.ndarray,
    fused_dvt: PositionMeanArtifactDenoiser,
    layer_dvt: Mapping[int, PositionMeanArtifactDenoiser],
    dvt_stats: Mapping[str, float],
    states: Mapping[str, PipelineState],
    arms: Mapping[str, ArmSpec],
    thresholds: Mapping[str, float],
    base_scores: Mapping[str, ScoreOutput],
    mask: np.ndarray | None,
    stimulus_metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    dvt_fused, dvt_layers = _apply_dvt(
        raw_fused, raw_layers, fused_dvt=fused_dvt, layer_dvt=layer_dvt
    )
    base_dvt_fused, base_dvt_layers = _apply_dvt(
        base_raw_fused, base_raw_layers, fused_dvt=fused_dvt, layer_dvt=layer_dvt
    )
    rows: list[dict[str, Any]] = []
    for arm, spec in arms.items():
        current_map = _arm_map(spec, raw_layers, raw_fused, dvt_layers, dvt_fused)
        result = _score(states[spec.state], current_map, density_weight=spec.density_weight)
        state = states[spec.state]
        row = _base_row(
            object_name=object_name,
            entry=entry,
            arm=arm,
            condition=condition,
            stimulus_id=stimulus_id,
            shift_axis=shift_axis,
            shift_sign=shift_sign,
            shift_magnitude=shift_magnitude,
            defect=defect,
        )
        if stimulus_metadata is not None:
            row.update(stimulus_metadata)
        row.update(
            _row_metrics(
                result,
                base_scores[arm],
                state=state,
                mask=mask,
                threshold=thresholds[arm],
                seed=stable_seed(object_name, entry.relative_path, stimulus_id, arm, "geometry"),
            )
        )
        raw_base = base_raw_fused if spec.layer is None else base_raw_layers[spec.layer]
        raw_current = raw_fused if spec.layer is None else raw_layers[spec.layer]
        dvt_base = base_dvt_fused if spec.layer is None else base_dvt_layers[spec.layer]
        dvt_current = dvt_fused if spec.layer is None else dvt_layers[spec.layer]
        row.update(
            _diagnostic_fields(
                spec=spec,
                raw_base=raw_base,
                raw_current=raw_current,
                dvt_base=dvt_base,
                dvt_current=dvt_current,
                dvt_stats=dvt_stats,
            )
        )
        rows.append(row)
    return rows


def _read_whitelisted_rgb(entry: synthetic.SupportEntry, whitelist: set[Path]) -> np.ndarray:
    resolved = entry.path.resolve(strict=True)
    if resolved not in whitelist:
        raise PermissionError(f"path is not in the frozen support whitelist: {resolved}")
    return synthetic.read_rgb(resolved)


def run_real_object(
    args: argparse.Namespace,
    backbone: DINOv2Backbone,
    entries: Sequence[synthetic.SupportEntry],
    object_name: str,
    output_root: Path,
) -> dict[str, Any]:
    backbone.set_resolution(RESOLUTIONS[object_name])
    whitelist = {entry.path.resolve(strict=True) for entry in entries}
    clean_images: dict[str, np.ndarray] = {}
    clean_maps: dict[str, tuple[dict[int, np.ndarray], np.ndarray]] = {}
    grids: set[tuple[int, int]] = set()
    for entry in entries:
        image = _read_whitelisted_rgb(entry, whitelist)
        clean_images[entry.relative_path] = image
        layers, fused, grid = _extract_maps(backbone, image)
        clean_maps[entry.relative_path] = (layers, fused)
        grids.add(grid)
    if len(grids) != 1:
        raise RuntimeError(f"{object_name}: support feature grids are not identical: {grids}")

    arms = arm_specs()
    rows: list[dict[str, Any]] = []
    m0_error = 0.0
    m0_mismatch = 0
    for fold in range(FOLD_COUNT):
        states, fused_dvt, layer_dvt, dvt_stats = _fit_fold(
            clean_maps, entries, fold, device=args.device
        )
        fit_entries = [entry for entry in entries if entry.fold != fold]
        thresholds = _fit_thresholds(
            states, arms, clean_maps, fit_entries, fused_dvt, layer_dvt
        )
        held_out = [entry for entry in entries if entry.fold == fold]
        if len(held_out) != 4:
            raise RuntimeError(f"fold {fold}: expected four held-out sources")
        first_layers, first_fused = clean_maps[held_out[0].relative_path]
        first_dvt_fused, _ = _apply_dvt(
            first_fused, first_layers, fused_dvt=fused_dvt, layer_dvt=layer_dvt
        )
        error, mismatch = m0_parity_assertion(
            states["dvt_flow"],
            first_dvt_fused,
            density_weight=0.25,
            atol=CUDA_CDIST_PARITY_ATOL,
        )
        m0_error = max(m0_error, error)
        m0_mismatch += mismatch

        for entry in held_out:
            image, clean_stimulus_metadata = clean_source_with_metadata(
                clean_images, entry.relative_path,
            )
            raw_layers, raw_fused = clean_maps[entry.relative_path]
            dvt_fused, dvt_layers = _apply_dvt(
                raw_fused, raw_layers, fused_dvt=fused_dvt, layer_dvt=layer_dvt
            )
            clean_scores = {
                arm: _score(
                    states[spec.state],
                    _arm_map(spec, raw_layers, raw_fused, dvt_layers, dvt_fused),
                    density_weight=spec.density_weight,
                )
                for arm, spec in arms.items()
            }
            rows.extend(
                _emit_condition_rows(
                    object_name=object_name,
                    entry=entry,
                    condition="C",
                    stimulus_id="C",
                    shift_axis=None,
                    shift_sign=None,
                    shift_magnitude=None,
                    defect="none",
                    raw_layers=raw_layers,
                    raw_fused=raw_fused,
                    base_raw_layers=raw_layers,
                    base_raw_fused=raw_fused,
                    fused_dvt=fused_dvt,
                    layer_dvt=layer_dvt,
                    dvt_stats=dvt_stats,
                    states=states,
                    arms=arms,
                    thresholds=thresholds,
                    base_scores=clean_scores,
                    mask=None,
                    stimulus_metadata=clean_stimulus_metadata,
                )
            )
            for axis, sign, magnitude, transform in shift_specs():
                shifted_image = apply_rgb_transform(image, transform)
                shifted_layers, shifted_fused, shifted_grid = _extract_maps(
                    backbone, shifted_image
                )
                if shifted_grid != next(iter(grids)):
                    raise RuntimeError("photometric shift changed the feature grid")
                rows.extend(
                    _emit_condition_rows(
                        object_name=object_name,
                        entry=entry,
                        condition="S",
                        stimulus_id=f"S:{axis}:{sign:+d}:{magnitude}",
                        shift_axis=axis,
                        shift_sign=sign,
                        shift_magnitude=magnitude,
                        defect="none",
                        raw_layers=shifted_layers,
                        raw_fused=shifted_fused,
                        base_raw_layers=raw_layers,
                        base_raw_fused=raw_fused,
                        fused_dvt=fused_dvt,
                        layer_dvt=layer_dvt,
                        dvt_stats=dvt_stats,
                        states=states,
                        arms=arms,
                        thresholds=thresholds,
                        base_scores=clean_scores,
                        mask=None,
                        stimulus_metadata={
                            "evaluated_image_sha256": sha256_bytes(shifted_image.tobytes())
                        },
                    )
                )

            for defect in DEFECTS:
                selected: tuple[
                    np.ndarray,
                    np.ndarray,
                    dict[int, np.ndarray],
                    np.ndarray,
                    dict[str, Any],
                ] | None = None
                for retry in range(synthetic.DEFECT_GENERATION_ATTEMPTS):
                    try:
                        defect_image, mask, metadata = make_defect(
                            image,
                            object_name=object_name,
                            relative_path=entry.relative_path,
                            kind=defect,
                            feature_retry=retry,
                        )
                    except RuntimeError:
                        # The frozen retry index changes only the SHA-derived
                        # stimulus, never in response to a downstream metric.
                        continue
                    defect_layers, defect_fused, defect_grid = _extract_maps(backbone, defect_image)
                    if defect_grid != next(iter(grids)):
                        raise RuntimeError("synthetic defect changed the feature grid")
                    patch_mask = synthetic.mask_to_patch_grid(mask, defect_grid)
                    feature_visible = float(np.ptp(defect_fused - raw_fused)) > 0
                    if patch_mask.any() and not patch_mask.all() and feature_visible:
                        selected = defect_image, mask, defect_layers, defect_fused, metadata
                        break
                if selected is None:
                    raise RuntimeError(
                        f"{object_name}/{entry.relative_path}/{defect}: no feature-visible defect"
                    )
                defect_image, mask, defect_layers, defect_fused, defect_metadata = selected
                defect_dvt_fused, defect_dvt_layers = _apply_dvt(
                    defect_fused,
                    defect_layers,
                    fused_dvt=fused_dvt,
                    layer_dvt=layer_dvt,
                )
                defect_scores = {
                    arm: _score(
                        states[spec.state],
                        _arm_map(
                            spec,
                            defect_layers,
                            defect_fused,
                            defect_dvt_layers,
                            defect_dvt_fused,
                        ),
                        density_weight=spec.density_weight,
                    )
                    for arm, spec in arms.items()
                }
                rows.extend(
                    _emit_condition_rows(
                        object_name=object_name,
                        entry=entry,
                        condition="D",
                        stimulus_id=f"D:{defect}",
                        shift_axis=None,
                        shift_sign=None,
                        shift_magnitude=None,
                        defect=defect,
                        raw_layers=defect_layers,
                        raw_fused=defect_fused,
                        base_raw_layers=defect_layers,
                        base_raw_fused=defect_fused,
                        fused_dvt=fused_dvt,
                        layer_dvt=layer_dvt,
                        dvt_stats=dvt_stats,
                        states=states,
                        arms=arms,
                        thresholds=thresholds,
                        base_scores=defect_scores,
                        mask=mask,
                        stimulus_metadata={
                            "defect_seed": int(defect_metadata["seed"]),
                            "defect_feature_retry": int(defect_metadata["feature_retry"]),
                            "defect_image_sha256": str(defect_metadata["image_sha256"]),
                            "mask_sha256": str(defect_metadata["mask_sha256"]),
                            "evaluated_image_sha256": sha256_bytes(defect_image.tobytes()),
                        },
                    )
                )
                for axis, sign, magnitude, transform in shift_specs():
                    shifted_defect_image = apply_rgb_transform(defect_image, transform)
                    shifted_defect_layers, shifted_defect_fused, shifted_grid = _extract_maps(
                        backbone, shifted_defect_image
                    )
                    if shifted_grid != next(iter(grids)):
                        raise RuntimeError("shifted defect changed the feature grid")
                    rows.extend(
                        _emit_condition_rows(
                            object_name=object_name,
                            entry=entry,
                            condition="SD",
                            stimulus_id=f"SD:{defect}:{axis}:{sign:+d}:{magnitude}",
                            shift_axis=axis,
                            shift_sign=sign,
                            shift_magnitude=magnitude,
                            defect=defect,
                            raw_layers=shifted_defect_layers,
                            raw_fused=shifted_defect_fused,
                            base_raw_layers=defect_layers,
                            base_raw_fused=defect_fused,
                            fused_dvt=fused_dvt,
                            layer_dvt=layer_dvt,
                            dvt_stats=dvt_stats,
                            states=states,
                            arms=arms,
                            thresholds=thresholds,
                            base_scores=defect_scores,
                            mask=mask,
                            stimulus_metadata={
                                "defect_seed": int(defect_metadata["seed"]),
                                "defect_feature_retry": int(
                                    defect_metadata["feature_retry"]
                                ),
                                "defect_image_sha256": str(
                                    defect_metadata["image_sha256"]
                                ),
                                "mask_sha256": str(defect_metadata["mask_sha256"]),
                                "evaluated_image_sha256": sha256_bytes(
                                    shifted_defect_image.tobytes()
                                ),
                            },
                        )
                    )
        del states
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    expected = expected_row_counts()
    condition_counts = {
        condition: sum(row["condition"] == condition for row in rows)
        for condition in ("C", "S", "D", "SD")
    }
    if condition_counts != {key: expected[key] for key in condition_counts}:
        raise RuntimeError(f"{object_name}: factorial coverage mismatch {condition_counts} != {expected}")
    if any(
        not np.isfinite(value)
        for row in rows
        for value in row.values()
        if isinstance(value, float)
    ):
        raise RuntimeError(f"{object_name}: non-finite compact metric")
    records_path = output_root / "objects" / f"{object_name}.jsonl"
    observed = atomic_write_jsonl(records_path, rows)
    summary_path = output_root / "objects" / f"{object_name}_summary.json"
    summary = {
        "schema": SUMMARY_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "object": object_name,
        "mode": "stage_m1_support_only",
        "support_count": len(entries),
        "fold_counts": np.bincount([entry.fold for entry in entries], minlength=4).tolist(),
        "row_count": observed,
        "expected_row_count": expected["total"],
        "condition_counts": condition_counts,
        "expected_condition_counts": {key: expected[key] for key in condition_counts},
        "arms": list(arms),
        "conditions": ["C", "S", "D", "SD"],
        "m0_score_max_abs_error": m0_error,
        "m0_nn_mismatch_count": m0_mismatch,
        "m0_score_atol": CUDA_CDIST_PARITY_ATOL,
        "parity_complete": (
            m0_error <= CUDA_CDIST_PARITY_ATOL
            and m0_mismatch == 0
            and observed == expected["total"]
        ),
        "finite_complete": True,
        "static_scoring_only": True,
        "test_public_accessed": False,
        "test_private_accessed": False,
        "tta_calls": 0,
        "query_update_calls": 0,
        "artifacts": {"records": str(records_path), "summary": str(summary_path)},
        "artifact_sha256": {"records": sha256_file(records_path)},
    }
    atomic_write_json(summary_path, summary)
    summary["artifact_sha256"]["summary"] = sha256_file(summary_path)
    return summary


def _smoke_entry(object_name: str) -> synthetic.SupportEntry:
    return synthetic.SupportEntry(
        object_name=object_name,
        path=Path("synthetic") / object_name / "train" / "good" / "fixture.png",
        relative_path=f"{object_name}/train/good/fixture.png",
        path_sha256=sha256_bytes(object_name.encode()),
        content_sha256=sha256_bytes(f"fixture:{object_name}".encode()),
        fold=0,
    )


def run_smoke_object(args: argparse.Namespace, object_name: str, output_root: Path) -> dict[str, Any]:
    rng = np.random.default_rng(stable_seed(object_name, "smoke"))
    fit_maps = [rng.normal(size=(2, 2, 8)).astype(np.float32) + index * 0.05 for index in range(4)]
    dvt = PositionMeanArtifactDenoiser.fit(fit_maps, alpha=1.0)
    dvt_maps = [dvt.transform(value) for value in fit_maps]
    states = {
        "dvt_flow": _fit_state("dvt_flow", dvt_maps, transform_mode="flow", device="cpu", smoke=True),
        "raw_flow": _fit_state("raw_flow", fit_maps, transform_mode="flow", device="cpu", smoke=True),
        "dvt_identity": _fit_state(
            "dvt_identity", dvt_maps, transform_mode="identity", device="cpu", smoke=True
        ),
        "raw_identity": _fit_state(
            "raw_identity", fit_maps, transform_mode="identity", device="cpu", smoke=True
        ),
    }
    for layer in LAYERS:
        states[f"layer_{layer}_identity"] = _fit_state(
            f"layer_{layer}_identity", dvt_maps, transform_mode="identity", device="cpu", smoke=True
        )
    query = rng.normal(size=(2, 2, 8)).astype(np.float32)
    shifted = query + rng.normal(scale=0.03, size=query.shape).astype(np.float32)
    mask = np.array([[True, False], [False, False]])
    rows: list[dict[str, Any]] = []
    entry = _smoke_entry(object_name)
    max_error = 0.0
    mismatch = 0
    for arm, spec in arm_specs().items():
        current = dvt.transform(query) if spec.query_dvt else query
        shifted_current = dvt.transform(shifted) if spec.query_dvt else shifted
        state = states[spec.state]
        base = _score(state, current, density_weight=spec.density_weight)
        error, misses = m0_parity_assertion(state, current, density_weight=spec.density_weight)
        max_error = max(max_error, error)
        mismatch += misses
        for condition, feature, row_mask in (
            ("C", current, None),
            ("S", shifted_current, None),
            ("D", current + 0.1, mask),
            ("SD", shifted_current + 0.1, mask),
        ):
            result = _score(state, feature, density_weight=spec.density_weight)
            row = _base_row(
                object_name=object_name,
                entry=entry,
                arm=arm,
                condition=condition,
                stimulus_id=f"smoke:{condition}",
                shift_axis="exposure" if condition in {"S", "SD"} else None,
                shift_sign=1 if condition in {"S", "SD"} else None,
                shift_magnitude=1 if condition in {"S", "SD"} else None,
                defect="cutpaste" if condition in {"D", "SD"} else "none",
            )
            row.update(
                _row_metrics(
                    result,
                    base,
                    state=state,
                    mask=row_mask,
                    threshold=float(np.quantile(base.scores, 0.995)),
                    seed=stable_seed(object_name, arm, condition),
                )
            )
            row.update(
                {
                    "dvt_field_energy": float(np.mean(np.square(dvt.artifact))),
                    "dvt_variance_removed": dvt_variance_removed(fit_maps, dvt_maps),
                    "dvt_cosine_displacement_before": cosine_displacement(
                        query.reshape(-1, 8), feature.reshape(-1, 8)
                    ),
                    "dvt_cosine_displacement_after": cosine_displacement(
                        dvt.transform(query).reshape(-1, 8),
                        dvt.transform(feature).reshape(-1, 8),
                    ),
                }
            )
            rows.append(row)
    records_path = output_root / "objects" / f"{object_name}.jsonl"
    count = atomic_write_jsonl(records_path, rows)
    summary_path = output_root / "objects" / f"{object_name}_summary.json"
    condition_counts = {key: sum(row["condition"] == key for row in rows) for key in ("C", "S", "D", "SD")}
    summary = {
        "schema": SUMMARY_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "object": object_name,
        "mode": "smoke",
        "support_count": 4,
        "fold_counts": [0, 0, 0, 0],
        "row_count": count,
        "expected_row_count": len(arm_specs()) * 4,
        "condition_counts": condition_counts,
        "expected_condition_counts": {key: len(arm_specs()) for key in condition_counts},
        "arms": list(arm_specs()),
        "conditions": ["C", "S", "D", "SD"],
        "m0_score_max_abs_error": max_error,
        "m0_nn_mismatch_count": mismatch,
        "m0_score_atol": FIXTURE_PARITY_ATOL,
        "parity_complete": max_error <= FIXTURE_PARITY_ATOL and mismatch == 0,
        "finite_complete": True,
        "static_scoring_only": True,
        "test_public_accessed": False,
        "test_private_accessed": False,
        "tta_calls": 0,
        "query_update_calls": 0,
        "artifacts": {"records": str(records_path), "summary": str(summary_path)},
        "artifact_sha256": {"records": sha256_file(records_path)},
    }
    atomic_write_json(summary_path, summary)
    summary["artifact_sha256"]["summary"] = sha256_file(summary_path)
    return summary


def frozen_config(mode: str) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "mode": mode,
        "evaluation_role": "support_only_default_mechanism_audit",
        "maximum_verdict": "MECHANISM_SUPPORTED_ON_SUPPORT_ONLY",
        "backbone": BACKBONE_MODEL,
        "layers": list(LAYERS),
        "resolution": RESOLUTIONS,
        "support_count": SUPPORT_COUNT,
        "required_support_manifest_sha256": REQUIRED_SUPPORT_SHA256,
        "folds": "four source-disjoint 12-fit/4-held-out rotations",
        "feature_fusion": "layer_norm_mean",
        "dvt": {"mode": "position_mean", "alpha": 1.0},
        "flow": asdict(_flow_config("flow")),
        "score": {
            "distance_weight": 1.0,
            "density_weight": 0.25,
            "density_quantile": 0.9,
            "top_percent": 0.01,
            "query_chunk_size": QUERY_CHUNK_SIZE,
            "loo_standardization": False,
            "squared_distance": False,
        },
        "arms": {name: asdict(spec) for name, spec in arm_specs().items()},
        "conditions": ["C", "S", "D", "SD"],
        "shift_axes": list(AXES),
        "shift_signs": list(SIGNS),
        "shift_magnitudes": list(MAGNITUDES),
        "defects": list(DEFECTS),
        "memory": "static exact fit-support patch bank",
        "parity": {
            "cpu_fixture_atol": FIXTURE_PARITY_ATOL,
            "cuda_chunked_vs_full_cdist_atol": CUDA_CDIST_PARITY_ATOL,
            "nearest_neighbor_index_mismatches_allowed": 0,
        },
        "tta": False,
        "query_update": False,
        "test_public_access": False,
        "test_private_access": False,
    }


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    if args.mode == "real" and args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real Stage-M1 execution")
    output_root = Path(args.output_root)
    _prepare_output_root(output_root)
    config_path = output_root / "frozen_config.json"
    atomic_write_json(config_path, frozen_config(args.mode))
    synthetic.PROTOCOL_ID = PROTOCOL_ID

    support_manifest_hash = "synthetic_fixture"
    support: dict[str, tuple[synthetic.SupportEntry, ...]] | None = None
    if args.mode == "real":
        support_path = Path(args.support_json).resolve(strict=True)
        support_manifest_hash = sha256_file(support_path)
        if support_manifest_hash != REQUIRED_SUPPORT_SHA256:
            raise RuntimeError(
                "support manifest SHA-256 mismatch: "
                f"{support_manifest_hash} != {REQUIRED_SUPPORT_SHA256}"
            )
        support = synthetic.load_support_manifest(support_path, Path(args.data_root))

    ledger = {
        "protocol_id": PROTOCOL_ID,
        "mode": args.mode,
        # The reused manifest loader hashes all 128 allowlisted files once;
        # selected shard images are then decoded exactly once for extraction.
        "support_reads": (
            SUPPORT_COUNT * len(OBJECTS) + SUPPORT_COUNT * len(args.objects)
            if args.mode == "real"
            else 0
        ),
        "support_content_hash_reads": SUPPORT_COUNT * len(OBJECTS) if args.mode == "real" else 0,
        "support_image_reads": SUPPORT_COUNT * len(args.objects) if args.mode == "real" else 0,
        "test_public_reads": 0,
        "test_private_reads": 0,
        "query_update_calls": 0,
        "tta_calls": 0,
        "static_scoring_only": True,
        "whitelist_complete": True,
        "enumeration_api_present": False,
    }
    ledger_path = output_root / "access_ledger.json"
    atomic_write_json(ledger_path, ledger)

    if args.mode == "real":
        backbone = DINOv2Backbone(
            model_name=BACKBONE_MODEL,
            device=args.device,
            smaller_edge_size=672,
            feature_layers=LAYERS,
        )
        assert support is not None
        summaries = [
            run_real_object(args, backbone, support[object_name], object_name, output_root)
            for object_name in args.objects
        ]
    else:
        summaries = [run_smoke_object(args, object_name, output_root) for object_name in args.objects]

    manifest = {
        "schema": RUN_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "objects": list(args.objects),
        "backbone": BACKBONE_MODEL,
        "layers": list(LAYERS),
        "required_support_manifest_sha256": REQUIRED_SUPPORT_SHA256,
        "support_manifest_sha256": support_manifest_hash,
        "runner_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(config_path),
        "access_ledger_sha256": sha256_file(ledger_path),
        "summaries": [f"objects/{summary['object']}_summary.json" for summary in summaries],
        "summary_sha256": {
            summary["object"]: sha256_file(
                output_root / "objects" / f"{summary['object']}_summary.json"
            )
            for summary in summaries
        },
        "m0_complete": all(summary["parity_complete"] for summary in summaries),
        "factorial_complete": all(
            summary["row_count"] == summary["expected_row_count"] for summary in summaries
        ),
        "static_scoring_only": True,
        "test_public_accessed": False,
        "test_private_accessed": False,
        "host_gpu_id": str(args.host_gpu_id),
        "container_cuda_slot": str(args.container_cuda_slot),
    }
    atomic_write_json(output_root / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
