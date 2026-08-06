#!/usr/bin/env python3
"""Run the frozen compact local-soft physical nuisance diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from flow_tte_mvtec_ad2_core import (  # noqa: E402
    BackboneLike,
    FeatureExtractionConfig,
    FeatureMap,
    ImageItem,
    LayeredFeatureMap,
    apply_feature_denoiser,
    build_pipeline,
    extract_layer_feature_maps_from_rgb,
    flatten_support_feature_maps,
    merge_layer_features,
    read_rgb,
    save_prediction,
    select_support_paths_for_backbone,
)
from run_flow_tte_mvtec_ad2 import build_runtime  # noqa: E402
from run_flow_tte_support_bank_routing import build_default_config  # noqa: E402

from flow_tte.cfntr import (  # noqa: E402
    TRAIN_AXIS_AMPLITUDES,
    apply_rgb_transform,
    make_axis_transform,
)
from flow_tte.denoising import fit_feature_denoiser  # noqa: E402

OBJECTS = (
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
)
AXES = ("exposure", "gamma", "white_balance")
ARMS = ("identity", "global_hard_rank4", "local_soft_rank2")
SOURCE_PATTERN = re.compile(r"^(?P<source>\d+)_")
MAX_BASIS_ROWS = 4096
RANDOMIZED_OVERSAMPLE = 8


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def source_id(path: Path) -> str:
    match = SOURCE_PATTERN.match(path.stem)
    if match is None:
        message = f"cannot parse source identity from {path.name}"
        raise ValueError(message)
    return match.group("source")


def compact_orbit_items(test_images: Mapping[str, Sequence[str]]) -> list[tuple[str, Path]]:
    selected: list[tuple[str, Path]] = []
    for split in ("good", "bad"):
        paths = sorted(Path(value) for value in test_images.get(split, ()))
        identities = sorted({source_id(path) for path in paths})[:2]
        if len(identities) != 2:
            message = f"compact orbit requires two source identities for {split}"
            raise RuntimeError(message)
        selected.extend((split, path) for path in paths if source_id(path) in identities)
    return sorted(selected, key=lambda row: (row[1].name, row[0]))


def quadrant_slices(height: int, width: int) -> tuple[tuple[slice, slice], ...]:
    y_mid = height // 2
    x_mid = width // 2
    return (
        (slice(0, y_mid), slice(0, x_mid)),
        (slice(0, y_mid), slice(x_mid, width)),
        (slice(y_mid, height), slice(0, x_mid)),
        (slice(y_mid, height), slice(x_mid, width)),
    )


def sampled_rows(rows: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("basis rows must be a nonempty matrix")
    if matrix.shape[0] <= max_rows:
        return matrix
    indices = np.random.default_rng(seed).choice(matrix.shape[0], size=max_rows, replace=False)
    return matrix[np.sort(indices)]


def randomized_right_basis(rows: np.ndarray, rank: int, seed: int) -> np.ndarray:
    matrix = sampled_rows(rows, MAX_BASIS_ROWS, seed).astype(np.float64)
    usable_rank = min(rank, matrix.shape[0], matrix.shape[1])
    if usable_rank <= 0:
        raise ValueError("basis rank must be positive")
    width = min(matrix.shape[1], usable_rank + RANDOMIZED_OVERSAMPLE)
    omega = np.random.default_rng(seed).standard_normal((matrix.shape[1], width))
    projected = matrix @ omega
    q_values, _ = np.linalg.qr(projected, mode="reduced")
    small = q_values.T @ matrix
    _left, _singular, right = np.linalg.svd(small, full_matrices=False)
    basis = right[:usable_rank].T
    q_basis, _ = np.linalg.qr(basis, mode="reduced")
    return q_basis[:, :usable_rank].astype(np.float32)


def energy_coverage(rows: np.ndarray, basis: np.ndarray) -> float:
    matrix = np.asarray(rows, dtype=np.float32)
    total = float(np.square(matrix, dtype=np.float64).sum())
    if total <= 1e-12:
        return 0.0
    projected = matrix @ basis
    return float(np.square(projected, dtype=np.float64).sum() / total)


def fused_map(layered: LayeredFeatureMap) -> FeatureMap:
    layer_rows = [
        layer.values.reshape(-1, layer.values.shape[-1]) for layer in layered.layers
    ]
    fused = merge_layer_features(layer_rows, "layer_norm_mean")
    height, width = layered.layers[0].values.shape[:2]
    return FeatureMap(
        values=fused.reshape(height, width, fused.shape[-1]),
        image_shape=layered.layers[0].image_shape,
    )


def correct_layer(
    values: np.ndarray,
    position_mean: np.ndarray,
    bases: Sequence[np.ndarray],
    strength: float,
) -> np.ndarray:
    corrected = np.asarray(values, dtype=np.float32).copy()
    for quadrant, (y_slice, x_slice) in enumerate(
        quadrant_slices(values.shape[0], values.shape[1]),
    ):
        residual = values[y_slice, x_slice] - position_mean[y_slice, x_slice]
        flat = residual.reshape(-1, residual.shape[-1])
        basis = bases[quadrant]
        corrected[y_slice, x_slice] -= (
            strength * ((flat @ basis) @ basis.T).reshape(residual.shape)
        )
    return corrected.astype(np.float32, copy=False)


def extract_layers(backbone: BackboneLike, image: np.ndarray) -> LayeredFeatureMap:
    return extract_layer_feature_maps_from_rgb(
        backbone,
        image,
        FeatureExtractionConfig(feature_fusion="layer_norm_mean", context_source="none"),
    )


def fit_tangent_state(
    identity: Sequence[LayeredFeatureMap],
    transformed: Mapping[tuple[str, int], Sequence[LayeredFeatureMap]],
    object_name: str,
) -> dict[str, Any]:
    layer_count = len(identity[0].layers)
    position_means: list[np.ndarray] = []
    global_bases: list[np.ndarray] = []
    local_bases: list[list[np.ndarray]] = []
    diagnostics: list[dict[str, Any]] = []
    for layer_index in range(layer_count):
        identity_values = np.stack(
            [item.layers[layer_index].values for item in identity],
            axis=0,
        )
        position_mean = identity_values.mean(axis=0).astype(np.float32)
        position_means.append(position_mean)
        differences: list[np.ndarray] = []
        for axis in AXES:
            negative = transformed[(axis, -1)]
            positive = transformed[(axis, 1)]
            scale = 2.0 * TRAIN_AXIS_AMPLITUDES[axis]
            for neg_item, pos_item in zip(negative, positive):
                differences.append(
                    (pos_item.layers[layer_index].values - neg_item.layers[layer_index].values)
                    / scale,
                )
        global_rows = np.concatenate(
            [difference.reshape(-1, difference.shape[-1]) for difference in differences],
            axis=0,
        )
        global_sample = sampled_rows(
            global_rows,
            MAX_BASIS_ROWS,
            stable_seed(object_name, layer_index, "global"),
        )
        global_basis = randomized_right_basis(
            global_sample,
            rank=4,
            seed=stable_seed(object_name, layer_index, "global_basis"),
        )
        global_bases.append(global_basis)
        layer_local: list[np.ndarray] = []
        local_coverage: list[float] = []
        for quadrant, (y_slice, x_slice) in enumerate(
            quadrant_slices(position_mean.shape[0], position_mean.shape[1]),
        ):
            centered_rows = []
            for difference in differences:
                quadrant_values = difference[y_slice, x_slice]
                centered = quadrant_values - quadrant_values.mean(
                    axis=(0, 1),
                    keepdims=True,
                )
                centered_rows.append(centered.reshape(-1, centered.shape[-1]))
            rows = np.concatenate(centered_rows, axis=0)
            sample = sampled_rows(
                rows,
                MAX_BASIS_ROWS,
                stable_seed(object_name, layer_index, quadrant, "sample"),
            )
            basis = randomized_right_basis(
                sample,
                rank=2,
                seed=stable_seed(object_name, layer_index, quadrant, "basis"),
            )
            layer_local.append(basis)
            local_coverage.append(energy_coverage(sample, basis))
        local_bases.append(layer_local)
        diagnostics.append({
            "layer_index": layer_index,
            "global_rank": int(global_basis.shape[1]),
            "global_sample_count": int(global_sample.shape[0]),
            "global_energy_coverage": energy_coverage(global_sample, global_basis),
            "local_rank": [int(basis.shape[1]) for basis in layer_local],
            "local_energy_coverage": local_coverage,
        })
    return {
        "position_means": position_means,
        "global_bases": global_bases,
        "local_bases": local_bases,
        "diagnostics": diagnostics,
    }


def corrected_layered(
    layered: LayeredFeatureMap,
    state: Mapping[str, Any],
    mode: str,
) -> LayeredFeatureMap:
    layers = []
    for layer_index, layer in enumerate(layered.layers):
        if mode == "identity":
            values = layer.values
        elif mode == "global_hard_rank4":
            basis = state["global_bases"][layer_index]
            values = correct_layer(
                layer.values,
                state["position_means"][layer_index],
                [basis, basis, basis, basis],
                strength=1.0,
            )
        elif mode == "local_soft_rank2":
            values = correct_layer(
                layer.values,
                state["position_means"][layer_index],
                state["local_bases"][layer_index],
                strength=0.25,
            )
        else:
            message = f"unknown correction mode: {mode}"
            raise ValueError(message)
        layers.append(FeatureMap(values=values, image_shape=layer.image_shape))
    return LayeredFeatureMap(layers=tuple(layers))


def run_object(args: argparse.Namespace, object_name: str) -> dict[str, Any]:
    started = time.time()
    config = build_default_config(args)
    dataset, backbone = build_runtime(config)
    info = dataset.get_object_info(object_name)
    backbone.set_resolution(config.backbone_resolution or info.resolution)
    train_paths = [Path(path) for path in dataset.get_train_images(object_name)]
    support_paths = select_support_paths_for_backbone(
        backbone,
        train_paths,
        shots=16,
        policy=f"fixed_json={args.support_json}",
        seed=0,
    )
    identity: list[LayeredFeatureMap] = []
    transformed: dict[tuple[str, int], list[LayeredFeatureMap]] = {
        (axis, sign): [] for axis in AXES for sign in (-1, 1)
    }
    for path in support_paths:
        image = read_rgb(path)
        identity.append(extract_layers(backbone, image))
        for axis in AXES:
            for sign in (-1, 1):
                transformed[(axis, sign)].append(
                    extract_layers(
                        backbone,
                        apply_rgb_transform(image, make_axis_transform(axis, sign=sign)),
                    ),
                )
    state = fit_tangent_state(identity, transformed, object_name)
    identity_fused = [fused_map(item) for item in identity]
    denoiser = fit_feature_denoiser(
        "position_mean",
        [item.values for item in identity_fused],
        alpha=1.0,
    )
    support_features = flatten_support_feature_maps(
        identity_fused,
        require_contexts=False,
        feature_denoiser=denoiser,
    )
    pipeline = build_pipeline(config)
    stats = pipeline.fit(support_features.values)

    test_images = dataset.get_test_images(object_name, split="test_public")
    orbit = compact_orbit_items(test_images)
    for split, path in orbit:
        layered = extract_layers(backbone, read_rgb(path))
        item = ImageItem(anomaly_type=split, path=path)
        for arm in ARMS:
            candidate = fused_map(corrected_layered(layered, state, arm))
            candidate = apply_feature_denoiser(candidate, denoiser)
            result = pipeline.score_static(candidate.values[np.newaxis, ...])
            patch_scores = result.patch_scores[0]
            score_map = cv2.resize(
                patch_scores.astype(np.float32, copy=False),
                (candidate.image_shape[1], candidate.image_shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            save_prediction(
                Path(args.output_root) / "objects" / object_name / arm,
                object_name,
                item,
                score_map,
            )

    object_root = Path(args.output_root) / "objects" / object_name
    object_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "object": object_name,
        "protocol": "local-soft-nuisance-stage4-20260718",
        "support_paths": [str(path) for path in support_paths],
        "axes": list(AXES),
        "arms": list(ARMS),
        "orbit": [{"split": split, "stem": path.stem} for split, path in orbit],
        "orbit_count": len(orbit),
        "tangent_diagnostics": state["diagnostics"],
        "global_hard_strength": 1.0,
        "local_soft_strength": 0.25,
        "train_nll_mean": stats.train_nll_mean,
        "train_nll_std": stats.train_nll_std,
        "density_threshold": stats.density_threshold,
        "query_memory_update": False,
        "test_labels_used_by_correction": False,
        "elapsed_seconds": time.time() - started,
    }
    (object_root / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--support-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--project-root", default="/workspace")
    parser.add_argument("--fsad-root", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    objects = tuple(value for value in args.objects.replace(",", " ").split() if value)
    if not objects or any(value not in OBJECTS for value in objects):
        parser.error(f"--objects must be a nonempty subset of {OBJECTS}")
    args.object_names = objects
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_cli(argv)
    for object_name in args.object_names:
        run_object(args, object_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
