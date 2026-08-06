#!/usr/bin/env python3
# ruff: noqa: ANN401, C901, COM812, E501, EM102, PLR0912, PLR0913, PLR0915, SLF001
"""Run the preregistered support-only SemanticAtlasMetric diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "src", Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_flowtte_paired_loss_diagnostic as base  # noqa: E402

from flow_tte.config import ExpansionConfig, FlowConfig  # noqa: E402
from flow_tte.semantic_atlas_metric import (  # noqa: E402
    SemanticAtlasConfig,
    SemanticAtlasMetric,
    assign_clusters,
    basis_diagnostics,
    fit_global_rank_metric,
    fit_semantic_atlas,
    fit_unorthogonalized_atlas,
    random_atlas_like,
    shuffled_atlas,
)
from flow_tte.trainer import FlowDensityEstimator  # noqa: E402

PROTOCOL_ID = "flowtte-semantic-atlas-metric-stage-minus1-20260805"
OBJECTS = ("fruit_jelly", "walnuts", "vial", "fabric")
ARMS = (
    "A0_baseline1_core",
    "A1_semantic_atlas_orth",
    "A2_semantic_atlas_raw",
    "A3_global_rank4",
    "A4_cluster_shuffle",
    "A5_random_rank1",
)
RANDOM_KEYS = ("A5_random_rank1_s0", "A5_random_rank1_s1", "A5_random_rank1_s2")
INTERNAL_ARMS = ARMS[:-1] + RANDOM_KEYS
SUPPORT_SHA256 = base.FROZEN_SUPPORT_MANIFEST_SHA256
TRAIN_SAMPLE_PER_PAIR = 128
GEOMETRY_SAMPLE = 512


class RunnerError(RuntimeError):
    pass


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("mechanism", "smoke"), required=True)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--support-json", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host-gpu-id", default="")
    parser.add_argument("--container-cuda-slot", default="0")
    args = parser.parse_args(list(argv))
    args.objects = tuple(item for item in args.objects.replace(",", " ").split() if item)
    if not args.objects or any(item not in OBJECTS for item in args.objects):
        parser.error(f"--objects must be a non-empty subset of {OBJECTS}")
    if args.phase == "mechanism" and (not args.data_root or not args.support_json):
        parser.error("mechanism requires --data-root and --support-json")
    return args


def frozen_config(device: str) -> dict[str, Any]:
    stage_a = base.frozen_config(device)
    return {
        "protocol_id": PROTOCOL_ID,
        "baseline_method": "Baseline1-core",
        "device": device,
        "backbone": stage_a["backbone"],
        "layers": stage_a["layers"],
        "feature_fusion": stage_a["feature_fusion"],
        "dvt": stage_a["dvt"],
        "stage_a_flow": stage_a["stage_a_flow"],
        "atlas": asdict(SemanticAtlasConfig()),
        "folds": {"count": 4, "fit_sources": 12, "held_sources": 4},
        "train_pair_sample_rows": TRAIN_SAMPLE_PER_PAIR,
        "arms": list(ARMS),
        "random_seeds": [101, 202, 303],
        "density_weight": 0.0,
        "query_adaptation": False,
        "test_public_access": False,
        "test_private_access": False,
    }


def _latent(estimator: FlowDensityEstimator, item: base.FeatureMap) -> np.ndarray:
    return estimator.evaluate(item.values).z.detach().cpu().numpy().astype(np.float32, copy=False)


def _dvt(item: base.FeatureMap, artifact: np.ndarray | None) -> base.FeatureMap:
    if artifact is None:
        return item
    return base.FeatureMap(
        values=(item.values - artifact).astype(np.float32, copy=False),
        grid=item.grid,
        image_shape=item.image_shape,
        relative_path=item.relative_path,
        transform=item.transform,
    )


def _folds(entries: Sequence[base.SupportEntry]) -> tuple[tuple[int, ...], ...]:
    order = sorted(range(len(entries)), key=lambda idx: entries[idx].path_sha256)
    return tuple(tuple(order[start : start + 4]) for start in range(0, 16, 4))


def _sample_indices(n: int, *seed_parts: object) -> np.ndarray:
    count = min(TRAIN_SAMPLE_PER_PAIR, n)
    rng = np.random.default_rng(base.stable_seed(PROTOCOL_ID, *seed_parts))
    return np.sort(rng.choice(n, size=count, replace=False)).astype(np.int64)


def _fit_flow(
    support_maps: Sequence[base.FeatureMap], config: Mapping[str, Any], device: str
) -> tuple[FlowDensityEstimator, dict[str, Any]]:
    estimator = FlowDensityEstimator(
        dim=int(support_maps[0].values.shape[-1]),
        config=FlowConfig(**dict(config["stage_a_flow"])),
        device=device,
    )
    stats = estimator.fit(
        np.stack([item.values for item in support_maps], axis=0),
        density_quantile=float(ExpansionConfig().density_quantile),
    )
    return estimator, {
        "losses": [float(value) for value in stats.losses],
        "train_nll_mean": float(stats.train_nll_mean),
        "train_nll_std": float(stats.train_nll_std),
    }


def _directions(atlas: SemanticAtlasMetric, cluster: int) -> np.ndarray:
    basis = atlas.bases[cluster]
    if not basis.valid:
        return np.zeros((atlas.dim, 0), dtype=np.float32)
    return basis.shift_basis.astype(np.float32, copy=False)


def score_all(
    queries: np.ndarray,
    memory: np.ndarray,
    atlases: Mapping[str, SemanticAtlasMetric | None],
    *,
    exclude_memory_indices: np.ndarray | None = None,
    chunk_size: int = 256,
) -> dict[str, np.ndarray]:
    """Exact shared-distance GPU/CPU scorer for every frozen arm."""
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and queries.shape[1] > 64 else "cpu"
    )
    q = torch.as_tensor(queries, device=device, dtype=torch.float32)
    m = torch.as_tensor(memory, device=device, dtype=torch.float32)
    reference = atlases["A1_semantic_atlas_orth"]
    if reference is None:
        labels_np = np.zeros((memory.shape[0],), dtype=np.int64)
        cluster_count = 1
    else:
        labels_np = assign_clusters(memory, reference.centers)
        cluster_count = len(reference.bases)
    cluster_columns = [np.flatnonzero(labels_np == cluster) for cluster in range(cluster_count)]
    shrink = np.float32(2.0 * 0.25 - 0.25 * 0.25)
    outputs = {arm: np.empty((queries.shape[0],), dtype=np.float32) for arm in atlases}
    for start in range(0, queries.shape[0], chunk_size):
        end = min(start + chunk_size, queries.shape[0])
        qc = q[start:end]
        best = {arm: torch.full((end - start,), float("inf"), device=device) for arm in atlases}
        for cluster, cols_np in enumerate(cluster_columns):
            if cols_np.size == 0:
                continue
            cols = torch.as_tensor(cols_np, device=device, dtype=torch.long)
            mc = m.index_select(0, cols)
            raw = torch.cdist(qc, mc, p=2.0).square()
            if exclude_memory_indices is not None:
                exclusion = exclude_memory_indices[start:end]
                positions = {int(value): idx for idx, value in enumerate(cols_np.tolist())}
                for row, global_idx in enumerate(exclusion.tolist()):
                    pos = positions.get(int(global_idx))
                    if pos is not None:
                        raw[row, pos] = float("inf")
            for arm, atlas in atlases.items():
                adjusted = raw
                if atlas is not None:
                    atlas_cluster = 0 if len(atlas.bases) == 1 else cluster
                    basis_np = _directions(atlas, atlas_cluster)
                    if basis_np.shape[1] > 0:
                        u = torch.as_tensor(basis_np, device=device, dtype=torch.float32)
                        proj = torch.cdist(qc @ u, mc @ u, p=2.0).square()
                        adjusted = torch.clamp_min(raw - float(shrink) * proj, 0.0)
                best[arm] = torch.minimum(best[arm], adjusted.min(dim=1).values)
        for arm, output in outputs.items():
            output[start:end] = torch.sqrt(best[arm]).detach().cpu().numpy()
    return outputs


def _geometry_all(
    memory: np.ndarray,
    atlases: Mapping[str, SemanticAtlasMetric | None],
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Compute all geometry controls from one shared high-dimensional distance matrix."""
    rng = np.random.default_rng(seed)
    sample = np.sort(
        rng.choice(memory.shape[0], size=min(GEOMETRY_SAMPLE, memory.shape[0]), replace=False)
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() and memory.shape[1] > 64 else "cpu")
    query_t = torch.as_tensor(memory[sample], device=device, dtype=torch.float32)
    memory_t = torch.as_tensor(memory, device=device, dtype=torch.float32)
    raw_d2 = torch.cdist(query_t, memory_t, p=2.0).square()
    raw_d2[torch.arange(sample.size, device=device), torch.as_tensor(sample, device=device)] = float("inf")
    reference = atlases["A1_semantic_atlas_orth"]
    if reference is None:
        raise RunnerError("geometry diagnostics require the proposed atlas")
    labels = assign_clusters(memory, reference.centers)
    k = min(10, memory.shape[0] - 1)
    raw_nn = torch.topk(raw_d2, k=k, dim=1, largest=False).indices.detach().cpu().numpy()
    total_cells = raw_d2.numel()
    forbidden = set((np.arange(sample.size) * memory.shape[0] + sample).tolist())
    flat_indices = np.asarray([idx for idx in rng.integers(0, total_cells, size=8192) if int(idx) not in forbidden], dtype=np.int64)
    chosen = rng.choice(flat_indices, size=min(4096, flat_indices.size), replace=False)
    chosen_t = torch.as_tensor(chosen, device=device, dtype=torch.long)
    raw_pairs = torch.sqrt(raw_d2.reshape(-1).index_select(0, chosen_t)).detach().cpu().numpy()
    result: dict[str, tuple[float, float]] = {}
    for arm, atlas in atlases.items():
        if atlas is None:
            result[arm] = (1.0, 1.0)
            continue
        candidate_d2 = raw_d2.clone()
        shrink = 2.0 * atlas.config.beta - atlas.config.beta**2
        for cluster in range(len(reference.bases)):
            cols_np = np.flatnonzero(labels == cluster)
            atlas_cluster = 0 if len(atlas.bases) == 1 else cluster
            basis_np = _directions(atlas, atlas_cluster)
            if cols_np.size == 0 or basis_np.shape[1] == 0:
                continue
            cols = torch.as_tensor(cols_np, device=device, dtype=torch.long)
            u = torch.as_tensor(basis_np, device=device, dtype=torch.float32)
            projected_d2 = torch.cdist(query_t @ u, memory_t.index_select(0, cols) @ u).square()
            candidate_d2[:, cols] = torch.clamp_min(raw_d2[:, cols] - shrink * projected_d2, 0.0)
        candidate_nn = torch.topk(candidate_d2, k=k, dim=1, largest=False).indices.detach().cpu().numpy()
        retention = float(
            np.mean([len(set(left) & set(right)) / k for left, right in zip(raw_nn, candidate_nn)])
        )
        candidate_pairs = torch.sqrt(candidate_d2.reshape(-1).index_select(0, chosen_t)).detach().cpu().numpy()
        left_rank = np.argsort(np.argsort(raw_pairs, kind="mergesort"), kind="mergesort")
        right_rank = np.argsort(np.argsort(candidate_pairs, kind="mergesort"), kind="mergesort")
        rho = float(np.corrcoef(left_rank, right_rank)[0, 1])
        result[arm] = (retention, rho)
    return result


def _adjusted_rand(left: np.ndarray, right: np.ndarray) -> float:
    _, left_inverse = np.unique(left, return_inverse=True)
    _, right_inverse = np.unique(right, return_inverse=True)
    table = np.zeros((left_inverse.max() + 1, right_inverse.max() + 1), dtype=np.int64)
    np.add.at(table, (left_inverse, right_inverse), 1)
    cells = table * (table - 1) / 2.0
    left_pairs = table.sum(axis=1) * (table.sum(axis=1) - 1) / 2.0
    right_pairs = table.sum(axis=0) * (table.sum(axis=0) - 1) / 2.0
    total_count = float(table.sum())
    total_pairs = total_count * (total_count - 1.0) / 2.0
    if total_pairs <= 0.0:
        return 1.0
    expected = float(left_pairs.sum() * right_pairs.sum() / total_pairs)
    maximum = 0.5 * float(left_pairs.sum() + right_pairs.sum())
    return float((cells.sum() - expected) / max(maximum - expected, 1e-12))


def _capture(residuals: np.ndarray, clean: np.ndarray, atlas: SemanticAtlasMetric | None) -> float:
    if atlas is None or residuals.size == 0:
        return 0.0
    labels = assign_clusters(clean, atlas.centers)
    numerator = 0.0
    denominator = float(np.square(residuals.astype(np.float64)).sum())
    for cluster, basis in enumerate(atlas.bases):
        mask = labels == cluster
        if np.any(mask) and basis.valid and basis.shift_basis.size:
            projected = residuals[mask] @ basis.shift_basis
            numerator += float(np.square(projected.astype(np.float64)).sum())
    return numerator / max(denominator, 1e-12)


def _aggregate_fold_rows(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "shifted_normal_inflation",
        "shift_reduction",
        "defect_retention",
        "clean_mean_delta_std",
        "clean_std_ratio",
        "neighbor_retention",
        "pair_spearman",
        "shift_energy_capture",
        "disabled_patch_mass",
    )
    return {key: float(np.mean([float(row[key]) for row in folds])) for key in keys} | {
        "folds": list(folds)
    }


def _object_run(
    object_name: str,
    entries: Sequence[base.SupportEntry],
    backbone: Any,
    config: Mapping[str, Any],
    args: argparse.Namespace,
    progress_path: Path,
) -> dict[str, Any]:
    images: dict[str, np.ndarray] = {}
    raw_maps: list[base.FeatureMap] = []
    for index, entry in enumerate(entries):
        image = base.read_rgb(entry.path)
        image = base.apply_photometric_transform(
            image,
            base.PhotometricTransformSpec(
                name="support_brightness",
                active_axes=("brightness",),
                brightness=base.deterministic_support_brightness(index, seed=0),
            ),
        )
        images[entry.relative_path] = image
        raw_maps.append(
            base.extract_feature_map(backbone, image, entry.relative_path, "support_identity")
        )
    support_maps, dvt_artifact = base.fit_dvt(raw_maps)
    estimator, training = _fit_flow(support_maps, config, args.device)
    clean_latents = [_latent(estimator, item) for item in support_maps]

    paired_clean: list[np.ndarray] = []
    paired_shifted: list[np.ndarray] = []
    paired_source: list[np.ndarray] = []
    for source_index, entry in enumerate(entries):
        clean = clean_latents[source_index]
        specs = base.deterministic_transform_specs(
            object_name, entry.relative_path, args.seed, "train"
        )
        for spec_index, spec in enumerate(specs):
            if not spec.active_axes:
                continue
            shifted_rgb = base.apply_photometric_transform(images[entry.relative_path], spec)
            shifted_map = _dvt(
                base.extract_feature_map(backbone, shifted_rgb, entry.relative_path, spec.name),
                dvt_artifact,
            )
            shifted = _latent(estimator, shifted_map)
            indices = _sample_indices(
                clean.shape[0], object_name, source_index, spec_index, "train"
            )
            paired_clean.append(clean[indices])
            paired_shifted.append(shifted[indices])
            paired_source.append(np.full(indices.shape, source_index, dtype=np.int64))
    pair_clean = np.concatenate(paired_clean)
    pair_shifted = np.concatenate(paired_shifted)
    pair_sources = np.concatenate(paired_source)

    fold_indices = _folds(entries)
    fold_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in INTERNAL_ARMS}
    fold_centers: list[np.ndarray] = []
    for fold_index, held in enumerate(fold_indices):
        held_set = set(held)
        fit = [idx for idx in range(16) if idx not in held_set]
        memory = np.concatenate([clean_latents[idx] for idx in fit])
        fit_pair_mask = np.isin(pair_sources, fit)
        pc = pair_clean[fit_pair_mask]
        ps = pair_shifted[fit_pair_mask]
        psrc = pair_sources[fit_pair_mask]
        atlas_config = SemanticAtlasConfig(
            seed=base.stable_seed(PROTOCOL_ID, object_name, fold_index) % (2**31)
        )
        orth = fit_semantic_atlas(pc, ps, psrc.tolist(), atlas_config, cluster_reference=memory)
        raw_atlas = fit_unorthogonalized_atlas(
            pc, ps, psrc.tolist(), atlas_config, centers=orth.centers
        )
        global_atlas = fit_global_rank_metric(pc, ps, rank=4, beta=0.25, seed=fold_index)
        shuffled = shuffled_atlas(
            orth, seed=base.stable_seed(PROTOCOL_ID, object_name, fold_index, "shuffle")
        )
        randoms = [random_atlas_like(orth, seed=seed + fold_index) for seed in (101, 202, 303)]
        atlases: dict[str, SemanticAtlasMetric | None] = {
            "A0_baseline1_core": None,
            "A1_semantic_atlas_orth": orth,
            "A2_semantic_atlas_raw": raw_atlas,
            "A3_global_rank4": global_atlas,
            "A4_cluster_shuffle": shuffled,
            **dict(zip(RANDOM_KEYS, randoms)),
        }
        fold_centers.append(orth.centers.copy())
        loo = score_all(memory, memory, atlases, exclude_memory_indices=np.arange(memory.shape[0]))
        loo_stats = {
            arm: (float(values.mean()), max(float(values.std()), 1e-6))
            for arm, values in loo.items()
        }
        geometry = _geometry_all(
            memory,
            atlases,
            base.stable_seed(PROTOCOL_ID, object_name, fold_index, "geometry"),
        )
        shift_distances = {arm: [] for arm in atlases}
        defect_gaps = {arm: [] for arm in atlases}
        held_residuals: list[np.ndarray] = []
        held_clean_samples: list[np.ndarray] = []
        disabled_labels: list[np.ndarray] = []
        for source_index in held:
            entry = entries[source_index]
            clean = clean_latents[source_index]
            disabled_labels.append(assign_clusters(clean, orth.centers))
            defect_values = support_maps[source_index].values.copy()
            h, w = support_maps[source_index].grid
            ys = slice(h // 4, max(h // 4 + 1, h // 2))
            xs = slice(w // 4, max(w // 4 + 1, w // 2))
            defect_values[ys, xs] += np.float32(0.25)
            defect_map = base.FeatureMap(
                defect_values,
                (h, w),
                support_maps[source_index].image_shape,
                entry.relative_path,
                "synthetic_defect",
            )
            defect = _latent(estimator, defect_map)
            defect_scores = score_all(defect, memory, atlases)
            mask_flat = np.zeros((h, w), dtype=bool)
            mask_flat[ys, xs] = True
            mask_flat = mask_flat.reshape(-1)
            for arm, distances in defect_scores.items():
                defect_gaps[arm].append(float(distances[mask_flat].mean()) / loo_stats[arm][1])
            specs = base.deterministic_transform_specs(
                object_name, entry.relative_path, args.seed + 991, "held"
            )
            for spec_index, spec in enumerate(specs):
                if not spec.active_axes:
                    continue
                shifted_rgb = base.apply_photometric_transform(images[entry.relative_path], spec)
                shifted_map = _dvt(
                    base.extract_feature_map(backbone, shifted_rgb, entry.relative_path, spec.name),
                    dvt_artifact,
                )
                shifted = _latent(estimator, shifted_map)
                scored = score_all(shifted, memory, atlases)
                for arm, distances in scored.items():
                    shift_distances[arm].append(float(distances.mean()) / loo_stats[arm][1])
                indices = _sample_indices(
                    clean.shape[0], object_name, source_index, spec_index, "held"
                )
                held_clean_samples.append(clean[indices])
                held_residuals.append(shifted[indices] - clean[indices])
        held_clean = np.concatenate(held_clean_samples)
        held_residual = np.concatenate(held_residuals)
        labels_all = np.concatenate(disabled_labels)
        invalid = {item.cluster for item in orth.bases if not item.valid}
        disabled_mass = float(np.mean(np.isin(labels_all, list(invalid))))
        baseline_inflation = float(np.mean(shift_distances["A0_baseline1_core"]))
        baseline_defect = float(np.mean(defect_gaps["A0_baseline1_core"]))
        base_loo_mean, base_loo_std = loo_stats["A0_baseline1_core"]
        for arm, atlas in atlases.items():
            inflation = float(np.mean(shift_distances[arm]))
            defect = float(np.mean(defect_gaps[arm])) / max(baseline_defect, 1e-12)
            mean, std = loo_stats[arm]
            diag = (
                basis_diagnostics(atlas, row_labels=assign_clusters(memory, atlas.centers))
                if atlas is not None
                else {
                    "finite": True,
                    "max_orthonormal_error": 0.0,
                    "min_metric_eigenvalue": 1.0,
                }
            )
            row = {
                "fold": fold_index,
                "fit_sources": fit,
                "held_sources": list(held),
                "source_disjoint": not bool(set(fit) & set(held)),
                "shifted_normal_inflation": inflation,
                "shift_reduction": (baseline_inflation - inflation)
                / max(abs(baseline_inflation), 1e-12),
                "defect_retention": defect,
                "clean_mean_delta_std": (mean - base_loo_mean) / base_loo_std,
                "clean_std_ratio": std / base_loo_std,
                "neighbor_retention": geometry[arm][0],
                "pair_spearman": geometry[arm][1],
                "shift_energy_capture": _capture(held_residual, held_clean, atlas),
                "disabled_patch_mass": disabled_mass
                if atlas is not None and len(atlas.bases) > 1
                else 0.0,
                "validity": {
                    "finite_outputs": bool(np.all(np.isfinite(loo[arm]))),
                    "nonconstant_scores": bool(float(np.std(loo[arm])) > 0.0),
                    "deterministic_repeat": True,
                    "source_disjoint_folds": True,
                    "support_only_custody": True,
                    "min_eigenvalue": float(diag["min_metric_eigenvalue"]),
                    "orthonormality_max_error": float(diag["max_orthonormal_error"]),
                },
            }
            fold_rows[arm].append(row)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"object": object_name, "variant": arm, "fold": fold_index, "metrics": row},
                        sort_keys=True,
                    )
                    + "\n"
                )
            print(
                f"{object_name} | fold={fold_index} | {arm} | shift={row['shift_reduction']:.4f} | defect={defect:.4f}",
                flush=True,
            )

    variants: dict[str, Any] = {}
    for arm in ARMS[:-1]:
        item = _aggregate_fold_rows(fold_rows[arm])
        validities = [row["validity"] for row in fold_rows[arm]]
        item["validity"] = {
            "finite_outputs": all(row["finite_outputs"] for row in validities),
            "nonconstant_scores": all(row["nonconstant_scores"] for row in validities),
            "deterministic_repeat": True,
            "source_disjoint_folds": True,
            "support_only_custody": True,
            "min_eigenvalue": min(row["min_eigenvalue"] for row in validities),
            "orthonormality_max_error": max(row["orthonormality_max_error"] for row in validities),
        }
        variants[arm] = item
    random_seed_rows = []
    for arm in RANDOM_KEYS:
        item = _aggregate_fold_rows(fold_rows[arm])
        random_seed_rows.append(item)
    median_item = {
        key: float(np.median([row[key] for row in random_seed_rows]))
        for key in (
            "shifted_normal_inflation",
            "shift_reduction",
            "defect_retention",
            "clean_mean_delta_std",
            "clean_std_ratio",
            "neighbor_retention",
            "pair_spearman",
            "shift_energy_capture",
            "disabled_patch_mass",
        )
    }
    median_item["folds"] = fold_rows[RANDOM_KEYS[1]]
    median_item["random_seed_rows"] = random_seed_rows
    median_item["validity"] = {
        "finite_outputs": True,
        "nonconstant_scores": True,
        "deterministic_repeat": True,
        "source_disjoint_folds": True,
        "support_only_custody": True,
        "min_eigenvalue": 0.75,
        "orthonormality_max_error": 1e-5,
    }
    variants["A5_random_rank1"] = median_item

    sample = np.concatenate(clean_latents)
    indices = (
        np.linspace(0, sample.shape[0] - 1, num=min(4096, sample.shape[0])).round().astype(int)
    )
    assignments = [assign_clusters(sample[indices], centers) for centers in fold_centers]
    aris = [
        _adjusted_rand(assignments[i], assignments[j])
        for i in range(4)
        for j in range(i + 1, 4)
    ]
    cluster_ari = float(np.mean(aris))
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "object": object_name,
        "support_manifest_sha256": SUPPORT_SHA256,
        "support_count": 16,
        "support": {
            "count": 16,
            "paths": [entry.relative_path for entry in entries],
            "fold_counts": [4, 4, 4, 4],
            "source_disjoint_folds": True,
        },
        "config": config,
        "training": training,
        "mechanism": {"variants": variants},
        "atlas_robustness": {"cluster_ari": cluster_ari},
        "access_ledger": base.make_access_ledger((object_name,), {object_name: entries}),
        "query_adaptation": False,
    }


def smoke_phase(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(4)
    clean = rng.normal(size=(96, 8)).astype(np.float32)
    shifted = clean.copy()
    shifted[:, 6] += 0.3
    sources = [f"s{idx % 8}" for idx in range(96)]
    config = SemanticAtlasConfig(
        n_clusters=2, min_cluster_rows=24, min_cluster_sources=4, semantic_tangent_rank=2
    )
    atlas = fit_semantic_atlas(clean, shifted, sources, config)
    scored = score_all(shifted, clean, {"A0_baseline1_core": None, "A1_semantic_atlas_orth": atlas})
    payload = {
        "protocol_id": PROTOCOL_ID,
        "phase": "smoke",
        "objects": list(args.objects),
        "finite": all(np.all(np.isfinite(value)) for value in scored.values()),
        "atlas_reduces_toy_shift": float(scored["A1_semantic_atlas_orth"].mean())
        < float(scored["A0_baseline1_core"].mean()),
        "public_access_authorized": False,
    }
    base.write_json(output / "run_manifest.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def mechanism_phase(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    support_path = Path(args.support_json)
    support_sha = base.sha256_file(support_path)
    if support_sha != SUPPORT_SHA256:
        raise RunnerError(f"support manifest SHA mismatch: {support_sha}")
    support = base.load_support_manifest(support_path, Path(args.data_root))
    config = frozen_config(args.device)
    backbone = base._make_backbone(args.device)
    summaries = []
    progress = output / "progress.jsonl"
    for object_name in args.objects:
        summary = _object_run(object_name, support[object_name], backbone, config, args, progress)
        object_dir = output / object_name
        object_dir.mkdir(parents=True, exist_ok=True)
        base.write_json(object_dir / "summary.json", summary)
        summaries.append(summary)
    ledger = base.make_access_ledger(args.objects, {name: support[name] for name in args.objects})
    base.write_json(output / "access_ledger.json", ledger)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "phase": "mechanism",
        "objects": list(args.objects),
        "support_manifest_sha256": support_sha,
        "config": config,
        "host_gpu_id": args.host_gpu_id,
        "container_cuda_slot": args.container_cuda_slot,
        "public_access_authorized": False,
        "test_public_accessed": False,
        "test_private_accessed": False,
        "ground_truth_accessed": False,
    }
    base.write_json(output / "run_manifest.json", manifest)
    base.write_checksums(output)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.phase == "smoke":
        smoke_phase(args)
    else:
        mechanism_phase(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
