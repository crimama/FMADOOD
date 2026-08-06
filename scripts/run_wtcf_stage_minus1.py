#!/usr/bin/env python3
"""Run one frozen WTCF Stage -1 support-only object/seed case."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import tempfile
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_cfntr_stage_minus1 as synthetic  # noqa: E402
from run_dinov2l_default_mechanism_stage_m1 import (  # noqa: E402
    _extract_maps,
    _flow_config,
    make_defect,
)

from flow_tte.aupro import aupro_score  # noqa: E402
from flow_tte.cfntr import apply_rgb_transform, make_axis_transform  # noqa: E402
from flow_tte.dinov3_early_exit import patch_torch_pytree_compat  # noqa: E402
from flow_tte.memory import TorchMemoryBank  # noqa: E402
from flow_tte.trainer import FlowDensityEstimator  # noqa: E402
from flow_tte.wtcf import (  # noqa: E402
    WTCFConfig,
    make_image_disjoint_folds,
    refine_wtcf,
    result_to_dict,
    soft_nearest_distance,
    stable_seed,
)
from fmad.backbones.dinov2 import DINOv2Backbone  # noqa: E402

patch_torch_pytree_compat()

PROTOCOL_ID = "wtcf-stage-minus1-support-only-20260804"
OBJECTS = ("can", "fruit_jelly", "rice", "fabric", "sheet_metal")
SEEDS = (0, 1, 2)
ARMS = (
    "a6",
    "a7_identity",
    "pca_residual",
    "shrinkage_mahalanobis",
    "crossfit_only",
    "global_isometry",
    "random_witness",
    "witness_only",
    "wtcf",
)
FLOW_MODES = ARMS[4:]
LAYERS = (5, 11, 17, 23)
AXES = ("exposure", "gamma", "contrast", "white_balance")
DEFECTS = ("cutpaste", "chromatic_blotch", "gaussian_blur", "smooth_local_warp")
RESOLUTIONS = {name: (448 if name == "sheet_metal" else 672) for name in OBJECTS}
SUPPORT_COUNT = 16
DIAGNOSTIC_COUNT = 4
PCA_RANK = 32
PCA_ROWS = 4096
MAHALANOBIS_SHRINKAGE = 0.10


@dataclass(frozen=True)
class LinearNull:
    kind: str
    center: np.ndarray
    transform: np.ndarray
    bank: np.ndarray | None


@dataclass
class FlowArm:
    estimator: FlowDensityEstimator
    bank: TorchMemoryBank
    bank_source_ids: torch.Tensor


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--fixed-support-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--object", choices=OBJECTS, required=True)
    parser.add_argument("--support-seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--host-gpu-id", default="3")
    parser.add_argument("--container-cuda-slot", default="0")
    args = parser.parse_args(list(argv))
    if args.steps != 300:
        parser.error("the frozen real protocol requires --steps=300")
    return args


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    except BaseException:
        with suppress(FileNotFoundError):
            Path(temporary).unlink()
        raise


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _prepare_support_manifest(
    data_root: Path,
    fixed_json: Path,
    object_name: str,
    support_seed: int,
    output_root: Path,
) -> tuple[Path, ...]:
    train_root = (data_root / object_name / "train" / "good").resolve(strict=True)
    candidates = tuple(sorted(path.resolve() for path in train_root.glob("*.png")))
    if len(candidates) < SUPPORT_COUNT:
        message = f"{object_name}: fewer than 16 train/good images"
        raise RuntimeError(message)
    if support_seed == 0:
        fixed = json.loads(fixed_json.read_text(encoding="utf-8"))
        names = [Path(value).name for value in fixed[object_name]]
        selected = tuple((train_root / name).resolve(strict=True) for name in names)
    else:
        selected = tuple(
            sorted(
                candidates,
                key=lambda path: _sha256_bytes(
                    f"{PROTOCOL_ID}\t{object_name}\t{support_seed}\t{path.name}".encode(),
                ),
            )[:SUPPORT_COUNT],
        )
    if len(selected) != SUPPORT_COUNT or len(set(selected)) != SUPPORT_COUNT:
        raise RuntimeError("support manifest is not an exact unique 16-shot set")
    folds = make_image_disjoint_folds(
        np.arange(SUPPORT_COUNT),
        fold_count=4,
        seed=stable_seed(PROTOCOL_ID, object_name, support_seed, "folds"),
    )
    rows = [
        {
            "source_id": source,
            "fold": folds[source],
            "filename": path.name,
            "relative_path": str(path.relative_to(data_root)),
            "path_sha256": _sha256_bytes(str(path.relative_to(data_root)).encode()),
        }
        for source, path in enumerate(selected)
    ]
    _atomic_json(output_root / "sealed_support_manifest.json", rows)
    return selected


def _read_rgb(path: Path, whitelist: set[Path], ledger: list[str]) -> np.ndarray:
    resolved = path.resolve(strict=True)
    if resolved not in whitelist:
        message = f"non-whitelisted image read: {resolved}"
        raise PermissionError(message)
    image = cv2.imread(str(resolved), cv2.IMREAD_COLOR)
    if image is None:
        message = f"failed to read support image: {resolved}"
        raise RuntimeError(message)
    ledger.append(resolved.name)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _flat(value: np.ndarray) -> np.ndarray:
    return value.reshape(-1, value.shape[-1]).astype(np.float32, copy=False)


def _fit_a6(clean_maps: Mapping[int, np.ndarray], device: str, seed: int) -> FlowDensityEstimator:
    maps = [clean_maps[source] for source in sorted(clean_maps)]
    support = np.stack(maps).astype(np.float32, copy=False)
    config = replace(_flow_config("flow"), seed=seed)
    estimator = FlowDensityEstimator(support.shape[-1], config, device)
    estimator.fit(support, density_quantile=0.9)
    return estimator


def _make_flow_arm(
    estimator: FlowDensityEstimator,
    clean_by_source: Mapping[int, np.ndarray],
) -> FlowArm:
    features = np.concatenate([clean_by_source[source] for source in sorted(clean_by_source)])
    source_ids = np.concatenate(
        [
            np.full(clean_by_source[source].shape[0], source, dtype=np.int64)
            for source in sorted(clean_by_source)
        ],
    )
    latent = estimator.transform(features)
    bank = TorchMemoryBank()
    bank.fit(latent)
    return FlowArm(estimator, bank, torch.as_tensor(source_ids, device=estimator.device))


def _flow_score(arm: FlowArm, features: np.ndarray, source_id: int) -> np.ndarray:
    evaluation = arm.estimator.evaluate(features)
    keep = arm.bank_source_ids != source_id
    bank = TorchMemoryBank()
    bank.fit(arm.bank.features[keep])
    result = bank.query(evaluation.z, k=1, chunk_size=512, squared=False)
    return result.distances[:, 0].detach().cpu().numpy().astype(np.float32, copy=False)


def _standardized_raw(estimator: FlowDensityEstimator, features: np.ndarray) -> np.ndarray:
    if estimator.standardizer is None:
        raise RuntimeError("A6 standardizer is missing")
    tensor = torch.as_tensor(features, device=estimator.device)
    return estimator.standardizer.transform(tensor).detach().cpu().numpy().astype(np.float32)


def _fit_linear_null(
    kind: str,
    estimator: FlowDensityEstimator,
    clean_by_source: Mapping[int, np.ndarray],
    excluded_source: int,
    seed: int,
) -> LinearNull:
    train = np.concatenate(
        [values for source, values in clean_by_source.items() if source != excluded_source],
    )
    train = _standardized_raw(estimator, train)
    if kind == "pca_residual":
        if train.shape[0] > PCA_ROWS:
            indices = np.sort(
                np.random.default_rng(seed).choice(train.shape[0], PCA_ROWS, replace=False),
            )
            train = train[indices]
        center = train.mean(axis=0).astype(np.float32)
        centered = train - center
        tensor = torch.as_tensor(centered, device=estimator.device)
        torch.manual_seed(seed)
        rank = min(PCA_RANK, tensor.shape[0] - 1, tensor.shape[1])
        _, _, basis = torch.pca_lowrank(tensor, q=rank, center=False, niter=2)
        return LinearNull(kind, center, basis.detach().cpu().numpy().astype(np.float32), None)
    center = train.mean(axis=0).astype(np.float32)
    centered = train - center
    variance = centered.var(axis=0).astype(np.float32)
    target = float(variance.mean())
    shrunk = (1.0 - MAHALANOBIS_SHRINKAGE) * variance + MAHALANOBIS_SHRINKAGE * target
    scale = np.reciprocal(np.sqrt(np.maximum(shrunk, 1e-6))).astype(np.float32)
    bank = centered * scale
    return LinearNull(kind, center, scale, bank.astype(np.float32, copy=False))


def _linear_score(
    null: LinearNull,
    estimator: FlowDensityEstimator,
    features: np.ndarray,
) -> np.ndarray:
    values = _standardized_raw(estimator, features) - null.center
    if null.kind == "pca_residual":
        projected = values @ null.transform
        residual = values - projected @ null.transform.T
        return np.linalg.norm(residual, axis=1).astype(np.float32)
    if null.bank is None:
        raise RuntimeError("nearest-neighbor linear null is missing its bank")
    query = torch.as_tensor(values * null.transform, device=estimator.device)
    bank = TorchMemoryBank()
    bank.fit(torch.as_tensor(null.bank, device=estimator.device))
    result = bank.query(query, k=1, chunk_size=512, squared=False)
    return result.distances[:, 0].detach().cpu().numpy().astype(np.float32)


def _score(  # noqa: PLR0913
    arm_name: str,
    flow_arms: Mapping[str, FlowArm],
    nulls: Mapping[tuple[str, int], LinearNull],
    a6: FlowDensityEstimator,
    features: np.ndarray,
    source_id: int,
) -> np.ndarray:
    if arm_name in flow_arms:
        return _flow_score(flow_arms[arm_name], features, source_id)
    if arm_name == "a7_identity":
        null = nulls[(arm_name, source_id)]
        return _linear_score(null, a6, features)
    return _linear_score(nulls[(arm_name, source_id)], a6, features)


def _fixed_f1(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    predicted = scores >= threshold
    tp = int(np.sum(predicted & labels))
    fp = int(np.sum(predicted & ~labels))
    fn = int(np.sum(~predicted & labels))
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def _proxy_audit(
    arm: FlowArm,
    clean_by_source: Mapping[int, np.ndarray],
    diagnostic_sources: Sequence[int],
    seed: int,
) -> dict[str, float | bool]:
    soft_parts: list[np.ndarray] = []
    hard_parts: list[np.ndarray] = []
    for source in diagnostic_sources:
        query_raw = clean_by_source[source]
        count = min(64, query_raw.shape[0])
        indices = np.sort(
            np.random.default_rng(stable_seed(seed, source)).choice(
                query_raw.shape[0], count, replace=False,
            ),
        )
        evaluation = arm.estimator.evaluate(query_raw[indices])
        keep = arm.bank_source_ids != source
        bank_z = arm.bank.features[keep]
        with torch.no_grad():
            soft = soft_nearest_distance(evaluation.z, bank_z, tau=0.05)
            hard = torch.cdist(evaluation.z, bank_z).min(dim=1).values
        soft_parts.append(soft.cpu().numpy())
        hard_parts.append(hard.cpu().numpy())
    soft_values = np.concatenate(soft_parts)
    hard_values = np.concatenate(hard_parts)
    soft_rank = np.argsort(np.argsort(soft_values, kind="mergesort"), kind="mergesort")
    hard_rank = np.argsort(np.argsort(hard_values, kind="mergesort"), kind="mergesort")
    spearman = float(np.corrcoef(soft_rank, hard_rank)[0, 1])
    top_count = max(1, math.ceil(soft_values.size * 0.005))
    soft_top = set(np.argpartition(soft_values, -top_count)[-top_count:].tolist())
    hard_top = set(np.argpartition(hard_values, -top_count)[-top_count:].tolist())
    overlap = len(soft_top & hard_top) / top_count
    scale = max(float(np.std(hard_values)), 1e-6)
    qdiff = abs(float(np.quantile(soft_values, 0.995) - np.quantile(hard_values, 0.995))) / scale
    return {
        "spearman": spearman,
        "top_0_5pct_overlap": overlap,
        "normalized_q995_difference": qdiff,
        "passed": bool(spearman >= 0.98 and overlap >= 0.90 and qdiff <= 0.01),
    }


def _diagnostic_sources(object_name: str, support_seed: int) -> tuple[int, ...]:
    folds = make_image_disjoint_folds(
        np.arange(SUPPORT_COUNT),
        fold_count=4,
        seed=stable_seed(PROTOCOL_ID, object_name, support_seed, "folds"),
    )
    return tuple(
        min(source for source, value in folds.items() if value == fold)
        for fold in range(4)
    )


def _build_posthoc_maps(  # noqa: PLR0913
    backbone: DINOv2Backbone,
    object_name: str,
    source_id: int,
    path: Path,
    image: np.ndarray,
    clean_map: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, object]]:
    axis = AXES[source_id % len(AXES)]
    family = DEFECTS[source_id % len(DEFECTS)]
    sign = -1 if stable_seed(PROTOCOL_ID, object_name, source_id, "shift-sign") % 2 else 1
    transform = make_axis_transform(axis, sign=sign, amplitude_scale=2.0)
    shifted_image = apply_rgb_transform(image, transform)
    _, shifted, shifted_grid = _extract_maps(backbone, shifted_image)
    selected: tuple[np.ndarray, np.ndarray, dict[str, Any]] | None = None
    for retry in range(synthetic.DEFECT_GENERATION_ATTEMPTS):
        try:
            defect_image, native_mask, metadata = make_defect(
                image,
                object_name=object_name,
                relative_path=path.name,
                kind=family,
                feature_retry=retry,
            )
        except RuntimeError:
            continue
        _, defect, defect_grid = _extract_maps(backbone, defect_image)
        patch_mask = np.asarray(
            synthetic.mask_to_patch_grid(native_mask, defect_grid),
            dtype=bool,
        ).reshape(defect_grid)
        if patch_mask.any() and not patch_mask.all() and float(np.ptp(defect - clean_map)) > 0:
            selected = defect_image, np.asarray(patch_mask, dtype=bool), metadata
            defect_map = defect
            break
    if selected is None:
        message = f"{object_name}/{path.name}: no valid post-hoc defect"
        raise RuntimeError(message)
    defect_image, patch_mask, metadata = selected
    _, shifted_defect, sd_grid = _extract_maps(
        backbone,
        apply_rgb_transform(defect_image, transform),
    )
    if shifted_grid != clean_map.shape[:2] or sd_grid != clean_map.shape[:2]:
        raise RuntimeError("post-hoc transform changed patch grid")
    return {
        "C": _flat(clean_map),
        "S": _flat(shifted),
        "D": _flat(defect_map),
        "SD": _flat(shifted_defect),
    }, patch_mask, {"axis": axis, "sign": sign, "family": family, **metadata}


def run_case(args: argparse.Namespace) -> dict[str, object]:  # noqa: C901, PLR0915
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    data_root = Path(args.data_root).resolve(strict=True)
    fixed_json = Path(args.fixed_support_json).resolve(strict=True)
    paths = _prepare_support_manifest(
        data_root, fixed_json, args.object, args.support_seed, output_root,
    )
    manifest_hash = _sha256_file(output_root / "sealed_support_manifest.json")
    whitelist = set(paths)
    ledger: list[str] = []
    images = {source: _read_rgb(path, whitelist, ledger) for source, path in enumerate(paths)}
    if len(ledger) != SUPPORT_COUNT or len(set(ledger)) != SUPPORT_COUNT:
        raise RuntimeError("selected-support access coverage is incomplete")

    backbone = DINOv2Backbone(
        "dinov2_vitl14",
        args.device,
        smaller_edge_size=RESOLUTIONS[args.object],
        feature_layers=LAYERS,
    )
    clean_maps: dict[int, np.ndarray] = {}
    for source in range(SUPPORT_COUNT):
        _, fused, _ = _extract_maps(backbone, images[source])
        clean_maps[source] = fused
    if len({value.shape for value in clean_maps.values()}) != 1:
        raise RuntimeError("support feature maps are not grid-aligned")
    clean_by_source = {source: _flat(value) for source, value in clean_maps.items()}
    a6 = _fit_a6(
        clean_maps,
        args.device,
        stable_seed(PROTOCOL_ID, args.object, args.support_seed, "a6"),
    )
    flow_arms: dict[str, FlowArm] = {"a6": _make_flow_arm(a6, clean_by_source)}
    training: dict[str, object] = {"a6": {"mode": "nll_only", "finite": True}}
    refinement_config = WTCFConfig(steps=args.steps)
    for mode in FLOW_MODES:
        candidate = copy.deepcopy(a6)
        result = refine_wtcf(
            candidate,
            clean_by_source,
            mode=mode,  # type: ignore[arg-type]
            config=refinement_config,
            seed=stable_seed(PROTOCOL_ID, args.object, args.support_seed, "shared-refinement"),
        )
        flow_arms[mode] = _make_flow_arm(candidate, clean_by_source)
        training[mode] = result_to_dict(result)

    diagnostic_sources = _diagnostic_sources(args.object, args.support_seed)
    nulls: dict[tuple[str, int], LinearNull] = {}
    for source in diagnostic_sources:
        # Identity and diagonal Mahalanobis are both exact hard-1NN scorers;
        # identity uses unit scale after the frozen A6 standardizer.
        dimension = next(iter(clean_by_source.values())).shape[1]
        train_identity = np.concatenate(
            [
                _standardized_raw(a6, value)
                for key, value in clean_by_source.items()
                if key != source
            ],
        )
        nulls[("a7_identity", source)] = LinearNull(
            "shrinkage_mahalanobis",
            np.zeros(dimension, dtype=np.float32),
            np.ones(dimension, dtype=np.float32),
            train_identity,
        )
        for kind in ("pca_residual", "shrinkage_mahalanobis"):
            nulls[(kind, source)] = _fit_linear_null(
                kind,
                a6,
                clean_by_source,
                source,
                stable_seed(PROTOCOL_ID, args.object, args.support_seed, source, kind),
            )

    posthoc: dict[int, tuple[dict[str, np.ndarray], np.ndarray, dict[str, object]]] = {}
    # All candidate states and hashes are frozen before the first synthetic map.
    state_hashes = {
        arm: _sha256_bytes(
            b"".join(
                value.detach().cpu().contiguous().numpy().tobytes()
                for _, value in sorted(flow_arm.estimator.flow.state_dict().items())
            ),
        )
        for arm, flow_arm in flow_arms.items()
    }
    for source in diagnostic_sources:
        posthoc[source] = _build_posthoc_maps(
            backbone,
            args.object,
            source,
            paths[source],
            images[source],
            clean_maps[source],
        )

    scores: dict[str, dict[int, dict[str, np.ndarray]]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for source in diagnostic_sources:
            maps, _, _ = posthoc[source]
            scores[arm][source] = {
                condition: _score(arm, flow_arms, nulls, a6, values, source)
                for condition, values in maps.items()
            }
    a6_clean = np.concatenate([scores["a6"][source]["C"] for source in diagnostic_sources])
    a6_center = float(np.median(a6_clean))
    a6_scale = max(float(np.std(a6_clean)), 1e-6)
    arm_results: dict[str, object] = {}
    for arm in ARMS:
        clean_scores = np.concatenate([scores[arm][source]["C"] for source in diagnostic_sources])
        threshold = float(np.quantile(clean_scores, 0.995))
        masks = np.stack([posthoc[source][1] for source in diagnostic_sources])
        defect_maps = np.stack(
            [scores[arm][source]["D"].reshape(masks.shape[1:]) for source in diagnostic_sources],
        ).astype(np.float32)
        shifted_defect_maps = np.stack(
            [scores[arm][source]["SD"].reshape(masks.shape[1:]) for source in diagnostic_sources],
        ).astype(np.float32)
        labels = masks.reshape(-1)
        defect_flat = defect_maps.reshape(-1)
        shifted_defect_flat = shifted_defect_maps.reshape(-1)
        finite = all(
            bool(np.isfinite(value).all()) and float(np.std(value)) > 0.0
            for source_scores in scores[arm].values()
            for value in source_scores.values()
        )
        metrics = {
            "held_normal_q995_a6_units": float(
                (np.quantile(clean_scores, 0.995) - a6_center) / a6_scale,
            ),
            "shifted_normal_q995_a6_units": float(
                (
                    np.quantile(
                        np.concatenate(
                            [scores[arm][s]["S"] for s in diagnostic_sources],
                        ),
                        0.995,
                    )
                    - a6_center
                )
                / a6_scale,
            ),
            "pAUROC_0.05": float(roc_auc_score(labels, shifted_defect_flat, max_fpr=0.05)),
            "fixed_F1": _fixed_f1(labels, shifted_defect_flat, threshold),
            "AUPRO_0.05": float(aupro_score(masks, shifted_defect_maps, 0.05, 200)),
            "raw_pAUROC_0.05": float(roc_auc_score(labels, defect_flat, max_fpr=0.05)),
            "raw_fixed_F1": _fixed_f1(labels, defect_flat, threshold),
            "raw_AUPRO_0.05": float(aupro_score(masks, defect_maps, 0.05, 200)),
        }
        proxy = (
            _proxy_audit(
                flow_arms[arm],
                clean_by_source,
                diagnostic_sources,
                stable_seed(PROTOCOL_ID, args.object, args.support_seed, arm, "proxy"),
            )
            if arm in flow_arms
            else {"passed": True, "not_applicable": True}
        )
        arm_results[arm] = {
            "metrics": metrics,
            "threshold_q995": threshold,
            "finite_nonconstant_complete": finite,
            "proxy": proxy,
            "training": training.get(arm, {"mode": "fixed_null", "finite": True}),
            "state_sha256": state_hashes.get(arm),
        }

    return {
        "schema": "wtcf-stage-minus1-case-v1",
        "protocol_id": PROTOCOL_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "object": args.object,
        "support_seed": args.support_seed,
        "support_manifest_sha256": manifest_hash,
        "diagnostic_sources": list(diagnostic_sources),
        "grid": list(next(iter(clean_maps.values())).shape[:2]),
        "feature_dim": int(next(iter(clean_maps.values())).shape[-1]),
        "arms": arm_results,
        "access_ledger": {
            "selected_support_count": SUPPORT_COUNT,
            "opened_support_count": len(ledger),
            "opened_support_names": ledger,
            "coverage": len(set(ledger)) / SUPPORT_COUNT,
            "public_private_test_ground_truth_attempts": 0,
        },
        "synthetic_generation_after_all_arm_states_frozen": True,
        "state_hashes_before_synthetic": state_hashes,
        "host_gpu_id": args.host_gpu_id,
        "container_cuda_slot": args.container_cuda_slot,
        "query_adaptation": False,
        "memory_expansion": False,
        "claim_authorized": False,
    }


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    root = Path(args.output_root)
    payload = run_case(args)
    _atomic_json(root / "summary.json", payload)
    (root / "complete.txt").write_text("status=complete\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
