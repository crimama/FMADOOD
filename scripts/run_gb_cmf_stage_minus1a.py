#!/usr/bin/env python3
"""Run one frozen GB-CMF Stage -1A support-only case.

The runner can open only paths sealed in the exact SuperAD-16 train/good
manifest.  It has no dataset object and no public/private test enumeration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_cfntr_stage_minus1 as synthetic  # noqa: E402
from dinov3_backbone import DINOv3Backbone  # noqa: E402
from flow_tte_mvtec_ad2_core import (  # noqa: E402
    FeatureExtractionConfig,
    extract_feature_map_from_rgb,
)
from run_cfntr_stage_minus1 import (  # noqa: E402
    AXES,
    RESOLUTIONS,
    SupportEntry,
    load_support_manifest,
    mask_to_patch_grid,
    read_rgb,
)

from flow_tte.cfntr import apply_rgb_transform, make_axis_transform  # noqa: E402
from flow_tte.config import ExpansionConfig, FlowConfig, FlowTTEConfig, ScoreConfig  # noqa: E402
from flow_tte.denoising import PositionMeanArtifactDenoiser  # noqa: E402
from flow_tte.gb_cmf import (  # noqa: E402
    GBCMFCell,
    GBCMFConfig,
    ScoreNull,
    audit_flow_geometry,
    audit_proxy_fidelity,
    refine_flow,
    score_null,
    stable_seed,
    train_score_null,
)
from flow_tte.pipeline import FlowTTE  # noqa: E402

PROTOCOL_ID = "gb-cmf-stage-minus1a-support-only-20260719"
EXPECTED_SUPPORT_SHA256 = "9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5"
FEATURE_LAYERS = (7, 15, 23, 31)
DEFECT_FAMILIES = ("cutpaste", "chromatic_blotch", "gaussian_blur")
ARMS = ("default", "nonfactorial", "gb_cmf", "affine_null", "nonlinear_head_null")
ASSIGNMENTS = {
    ("rice", 0): (0, "cutpaste", "exposure", 0),
    ("can", 1): (1, "chromatic_blotch", "gamma", 1),
    ("rice", 1): (2, "gaussian_blur", "contrast", 2),
    ("can", 0): (3, "cutpaste", "white_balance", 3),
}
EXPECTED_HELD_FILES = {
    ("rice", 0): ("128_regular.png", "205_regular.png", "064_regular.png", "060_regular.png"),
    ("can", 1): ("322_regular.png", "107_regular.png", "262_regular.png", "217_regular.png"),
    ("rice", 1): ("273_regular.png", "091_regular.png", "093_regular.png", "036_regular.png"),
    ("can", 0): ("105_regular.png", "211_regular.png", "020_regular.png", "289_regular.png"),
}
DEFECT_RETRIES = 32


@dataclass(frozen=True)
class ComponentCell:
    source_id: int
    axis: str
    family: str
    clean: np.ndarray
    shifted_clean: np.ndarray
    defect: np.ndarray
    shifted_defect: np.ndarray
    defect_mask: np.ndarray
    background_mask: np.ndarray


@dataclass(frozen=True)
class HeldCell:
    source: str
    clean: np.ndarray
    shifted_clean: np.ndarray
    defect: np.ndarray
    shifted_defect: np.ndarray
    defect_mask: np.ndarray
    interior_mask: np.ndarray
    background_mask: np.ndarray
    generator: dict[str, Any]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--support-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--object", choices=("can", "rice"), required=True)
    parser.add_argument("--experiment-seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host-gpu-id", required=True)
    parser.add_argument("--container-cuda-slot", default="0")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(list(argv))
    key = (args.object, args.experiment_seed)
    if key not in ASSIGNMENTS:
        parser.error(f"case is not frozen in Stage -1A: {key}")
    if args.steps != 300:
        parser.error("Stage -1A freezes --steps=300")
    if int(args.host_gpu_id) != ASSIGNMENTS[key][3]:
        parser.error("case is assigned to a different dsba3 GPU")
    return args


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _write_arm_artifact(root: Path, arm: str, payload: dict[str, Any]) -> None:
    arm_root = root / "arms" / arm
    arm_root.mkdir(parents=True, exist_ok=False)
    temporary = arm_root / "result.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arm_root / "result.json")
    (arm_root / "result.sha256").write_text(
        _sha256(arm_root / "result.json") + "  result.json\n",
        encoding="utf-8",
    )
    (arm_root / "complete.txt").write_text("status=complete\n", encoding="utf-8")


def _load_complete_arm(root: Path, arm: str, theta0_sha256: str) -> dict[str, Any] | None:
    arm_root = root / "arms" / arm
    if not arm_root.exists():
        return None
    required = (arm_root / "result.json", arm_root / "result.sha256", arm_root / "complete.txt")
    if not all(path.is_file() for path in required):
        message = f"{arm}: incomplete resume artifact"
        raise RuntimeError(message)
    expected = (arm_root / "result.sha256").read_text(encoding="utf-8").split()[0]
    if _sha256(arm_root / "result.json") != expected:
        message = f"{arm}: resume checksum mismatch"
        raise RuntimeError(message)
    payload = json.loads((arm_root / "result.json").read_text(encoding="utf-8"))
    if payload.get("parent_theta0_sha256") != theta0_sha256:
        message = f"{arm}: foreign theta0 parent"
        raise RuntimeError(message)
    result = payload.get("result")
    if not isinstance(result, dict) or result.get("status") != "complete":
        message = f"{arm}: invalid resume result"
        raise RuntimeError(message)
    return result


def _read_whitelisted_image(
    entry: SupportEntry,
    whitelist: set[Path],
    opened: list[str],
) -> np.ndarray:
    resolved = entry.path.resolve(strict=True)
    if resolved not in whitelist:
        raise RuntimeError("image access sentinel rejected a non-whitelisted path")
    if any(part in {"test_public", "test_private", "ground_truth"} for part in resolved.parts):
        raise RuntimeError("image access sentinel rejected a forbidden path")
    image = read_rgb(resolved)
    opened.append(entry.relative_path)
    return image


def _sign(*parts: object) -> int:
    return 1 if stable_seed(PROTOCOL_ID, *parts) & 1 else -1


def _extract(
    backbone: DINOv3Backbone,
    image: np.ndarray,
    denoiser: PositionMeanArtifactDenoiser | None,
) -> np.ndarray:
    feature_map = extract_feature_map_from_rgb(
        backbone,
        image,
        FeatureExtractionConfig(feature_fusion="layer_norm_mean", context_source="none"),
    ).values
    if denoiser is not None:
        feature_map = denoiser.transform(feature_map)
    return np.asarray(feature_map, dtype=np.float32)


def _pipeline(seed: int, device: str) -> FlowTTE:
    return FlowTTE(
        FlowTTEConfig(
            flow=FlowConfig(
                n_coupling_layers=2,
                hidden_multiplier=1,
                n_epochs=3,
                lr=2e-4,
                clamp=1.9,
                batch_size=512,
                seed=seed,
                tail_weight=0.3,
                tail_top_k_ratio=0.05,
                lambda_logdet=1e-3,
            ),
            expansion=ExpansionConfig(budget=1.0, density_quantile=0.9, random_seed=seed),
            score=ScoreConfig(
                distance_weight=1.0,
                density_weight=0.25,
                top_percent=0.01,
                query_chunk_size=512,
                calibration_sample_size=0,
                loo_standardize=True,
            ),
            device=device,
        ),
    )


def _interior_background(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(mask, dtype=np.uint8)
    interior = cv2.erode(value, np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
    return interior.reshape(-1), (~value.astype(bool)).reshape(-1)


def _make_defect(
    image: np.ndarray,
    entry: SupportEntry,
    family: str,
    grid: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    for retry in range(DEFECT_RETRIES):
        try:
            defect, native_mask, metadata = synthetic.make_synthetic_defect(
                image,
                object_name=entry.object_name,
                relative_path=entry.relative_path,
                kind=family,
                feature_retry=retry,
            )
        except RuntimeError:
            continue
        patch_mask = np.asarray(mask_to_patch_grid(native_mask, grid), dtype=bool).reshape(grid)
        interior, background = _interior_background(patch_mask)
        if interior.any() and background.any():
            return defect, patch_mask.reshape(-1), interior, background, metadata
    message = f"{entry.relative_path}/{family}: no interior-capable defect"
    raise RuntimeError(message)


def _flat(feature_map: np.ndarray) -> np.ndarray:
    return feature_map.reshape(-1, feature_map.shape[-1]).astype(np.float32, copy=False)


def _components(
    pipeline: FlowTTE,
    features: np.ndarray,
    *,
    source_id: int | None = None,
    bank_source_ids: np.ndarray | None = None,
) -> np.ndarray:
    if source_id is None:
        return pipeline.factorial_score_components(features)
    if bank_source_ids is None:
        raise ValueError("source-excluded scoring requires bank source ids")
    query = np.full(features.shape[0], source_id, dtype=np.int64)
    return pipeline.factorial_score_components(
        features,
        query_source_ids=query,
        bank_source_ids=bank_source_ids,
    )


def _component_cell(
    pipeline: FlowTTE,
    cell: GBCMFCell,
    bank_source_ids: np.ndarray,
) -> ComponentCell:
    kwargs = {"source_id": cell.source_id, "bank_source_ids": bank_source_ids}
    return ComponentCell(
        source_id=cell.source_id,
        axis=cell.axis,
        family=cell.family,
        clean=_components(pipeline, cell.clean, **kwargs),
        shifted_clean=_components(pipeline, cell.shifted_clean, **kwargs),
        defect=_components(pipeline, cell.defect, **kwargs),
        shifted_defect=_components(pipeline, cell.shifted_defect, **kwargs),
        defect_mask=cell.defect_mask,
        background_mask=cell.background_mask,
    )


def _score(kind: str, components: np.ndarray, null: ScoreNull | None = None) -> np.ndarray:
    if kind in {"default", "nonfactorial", "gb_cmf"}:
        return np.asarray(components @ np.asarray([1.0, 0.25], dtype=np.float32))
    if null is None:
        raise RuntimeError("null arm has no fitted scorer")
    return score_null(null, components)


def _rank_loss(scores: np.ndarray, interior: np.ndarray, background: np.ndarray) -> float:
    background_scores = scores[background]
    k = max(1, math.ceil(background_scores.size * 0.05))
    top_background = np.partition(background_scores, -k)[-k:].mean()
    return float(np.logaddexp(0.0, 0.5 + top_background - scores[interior].mean()))


def _fixed_f1(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    predicted = scores >= threshold
    true_positive = int(np.sum(predicted & labels))
    false_positive = int(np.sum(predicted & ~labels))
    false_negative = int(np.sum(~predicted & labels))
    return 2.0 * true_positive / max(2 * true_positive + false_positive + false_negative, 1)


def _oracle_f1(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(labels.astype(np.uint8), scores)
    return float(np.max(2.0 * precision * recall / np.maximum(precision + recall, 1e-12)))


def _pixel_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    return {
        "pAUROC_0.05": float(roc_auc_score(labels, scores, max_fpr=0.05)),
        "fixed_F1": _fixed_f1(labels, scores, threshold),
        "oracle_F1_diagnostic": _oracle_f1(labels, scores),
    }


def _bounded_rows(values: np.ndarray, rows: int, *seed_parts: object) -> np.ndarray:
    if values.shape[0] <= rows:
        return values
    indices = np.random.default_rng(stable_seed(PROTOCOL_ID, *seed_parts)).choice(
        values.shape[0],
        size=rows,
        replace=False,
    )
    return values[np.sort(indices)]


def _std_ratio(candidate: np.ndarray, baseline: np.ndarray) -> float:
    candidate_std = float(np.std(candidate))
    baseline_std = float(np.std(baseline))
    if baseline_std < 1e-6:
        return 1.0 if candidate_std <= 1e-6 else float(np.finfo(np.float32).max)
    return candidate_std / baseline_std


def _build_cells(  # noqa: PLR0913
    backbone: DINOv3Backbone,
    denoiser: PositionMeanArtifactDenoiser,
    entries: Sequence[SupportEntry],
    images: dict[str, np.ndarray],
    clean_maps: dict[str, np.ndarray],
    axes: Sequence[str],
    families: Sequence[str],
    grid: tuple[int, int],
) -> tuple[list[GBCMFCell], list[dict[str, Any]]]:
    cells: list[GBCMFCell] = []
    audit: list[dict[str, Any]] = []
    for source_id, entry in enumerate(entries):
        image = images[entry.relative_path]
        clean = _flat(clean_maps[entry.relative_path])
        shifted_clean: dict[str, np.ndarray] = {}
        specs: dict[str, Any] = {}
        for axis in axes:
            spec = make_axis_transform(
                axis,
                sign=_sign(entry.object_name, entry.relative_path, axis),
                amplitude_scale=2.0,
            )
            specs[axis] = spec
            shifted_clean[axis] = _flat(
                _extract(backbone, apply_rgb_transform(image, spec), denoiser),
            )
        for family in families:
            defect_image, _, interior, background, metadata = _make_defect(
                image,
                entry,
                family,
                grid,
            )
            defect = _flat(_extract(backbone, defect_image, denoiser))
            for axis in axes:
                shifted_defect = _flat(
                    _extract(backbone, apply_rgb_transform(defect_image, specs[axis]), denoiser),
                )
                cells.append(
                    GBCMFCell(
                        source_id=source_id,
                        clean=clean,
                        shifted_clean=shifted_clean[axis],
                        defect=defect,
                        shifted_defect=shifted_defect,
                        defect_mask=interior,
                        background_mask=background,
                        axis=axis,
                        family=family,
                    ),
                )
            audit.append({"source": entry.relative_path, "family": family, **metadata})
    return cells, audit


def _build_held_cells(  # noqa: PLR0913
    backbone: DINOv3Backbone,
    denoiser: PositionMeanArtifactDenoiser,
    entries: Sequence[SupportEntry],
    images: dict[str, np.ndarray],
    clean_maps: dict[str, np.ndarray],
    axis: str,
    family: str,
    grid: tuple[int, int],
) -> list[HeldCell]:
    held: list[HeldCell] = []
    for entry in entries:
        image = images[entry.relative_path]
        spec = make_axis_transform(
            axis,
            sign=_sign(entry.object_name, entry.relative_path, axis),
            amplitude_scale=2.0,
        )
        shifted_clean = _flat(_extract(backbone, apply_rgb_transform(image, spec), denoiser))
        defect_image, defect_mask, interior, background, metadata = _make_defect(
            image,
            entry,
            family,
            grid,
        )
        defect = _flat(_extract(backbone, defect_image, denoiser))
        shifted_defect = _flat(
            _extract(backbone, apply_rgb_transform(defect_image, spec), denoiser),
        )
        held.append(
            HeldCell(
                source=entry.relative_path,
                clean=_flat(clean_maps[entry.relative_path]),
                shifted_clean=shifted_clean,
                defect=defect,
                shifted_defect=shifted_defect,
                defect_mask=defect_mask,
                interior_mask=interior,
                background_mask=background,
                generator=metadata,
            ),
        )
    return held


def _threshold(
    arm: str,
    pipeline: FlowTTE,
    cells: Sequence[GBCMFCell],
    bank_source_ids: np.ndarray,
    null: ScoreNull | None,
) -> float:
    normal_scores: list[np.ndarray] = []
    seen: set[tuple[int, str]] = set()
    for cell in cells:
        clean_key = (cell.source_id, "clean")
        if clean_key not in seen:
            normal_scores.append(
                _score(
                    arm,
                    _components(
                        pipeline,
                        cell.clean,
                        source_id=cell.source_id,
                        bank_source_ids=bank_source_ids,
                    ),
                    null,
                ),
            )
            seen.add(clean_key)
        shifted_key = (cell.source_id, cell.axis)
        if shifted_key not in seen:
            normal_scores.append(
                _score(
                    arm,
                    _components(
                        pipeline,
                        cell.shifted_clean,
                        source_id=cell.source_id,
                        bank_source_ids=bank_source_ids,
                    ),
                    null,
                ),
            )
            seen.add(shifted_key)
    return float(np.quantile(np.concatenate(normal_scores), 0.995))


def _evaluate_arm(  # noqa: PLR0913
    arm: str,
    pipeline: FlowTTE,
    default_pipeline: FlowTTE,
    held_cells: Sequence[HeldCell],
    train_features: np.ndarray,
    threshold: float,
    null: ScoreNull | None,
    opened_sources: set[str],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    for held in held_cells:
        component = {
            name: _components(pipeline, getattr(held, name))
            for name in ("clean", "shifted_clean", "defect", "shifted_defect")
        }
        scores = {name: _score(arm, values, null) for name, values in component.items()}
        default_component = _components(default_pipeline, held.shifted_defect)
        default_scores = _score("default", default_component)
        labels = held.defect_mask
        raw = _pixel_metrics(labels, scores["defect"], threshold)
        shifted = _pixel_metrics(labels, scores["shifted_defect"], threshold)
        component_std_ratios = [
            _std_ratio(component["shifted_defect"][:, index], default_component[:, index])
            for index in range(2)
        ]
        total_ratio = _std_ratio(scores["shifted_defect"], default_scores)
        if arm in {"nonfactorial", "gb_cmf"}:
            geometry = audit_flow_geometry(
                default_pipeline.estimator,  # type: ignore[arg-type]
                pipeline.estimator,  # type: ignore[arg-type]
                held.clean,
                sample_rows=256,
                seed=stable_seed(PROTOCOL_ID, held.source, arm, "geometry"),
            )
            proxy = audit_proxy_fidelity(
                pipeline.estimator,  # type: ignore[arg-type]
                pipeline.score_calibration,  # type: ignore[arg-type]
                _bounded_rows(
                    np.concatenate((held.clean, held.shifted_clean)),
                    256,
                    held.source,
                    arm,
                    "proxy-query",
                ),
                train_features,
            )
            geometry_payload = asdict(geometry)
            proxy_payload = asdict(proxy)
        else:
            geometry_payload = {"passed": True, "inherited_unchanged_flow": True}
            proxy_payload = {"passed": True, "inherited_unchanged_flow": True}
        component_pass = min(component_std_ratios) >= 0.5 and 0.8 <= total_ratio <= 1.25
        map_coverage = float(
            len(scores) == 4
            and all(value.shape == held.defect_mask.shape for value in scores.values()),
        )
        access_coverage = float(held.source in opened_sources)
        rows.append(
            {
                "source": held.source,
                "family": held.generator["kind"],
                "raw_pAUROC_0.05": raw["pAUROC_0.05"],
                "raw_fixed_F1": raw["fixed_F1"],
                "raw_oracle_F1_diagnostic": raw["oracle_F1_diagnostic"],
                "pAUROC_0.05": shifted["pAUROC_0.05"],
                "fixed_F1": shifted["fixed_F1"],
                "oracle_F1_diagnostic": shifted["oracle_F1_diagnostic"],
                "g_tail": float(
                    np.quantile(scores["shifted_clean"], 0.995)
                    - np.quantile(scores["clean"], 0.995),
                ),
                "g_drop": _rank_loss(
                    scores["shifted_defect"], held.interior_mask, held.background_mask,
                )
                - _rank_loss(scores["defect"], held.interior_mask, held.background_mask),
                "component_std_ratios": component_std_ratios,
                "component_std_ratio_min": min(component_std_ratios),
                "total_score_std_ratio": total_ratio,
                "geometry": geometry_payload,
                "geometry_pass": bool(geometry_payload["passed"] and component_pass),
                "proxy": proxy_payload,
                "proxy_pass": bool(proxy_payload["passed"]),
                "map_coverage": map_coverage,
                "access_coverage": access_coverage,
            },
        )
    macro = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in ("raw_pAUROC_0.05", "raw_fixed_F1", "pAUROC_0.05", "fixed_F1")
    }
    return rows, macro


def run_case(args: argparse.Namespace) -> dict[str, Any]:  # noqa: C901, PLR0915
    key = (args.object, args.experiment_seed)
    held_fold, held_family, held_axis, _ = ASSIGNMENTS[key]
    support_json = Path(args.support_json).resolve(strict=True)
    if _sha256(support_json) != EXPECTED_SUPPORT_SHA256:
        raise RuntimeError("support manifest checksum does not match frozen SuperAD-16")
    manifests = load_support_manifest(support_json, Path(args.data_root))
    manifest_hash_read_paths = [
        entry.relative_path
        for object_entries in manifests.values()
        for entry in object_entries
    ]
    entries = tuple(
        sorted(
            manifests[args.object],
            key=lambda entry: (entry.path_sha256, entry.relative_path),
        ),
    )
    train_entries = tuple(entry for entry in entries if entry.fold != held_fold)
    held_entries = tuple(entry for entry in entries if entry.fold == held_fold)
    held_files = tuple(entry.path.name for entry in held_entries)
    if len(train_entries) != 12 or held_files != EXPECTED_HELD_FILES[key]:
        message = f"frozen fold custody mismatch: {held_files}"
        raise RuntimeError(message)
    train_axes = tuple(axis for axis in AXES if axis != held_axis)
    train_families = tuple(family for family in DEFECT_FAMILIES if family != held_family)

    synthetic.PROTOCOL_ID = PROTOCOL_ID
    backbone = DINOv3Backbone(
        model_name="dinov3_vith16plus",
        device=args.device,
        smaller_edge_size=RESOLUTIONS[args.object],
        feature_layers=FEATURE_LAYERS,
    )
    whitelist = {entry.path.resolve(strict=True) for entry in entries}
    opened_paths: list[str] = []
    images = {
        entry.relative_path: _read_whitelisted_image(entry, whitelist, opened_paths)
        for entry in entries
    }
    if len(opened_paths) != 16 or set(opened_paths) != {entry.relative_path for entry in entries}:
        raise RuntimeError("image access sentinel did not observe exact support coverage")
    raw_train = [_extract(backbone, images[entry.relative_path], None) for entry in train_entries]
    denoiser = PositionMeanArtifactDenoiser.fit(raw_train, alpha=1.0)
    clean_maps = {
        entry.relative_path: denoiser.transform(raw_train[index])
        for index, entry in enumerate(train_entries)
    }
    for entry in held_entries:
        clean_maps[entry.relative_path] = _extract(backbone, images[entry.relative_path], denoiser)
    grid = tuple(int(value) for value in next(iter(clean_maps.values())).shape[:2])
    if any(tuple(value.shape[:2]) != grid for value in clean_maps.values()):
        raise RuntimeError("support feature grids are not aligned")

    train_cells, generator_audit = _build_cells(
        backbone,
        denoiser,
        train_entries,
        images,
        clean_maps,
        train_axes,
        train_families,
        grid,
    )
    held_cells = _build_held_cells(
        backbone,
        denoiser,
        held_entries,
        images,
        clean_maps,
        held_axis,
        held_family,
        grid,
    )
    clean_by_source = {
        index: _flat(clean_maps[entry.relative_path])
        for index, entry in enumerate(train_entries)
    }
    train_features = np.concatenate(list(clean_by_source.values())).astype(np.float32, copy=False)
    patches_per_source = int(np.prod(grid))
    bank_source_ids = np.repeat(np.arange(12, dtype=np.int64), patches_per_source)
    default_pipeline = _pipeline(stable_seed(PROTOCOL_ID, *key, "theta0"), args.device)
    flow_stats = default_pipeline.fit(train_features)
    theta0_sha256 = _state_sha256(default_pipeline.estimator.flow.state_dict())  # type: ignore[union-attr]
    frozen_calibration = default_pipeline.score_calibration
    default_memory_size = default_pipeline.memory.bank.size()  # type: ignore[union-attr]

    shared_schedule_seed = stable_seed(PROTOCOL_ID, *key, "shared-row-schedule")
    default_component_cells = [
        _component_cell(default_pipeline, cell, bank_source_ids) for cell in train_cells
    ]
    arm_results: dict[str, Any] = {}
    for arm in ARMS:
        if args.resume:
            resumed = _load_complete_arm(Path(args.output_root), arm, theta0_sha256)
            if resumed is not None:
                arm_results[arm] = resumed
                continue
        pipeline = default_pipeline
        null: ScoreNull | None = None
        training: dict[str, Any] = {"mode": "clean_only", "finite": True}
        if arm in {"nonfactorial", "gb_cmf"}:
            pipeline = default_pipeline.clone_offline()
            result = refine_flow(
                pipeline.estimator,  # type: ignore[arg-type]
                clean_by_source,
                train_cells,
                pipeline.score_calibration,  # type: ignore[arg-type]
                mode=arm,  # type: ignore[arg-type]
                config=GBCMFConfig(steps=args.steps),
                seed=shared_schedule_seed,
            )
            pipeline.refresh_static_memory(train_features)
            if (
                pipeline.score_calibration is not frozen_calibration
                and pipeline.score_calibration != frozen_calibration
            ):
                raise RuntimeError("flow arm changed frozen score calibration")
            training = asdict(result)
        elif arm in {"affine_null", "nonlinear_head_null"}:
            kind = "affine" if arm == "affine_null" else "nonlinear"
            null = train_score_null(
                default_component_cells,
                kind=kind,
                steps=args.steps,
                learning_rate=2e-5,
                seed=shared_schedule_seed,
                device=args.device,
                config=GBCMFConfig(steps=args.steps),
            )
            training = {"mode": null.kind, "finite": True}
        threshold = _threshold(arm, pipeline, train_cells, bank_source_ids, null)
        source_metrics, macro = _evaluate_arm(
            arm,
            pipeline,
            default_pipeline,
            held_cells,
            train_features,
            threshold,
            null,
            set(opened_paths),
        )
        arm_results[arm] = {
            "status": "complete",
            "threshold_q995": threshold,
            "training": training,
            "source_metrics": source_metrics,
            "macro_metrics": macro,
            "static_memory_size": pipeline.memory.bank.size(),  # type: ignore[union-attr]
            "static_memory_size_invariant": (
                pipeline.memory.bank.size() == default_memory_size  # type: ignore[union-attr]
            ),
        }
        pipeline_state_sha256 = _state_sha256(pipeline.estimator.flow.state_dict())  # type: ignore[union-attr]
        null_state_sha256 = None if null is None else _state_sha256(dict(null.state_dict))
        _write_arm_artifact(
            Path(args.output_root),
            arm,
            {
                "protocol_id": PROTOCOL_ID,
                "arm": arm,
                "parent_theta0_sha256": theta0_sha256,
                "pipeline_state_sha256": pipeline_state_sha256,
                "null_state_sha256": null_state_sha256,
                "result": arm_results[arm],
            },
        )

    return {
        "schema": "gb-cmf-stage-minus1a-case-v1",
        "protocol_id": PROTOCOL_ID,
        "support_manifest_sha256": EXPECTED_SUPPORT_SHA256,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "Stage_-1A_support_only_structural_falsification",
        "object": args.object,
        "seed": args.experiment_seed,
        "held_fold": held_fold,
        "held_family": held_family,
        "held_axis": held_axis,
        "held_sources": [entry.relative_path for entry in held_entries],
        "train_sources": [entry.relative_path for entry in train_entries],
        "train_axes": list(train_axes),
        "train_families": list(train_families),
        "grid": list(grid),
        "feature_dim": int(train_features.shape[1]),
        "flow_stats": asdict(flow_stats),
        "provenance": {
            "support_manifest_sha256": EXPECTED_SUPPORT_SHA256,
            "source_set_sha256": hashlib.sha256(
                "\n".join(
                    f"{entry.relative_path}\t{entry.content_sha256}" for entry in entries
                ).encode(),
            ).hexdigest(),
            "dvt_artifact_sha256": _array_sha256(denoiser.artifact),
            "train_feature_sha256": _array_sha256(train_features),
            "theta0_sha256": theta0_sha256,
            "cache_policy": "in_memory_only_no_cross_arm_post_flow_cache",
        },
        "generator_audit": generator_audit,
        "access_ledger": {
            "manifest_content_hash_read_count": len(manifest_hash_read_paths),
            "manifest_content_hash_read_paths": manifest_hash_read_paths,
            "opened_image_count": len(opened_paths),
            "opened_support_paths": opened_paths,
            "opened_path_set_sha256": hashlib.sha256(
                "\n".join(sorted(opened_paths)).encode(),
            ).hexdigest(),
            "whitelist_coverage": len(set(opened_paths)) / len(whitelist),
        },
        "fit_count": 5,
        "arms": arm_results,
        "gpu": int(args.host_gpu_id),
        "host_gpu_id": str(args.host_gpu_id),
        "container_cuda_slot": str(args.container_cuda_slot),
        "query_or_test_access": False,
        "test_access_attempts": 0,
        "public_metrics_computed": False,
        "access_attempts": {"public": 0, "test": 0, "private": 0, "ground_truth": 0},
        "claim_authorized": False,
        "stage_minus1b_authorized": False,
    }


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    output_root = Path(args.output_root)
    if args.resume:
        output_root.resolve(strict=True)
        if (output_root / "summary.json").exists():
            raise RuntimeError("cannot resume an already summarized case")
    else:
        output_root.mkdir(parents=True, exist_ok=False)
    payload = run_case(args)
    temporary = output_root / "summary.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_root / "summary.json")
    (output_root / "complete.txt").write_text("status=complete\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
