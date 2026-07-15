"""Pure evaluation and sequencing helpers for the FlowTTE resolution ladder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import ndimage

from flow_tte_gap_decomposition import boundary_tolerant_counts, harmonic, safe_ratio
from flow_tte_phase3_scorer_suite import NativeMapRecord, evaluate_native_records

RESOLUTIONS = (672, 896, 1120)
BOUNDARY_TOLERANCES = (0, 4, 8)
STRUCTURE_8 = np.ones((3, 3), dtype=np.uint8)


def token_grid_shape(resolution: int, patch_size: int = 16) -> tuple[int, int]:
    """Return the square full-frame token grid, rejecting partial patches."""
    if resolution <= 0 or patch_size <= 0 or resolution % patch_size:
        raise ValueError("resolution and patch size must be positive and evenly divisible")
    side = resolution // patch_size
    return side, side


def small_component_counts(
    prediction: np.ndarray,
    gt: np.ndarray,
    native_shape: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Count hit sub-672-token GT components using each image's native scale."""
    shape = tuple(native_shape or gt.shape)
    native_px_per_672_token = min(int(shape[0]), int(shape[1])) / 42.0
    area_limit = native_px_per_672_token**2
    labels, count = ndimage.label(np.asarray(gt, dtype=bool), structure=STRUCTURE_8)
    hit = total = 0
    areas: list[int] = []
    for component_id in range(1, count + 1):
        mask = labels == component_id
        area = int(np.count_nonzero(mask))
        if area < area_limit:
            total += 1
            hit += int(bool(np.any(np.asarray(prediction, dtype=bool)[mask])))
            areas.append(area)
    return {
        "hit": hit,
        "total": total,
        "recall": safe_ratio(hit, total),
        "area_limit_native_px": area_limit,
        "native_px_per_672_token": native_px_per_672_token,
        "small_component_areas_native_px": areas,
    }


def evaluate_resolution_records(records: Sequence[NativeMapRecord]) -> dict[str, Any]:
    """Extend the Phase-3 pooled evaluator with Phase-5 spatial diagnostics."""
    metrics = evaluate_native_records(records)
    threshold = np.float16(metrics["oracle_threshold_float16"])
    boundary: dict[str, Any] = {}
    small_hit = small_total = 0
    for tolerance in BOUNDARY_TOLERANCES:
        counts = np.sum(
            [
                boundary_tolerant_counts(
                    np.asarray(row.score, dtype=np.float16) >= threshold,
                    row.gt,
                    tolerance,
                )
                for row in records
            ],
            axis=0,
        )
        precision = safe_ratio(int(counts[0]), int(counts[1]))
        recall = safe_ratio(int(counts[2]), int(counts[3]))
        boundary[str(tolerance)] = {
            "precision": precision,
            "recall": recall,
            "f1": harmonic(precision, recall),
        }
    for row in records:
        prediction = np.asarray(row.score, dtype=np.float16) >= threshold
        counts = small_component_counts(prediction, row.gt, row.gt.shape)
        small_hit += int(counts["hit"])
        small_total += int(counts["total"])
    metrics["boundary_tolerant_f1_native_px"] = boundary
    metrics["small_defect_component_recall_at_oracle"] = safe_ratio(small_hit, small_total)
    metrics["small_defect_components_hit"] = small_hit
    metrics["small_defect_components_total"] = small_total
    metrics["small_defect_definition"] = {
        "connectivity": 8,
        "reference_grid": "672/16=42 tokens on native shorter edge",
        "criterion": "native component area < (min(native_h,native_w)/42)^2",
    }
    return metrics


@dataclass(frozen=True)
class StageResult:
    resolution: int
    valid: bool
    leaderboard: Path
    summary: Mapping[str, Any]


def run_resolution_sequence(
    execute_stage: Callable[[int], StageResult],
    resolutions: Sequence[int] = RESOLUTIONS,
) -> list[StageResult]:
    """Run validated stages strictly resolution-major and stop on invalidity."""
    completed: list[StageResult] = []
    for resolution in resolutions:
        result = execute_stage(int(resolution))
        if result.resolution != resolution:
            raise RuntimeError("stage executor returned the wrong resolution")
        completed.append(result)
        if not result.valid:
            break
        if not result.leaderboard.is_file():
            raise RuntimeError(f"stage {resolution} did not write its leaderboard")
    return completed


def keep_gate_status(stage_means: Mapping[int, Mapping[str, float]]) -> dict[str, Any]:
    """Report section 10.4 gates without making the keep/stop decision."""
    base = stage_means.get(672)
    if base is None:
        return {"evaluable": False, "reason": "672 baseline unavailable"}
    rows: dict[str, Any] = {}
    for resolution in (896, 1120):
        row = stage_means.get(resolution)
        if row is None:
            continue
        rows[str(resolution)] = {
            "ap_delta_vs_672": row["mean_pixel_ap"] - base["mean_pixel_ap"],
            "component_recall_delta_vs_672": (
                row["mean_component_recall"] - base["mean_component_recall"]
            ),
            "pauroc_loss_vs_672": base["mean_pauroc_0.05"] - row["mean_pauroc_0.05"],
        }
    consecutive_ap = all(
        r in stage_means and stage_means[r]["mean_pixel_ap"] > base["mean_pixel_ap"]
        for r in (896, 1120)
    )
    consecutive_component = all(
        r in stage_means
        and stage_means[r]["mean_component_recall"] > base["mean_component_recall"]
        for r in (896, 1120)
    )
    best_ap_delta = max(
        (row["mean_pixel_ap"] - base["mean_pixel_ap"] for r, row in stage_means.items() if r != 672),
        default=float("-inf"),
    )
    best_component_delta = max(
        (
            row["mean_component_recall"] - base["mean_component_recall"]
            for r, row in stage_means.items()
            if r != 672
        ),
        default=float("-inf"),
    )
    pauroc_ok = all(
        base["mean_pauroc_0.05"] - row["mean_pauroc_0.05"] <= 0.005
        for r, row in stage_means.items()
        if r != 672
    )
    return {
        "evaluable": len(stage_means) > 1,
        "per_stage": rows,
        "ap_or_component_increase_at_two_consecutive_resolutions": (
            consecutive_ap or consecutive_component
        ),
        "best_vs_672_ap_plus_0.01": best_ap_delta >= 0.01,
        "best_vs_672_component_recall_plus_0.10": best_component_delta >= 0.10,
        "mean_pauroc_loss_le_0.005": pauroc_ok,
        "decision_owner": "orchestrator",
    }
