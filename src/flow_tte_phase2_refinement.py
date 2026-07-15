"""Structure-guided refinement diagnostics for retained FlowTTE anomaly maps.

All score transforms are class-agnostic and use only the retained score map and
its corresponding RGB image.  Except for identity, transforms run at half
native resolution and are bilinearly resized back before native-resolution
evaluation.  Guided variants min-max normalize each image's score map, filter
the normalized map, and then apply ``filtered * (max - min) + min``.
"""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Sequence

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image
from sklearn.metrics import roc_auc_score

from src.flow_tte_gap_decomposition import (
    boundary_pair_summary,
    component_recall,
    load_records,
    locate_map_dir,
    oracle_f1,
    safe_ratio,
)

if TYPE_CHECKING:
    from pathlib import Path

FloatArray = npt.NDArray[np.float32]
Record = Dict[str, Any]

OBJECTS = (
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "vial",
    "wallplugs",
    "sheet_metal",
    "walnuts",
)
GUIDED_VARIANTS = (
    "guided_r8_eps1e-2",
    "guided_r16_eps1e-2",
    "guided_r16_eps1e-1",
    "guided_r32_eps1e-2",
)
FULL_VARIANTS = (
    "identity",
    "gaussian_blur_sigma4",
    "gaussian_blur_sigma8",
    *GUIDED_VARIANTS,
    "joint_bilateral_r16_sigmac0.1",
)
CORE_VARIANTS = (
    "identity",
    "gaussian_blur_sigma8",
    "guided_r16_eps1e-2",
    "guided_r16_eps1e-1",
)
ANCHOR_MEAN_F1 = 0.5306352134494448
ANCHOR_MEAN_PAUC = 0.8374260727573787
ANCHOR_F1_ROUNDED = {
    "can": 0.0007,
    "fabric": 0.6979,
    "fruit_jelly": 0.4768,
    "rice": 0.7125,
    "vial": 0.4396,
    "wallplugs": 0.6657,
    "sheet_metal": 0.5161,
    "walnuts": 0.7357,
}
CLAIM_SCOPE = "AD2-public-shadow-diagnostic"
WORK_SCALE = 0.5
BOUNDARY_TOLERANCES = (0, 4, 8, 16)


def box_mean(array: npt.ArrayLike, radius: int) -> FloatArray:
    """Return a normalized square-window mean with reflected boundaries."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    source = np.asarray(array, dtype=np.float32)
    if radius == 0:
        return source.copy()
    size = 2 * radius + 1
    return np.asarray(
        cv2.boxFilter(
            source,
            ddepth=-1,
            ksize=(size, size),
            normalize=True,
            borderType=cv2.BORDER_REFLECT,
        ),
        dtype=np.float32,
    )


def fast_guided_filter(
    guidance: npt.ArrayLike,
    source: npt.ArrayLike,
    radius: int,
    eps: float,
) -> FloatArray:
    """Apply the O(N) grayscale guided filter of He et al. using box means."""
    guide = np.asarray(guidance, dtype=np.float32)
    src = np.asarray(source, dtype=np.float32)
    if guide.shape != src.shape:
        message = f"guidance/source shape mismatch: {guide.shape} versus {src.shape}"
        raise ValueError(message)
    if eps <= 0:
        raise ValueError("eps must be positive")
    mean_i, mean_p = box_mean(guide, radius), box_mean(src, radius)
    corr_i = box_mean(guide * guide, radius)
    corr_ip = box_mean(guide * src, radius)
    variance_i = corr_i - mean_i * mean_i
    covariance_ip = corr_ip - mean_i * mean_p
    a = covariance_ip / (variance_i + np.float32(eps))
    b = mean_p - a * mean_i
    output = box_mean(a, radius) * guide + box_mean(b, radius)
    return np.asarray(output, dtype=np.float32)


def _half_size(shape: Sequence[int]) -> tuple[int, int]:
    height, width = int(shape[0]), int(shape[1])
    return max(1, round(width * WORK_SCALE)), max(1, round(height * WORK_SCALE))


def load_half_guidance(rgb_path: str | Path, native_shape: Sequence[int]) -> FloatArray:
    """Load RGB as grayscale [0,1] guidance and resize it to half resolution."""
    with Image.open(rgb_path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float32) / np.float32(255.0)
    if gray.shape != tuple(native_shape):
        message = (
            f"RGB/map shape mismatch for {rgb_path}: "
            f"{gray.shape} versus {tuple(native_shape)}"
        )
        raise ValueError(message)
    return np.asarray(
        cv2.resize(gray, _half_size(native_shape), interpolation=cv2.INTER_AREA),
        dtype=np.float32,
    )


def _resize_to_native(score: FloatArray, native_shape: Sequence[int]) -> FloatArray:
    return np.asarray(
        cv2.resize(
            score,
            (int(native_shape[1]), int(native_shape[0])),
            interpolation=cv2.INTER_LINEAR,
        ),
        dtype=np.float32,
    )


def _normalized_half_score(score: npt.ArrayLike) -> tuple[FloatArray, float, float]:
    native = np.asarray(score, dtype=np.float32)
    minimum, maximum = float(np.min(native)), float(np.max(native))
    span = maximum - minimum
    if span == 0.0:
        normalized = np.zeros(native.shape, dtype=np.float32)
    else:
        normalized = np.asarray((native - minimum) / span, dtype=np.float32)
    half = cv2.resize(normalized, _half_size(native.shape), interpolation=cv2.INTER_AREA)
    return np.asarray(half, dtype=np.float32), minimum, span


def has_joint_bilateral() -> bool:
    """Return whether OpenCV contrib exposes the required joint filter."""
    return bool(
        hasattr(cv2, "ximgproc")
        and hasattr(cv2.ximgproc, "jointBilateralFilter"),  # type: ignore[attr-defined]
    )


def transform_score(
    score: npt.ArrayLike,
    guidance_half: FloatArray | None,
    variant: str,
) -> FloatArray:
    """Apply one preregistered transform without consulting GT or split labels."""
    native = np.asarray(score, dtype=np.float32)
    if variant == "identity":
        return native
    if variant.startswith("gaussian_blur_sigma"):
        sigma = float(variant.rsplit("sigma", 1)[1])
        half = cv2.resize(native, _half_size(native.shape), interpolation=cv2.INTER_AREA)
        filtered = cv2.GaussianBlur(
            half,
            ksize=(0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT,
        )
        return _resize_to_native(np.asarray(filtered, dtype=np.float32), native.shape)
    if guidance_half is None:
        message = f"variant {variant} requires RGB guidance"
        raise ValueError(message)

    normalized, minimum, span = _normalized_half_score(native)
    if variant.startswith("guided_"):
        parameters = {
            "guided_r8_eps1e-2": (8, 1e-2),
            "guided_r16_eps1e-2": (16, 1e-2),
            "guided_r16_eps1e-1": (16, 1e-1),
            "guided_r32_eps1e-2": (32, 1e-2),
        }
        if variant not in parameters:
            message = f"unknown guided variant: {variant}"
            raise ValueError(message)
        radius, eps = parameters[variant]
        filtered = fast_guided_filter(guidance_half, normalized, radius, eps)
    elif variant == "joint_bilateral_r16_sigmac0.1":
        if not has_joint_bilateral():
            raise RuntimeError("cv2.ximgproc.jointBilateralFilter is unavailable")
        filtered = cv2.ximgproc.jointBilateralFilter(  # type: ignore[attr-defined]
            guidance_half,
            normalized,
            d=33,
            sigmaColor=0.1,
            sigmaSpace=16.0,
            borderType=cv2.BORDER_REFLECT,
        )
    else:
        message = f"unknown refinement variant: {variant}"
        raise ValueError(message)
    restored = np.asarray(filtered, dtype=np.float32) * np.float32(span) + np.float32(minimum)
    return _resize_to_native(restored, native.shape)


def transform_records(
    records: Sequence[Record],
    variant: str,
    guidances: Sequence[FloatArray] | None = None,
) -> list[FloatArray]:
    """Transform every map using only its score and corresponding RGB image."""
    if variant == "identity":
        return [np.asarray(record["score"], dtype=np.float32) for record in records]
    if variant.startswith("gaussian_blur_sigma"):
        return [transform_score(record["score"], None, variant) for record in records]
    if guidances is not None and len(guidances) != len(records):
        raise ValueError("guidance/record count mismatch")
    output = []
    for index, record in enumerate(records):
        score = np.asarray(record["score"], dtype=np.float32)
        guidance = (
            guidances[index]
            if guidances is not None
            else load_half_guidance(record["rgb_path"], score.shape)
        )
        output.append(transform_score(score, guidance, variant))
    return output


def _tolerant_metrics(summaries: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    output = {}
    for tolerance in BOUNDARY_TOLERANCES:
        counts = np.sum([row["tolerant"][tolerance] for row in summaries], axis=0)
        precision = safe_ratio(counts[0], counts[1])
        recall = safe_ratio(counts[2], counts[3])
        f1 = safe_ratio(2.0 * precision * recall, precision + recall)
        output[str(tolerance)] = {"precision": precision, "recall": recall, "f1": f1}
    return output


def evaluate_variant(records: Sequence[Record], scores: Sequence[FloatArray]) -> dict[str, Any]:
    """Evaluate one object/variant with float16 pooled evaluator parity."""
    labels = np.concatenate(
        [np.asarray(record["gt"], dtype=np.bool_).ravel() for record in records],
    )
    pooled_scores = np.concatenate(
        [np.asarray(score, dtype=np.float32).ravel() for score in scores],
    )
    pooled = oracle_f1(labels, pooled_scores, cast_float16=True)
    threshold = np.float16(pooled["threshold"])
    partial_auc = float(
        roc_auc_score(labels.astype(np.uint8), pooled_scores.astype(np.float16), max_fpr=0.05),
    )

    summaries = []
    gt_hit = gt_total = 0
    good_positive = good_pixels = 0
    good_image_fprs = []
    for record, score in zip(records, scores):
        prediction = np.asarray(score, dtype=np.float32).astype(np.float16) >= threshold
        gt = np.asarray(record["gt"], dtype=np.bool_)
        summaries.append(boundary_pair_summary(prediction, gt))
        components = component_recall(prediction, gt)
        gt_hit += int(components["hit"])
        gt_total += int(components["total"])
        if record["split"] == "good":
            positive = int(np.count_nonzero(prediction))
            pixels = int(prediction.size)
            good_positive += positive
            good_pixels += pixels
            good_image_fprs.append(safe_ratio(positive, pixels))

    interior_numerator = sum(row["interior"][8][0] for row in summaries)
    interior_denominator = sum(row["interior"][8][1] for row in summaries)
    return {
        "pooled_oracle_f1": pooled["f1"],
        "pooled_oracle_threshold": pooled["threshold"],
        "seg_AUROC_0.05": partial_auc,
        "boundary_tolerant_f1_native_px": _tolerant_metrics(summaries),
        "interior_recall_eroded_8px": safe_ratio(interior_numerator, interior_denominator),
        "gt_component_recall": safe_ratio(gt_hit, gt_total),
        "gt_components_hit": gt_hit,
        "gt_components_total": gt_total,
        "normal_image_fpr": {
            "mean_per_image": float(np.mean(good_image_fprs)),
            "pooled_pixel": safe_ratio(good_positive, good_pixels),
        },
    }


def aggregate_variant(objects: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    """Compute unweighted object means for Phase 2 metrics."""
    rows = list(objects.values())
    return {
        "pooled_oracle_f1": float(np.mean([row["pooled_oracle_f1"] for row in rows])),
        "seg_AUROC_0.05": float(np.mean([row["seg_AUROC_0.05"] for row in rows])),
        "boundary_tolerant_f1_native_px": {
            str(tolerance): float(
                np.mean(
                    [row["boundary_tolerant_f1_native_px"][str(tolerance)]["f1"] for row in rows],
                ),
            )
            for tolerance in BOUNDARY_TOLERANCES
        },
        "interior_recall_eroded_8px": float(
            np.mean([row["interior_recall_eroded_8px"] for row in rows]),
        ),
        "gt_component_recall": float(np.mean([row["gt_component_recall"] for row in rows])),
        "normal_image_fpr_mean_per_image": float(
            np.mean([row["normal_image_fpr"]["mean_per_image"] for row in rows]),
        ),
        "normal_image_fpr_pooled_pixel": float(
            np.mean([row["normal_image_fpr"]["pooled_pixel"] for row in rows]),
        ),
    }


def _anchor_reference(result_root: Path) -> dict[str, float]:
    path = result_root / "summary_gapdecomp_anchor.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["object"]): float(row["seg_F1"]) for row in payload.get("objects", [])}


def identity_parity(identity: Mapping[str, Any], result_root: Path) -> dict[str, Any]:
    """Validate identity against exact retained-anchor metrics when available."""
    observed = {name: float(row["pooled_oracle_f1"]) for name, row in identity["objects"].items()}
    exact_reference = _anchor_reference(result_root)
    rounded_pass = {
        name: round(observed[name], 4) == ANCHOR_F1_ROUNDED[name] for name in OBJECTS
    }
    exact_pass = {
        name: observed[name] == exact_reference[name] for name in OBJECTS if name in exact_reference
    }
    mean_value = float(identity["mean"]["pooled_oracle_f1"])
    return {
        "observed": observed,
        "exact_reference": exact_reference,
        "exact_object_pass": exact_pass,
        "rounded_reference": ANCHOR_F1_ROUNDED,
        "rounded_object_pass": rounded_pass,
        "observed_mean": mean_value,
        "reference_mean": ANCHOR_MEAN_F1,
        "mean_absolute_error": abs(mean_value - ANCHOR_MEAN_F1),
        "pass": (
            all(rounded_pass.values())
            and (
                not exact_reference
                or (len(exact_pass) == len(OBJECTS) and all(exact_pass.values()))
            )
            and abs(mean_value - ANCHOR_MEAN_F1) <= 1e-6
        ),
    }


def floor_check(variant: Mapping[str, Any], identity: Mapping[str, Any]) -> dict[str, Any]:
    """Check that no per-object F1 or pAUROC loss exceeds 0.02."""
    objects = {}
    for name in OBJECTS:
        current, anchor = variant["objects"][name], identity["objects"][name]
        f1_delta = float(current["pooled_oracle_f1"] - anchor["pooled_oracle_f1"])
        pauc_delta = float(current["seg_AUROC_0.05"] - anchor["seg_AUROC_0.05"])
        objects[name] = {
            "f1_delta": f1_delta,
            "seg_AUROC_0.05_delta": pauc_delta,
            "pass": f1_delta >= -0.02 and pauc_delta >= -0.02,
        }
    return {"objects": objects, "pass": all(row["pass"] for row in objects.values())}


def keep_gate(
    variant: Mapping[str, Any],
    identity: Mapping[str, Any],
    gaussian: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the preregistered Phase 2 criteria without making a decision."""
    mean, identity_mean, gaussian_mean = variant["mean"], identity["mean"], gaussian["mean"]
    boundary_gap = float(
        mean["boundary_tolerant_f1_native_px"]["8"]
        - mean["boundary_tolerant_f1_native_px"]["0"],
    )
    identity_gap = float(
        identity_mean["boundary_tolerant_f1_native_px"]["8"]
        - identity_mean["boundary_tolerant_f1_native_px"]["0"],
    )
    interior_loss = float(
        identity_mean["interior_recall_eroded_8px"] - mean["interior_recall_eroded_8px"],
    )
    pauc_loss = float(identity_mean["seg_AUROC_0.05"] - mean["seg_AUROC_0.05"])
    checks = {
        "mean_f1_at_least_anchor_plus_0.01": (
            mean["pooled_oracle_f1"] >= identity_mean["pooled_oracle_f1"] + 0.01
        ),
        "tolerant_f1_gap_t8_minus_t0_shrinks_by_at_least_0.03": (
            identity_gap - boundary_gap >= 0.03
        ),
        "interior_recall_loss_at_most_0.01": interior_loss <= 0.01,
        "mean_pAUROC_loss_at_most_0.002": pauc_loss <= 0.002,
        "guided_mean_f1_beats_gaussian_control": (
            mean["pooled_oracle_f1"] > gaussian_mean["pooled_oracle_f1"]
        ),
    }
    return {
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "values": {
            "mean_f1": mean["pooled_oracle_f1"],
            "required_mean_f1": identity_mean["pooled_oracle_f1"] + 0.01,
            "identity_t8_minus_t0_gap": identity_gap,
            "variant_t8_minus_t0_gap": boundary_gap,
            "gap_shrinkage": identity_gap - boundary_gap,
            "interior_recall_loss": interior_loss,
            "mean_pAUROC_loss": pauc_loss,
            "gaussian_mean_f1": gaussian_mean["pooled_oracle_f1"],
        },
        "decision_owner": "requester",
    }


def requested_variants(variant_set: str, include_joint: bool | None = None) -> tuple[str, ...]:
    """Resolve full/core sweep and the OpenCV-contrib capability gate."""
    if variant_set == "core":
        return CORE_VARIANTS
    if variant_set != "full":
        message = f"unknown variant set: {variant_set}"
        raise ValueError(message)
    joint = has_joint_bilateral() if include_joint is None else include_joint
    return FULL_VARIANTS if joint else FULL_VARIANTS[:-1]


def analyze_object_variants(
    result_root: Path,
    data_root: Path,
    object_name: str,
    variants: Sequence[str],
) -> dict[str, Any]:
    """Load one object once, then evaluate variants sequentially to bound memory."""
    cv2.setNumThreads(1)
    print(
        f"loading object={object_name} map_dir={locate_map_dir(result_root, object_name)}",
        flush=True,
    )
    records = load_records(result_root, data_root, object_name)
    print(f"loaded object={object_name} images={len(records)}", flush=True)
    needs_guidance = any(
        name in GUIDED_VARIANTS or name.startswith("joint_bilateral") for name in variants
    )
    guidances = (
        [load_half_guidance(record["rgb_path"], record["score"].shape) for record in records]
        if needs_guidance
        else None
    )
    metrics = {}
    for name in variants:
        scores = transform_records(records, name, guidances)
        metrics[name] = evaluate_variant(records, scores)
        print(
            f"evaluated object={object_name} variant={name} "
            f"f1={metrics[name]['pooled_oracle_f1']:.6f}",
            flush=True,
        )
    return {"object": object_name, "metrics": metrics}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_leaderboard(path: Path, variants: Mapping[str, dict[str, Any]]) -> None:
    fields = ["variant", "mean_f1", "mean_seg_AUROC_0.05"]
    for object_name in OBJECTS:
        fields.extend((f"{object_name}_f1", f"{object_name}_seg_AUROC_0.05"))
    rows = []
    for name, payload in variants.items():
        row: dict[str, Any] = {
            "variant": name,
            "mean_f1": payload["mean"]["pooled_oracle_f1"],
            "mean_seg_AUROC_0.05": payload["mean"]["seg_AUROC_0.05"],
        }
        for object_name in OBJECTS:
            metrics = payload["objects"][object_name]
            row[f"{object_name}_f1"] = metrics["pooled_oracle_f1"]
            row[f"{object_name}_seg_AUROC_0.05"] = metrics["seg_AUROC_0.05"]
        rows.append(row)
    rows.sort(key=lambda row: float(row["mean_f1"]), reverse=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def analyze_run(  # noqa: C901, PLR0912, PLR0913
    result_root: Path,
    data_root: Path,
    objects: Sequence[str],
    output_dir: Path,
    workers: int = 1,
    variant_set: str = "full",
    include_joint: bool | None = None,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run Phase 2 and write the compact leaderboard/JSON artifact set."""
    if tuple(objects) != OBJECTS:
        message = f"Phase 2 requires the fixed object order/set: {OBJECTS}"
        raise ValueError(message)
    if workers < 1:
        raise ValueError("workers must be positive")
    variants_to_run = requested_variants(variant_set, include_joint)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants: dict[str, dict[str, Any]] = {
        name: {"variant": name, "objects": {}} for name in variants_to_run
    }
    if workers == 1:
        results = [
            analyze_object_variants(result_root, data_root, name, variants_to_run)
            for name in objects
        ]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=min(workers, len(objects))) as executor:
            futures = {
                executor.submit(
                    analyze_object_variants,
                    result_root,
                    data_root,
                    name,
                    variants_to_run,
                ): name
                for name in objects
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                progress(f"completed object={result['object']}")
    for result in results:
        for name in variants_to_run:
            variants[name]["objects"][result["object"]] = result["metrics"][name]
    for payload in variants.values():
        payload["mean"] = aggregate_variant(payload["objects"])

    parity = identity_parity(variants["identity"], result_root)
    if not parity["pass"]:
        metadata = {
            "claim_scope": CLAIM_SCOPE,
            "status": "identity_parity_failed",
            "identity_parity": parity,
            "interpretation_blocked": True,
        }
        write_json(output_dir / "metadata.json", metadata)
        write_json(output_dir / "identity.json", variants["identity"])
        write_leaderboard(output_dir / "leaderboard.tsv", variants)
        raise RuntimeError("identity parity failed; refinement interpretation suppressed")

    gaussian = variants["gaussian_blur_sigma8"]
    floor_checks = {
        name: floor_check(payload, variants["identity"]) for name, payload in variants.items()
    }
    gates = {
        name: keep_gate(payload, variants["identity"], gaussian)
        for name, payload in variants.items()
        if name in GUIDED_VARIANTS
    }
    deviations = []
    if variant_set == "core":
        deviations.append(
            "Projected wall time exceeded approximately 3 hours: reduced to the preregistered "
            "four-variant core set.",
        )
    if variant_set == "full" and "joint_bilateral_r16_sigmac0.1" not in variants_to_run:
        deviations.append(
            "Dropped joint_bilateral_r16_sigmac0.1 because "
            "cv2.ximgproc.jointBilateralFilter is unavailable.",
        )
    metadata = {
        "claim_scope": CLAIM_SCOPE,
        "status": "complete",
        "class_agnostic": True,
        "gt_used_in_score_transform": False,
        "per_object_tuning": False,
        "result_root_read_only": str(result_root),
        "data_root": str(data_root),
        "objects": list(objects),
        "variants": list(variants_to_run),
        "work_scale": WORK_SCALE,
        "radius_units": "pixels at half-resolution working scale",
        "metric_resolution": "full native resolution",
        "guided_score_scale_procedure": (
            "Per image: native float32 min/max; normalize to [0,1] (constant maps become zero); "
            "INTER_AREA resize to half resolution; guided filtering; affine de-normalize with "
            "the same min/max; INTER_LINEAR resize to native resolution."
        ),
        "guided_filter_backend": "plain O(N) box-filter NumPy/OpenCV implementation",
        "opencv_version": cv2.__version__,
        "opencv_has_ximgproc": hasattr(cv2, "ximgproc"),
        "identity_parity": parity,
        "anchor_mean_pAUROC_0.05": ANCHOR_MEAN_PAUC,
        "floor_checks": floor_checks,
        "keep_gates": gates,
        "deviations": deviations,
        "decision_owner": "requester",
    }
    for name, payload in variants.items():
        payload["metadata"] = {
            "claim_scope": CLAIM_SCOPE,
            "work_scale": WORK_SCALE,
            "radius_units": "half-resolution pixels",
        }
        payload["floor_check"] = floor_checks[name]
        if name in gates:
            payload["keep_gate"] = gates[name]
        write_json(output_dir / f"{name}.json", payload)
    write_leaderboard(output_dir / "leaderboard.tsv", variants)
    write_json(output_dir / "metadata.json", metadata)
    progress(f"identity_parity_pass={parity['pass']}")
    for name, gate in gates.items():
        for criterion, passed in gate["checks"].items():
            progress(f"KEEP_GATE variant={name} criterion={criterion} pass={passed}")
        progress(f"KEEP_GATE variant={name} all_checks_pass={gate['all_checks_pass']}")
    progress("phase2_complete")
    return {"metadata": metadata, "variants": variants}
