#!/usr/bin/env python3
"""Evaluate public v13 F1 at one leave-one-support-out normal threshold.

This runner is deliberately separate from the shared inference/evaluation
pipeline. It uses only the configured support images to calibrate the
threshold, never stores dense maps, and opens test masks only after the
threshold is frozen. Run one object per process to bound memory and to make
the saved threshold provenance easy to audit.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

import fmad.backbones.dinov2  # noqa: F401
import fmad.backbones.dinov3  # noqa: F401
import fmad.datasets  # noqa: F401
from fmad.datasets.base import BaseDataset
from fmad.methods import register_methods
from fmad.pipeline import _ground_truth_mask, _input_fingerprint
from fmad.protocol import MethodContext, seed_generators, select_support_paths, validate_protocol
from fmad.registry import BACKBONE_REGISTRY, DATASET_REGISTRY, METHOD_REGISTRY
from fmad.results import _code_sha256


THRESHOLD_PROTOCOL = "public_v13_support_loo_pooled_pixel_q95_superadd_1.421_v1"
THRESHOLD_PERCENTILE = 95.0
THRESHOLD_FACTOR = 1.421
QUERY_SPLIT = "test_public"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    for section in ("dataset_config", "backbone_config", "method_config", "protocol", "evaluation"):
        if section in config and not isinstance(config[section], dict):
            raise ValueError(f"{section} must be a mapping")
    if config.get("dataset", "mvtec_ad2") != "mvtec_ad2":
        raise ValueError("public v13 threshold evaluation requires dataset=mvtec_ad2")
    if not isinstance(config.get("method"), str) or not config["method"]:
        raise ValueError("configuration requires a method name")
    if not isinstance(config.get("backbone"), str) or not config["backbone"]:
        raise ValueError("configuration requires a backbone name")
    config.setdefault("seed", 0)
    config.setdefault("protocol", {})
    config.setdefault("dataset_config", {})
    config.setdefault("backbone_config", {})
    config.setdefault("method_config", {})
    validate_protocol(config["protocol"])
    if config["protocol"].get("shots", 16) != 16:
        raise ValueError("this threshold extension is defined for exactly 16 support images")
    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool):
        raise ValueError("seed must be an integer")
    return config


def _validate_score_map(prediction: Any, image_path: str) -> np.ndarray:
    scores = np.asarray(prediction.score_map, dtype=np.float32)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError(f"invalid finite 2D score map for {image_path}: {scores.shape}")
    with Image.open(image_path) as image:
        native_shape = (image.height, image.width)
    if scores.shape != native_shape:
        raise ValueError(
            f"score map shape {scores.shape} does not match native image "
            f"{native_shape}: {image_path}"
        )
    return scores


def _new_fitted_method(
    config: dict[str, Any],
    dataset: BaseDataset,
    object_name: str,
    device: str,
    support_paths: list[str],
):
    """Construct an isolated model and fit only the supplied normal paths."""
    training_events: list[dict[str, Any]] = []
    seed = int(config["seed"])
    seed_generators(seed)
    info = dataset.get_object_info(object_name)
    context = MethodContext(
        object_name,
        seed,
        info.resolution,
        info.masking,
        info.rotation,
    )
    backbone_cls = BACKBONE_REGISTRY.get(config["backbone"])
    backbone = backbone_cls(
        model_name=config["backbone"],
        device=device,
        **dict(config.get("backbone_config", {})),
    )
    if hasattr(backbone, "set_resolution"):
        backbone.set_resolution(context.resolution)
    warmup_iters = int(config["protocol"].get("warmup_iters", 25))
    if hasattr(backbone, "warmup") and support_paths and warmup_iters:
        backbone.warmup(support_paths[0], iters=warmup_iters)
    method_cls = METHOD_REGISTRY.get(config["method"])
    method = method_cls(backbone, dict(config.get("method_config", {})))
    method._training_log_sink = lambda step, metrics: training_events.append(
        {"step": step, **metrics}
    )
    fit_started = time.perf_counter()
    method.fit(list(support_paths), context)
    return method, backbone, context, {
        "fit_seconds": time.perf_counter() - fit_started,
        "training_events": training_events,
    }


def _calibrate_threshold(
    config: dict[str, Any],
    dataset: BaseDataset,
    object_name: str,
    device: str,
    support_paths: list[str],
) -> tuple[float, dict[str, Any]]:
    leave_one_out_scores: list[np.ndarray] = []
    fold_records: list[dict[str, Any]] = []
    pixels_per_fold: list[int] = []

    for fold_index, held_out_path in enumerate(support_paths):
        fold_support = [path for path in support_paths if path != held_out_path]
        if len(fold_support) != 15 or held_out_path in fold_support:
            raise ValueError("each calibration fold must fit exactly the other 15 support images")
        method = backbone = None
        try:
            method, backbone, _, fit_log = _new_fitted_method(
                config, dataset, object_name, device, fold_support
            )
            predictions = method.predict_batch([held_out_path])
            if len(predictions) != 1:
                raise ValueError("predict_batch must return one output for one held-out support image")
            score_map = _validate_score_map(predictions[0], held_out_path)
            flat = score_map.reshape(-1).copy()
            leave_one_out_scores.append(flat)
            pixels_per_fold.append(int(flat.size))
            fold_records.append(
                {
                    "fold": fold_index,
                    "held_out_support": Path(held_out_path).name,
                    "fitted_support": [Path(path).name for path in fold_support],
                    "fit_support_count": len(fold_support),
                    "fit_seconds": fit_log["fit_seconds"],
                    "training_events": fit_log["training_events"],
                    "score_map_shape": list(score_map.shape),
                    "score_pixel_count": int(flat.size),
                }
            )
        finally:
            method = None
            backbone = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if len(leave_one_out_scores) != 16:
        raise ValueError(f"expected 16 leave-one-out calibration maps, got {len(leave_one_out_scores)}")
    pooled = np.concatenate(leave_one_out_scores)
    if not pooled.size or not np.isfinite(pooled).all():
        raise ValueError("pooled support-only calibration scores are empty or non-finite")
    q95 = float(np.percentile(pooled, THRESHOLD_PERCENTILE, method="linear"))
    threshold = float(q95 * THRESHOLD_FACTOR)
    if not np.isfinite(threshold):
        raise ValueError("calibrated threshold is non-finite")
    provenance = {
        "calibration_protocol": THRESHOLD_PROTOCOL,
        "calibration_source": "only the configured 16 normal support images, predicted leave-one-out",
        "support_count": 16,
        "leave_one_out_fold_count": len(fold_records),
        "fit_support_count_per_fold": 15,
        "calibration_map_shapes": [item["score_map_shape"] for item in fold_records],
        "calibration_pixels_per_fold": pixels_per_fold,
        "pooled_calibration_pixel_count": int(pooled.size),
        "quantile": THRESHOLD_PERCENTILE,
        "quantile_method": "numpy_percentile_linear",
        "normal_pixel_q95": q95,
        "factor": THRESHOLD_FACTOR,
        "factor_precedent": "scripts/prepare_rsg4_private_submission.py: THRESHOLD_FACTOR=1.421",
        "threshold": threshold,
        "comparison": "score > threshold",
        "folds": fold_records,
    }
    del pooled, leave_one_out_scores
    return threshold, provenance


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0


def _evaluate_queries(
    method,
    dataset: BaseDataset,
    object_name: str,
    threshold: float,
) -> dict[str, Any]:
    grouped = dataset.get_test_images(object_name, split=QUERY_SPLIT)
    queries = sorted(
        [(kind, path) for kind, paths in grouped.items() for path in paths],
        key=lambda item: (Path(item[1]).name, item[0]),
    )
    if not queries:
        raise ValueError(f"{object_name}: {QUERY_SPLIT} contains no query images")

    seg_tp = seg_fp = seg_fn = 0
    class_tp = class_fp = class_fn = 0
    positive_images = 0
    negative_images = 0
    query_identity: list[list[str]] = []
    ground_truth_digest = hashlib.sha256()

    for anomaly_type, image_path in queries:
        predictions = method.predict_batch([image_path])
        if len(predictions) != 1:
            raise ValueError("predict_batch must return one output for one query image")
        scores = _validate_score_map(predictions[0], image_path)
        predicted_mask = scores > threshold

        # The label path is deliberately downstream of prediction and the
        # frozen support threshold. No label reaches the method.
        truth = _ground_truth_mask(
            dataset,
            object_name,
            anomaly_type,
            image_path,
            scores.shape,
        )
        sample_key = f"{anomaly_type}/{Path(image_path).name}"
        ground_truth_digest.update(json.dumps([sample_key, list(truth.shape)]).encode("utf-8"))
        ground_truth_digest.update(truth.tobytes())
        positive = bool(truth.any())
        predicted_positive = bool(predicted_mask.any())
        if positive:
            positive_images += 1
        else:
            negative_images += 1
        if positive and predicted_positive:
            class_tp += 1
        elif predicted_positive:
            class_fp += 1
        elif positive:
            class_fn += 1

        seg_tp += int(np.count_nonzero(predicted_mask & truth))
        seg_fp += int(np.count_nonzero(predicted_mask & ~truth))
        seg_fn += int(np.count_nonzero(~predicted_mask & truth))
        query_identity.append([anomaly_type, Path(image_path).name])

    return {
        "query_split": QUERY_SPLIT,
        "query_count": len(queries),
        "query_identity_sha256": _canonical_hash(query_identity),
        "ground_truth_sha256": ground_truth_digest.hexdigest(),
        "positive_image_count": positive_images,
        "negative_image_count": negative_images,
        "segmentation": {
            "seg_f1": _f1(seg_tp, seg_fp, seg_fn),
            "true_positive_pixels": seg_tp,
            "false_positive_pixels": seg_fp,
            "false_negative_pixels": seg_fn,
            "thresholded_rule": "score > fixed support-only threshold",
        },
        "classification": {
            "class_f1": _f1(class_tp, class_fp, class_fn),
            "true_positive_images": class_tp,
            "false_positive_images": class_fp,
            "false_negative_images": class_fn,
            "prediction_rule": "image is anomalous iff any thresholded pixel is positive",
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite threshold result: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--device", default=None, help="defaults to the device in the config")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = _load_config(config_path)
    seed = int(config["seed"])
    device = args.device or config.get("device", "cuda:0")
    if not isinstance(device, str) or not (device == "cpu" or device == "cuda" or device.startswith("cuda:")):
        raise ValueError(f"unsupported device string: {device!r}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device!r} requested but CUDA is unavailable")

    manifest_value = config["protocol"].get("fixed_reference_manifest")
    manifest_path = Path(manifest_value).resolve() if manifest_value else None
    if manifest_path is None or not manifest_path.is_file():
        raise ValueError("public v13 threshold evaluation requires an existing fixed support manifest")

    dataset_cls = DATASET_REGISTRY.get(config.get("dataset", "mvtec_ad2"))
    dataset = dataset_cls(str(args.data_root.resolve()), dict(config.get("dataset_config", {})))
    object_info = dataset.get_object_info(args.object)
    train_paths = sorted(dataset.get_train_images(args.object))
    support_paths, support_manifest_hash = select_support_paths(
        train_paths,
        16,
        seed,
        manifest_path=manifest_path,
        object_name=args.object,
    )
    if len(support_paths) != 16 or len(set(support_paths)) != 16:
        raise ValueError("support manifest did not resolve to 16 unique normal images")

    register_methods(config["method"])
    start = time.perf_counter()
    threshold, calibration = _calibrate_threshold(
        config,
        dataset,
        args.object,
        device,
        support_paths,
    )

    # Final query model is freshly instantiated and fitted on the complete
    # 16-image support set, as declared in the benchmark configuration.
    method = backbone = None
    try:
        method, backbone, context, final_fit_log = _new_fitted_method(
            config,
            dataset,
            args.object,
            device,
            support_paths,
        )
        metrics = _evaluate_queries(method, dataset, args.object, threshold)
    finally:
        method = None
        backbone = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    support_identity_hash, support_content_hash = _input_fingerprint(
        support_paths, dataset.data_root
    )
    result = {
        "protocol": THRESHOLD_PROTOCOL,
        "method": config["method"],
        "dataset": config.get("dataset", "mvtec_ad2"),
        "backbone": config["backbone"],
        "seed": seed,
        "object": args.object,
        "shots": 16,
        "device": device,
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "script_sha256": _sha256_file(Path(__file__).resolve()),
        "fmad_code_sha256": _code_sha256(),
        "support_manifest": str(manifest_path),
        "support_manifest_sha256": support_manifest_hash,
        "support_identity_sha256": support_identity_hash,
        "support_content_sha256": support_content_hash,
        "support_images": [Path(path).name for path in support_paths],
        "context": context.__dict__,
        "final_fit": final_fit_log,
        "threshold_calibration": calibration,
        "metrics": metrics,
        "execution": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "elapsed_seconds": time.perf_counter() - start,
            "dense_maps_persisted": False,
            "test_labels_used_for_threshold": False,
            "method_or_threshold_query_adaptation": False,
        },
    }
    output_path = args.output_dir / f"{args.object}.json"
    _write_json(output_path, result)
    print(
        f"{config['method']} seed={seed} object={args.object}: "
        f"threshold={threshold:.10g} SegF1={metrics['segmentation']['seg_f1']:.6f} "
        f"ClassF1={metrics['classification']['class_f1']:.6f}; wrote {output_path}"
    )


if __name__ == "__main__":
    main()
