"""Main pipeline — orchestrates method, dataset, backbone, evaluation, results."""

import hashlib
import json
import os
from pathlib import Path
import time
from contextlib import nullcontext
from typing import Dict, List

import numpy as np
from PIL import Image

from fmad.methods.base import BaseMethod, ObjectResult
from fmad.protocol import MethodContext, select_support_paths, seed_generators, validate_protocol
from fmad.results import ResultsManager, validate_output_options
from fmad.evaluation.streaming import StreamingSegmentationAccumulator

_STREAMING_METRIC_CONTRACT = "mvtec_ad2_float16_streaming_v1"


def _top_fraction_mean(values: np.ndarray, fraction: float = 0.01) -> float:
    flat = np.asarray(values).reshape(-1)
    if flat.size == 0:
        raise ValueError("cannot aggregate an empty score map")
    k = max(1, int(np.floor(flat.size * fraction)))
    return float(np.mean(np.partition(flat, -k)[-k:]))


def _input_fingerprint(paths, data_root):
    """Hash ordered identities and bytes independently of the machine's data root."""
    root = Path(data_root).resolve()
    identities = []
    content = hashlib.sha256()
    for value in paths:
        path = Path(value).resolve()
        try:
            identity = path.relative_to(root).as_posix()
        except ValueError:
            # When paths are symlinked to external canonical data, preserve the call path structure
            identity = Path(value).as_posix()
        identities.append(identity)
        content.update(json.dumps(identity).encode())
        content.update(hashlib.sha256(path.read_bytes()).digest())
    identity_hash = hashlib.sha256(json.dumps(identities).encode()).hexdigest()
    return identity_hash, content.hexdigest()


def _ground_truth_mask(
    dataset, object_name: str, anomaly_type: str, image_path: str, score_shape: tuple[int, ...]
) -> np.ndarray:
    with Image.open(image_path) as image:
        native_shape = (image.height, image.width)
    if score_shape != native_shape:
        raise ValueError(
            f"score map shape {score_shape} does not match native image {native_shape}"
        )
    if anomaly_type == "good":
        return np.zeros(score_shape, dtype=bool)
    gt_dir = dataset.get_ground_truth_dir(object_name)
    if gt_dir is None:
        raise FileNotFoundError(f"{object_name}: ground-truth directory is missing")
    gt_path = Path(gt_dir) / anomaly_type / f"{Path(image_path).stem}_mask.png"
    if not gt_path.is_file():
        gt_path = Path(gt_dir) / anomaly_type / f"{Path(image_path).stem}.png"
    if not gt_path.is_file():
        raise FileNotFoundError(f"ground truth not found: {gt_path}")
    with Image.open(gt_path) as image:
        mask = np.asarray(image) > 0
    if mask.ndim > 2:
        mask = np.any(mask, axis=-1)
    if mask.shape != score_shape:
        raise ValueError(f"GT shape {mask.shape} does not match score map {score_shape}")
    return np.asarray(mask, dtype=bool)


def _run_streaming_object(
    method,
    dataset,
    object_name: str,
    seed: int,
    run_dir: str,
    batch_size: int,
    *,
    config: dict | None = None,
    debug_output: bool = False,
    results_mgr: ResultsManager | None = None,
) -> ObjectResult:
    """Run one swappable model through the common streaming evaluator."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("inference_batch_size must be positive")
    config = config or {}
    cost_profile = config.get("evaluation", {}).get("cost_profile", False)
    if not isinstance(cost_profile, bool):
        raise ValueError("evaluation.cost_profile must be a boolean")
    if cost_profile and batch_size != 1:
        raise ValueError("cost profiling requires inference_batch_size=1")
    seed_generators(seed)
    obj_info = dataset.get_object_info(object_name)
    train_paths = sorted(dataset.get_train_images(object_name))
    protocol = config.get("protocol", {})
    shots = protocol.get("shots", 16)
    manifest = protocol.get("fixed_reference_manifest")
    warmup_iters = protocol.get("warmup_iters", 25)
    if isinstance(warmup_iters, bool) or not isinstance(warmup_iters, int) or warmup_iters < 0:
        raise ValueError("warmup_iters must be a non-negative integer")
    support_paths, manifest_digest = select_support_paths(
        train_paths,
        shots,
        seed,
        manifest_path=Path(manifest) if manifest else None,
        object_name=object_name,
    )
    context = MethodContext(
        object_name, seed, obj_info.resolution, obj_info.masking, obj_info.rotation
    )
    if hasattr(method.backbone, "set_resolution"):
        method.backbone.set_resolution(context.resolution)
    if hasattr(method.backbone, "warmup") and support_paths:
        method.backbone.warmup(support_paths[0], iters=warmup_iters)
    fit_timing = None
    if cost_profile:
        import torch
        from fmad.cost_profile import NvidiaSmiPeakSampler

        profile_device = getattr(method.backbone, "device", "cuda")
        if not torch.cuda.is_available() or not str(profile_device).startswith("cuda"):
            raise RuntimeError("cost profiling requires a CUDA device")
        fit_memory_sampler = NvidiaSmiPeakSampler.from_environment()
        fit_memory_sampler.start()
        torch.cuda.synchronize(profile_device)
        torch.cuda.reset_peak_memory_stats(profile_device)
    started = time.perf_counter()
    training_session = (
        results_mgr.training_session(method, object_name, len(support_paths))
        if results_mgr is not None
        else nullcontext()
    )
    try:
        with training_session:
            method.fit(support_paths, context)
    finally:
        if cost_profile:
            torch.cuda.synchronize(profile_device)
            fit_seconds = time.perf_counter() - started
            fit_torch_peak_gib = torch.cuda.max_memory_allocated(profile_device) / (1024 ** 3)
            fit_nvidia_smi_peak = fit_memory_sampler.stop()
            fit_timing = {
                "seconds": fit_seconds,
                "torch_peak_allocated_gib": fit_torch_peak_gib,
                "nvidia_smi_peak": fit_nvidia_smi_peak,
            }
        else:
            fit_seconds = time.perf_counter() - started
    checkpoint_path = (
        results_mgr.save_weights(method, object_name) if results_mgr is not None else None
    )
    grouped = dataset.get_test_images(object_name)
    queries = sorted(
        [
            (anomaly_type, image_path)
            for anomaly_type, image_paths in grouped.items()
            for image_path in image_paths
        ],
        key=lambda item: (Path(item[1]).name, item[0]),
    )
    if not queries:
        raise ValueError(f"{object_name}: test image set is empty")
    query_paths = [path for _, path in queries]
    sample_keys = [f"{kind}/{Path(path).name}" for kind, path in queries]
    if len(set(sample_keys)) != len(sample_keys) or len(set(query_paths)) != len(query_paths):
        raise ValueError(f"{object_name}: duplicate query identities")
    support_identity, support_content = _input_fingerprint(support_paths, dataset.data_root)
    query_identity, query_content = _input_fingerprint(query_paths, dataset.data_root)
    ground_truth_digest = hashlib.sha256()

    accumulator = StreamingSegmentationAccumulator(max_fpr=0.05)
    fusion_diagnostic = None
    calibration_diagnostic = None
    if config.get('evaluation', {}).get('calibration_diagnostic', False):
        from fmad.evaluation.calibration_diagnostic import CalibrationDiagnostic
        calibration_diagnostic = CalibrationDiagnostic()
    if config.get("evaluation", {}).get("fusion_diagnostic", False):
        from fmad.evaluation.fusion_diagnostic import FusionDiagnostic
        fusion_diagnostic = FusionDiagnostic()
    source_shift = None
    if config.get("evaluation", {}).get("source_shift_diagnostic", False):
        if fusion_diagnostic is None:
            raise ValueError("source_shift_diagnostic requires fusion_diagnostic")
        from fmad.evaluation.source_shift import SourceShiftDiagnostic
        source_shift = SourceShiftDiagnostic.calibrate(method, train_paths, support_paths, object_name)
    condition_strata_accumulators: dict[str, StreamingSegmentationAccumulator] = {}
    condition_strata_queries: dict[str, list[tuple[str, str]]] = {}
    condition_strata_enabled = False
    if config.get("evaluation", {}).get("condition_strata", False):
        if not isinstance(config["evaluation"]["condition_strata"], bool):
            raise ValueError("condition_strata must be a boolean")
        condition_strata_enabled = True
    anomaly_scores: dict[str, float] = {}
    aggregation_diagnostic_enabled = config.get("evaluation", {}).get("image_aggregation_diagnostic", False)
    if not isinstance(aggregation_diagnostic_enabled, bool):
        raise ValueError("image_aggregation_diagnostic must be a boolean")
    aggregation_scores: dict[str, dict[str, float | str]] = {}
    candidate_scores: dict[str, dict[str, float | str]] = {}
    inference_times: dict[str, float] = {}
    diagnostics: dict[str, dict] = {}
    cost_query_latencies_seconds: list[float] = []
    if cost_profile:
        query_memory_sampler = NvidiaSmiPeakSampler.from_environment()
        query_memory_sampler.start()
        torch.cuda.synchronize(profile_device)
        torch.cuda.reset_peak_memory_stats(profile_device)
    progress_dir = Path(run_dir) / "streaming_metrics" / object_name
    if debug_output:
        progress_dir.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]
        if cost_profile:
            torch.cuda.synchronize(profile_device)
            query_started = time.perf_counter()
        predictions = method.predict_batch([path for _, path in batch])
        if cost_profile:
            torch.cuda.synchronize(profile_device)
            batch_latency_seconds = time.perf_counter() - query_started
            cost_query_latencies_seconds.extend(
                [batch_latency_seconds / len(batch)] * len(batch)
            )
        if len(predictions) != len(batch):
            raise ValueError(
                f"{object_name}: predict_batch returned {len(predictions)} "
                f"outputs for {len(batch)} queries",
            )
        score_maps: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for (anomaly_type, image_path), prediction in zip(batch, predictions, strict=True):
            score_map = np.asarray(prediction.score_map, dtype=np.float32)
            if score_map.ndim != 2:
                raise ValueError(f"{object_name}: model score map must be 2D")
            score_maps.append(score_map)
            masks.append(
                _ground_truth_mask(
                    dataset,
                    object_name,
                    anomaly_type,
                    image_path,
                    score_map.shape,
                )
            )
            sample_key = f"{anomaly_type}/{Path(image_path).name}"
            ground_truth_digest.update(json.dumps([sample_key, list(masks[-1].shape)]).encode())
            ground_truth_digest.update(masks[-1].tobytes())
            if (
                not np.isfinite(prediction.image_score)
                or not np.isfinite(prediction.inference_seconds)
                or prediction.inference_seconds < 0
            ):
                raise ValueError("prediction score and non-negative timing must be finite")
            anomaly_scores[sample_key] = float(prediction.image_score)
            inference_times[sample_key] = float(
                batch_latency_seconds / len(batch)
                if cost_profile
                else prediction.inference_seconds
            )
            compact_diagnostics = dict(prediction.diagnostics)
            calibration_auxiliary = compact_diagnostics.pop('calibration_audit', None)
            if calibration_diagnostic is not None:
                if calibration_auxiliary is None:
                    raise ValueError('calibration_diagnostic requires calibration_audit')
                calibration_diagnostic.update(image_path, score_map, masks[-1], calibration_auxiliary)
            fusion_auxiliary = compact_diagnostics.pop("fusion_audit", None)
            image_candidates = compact_diagnostics.pop("image_score_candidates", None)
            if image_candidates is not None:
                if not all(np.isfinite(float(v)) for v in image_candidates.values()):
                    raise ValueError("image score candidates must be finite")
                candidate_scores[sample_key] = {
                    "label": "normal" if anomaly_type == "good" else "anomaly",
                    **{name: float(value) for name, value in image_candidates.items()},
                }
            if aggregation_diagnostic_enabled:
                if fusion_auxiliary is None or "raw_map" not in fusion_auxiliary:
                    raise ValueError("image_aggregation_diagnostic requires fusion raw_map auxiliary predictions")
                raw_map = np.asarray(fusion_auxiliary["raw_map"], dtype=np.float32)
                if raw_map.shape != score_map.shape:
                    raise ValueError("fusion raw_map must match prediction score map shape")
                raw_flat = raw_map.reshape(-1)
                final_flat = score_map.reshape(-1)
                aggregation_scores[sample_key] = {
                    "label": "normal" if anomaly_type == "good" else "anomaly",
                    "max_raw": float(np.max(raw_flat)),
                    "top1pct_mean_raw": _top_fraction_mean(raw_flat),
                    "max_final": float(np.max(final_flat)),
                    "top1pct_mean_final": _top_fraction_mean(final_flat),
                }
            if fusion_diagnostic is not None:
                if fusion_auxiliary is None:
                    raise ValueError("fusion_diagnostic requires method auxiliary predictions")
                fusion_diagnostic.update(image_path, anomaly_type, score_map, masks[-1], fusion_auxiliary)
            if source_shift is not None:
                source_shift.update(image_path, anomaly_type, score_map, masks[-1], fusion_auxiliary)
            if condition_strata_enabled:
                from fmad.evaluation.shift_diagnostic import acquisition_identity
                _, condition = acquisition_identity(image_path)
                if condition not in condition_strata_accumulators:
                    condition_strata_accumulators[condition] = StreamingSegmentationAccumulator(max_fpr=0.05)
                    condition_strata_queries[condition] = []
                condition_strata_accumulators[condition].update([score_map], [masks[-1]])
                condition_strata_queries[condition].append((anomaly_type, image_path))
            if debug_output and compact_diagnostics:
                diagnostics[sample_key] = compact_diagnostics
            if results_mgr is not None:
                results_mgr.save_anomaly_map(object_name, sample_key, score_map)
        accumulator.update(score_maps, masks)
        if debug_output:
            print(f"    processed {min(start + len(batch), len(queries))}/{len(queries)}")
        elif fusion_diagnostic is not None and (start % 32 == 0 or start + len(batch) == len(queries)):
            print(f"    fusion audit processed {min(start + len(batch), len(queries))}/{len(queries)}", flush=True)

    if cost_profile:
        torch.cuda.synchronize(profile_device)
        query_torch_peak_gib = torch.cuda.max_memory_allocated(profile_device) / (1024 ** 3)
        query_nvidia_smi_peak = query_memory_sampler.stop()

    final = accumulator.compute()
    labels = [1 if kind != "good" else 0 for kind, _ in queries]
    scores = [anomaly_scores[f"{kind}/{Path(path).name}"] for kind, path in queries]
    if len(set(labels)) > 1:
        try:
            from sklearn.metrics import roc_auc_score
            img_auroc = float(roc_auc_score(labels, scores))
        except Exception:
            img_auroc = 0.0
    else:
        img_auroc = 0.0

    metrics = {
        "seg_AUROC_0.05": final.seg_auroc_0_05,
        "seg_F1_oracle": final.seg_f1,
        "best_thre": final.best_threshold,
        "seg_AUPRO_0.05": final.seg_aupro_0_05,
        "seg_AUROC_1.00": getattr(final, "seg_auroc_1_00", 0.0),
        "seg_AUPRO_0.30": getattr(final, "seg_aupro_0_30", 0.0),
        "image_AUROC": img_auroc,
        "pixel_score_quantization": final.pixel_score_quantization,
        "metric_contract": _STREAMING_METRIC_CONTRACT,
        "evaluation_mode": "mini_batch_streaming",
        "inference_batch_size": batch_size,
        "coverage": {
            "prediction_map_count": final.prediction_map_count,
            "ground_truth_map_count": final.prediction_map_count,
            "shape_match_count": final.shape_match_count,
            "nonconstant_prediction_map_count": final.nonconstant_prediction_map_count,
            "non_finite_prediction_count": final.non_finite_prediction_count,
            "anomalous_region_count": accumulator.region_count,
        },
    }
    if debug_output:
        progress_dir.mkdir(parents=True, exist_ok=True)
        (progress_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    shift_diagnostic = None
    if config.get("evaluation", {}).get("shift_diagnostic", False):
        from fmad.evaluation.shift_diagnostic import run_shift_diagnostic
        shift_diagnostic = run_shift_diagnostic(
            method, dataset, object_name, seed, queries, _ground_truth_mask,
            mode=config["evaluation"]["shift_diagnostic"])
    metadata = dict(method.object_metadata())
    if calibration_diagnostic is not None:
        metadata['calibration_diagnostic'] = calibration_diagnostic.compute()
    if candidate_scores:
        from sklearn.metrics import roc_auc_score
        query_keys = [f"{kind}/{Path(path).name}" for kind, path in queries]
        if set(candidate_scores) != set(query_keys):
            raise ValueError("image score candidates must cover every query")
        names = sorted(set.intersection(*(set(row) - {"label"} for row in candidate_scores.values())))
        candidate_metrics = {
            name: (float(roc_auc_score(labels, [candidate_scores[key][name] for key in query_keys]))
                   if len(set(labels)) > 1 else 0.0)
            for name in names
        }
        primary_key = getattr(method, "image_score_key", None)
        metadata["image_score_candidates"] = {
            "metrics": {"image_AUROC": candidate_metrics},
            "top_fraction": 0.01,
            "k_rule": "max(1, floor(pixel_count * top_fraction))",
            "primary_key": primary_key,
            "primary_matches": primary_key in names and all(
                np.isclose(candidate_scores[key][primary_key], anomaly_scores[key], rtol=1e-6, atol=1e-7)
                for key in query_keys),
            "query_scores": candidate_scores,
        }
    if aggregation_diagnostic_enabled:
        from sklearn.metrics import roc_auc_score
        aggregate_metrics = {}
        for score_name in ("max_raw", "top1pct_mean_raw", "max_final", "top1pct_mean_final"):
            values = [aggregation_scores[f"{kind}/{Path(path).name}"][score_name] for kind, path in queries]
            aggregate_metrics[score_name] = float(roc_auc_score(labels, values)) if len(set(labels)) > 1 else 0.0
        quantiles = {}
        for label_name in ("normal", "anomaly"):
            label_rows = [row for row in aggregation_scores.values() if row["label"] == label_name]
            quantiles[label_name] = {
                score_name: {
                    str(q): float(np.quantile([row[score_name] for row in label_rows], q))
                    for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
                }
                for score_name in aggregate_metrics
            }
        metadata["image_aggregation_diagnostic"] = {
            "metrics": {"image_AUROC": aggregate_metrics},
            "top_fraction": 0.01,
            "k_rule": "max(1, floor(pixel_count * top_fraction))",
            "query_scores": aggregation_scores,
            "score_quantiles_by_label": quantiles,
            "max_final_matches_primary": all(
                np.isclose(aggregation_scores[f"{kind}/{Path(path).name}"]["max_final"], anomaly_scores[f"{kind}/{Path(path).name}"], rtol=1e-6, atol=1e-7)
                for kind, path in queries
            ),
        }
    if fusion_diagnostic is not None:
        metadata["fusion_diagnostic"] = fusion_diagnostic.compute()
    if source_shift is not None:
        metadata["source_shift_diagnostic"] = source_shift.compute()
    if shift_diagnostic is not None:
        metadata["shift_diagnostic"] = shift_diagnostic
    if condition_strata_enabled:
        from fmad.evaluation.fusion_diagnostic import summarize
        condition_strata_result = {}
        for condition, accumulator in condition_strata_accumulators.items():
            condition_summary = summarize(accumulator)
            queries_in_condition = condition_strata_queries[condition]
            condition_strata_result[condition] = {
                # summarize() returns metrics=None when a stratum lacks normal or defect mass.
                "AUPRO_0_05": (condition_summary.get("metrics") or {}).get("AUPRO_0_05"),
                "pAUROC_0_05": (condition_summary.get("metrics") or {}).get("pAUROC_0_05"),
                "reason": condition_summary.get("reason"),
                "queries": [f"{kind}/{Path(path).name}" for kind, path in queries_in_condition],
                "normal_pixels": condition_summary.get("coverage", {}).get("normal_pixels"),
                "defect_pixels": condition_summary.get("coverage", {}).get("defect_pixels"),
            }
        metadata["condition_strata"] = condition_strata_result
    if config.get("evaluation", {}).get("feature_mechanism", False):
        from fmad.evaluation.feature_mechanism import run_feature_mechanism
        metadata["feature_mechanism"] = run_feature_mechanism(
            method, dataset, object_name, queries, _ground_truth_mask)
    metadata.update(
        {
            "streaming_metrics": metrics,
            "prediction_diagnostics": diagnostics if debug_output else {},
            "protocol": {
                "support_paths": support_paths,
                "support_count": len(support_paths),
                "support_manifest_sha256": manifest_digest,
                "support_identity_sha256": support_identity,
                "support_content_sha256": support_content,
                "query_count": len(query_paths),
                "query_identity_sha256": query_identity,
                "query_content_sha256": query_content,
                "ground_truth_sha256": ground_truth_digest.hexdigest(),
                "context": context.__dict__,
            },
        }
    )
    if cost_profile:
        latency_ms = np.asarray(cost_query_latencies_seconds, dtype=np.float64) * 1000.0
        metadata["cost_profile"] = {
            "measurement_version": 1,
            "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",", 1)[0].strip(),
            "fit_seconds": fit_timing["seconds"],
            "fit_includes_support_image_decoding_and_feature_extraction": True,
            "fit_includes_bank_index_and_support_calibration": True,
            "fit_torch_peak_allocated_gib": fit_timing["torch_peak_allocated_gib"],
            "fit_nvidia_smi_peak": fit_timing["nvidia_smi_peak"],
            "query_count": int(latency_ms.size),
            "query_mean_ms": float(np.mean(latency_ms)),
            "query_p95_ms": float(np.percentile(latency_ms, 95)),
            "query_latencies_ms": latency_ms.tolist(),
            "query_torch_peak_allocated_gib": query_torch_peak_gib,
            "query_nvidia_smi_peak": query_nvidia_smi_peak,
            "query_timer_scope": "method.predict_batch with CUDA synchronization before and after each batch",
            "batch_size": batch_size,
            "backbone_warmup_iters_excluded": warmup_iters,
            "model_loading_excluded": True,
            "support_selection_and_input_fingerprinting_excluded": True,
            "checkpoint_loading_excluded": True,
        }
    metadata["backbone"] = {
        "name": getattr(method.backbone, "model_name", type(method.backbone).__name__),
        "device": str(getattr(method.backbone, "device", "unknown")),
        "resolution": context.resolution,
        "feature_layers": list(getattr(method.backbone, "feature_layers", ())),
        "provenance": getattr(method.backbone, "provenance", None),
    }
    metadata["artifacts"] = {
        "checkpoint": checkpoint_path,
        "anomaly_maps": (
            {
                "directory": f"anomaly_maps/{object_name}",
                "count": len(queries),
                "format": "npz",
                "array": "score_map",
                "dtype": "float32",
                "score_stage": "before_evaluator_float16_quantization",
            }
            if results_mgr is not None and results_mgr.output_options["save_anomaly_maps"]
            else None
        ),
    }
    return ObjectResult(
        object_name=object_name,
        anomaly_maps={},
        anomaly_scores=anomaly_scores,
        time_memorybank=fit_seconds,
        inference_times=inference_times,
        metadata=metadata,
    )


def run_pipeline(method, dataset, backbone, config: dict, output_dir: str) -> Dict:
    """Run the full anomaly detection pipeline.

    Args:
        method: Initialized BaseMethod instance.
        dataset: Initialized BaseDataset instance.
        backbone: Initialized BaseBackbone instance.
        config: Full configuration dict.
        output_dir: Base output directory.

    Returns:
        Dict with evaluation metrics.
    """
    evaluation = config.get("evaluation", {})
    if evaluation.get("streaming", True) is not True:
        raise ValueError("evaluation.streaming=false is not supported; streaming is mandatory")
    if config.get("eval_clf", False) or evaluation.get("eval_clf", False):
        raise ValueError("eval_clf is not supported by the segmentation pipeline")
    if evaluation.get("eval_segm", True) is not True:
        raise ValueError("eval_segm must be true for the segmentation pipeline")
    validate_protocol(config.get("protocol", {}))
    output_options = validate_output_options(config)
    if output_options["save_weights"]:
        saver = getattr(method, "save_checkpoint", None)
        if not callable(saver) or getattr(saver, "__func__", None) is BaseMethod.save_checkpoint:
            raise NotImplementedError(f"{type(method).__name__} does not support checkpoint saving")
    if not isinstance(config.get("debug_output", False), bool):
        raise ValueError("debug_output must be a boolean")
    condition_strata_val = config.get("evaluation", {}).get("condition_strata", False)
    if condition_strata_val is not False and not isinstance(condition_strata_val, bool):
        raise ValueError("evaluation.condition_strata must be a boolean")
    seed = config.get("seed", 0)
    seed_generators(seed)
    method_name = config["method"]
    dataset_name = config["dataset"]
    backbone_name = config["backbone"]

    # Run per-object
    objects = dataset.get_objects()
    if not objects:
        raise ValueError("dataset object selection is empty")
    object_results: List[ObjectResult] = []
    metric_contract = config.get("evaluation", {}).get(
        "metric_contract",
        _STREAMING_METRIC_CONTRACT,
    )
    if metric_contract != _STREAMING_METRIC_CONTRACT:
        raise ValueError(
            f"streaming evaluation requires metric_contract={_STREAMING_METRIC_CONTRACT}",
        )
    if config.get("evaluation", {}).get("binary_postprocess", "none") != "none":
        raise ValueError(
            "streaming evaluation does not retain dense maps required by binary_postprocess",
        )
    pro_limit = float(config.get("evaluation", {}).get("pro_integration_limit", 0.05))
    if pro_limit != 0.05:
        raise ValueError("streaming evaluation currently enforces pro_integration_limit=0.05")
    inference_batch_size = evaluation.get("inference_batch_size", 1)
    if (
        isinstance(inference_batch_size, bool)
        or not isinstance(inference_batch_size, int)
        or inference_batch_size <= 0
    ):
        raise ValueError("inference_batch_size must be positive")

    results_mgr = ResultsManager(
        base_dir=output_dir,
        method=method_name,
        dataset=dataset_name,
        backbone=backbone_name,
        seed=seed,
        config=config,
    )
    results_mgr.save_config(config)
    run_dir = results_mgr.run_dir

    print(f"\n{'=' * 60}")
    print(f"  Method: {method_name} | Dataset: {dataset_name} | Backbone: {backbone_name}")
    print(f"  Seed: {seed} | Objects: {len(objects)}")
    print(f"  Output: {run_dir}")
    print(f"{'=' * 60}\n")

    for obj_info in objects:
        print(
            f"--- {obj_info.name} (resolution={obj_info.resolution}, "
            f"masking={obj_info.masking}, rotation={obj_info.rotation}) ---"
        )

        t0 = time.time()
        result = _run_streaming_object(
            method,
            dataset,
            obj_info.name,
            seed,
            run_dir,
            inference_batch_size,
            config=config,
            debug_output=bool(config.get("debug_output", False)),
            results_mgr=results_mgr,
        )
        elapsed = time.time() - t0
        object_results.append(result)

        n_samples = len(result.inference_times)
        mean_time = sum(result.inference_times.values()) / max(n_samples, 1)
        print(
            f"    {n_samples} samples, mean inference: {mean_time:.3f} s/sample, "
            f"total: {elapsed:.1f}s"
        )

    # Save timing CSV
    if config.get("debug_output", False):
        results_mgr.save_object_times(object_results)
    results_mgr.save_run_manifest(object_results)

    # Evaluate
    print(f"\n{'=' * 60}")
    print("  Evaluating...")
    print(f"{'=' * 60}")

    metrics = {
        result.object_name: result.metadata["streaming_metrics"] for result in object_results
    }
    object_metrics = list(metrics.values())
    metrics["mean_segmentation_auroc_0.05"] = float(
        sum(row["seg_AUROC_0.05"] for row in object_metrics) / len(object_metrics)
    )
    metrics["mean_segmentation_f1_oracle"] = float(
        sum(row["seg_F1_oracle"] for row in object_metrics) / len(object_metrics)
    )
    metrics["mean_segmentation_aupro_0.05"] = float(
        sum(row["seg_AUPRO_0.05"] for row in object_metrics) / len(object_metrics)
    )
    if any("seg_AUROC_1.00" in row for row in object_metrics):
        metrics["mean_segmentation_auroc_1.00"] = float(
            sum(row.get("seg_AUROC_1.00", 0.0) for row in object_metrics) / len(object_metrics)
        )
    if any("seg_AUPRO_0.30" in row for row in object_metrics):
        metrics["mean_segmentation_aupro_0.30"] = float(
            sum(row.get("seg_AUPRO_0.30", 0.0) for row in object_metrics) / len(object_metrics)
        )
    if any("image_AUROC" in row for row in object_metrics):
        metrics["mean_image_auroc"] = float(
            sum(row.get("image_AUROC", 0.0) for row in object_metrics) / len(object_metrics)
        )

    # Save & accumulate
    results_mgr.save_metrics(metrics)
    results_mgr.append_summary(metrics, object_results)

    # Print summary
    print(f"\n{'=' * 72}")
    print("  Results")
    print(f"{'=' * 72}")
    print(f"  {'Object':<14} {'Pix-AUC':>10} {'PRO@.30':>10} {'PRO@.05':>10} {'Img-AUC':>10} {'F1-ora':>9}")
    print(f"  {'-' * 66}")
    for obj_name, obj_m in metrics.items():
        if not isinstance(obj_m, dict) or "seg_AUROC_0.05" not in obj_m:
            continue
        p_auc = obj_m.get("seg_AUROC_1.00", 0.0)
        pro30 = obj_m.get("seg_AUPRO_0.30", 0.0)
        pro05 = obj_m.get("seg_AUPRO_0.05", 0.0)
        i_auc = obj_m.get("image_AUROC", 0.0)
        f1 = obj_m.get("seg_F1_oracle", 0.0)
        print(
            f"  {obj_name:<14} {p_auc:>10.4f} {pro30:>10.4f} {pro05:>10.4f} {i_auc:>10.4f} {f1:>9.4f}"
        )

    m_pauc = metrics.get("mean_segmentation_auroc_1.00", 0.0)
    m_pro30 = metrics.get("mean_segmentation_aupro_0.30", 0.0)
    m_pro05 = metrics.get("mean_segmentation_aupro_0.05", 0.0)
    m_iauc = metrics.get("mean_image_auroc", 0.0)
    m_f1 = metrics.get("mean_segmentation_f1_oracle", 0.0)
    print(f"  {'-' * 66}")
    print(f"  {'MEAN':<14} {m_pauc:>10.4f} {m_pro30:>10.4f} {m_pro05:>10.4f} {m_iauc:>10.4f} {m_f1:>9.4f}")
    print(f"\n  Summary appended to: {results_mgr.summary_path}")

    return metrics
