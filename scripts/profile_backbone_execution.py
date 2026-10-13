#!/usr/bin/env python3
"""Bounded same-input DINOv2 execution probe for attention and compile options.

The probe selects at most two dataset query images and the configured normal
support identities, but never reads ground-truth masks, fits a method, or saves
feature maps or model weights.  It reuses one preprocessed CPU tensor per image
and resolution across every execution arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any


ARMS = {
    "default": {"inference_attention_backend": "default", "inference_compile_mode": "none"},
    "strictflash": {"inference_attention_backend": "sdpa_flash", "inference_compile_mode": "none"},
    "compiledefault": {"inference_attention_backend": "default", "inference_compile_mode": "default"},
}
DEFAULT_RESOLUTIONS = [448, 672, 896, 1120]
LOGGER = logging.getLogger("profile_backbone_execution")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    return _sha256_bytes(value.numpy().tobytes())


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    """Atomically checkpoint JSON so earlier arms survive a compile timeout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _nvidia_smi_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=uuid,name,clocks.sm,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
        rows = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 5:
                rows.append(
                    {
                        "uuid": parts[0],
                        "name": parts[1],
                        "sm_clock_mhz": parts[2],
                        "power_draw_w": parts[3],
                        "temperature_c": parts[4],
                    }
                )
        return {"visible_devices": rows, "error": None}
    except Exception as error:  # Diagnostics must not stop a probe.
        return {"visible_devices": [], "error": f"{type(error).__name__}: {error}"}


def _operator_profile(model_call, image_tensor, device: str) -> dict[str, Any]:
    import torch

    activities = [torch.profiler.ProfilerActivity.CPU]
    if str(device).startswith("cuda"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as profile:
        _ = model_call(image_tensor)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)

    interesting = []
    all_names = []
    for item in profile.key_averages(group_by_input_shape=True):
        name = item.key
        all_names.append(name)
        normalized = name.lower()
        if (
            "scaled_dot_product" in normalized
            or "flash_attention" in normalized
            or "efficient_attention" in normalized
            or "xformers" in normalized
            or "attention" in normalized and ("cuda" in normalized or "aten" in normalized)
        ):
            interesting.append(
                {
                    "operator": name,
                    "input_shapes": _json_safe(item.input_shapes),
                    "calls": int(item.count),
                    "self_cuda_time_us": float(getattr(item, "self_cuda_time_total", 0.0)),
                    "cuda_time_us": float(getattr(item, "cuda_time_total", 0.0)),
                    "self_cpu_time_us": float(getattr(item, "self_cpu_time_total", 0.0)),
                }
            )
    return {
        "attention_operators": interesting,
        "all_operator_names": sorted(set(all_names)),
    }


def _as_cpu_float32(value: Any):
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _paired_difference(reference: list[Any], candidate: list[Any]) -> list[dict[str, Any]]:
    import numpy as np

    if len(reference) != len(candidate):
        return [{"error": f"feature layer count mismatch: {len(reference)} vs {len(candidate)}"}]
    rows = []
    for index, (ref_value, cand_value) in enumerate(zip(reference, candidate, strict=True)):
        ref = _as_cpu_float32(ref_value)
        cand = _as_cpu_float32(cand_value)
        if ref.shape != cand.shape:
            rows.append(
                {
                    "layer_index": index,
                    "reference_shape": list(ref.shape),
                    "candidate_shape": list(cand.shape),
                    "error": "feature shape mismatch",
                }
            )
            continue
        delta = np.abs(cand - ref)
        denom = float(np.abs(ref).sum(dtype=np.float64))
        rows.append(
            {
                "layer_index": index,
                "shape": list(ref.shape),
                "max_absolute_difference": float(delta.max(initial=0.0)),
                "relative_l1_difference": float(delta.sum(dtype=np.float64) / max(denom, 1e-30)),
            }
        )
    return rows


def _pick_queries(dataset, object_name: str, count: int, start_index: int) -> list[str]:
    grouped = dataset.get_test_images(object_name)
    paths = sorted({str(path) for members in grouped.values() for path in members})
    if not paths:
        raise ValueError(f"{object_name}: dataset API returned no query images")
    if start_index < 0 or start_index >= len(paths):
        raise ValueError(f"query index {start_index} is outside [0, {len(paths) - 1}]")
    selected = paths[start_index : start_index + count]
    # If the requested count crosses the final item, wrap in deterministic order.
    if len(selected) < count:
        selected.extend(paths[: count - len(selected)])
    return selected


def _resolve_resolutions(config: dict, explicit: list[int] | None) -> list[int]:
    if explicit is not None:
        values = explicit
    else:
        values = config.get("method_config", {}).get("resolutions", DEFAULT_RESOLUTIONS)
        if not isinstance(values, list) or len(values) != 4:
            values = DEFAULT_RESOLUTIONS
    values = [int(value) for value in values]
    if len(values) != 4 or any(value <= 0 for value in values) or len(set(values)) != 4:
        raise ValueError("exactly four distinct positive resolutions are required")
    return values


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)


def _load_config(path: Path) -> dict:
    import yaml

    config = yaml.safe_load(path.read_text()) or {}
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    return config


def profile(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import yaml

    from fmad.protocol import seed_generators, select_support_paths
    import fmad.datasets  # noqa: F401 - register dataset classes
    from fmad.registry import DATASET_REGISTRY
    from fmad.backbones.dinov2 import DINOv2Backbone

    config_path = Path(args.config).resolve()
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes) or {}
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    if config.get("backbone", "dinov2_vitl14") not in {
        "dinov2_vitl14", "dinov2_vitl14_reg", "dinov2_vits14"
    }:
        raise ValueError("the execution probe currently supports the DINOv2 registered backbones")
    data_root = args.data_root or config.get("data_root")
    if not data_root:
        raise ValueError("pass --data-root or set data_root in the config")
    device = args.device or config.get("device", "cuda:0")
    if not str(device).startswith("cuda"):
        raise ValueError("the backbone execution probe requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device_index = torch.device(device).index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)

    seed = int(config.get("seed", 0))
    seed_generators(seed)
    dataset_config = dict(config.get("dataset_config", {}))
    object_name = args.object
    if object_name is None:
        configured_objects = dataset_config.get("objects")
        if isinstance(configured_objects, list) and configured_objects:
            object_name = configured_objects[0]
        else:
            objects = DATASET_REGISTRY.get(config.get("dataset", "mvtec_ad2"))(
                str(data_root), dataset_config
            ).get_objects()
            if not objects:
                raise ValueError("dataset contains no objects")
            object_name = objects[0].name
    dataset = DATASET_REGISTRY.get(config.get("dataset", "mvtec_ad2"))(
        data_root=str(data_root), config=dataset_config
    )
    object_info = dataset.get_object_info(object_name)
    support, support_manifest_sha256 = select_support_paths(
        sorted(dataset.get_train_images(object_name)),
        config.get("protocol", {}).get("shots", 16),
        seed,
        manifest_path=(
            Path(config["protocol"]["fixed_reference_manifest"])
            if config.get("protocol", {}).get("fixed_reference_manifest")
            else None
        ),
        object_name=object_name,
    )
    query_paths = _pick_queries(dataset, object_name, args.queries, args.query_index)
    resolutions = _resolve_resolutions(config, args.resolutions)
    selected_modes = args.modes
    backbone_config = dict(config.get("backbone_config", {}))
    backbone_config["inference_autocast_dtype"] = args.autocast_dtype

    report: dict[str, Any] = {
        "status": "running",
        "scope": "single-object development execution probe; query labels and ground truth unused",
        "config_path": str(config_path),
        "config_sha256": _sha256_bytes(config_bytes),
        "probe_sha256": _sha256_file(Path(__file__)),
        "dataset": config.get("dataset", "mvtec_ad2"),
        "data_root": str(Path(data_root).resolve()),
        "object": object_name,
        "object_resolution": int(object_info.resolution),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device_index),
        "cuda_device_capability": list(torch.cuda.get_device_capability(device_index)),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "seed": seed,
        "support_selection": {
            "shots": len(support),
            "paths": support,
            "manifest_sha256": support_manifest_sha256,
            "selection_only": True,
            "support_images_loaded_or_fitted": False,
        },
        "queries": [
            {"path": path, "file_sha256": _sha256_file(path)} for path in query_paths
        ],
        "resolutions": resolutions,
        "input_preprocessing": {
            "source": "DINOv2Backbone.prepare_image",
            "same_preprocessed_tensor_reused_for_every_execution_arm": True,
            "feature_layer_indices": list(backbone_config.get("feature_layers") or (5, 11, 17, 23)),
        },
        "backbone_config": backbone_config,
        "timing_contract": {
            "cold_call": "first synchronized extract_features call at each input resolution; model initialization measured separately",
            "warmup_iterations_per_resolution": args.warmup,
            "timed_repetitions_per_resolution": args.repeats,
            "synchronized": True,
            "measurement_warning": "GPU clock and power state can vary; compare paired arms within this run and inspect per-arm GPU snapshots.",
        },
        "arm_order": selected_modes,
        "arms": {},
    }
    _write_report(args.output, report)

    # Construct the model for the first selected arm and preprocess once.
    tensors_by_query_resolution: dict[tuple[str, int], tuple[Any, tuple[int, int]]] = {}
    backbone_instances: dict[str, Any] = {}
    feature_outputs: dict[tuple[str, str, int], list[Any]] = {}
    baseline_mode = selected_modes[0]

    def make_backbone(mode: str):
        options = dict(backbone_config)
        options.update(ARMS[mode])
        return DINOv2Backbone(
            model_name=config.get("backbone", "dinov2_vitl14"),
            device=str(device),
            **options,
        )

    try:
        LOGGER.info("preparing common input tensors before execution arms")
        preprocessing_backbone = make_backbone(baseline_mode)
        # prepare_image follows production resizing, normalization, and patch crop.
        tensor_metadata = {}
        for query in query_paths:
            for resolution in resolutions:
                tensor, grid = preprocessing_backbone.prepare_image(query, resolution=resolution)
                tensors_by_query_resolution[(query, resolution)] = (tensor.detach().cpu().contiguous(), grid)
                key = f"{Path(query).name}@{resolution}"
                tensor_metadata[key] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "grid_size": list(grid),
                    "sha256": _tensor_sha256(tensor),
                }
        report["input_preprocessing"]["tensors"] = tensor_metadata
        # Reuse the loaded first-arm backbone after preprocessing.
        backbone_instances[baseline_mode] = preprocessing_backbone
        report["arms"][baseline_mode] = {
            "status": "running",
            "options": dict(ARMS[baseline_mode]),
            "started_unix_seconds": time.time(),
            "gpu_snapshot_before": _nvidia_smi_snapshot(),
            "resolutions": {},
        }
        _write_report(args.output, report)

        baseline_outputs: dict[tuple[str, int], list[Any]] = {}
        for mode in selected_modes:
            arm = report["arms"].setdefault(
                mode,
                {
                    "status": "running",
                    "options": dict(ARMS[mode]),
                    "started_unix_seconds": time.time(),
                    "gpu_snapshot_before": _nvidia_smi_snapshot(),
                    "resolutions": {},
                },
            )
            _write_report(args.output, report)
            load_seconds = 0.0
            try:
                if mode in backbone_instances:
                    backbone = backbone_instances[mode]
                else:
                    load_start = time.perf_counter()
                    backbone = make_backbone(mode)
                    # Force lazy model initialization without doing a measured forward.
                    backbone._ensure_loaded()
                    torch.cuda.synchronize(device)
                    load_seconds = time.perf_counter() - load_start
                    backbone_instances[mode] = backbone
                wrapper = backbone._wrapper
                if wrapper is None:
                    raise RuntimeError("DINOv2 wrapper was not initialized")
                torch.cuda.synchronize(device)
                arm["model_initialization_seconds"] = load_seconds
                arm["model_hash"] = backbone.backbone_state_sha256
                arm["backbone_provenance"] = dict(backbone.provenance or {})
                arm["feature_output_dtype"] = "float32"
                arm["gpu_snapshot_after_model_load"] = _nvidia_smi_snapshot()

                for query in query_paths:
                    query_key = Path(query).name
                    for resolution in resolutions:
                        shape_key = f"{query_key}@{resolution}"
                        row: dict[str, Any] = {
                            "status": "running",
                            "input_tensor_sha256": tensor_metadata[shape_key]["sha256"],
                            "input_shape": tensor_metadata[shape_key]["shape"],
                            "input_dtype": tensor_metadata[shape_key]["dtype"],
                            "grid_size": tensor_metadata[shape_key]["grid_size"],
                        }
                        arm["resolutions"][shape_key] = row
                        _write_report(args.output, report)
                        try:
                            image_tensor, _ = tensors_by_query_resolution[(query, resolution)]
                            torch.cuda.synchronize(device)
                            torch.cuda.reset_peak_memory_stats(device)
                            cold_start = time.perf_counter()
                            output = backbone.extract_features(image_tensor)
                            torch.cuda.synchronize(device)
                            cold_seconds = time.perf_counter() - cold_start
                            if not output:
                                raise RuntimeError("extract_features returned no feature layers")
                            for index, feature in enumerate(output):
                                value = _as_cpu_float32(feature)
                                if not bool(torch.isfinite(torch.from_numpy(value)).all()):
                                    raise FloatingPointError(f"non-finite values in feature layer {index}")
                            row["cold_first_call_seconds"] = cold_seconds
                            row["feature_layers"] = [
                                {
                                    "layer_index": index,
                                    "shape": list(_as_cpu_float32(feature).shape),
                                    "dtype": str(feature.dtype) if hasattr(feature, "dtype") else "float32",
                                }
                                for index, feature in enumerate(output)
                            ]
                            feature_outputs[(mode, query, resolution)] = output
                            if mode == baseline_mode:
                                baseline_outputs[(query, resolution)] = output

                            warmup_times = []
                            for _ in range(args.warmup):
                                started = time.perf_counter()
                                _ = backbone.extract_features(image_tensor)
                                torch.cuda.synchronize(device)
                                warmup_times.append(time.perf_counter() - started)
                            row["warmup_seconds"] = warmup_times

                            profile_start = time.perf_counter()
                            row["operator_profile"] = _operator_profile(
                                backbone.extract_features, image_tensor, str(device)
                            )
                            torch.cuda.synchronize(device)
                            row["operator_profile_seconds"] = time.perf_counter() - profile_start

                            repetitions = []
                            for _ in range(args.repeats):
                                torch.cuda.synchronize(device)
                                started = time.perf_counter()
                                _ = backbone.extract_features(image_tensor)
                                torch.cuda.synchronize(device)
                                repetitions.append(time.perf_counter() - started)
                            row["warmed_repeated_seconds"] = repetitions
                            row["warmed_summary_seconds"] = {
                                "min": min(repetitions),
                                "median": statistics.median(repetitions),
                                "mean": statistics.mean(repetitions),
                                "max": max(repetitions),
                            }
                            row["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
                            row["peak_allocated_gib"] = float(
                                torch.cuda.max_memory_allocated(device) / (1024**3)
                            )
                            if mode != baseline_mode and (query, resolution) in baseline_outputs:
                                row["paired_feature_difference_vs"] = baseline_mode
                                row["paired_feature_difference"] = _paired_difference(
                                    baseline_outputs[(query, resolution)], output
                                )
                            row["status"] = "completed"
                            LOGGER.info(
                                "%s %s cold %.4fs warmed median %.4fs operators=%s",
                                mode,
                                shape_key,
                                cold_seconds,
                                row["warmed_summary_seconds"]["median"],
                                [op["operator"] for op in row["operator_profile"]["attention_operators"]],
                            )
                        except Exception as error:
                            row["status"] = "failed"
                            row["error"] = {
                                "type": type(error).__name__,
                                "message": str(error),
                                "traceback": traceback.format_exc(),
                            }
                            LOGGER.exception("arm %s failed at %s", mode, shape_key)
                        _write_report(args.output, report)
                arm["status"] = (
                    "completed"
                    if all(value.get("status") == "completed" for value in arm["resolutions"].values())
                    else "partial_failure"
                )
            except Exception as error:
                arm["status"] = "failed"
                arm["error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
                LOGGER.exception("execution arm %s failed", mode)
            finally:
                arm["finished_unix_seconds"] = time.time()
                arm["gpu_snapshot_after"] = _nvidia_smi_snapshot()
                _write_report(args.output, report)

        report["status"] = (
            "completed"
            if all(arm.get("status") == "completed" for arm in report["arms"].values())
            else "completed_with_failures"
        )
    except Exception as error:
        report["status"] = "failed"
        report["fatal_error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        LOGGER.exception("probe setup failed")
    finally:
        report["finished_unix_seconds"] = time.time()
        _write_report(args.output, report)
        # Avoid holding additional parameter copies if one arm failed to finish.
        backbone_instances.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment YAML config")
    parser.add_argument("--data-root", help="Override config data_root")
    parser.add_argument("--object", help="Dataset object; default is first configured/dataset object")
    parser.add_argument("--device", help="CUDA device, for example cuda:0")
    parser.add_argument("--output", default="results/local_runs/backbone_execution/profile.json")
    parser.add_argument("--queries", type=int, choices=(1, 2), default=1)
    parser.add_argument("--query-index", type=int, default=0, help="Start index in sorted dataset query paths")
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        help="Four short-edge sizes; otherwise use config method_config.resolutions or 448 672 896 1120",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(ARMS),
        default=["default", "strictflash", "compiledefault"],
        help="Execution arms in paired measurement order",
    )
    parser.add_argument("--autocast-dtype", choices=("fp16", "bf16", "none"), default="fp16")
    parser.add_argument("--warmup", type=int, default=2, help="Warm calls after the cold call, per shape")
    parser.add_argument("--repeats", type=int, default=5, help="Synchronized warmed timings per shape (minimum 5)")
    args = parser.parse_args(argv)
    if not args.modes or len(set(args.modes)) != len(args.modes):
        parser.error("--modes must be non-empty and unique")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.repeats < 5:
        parser.error("--repeats must be >= 5 for a warmed timing estimate")
    if args.query_index < 0:
        parser.error("--query-index must be >= 0")
    args.config = str(Path(args.config).resolve())
    args.output = Path(args.output).resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.output.with_suffix(".log"))
    LOGGER.info("starting bounded DINOv2 backbone execution probe")
    report = profile(args)
    LOGGER.info("probe status=%s report=%s", report["status"], args.output)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
