#!/usr/bin/env python3
"""Synchronized, non-mutating component timing for a local RSG4 sample."""

import argparse
import gc
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch
import yaml

from fmad.backbones.dinov2 import DINOv2Backbone
import fmad.datasets  # register datasets
import fmad.methods  # register methods
from fmad.registry import DATASET_REGISTRY, METHOD_REGISTRY
import fmad.methods.reference_matching as matching
from fmad.protocol import MethodContext, select_support_paths, seed_generators


class Timer:
    def __init__(self):
        self.phase = "fit"
        self.last_resolution = None
        self.values = defaultdict(float)
        self.calls = defaultdict(int)
        self.stack = []
        self.originals = []

    def sync(self):
        torch.cuda.synchronize()

    @contextmanager
    def measure(self, label):
        self.sync()
        start = time.perf_counter()
        entry = [label, start, 0.0]
        self.stack.append(entry)
        try:
            yield
        finally:
            self.sync()
            elapsed = time.perf_counter() - start
            self.stack.pop()
            self.values[label] += elapsed - entry[2]
            self.calls[label] += 1
            if self.stack:
                self.stack[-1][2] += elapsed

    def wrap(self, obj, name, label_fn):
        original = getattr(obj, name)

        @wraps(original)
        def timed(*args, **kwargs):
            label = label_fn(*args, **kwargs)
            with self.measure(label):
                return original(*args, **kwargs)

        self.originals.append((obj, name, original))
        setattr(obj, name, timed)

    def restore(self):
        for obj, name, original in reversed(self.originals):
            setattr(obj, name, original)
        self.originals.clear()


def profile(args, reference=None):
    config_bytes = Path(args.config).read_bytes()
    config = yaml.safe_load(config_bytes)
    source_marker = Path(__file__).resolve().parents[1] / ".fmad-source-sha256"
    dataset = DATASET_REGISTRY.get(config["dataset"])(args.data_root, config["dataset_config"])
    info = dataset.get_object_info(args.object)
    support, manifest_hash = select_support_paths(
        sorted(dataset.get_train_images(args.object)), config["protocol"]["shots"],
        config["seed"], manifest_path=Path(config["protocol"]["fixed_reference_manifest"]),
        object_name=args.object)
    grouped = dataset.get_test_images(args.object)
    good = sorted(grouped.get("good", []))
    bad = sorted(p for kind, paths in grouped.items() if kind != "good" for p in paths)
    if not good or not bad:
        raise ValueError("profile needs both normal and anomalous query images")
    n_good = min(len(good), args.queries // 2)
    n_bad = min(len(bad), args.queries - n_good)
    def spread(paths, count):
        return [paths[i] for i in np.linspace(0, len(paths) - 1, count, dtype=int)]
    queries = spread(good, n_good) + spread(bad, n_bad)
    seed_generators(config["seed"])
    fmad.methods.register_methods(config["method"])
    backbone = DINOv2Backbone(config["backbone"], "cuda:0", **config["backbone_config"])
    method = METHOD_REGISTRY.get(config["method"])(backbone, config["method_config"])
    context = MethodContext(args.object, config["seed"], info.resolution, info.masking, info.rotation)
    backbone.set_resolution(info.resolution)
    backbone.warmup(support[0], iters=config["protocol"]["warmup_iters"])
    timer = Timer()
    timer.wrap(matching, "_read_rgb", lambda *_a, **_kw: f"{timer.phase}.image_read")
    timer.wrap(matching, "immerkaer_sigma", lambda *_a, **_kw: f"{timer.phase}.noise_guard")
    timer.wrap(backbone, "prepare_image", lambda *_a, **_kw: f"{timer.phase}.preprocess")
    timer.wrap(backbone, "extract_features", lambda *_a, **_kw: f"{timer.phase}.backbone")
    timer.wrap(method, "_extract", lambda _image, resolution, **_kw: f"{timer.phase}.feature_r{resolution}")
    timer.wrap(method, "_bank_for", lambda *_a, **_kw: "fit.bank_assembly")
    timer.wrap(method, "_fit_scales_for_resolution", lambda resolution: f"fit.calibration_r{resolution}")
    timer.wrap(matching, "_distance_chunks", lambda *_a, **_kw: f"{timer.phase}.nearest_neighbor")
    if config["method"] == "reference_matching_fast":
        import fmad.methods.reference_matching_fast as fast_matching
        timer.wrap(fast_matching, "_build_shared_bag_union", lambda *_a, **_kw: "fit.bank_assembly")
        for name in ("_masked_loo_distances", "_shared_bag_distances"):
            if hasattr(fast_matching, name):
                timer.wrap(fast_matching, name, lambda *_a, **_kw: f"{timer.phase}.nearest_neighbor")
    timer.wrap(method, "_fuse_layers", lambda *_a, **_kw: f"{timer.phase}.layer_fusion")
    timer.wrap(method, "_score_tokens", lambda *_a, **_kw: f"query.matching_r{timer.last_resolution}")
    timer.wrap(method, "_postprocess_prediction", lambda *_a, **_kw: "query.component_quantile")

    # _score_tokens follows _extract in the production query path.
    extracted = method._extract
    @wraps(extracted)
    def remember_resolution(image, resolution, *a, **kw):
        timer.last_resolution = resolution
        return extracted(image, resolution, *a, **kw)
    method._extract = remember_resolution

    timer.sync()
    torch.cuda.reset_peak_memory_stats()
    fit_start = time.perf_counter()
    method.fit(support, context)
    timer.sync()
    fit_s = time.perf_counter() - fit_start
    fit_peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    torch.cuda.reset_peak_memory_stats()
    timer.phase = "query"
    latencies = []
    output_hash = hashlib.sha256()
    predictions, deviations = [], []
    for path in queries:
        timer.sync()
        start = time.perf_counter()
        score, image_score = method.predict(path)
        timer.sync()
        latencies.append(time.perf_counter() - start)
        if not np.isfinite(score).all() or not np.isfinite(image_score):
            raise ValueError(f"Nonfinite prediction: {path}")
        if reference is None:
            predictions.append((score.copy(), float(image_score)))
        else:
            expected, expected_image = reference["predictions"][len(deviations)]
            if expected.shape != score.shape:
                raise ValueError("Reference/candidate map shapes differ")
            difference = np.abs(score.astype(np.float64) - expected)
            deviations.append({"path": str(path), "map_max_abs": float(difference.max()),
                               "map_mean_abs": float(difference.mean()),
                               "map_relative_l1": float(difference.sum() / max(np.abs(expected).sum(dtype=np.float64), 1e-12)),
                               "image_score_abs": abs(float(image_score) - expected_image)})
        output_hash.update(score.tobytes())
        output_hash.update(str(image_score).encode())
    query_peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    totals = {"fit": fit_s, "query": sum(latencies)}
    components = {}
    for phase in totals:
        parts = {k.split(".", 1)[1]: v for k, v in timer.values.items() if k.startswith(phase + ".")}
        # Nested wrappers were recorded exclusive of their children. The unwrapped
        # residual covers method setup, aggregation, transfers, and timer overhead.
        parts["other_and_instrumentation"] = max(0.0, totals[phase] - sum(parts.values()))
        components[phase] = {k: {"seconds": v, "fraction": v / totals[phase]}
                             for k, v in sorted(parts.items())}
    report = {
        "scope": f"single-class development descriptive profile, {config['dataset']}",
        "method": config["method"], "backbone_provenance": backbone.provenance,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "source_sha256": source_marker.read_text().strip() if source_marker.exists() else None,
        "profiler_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "support_manifest_sha256": manifest_hash,
        "object": args.object, "shots": len(support), "query_count": len(queries),
        "query_paths": queries, "output_sha256": output_hash.hexdigest(),
        "fit_seconds": fit_s, "query_total_seconds": totals["query"],
        "query_mean_seconds": statistics.mean(latencies),
        "query_p95_seconds": float(np.percentile(latencies, 95)),
        "query_seconds": latencies,
        "fit_peak_allocated_gib": fit_peak_gib,
        "query_peak_allocated_gib": query_peak_gib,
        "components": components,
        "notes": "Exclusive synchronized wall time; instrumentation adds CUDA synchronizations. Percentages include explicit unwrapped residual. GPU peak is not additive across components.",
    }
    calibration = {str(res): [x.detach().cpu().tolist() for x in values]
                   for res, values in method.bag_scales.items()}
    report["bag_scales"] = calibration
    report["bag_spike_tau"] = {str(res): [[float(tail.get("spike_tau", 0)) for tail in bag]
                                          for bag in bags]
                               for res, bags in method.bag_tails.items()}
    report["noise_guard_log"] = method.noise_guard_log
    if reference is not None:
        if queries != reference["query_paths"] or manifest_hash != reference["support_manifest_sha256"]:
            raise ValueError("Reference/candidate query or support identities differ")
        report["reference_config_sha256"] = reference["config_sha256"]
        report["prediction_deviations"] = deviations
        report["max_map_abs_deviation"] = max(x["map_max_abs"] for x in deviations)
        report["max_map_relative_l1"] = max(x["map_relative_l1"] for x in deviations)
        report["max_scale_abs_deviation"] = max(
            float(np.max(np.abs(np.asarray(values) - np.asarray(reference["bag_scales"][res]))))
            for res, values in calibration.items())
        report["max_spike_tau_abs_deviation"] = max(
            float(np.max(np.abs(np.asarray(values) - np.asarray(reference["bag_spike_tau"][res]))))
            for res, values in report["bag_spike_tau"].items())
        report["noise_guard_decisions_equal"] = (
            [entry["fired"] for entry in report["noise_guard_log"]] ==
            [entry["fired"] for entry in reference["noise_guard_log"]])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_name("config.yaml").write_bytes(config_bytes)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("object", "shots", "query_count", "fit_seconds", "query_mean_seconds", "fit_peak_allocated_gib", "query_peak_allocated_gib")}, indent=2))
    print(output, flush=True)
    timer.restore()
    result = {key: report[key] for key in ("query_paths", "support_manifest_sha256", "config_sha256", "bag_scales", "bag_spike_tau", "noise_guard_log")}
    result["predictions"] = predictions
    return result


def main():
    import copy
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmarks/rsg4_cost_profile_20260930/rsg4.yaml")
    parser.add_argument("--configs", nargs="+", help="paired suite; first config is reference, maps retained only in RAM")
    parser.add_argument("--data-root", default="/home/hun/Volume/DATA/mvtec_ad_2")
    parser.add_argument("--object", default="fabric")
    parser.add_argument("--queries", type=int, default=12)
    parser.add_argument("--output", default="results/local_runs/rsg4_component_cost_20261001/profile.json")
    args = parser.parse_args()
    if args.queries < 2:
        raise ValueError("--queries must be >=2")
    reference = None
    for config in args.configs or [args.config]:
        current = copy.copy(args)
        current.config = config
        current.output = str(Path(args.output) / Path(config).stem / "profile.json") if args.configs else args.output
        result = profile(current, reference)
        if reference is None:
            reference = result
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
