#!/usr/bin/env python3
"""Direct, post-hoc test of high-score overlap using cached AD2 maps.

Labels are used only to form diagnostic background/defect pairs after all
scores have been computed. No fitting, prediction, CQ, or noise guard occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F


CLASSES = ("can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts")
SHOTS = (2, 16)
SEEDS = (0, 1, 2)
RESOLUTIONS = (448, 672, 896, 1120)
CALIPERS = (0.01, 0.02, 0.05)
PRIMARY_CALIPER = 0.02
NORMAL_TAIL_FRACTION = 0.05
EPS = 1e-8
SOURCE_ROOTS = (
    Path("/workspace/results_local_compatible/scopead_mechanism_20261020"),
    Path("/var/tmp/fsad-analysis/scopead_mechanism_20261020"),
)


def condition_from_path(image_path: str) -> str:
    stem = Path(image_path).stem
    if "_" not in stem:
        raise ValueError(f"cannot parse AD2 condition from filename: {image_path}")
    return stem.split("_", 1)[1]


def match_sorted(normal_values: np.ndarray, defect_values: np.ndarray, tolerance: float):
    """Maximum-cardinality, order-preserving one-to-one matching in 1-D."""
    i = j = 0
    normal_match = []
    defect_match = []
    n, m = len(normal_values), len(defect_values)
    while i < n and j < m:
        difference = defect_values[j] - normal_values[i]
        if abs(difference) <= tolerance:
            normal_match.append(i)
            defect_match.append(j)
            i += 1
            j += 1
        elif difference > tolerance:
            i += 1
        else:
            j += 1
    return np.asarray(normal_match, dtype=np.int64), np.asarray(defect_match, dtype=np.int64)


def self_check_matcher() -> None:
    """Compare the greedy matcher with dynamic-programming optima on tiny cases."""
    def optimum(a, b, tol):
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                value = max(dp[i - 1][j], dp[i][j - 1])
                if abs(a[i - 1] - b[j - 1]) <= tol:
                    value = max(value, dp[i - 1][j - 1] + 1)
                dp[i][j] = value
        return dp[-1][-1]

    sequences = [tuple(x) for n in range(5) for x in itertools.combinations_with_replacement(range(5), n)]
    for a in sequences:
        for b in sequences:
            got = len(match_sorted(np.asarray(a), np.asarray(b), 1)[0])
            if got != optimum(a, b, 1):
                raise AssertionError(f"matcher mismatch for {a}, {b}: {got} != {optimum(a, b, 1)}")


def load_sources():
    sources = {}
    for root in SOURCE_ROOTS:
        if not root.is_dir():
            continue
        for index_path in sorted(root.rglob("mechanism_cache_index.json")):
            info = json.loads(index_path.read_text())
            config = info.get("config", {})
            key = (int(config.get("protocol", {}).get("shots", -1)), int(config.get("seed", -1)), info.get("object"))
            if key in sources:
                raise ValueError(f"duplicate source identity {key}: {sources[key]['index_path']} and {index_path}")
            if config.get("dataset") != "mvtec_ad2" or config.get("method") != "multiscale_aggregation_reference_matching":
                raise ValueError(f"unexpected source configuration: {index_path}")
            method = config.get("method_config", {})
            if (method.get("statistical_normalization") != "q95"
                    or method.get("feature_layers") != [5, 11, 23]
                    or method.get("resolutions") != list(RESOLUTIONS)
                    or method.get("layer_fusion_gamma") != 0.125):
                raise ValueError(f"source is not the fixed Q95 ScopeAD setting: {index_path}")
            if info.get("label_use") != "AD2 public seen development GT, only after prediction, post-hoc mechanism analysis":
                raise ValueError(f"unexpected source label provenance: {index_path}")
            for image, record in info.get("records", {}).items():
                error = float(record.get("replay_max_abs_error", math.inf))
                if not math.isfinite(error) or error > 2e-5:
                    raise ValueError(f"source replay parity failed: {index_path} {image}")
            sources[key] = dict(index_path=index_path, info=info,
                                 index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest())
    expected = {(shot, seed, name) for shot in SHOTS for seed in SEEDS for name in CLASSES}
    if set(sources) != expected:
        missing = sorted(expected - set(sources))
        extra = sorted(set(sources) - expected)
        raise ValueError(f"source identity coverage mismatch; missing={missing}, extra={extra}")
    return sorted(sources.items())


def resolution_maps(item, device):
    """Replay cached layer scores with the same layer/bag/interpolation order."""
    maps = []
    for resolution in item["resolutions"]:
        values = torch.as_tensor(item[f"q95_{int(resolution)}"], device=device)
        bags = [torch.stack([value[1], value[2]]).mean(0) + 0.125 * value[0] for value in values]
        score = torch.stack(bags).mean(0)
        output_shape = tuple(int(value) for value in item["shape"])
        maps.append(F.interpolate(score[None, None], size=output_shape,
                                  mode="bilinear", align_corners=False)[0, 0])
    return maps


def combine_mean(maps):
    return torch.stack(maps, dim=0).mean(dim=0)


def combine_renyi(maps):
    stacked = torch.stack(maps, dim=0)
    if stacked.shape[0] != 4:
        raise ValueError(f"Rényi replay requires four resolutions, got {stacked.shape[0]}")
    ordered = torch.sort(stacked, dim=0, descending=False).values
    weights = torch.tensor([1 / 16, 7 / 48, 13 / 48, 25 / 48],
                           device=stacked.device, dtype=stacked.dtype)[:, None, None]
    return (ordered * weights).sum(dim=0)


def log_values(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("score maps must be finite and nonnegative")
    return np.log(np.maximum(values, EPS))


def pair_statistics(delta, baseline_delta):
    if len(delta) == 0:
        return dict(concordance=None, mean_log_margin=None, median_log_margin=None,
                    delta_concordance_vs_1120=None, delta_margin_vs_1120=None,
                    rescue_rate=None, regression_rate=None)
    ties = np.isclose(delta, 0.0, atol=1e-10, rtol=0.0)
    concordance = float(np.mean(np.where(ties, 0.5, (delta > 0).astype(float))))
    baseline_ties = np.isclose(baseline_delta, 0.0, atol=1e-10, rtol=0.0)
    baseline_correct_value = np.where(baseline_ties, 0.5, (baseline_delta > 0).astype(float))
    current_correct_value = np.where(ties, 0.5, (delta > 0).astype(float))
    return dict(
        concordance=concordance,
        mean_log_margin=float(np.mean(delta)),
        median_log_margin=float(np.median(delta)),
        delta_concordance_vs_1120=float(np.mean(current_correct_value - baseline_correct_value)),
        delta_margin_vs_1120=float(np.mean(delta - baseline_delta)),
        rescue_rate=float(np.mean((baseline_correct_value == 0) & (current_correct_value == 1))),
        regression_rate=float(np.mean((baseline_correct_value == 1) & (current_correct_value == 0))),
    )


def process_image(dataset, data_root, name, shots, seed, image_path, record, cache_dir, device):
    if Path(image_path).parent.name != "bad":
        raise ValueError(f"expected bad query image: {image_path}")
    cache_file = cache_dir / record["file"]
    with np.load(cache_file, allow_pickle=False) as archive:
        item = {key: archive[key] for key in archive.files}
    resolutions = tuple(int(value) for value in item["resolutions"])
    if resolutions != RESOLUTIONS:
        raise ValueError(f"unexpected resolutions in {cache_file}: {resolutions}")
    maps = resolution_maps(item, device)
    if len(maps) != 4 or any(tuple(score.shape) != tuple(item["shape"]) for score in maps):
        raise ValueError(f"invalid replay map geometry for {cache_file}")
    map_arrays = {str(resolution): score.detach().cpu().numpy().reshape(-1)
                  for resolution, score in zip(resolutions, maps)}
    map_arrays["4scale_mean"] = combine_mean(maps).detach().cpu().numpy().reshape(-1)
    map_arrays["4scale_renyi"] = combine_renyi(maps).detach().cpu().numpy().reshape(-1)
    map_logs = {key: log_values(value) for key, value in map_arrays.items()}
    baseline = map_logs["1120"]

    from fmad.pipeline import _ground_truth_mask
    mask = _ground_truth_mask(dataset, name, Path(image_path).parent.name,
                              image_path, tuple(item["shape"]))
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    defect_ids = np.flatnonzero(mask)
    background_ids = np.flatnonzero(~mask)
    if not len(defect_ids) or not len(background_ids):
        raise ValueError(f"empty defect/background region in {image_path}")

    normal_count = max(1, int(math.ceil(NORMAL_TAIL_FRACTION * len(background_ids))))
    normal_order = np.lexsort((background_ids, -baseline[background_ids]))
    normal_ids = background_ids[normal_order[:normal_count]]
    normal_order = np.lexsort((normal_ids, baseline[normal_ids]))
    defect_order = np.lexsort((defect_ids, baseline[defect_ids]))
    normal_sorted_ids = normal_ids[normal_order]
    defect_sorted_ids = defect_ids[defect_order]
    normal_sorted_values = baseline[normal_sorted_ids]
    defect_sorted_values = baseline[defect_sorted_ids]

    condition = condition_from_path(image_path)
    row = dict(
        shots=shots, seed=seed, object=name,
        image=str(Path(image_path).relative_to(Path(data_root))),
        condition=condition,
        height=int(item["shape"][0]), width=int(item["shape"][1]),
        defect_pixels=int(len(defect_ids)), background_pixels=int(len(background_ids)),
        normal_tail_fraction=NORMAL_TAIL_FRACTION,
        normal_tail_candidates=int(normal_count),
        calipers={},
    )
    arm_names = ["448", "672", "896", "1120", "4scale_mean", "4scale_renyi"]
    for tolerance in CALIPERS:
        left = int(np.searchsorted(defect_sorted_values,
                                   normal_sorted_values[0] - tolerance, side="left"))
        right = int(np.searchsorted(defect_sorted_values,
                                    normal_sorted_values[-1] + tolerance, side="right"))
        near_defect_values = defect_sorted_values[left:right]
        n_match, d_match = match_sorted(normal_sorted_values, near_defect_values, tolerance)
        normal_pair_ids = normal_sorted_ids[n_match]
        defect_pair_ids = defect_sorted_ids[left + d_match]
        base_delta = baseline[defect_pair_ids] - baseline[normal_pair_ids]
        arm_results = {}
        for arm in arm_names:
            delta = map_logs[arm][defect_pair_ids] - map_logs[arm][normal_pair_ids]
            arm_results[arm] = pair_statistics(delta, base_delta)
        row["calipers"][f"{tolerance:.2f}"] = dict(
            pair_count=int(len(normal_pair_ids)),
            defect_coverage=float(len(defect_pair_ids) / len(defect_ids)),
            mean_abs_1120_log_gap=float(np.mean(np.abs(base_delta))) if len(base_delta) else None,
            median_abs_1120_log_gap=float(np.median(np.abs(base_delta))) if len(base_delta) else None,
            max_abs_1120_log_gap=float(np.max(np.abs(base_delta))) if len(base_delta) else None,
            arms=arm_results,
        )
    return row


def analyze(args):
    self_check_matcher()
    from fmad.cli import load_config as _load_config  # noqa: F401
    from fmad.methods import register_methods
    from fmad.registry import DATASET_REGISTRY
    register_methods("multiscale_aggregation_reference_matching")

    sources = load_sources()
    if not (0 <= args.shard < args.shards):
        raise ValueError("shard must be in [0, shards)")
    selected = [entry for index, entry in enumerate(sources) if index % args.shards == args.shard]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"partial_{args.shard:02d}.json"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing shard: {target}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(4)
    rows = []
    source_rows = []
    for (shots, seed, name), source in selected:
        info = source["info"]
        config = info["config"]
        data_root = config["data_root"]
        dataset = DATASET_REGISTRY.get(config["dataset"])(
            data_root=data_root, config=config.get("dataset_config", {}))
        cache_dir = source["index_path"].parent / "mechanism_cache"
        bad_records = [(path, record) for path, record in sorted(info["records"].items())
                       if Path(path).parent.name == "bad"]
        if not bad_records:
            raise ValueError(f"no bad query records in {source['index_path']}")
        conditions = {}
        for image_path, record in bad_records:
            row = process_image(dataset, data_root, name, shots, seed,
                                image_path, record, cache_dir, device)
            rows.append(row)
            conditions[row["condition"]] = conditions.get(row["condition"], 0) + 1
            if len(rows) % 100 == 0:
                print(f"HSA_REPLAY shard={args.shard} images={len(rows)}", flush=True)
        source_rows.append(dict(shots=shots, seed=seed, object=name,
            index_path=str(source["index_path"]), index_sha256=source["index_sha256"],
            image_count=len(bad_records), bad_images_by_condition=conditions))
        print(f"HSA_IDENTITY_COMPLETE shard={args.shard} shot={shots} seed={seed} class={name} bad_images={len(bad_records)}",
              flush=True)

    release_path = Path("/workspace/.local-sync/current/.fmad-source-sha256")
    release = release_path.read_text().strip() if release_path.exists() else "unknown"
    result = dict(
        schema="scopead_hsa_overlap_v1", shard=args.shard, shards=args.shards,
        created_unix=time.time(), source_release_sha256=release,
        source_scope="MVTec AD 2 test_public; previously seen development population",
        label_use="Ground truth is used after cached scoring only to select bad-image defect and background pixels; it is not used for fitting or prediction.",
        replay="Cached Q95 per-layer/per-bag score maps; layer fusion, bag averaging, and bilinear upsampling replayed before label access.",
        postprocessing="CQ omitted; noise guard omitted; all four cached resolutions are replayed directly.",
        matching=dict(primary_caliper_log_score=PRIMARY_CALIPER,
            sensitivity_calipers_log_score=list(CALIPERS),
            baseline="1120-resolution Q95 score",
            normal_candidates="exact highest-scoring 5% of background pixels in each bad image, selected by stable descending score then row-major index",
            matching="maximum-cardinality order-preserving one-to-one matching between selected background pixels and all defect pixels, with absolute natural-log score difference <= caliper",
            score_transform="natural logarithm after clipping nonnegative score to 1e-8",
            interpretation="paired concordance is 1 for defect score > background score, 0.5 for a tie, and 0 otherwise; pairs are within-image diagnostic comparisons, not independent statistical samples."),
        conditions="regular; illumination (overexposed/underexposed); acquisition shift (shift_1 through shift_4); nonregular is the union of the latter two, with unavailable class-condition cells retained as missing.",
        data_root=selected[0][1]["info"]["config"]["data_root"],
        source_indices=source_rows, image_count=len(rows), images=rows,
    )
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    temp.replace(target)
    print(f"HSA_SHARD_COMPLETE shard={args.shard} images={len(rows)} output={target}", flush=True)


def condition_groups(condition):
    return {
        "all": True,
        "regular": condition == "regular",
        "illumination": condition in {"overexposed", "underexposed"},
        "acquisition_shift": condition.startswith("shift_"),
        "nonregular": condition != "regular",
    }


def class_bootstrap(values, seed, draws=10000):
    items = np.asarray(list(values.values()), dtype=np.float64)
    if not len(items):
        return None
    if len(items) == 1:
        return [float(items[0]), float(items[0])]
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(items), size=(draws, len(items)))
    means = items[index].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def summarize_cells(cells, metric_name, seed):
    class_seed = {key: float(np.mean(value)) for key, value in cells.items()
                  if value is not None and np.asarray(value).size > 0}
    grouped_classes = {}
    for (object_name, _support_seed), value in class_seed.items():
        grouped_classes.setdefault(object_name, []).append(value)
    class_values = {name: float(np.mean(values)) for name, values in grouped_classes.items()}
    seed_values = {}
    for support_seed in SEEDS:
        values = [value for (_object, cell_seed), value in class_seed.items()
                  if cell_seed == support_seed]
        if values:
            seed_values[str(support_seed)] = float(np.mean(values))
    seed_array = np.asarray(list(seed_values.values()), dtype=np.float64)
    return dict(
        metric=metric_name,
        mean_class_macro=float(np.mean(list(class_values.values()))) if class_values else None,
        class_cluster_bootstrap_95=class_bootstrap(class_values, seed) if class_values else None,
        class_values=class_values,
        seed_macro_values=seed_values,
        seed_macro_mean=float(seed_array.mean()) if len(seed_array) else None,
        seed_macro_sd=float(seed_array.std(ddof=1)) if len(seed_array) > 1 else None,
        class_count=len(class_values), class_seed_cells=len(class_seed),
    )


def self_check_statistics():
    delta = np.asarray([-1., -5e-11, 0., 5e-11, 1.])
    stats = pair_statistics(delta, np.zeros_like(delta))
    if stats["concordance"] != 0.5 or stats["rescue_rate"] != 0 or stats["regression_rate"] != 0:
        raise AssertionError(f"inconsistent tie handling: {stats}")
    zeros = summarize_cells({("a", 0): 0., ("b", 0): 1.}, "coverage", 1)
    if zeros["class_count"] != 2 or zeros["mean_class_macro"] != 0.5:
        raise AssertionError("zero-valued coverage cells must remain in the macro")
    strict = pair_statistics(np.asarray([1., -1.]), np.asarray([-1., 1.]))
    if strict["rescue_rate"] != 0.5 or strict["regression_rate"] != 0.5:
        raise AssertionError("strict rescue/regression transitions are incorrect")


def aggregate_metric(rows, tolerance, group_name, seed_offset):
    selected = [row for row in rows if condition_groups(row["condition"]).get(group_name, False)]
    methods = ("448", "672", "896", "1120", "4scale_mean", "4scale_renyi")
    by_method = {method: {} for method in methods}
    delta_vs_1120 = {method: {} for method in methods if method != "1120"}
    delta_renyi_vs_mean = {}
    coverage = {}
    diagnostic_names = ("mean_log_margin", "median_log_margin", "delta_margin_vs_1120",
                        "rescue_rate", "regression_rate")
    diagnostics = {method: {metric: {} for metric in diagnostic_names} for method in methods}
    matching_gaps = {}
    total_pair_count = 0
    image_rows_with_pairs = 0
    for row in selected:
        caliper = row["calipers"][f"{tolerance:.2f}"]
        total_pair_count += caliper["pair_count"]
        image_rows_with_pairs += int(caliper["pair_count"] > 0)
        key = (row["object"], row["seed"])
        coverage.setdefault(key, []).append(caliper["defect_coverage"])
        gap = caliper.get("mean_abs_1120_log_gap")
        if gap is not None:
            matching_gaps.setdefault(key, []).append(gap)
        base = caliper["arms"]["1120"].get("concordance")
        for method in methods:
            for metric in diagnostic_names:
                diagnostic = caliper["arms"][method].get(metric)
                if diagnostic is not None:
                    diagnostics[method][metric].setdefault(key, []).append(diagnostic)
            value = caliper["arms"][method].get("concordance")
            if value is None:
                continue
            by_method[method].setdefault(key, []).append(value)
            if method != "1120" and base is not None:
                delta_vs_1120[method].setdefault(key, []).append(value - base)
        renyi = caliper["arms"]["4scale_renyi"].get("concordance")
        mean = caliper["arms"]["4scale_mean"].get("concordance")
        if renyi is not None and mean is not None:
            delta_renyi_vs_mean.setdefault(key, []).append(renyi - mean)
    coverage_cells = {key: float(np.mean(value)) for key, value in coverage.items()}
    coverage_summary = summarize_cells(coverage_cells, "defect_coverage", seed_offset + 1)
    return dict(
        group=group_name, caliper_log_score=tolerance,
        image_count=len(selected), image_rows_with_pairs=image_rows_with_pairs,
        total_matched_pairs=int(total_pair_count),
        mean_image_defect_coverage=float(np.mean([x for values in coverage.values() for x in values])) if coverage else None,
        class_macro_defect_coverage=coverage_summary,
        baseline_mean_abs_log_gap=summarize_cells(matching_gaps, "mean_abs_log_gap", seed_offset + 2),
        pair_diagnostics={method: {
            metric: summarize_cells(cells, metric, seed_offset + 100 + i * 10 + j)
            for j, (metric, cells) in enumerate(metrics.items())}
            for i, (method, metrics) in enumerate(diagnostics.items())},
        methods={method: summarize_cells(cells, "concordance", seed_offset + i + 10)
                 for i, (method, cells) in enumerate(by_method.items())},
        paired_deltas=dict(
            vs_1120={method: summarize_cells(cells, "concordance_delta", seed_offset + i + 30)
                     for i, (method, cells) in enumerate(delta_vs_1120.items())},
            renyi_vs_mean=summarize_cells(delta_renyi_vs_mean, "concordance_delta", seed_offset + 50),
        ),
    )


def summarize(args):
    output_dir = Path(args.output_dir)
    input_dir = Path(args.input_dir) if args.input_dir else output_dir
    partials = sorted(input_dir.glob("partial_*.json"))
    if len(partials) != args.shards:
        raise ValueError(f"expected {args.shards} shard files, found {len(partials)}")
    payloads = [json.loads(path.read_text()) for path in partials]
    for payload in payloads:
        for row in payload["images"]:
            for caliper in row["calipers"].values():
                base = caliper["arms"]["1120"]["concordance"]
                for value in caliper["arms"].values():
                    score = value["concordance"]
                    if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
                        raise ValueError("invalid stored concordance; replay pixel scores before summarizing")
                    if score is not None and abs(score - base - value["delta_concordance_vs_1120"]) > 1e-12:
                        raise ValueError("stored tie calculation differs; replay pixel scores before summarizing")
    if sorted(item["shard"] for item in payloads) != list(range(args.shards)):
        raise ValueError("shard IDs are incomplete or duplicated")
    rows = [row for payload in payloads for row in payload["images"]]
    identities = set()
    for row in rows:
        identity = (row["shots"], row["seed"], row["object"], row["image"])
        if identity in identities:
            raise ValueError(f"duplicate image identity {identity}")
        identities.add(identity)
    identity_set = {(row["shots"], row["seed"], row["object"]) for row in rows}
    expected = {(shot, seed, name) for shot in SHOTS for seed in SEEDS for name in CLASSES}
    if identity_set != expected:
        raise ValueError("not all 48 class/support identities are represented")
    source_indices = [item for payload in payloads for item in payload["source_indices"]]
    if len(source_indices) != 48 or len({(x["shots"], x["seed"], x["object"]) for x in source_indices}) != 48:
        raise ValueError("source index provenance is incomplete or duplicated")
    observed_counts = {}
    for row in rows:
        key = (row["shots"], row["seed"], row["object"])
        observed_counts[key] = observed_counts.get(key, 0) + 1
    for source in source_indices:
        key = (source["shots"], source["seed"], source["object"])
        if observed_counts.get(key) != source["image_count"]:
            raise ValueError(f"source image count mismatch: {key}")
    groups = ("all", "regular", "illumination", "acquisition_shift", "nonregular")
    report = dict(
        schema="scopead_hsa_overlap_summary_v2",
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_shards=[dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                      for path in partials],
        statistics_status="Reaggregation of existing image statistics, not pixel replay. Legacy rescue/regression use exact signs; concordance uses a 1e-10 tie tolerance. Stored concordance consistency checked.",
        source="Existing 2026-10-20 Q95 mechanism caches; no new feature or nearest-neighbor computation.",
        source_release_sha256=sorted({item["source_release_sha256"] for item in payloads}),
        source_indices=source_indices,
        label_use="Post-hoc diagnostic on previously seen MVTec AD 2 test_public development labels; labels enter only after cached scores are produced.",
        inference_limit="Eight class clusters and three repeated support seeds; class bootstrap intervals are descriptive. Pixel pairs and repeated queries are not treated as independent samples.",
        primary_caliper_log_score=PRIMARY_CALIPER,
        image_count=len(rows), identities=len(identity_set),
        unique_query_images=len({(row["object"], row["image"]) for row in rows}),
        condition_counts={condition: sum(row["condition"] == condition for row in rows)
                          for condition in sorted({row["condition"] for row in rows})},
        summaries={},
    )
    for shots in SHOTS:
        report["summaries"][str(shots)] = {}
        shot_rows = [row for row in rows if row["shots"] == shots]
        for group_name in groups:
            report["summaries"][str(shots)][group_name] = {
                f"{tolerance:.2f}": aggregate_metric(
                    shot_rows, tolerance, group_name,
                    shots * 1000 + int(tolerance * 100) + len(group_name))
                for tolerance in CALIPERS
            }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "hsa_overlap_analysis.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# Direct high-score overlap analysis", "",
        "Post-hoc paired diagnostic using the existing MVTec AD 2 test_public Q95 map caches (8 classes, 2/16-shot, 3 support seeds). Ground truth is used only after scoring to form same-image pairs. CQ and noise guard are omitted. The primary comparison matches the exact top 5% of normal-background pixels under the 1120-resolution score to defect pixels within an absolute natural-log score gap of 0.02, without replacement.", "",
        "Pairwise concordance is 1 when the defect receives the higher score, 0.5 for a tie, and 0 otherwise. We first average image values within each class and support seed, then average classes equally. Bracketed intervals resample the eight class clusters; these intervals are descriptive, not an untouched-test confidence claim.", "",
        "## Primary matched-pair results", "",
        "| Shots | Condition | Matched images / images | Pairs | Defect coverage | 1120 only | 4-scale mean | 4-scale Rényi | Rényi − 1120 (pp) [class bootstrap 95%] | Rényi − mean (pp) [class bootstrap 95%] |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def fmt_pct(value):
        return "—" if value is None else f"{100 * value:.2f}%"
    def fmt_pp(value, interval=None):
        if value is None:
            return "—"
        text = f"{100 * value:+.2f}"
        return text if interval is None else text + f" [{100 * interval[0]:+.2f}, {100 * interval[1]:+.2f}]"
    main_groups = ("regular", "illumination", "acquisition_shift", "nonregular")
    for shots in SHOTS:
        for group_name in main_groups:
            item = report["summaries"][str(shots)][group_name][f"{PRIMARY_CALIPER:.2f}"]
            methods = item["methods"]
            paired = item["paired_deltas"]
            d_base = paired["vs_1120"]["4scale_renyi"]
            d_mean = paired["renyi_vs_mean"]
            cov = item["class_macro_defect_coverage"]["mean_class_macro"]
            lines.append(f"| {shots} | {group_name} | {item['image_rows_with_pairs']} / {item['image_count']} | {item['total_matched_pairs']:,} | {fmt_pct(cov)} | {fmt_pct(methods['1120']['mean_class_macro'])} | {fmt_pct(methods['4scale_mean']['mean_class_macro'])} | {fmt_pct(methods['4scale_renyi']['mean_class_macro'])} | {fmt_pp(d_base['mean_class_macro'], d_base['class_cluster_bootstrap_95'])} | {fmt_pp(d_mean['mean_class_macro'], d_mean['class_cluster_bootstrap_95'])} |")
    lines.extend(["", "## Resolution-specific diagnostics", "",
        "| Shots | Condition | 448 | 672 | 896 | 1120 | 4-scale mean | 4-scale Rényi |",
        "|---:|---|---:|---:|---:|---:|---:|---:|"])
    for shots in SHOTS:
        for group_name in main_groups:
            item = report["summaries"][str(shots)][group_name][f"{PRIMARY_CALIPER:.2f}"]
            vals = [fmt_pct(item["methods"][key]["mean_class_macro"])
                    for key in ("448", "672", "896", "1120", "4scale_mean", "4scale_renyi")]
            lines.append(f"| {shots} | {group_name} | " + " | ".join(vals) + " |")
    lines.extend(["", "## Caliper sensitivity", "",
        "| Shots | Condition | Log-score caliper | Pairs | 1120 only | 4-scale mean | 4-scale Rényi | Rényi − 1120 (pp) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|"])
    for shots in SHOTS:
        for group_name in ("regular", "nonregular"):
            for tolerance in CALIPERS:
                item = report["summaries"][str(shots)][group_name][f"{tolerance:.2f}"]
                delta = item["paired_deltas"]["vs_1120"]["4scale_renyi"]["mean_class_macro"]
                lines.append(f"| {shots} | {group_name} | {tolerance:.2f} | {item['total_matched_pairs']:,} | {fmt_pct(item['methods']['1120']['mean_class_macro'])} | {fmt_pct(item['methods']['4scale_mean']['mean_class_macro'])} | {fmt_pct(item['methods']['4scale_renyi']['mean_class_macro'])} | {fmt_pp(delta)} |")
    lines.extend(["", "## Matching and score-change audit", "",
        f"{report['unique_query_images']} unique bad query images are evaluated {len(rows)} times across support settings. Original pixel pairs and all original concordance results are retained. This is reaggregation, not a pixel-score replay. Legacy rescue/regression rates use exact signs; the revised implementation treats ties consistently for future replay.", "",
        "Signed margin is log(defect score) minus log(background score). Absolute baseline gap measures matching closeness. Rescue/regression are unconditional fractions of all matched pairs, not rates conditional on baseline errors/correct pairs. Image medians are averaged; this is not a pooled pixel median.", "",
        "| Shots | Condition | Baseline signed gap | Baseline absolute gap | Mean signed gap | Rényi signed gap | Rényi rescue | Rényi regression |",
        "|---:|---|---:|---:|---:|---:|---:|---:|"])
    for shots in SHOTS:
        for group_name in main_groups:
            item = report["summaries"][str(shots)][group_name][f"{PRIMARY_CALIPER:.2f}"]
            stats = item["pair_diagnostics"]
            signed = [stats[method]["mean_log_margin"]["mean_class_macro"]
                      for method in ("1120", "4scale_mean", "4scale_renyi")]
            absolute = item["baseline_mean_abs_log_gap"]["mean_class_macro"]
            rescue = stats["4scale_renyi"]["rescue_rate"]["mean_class_macro"]
            regression = stats["4scale_renyi"]["regression_rate"]["mean_class_macro"]
            lines.append(f"| {shots} | {group_name} | {signed[0]:+.6f} | {absolute:.6f} | {signed[1]:+.6f} | {signed[2]:+.6f} | {fmt_pct(rescue)} | {fmt_pct(regression)} |")
    lines.extend(["", "## Interpretation limits", "",
        "Maximum-cardinality matching does not minimize score gaps or balance their signs. Candidate counts and score densities can change baseline concordance even for uniform score populations. Baseline-selected pairs are a conditional diagnostic; a negative difference cannot by itself establish a population-wide HSA deterioration. All arms use Q95 and the same layer fusion, so this comparison cannot isolate calibration or layer expansion effects.", "",
        "Pairs are selected using defect labels and the baseline map, so this diagnostic explains score ordering on already-seen examples; it does not estimate deployment error, calibrate a threshold, or establish generalization to unseen acquisition shifts. Missing class-condition combinations remain absent (for example, not every class has every shift condition). Read class-level values in hsa_overlap_analysis.json with the macro averages.", ""])
    (output_dir / "comparison.md").write_text("\n".join(lines))
    print(f"HSA_SUMMARY_COMPLETE images={len(rows)} identities={len(identity_set)} output={report_path}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--output-dir", required=True)
    analyze_parser.add_argument("--shard", type=int, required=True)
    analyze_parser.add_argument("--shards", type=int, required=True)
    analyze_parser.add_argument("--device", default="cuda:0")
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--output-dir", required=True)
    summary_parser.add_argument("--input-dir", help="Read existing shards without overwriting their summary")
    summary_parser.add_argument("--shards", type=int, default=4)
    check_parser = sub.add_parser("self-check")
    args = parser.parse_args()
    if args.command == "analyze":
        analyze(args)
    elif args.command == "summarize":
        summarize(args)
    else:
        self_check_matcher()
        self_check_statistics()
        print("HSA_MATCHER_AND_STATISTICS_SELF_CHECK_PASSED")


if __name__ == "__main__":
    main()
