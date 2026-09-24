"""Evaluation-only QR1 interventions; no labels or paired data enter estimation."""
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from fmad.evaluation.shift_diagnostic import acquisition_identity
from fmad.evaluation.streaming import StreamingSegmentationAccumulator
from fmad.methods.PSMA_MR import _read_rgb
from fmad.methods.query_reference import controlled_translation, fit_translation
from fmad.methods.reference_matching import _distance_chunks


ARMS = ("baseline", "fitted", "random", "opposite", "fixed", "baseline5", "fitted5")


def cosine(a, b):
    denominator = a.norm() * b.norm()
    return float(torch.dot(a, b) / denominator) if float(denominator) > 1e-12 else None


@torch.no_grad()
def token_evidence(query, reference, *, chunk, ridge, max_norm, min_tokens, seed):
    """Fit from one unlabeled query; masks and paired queries are not accepted."""
    baseline, indices = _distance_chunks(query, reference, chunk, return_indices=True)
    indices = indices[:, 0]
    delta, stats = fit_translation(query - reference[indices], keep_fraction=1.0,
                                   ridge=ridge, max_norm=max_norm, min_tokens=min_tokens)
    scores = {"baseline": baseline,
              "baseline5": _distance_chunks(query, reference, chunk, k=5)}
    deltas = {}
    for arm in ("fitted", "random", "opposite"):
        d = controlled_translation(delta, arm, seed)
        shifted = reference if stats["fallback"] else F.normalize(reference + d, dim=-1)
        scores[arm] = _distance_chunks(query, shifted, chunk)
        deltas[arm] = d
        if arm == "fitted":
            scores["fixed"] = 1.0 - (query * shifted[indices]).sum(-1)
            scores["fitted5"] = _distance_chunks(query, shifted, chunk, k=5)
    return scores, {"query": query, "indices": indices, "deltas": deltas,
                    "fallback": stats["fallback"]}


@torch.no_grad()
def image_evidence(method, path, *, retain_trace=False):
    image = _read_rgb(path)
    totals = {}
    traces = {}
    for resolution in method.resolutions:
        layers, grid = method._extract(image, resolution)
        layer_scores = {arm: [] for arm in ARMS}
        for index, (query, reference) in enumerate(zip(layers, method.banks[resolution]["layers"])):
            scores, trace = token_evidence(
                query, reference, chunk=method.chunk, ridge=method.ridge,
                max_norm=method.max_norm, min_tokens=method.min_tokens,
                seed=20260920 + index * 1000003 + grid[0] * 1009 + grid[1])
            scale = method.scales[resolution][index].to(query.device)
            for arm in ARMS:
                layer_scores[arm].append(scores[arm] / scale)
            if retain_trace:
                traces[(resolution, index)] = trace
        for arm in ARMS:
            score = torch.stack(layer_scores[arm]).mean(0)
            native = F.interpolate(score.reshape(1, 1, *grid), size=image.shape[:2],
                                   mode="bilinear", align_corners=False)[0, 0]
            totals[arm] = native if arm not in totals else totals[arm] + native
    maps = {arm: (value / len(method.resolutions)).cpu().numpy()
            for arm, value in totals.items()}
    if any(not np.isfinite(value).all() for value in maps.values()):
        raise ValueError("nonfinite mechanism scores")
    return maps, traces


@torch.no_grad()
def paired_alignment(clean, shifted, reference):
    """Normal paired evaluator only; fixed regular-query correspondences."""
    q0, q1 = clean["query"], shifted["query"]
    if q0.shape != q1.shape:
        raise ValueError("paired feature shape mismatch")
    displacement = q1 - q0
    mean = displacement.mean(0)
    indices = clean["indices"]
    result = {"mean_shift_norm": float(mean.norm()),
              "baseline_residual_displacement_mse": float(displacement.square().sum(-1).mean())}
    for arm in ("fitted", "random", "opposite"):
        d0, d1 = clean["deltas"][arm], shifted["deltas"][arm]
        b0 = reference if clean["fallback"] else F.normalize(reference + d0, dim=-1)
        b1 = reference if shifted["fallback"] else F.normalize(reference + d1, dim=-1)
        actual = b1 - b0
        differential = d1 - d0
        residual_change = displacement - actual[indices]
        result[arm] = {
            "differential_cosine": cosine(differential, mean),
            "shift_delta_cosine": cosine(d1, mean),
            "actual_bank_cosine": cosine(actual.mean(0), mean),
            "differential_norm": float(differential.norm()),
            "projection_coefficient": (float(torch.dot(differential, mean) / mean.square().sum())
                                       if float(mean.norm()) > 1e-12 else None),
            "residual_displacement_mse": float(residual_change.square().sum(-1).mean()),
        }
    return result


def score_summary(score, mask):
    normal, defect = score[~mask], score[mask]
    normal_mean = float(normal.mean()) if normal.size else None
    defect_mean = float(defect.mean()) if defect.size else None
    return {"normal_mean": normal_mean, "defect_mean": defect_mean,
            "margin": (defect_mean - normal_mean
                       if normal_mean is not None and defect_mean is not None else None)}


def run_feature_mechanism(method, dataset, object_name, queries, mask_loader):
    # Fail closed for unsupported production settings rather than silently
    # auditing a different scorer. This hook never changes method state.
    if (not method.query_adaptation or method.robust_trim or method.spatial_crossfit
            or method.translation_control != "fitted" or method.bank_bags != 1
            or method.variant != "R1" or method.score_normalization != "mad"
            or method.layer_fusion != "mean" or method.scale_fusion != "mean"
            or method.local_reliability != "off" or method.prediction_postprocess != "none"):
        raise ValueError("feature mechanism requires original untrimmed QR1/MAD/mean")
    pairs = {}
    for kind, path in queries:
        scene, condition = acquisition_identity(path)
        if condition not in {"regular", "shift_1"}:
            continue
        pair = pairs.setdefault((kind, scene), {})
        if condition in pair:
            raise ValueError("duplicate acquisition scene")
        pair[condition] = path
    if not pairs or any(set(pair) != {"regular", "shift_1"} for pair in pairs.values()):
        raise ValueError("incomplete regular/shift_1 pairs")
    accumulators = {(condition, arm): StreamingSegmentationAccumulator(max_fpr=.05)
                    for condition in ("regular", "shift_1") for arm in ARMS}
    records, alignments = [], []
    max_parity_error = 0.0
    for pair_number, ((kind, scene), paths) in enumerate(sorted(pairs.items())):
        pair_traces = {}
        for condition in ("regular", "shift_1"):
            path = paths[condition]
            maps, traces = image_evidence(method, path, retain_trace=kind == "good")
            # One paired scene/object is independently replayed through the
            # production scorer. Do not accept an approximate replacement.
            if pair_number == 0:
                production = np.asarray(method.predict(path)[0])
                error = float(np.max(np.abs(production - maps["fitted"])))
                max_parity_error = max(max_parity_error, error)
                if not np.allclose(production, maps["fitted"], atol=1e-6, rtol=1e-6):
                    raise ValueError(f"QR1 mechanism replay mismatch: {error}")
            mask = mask_loader(dataset, object_name, kind, path, maps["baseline"].shape)
            record = {"query": f"{kind}/{Path(path).name}", "scene_id": scene,
                      "normal_image": kind == "good", "condition": condition, "arms": {}}
            for arm, score in maps.items():
                accumulators[(condition, arm)].update([score], [mask])
                record["arms"][arm] = score_summary(score, mask)
            records.append(record)
            if kind == "good":
                pair_traces[condition] = traces
        if kind == "good":
            for (resolution, index), clean in pair_traces["regular"].items():
                result = paired_alignment(clean, pair_traces["shift_1"][(resolution, index)],
                                          method.banks[resolution]["layers"][index])
                alignments.append({"scene_id": scene, "resolution": resolution,
                                   "layer": method.layers[index], **result})
        print(f"    mechanism {object_name}: pair {pair_number + 1}/{len(pairs)}", flush=True)
    metrics = {}
    for (condition, arm), accumulator in accumulators.items():
        value = accumulator.compute()
        metrics.setdefault(condition, {})[arm] = {
            "seg_AUPRO_0.05": value.seg_aupro_0_05,
            "seg_AUROC_0.05": value.seg_auroc_0_05,
            "prediction_map_count": value.prediction_map_count}
    return {"scope": "posthoc_public_seen_development_regular_shift1",
            "interpretation": "reference_relative_residuals_not_changed_backbone_features",
            "query_features_changed": False, "labels_used_for_estimation": False,
            "scoring_arms": list(ARMS), "alternative_readout": "5NN_same_original_1NN_support_MAD",
            "max_production_replay_absolute_error": max_parity_error,
            "condition_metrics": metrics, "records": records, "normal_pair_alignment": alignments}
