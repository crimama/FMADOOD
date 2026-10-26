#!/usr/bin/env python3
"""Evaluation-only rank diagnostic for order aggregation across resolutions.

The shared pipeline fits and predicts exactly as in a normal run. Just before
resolution aggregation, the per-resolution native score maps are captured.
Only after the method has returned its prediction is the GT mask read, and only
for this diagnostic: each pixel's rank position per resolution is counted for
normal and defect pixels. Labels never enter fitting or prediction.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from fmad.cli import parse_args, load_config, resolve_runtime_device
from fmad.registry import BACKBONE_REGISTRY, DATASET_REGISTRY, METHOD_REGISTRY
from fmad.methods import register_methods
from fmad.pipeline import run_pipeline, _ground_truth_mask
from fmad.protocol import seed_generators

SIZE_BINS = (0.001, 0.01)  # defect region area / image area: small | medium | large
MAX_QUANTILE_SAMPLES = 1_000_000


def renyi_weights_top_first(r_count):
    spacings = [sum(1.0 / (r_count - j) for j in range(k)) for k in range(1, r_count + 1)]
    weights = np.asarray(spacings) / sum(spacings)  # ascending order statistics
    return weights[::-1].copy()  # index 0 = highest-ranked map


def rank_positions(maps):
    """[R, H, W] -> [R, H, W] rank of each map per pixel; 0 is the highest score."""
    return (maps.unsqueeze(0) > maps.unsqueeze(1)).sum(dim=1)


def image_statistics(maps, mask):
    """Per-image rank counts. The mask is used only after prediction."""
    r_count = maps.shape[0]
    rank = rank_positions(maps)
    counts = {}
    for kind, selected in (("normal", ~mask), ("defect", mask)):
        table = np.zeros((r_count, r_count), dtype=np.int64)
        if bool(selected.any()):
            for r in range(r_count):
                table[r] = torch.bincount(rank[r][selected], minlength=r_count)[:r_count].cpu().numpy()
        counts[kind] = table
    regions = dict(size_top1=np.zeros((3, r_count), dtype=np.int64), size_regions=np.zeros(3, dtype=np.int64),
                   agree_pixels=0, region_pixels=0, region_count=0,
                   evidence_best=np.zeros(r_count, dtype=np.int64))
    if bool(mask.any()) and bool((~mask).any()):
        top1 = maps.argmax(dim=0)
        normal_values = maps[:, ~mask]
        stride = max(1, normal_values.shape[1] // MAX_QUANTILE_SAMPLES)
        reference = torch.quantile(normal_values[:, ::stride].float(), 0.95, dim=1).clamp_min(1e-12)
        count, labels = cv2.connectedComponents(mask.cpu().numpy().astype(np.uint8), connectivity=8)
        labels = torch.from_numpy(labels).to(maps.device)
        total = mask.numel()
        for k in range(1, count):
            region = labels == k
            pixels = int(region.sum())
            # Scale-free evidence: region mean relative to the same image's normal Q95.
            best = int((maps[:, region].mean(dim=1) / reference).argmax())
            fraction = pixels / total
            size = 0 if fraction < SIZE_BINS[0] else (1 if fraction < SIZE_BINS[1] else 2)
            regions["size_top1"][size] += torch.bincount(top1[region], minlength=r_count)[:r_count].cpu().numpy()
            regions["size_regions"][size] += 1
            regions["agree_pixels"] += int((top1[region] == best).sum())
            regions["region_pixels"] += pixels
            regions["region_count"] += 1
            regions["evidence_best"][best] += 1
    return counts, regions


def summarize(acc, resolutions):
    weights = renyi_weights_top_first(len(resolutions))
    out = {}
    for kind in ("normal", "defect"):
        table = acc[kind].astype(np.float64)
        total = table.sum(axis=1, keepdims=True)
        share = np.divide(table, total, out=np.zeros_like(table), where=total > 0)
        out[kind + "_top1_share"] = share[:, 0].tolist()
        out[kind + "_effective_renyi_weight"] = (share @ weights).tolist()
        out[kind + "_rank_share"] = share.tolist()
    normal_top1 = np.asarray(out["normal_top1_share"])
    defect_top1 = np.asarray(out["defect_top1_share"])
    out["top1_total_variation_defect_vs_normal"] = float(0.5 * np.abs(defect_top1 - normal_top1).sum())
    out["normal_top1_dominance"] = float(normal_top1.max())
    size = acc["size_top1"].astype(np.float64)
    size_total = size.sum(axis=1, keepdims=True)
    out["size_bins"] = ["small(<0.1%)", "medium(0.1-1%)", "large(>=1%)"]
    out["size_top1_share"] = np.divide(size, size_total, out=np.zeros_like(size), where=size_total > 0).tolist()
    out["evidence_best_agreement"] = (acc["agree_pixels"] / acc["region_pixels"]) if acc["region_pixels"] else None
    return out


def main():
    config = load_config(parse_args())
    options = config.get("evaluation", {}).get("order_rank_diagnostic")
    if not isinstance(options, dict) or options.get("use_evaluation_ground_truth") is not True:
        raise ValueError("rank diagnostic requires explicit evaluation GT authorization in config")
    device = resolve_runtime_device(config["device"])
    seed_generators(config["seed"])
    register_methods(config["method"])
    dataset = DATASET_REGISTRY.get(config["dataset"])(
        data_root=config["data_root"], config=config.get("dataset_config", {}))
    objects = dataset.get_objects()
    if len(objects) != 1:
        raise ValueError("rank diagnostic processes exactly one class per runner cell")
    name = objects[0].name
    backbone = BACKBONE_REGISTRY.get(config["backbone"])(model_name=config["backbone"],
        device=device, **config.get("backbone_config", {}))
    method = METHOD_REGISTRY.get(config["method"])(backbone=backbone, config=config.get("method_config", {}))
    resolutions = [int(r) for r in config["method_config"]["resolutions"]]
    r_count = len(resolutions)
    if config["method_config"].get("multiscale_aggregation") in (None, "mean"):
        raise ValueError("rank diagnostic needs an order aggregation path")

    holder = {}
    acc = dict(normal=np.zeros((r_count, r_count), np.int64), defect=np.zeros((r_count, r_count), np.int64),
               size_top1=np.zeros((3, r_count), np.int64), size_regions=np.zeros(3, np.int64),
               agree_pixels=0, region_pixels=0, region_count=0, evidence_best=np.zeros(r_count, np.int64),
               images=0, defect_images=0, routed_images=0)
    original_combine = method._combine_maps

    def combine(maps, resolutions_to_use):
        if [int(r) for r in resolutions_to_use] == resolutions:
            holder["maps"] = torch.stack([m.detach().float() for m in maps])
        return original_combine(maps, resolutions_to_use)

    original_predict = method.predict

    def predict(path):
        holder.clear()
        result = original_predict(path)
        maps = holder.pop("maps", None)
        if maps is None:
            acc["routed_images"] += 1
            return result
        anomaly = Path(path).parent.name
        mask = _ground_truth_mask(dataset, name, anomaly, path, tuple(maps.shape[1:]))
        mask = torch.from_numpy(mask).to(maps.device)
        counts, regions = image_statistics(maps, mask)
        acc["normal"] += counts["normal"]
        acc["defect"] += counts["defect"]
        for key in ("size_top1", "size_regions", "evidence_best"):
            acc[key] += regions[key]
        for key in ("agree_pixels", "region_pixels", "region_count"):
            acc[key] += regions[key]
        acc["images"] += 1
        acc["defect_images"] += int(bool(mask.any()))
        return result

    method._combine_maps = combine
    method.predict = predict
    root = Path(config["output_dir"])
    previous = set(root.rglob("run_manifest.json"))
    run_pipeline(method, dataset, backbone, config, config["output_dir"])
    manifests = list(set(root.rglob("run_manifest.json")) - previous)
    if len(manifests) != 1:
        raise RuntimeError("rank diagnostic expected exactly one new run manifest")
    run_root = manifests[0].parent
    record = dict(
        object=name, resolutions=resolutions,
        statistical_normalization=config["method_config"].get("statistical_normalization"),
        multiscale_aggregation=config["method_config"].get("multiscale_aggregation"),
        shots=config.get("protocol", {}).get("shots"),
        renyi_weights_top_first=renyi_weights_top_first(r_count).tolist(),
        counts={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in acc.items()},
        summary=summarize(acc, resolutions),
        label_use="test_public GT read after prediction for this diagnostic only; not used in fit/predict",
        completed_at=time.time(),
    )
    (run_root / "order_rank_diagnostic.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"RANK_DIAGNOSTIC_COMPLETE {name} images={acc['images']} routed={acc['routed_images']}", flush=True)


if __name__ == "__main__":
    main()

