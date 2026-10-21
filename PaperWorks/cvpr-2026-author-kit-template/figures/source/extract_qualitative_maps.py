"""Cache final maps for the qualitative figure (Figure 4).

Fits the final ScopeAD configuration (seed 0, 16-shot, fixed manifest) twice,
with and without Q95 calibration. For every test_public image the
per-resolution maps are captured just before resolution aggregation; Mean and
Renyi aggregation are both computed from the same maps and both receive the
method's own CQ99 postprocessing. Ground truth is read only after prediction,
for display and threshold statistics. Nothing here is a new benchmark run.

Usage:
    .venv/bin/python PaperWorks/cvpr-2026-author-kit-template/figures/source/extract_qualitative_maps.py fabric walnuts
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.nn import functional as F

import fmad.backbones  # noqa: F401
import fmad.datasets  # noqa: F401
from fmad.methods import register_methods
from fmad.methods.multiscale_aggregation_reference_matching import combine_resolution_maps
from fmad.methods.reference_matching import _read_rgb
from fmad.pipeline import _ground_truth_mask
from fmad.protocol import MethodContext, seed_generators, select_support_paths
from fmad.registry import BACKBONE_REGISTRY, DATASET_REGISTRY, METHOD_REGISTRY

ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "configs/benchmarks/scopead_renyi_cq_20261018/ad2_scale_order_renyi_cq_s16.yaml"
CACHE = ROOT / "PaperWorks/figures/cache_scopead_qualitative"
DATA = Path("/home/hun/Volume/DATA/mvtec_ad_2")
SHORT = 256  # cached display short edge


def small(array, mode):
    h, w = array.shape[:2]
    size = (round(w * SHORT / min(h, w)), round(h * SHORT / min(h, w)))
    if mode == "rgb":
        return np.asarray(Image.fromarray(array).resize(size, Image.BICUBIC))
    if mode == "mask":
        return np.asarray(Image.fromarray(array.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 0
    t = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))[None, None]
    return F.interpolate(t, size=size[::-1], mode="area")[0, 0].numpy()


def run(name, config, device):
    seed = int(config["seed"])
    dataset = DATASET_REGISTRY.get(config["dataset"])(
        data_root=str(DATA), config={**config.get("dataset_config", {}), "objects": [name]})
    info = dataset.get_object_info(name)
    train = sorted(dataset.get_train_images(name))
    proto = config["protocol"]
    support, _ = select_support_paths(train, proto["shots"], seed,
                                      manifest_path=ROOT / proto["fixed_reference_manifest"],
                                      object_name=name)
    tests = dataset.get_test_images(name)
    paths = [(kind, p) for kind in sorted(tests) for p in sorted(tests[kind])]
    out = {"paths": np.array([p for _, p in paths]), "kind": np.array([k for k, _ in paths])}
    for norm in ("none", "q95"):
        seed_generators(seed)
        register_methods(config["method"])
        backbone = BACKBONE_REGISTRY.get(config["backbone"])(
            model_name=config["backbone"], device=device, **config.get("backbone_config", {}))
        mcfg = copy.deepcopy(config["method_config"])
        mcfg["statistical_normalization"] = norm
        method = METHOD_REGISTRY.get(config["method"])(backbone=backbone, config=mcfg)
        context = MethodContext(name, seed, info.resolution, info.masking, info.rotation)
        if hasattr(backbone, "set_resolution"):
            backbone.set_resolution(context.resolution)
        backbone.warmup(support[0], iters=int(proto.get("warmup_iters", 1)))
        method.fit(support, context)
        holder = {}
        original = method._combine_maps

        def combine(maps, used):
            holder["maps"] = [m.detach().float() for m in maps]
            holder["used"] = list(used)
            return original(maps, used)

        method._combine_maps = combine
        finals = {"mean": [], "renyi": []}
        routed = 0
        for kind, path in paths:
            holder.clear()
            method.predict(path)
            if holder.get("used") != list(method.resolutions):
                routed += 1
            image = _read_rgb(path)
            for agg, mode in (("mean", "mean"), ("renyi", "scale_order_renyi")):
                fused = combine_resolution_maps(holder["maps"], mode).cpu().numpy()
                final = method._postprocess_prediction(fused, image)
                finals[agg].append(small(final, "map").astype(np.float16))
        for agg, maps in finals.items():
            out[f"{norm}_{agg}"] = np.stack(maps)
        out[f"{norm}_routed"] = routed
        del method, backbone
        torch.cuda.empty_cache()
        print(name, norm, "done", "routed", routed, flush=True)
    rgbs, gts = [], []
    for kind, path in paths:  # labels read only after all predictions
        rgb = np.asarray(Image.open(path).convert("RGB"))
        mask = _ground_truth_mask(dataset, name, kind, path, rgb.shape[:2])
        rgbs.append(small(rgb, "rgb"))
        gts.append(small(np.asarray(mask, bool), "mask"))
    out["rgb"] = np.stack(rgbs)
    out["gt"] = np.stack(gts)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / f"{name}_s{proto['shots']}_seed{seed}.npz", **out)
    (CACHE / f"{name}_s{proto['shots']}_seed{seed}.json").write_text(json.dumps({
        "config": str(CONFIG.relative_to(ROOT)), "support": support,
        "arms": ["none_mean", "none_renyi", "q95_mean", "q95_renyi"],
        "postprocess": "method CQ99 applied to every arm", "short_edge": SHORT,
        "label_use": "GT read after all predictions; display and threshold statistics only",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("objects", nargs="+")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text())
    for name in args.objects:
        run(name, config, args.device)


if __name__ == "__main__":
    main()
