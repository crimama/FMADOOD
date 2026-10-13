"""Extract real RSG4 intermediate maps for the method-intuition figures.

Stage 1 (this file, GPU): fit RSG4 on normal support images and cache the
intermediate maps of a few queries. Ground-truth masks are loaded only for
display and pixel statistics after prediction; they never enter fitting or
prediction.

Usage:
    .venv/bin/python PaperWorks/figures/extract_rsg4_mechanism_maps.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

import fmad.backbones  # noqa: F401  (registers backbones)
from fmad.methods import register_methods
from fmad.protocol import MethodContext, select_support_paths, seed_generators
from fmad.registry import BACKBONE_REGISTRY, METHOD_REGISTRY

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(__file__).resolve().parent / "cache_rsg4_mechanism"
DATA = Path("/home/hun/Volume/DATA")
MANIFEST = ROOT / "configs/methods/VOID_support_16.json"

# Canonical RSG4 method_config (configs/methods/tail_bag_RSG4.yaml) plus the
# audit flags of the matched fusion_factorial implementation.
RSG4 = dict(
    shallow_mode="gated", mechanism_audit=True,
    variant="R1", feature_layers=[5, 11, 23], resolutions=[448, 672, 896, 1120],
    bank_tokens_per_source=256, query_chunk_size=128, bank_bags=4, knn_k=1,
    score_normalization="tailscale", layer_fusion="residual_spike",
    layer_fusion_tau_mode="support_quantile", layer_fusion_tau_q=0.990,
    layer_fusion_gamma=1.0, layer_fusion_shallow_idx=0,
    prediction_postprocess="component_quantile", component_percentile=99.0,
    component_q=1.0, component_dilate_size=0,
    noise_guard=True, noise_guard_k=9.0, noise_guard_action="lowres",
    noise_guard_resolutions=[448, 672],
)


def build_method(device: str):
    register_methods("fusion_factorial")
    base = METHOD_REGISTRY.get("fusion_factorial")

    class Recording(base):
        """Adds per-bag fused maps; everything else is the unmodified method."""

        def _score_tokens(self, layers, grid, bank, *, normalize=True):
            self._fused_bags = []
            score = super()._score_tokens(layers, grid, bank, normalize=normalize)
            if self._audit_active:
                self._bag_grids.append(np.stack([
                    b.reshape(*grid).cpu().numpy() for b in self._fused_bags]))
            return score

        def _fuse_layers(self, scores, tau=None):
            fused = super()._fuse_layers(scores, tau=tau)
            if self._audit_active:
                self._fused_bags.append(fused.detach())
            return fused

    backbone = BACKBONE_REGISTRY.get("dinov2_vitl14")(
        model_name="dinov2_vitl14", device=device,
        smaller_edge_size=448, feature_layers=[5, 11, 23])
    return Recording(backbone=backbone, config=dict(RSG4))


def display_resize(array: np.ndarray, short_edge: int, mode: str) -> np.ndarray:
    h, w = array.shape[:2]
    scale = short_edge / min(h, w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    resample = {"rgb": Image.BICUBIC, "mask": Image.NEAREST}.get(mode)
    if resample is not None:
        return np.asarray(Image.fromarray(array).resize(size, resample))
    tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))[None, None]
    return F.interpolate(tensor, size=size[::-1], mode="bilinear",
                         align_corners=False)[0, 0].numpy()


def upsample(grid: np.ndarray, shape) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(grid, dtype=np.float32))[None, None]
    return F.interpolate(tensor, size=tuple(shape), mode="bilinear",
                         align_corners=False)[0, 0].numpy()


def predict_query(method, path: Path, gt_path: Path | None, short_edge: int) -> dict:
    method._bag_grids = []
    [prediction] = method.predict_batch([str(path)])
    audit = prediction.diagnostics["fusion_audit"]
    rgb = np.asarray(Image.open(path).convert("RGB"))
    native = rgb.shape[:2]
    gt = np.zeros(native, dtype=bool)
    if gt_path is not None and gt_path.exists():
        gt = np.asarray(Image.open(gt_path).convert("L")) > 0
    out = {
        "rgb": display_resize(rgb, short_edge, "rgb"),
        "gt": display_resize(gt.astype(np.uint8) * 255, short_edge, "mask") > 0,
        "raw_map": display_resize(audit["raw_map"], short_edge, "map"),
        "final_map": display_resize(prediction.score_map, short_edge, "map"),
        "resolutions": np.asarray(method.last_resolutions),
        "noise_ratio": float(method.noise_guard_log[-1]["ratio"]) if method.noise_guard else float("nan"),
        "noise_fired": bool(method.noise_guard_log[-1]["fired"]) if method.noise_guard else False,
    }
    shape = out["raw_map"].shape
    for resolution, grids, gate, bags in zip(method.last_resolutions, audit["component_grids"],
                                             audit["gate_grids"], method._bag_grids):
        for key, grid in grids.items():
            out[f"r{resolution}_{key}"] = upsample(grid, shape)
        out[f"r{resolution}_gate"] = upsample(gate, shape)
        for index, bag in enumerate(bags):
            out[f"r{resolution}_bag{index}"] = upsample(bag, shape)
    # Native-resolution pixel statistics for the score-distribution panel.
    # The labels are evaluator-side only and are applied after prediction.
    stats = {}
    for name in ("deep", "shallow_full", "shallow_residual"):
        grids = [upsample(g[name], native) for g in audit["component_grids"]]
        stats[name] = np.mean(grids, axis=0)
    deep = stats["deep"]
    for label, fused in (("deep_only", deep), ("all_add", deep + stats["shallow_full"]),
                         ("gated", deep + stats["shallow_residual"])):
        out[f"stat_{label}_normal"] = np.random.default_rng(0).choice(
            fused[~gt], size=min(200_000, int((~gt).sum())), replace=False)
        if gt.any():
            out[f"stat_{label}_defect"] = fused[gt]
    return out


def run_case(method, name, support, queries, short_edge, seed=0, guard_off=()):
    seed_generators(seed)
    context = MethodContext(name, seed, 448, False, False)
    method.backbone.warmup(support[0], iters=1)
    method.fit(support, context)
    meta = {"support": support, "noise_reference_sigma": method.noise_reference_sigma,
            "taus": {}, "scales": {}}
    for resolution in method.resolutions:
        meta["taus"][str(resolution)] = [t[0]["spike_tau"] for t in [
            method.bag_tails[resolution][b] for b in range(method.bank_bags)]]
        meta["scales"][str(resolution)] = [
            [float(v) for v in method.bag_scales[resolution][b]] for b in range(method.bank_bags)]
    CACHE.mkdir(parents=True, exist_ok=True)
    for tag, (path, gt_path) in queries.items():
        # guard_off tags re-score the same query with the guard disabled, to
        # show what the guard prevents; the fitted state is unchanged.
        method.noise_guard = tag not in guard_off
        try:
            arrays = predict_query(method, Path(path), gt_path and Path(gt_path), short_edge)
        finally:
            method.noise_guard = True
        np.savez_compressed(CACHE / f"{name}__{tag}.npz", **{
            k: (v.astype(np.float16) if isinstance(v, np.ndarray) and v.dtype == np.float32
                and not k.startswith("stat_") else v) for k, v in arrays.items()})
        meta[tag] = {"path": str(path), "noise_ratio": arrays["noise_ratio"],
                     "noise_fired": arrays["noise_fired"],
                     "resolutions": arrays["resolutions"].tolist()}
        print(name, tag, "rho=%.3f" % arrays["noise_ratio"], arrays["resolutions"].tolist())
    (CACHE / f"{name}__meta.json").write_text(json.dumps(meta, indent=2))


def scan_class(method, name, support, items, short_edge=256, seed=0):
    """Light per-image cache (deep, z5, residual, gt) for picking illustrative pairs.

    GT is loaded after prediction for display/selection only.
    """
    seed_generators(seed)
    method.backbone.warmup(support[0], iters=1)
    method.fit(support, MethodContext(name, seed, 448, False, False))
    out = {}
    for tag, path, gt_path in items:
        method._bag_grids = []
        [prediction] = method.predict_batch([str(path)])
        audit = prediction.diagnostics["fusion_audit"]
        native = np.asarray(Image.open(path)).shape[:2]
        gt = np.zeros(native, dtype=np.uint8)
        if gt_path is not None and Path(gt_path).exists():
            gt = (np.asarray(Image.open(gt_path).convert("L")) > 0).astype(np.uint8) * 255
        shape = display_resize(gt, short_edge, "mask").shape
        for key in ("deep", "normalized_l5", "shallow_residual"):
            maps = [upsample(g[key], shape) for g in audit["component_grids"]]
            out[f"{tag}__{key}"] = np.mean(maps, axis=0).astype(np.float16)
        out[f"{tag}__gt"] = display_resize(gt, short_edge, "mask") > 0
        print(name, tag, flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / f"scan_{name}.npz", **out)


def scan_main(device, classes, stride):
    method = build_method(device)
    for name in classes:
        root = DATA / "mvtec_ad_2" / name
        train = sorted(str(p) for p in (root / "train/good").glob("*.png"))
        support, _ = select_support_paths(train, 16, 0, manifest_path=MANIFEST, object_name=name)
        test = root / "test_public"
        items = [("bad_" + p.stem, p, test / f"ground_truth/bad/{p.stem}_mask.png")
                 for p in sorted((test / "bad").glob("*.png"))[::stride]]
        items += [("good_" + p.stem, p, None) for p in sorted((test / "good").glob("*.png"))[::stride]]
        scan_class(method, name, support, items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ad2-class", default="fabric")
    parser.add_argument("--ad2-defect", default="000_regular")
    parser.add_argument("--ad2-good", default="000_regular")
    parser.add_argument("--ad-class", default="hazelnut")
    parser.add_argument("--ad-defect", default="crack/000")
    parser.add_argument("--noise-root", default="impulse_noise_s3")
    parser.add_argument("--ad-good", default="good/000")
    parser.add_argument("--scan", nargs="*", default=None,
                        help="MVTec AD 2 classes to scan for illustrative pairs")
    parser.add_argument("--scan-stride", type=int, default=3)
    args = parser.parse_args()
    if args.scan:
        scan_main(args.device, args.scan, args.scan_stride)
        return
    method = build_method(args.device)

    # (1) MVTec AD 2 test_public: fixed 16-shot manifest used by the paper runs.
    root = DATA / "mvtec_ad_2" / args.ad2_class
    train = sorted(str(p) for p in (root / "train/good").glob("*.png"))
    support, _ = select_support_paths(train, 16, 0, manifest_path=MANIFEST,
                                      object_name=args.ad2_class)
    test = root / "test_public"
    run_case(method, f"ad2_{args.ad2_class}", support, {
        "defect": (test / f"bad/{args.ad2_defect}.png",
                   test / f"ground_truth/bad/{args.ad2_defect}_mask.png"),
        "good": (test / f"good/{args.ad2_good}.png", None),
    }, short_edge=512)

    # (2) MVTec AD clean vs. noise-corrupted query; support stays clean.
    clean = DATA / "MVTecAD" / args.ad_class
    noisy = DATA / "MVTecAD_corrupted_heldout" / args.noise_root / args.ad_class
    train = sorted(str(p) for p in (clean / "train/good").glob("*.png"))
    support, _ = select_support_paths(train, 16, 0)
    defect_type, stem = args.ad_defect.split("/")
    gt = clean / f"ground_truth/{defect_type}/{stem}_mask.png"
    run_case(method, f"ad_{args.ad_class}", support, {
        "clean": (clean / f"test/{defect_type}/{stem}.png", gt),
        "noisy": (noisy / f"test/{defect_type}/{stem}.png", gt),
        "noisy_guardoff": (noisy / f"test/{defect_type}/{stem}.png", gt),
        "good_noisy": (noisy / f"test/{args.ad_good}.png", None),
        "good_noisy_guardoff": (noisy / f"test/{args.ad_good}.png", None),
    }, short_edge=448, guard_off=("noisy_guardoff", "good_noisy_guardoff"))


if __name__ == "__main__":
    main()
