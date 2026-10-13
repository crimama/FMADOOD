"""Cache real RSG4 maps for the SCARF overview figure (draw_scarf_overview.py).

Reuses the canonical RSG4 configuration and audit hooks of
extract_rsg4_mechanism_maps.py. Ground-truth masks are read only after
prediction, for display; they never enter fitting or prediction. The query was
picked post hoc from the walnuts test_public scan for illustration.

Usage:
    .venv/bin/python PaperWorks/figures/extract_scarf_overview_maps.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from extract_rsg4_mechanism_maps import (CACHE, DATA, MANIFEST, build_method,
                                         display_resize, predict_query)
from fmad.protocol import MethodContext, seed_generators, select_support_paths

LSO_RESOLUTION = 672
LSO_BAG = 0
THUMB_EDGE = 160


def fit(method, name: str, support: list[str], seed: int) -> None:
    seed_generators(seed)
    method.backbone.warmup(support[0], iters=1)
    method.fit(support, MethodContext(name, seed, 448, False, False))


def lso_samples(method) -> dict[str, np.ndarray]:
    """Support-only leave-source-out NN distances kept by the tail fit."""
    tails = method.bag_tails[LSO_RESOLUTION][LSO_BAG]
    out = {}
    for layer, tail in zip(method.layers, tails):
        out[f"lso_l{layer}"] = tail["sample"].numpy().astype(np.float32)
        out[f"lso_u_l{layer}"] = np.float32(tail["u"].item())
    out["lso_tau_l5"] = np.float32(tails[0]["spike_tau"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cls", default="walnuts")
    parser.add_argument("--defect", default="006_shift_2")
    parser.add_argument("--good", default="000_shift_2")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = DATA / "mvtec_ad_2" / args.cls
    train = sorted(str(p) for p in (root / "train/good").glob("*.png"))
    support, _ = select_support_paths(train, 16, args.seed, manifest_path=MANIFEST,
                                      object_name=args.cls)
    method = build_method(args.device)
    fit(method, f"ad2_{args.cls}", support, args.seed)

    test = root / "test_public"
    cases = {
        "defect": (test / f"bad/{args.defect}.png",
                   test / f"ground_truth/bad/{args.defect}_mask.png"),
        "good": (test / f"good/{args.good}.png", None),
    }
    name = f"overview_{args.cls}"
    CACHE.mkdir(parents=True, exist_ok=True)
    meta = {"class": args.cls, "seed": args.seed, "support": support,
            "manifest": str(MANIFEST.relative_to(MANIFEST.parents[2])),
            "noise_reference_sigma": method.noise_reference_sigma,
            "lso": {"resolution": LSO_RESOLUTION, "bag": LSO_BAG}, "queries": {}}
    for tag, (path, gt_path) in cases.items():
        arrays = predict_query(method, path, gt_path, short_edge=512)
        meta["queries"][tag] = {"path": str(path), "noise_ratio": float(arrays["noise_ratio"]),
                                "noise_fired": bool(arrays["noise_fired"]),
                                "resolutions": [int(r) for r in arrays["resolutions"]]}
        np.savez_compressed(CACHE / f"{name}__{tag}.npz", **{
            k: (v.astype(np.float16) if isinstance(v, np.ndarray) and v.dtype == np.float32
                and not k.startswith("stat_") else v) for k, v in arrays.items()})
        print(name, tag, meta["queries"][tag], flush=True)

    thumbs = np.stack([display_resize(np.asarray(Image.open(p).convert("RGB")),
                                      THUMB_EDGE, "rgb")[:THUMB_EDGE, :THUMB_EDGE]
                       for p in support[:4]])
    np.savez_compressed(CACHE / f"{name}__support.npz", thumbs=thumbs, **lso_samples(method))
    (CACHE / f"{name}__meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
