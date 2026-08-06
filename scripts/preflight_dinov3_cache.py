#!/usr/bin/env python3
"""Load a cached DINOv3 model through the compatibility-aware project wrapper."""

from __future__ import annotations

import argparse

from dinov3_backbone import DINOv3Backbone


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="dinov3_vith16plus")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--feature-layers", default="7,15,23,31")
    args = parser.parse_args()
    layers = tuple(int(part) for part in args.feature_layers.split(","))
    backbone = DINOv3Backbone(
        args.model,
        args.device,
        smaller_edge_size=672,
        feature_layers=layers,
    )
    backbone._ensure_loaded()  # noqa: SLF001
    print(type(backbone._model).__name__)  # noqa: SLF001
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
