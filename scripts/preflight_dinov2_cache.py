"""Load a cached DINOv2 model to verify an offline remote runtime."""

from __future__ import annotations

import argparse

import src.backbones  # noqa: F401
from src.dinov2_loader import load_dinov2_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dinov2_vitl14")
    args = parser.parse_args()
    model = load_dinov2_model(args.model, strict_offline=True)
    print(type(model).__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
