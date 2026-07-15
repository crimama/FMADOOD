#!/usr/bin/env python3
"""Evaluate half-resolution structure-guided refinement of retained FlowTTE maps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.flow_tte_phase2_refinement import OBJECTS, analyze_run  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--objects", nargs="+", default=list(OBJECTS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--variant-set", choices=("full", "core"), default="full")
    parser.add_argument(
        "--drop-joint-bilateral",
        action="store_true",
        help="Documented fallback when the remote OpenCV build lacks ximgproc joint bilateral.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    objects = [part for value in args.objects for part in value.replace(",", " ").split() if part]
    analyze_run(
        args.result_root,
        args.data_root,
        objects,
        args.output_dir,
        workers=args.workers,
        variant_set=args.variant_set,
        include_joint=False if args.drop_joint_bilateral else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
