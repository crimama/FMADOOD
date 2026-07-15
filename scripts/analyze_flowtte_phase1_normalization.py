#!/usr/bin/env python3
"""Evaluate label-free per-image normalization on retained FlowTTE maps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.flow_tte_phase1_normalization import (  # noqa: E402
    OBJECTS,
    analyze_run,
    analyze_supplementary_run,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--objects", nargs="+", default=list(OBJECTS))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--supplementary-only",
        action="store_true",
        help=(
            "evaluate only condition_group_quantile_match_to_regular_q4096 and "
            "condition_tail_affine_to_regular; write collision-safe supplementary artifacts"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    objects = [part for value in args.objects for part in value.replace(",", " ").split() if part]
    analyzer = analyze_supplementary_run if args.supplementary_only else analyze_run
    analyzer(args.result_root, args.data_root, objects, args.output_dir, workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
