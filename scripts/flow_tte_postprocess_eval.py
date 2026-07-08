# /// script
# requires-python = ">=3.8"
# dependencies = ["numpy", "opencv-python-headless", "pillow", "tifffile"]
# ///
# pyright: reportMissingImports=false
"""Evaluate threshold and morphology variants on saved FlowTTE anomaly maps."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.flow_tte_postprocess_core import (  # noqa: E402
    EvalConfig,
    VariantRow,
    collect_object_rows,
)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-root", action="append", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-tsv", required=True)
    parser.add_argument("--threshold-count", type=int, default=96)
    parser.add_argument("--line-length", type=int, default=17)
    parser.add_argument("--angle-count", type=int, default=16)
    return parser.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    config = EvalConfig(
        data_root=Path(args.data_root),
        run_roots=tuple(Path(raw) for raw in args.run_root),
        threshold_count=args.threshold_count,
        line_length=args.line_length,
        angle_count=args.angle_count,
    )
    rows = tuple(
        row
        for object_name in args.objects.replace(",", " ").split()
        for row in collect_object_rows(config, object_name)
    )
    write_outputs(Path(args.output_json), Path(args.output_tsv), rows)
    return 0


def write_outputs(json_path: Path, tsv_path: Path, rows: Sequence[VariantRow]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summarize(rows), "rows": [asdict(row) for row in rows]}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def summarize(rows: Sequence[VariantRow]) -> Dict[str, Dict[str, float]]:
    variants = sorted({row.variant for row in rows})
    return {
        variant: {
            "mean_f1": float(np.mean([row.f1 for row in rows if row.variant == variant])),
            "mean_source_seg_auroc": float(
                np.mean([row.source_seg_auroc for row in rows if row.variant == variant]),
            ),
            "mean_source_seg_f1": float(
                np.mean([row.source_seg_f1 for row in rows if row.variant == variant]),
            ),
        }
        for variant in variants
    }


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
