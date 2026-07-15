"""Merge disjoint classic MVTec AD1 per-object metric chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

_METRICS = ("i_AUROC", "i_AUPRC", "p_AUROC", "p_AUPRC", "p_AUPRO")
_ALIASES = {
    "image_AUROC": "i_AUROC",
    "image_AP": "i_AUPRC",
    "pixel_AUROC": "p_AUROC",
    "pixel_AP": "p_AUPRC",
    "pixel_PRO": "p_AUPRO",
}


def aggregate(paths: Sequence[Path], expected_objects: Sequence[str]) -> dict[str, Any]:
    per_object: dict[str, Any] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for object_name, row in payload["per_object"].items():
            if object_name in per_object:
                raise ValueError(f"Duplicate object metric: {object_name}")
            per_object[object_name] = row
    if set(per_object) != set(expected_objects):
        missing = sorted(set(expected_objects) - set(per_object))
        extra = sorted(set(per_object) - set(expected_objects))
        raise ValueError(f"Object coverage mismatch: missing={missing}, extra={extra}")
    ordered = {name: per_object[name] for name in expected_objects}
    output: dict[str, Any] = {
        "dataset": "MVTec AD1 classic",
        "objects": list(expected_objects),
        "seed": 0,
        "pro_integration_limit": 0.05,
        "image_score_aggregation": "mean_top_1_percent_full_resolution_map",
        "pixel_score_quantization": "signed_log1p_linear_uint16_65536_per_object",
        "pixel_PRO_max_fpr": 0.30,
        "per_object": ordered,
    }
    for metric in _METRICS:
        output[metric] = sum(float(row[metric]) for row in ordered.values()) / len(ordered)
    for alias, metric in _ALIASES.items():
        output[alias] = output[metric]
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    objects = tuple(part for part in args.objects.replace(",", " ").split() if part)
    payload = aggregate([Path(path) for path in args.metrics], objects)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
