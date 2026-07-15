"""Combine raw-map image metrics with refined-map pixel metrics."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


IMAGE_KEYS = ("i_AUROC", "i_AUPRC", "image_AUROC", "image_AP")
PIXEL_KEYS = (
    "p_AUROC",
    "p_AUPRC",
    "p_AUPRO",
    "pixel_AUROC",
    "pixel_AP",
    "pixel_PRO",
)


def combine_metrics(
    raw: Mapping[str, Any],
    refined: Mapping[str, Any],
) -> dict[str, Any]:
    """Return raw image metrics and refined pixel metrics under one contract."""
    raw_objects = tuple(raw.get("objects", ()))
    refined_objects = tuple(refined.get("objects", ()))
    if raw_objects != refined_objects or not raw_objects:
        raise ValueError(
            f"raw/refined object mismatch: {raw_objects} versus {refined_objects}",
        )
    if raw.get("dataset") != refined.get("dataset"):
        raise ValueError("raw/refined dataset mismatch")

    output = copy.deepcopy(dict(raw))
    for key in PIXEL_KEYS:
        output[key] = refined[key]
    for object_name in raw_objects:
        raw_row = raw["per_object"][object_name]
        refined_row = refined["per_object"][object_name]
        for key in IMAGE_KEYS:
            output["per_object"][object_name][key] = raw_row[key]
        for key in PIXEL_KEYS:
            output["per_object"][object_name][key] = refined_row[key]
    output["hybrid_contract"] = {
        "image_score_source": "density0_raw_map_mean_top_1_percent",
        "pixel_map_source": "guided_r8_eps1e-2_of_same_density0_raw_map",
        "gt_used_by_refinement": False,
    }
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-metrics", required=True)
    parser.add_argument("--refined-metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    raw = json.loads(Path(args.raw_metrics).read_text(encoding="utf-8"))
    refined = json.loads(Path(args.refined_metrics).read_text(encoding="utf-8"))
    output = combine_metrics(raw, refined)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
