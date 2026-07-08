# /// script
# requires-python = ">=3.8"
# dependencies = ["numpy", "opencv-python-headless", "pillow", "tifffile"]
# ///
# pyright: reportMissingImports=false
"""Summarize high-score connected-component fragmentation for FlowTTE maps."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.flow_tte_components import summarize_components  # noqa: E402
from scripts.flow_tte_postprocess_core import (  # noqa: E402
    MapSample,
    find_object_run_root,
    load_samples,
    read_metrics,
)


@dataclass(frozen=True)
class FragmentationRow:
    object_name: str
    threshold: float
    split_group: str
    image_count: int
    mean_component_count: float
    mean_positive_area: float
    mean_largest_component_share: float
    mean_component_area: float
    mean_predicted_gt_overlap: float
    mean_gt_recall: float


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-root", action="append", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-tsv", required=True)
    return parser.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    data_root = Path(args.data_root)
    run_roots = tuple(Path(raw) for raw in args.run_root)
    rows = tuple(
        row
        for object_name in args.objects.replace(",", " ").split()
        for row in collect_fragmentation_rows(data_root, run_roots, object_name)
    )
    write_outputs(Path(args.output_json), Path(args.output_tsv), rows)
    return 0


def collect_fragmentation_rows(
    data_root: Path,
    run_roots: Sequence[Path],
    object_name: str,
) -> Tuple[FragmentationRow, ...]:
    run_root = find_object_run_root(run_roots, object_name)
    samples = load_samples(data_root, run_root, object_name)
    threshold = read_metrics(run_root / "metrics.json", object_name).threshold
    return (
        summarize_group(object_name, threshold, "all", samples),
        summarize_group(
            object_name,
            threshold,
            "good",
            tuple(sample for sample in samples if sample.split == "good"),
        ),
        summarize_group(
            object_name,
            threshold,
            "bad",
            tuple(sample for sample in samples if sample.split != "good"),
        ),
    )


def summarize_group(
    object_name: str,
    threshold: float,
    split_group: str,
    samples: Sequence[MapSample],
) -> FragmentationRow:
    if not samples:
        return FragmentationRow(
            object_name=object_name,
            threshold=threshold,
            split_group=split_group,
            image_count=0,
            mean_component_count=0.0,
            mean_positive_area=0.0,
            mean_largest_component_share=0.0,
            mean_component_area=0.0,
            mean_predicted_gt_overlap=0.0,
            mean_gt_recall=0.0,
        )
    component_counts: List[float] = []
    positive_areas: List[float] = []
    largest_shares: List[float] = []
    component_areas: List[float] = []
    gt_overlaps: List[float] = []
    gt_recalls: List[float] = []
    for sample in samples:
        mask = sample.score >= threshold
        summary = summarize_components(mask)
        component_counts.append(float(summary.component_count))
        positive_areas.append(summary.positive_area)
        largest_shares.append(summary.largest_component_share)
        component_areas.append(summary.mean_component_area)
        predicted_positive = int(np.count_nonzero(mask))
        gt_positive = int(np.count_nonzero(sample.gt_mask))
        true_positive = int(np.count_nonzero(mask & sample.gt_mask))
        gt_overlaps.append(0.0 if predicted_positive == 0 else true_positive / predicted_positive)
        gt_recalls.append(0.0 if gt_positive == 0 else true_positive / gt_positive)
    return FragmentationRow(
        object_name=object_name,
        threshold=threshold,
        split_group=split_group,
        image_count=len(samples),
        mean_component_count=float(np.mean(component_counts)),
        mean_positive_area=float(np.mean(positive_areas)),
        mean_largest_component_share=float(np.mean(largest_shares)),
        mean_component_area=float(np.mean(component_areas)),
        mean_predicted_gt_overlap=float(np.mean(gt_overlaps)),
        mean_gt_recall=float(np.mean(gt_recalls)),
    )


def write_outputs(
    json_path: Path,
    tsv_path: Path,
    rows: Sequence[FragmentationRow],
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summarize_rows(rows), "rows": [asdict(row) for row in rows]}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def summarize_rows(rows: Sequence[FragmentationRow]) -> Dict[str, float]:
    all_rows = [row for row in rows if row.split_group == "all"]
    return {
        "mean_component_count_all": float(
            np.mean([row.mean_component_count for row in all_rows]),
        ),
        "mean_positive_area_all": float(np.mean([row.mean_positive_area for row in all_rows])),
        "mean_predicted_gt_overlap_all": float(
            np.mean([row.mean_predicted_gt_overlap for row in all_rows]),
        ),
        "object_count": float(len(all_rows)),
    }


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
