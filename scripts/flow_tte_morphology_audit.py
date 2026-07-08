# /// script
# requires-python = ">=3.8"
# dependencies = ["numpy", "opencv-python-headless", "pillow", "tifffile"]
# ///
# pyright: reportMissingImports=false
"""Summarize thresholded anomaly-map morphology for FlowTTE diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Union

import cv2
import numpy as np
import numpy.typing as npt
import tifffile as tiff
from PIL import Image

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
JsonScalar = Union[str, int, float, bool, None]
JsonValue = Union[JsonScalar, Dict[str, "JsonValue"], List["JsonValue"]]


@dataclass(frozen=True)
class RunRoot:
    name: str
    path: Path


@dataclass(frozen=True)
class MapStats:
    run: str
    object_name: str
    split: str
    n_images: int
    threshold: float
    mean_positive_area: float
    median_positive_area: float
    mean_component_count: float
    mean_largest_component_share: float
    mean_score: float
    mean_p99_score: float
    mean_gt_iou: float
    mean_gt_recall: float
    mean_gt_precision: float


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-root", action="append", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--output-tsv", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    data_root = Path(args.data_root)
    objects = tuple(part for part in args.objects.replace(",", " ").split() if part)
    run_roots = parse_run_roots(args.run_root)
    rows = collect_rows(data_root, run_roots, objects)
    write_tsv(Path(args.output_tsv), rows)
    write_json(Path(args.output_json), rows)
    return 0


def parse_run_roots(values: Sequence[str]) -> List[RunRoot]:
    roots: List[RunRoot] = []
    for raw in values:
        if "=" in raw:
            name, path_text = raw.split("=", 1)
        else:
            path_text = raw
            name = Path(raw).name
        roots.append(RunRoot(name=name, path=Path(path_text)))
    return roots


def collect_rows(
    data_root: Path,
    run_roots: Sequence[RunRoot],
    objects: Sequence[str],
) -> List[MapStats]:
    rows: List[MapStats] = []
    for run_root in run_roots:
        metrics = read_metrics(run_root.path / "metrics.json")
        for object_name in objects:
            threshold = object_threshold(metrics, object_name)
            rows.extend(summarize_object(data_root, run_root, object_name, threshold))
    return rows


def read_metrics(path: Path) -> Dict[str, JsonValue]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        message = f"metrics JSON must be an object: {path}"
        raise TypeError(message)
    return payload


def object_threshold(metrics: Dict[str, JsonValue], object_name: str) -> float:
    value = metrics.get(object_name)
    if not isinstance(value, dict):
        message = f"missing object metrics for {object_name}"
        raise TypeError(message)
    threshold = value.get("best_thre")
    if not isinstance(threshold, (float, int)):
        message = f"missing best_thre for {object_name}"
        raise TypeError(message)
    return float(threshold)


def summarize_object(
    data_root: Path,
    run_root: RunRoot,
    object_name: str,
    threshold: float,
) -> List[MapStats]:
    test_dir = run_root.path / "anomaly_maps" / object_name / "test"
    if not test_dir.is_dir():
        message = f"missing anomaly map directory: {test_dir}"
        raise RuntimeError(message)
    return [
        summarize_split(data_root, run_root, object_name, split_dir, threshold)
        for split_dir in sorted(path for path in test_dir.iterdir() if path.is_dir())
    ]


def summarize_split(
    data_root: Path,
    run_root: RunRoot,
    object_name: str,
    split_dir: Path,
    threshold: float,
) -> MapStats:
    split = split_dir.name
    map_paths = sorted(split_dir.glob("*.tiff"))
    if not map_paths:
        message = f"no maps for {run_root.name}/{object_name}/{split}"
        raise RuntimeError(message)
    areas: List[float] = []
    component_counts: List[float] = []
    largest_shares: List[float] = []
    means: List[float] = []
    p99s: List[float] = []
    ious: List[float] = []
    recalls: List[float] = []
    precisions: List[float] = []
    for map_path in map_paths:
        prediction = np.asarray(tiff.imread(map_path), dtype=np.float32)
        positive = np.asarray(prediction >= threshold, dtype=np.bool_)
        gt_mask = load_gt_mask(data_root, object_name, split, map_path.stem, prediction.shape)
        morphology = connected_morphology(positive)
        overlap = overlap_metrics(positive, gt_mask)
        areas.append(float(np.mean(positive)))
        component_counts.append(float(morphology[0]))
        largest_shares.append(morphology[1])
        means.append(float(np.mean(prediction)))
        p99s.append(float(np.quantile(prediction, 0.99)))
        ious.append(overlap[0])
        recalls.append(overlap[1])
        precisions.append(overlap[2])
    return MapStats(
        run=run_root.name,
        object_name=object_name,
        split=split,
        n_images=len(map_paths),
        threshold=threshold,
        mean_positive_area=float(np.mean(areas)),
        median_positive_area=float(np.median(areas)),
        mean_component_count=float(np.mean(component_counts)),
        mean_largest_component_share=float(np.mean(largest_shares)),
        mean_score=float(np.mean(means)),
        mean_p99_score=float(np.mean(p99s)),
        mean_gt_iou=float(np.mean(ious)),
        mean_gt_recall=float(np.mean(recalls)),
        mean_gt_precision=float(np.mean(precisions)),
    )


def load_gt_mask(
    data_root: Path,
    object_name: str,
    split: str,
    stem: str,
    prediction_shape: Sequence[int],
) -> BoolArray:
    if split == "good":
        return np.zeros(tuple(prediction_shape), dtype=np.bool_)
    gt_path = data_root / object_name / "test_public" / "ground_truth" / split / f"{stem}_mask.png"
    if not gt_path.is_file():
        return np.zeros(tuple(prediction_shape), dtype=np.bool_)
    mask = np.asarray(Image.open(gt_path)) > 0
    if tuple(mask.shape) != tuple(prediction_shape):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (int(prediction_shape[1]), int(prediction_shape[0])),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
    return np.asarray(mask, dtype=np.bool_)


def connected_morphology(mask: BoolArray) -> tuple[int, float]:
    positive_count = int(np.count_nonzero(mask))
    if positive_count == 0:
        return 0, 0.0
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    if count <= 1:
        return 0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return int(count - 1), float(np.max(areas) / positive_count)


def overlap_metrics(predicted: BoolArray, gt_mask: BoolArray) -> tuple[float, float, float]:
    gt_count = int(np.count_nonzero(gt_mask))
    pred_count = int(np.count_nonzero(predicted))
    if gt_count == 0:
        return 0.0, 0.0, 0.0
    intersection = int(np.count_nonzero(predicted & gt_mask))
    union = int(np.count_nonzero(predicted | gt_mask))
    iou = 0.0 if union == 0 else float(intersection / union)
    recall = float(intersection / gt_count)
    precision = 0.0 if pred_count == 0 else float(intersection / pred_count)
    return iou, recall, precision


def write_tsv(path: Path, rows: Sequence[MapStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json(path: Path, rows: Sequence[MapStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(row) for row in rows]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
