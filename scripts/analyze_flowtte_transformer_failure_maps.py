# /// script
# requires-python = ">=3.8"
# dependencies = ["numpy", "opencv-python-headless", "pillow", "tifffile"]
# ///
# pyright: reportMissingImports=false
"""Compare FlowTTE baseline and Transformer Flow anomaly maps image by image."""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

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
class ImageStats:
    object_name: str
    split: str
    stem: str
    shift_type: str
    gt_area: float
    baseline_mean: float
    transformer_mean: float
    baseline_p99: float
    transformer_p99: float
    baseline_gap_z: float
    transformer_gap_z: float
    delta_gap_z: float
    baseline_top1_gt_share: float
    transformer_top1_gt_share: float
    delta_top1_gt_share: float
    baseline_positive_area: float
    transformer_positive_area: float
    delta_positive_area: float
    baseline_component_count: int
    transformer_component_count: int
    delta_component_count: int
    baseline_largest_component_share: float
    transformer_largest_component_share: float
    delta_largest_component_share: float
    baseline_iou: float
    transformer_iou: float
    delta_iou: float
    baseline_precision: float
    transformer_precision: float
    delta_precision: float
    baseline_recall: float
    transformer_recall: float
    delta_recall: float


@dataclass(frozen=True)
class ObjectSummary:
    object_name: str
    n_images: int
    n_bad_images: int
    mean_delta_gap_z_bad: float
    mean_delta_top1_gt_share_bad: float
    mean_delta_iou_bad: float
    mean_delta_precision_bad: float
    mean_delta_recall_bad: float
    mean_delta_positive_area_all: float
    mean_delta_component_count_all: float
    baseline_mean_components_all: float
    transformer_mean_components_all: float
    worse_gap_fraction_bad: float


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--transformer-root", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-panels", type=int, default=24)
    return parser.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    objects = tuple(part for part in args.objects.replace(",", " ").split() if part)
    baseline = RunRoot("baseline", Path(args.baseline_root))
    transformer = RunRoot("transformer", Path(args.transformer_root))

    rows = collect_image_stats(data_root, baseline, transformer, objects)
    object_rows = summarize_objects(rows, objects)
    shift_rows = summarize_by_shift(rows)

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "image_pairs.csv", rows)
    write_csv(output_root / "object_summary.csv", object_rows)
    write_dict_csv(output_root / "shift_summary.csv", shift_rows)
    write_json(
        output_root / "summary.json",
        {
            "baseline_root": str(baseline.path),
            "transformer_root": str(transformer.path),
            "objects": list(objects),
            "n_image_pairs": len(rows),
            "object_summary": [asdict(row) for row in object_rows],
            "shift_summary": shift_rows,
        },
    )
    write_representative_panels(
        data_root=data_root,
        baseline=baseline,
        transformer=transformer,
        rows=rows,
        output_dir=output_root / "panels",
        max_panels=int(args.max_panels),
    )
    return 0


def collect_image_stats(
    data_root: Path,
    baseline: RunRoot,
    transformer: RunRoot,
    objects: Sequence[str],
) -> List[ImageStats]:
    baseline_metrics = collect_metrics_by_object(baseline.path)
    transformer_metrics = collect_metrics_by_object(transformer.path)
    rows: List[ImageStats] = []
    for object_name in objects:
        baseline_dir = locate_object_map_dir(baseline.path, object_name)
        transformer_dir = locate_object_map_dir(transformer.path, object_name)
        baseline_threshold = object_threshold(baseline_metrics, object_name)
        transformer_threshold = object_threshold(transformer_metrics, object_name)
        baseline_maps = map_paths_by_key(baseline_dir)
        transformer_maps = map_paths_by_key(transformer_dir)
        missing = sorted(set(baseline_maps) ^ set(transformer_maps))
        if missing:
            message = f"map set mismatch for {object_name}: {missing[:5]}"
            raise RuntimeError(message)
        for split, stem in sorted(baseline_maps):
            baseline_map = load_map(baseline_maps[(split, stem)])
            transformer_map = load_map(transformer_maps[(split, stem)])
            if baseline_map.shape != transformer_map.shape:
                message = f"shape mismatch for {object_name}/{split}/{stem}"
                raise RuntimeError(message)
            gt_mask = load_gt_mask(data_root, object_name, split, stem, baseline_map.shape)
            base_binary = baseline_map >= baseline_threshold
            transformer_binary = transformer_map >= transformer_threshold
            base_continuous = continuous_stats(baseline_map, gt_mask)
            transformer_continuous = continuous_stats(transformer_map, gt_mask)
            base_binary_stats = binary_stats(base_binary, gt_mask)
            transformer_binary_stats = binary_stats(transformer_binary, gt_mask)
            rows.append(
                ImageStats(
                    object_name=object_name,
                    split=split,
                    stem=stem,
                    shift_type=infer_shift_type(stem),
                    gt_area=float(np.mean(gt_mask)),
                    baseline_mean=base_continuous.mean,
                    transformer_mean=transformer_continuous.mean,
                    baseline_p99=base_continuous.p99,
                    transformer_p99=transformer_continuous.p99,
                    baseline_gap_z=base_continuous.gap_z,
                    transformer_gap_z=transformer_continuous.gap_z,
                    delta_gap_z=transformer_continuous.gap_z - base_continuous.gap_z,
                    baseline_top1_gt_share=base_continuous.top1_gt_share,
                    transformer_top1_gt_share=transformer_continuous.top1_gt_share,
                    delta_top1_gt_share=(
                        transformer_continuous.top1_gt_share - base_continuous.top1_gt_share
                    ),
                    baseline_positive_area=base_binary_stats.positive_area,
                    transformer_positive_area=transformer_binary_stats.positive_area,
                    delta_positive_area=(
                        transformer_binary_stats.positive_area - base_binary_stats.positive_area
                    ),
                    baseline_component_count=base_binary_stats.component_count,
                    transformer_component_count=transformer_binary_stats.component_count,
                    delta_component_count=(
                        transformer_binary_stats.component_count
                        - base_binary_stats.component_count
                    ),
                    baseline_largest_component_share=base_binary_stats.largest_component_share,
                    transformer_largest_component_share=(
                        transformer_binary_stats.largest_component_share
                    ),
                    delta_largest_component_share=(
                        transformer_binary_stats.largest_component_share
                        - base_binary_stats.largest_component_share
                    ),
                    baseline_iou=base_binary_stats.iou,
                    transformer_iou=transformer_binary_stats.iou,
                    delta_iou=transformer_binary_stats.iou - base_binary_stats.iou,
                    baseline_precision=base_binary_stats.precision,
                    transformer_precision=transformer_binary_stats.precision,
                    delta_precision=(
                        transformer_binary_stats.precision - base_binary_stats.precision
                    ),
                    baseline_recall=base_binary_stats.recall,
                    transformer_recall=transformer_binary_stats.recall,
                    delta_recall=transformer_binary_stats.recall - base_binary_stats.recall,
                )
            )
    return rows


def collect_metrics_by_object(run_root: Path) -> Dict[str, Dict[str, JsonValue]]:
    metrics: Dict[str, Dict[str, JsonValue]] = {}
    for path in sorted(run_root.glob("**/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            if isinstance(value, dict) and "best_thre" in value:
                metrics[key] = value
    if not metrics:
        message = f"no object metrics found under {run_root}"
        raise RuntimeError(message)
    return metrics


def object_threshold(metrics: Dict[str, Dict[str, JsonValue]], object_name: str) -> float:
    payload = metrics.get(object_name)
    if payload is None:
        message = f"missing metrics for {object_name}"
        raise RuntimeError(message)
    threshold = payload.get("best_thre")
    if not isinstance(threshold, (float, int)):
        message = f"missing best_thre for {object_name}"
        raise RuntimeError(message)
    return float(threshold)


def locate_object_map_dir(run_root: Path, object_name: str) -> Path:
    direct = run_root / "anomaly_maps" / object_name / "test"
    if direct.is_dir():
        return direct
    matches = sorted(run_root.glob(f"**/anomaly_maps/{object_name}/test"))
    if len(matches) != 1:
        message = f"expected one map dir for {object_name} under {run_root}, found {len(matches)}"
        raise RuntimeError(message)
    return matches[0]


def map_paths_by_key(test_dir: Path) -> Dict[Tuple[str, str], Path]:
    paths: Dict[Tuple[str, str], Path] = {}
    for path in sorted(test_dir.glob("*/*.tiff")):
        paths[(path.parent.name, path.stem)] = path
    if not paths:
        message = f"no TIFF anomaly maps found in {test_dir}"
        raise RuntimeError(message)
    return paths


def load_map(path: Path) -> FloatArray:
    return np.asarray(tiff.imread(path), dtype=np.float32)


def load_gt_mask(
    data_root: Path,
    object_name: str,
    split: str,
    stem: str,
    prediction_shape: Sequence[int],
) -> BoolArray:
    if split == "good":
        return np.zeros(tuple(prediction_shape), dtype=np.bool_)
    path = data_root / object_name / "test_public" / "ground_truth" / split / f"{stem}_mask.png"
    if not path.is_file():
        return np.zeros(tuple(prediction_shape), dtype=np.bool_)
    mask = np.asarray(Image.open(path)) > 0
    if tuple(mask.shape) != tuple(prediction_shape):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (int(prediction_shape[1]), int(prediction_shape[0])),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
    return np.asarray(mask, dtype=np.bool_)


@dataclass(frozen=True)
class ContinuousStats:
    mean: float
    p99: float
    gap_z: float
    top1_gt_share: float


def continuous_stats(prediction: FloatArray, gt_mask: BoolArray) -> ContinuousStats:
    mean = float(np.mean(prediction))
    p99 = float(np.quantile(prediction, 0.99))
    gt_count = int(np.count_nonzero(gt_mask))
    if gt_count == 0:
        gap_z = 0.0
        top1_gt_share = 0.0
    else:
        inside = float(np.mean(prediction[gt_mask]))
        outside = float(np.mean(prediction[~gt_mask]))
        std = float(np.std(prediction))
        gap_z = 0.0 if std <= 1e-12 else (inside - outside) / std
        top1_gt_share = top_fraction_gt_share(prediction, gt_mask, 0.01)
    return ContinuousStats(mean=mean, p99=p99, gap_z=float(gap_z), top1_gt_share=top1_gt_share)


def top_fraction_gt_share(prediction: FloatArray, gt_mask: BoolArray, fraction: float) -> float:
    flat = prediction.reshape(-1)
    mask = gt_mask.reshape(-1)
    top_count = max(1, int(math.ceil(float(flat.size) * fraction)))
    threshold_index = flat.size - top_count
    threshold = np.partition(flat, threshold_index)[threshold_index]
    selected = flat >= threshold
    selected_count = int(np.count_nonzero(selected))
    if selected_count == 0:
        return 0.0
    return float(np.count_nonzero(mask & selected) / selected_count)


@dataclass(frozen=True)
class BinaryStats:
    positive_area: float
    component_count: int
    largest_component_share: float
    iou: float
    precision: float
    recall: float


def binary_stats(predicted: BoolArray, gt_mask: BoolArray) -> BinaryStats:
    positive_count = int(np.count_nonzero(predicted))
    positive_area = float(np.mean(predicted))
    component_count, largest_component_share = connected_morphology(predicted)
    gt_count = int(np.count_nonzero(gt_mask))
    if gt_count == 0:
        return BinaryStats(
            positive_area=positive_area,
            component_count=component_count,
            largest_component_share=largest_component_share,
            iou=0.0,
            precision=0.0,
            recall=0.0,
        )
    intersection = int(np.count_nonzero(predicted & gt_mask))
    union = int(np.count_nonzero(predicted | gt_mask))
    iou = 0.0 if union == 0 else float(intersection / union)
    precision = 0.0 if positive_count == 0 else float(intersection / positive_count)
    recall = float(intersection / gt_count)
    return BinaryStats(
        positive_area=positive_area,
        component_count=component_count,
        largest_component_share=largest_component_share,
        iou=iou,
        precision=precision,
        recall=recall,
    )


def connected_morphology(mask: BoolArray) -> Tuple[int, float]:
    positive_count = int(np.count_nonzero(mask))
    if positive_count == 0:
        return 0, 0.0
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return 0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return int(count - 1), float(np.max(areas) / positive_count)


def infer_shift_type(stem: str) -> str:
    for suffix in ("overexposed", "underexposed", "shift_1", "regular"):
        if stem.endswith(suffix):
            return suffix
    return "unknown"


def summarize_objects(rows: Sequence[ImageStats], objects: Sequence[str]) -> List[ObjectSummary]:
    summaries: List[ObjectSummary] = []
    for object_name in objects:
        object_rows = [row for row in rows if row.object_name == object_name]
        bad_rows = [row for row in object_rows if row.split != "good"]
        summaries.append(
            ObjectSummary(
                object_name=object_name,
                n_images=len(object_rows),
                n_bad_images=len(bad_rows),
                mean_delta_gap_z_bad=mean_value(bad_rows, "delta_gap_z"),
                mean_delta_top1_gt_share_bad=mean_value(bad_rows, "delta_top1_gt_share"),
                mean_delta_iou_bad=mean_value(bad_rows, "delta_iou"),
                mean_delta_precision_bad=mean_value(bad_rows, "delta_precision"),
                mean_delta_recall_bad=mean_value(bad_rows, "delta_recall"),
                mean_delta_positive_area_all=mean_value(object_rows, "delta_positive_area"),
                mean_delta_component_count_all=mean_value(object_rows, "delta_component_count"),
                baseline_mean_components_all=mean_value(object_rows, "baseline_component_count"),
                transformer_mean_components_all=mean_value(object_rows, "transformer_component_count"),
                worse_gap_fraction_bad=fraction_where(bad_rows, "delta_gap_z", upper=0.0),
            )
        )
    return summaries


def summarize_by_shift(rows: Sequence[ImageStats]) -> List[Dict[str, JsonValue]]:
    grouped: Dict[Tuple[str, str], List[ImageStats]] = {}
    for row in rows:
        grouped.setdefault((row.object_name, row.shift_type), []).append(row)
    result: List[Dict[str, JsonValue]] = []
    for (object_name, shift_type), group in sorted(grouped.items()):
        bad_group = [row for row in group if row.split != "good"]
        result.append(
            {
                "object_name": object_name,
                "shift_type": shift_type,
                "n_images": len(group),
                "n_bad_images": len(bad_group),
                "mean_delta_gap_z_bad": mean_value(bad_group, "delta_gap_z"),
                "mean_delta_top1_gt_share_bad": mean_value(bad_group, "delta_top1_gt_share"),
                "mean_delta_iou_bad": mean_value(bad_group, "delta_iou"),
                "mean_delta_component_count_all": mean_value(group, "delta_component_count"),
                "mean_delta_positive_area_all": mean_value(group, "delta_positive_area"),
            }
        )
    return result


def mean_value(rows: Sequence[ImageStats], field_name: str) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([float(getattr(row, field_name)) for row in rows]))


def fraction_where(rows: Sequence[ImageStats], field_name: str, upper: float) -> float:
    if not rows:
        return float("nan")
    values = [float(getattr(row, field_name)) for row in rows]
    return float(np.mean([value < upper for value in values]))


def write_csv(path: Path, rows: Sequence[Any]) -> None:
    if not rows:
        raise RuntimeError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_dict_csv(path: Path, rows: Sequence[Dict[str, JsonValue]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_representative_panels(
    data_root: Path,
    baseline: RunRoot,
    transformer: RunRoot,
    rows: Sequence[ImageStats],
    output_dir: Path,
    max_panels: int,
) -> None:
    bad_rows = [row for row in rows if row.split != "good"]
    degraded = sorted(bad_rows, key=lambda row: (row.delta_gap_z, row.delta_iou))[
        : max_panels // 2
    ]
    improved = sorted(bad_rows, key=lambda row: (row.delta_gap_z, row.delta_iou), reverse=True)[
        : max_panels - len(degraded)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    for prefix, selected in (("degraded", degraded), ("improved", improved)):
        for index, row in enumerate(selected, start=1):
            path = output_dir / f"{prefix}_{index:02d}_{row.object_name}_{row.stem}.png"
            write_panel(data_root, baseline, transformer, row, path)


def write_panel(
    data_root: Path,
    baseline: RunRoot,
    transformer: RunRoot,
    row: ImageStats,
    output_path: Path,
) -> None:
    baseline_map_path = map_paths_by_key(locate_object_map_dir(baseline.path, row.object_name))[
        (row.split, row.stem)
    ]
    transformer_map_path = map_paths_by_key(locate_object_map_dir(transformer.path, row.object_name))[
        (row.split, row.stem)
    ]
    base_map = load_map(baseline_map_path)
    transformer_map = load_map(transformer_map_path)
    gt_mask = load_gt_mask(data_root, row.object_name, row.split, row.stem, base_map.shape)
    image = load_image(data_root, row.object_name, row.split, row.stem, base_map.shape)
    gt_panel = overlay_mask(image, gt_mask)
    base_heat = heatmap(base_map)
    transformer_heat = heatmap(transformer_map)
    delta_heat = signed_delta_heatmap(normalize01(transformer_map) - normalize01(base_map))
    panels = [
        label_panel(image, "image"),
        label_panel(gt_panel, "GT"),
        label_panel(base_heat, "MLP"),
        label_panel(transformer_heat, "Transformer"),
        label_panel(delta_heat, "T - MLP"),
    ]
    output = np.concatenate(panels, axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))


def load_image(
    data_root: Path,
    object_name: str,
    split: str,
    stem: str,
    shape: Sequence[int],
) -> npt.NDArray[np.uint8]:
    directory = data_root / object_name / "test_public" / split
    matches = sorted(directory.glob(f"{stem}.*"))
    if not matches:
        return np.zeros((int(shape[0]), int(shape[1]), 3), dtype=np.uint8)
    image = np.asarray(Image.open(matches[0]).convert("RGB"))
    if image.shape[:2] != tuple(shape):
        image = cv2.resize(image, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_AREA)
    return np.asarray(image, dtype=np.uint8)


def overlay_mask(image: npt.NDArray[np.uint8], mask: BoolArray) -> npt.NDArray[np.uint8]:
    overlay = image.copy()
    overlay[mask] = (255, 40, 40)
    return np.asarray(cv2.addWeighted(image, 0.65, overlay, 0.35, 0.0), dtype=np.uint8)


def heatmap(values: FloatArray) -> npt.NDArray[np.uint8]:
    normalized = (normalize01(values) * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    return np.asarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def signed_delta_heatmap(values: FloatArray) -> npt.NDArray[np.uint8]:
    clipped = np.clip(values, -1.0, 1.0)
    positive = np.clip(clipped, 0.0, 1.0)
    negative = np.clip(-clipped, 0.0, 1.0)
    image = np.zeros((*clipped.shape, 3), dtype=np.float32)
    image[..., 0] = positive * 255.0
    image[..., 2] = negative * 255.0
    image[..., 1] = (1.0 - np.maximum(positive, negative)) * 255.0
    return np.asarray(image, dtype=np.uint8)


def normalize01(values: FloatArray) -> FloatArray:
    low = float(np.quantile(values, 0.01))
    high = float(np.quantile(values, 0.995))
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def label_panel(image: npt.NDArray[np.uint8], label: str) -> npt.NDArray[np.uint8]:
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (min(panel.shape[1], 220), 28), (0, 0, 0), thickness=-1)
    cv2.putText(
        panel,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
