from __future__ import annotations

# pyright: reportMissingImports=false
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
import tifffile as tiff
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
JsonScalar = Union[str, int, float, bool, None]
JsonValue = Union[JsonScalar, Dict[str, "JsonValue"], List["JsonValue"]]


@dataclass(frozen=True)
class EvalConfig:
    data_root: Path
    run_roots: Tuple[Path, ...]
    threshold_count: int
    line_length: int
    angle_count: int


@dataclass(frozen=True)
class MapSample:
    score: FloatArray
    gt_mask: BoolArray
    split: str


@dataclass(frozen=True)
class SourceMetrics:
    threshold: float
    auroc: float
    f1: float


@dataclass(frozen=True)
class ObjectInput:
    object_name: str
    samples: Tuple[MapSample, ...]
    metrics: SourceMetrics


@dataclass(frozen=True)
class VariantProfile:
    name: str
    erosion_size: int
    use_morphology: bool


@dataclass(frozen=True)
class VariantRow:
    object_name: str
    variant: str
    threshold: float
    f1: float
    precision: float
    recall: float
    positive_area: float
    source_seg_auroc: float
    source_seg_f1: float


@dataclass(frozen=True)
class BinaryMaskMetrics:
    f1: float
    precision: float
    recall: float
    positive_area: float


def collect_object_rows(config: EvalConfig, object_name: str) -> Tuple[VariantRow, ...]:
    run_root = find_object_run_root(config.run_roots, object_name)
    object_input = ObjectInput(
        object_name=object_name,
        samples=load_samples(config.data_root, run_root, object_name),
        metrics=read_metrics(run_root / "metrics.json", object_name),
    )
    thresholds = candidate_thresholds(object_input.samples, config.threshold_count)
    rows: List[VariantRow] = []
    for profile in variant_profiles():
        rows.append(evaluate_variant(object_input, profile, object_input.metrics.threshold, config))
        rows.append(best_variant(object_input, profile, thresholds, config))
    return tuple(rows)


def variant_profiles() -> Tuple[VariantProfile, ...]:
    return (
        VariantProfile(name="raw", erosion_size=0, use_morphology=False),
        VariantProfile(name="closefill", erosion_size=0, use_morphology=True),
        VariantProfile(name="closefill_erode", erosion_size=3, use_morphology=True),
    )


def variant_profile(name: str) -> VariantProfile:
    profiles = {profile.name: profile for profile in variant_profiles()}
    try:
        return profiles[name]
    except KeyError as error:
        choices = ", ".join(sorted(profiles))
        raise ValueError(f"unknown morphology profile {name!r}; choose from {choices}") from error


def find_object_run_root(run_roots: Sequence[Path], object_name: str) -> Path:
    for run_root in run_roots:
        if (run_root / "anomaly_maps" / object_name).is_dir():
            return run_root
    message = f"no run root contains anomaly maps for {object_name}"
    raise RuntimeError(message)


def read_metrics(path: Path, object_name: str) -> SourceMetrics:
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload[object_name]
    return SourceMetrics(
        threshold=float(value["best_thre"]),
        auroc=float(value["seg_AUROC"]),
        f1=float(value["seg_F1"]),
    )


def load_samples(data_root: Path, run_root: Path, object_name: str) -> Tuple[MapSample, ...]:
    test_dir = run_root / "anomaly_maps" / object_name / "test"
    samples: List[MapSample] = []
    for split_dir in sorted(path for path in test_dir.iterdir() if path.is_dir()):
        for map_path in sorted(split_dir.glob("*.tiff")):
            score = np.asarray(tiff.imread(map_path), dtype=np.float32)
            samples.append(
                MapSample(
                    score=score,
                    gt_mask=load_gt_mask(
                        data_root,
                        object_name,
                        split_dir.name,
                        map_path.stem,
                        score.shape,
                    ),
                    split=split_dir.name,
                ),
            )
    return tuple(samples)


def load_gt_mask(
    data_root: Path,
    object_name: str,
    split: str,
    stem: str,
    shape: Sequence[int],
) -> BoolArray:
    if split == "good":
        return np.zeros(tuple(shape), dtype=np.bool_)
    path = data_root / object_name / "test_public" / "ground_truth" / split / f"{stem}_mask.png"
    mask = np.asarray(Image.open(path)) > 0
    if tuple(mask.shape) != tuple(shape):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (int(shape[1]), int(shape[0])),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
    return np.asarray(mask, dtype=np.bool_)


def candidate_thresholds(samples: Sequence[MapSample], threshold_count: int) -> Tuple[float, ...]:
    values = np.concatenate([sample.score.reshape(-1) for sample in samples])
    quantiles = np.linspace(0.01, 0.999, threshold_count, dtype=np.float32)
    thresholds = np.unique(np.quantile(values, quantiles).astype(np.float32))
    return tuple(float(threshold) for threshold in thresholds)


def best_variant(
    object_input: ObjectInput,
    profile: VariantProfile,
    thresholds: Sequence[float],
    config: EvalConfig,
) -> VariantRow:
    rows = tuple(
        evaluate_variant(object_input, profile, threshold, config)
        for threshold in thresholds
    )
    best = max(rows, key=lambda row: row.f1)
    return VariantRow(
        object_name=best.object_name,
        variant=f"{profile.name}_oracle_grid",
        threshold=best.threshold,
        f1=best.f1,
        precision=best.precision,
        recall=best.recall,
        positive_area=best.positive_area,
        source_seg_auroc=best.source_seg_auroc,
        source_seg_f1=best.source_seg_f1,
    )


def evaluate_variant(
    object_input: ObjectInput,
    profile: VariantProfile,
    threshold: float,
    config: EvalConfig,
) -> VariantRow:
    metrics = binary_mask_metrics(
        tuple(sample.score for sample in object_input.samples),
        tuple(sample.gt_mask for sample in object_input.samples),
        threshold,
        profile,
        config.line_length,
        config.angle_count,
    )
    return VariantRow(
        object_name=object_input.object_name,
        variant=f"{profile.name}_at_metrics_best",
        threshold=threshold,
        f1=metrics.f1,
        precision=metrics.precision,
        recall=metrics.recall,
        positive_area=metrics.positive_area,
        source_seg_auroc=object_input.metrics.auroc,
        source_seg_f1=object_input.metrics.f1,
    )


def binary_mask_metrics(
    scores: Sequence[FloatArray],
    gt_masks: Sequence[BoolArray],
    threshold: float,
    profile: VariantProfile,
    line_length: int = 17,
    angle_count: int = 16,
) -> BinaryMaskMetrics:
    if len(scores) != len(gt_masks):
        raise ValueError("scores and gt_masks must have the same length")
    tp = fp = fn = positive = total = 0
    for score, gt_mask in zip(scores, gt_masks):
        mask = score >= threshold
        if profile.use_morphology:
            mask = postprocess_mask(
                mask,
                line_length,
                angle_count,
                profile.erosion_size,
            )
        tp += int(np.count_nonzero(mask & gt_mask))
        fp += int(np.count_nonzero(mask & ~gt_mask))
        fn += int(np.count_nonzero(~mask & gt_mask))
        positive += int(np.count_nonzero(mask))
        total += int(mask.size)
    precision = 0.0 if tp + fp == 0 else float(tp / (tp + fp))
    recall = 0.0 if tp + fn == 0 else float(tp / (tp + fn))
    f1 = 0.0 if 2 * tp + fp + fn == 0 else float((2 * tp) / (2 * tp + fp + fn))
    return BinaryMaskMetrics(
        f1=f1,
        precision=precision,
        recall=recall,
        positive_area=0.0 if total == 0 else float(positive / total),
    )


def postprocess_mask(
    mask: BoolArray,
    line_length: int,
    angle_count: int,
    erosion_size: int,
) -> BoolArray:
    work = np.zeros(mask.shape, dtype=np.uint8)
    source = mask.astype(np.uint8)
    for index in range(angle_count):
        closed = np.asarray(
            cv2.morphologyEx(source, cv2.MORPH_CLOSE, line_kernel(line_length, index, angle_count)),
            dtype=np.uint8,
        )
        work = np.maximum(work, closed)
    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(work)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    if erosion_size > 0:
        kernel = np.ones((erosion_size, erosion_size), dtype=np.uint8)
        filled = cv2.erode(filled, kernel, iterations=1)
    return np.asarray(filled > 0, dtype=np.bool_)


def line_kernel(size: int, index: int, count: int) -> npt.NDArray[np.uint8]:
    kernel = np.zeros((size, size), dtype=np.uint8)
    center = (size - 1) / 2.0
    angle = np.pi * index / count
    dx = np.cos(angle) * center
    dy = np.sin(angle) * center
    start = (round(center - dx), round(center - dy))
    end = (round(center + dx), round(center + dy))
    cv2.line(kernel, start, end, 1, thickness=1)
    return kernel
