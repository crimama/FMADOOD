# /// script
# requires-python = ">=3.8"
# dependencies = []
# ///
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Final, List, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from typing_extensions import Literal

ContextSource = Literal["cls", "register", "cls_register"]
SplitName = Literal["good", "bad"]
JsonScalar = Union[str, int, float, bool, None]
JsonValue = Union[JsonScalar, Dict[str, "JsonValue"], List["JsonValue"]]
FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]
CONTEXT_SOURCES: Final[Tuple[ContextSource, ...]] = ("cls", "register", "cls_register")
_EPS: Final = 1e-12


@dataclass(frozen=True)
class AnalysisConfig:
    data_root: Path
    output_root: Path
    project_root: Path
    fsad_root: Path
    support_json: Path
    objects: Tuple[str, ...]
    device: str
    seed: int
    patch_samples_per_image: int
    context_top_m: int


@dataclass(frozen=True)
class ContextSet:
    cls: FloatArray
    register: FloatArray
    cls_register: FloatArray


@dataclass(frozen=True)
class ImageBundle:
    features: FloatArray
    contexts: ContextSet


@dataclass(frozen=True)
class SupportBundle:
    features: FloatArray
    register_patch_contexts: FloatArray
    group_ids: IntArray
    contexts: ContextSet


@dataclass(frozen=True)
class TestItem:
    anomaly_type: str
    path: Path

    @property
    def split(self) -> SplitName:
        return "good" if self.anomaly_type == "good" else "bad"


@dataclass(frozen=True)
class BinaryDistanceSummary:
    good_mean: float
    bad_mean: float
    delta_bad_good: float
    good_count: int
    bad_count: int


@dataclass(frozen=True)
class LatentVolumeSummary:
    mean_variance: float
    mean_log_variance: float
    effective_rank: float


def context_values(contexts: ContextSet, source: ContextSource) -> FloatArray:
    values: Dict[ContextSource, FloatArray] = {
        "cls": contexts.cls,
        "register": contexts.register,
        "cls_register": contexts.cls_register,
    }
    return values[source]


def cosine_distance_matrix(query: FloatArray, reference: FloatArray) -> FloatArray:
    if query.ndim != 2 or reference.ndim != 2:
        message = "cosine_distance_matrix expects 2D arrays"
        raise RuntimeError(message)
    if query.shape[1] != reference.shape[1]:
        message = "cosine_distance_matrix dimension mismatch"
        raise RuntimeError(message)
    query_norm = l2_normalize(query)
    reference_norm = l2_normalize(reference)
    return (1.0 - query_norm @ reference_norm.T).astype(np.float32, copy=False)


def summarize_good_bad_distances(
    good_values: FloatArray,
    bad_values: FloatArray,
) -> BinaryDistanceSummary:
    good_mean = mean_or_nan(good_values)
    bad_mean = mean_or_nan(bad_values)
    return BinaryDistanceSummary(
        good_mean=good_mean,
        bad_mean=bad_mean,
        delta_bad_good=bad_mean - good_mean,
        good_count=int(good_values.size),
        bad_count=int(bad_values.size),
    )


def latent_volume_summary(features: FloatArray) -> LatentVolumeSummary:
    if features.ndim != 2:
        message = "latent_volume_summary expects a 2D array"
        raise RuntimeError(message)
    if features.shape[0] == 0:
        message = "latent_volume_summary expects at least one feature"
        raise RuntimeError(message)
    values = features.astype(np.float32, copy=False)
    variance = np.asarray(np.var(values, axis=0), dtype=np.float32)
    safe_variance = np.maximum(variance, np.float32(_EPS)).astype(np.float32, copy=False)
    weights = (safe_variance / np.sum(safe_variance)).astype(np.float32, copy=False)
    entropy = -float(np.sum(weights * np.log(np.maximum(weights, np.float32(_EPS)))))
    return LatentVolumeSummary(
        mean_variance=float(np.mean(safe_variance)),
        mean_log_variance=float(np.mean(np.log(safe_variance))),
        effective_rank=float(np.exp(entropy)),
    )


def summarize_list(values: Sequence[float]) -> Tuple[int, float, float, float]:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return 0, float("nan"), float("nan"), float("nan")
    return (
        int(array.size),
        float(np.mean(array)),
        float(np.std(array)),
        float(np.percentile(array, 95)),
    )


def load_support_paths(path: Path) -> Dict[str, Tuple[Path, ...]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        message = f"support json must be an object: {path}"
        raise TypeError(message)
    result: Dict[str, Tuple[Path, ...]] = {}
    for object_name, values in raw.items():
        if not isinstance(object_name, str) or not isinstance(values, list):
            message = f"invalid support json entry: {object_name}"
            raise TypeError(message)
        result[object_name] = tuple(Path(item) for item in values if isinstance(item, str))
        if len(result[object_name]) != len(values):
            message = f"support paths must be strings for object: {object_name}"
            raise TypeError(message)
    return result


def write_tsv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def l2_normalize(values: FloatArray) -> FloatArray:
    norms = np.maximum(np.linalg.norm(values, axis=1, keepdims=True), np.float32(_EPS))
    return (values.astype(np.float32, copy=False) / norms).astype(np.float32, copy=False)


def mean_or_nan(values: FloatArray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))
