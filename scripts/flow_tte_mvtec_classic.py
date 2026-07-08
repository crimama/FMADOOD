from __future__ import annotations

# pyright: reportMissingImports=false
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Final, List, Optional, Tuple, Union

if __package__:
    from .flow_tte_map_metrics import compute_map_metric_set
else:
    from flow_tte_map_metrics import compute_map_metric_set

_IMAGE_SUFFIXES: Final = (".png", ".jpg", ".jpeg", ".bmp")

JsonScalar = Union[str, int, float, bool, None]
JsonValue = Union[JsonScalar, Dict[str, "JsonValue"], List["JsonValue"]]


@dataclass(frozen=True)
class ClassicObjectInfo:
    name: str
    anomaly_types: Tuple[str, ...]
    resolution: int
    masking: bool = False
    rotation: bool = False


@dataclass(frozen=True)
class EvaluationFileLists:
    gt_filenames: Tuple[Optional[str], ...]
    prediction_filenames: Tuple[str, ...]
    sample_count: int


@dataclass(frozen=True)
class ClassicEvaluationConfig:
    dataset: "ClassicMVTecDataset"
    output_root: Path
    objects: Tuple[str, ...]
    pro_integration_limit: float
    seed: int
    image_top_fraction: float = 0.01


@dataclass(frozen=True)
class ClassicMVTecDataset:
    data_root: str
    objects: Tuple[str, ...]
    resolution: int = 448

    def get_objects(self) -> List[ClassicObjectInfo]:
        return [self.get_object_info(object_name) for object_name in self.objects]

    def get_object_info(self, object_name: str) -> ClassicObjectInfo:
        anomaly_types = tuple(
            path.name
            for path in sorted((self._object_dir(object_name) / "test").iterdir())
            if path.is_dir() and path.name != "good"
        )
        return ClassicObjectInfo(
            name=object_name,
            anomaly_types=anomaly_types,
            resolution=self.resolution,
        )

    def get_train_images(self, object_name: str) -> List[str]:
        train_dir = self._object_dir(object_name) / "train" / "good"
        return [str(path) for path in _image_files(train_dir)]

    def get_test_images(
        self,
        object_name: str,
        split: str = "test",
    ) -> Dict[str, List[str]]:
        _ = split
        test_dir = self._object_dir(object_name) / "test"
        if not test_dir.is_dir():
            message = f"Test directory not found: {test_dir}"
            raise FileNotFoundError(message)
        result: Dict[str, List[str]] = {}
        for anomaly_dir in sorted(path for path in test_dir.iterdir() if path.is_dir()):
            images = [str(path) for path in _image_files(anomaly_dir)]
            if images:
                result[anomaly_dir.name] = images
        return result

    def get_ground_truth_dir(self, object_name: str, split: str = "test") -> Optional[str]:
        _ = split
        gt_dir = self._object_dir(object_name) / "ground_truth"
        if gt_dir.is_dir():
            return str(gt_dir)
        return None

    def ground_truth_path(self, object_name: str, anomaly_type: str, stem: str) -> Optional[Path]:
        if anomaly_type == "good":
            return None
        return self._object_dir(object_name) / "ground_truth" / anomaly_type / f"{stem}_mask.png"

    def _object_dir(self, object_name: str) -> Path:
        object_dir = Path(self.data_root) / object_name
        if not object_dir.is_dir():
            message = f"Object directory not found: {object_dir}"
            raise FileNotFoundError(message)
        return object_dir


def _image_files(directory: Path) -> Tuple[Path, ...]:
    if not directory.is_dir():
        message = f"Image directory not found: {directory}"
        raise FileNotFoundError(message)
    return tuple(
        sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        ),
    )


def build_evaluation_file_lists(
    dataset: ClassicMVTecDataset,
    output_root: Path,
    object_name: str,
) -> EvaluationFileLists:
    gt_paths: List[Optional[str]] = []
    prediction_paths: List[str] = []
    test_images = dataset.get_test_images(object_name)
    for anomaly_type, images in test_images.items():
        for image_path in images:
            stem = Path(image_path).stem
            gt_path = dataset.ground_truth_path(object_name, anomaly_type, stem)
            pred_path = output_root / "anomaly_maps" / object_name / "test" / anomaly_type / stem
            gt_paths.append(str(gt_path) if gt_path is not None else None)
            prediction_paths.append(str(pred_path))
    return EvaluationFileLists(
        gt_filenames=tuple(gt_paths),
        prediction_filenames=tuple(prediction_paths),
        sample_count=len(prediction_paths),
    )


def evaluate_classic_mvtec(config: ClassicEvaluationConfig) -> Dict[str, JsonValue]:
    from src.post_eval import eval_segmentation  # noqa: PLC0415

    per_object: Dict[str, JsonValue] = {}
    aurocs: List[float] = []
    f1s: List[float] = []
    image_aurocs: List[float] = []
    pixel_aurocs: List[float] = []
    image_aps: List[float] = []
    pixel_aps: List[float] = []
    pixel_pros: List[float] = []
    image_score_aggregation: Optional[str] = None
    for object_name in config.objects:
        files = build_evaluation_file_lists(config.dataset, config.output_root, object_name)
        auroc, f1, threshold = eval_segmentation(
            files.gt_filenames,
            files.prediction_filenames,
            pro_integration_limit=config.pro_integration_limit,
            delete_tiff_files=False,
        )
        map_metrics = compute_map_metric_set(
            files.gt_filenames,
            files.prediction_filenames,
            image_top_fraction=config.image_top_fraction,
        )
        image_score_aggregation = map_metrics.image_score_aggregation
        aurocs.append(float(auroc))
        f1s.append(float(f1))
        image_aurocs.append(map_metrics.image_auroc)
        pixel_aurocs.append(map_metrics.pixel_auroc)
        image_aps.append(map_metrics.image_ap)
        pixel_aps.append(map_metrics.pixel_ap)
        pixel_pros.append(map_metrics.pixel_pro)
        per_object[object_name] = {
            "seg_AUROC_0.05": float(auroc),
            "seg_F1": float(f1),
            "best_threshold": float(threshold),
            "sample_count": files.sample_count,
            **map_metrics.as_dict(),
        }
    if image_score_aggregation is None:
        raise ValueError("At least one object is required for classic MVTec evaluation")
    metrics: Dict[str, JsonValue] = {
        "dataset": "MVTec AD1 classic",
        "objects": list(config.objects),
        "seed": config.seed,
        "pro_integration_limit": config.pro_integration_limit,
        "seg_AUROC_0.05": float(sum(aurocs) / len(aurocs)),
        "seg_F1": float(sum(f1s) / len(f1s)),
        "image_AUROC": float(sum(image_aurocs) / len(image_aurocs)),
        "pixel_AUROC": float(sum(pixel_aurocs) / len(pixel_aurocs)),
        "image_AP": float(sum(image_aps) / len(image_aps)),
        "pixel_AP": float(sum(pixel_aps) / len(pixel_aps)),
        "pixel_PRO": float(sum(pixel_pros) / len(pixel_pros)),
        "image_score_aggregation": image_score_aggregation,
        "pixel_score_quantization": "float16_histogram",
        "pixel_PRO_max_fpr": 0.30,
        "per_object": per_object,
    }
    write_metrics(config.output_root, config.seed, metrics)
    return metrics


def write_metrics(output_root: Path, seed: int, metrics: Dict[str, JsonValue]) -> None:
    payload = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    (output_root / "metrics.json").write_text(payload, encoding="utf-8")
    (output_root / f"metrics_seed={seed}.json").write_text(payload, encoding="utf-8")
