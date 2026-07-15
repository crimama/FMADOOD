"""Auditable JSON/TSV artifacts for the common DARC map evaluator."""

from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Mapping, Sequence, Tuple, Union

from flow_tte.darc_common_eval import ObjectEvaluation
from flow_tte.darc_map_io import ObjectAuditSet
from flow_tte.superadd_morphology import MorphologyConfig

if TYPE_CHECKING:
    from pathlib import Path

JsonScalar = Union[None, bool, int, float, str]
JsonValue = Union[JsonScalar, Sequence["JsonValue"], Mapping[str, "JsonValue"]]


@dataclass(frozen=True)  # noqa: RUF100 -- Python 3.8; # noqa: SLOTS_OK
class RunMetadata:
    data_root: Path
    map_roots: Tuple[Path, ...]
    output_root: Path
    method_label: str
    resource_label: str
    comparable: bool


@dataclass(frozen=True)  # noqa: RUF100 -- Python 3.8; # noqa: SLOTS_OK
class ObjectResult:
    audits: ObjectAuditSet
    metrics: ObjectEvaluation


@dataclass(frozen=True)  # noqa: RUF100 -- Python 3.8; # noqa: SLOTS_OK
class EvaluationRun:
    metadata: RunMetadata
    objects: Tuple[ObjectResult, ...]


def write_evaluation_outputs(run: EvaluationRun) -> None:
    """Write metrics, TSV, and provenance without mutating or deleting input maps."""
    run.metadata.output_root.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "schema_version": "darc-common-eval-v1",
        "method_label": run.metadata.method_label,
        "resource_label": run.metadata.resource_label,
        "comparable": run.metadata.comparable,
        "objects": [asdict(result.metrics) for result in run.objects],
    }
    _write_json(run.metadata.output_root / "per_object_metrics.json", metrics_payload)
    _write_tsv(run)
    manifest_payload = {
        "schema_version": "darc-common-eval-manifest-v1",
        "method_label": run.metadata.method_label,
        "resource_label": run.metadata.resource_label,
        "comparable": run.metadata.comparable,
        "data_root": str(run.metadata.data_root),
        "map_roots": [str(path) for path in run.metadata.map_roots],
        "population": "test_public/good+bad; bad-only also reported",
        "memory_policy": (
            "one object of arrays at a time; completed objects retain audit metadata only"
        ),
        "common_grid": {
            "target": "corresponding native test RGB HxW",
            "score_resize": "cv2.INTER_LINEAR (OpenCV half-pixel semantics)",
            "gt_resize": "cv2.INTER_NEAREST when required",
            "pre_metric_normalization": True,
        },
        "threshold_protocol": {
            "name": "four-fold-image-disjoint-good-normal-cross-fit",
            "interpretation": "diagnostic only; not a deployment threshold claim",
            "folds": 4,
            "assignment": "sorted IDs, round-robin separately for good and bad",
            "normal_quantile": 0.9999,
            "quantile_method": "higher",
            "calibration": "good images outside the sample fold",
            "raw_comparator": ">",
            "oracle_comparator": ">=",
        },
        "morphology": {
            "name": "shared-superadd-v1",
            **asdict(MorphologyConfig()),
        },
        "continuous_metrics": {
            "implementation": "one shared stable mergesort for all-test and bad-only",
            "reference": "scikit-learn 1.3.2 public metric APIs; no private API",
            "p_auroc_005": "sklearn standardized partial AUROC, max_fpr=0.05",
            "p_ap": "sklearn average_precision_score",
            "oracle_f1": "maximum over attainable precision-recall thresholds",
            "component_recall": "8-connected GT components hit at oracle threshold",
        },
        "objects": [_object_manifest(result) for result in run.objects],
        "inputs_deleted": False,
    }
    _write_json(run.metadata.output_root / "run_manifest.json", manifest_payload)


def _object_manifest(result: ObjectResult) -> Mapping[str, JsonValue]:
    image_rows: list[Mapping[str, JsonValue]] = [
        {
            "image_id": audit.image_id,
            "population": audit.population.value,
            "map_path": str(audit.map_path),
            "source_path": str(audit.source_path),
            "gt_path": None if audit.gt_path is None else str(audit.gt_path),
            "map_sha256": audit.map_sha256,
            "source_sha256": audit.source_sha256,
            "gt_sha256": audit.gt_sha256,
            "original_map_shape": audit.original_map_shape,
            "original_gt_shape": audit.original_gt_shape,
            "common_shape": audit.common_shape,
            "score_resized": audit.original_map_shape != audit.common_shape,
            "gt_resized": audit.original_gt_shape not in (None, audit.common_shape),
        }
        for audit in result.audits.images
    ]
    return {
        "object_name": result.audits.object_name,
        "good_count": result.metrics.good_count,
        "bad_count": result.metrics.bad_count,
        "fixed_thresholds_by_fold": result.metrics.fixed_thresholds,
        "map_set_sha256": _set_hash(
            tuple(f"{audit.image_id}:{audit.map_sha256}" for audit in result.audits.images),
        ),
        "source_set_sha256": _set_hash(
            tuple(
                f"{audit.image_id}:{audit.source_sha256}:{audit.gt_sha256 or ''}"
                for audit in result.audits.images
            ),
        ),
        "images": image_rows,
    }


def _set_hash(entries: Tuple[str, ...]) -> str:
    joined = "\n".join(sorted(entries)).encode()
    return hashlib.sha256(joined).hexdigest()


def _write_tsv(run: EvaluationRun) -> None:
    path = run.metadata.output_root / "per_object_metrics.tsv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            (
                "method_label",
                "resource_label",
                "comparable",
                "object",
                "population",
                "p_auroc_005",
                "p_ap",
                "oracle_f1",
                "oracle_threshold",
                "oracle_component_recall",
                "fixed_raw_f1",
                "fixed_morphology_f1",
            ),
        )
        for result in run.objects:
            for population, metrics in (
                ("all_test", result.metrics.all_test),
                ("bad_only", result.metrics.bad_only),
            ):
                writer.writerow(
                    (
                        run.metadata.method_label,
                        run.metadata.resource_label,
                        str(run.metadata.comparable).lower(),
                        result.metrics.object_name,
                        population,
                        metrics.p_auroc_005,
                        metrics.p_ap,
                        metrics.oracle_f1,
                        metrics.oracle_threshold,
                        metrics.oracle_component_recall,
                        metrics.fixed_raw_f1,
                        metrics.fixed_morphology_f1,
                    ),
                )


def _write_json(path: Path, payload: Mapping[str, JsonValue]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
