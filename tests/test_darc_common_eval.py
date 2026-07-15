from __future__ import annotations

# pyright: reportMissingImports=false
import math
import weakref
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pytest

from flow_tte.darc_common_eval import evaluate_object
from flow_tte.darc_common_report import (
    EvaluationRun,
    ObjectResult,
    RunMetadata,
    write_evaluation_outputs,
)
from flow_tte.darc_map_io import (
    ImageAudit,
    ImageRecord,
    ObjectMapSet,
    Population,
    audit_only,
)


def _audit_result_with_array_ref() -> tuple[ObjectResult, weakref.ReferenceType[np.ndarray]]:
    records = (
        *(
            _record(Population.GOOD, f"g{index}", [0.1], [0])
            for index in range(4)
        ),
        _record(Population.BAD, "b0", [0.9, 0.0], [1, 0]),
    )
    maps = ObjectMapSet(object_name="can", records=records)
    score_ref = weakref.ref(maps.records[0].score_map)
    return ObjectResult(audits=audit_only(maps), metrics=evaluate_object(maps)), score_ref


def _record(
    population: Population,
    stem: str,
    scores: list[float],
    labels: list[int],
) -> ImageRecord:
    shape = (1, len(scores))
    path = Path(f"/{population.value}/{stem}.tiff")
    audit = ImageAudit(
        image_id=f"{population.value}/{stem}",
        population=population,
        map_path=path,
        source_path=Path(f"/{population.value}/{stem}.png"),
        gt_path=None if population is Population.GOOD else Path(f"/{stem}_mask.png"),
        map_sha256=f"map-{stem}",
        source_sha256=f"source-{stem}",
        gt_sha256=None if population is Population.GOOD else f"gt-{stem}",
        original_map_shape=shape,
        original_gt_shape=None if population is Population.GOOD else shape,
        common_shape=shape,
    )
    return ImageRecord(
        audit=audit,
        score_map=np.asarray([scores], dtype=np.float32),
        gt_mask=np.asarray([labels], dtype=np.bool_),
    )


def test_evaluate_object_uses_tie_attainable_oracle_f1() -> None:
    # Given: tied positive and negative bad pixels above four normal maps.
    records = (
        *(_record(Population.GOOD, f"g{index}", [0.1], [0]) for index in range(4)),
        _record(Population.BAD, "b0", [0.5, 0.5], [1, 0]),
    )

    # When: the standardized continuous views are evaluated.
    result = evaluate_object(ObjectMapSet(object_name="can", records=records))

    # Then: a threshold cannot split equal scores into an artificial prefix.
    assert math.isclose(result.bad_only.p_ap, 0.5)
    assert math.isclose(result.bad_only.oracle_f1, 2.0 / 3.0)
    assert math.isclose(result.bad_only.p_auroc_005, 0.5)
    assert result.bad_only.oracle_component_recall == 1.0


def test_evaluate_object_sorts_once_for_both_continuous_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: both all-test and bad-only views with tied float32 scores.
    records = (
        *(_record(Population.GOOD, f"g{index}", [0.1, 0.5], [0, 0]) for index in range(4)),
        _record(Population.BAD, "b0", [0.9, 0.5, 0.0], [1, 0, 0]),
    )
    original_argsort = np.argsort
    sort_count = 0

    def counted_argsort(
        values: npt.ArrayLike,
        *,
        kind: Literal["mergesort"],
    ) -> npt.NDArray[np.intp]:
        nonlocal sort_count
        sort_count += 1
        return original_argsort(values, kind=kind)

    monkeypatch.setattr(np, "argsort", counted_argsort)

    # When: the common evaluator computes all continuous metrics.
    evaluate_object(ObjectMapSet(object_name="can", records=records))

    # Then: one stable score ordering serves both views and all three metrics.
    assert sort_count == 1


def test_evaluate_object_uses_standardized_partial_auroc() -> None:
    # Given: the registered eight-pixel score fixture with one anomalous pixel.
    records = (
        *(
            _record(Population.GOOD, f"g{index}", [score], [0])
            for index, score in enumerate((0.9, 0.4, 0.3, 0.2))
        ),
        _record(Population.BAD, "b0", [0.8, 0.7, 0.6, 0.5], [1, 0, 0, 0]),
    )

    # When: the all-test continuous population is evaluated.
    result = evaluate_object(ObjectMapSet(object_name="can", records=records))

    # Then: max_fpr=.05 has sklearn's standardized partial-AUC semantics.
    assert math.isclose(result.all_test.p_auroc_005, 0.48717948717948717)
    assert math.isclose(result.all_test.p_ap, 0.5)
    assert math.isclose(result.all_test.oracle_f1, 2.0 / 3.0)


def test_evaluate_object_crossfit_excludes_each_good_fold() -> None:
    # Given: four sorted normal images with a distinct maximum per fold.
    records = tuple(
        _record(Population.GOOD, f"g{index}", [score], [0])
        for index, score in enumerate((0.1, 0.2, 0.3, 0.9))
    ) + tuple(_record(Population.BAD, f"b{index}", [1.0, 0.0], [1, 0]) for index in range(4))

    # When: four-fold image-disjoint higher-quantile calibration is applied.
    result = evaluate_object(ObjectMapSet(object_name="can", records=records))

    # Then: each fold threshold excludes the normal image assigned to that fold.
    np.testing.assert_allclose(result.fixed_thresholds, (0.9, 0.9, 0.9, 0.3))
    assert result.good_count == 4
    assert result.bad_count == 4


def test_write_evaluation_outputs_records_protocol_and_artifact_hashes(
    tmp_path: Path,
) -> None:
    # Given: one evaluated object with auditable source and map identities.
    records = (
        *(_record(Population.GOOD, f"g{index}", [0.1], [0]) for index in range(4)),
        _record(Population.BAD, "b0", [0.9, 0.0], [1, 0]),
    )
    maps = ObjectMapSet(object_name="can", records=records)
    run = EvaluationRun(
        metadata=RunMetadata(
            data_root=Path("/data"),
            map_roots=(Path("/maps"),),
            output_root=tmp_path,
            method_label="DARC-G0",
            resource_label="P16-random",
            comparable=False,
        ),
        objects=(ObjectResult(audits=audit_only(maps), metrics=evaluate_object(maps)),),
    )

    assert not hasattr(run.objects[0], "maps")
    assert all(not hasattr(audit, "score_map") for audit in run.objects[0].audits.images)

    # When: the common evaluator writes its three registered artifacts.
    write_evaluation_outputs(run)

    # Then: metrics, tabular rows, and a provenance manifest all exist.
    metrics = (tmp_path / "per_object_metrics.json").read_text(encoding="utf-8")
    manifest = (tmp_path / "run_manifest.json").read_text(encoding="utf-8")
    assert '"method_label": "DARC-G0"' in metrics
    assert '"quantile_method": "higher"' in manifest
    assert '"reference": "scikit-learn 1.3.2 public metric APIs; no private API"' in manifest
    assert '"map_sha256": "map-b0"' in manifest
    assert '"common_shape": [' in manifest
    assert (tmp_path / "per_object_metrics.tsv").is_file()


def test_completed_object_result_does_not_retain_score_arrays() -> None:
    # Given / When: a completed object is reduced to the report representation.
    result, score_ref = _audit_result_with_array_ref()

    # Then: only immutable audit metadata survives beyond object evaluation.
    assert score_ref() is None
    assert not hasattr(result, "maps")
