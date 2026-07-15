from __future__ import annotations

# pyright: reportMissingImports=false
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import pytest

from flow_tte.darc_gate2_evaluation import (
    compute_fold_component_thresholds,
    evaluate_source_maps,
)
from flow_tte.darc_gate2_evaluation_types import (
    FoldCleanCalibration,
    RungSourceMaps,
    SourceEvaluationInput,
    SourceMapBundle,
    SourceMasks,
)
from flow_tte.darc_gate2_metrics_types import InvalidGateInput, SourceAudit

FloatMap = npt.NDArray[np.float32]
BoolMask = npt.NDArray[np.bool_]


def _map(fill: float = 0.0) -> FloatMap:
    return np.full((3, 3), fill, dtype=np.float32)


def _masks() -> SourceMasks:
    first = np.zeros((3, 3), dtype=np.bool_)
    first[0, 0] = True
    second = np.zeros((3, 3), dtype=np.bool_)
    second[2, 2] = True
    broad = np.zeros((3, 3), dtype=np.bool_)
    broad[1, 1:3] = True
    return SourceMasks(thin_profiles=(first, second), broad=broad)


def _rung(
    clean_value: float,
    thin_values: Tuple[float, float],
    broad_positive: float,
) -> RungSourceMaps:
    masks = _masks()
    thin_maps: List[FloatMap] = []
    for mask, value in zip(masks.thin_profiles, thin_values):
        score_map = _map()
        score_map[mask] = value
        thin_maps.append(score_map)
    broad = _map()
    broad[masks.broad] = broad_positive
    return RungSourceMaps(
        clean=_map(clean_value),
        thin_profiles=tuple(thin_maps),
        broad=broad,
    )


def _calibration() -> FoldCleanCalibration:
    clean_maps = tuple(_map(0.5) for _ in range(4))
    return FoldCleanCalibration(l0=clean_maps, l1=clean_maps)


def _input() -> SourceEvaluationInput:
    l0 = _rung(0.1, (0.4, 0.4), 0.0)
    for score_map in l0.thin_profiles:
        score_map[0, 1] = 0.45
    l1 = _rung(0.2, (0.6, 0.8), 1.0)
    r1 = _rung(0.3, (0.1, 0.8), 1.0)
    r1.thin_profiles[0][1, 2] = 99.0
    return SourceEvaluationInput(
        object_name="bottle",
        seed=0,
        fold_index=2,
        source_id="source-01",
        maps=SourceMapBundle(l0=l0, l1=l1, r1=r1),
        masks=_masks(),
        calibration=_calibration(),
        audit=SourceAudit("population", "support", "fallback", "mask"),
    )


def test_evaluate_source_maps_builds_paired_metric_and_detailed_manifest() -> None:
    source = _input()

    result = evaluate_source_maps(source)

    assert result.metric.d_ap == pytest.approx(0.5)
    assert result.metric.d_component_recall == pytest.approx(1.0)
    assert result.metric.l1_responses == pytest.approx((0.4, 0.6))
    assert result.metric.r1_responses == pytest.approx((-0.2, 0.5))
    assert result.metric.broad_pauroc_delta == pytest.approx(0.5)
    assert result.metric.l0_audit is source.audit
    assert result.metric.l1_audit is source.audit
    assert result.metric.r1_audit is source.audit

    manifest = result.manifest
    assert manifest.thresholds.l0 == pytest.approx(0.5)
    assert manifest.thresholds.l1 == pytest.approx(0.5)
    assert manifest.l0_ap == pytest.approx(0.5)
    assert manifest.l1_ap == pytest.approx(1.0)
    assert manifest.l0_component_recall == pytest.approx(0.0)
    assert manifest.l1_component_recall == pytest.approx(1.0)
    assert manifest.l1_profiles[0].clean_mean == pytest.approx(0.2)
    assert manifest.l1_profiles[0].cue_mean == pytest.approx(0.6)
    assert manifest.r1_profiles[0].response == pytest.approx(-0.2)
    assert manifest.broad_l0_pauroc_005 == pytest.approx(0.5)
    assert manifest.broad_r1_pauroc_005 == pytest.approx(1.0)
    assert manifest.to_manifest()["thresholds"] == {"l0": 0.5, "l1": 0.5}


def test_component_threshold_uses_four_clean_maps_and_higher_boundary() -> None:
    l0 = tuple(_map(value) for value in (0.1, 0.2, 0.3, 0.4))
    l1 = tuple(_map(value) for value in (0.2, 0.3, 0.4, 0.5))

    thresholds = compute_fold_component_thresholds(FoldCleanCalibration(l0=l0, l1=l1))

    assert thresholds.l0 == pytest.approx(0.4)
    assert thresholds.l1 == pytest.approx(0.5)


def test_component_quantile_distinguishes_p9999_higher_from_p999() -> None:
    l0_maps: List[FloatMap] = []
    l1_maps: List[FloatMap] = []
    for map_index in range(4):
        l0_map = np.empty((50, 50), dtype=np.float32)
        l1_map = np.empty((50, 50), dtype=np.float32)
        for pixel_index in range(2_500):
            row, column = divmod(pixel_index, 50)
            value = float(map_index * 2_500 + pixel_index)
            l0_map[row, column] = value
            l1_map[row, column] = value + 10_000.0
        l0_maps.append(l0_map)
        l1_maps.append(l1_map)

    thresholds = compute_fold_component_thresholds(
        FoldCleanCalibration(l0=tuple(l0_maps), l1=tuple(l1_maps)),
    )

    assert thresholds.l0 == pytest.approx(9_999.0)
    assert thresholds.l1 == pytest.approx(19_999.0)


def test_ap_concatenates_both_differently_ranked_thin_profiles() -> None:
    source = _input()
    first = _map()
    first[source.masks.thin_profiles[0]] = 1.0
    second = _map()
    second[source.masks.thin_profiles[1]] = 0.4
    second[0, 1] = 0.9
    l0 = source.maps.l0._replace(thin_profiles=(first, second))

    manifest = evaluate_source_maps(
        source._replace(maps=source.maps._replace(l0=l0)),
    ).manifest

    assert manifest.l0_ap == pytest.approx(5.0 / 6.0)
    assert manifest.l0_ap != pytest.approx(1.0)
    assert manifest.l0_ap != pytest.approx(0.5)


def test_broad_diagnostic_uses_standardized_pauroc_at_point_zero_five() -> None:
    source = _input()
    broad_mask = np.zeros((3, 3), dtype=np.bool_)
    broad_mask[0, 0] = True
    l0_broad = _map()
    l0_broad[0, 0] = 0.8
    l0_broad[0, 1] = 0.9
    r1_broad = _map()
    r1_broad[0, 0] = 1.0
    diagnostic_input = source._replace(
        masks=source.masks._replace(broad=broad_mask),
        maps=source.maps._replace(
            l0=source.maps.l0._replace(broad=l0_broad),
            r1=source.maps.r1._replace(broad=r1_broad),
        ),
    )

    manifest = evaluate_source_maps(diagnostic_input).manifest

    assert manifest.broad_l0_pauroc_005 == pytest.approx(0.48717948717948717)
    assert manifest.broad_l0_pauroc_005 != pytest.approx(0.875)
    assert manifest.broad_pauroc_delta == pytest.approx(0.5128205128205128)


def test_component_predictions_are_strictly_greater_than_threshold() -> None:
    source = _input()
    l0 = _rung(0.1, (0.5, 0.5), 0.0)
    l1 = _rung(0.2, (0.5001, 0.5001), 1.0)
    strict_input = source._replace(
        maps=SourceMapBundle(l0=l0, l1=l1, r1=source.maps.r1),
    )

    manifest = evaluate_source_maps(strict_input).manifest

    assert manifest.l0_component_recall == pytest.approx(0.0)
    assert manifest.l1_component_recall == pytest.approx(1.0)


def test_component_recall_uses_exact_eight_connected_masks() -> None:
    source = _input()
    diagonal = np.zeros((3, 3), dtype=np.bool_)
    diagonal[0, 0] = True
    diagonal[1, 1] = True
    second = source.masks.thin_profiles[1]
    l1_first = _map()
    l1_first[0, 0] = 0.6
    l1_second = _map()
    l1_second[second] = 0.6
    exact_input = source._replace(
        masks=SourceMasks(thin_profiles=(diagonal, second), broad=source.masks.broad),
        maps=SourceMapBundle(
            l0=source.maps.l0,
            l1=source.maps.l1._replace(thin_profiles=(l1_first, l1_second)),
            r1=source.maps.r1,
        ),
    )

    manifest = evaluate_source_maps(exact_input).manifest

    assert manifest.l1_component_recall == pytest.approx(1.0)


@pytest.mark.parametrize("count", [3, 5])
def test_rejects_any_calibration_count_other_than_four(count: int) -> None:
    clean_maps = tuple(_map(0.5) for _ in range(count))

    with pytest.raises(InvalidGateInput, match="exactly four"):
        compute_fold_component_thresholds(FoldCleanCalibration(l0=clean_maps, l1=clean_maps))


def test_rejects_any_thin_profile_count_other_than_two() -> None:
    source = _input()
    short_l0 = source.maps.l0._replace(thin_profiles=source.maps.l0.thin_profiles[:1])

    with pytest.raises(InvalidGateInput, match="exactly two"):
        evaluate_source_maps(
            source._replace(maps=source.maps._replace(l0=short_l0)),
        )


def test_rejects_empty_boolean_masks() -> None:
    source = _input()
    empty = np.zeros((3, 3), dtype=np.bool_)

    with pytest.raises(InvalidGateInput, match="positive pixel"):
        evaluate_source_maps(
            source._replace(
                masks=source.masks._replace(
                    thin_profiles=(empty, source.masks.thin_profiles[1]),
                ),
            ),
        )


def test_rejects_nonfinite_maps() -> None:
    source = _input()
    nonfinite = _map()
    nonfinite[0, 0] = np.nan

    with pytest.raises(InvalidGateInput, match="finite"):
        evaluate_source_maps(
            source._replace(
                maps=source.maps._replace(
                    l1=source.maps.l1._replace(broad=nonfinite),
                ),
            ),
        )


def test_rejects_shape_mismatch() -> None:
    source = _input()
    wrong_shape = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(InvalidGateInput, match="identical shapes"):
        evaluate_source_maps(
            source._replace(
                maps=source.maps._replace(
                    r1=source.maps.r1._replace(clean=wrong_shape),
                ),
            ),
        )


def test_evaluation_does_not_mutate_caller_owned_arrays() -> None:
    source = _input()
    arrays = (
        source.maps.l0.clean,
        *source.maps.l0.thin_profiles,
        source.maps.l0.broad,
        source.maps.l1.clean,
        *source.maps.l1.thin_profiles,
        source.maps.l1.broad,
        source.maps.r1.clean,
        *source.maps.r1.thin_profiles,
        source.maps.r1.broad,
        *source.masks.thin_profiles,
        source.masks.broad,
        *source.calibration.l0,
        *source.calibration.l1,
    )
    snapshots: Tuple[bytes, ...] = tuple(value.tobytes() for value in arrays)

    evaluate_source_maps(source)

    assert tuple(value.tobytes() for value in arrays) == snapshots
