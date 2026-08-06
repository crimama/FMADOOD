from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts import aggregate_flowtte_semantic_atlas_metric as aggregate

if TYPE_CHECKING:
    from pathlib import Path


def _variant(
    *,
    shift_reduction: float,
    defect_retention: float = 0.99,
    shift_energy_capture: float = 0.50,
    disabled_patch_mass: float = 0.05,
    random_seed_rows: list[dict[str, float]] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "shifted_normal_inflation": 1.0 - shift_reduction,
        "shift_reduction": shift_reduction,
        "defect_retention": defect_retention,
        "clean_mean_delta_std": 0.01,
        "clean_std_ratio": 1.0,
        "neighbor_retention": 0.98,
        "pair_spearman": 0.99,
        "shift_energy_capture": shift_energy_capture,
        "disabled_patch_mass": disabled_patch_mass,
        "folds": [
            {"fold": fold, "shift_reduction": shift_reduction, "source_disjoint": True}
            for fold in range(4)
        ],
        "validity": {
            "finite_outputs": True,
            "nonconstant_scores": True,
            "deterministic_repeat": True,
            "source_disjoint_folds": True,
            "support_only_custody": True,
            "min_eigenvalue": 0.75,
            "orthonormality_max_error": 1e-5,
        },
    }
    if random_seed_rows is not None:
        payload["random_seed_rows"] = random_seed_rows
    return payload


def _summary(parent: Path, object_name: str, *, a1_shift: float = 0.13) -> Path:
    path = parent / f"{object_name}.json"
    random_seed_rows = [
        {
            "shift_reduction": 0.01,
            "defect_retention": 1.0,
            "clean_mean_delta_std": 0.01,
            "clean_std_ratio": 1.0,
            "neighbor_retention": 0.99,
            "pair_spearman": 0.99,
            "shift_energy_capture": 0.07,
        },
        {
            "shift_reduction": 0.02,
            "defect_retention": 0.99,
            "clean_mean_delta_std": 0.02,
            "clean_std_ratio": 1.01,
            "neighbor_retention": 0.98,
            "pair_spearman": 0.98,
            "shift_energy_capture": 0.08,
        },
        {
            "shift_reduction": 0.03,
            "defect_retention": 0.98,
            "clean_mean_delta_std": 0.03,
            "clean_std_ratio": 0.99,
            "neighbor_retention": 0.97,
            "pair_spearman": 0.97,
            "shift_energy_capture": 0.09,
        },
    ]
    payload = {
        "protocol_id": aggregate.PROTOCOL_ID,
        "object": object_name,
        "support_manifest_sha256": aggregate.FROZEN_SUPPORT_MANIFEST_SHA256,
        "support": {
            "count": 16,
            "fold_counts": [4, 4, 4, 4],
            "source_disjoint_folds": True,
        },
        "config": {"density_weight": 0.0},
        "access_ledger": {
            "test_public_enumerated": False,
            "test_public_opened": False,
            "test_private_enumerated": False,
            "test_private_opened": False,
            "ground_truth_enumerated": False,
            "ground_truth_opened": False,
            "query_adaptation": False,
        },
        "atlas_robustness": {"cluster_ari": 0.80},
        "mechanism": {
            "variants": {
                "A0_baseline1_core": _variant(shift_reduction=0.0, shift_energy_capture=0.0),
                "A1_semantic_atlas_orth": _variant(
                    shift_reduction=a1_shift,
                    defect_retention=0.99,
                    shift_energy_capture=0.40,
                ),
                "A2_semantic_atlas_raw": _variant(
                    shift_reduction=a1_shift + 0.01,
                    defect_retention=0.94,
                    shift_energy_capture=0.35,
                ),
                "A3_global_rank4": _variant(
                    shift_reduction=0.04,
                    defect_retention=0.95,
                    shift_energy_capture=0.20,
                ),
                "A4_cluster_shuffle": _variant(
                    shift_reduction=0.02,
                    defect_retention=0.99,
                    shift_energy_capture=0.10,
                ),
                "A5_random_rank1": _variant(
                    shift_reduction=0.02,
                    defect_retention=0.99,
                    shift_energy_capture=0.08,
                    random_seed_rows=random_seed_rows,
                ),
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_semantic_atlas_pass_writes_all8_support_sentinel(tmp_path: Path) -> None:
    summaries = [_summary(tmp_path, object_name) for object_name in aggregate.OBJECTS]

    payload = aggregate.aggregate("mechanism", summaries, tmp_path / "agg")

    assert payload["verdict"] == "MECHANISM_PASS"
    assert payload["baseline_method"] == "Baseline1-core"
    assert payload["public_access_authorized"] is False
    assert payload["all8_support_authorized"] is True
    assert (tmp_path / "agg" / "all8_support_authorized.txt").is_file()


def test_random_q95_control_blocks_claim(tmp_path: Path) -> None:
    summaries = [_summary(tmp_path, object_name) for object_name in aggregate.OBJECTS]
    for path in summaries:
        payload = json.loads(path.read_text())
        payload["mechanism"]["variants"]["A5_random_rank1"]["random_seed_rows"] = [
            {
                "shift_reduction": 0.20,
                "defect_retention": 0.99,
                "clean_mean_delta_std": 0.01,
                "clean_std_ratio": 1.0,
                "neighbor_retention": 0.99,
                "pair_spearman": 0.99,
                "shift_energy_capture": 0.05,
            },
            {
                "shift_reduction": 0.22,
                "defect_retention": 0.99,
                "clean_mean_delta_std": 0.01,
                "clean_std_ratio": 1.0,
                "neighbor_retention": 0.99,
                "pair_spearman": 0.99,
                "shift_energy_capture": 0.06,
            },
        ]
        path.write_text(json.dumps(payload), encoding="utf-8")

    result = aggregate.aggregate("mechanism", summaries, tmp_path / "agg")

    assert result["verdict"] == "KILL_FOR_CLAIM"
    assert result["controls"]["random_q95_same_gate"] is True
    assert not (tmp_path / "agg" / "all8_support_authorized.txt").exists()


def test_shift_gain_with_bad_defect_retention_is_absorption_kill(tmp_path: Path) -> None:
    summaries = [_summary(tmp_path, object_name) for object_name in aggregate.OBJECTS]
    for path in summaries:
        payload = json.loads(path.read_text())
        payload["mechanism"]["variants"]["A1_semantic_atlas_orth"]["defect_retention"] = 0.80
        path.write_text(json.dumps(payload), encoding="utf-8")

    result = aggregate.aggregate("mechanism", summaries, tmp_path / "agg")

    assert result["verdict"] == "ANOMALY_ABSORPTION_KILL"
    assert result["controls"]["anomaly_absorption_indicated"] is True


def test_safe_five_to_ten_percent_shift_reduction_is_continue_diagnostic(
    tmp_path: Path,
) -> None:
    summaries = [
        _summary(tmp_path, object_name, a1_shift=0.07) for object_name in aggregate.OBJECTS
    ]

    result = aggregate.aggregate("mechanism", summaries, tmp_path / "agg")

    assert result["verdict"] == "CONTINUE_DIAGNOSTIC"
    assert result["all8_support_authorized"] is False


def test_forbidden_access_is_invalid(tmp_path: Path) -> None:
    summaries = [_summary(tmp_path, object_name) for object_name in aggregate.OBJECTS]
    payload = json.loads(summaries[0].read_text())
    payload["access_ledger"]["ground_truth_opened"] = True
    summaries[0].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(aggregate.AggregateError, match="ground_truth_opened"):
        aggregate.aggregate("mechanism", summaries, tmp_path / "agg")


def test_public_phase_is_forbidden(tmp_path: Path) -> None:
    summaries = [_summary(tmp_path, object_name) for object_name in aggregate.OBJECTS]

    with pytest.raises(aggregate.AggregateError, match="public phase is forbidden"):
        aggregate.aggregate("public", summaries, tmp_path / "agg")
