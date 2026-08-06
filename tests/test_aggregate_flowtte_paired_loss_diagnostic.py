from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts import aggregate_flowtte_paired_loss_diagnostic as aggregate

if TYPE_CHECKING:
    from pathlib import Path


def _summary(parent: Path, object_name: str, *, raw_reduction: float = 0.13) -> Path:
    path = parent / f"{object_name}.json"
    variants = {}
    for variant in aggregate.VARIANTS:
        reduction = raw_reduction if variant in aggregate.PROMOTION_CANDIDATES else 0.0
        variants[variant] = {
            "shifted_normal_inflation": 1.0 - reduction,
            "shift_reduction": reduction,
            "loo_delta": 0.01,
            "std_ratio": 1.0,
            "effective_rank_ratio": 1.0,
            "neighbor_retention": 1.0,
            "defect_retention": 1.0,
            "gate_passed": variant in aggregate.PROMOTION_CANDIDATES,
            "validity": {
                "inverse_reconstruction_max_error": 1e-6,
                "logdet_cancellation_max_error": 1e-5,
                "finite_gradients": True,
                "deterministic_repeat": True,
                "finite_outputs": True,
                "raw_euclidean_memory_scoring": True,
            },
        }
    payload = {
        "protocol_id": aggregate.PROTOCOL_ID,
        "phase": "mechanism",
        "object": object_name,
        "support_manifest_sha256": aggregate.FROZEN_SUPPORT_MANIFEST_SHA256,
        "support_count": 16,
        "checkpoint": str(parent / object_name / "checkpoint.pt"),
        "query_adaptation": False,
        "test_public_accessed": False,
        "access_ledger": {
            "test_public_enumerated": False,
            "test_public_opened": False,
            "test_private_opened": False,
            "ground_truth_opened": False,
        },
        "mechanism": {"variants": variants},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_raw_pair_pass_writes_all8_support_sentinel(tmp_path: Path) -> None:
    summaries = [_summary(tmp_path, object_name) for object_name in aggregate.OBJECTS]
    payload = aggregate.aggregate("mechanism", summaries, tmp_path / "agg")

    assert payload["verdict"] == "MECHANISM_PASS"
    assert payload["baseline_method"] == "Baseline1"
    assert payload["baseline_arm_id"] == "V0_default"
    assert payload["public_access_authorized"] is False
    assert payload["all8_support_authorized"] is True
    sentinel = tmp_path / "agg" / "all8_support_authorized.txt"
    assert sentinel.is_file()
    assert json.loads(sentinel.read_text())["public_access_authorized"] is False


def test_random_pair_null_blocks_mechanism_pass(tmp_path: Path) -> None:
    summaries = [_summary(tmp_path, object_name) for object_name in aggregate.OBJECTS]
    for path in summaries:
        payload = json.loads(path.read_text())
        payload["mechanism"]["variants"]["C2_random_pair"]["shift_reduction"] = 0.2
        payload["mechanism"]["variants"]["C2_random_pair"]["shifted_normal_inflation"] = 0.8
        path.write_text(json.dumps(payload), encoding="utf-8")

    result = aggregate.aggregate("mechanism", summaries, tmp_path / "agg")

    assert result["verdict"] == "KILL_FOR_CLAIM"
    assert result["controls"]["c2_random_pair_pass"] is True
    assert result["controls"]["random_null_blocks_claim"] is True
    assert not (tmp_path / "agg" / "all8_support_authorized.txt").exists()


def test_macro_gate_does_not_require_each_object_ten_percent(tmp_path: Path) -> None:
    summaries = [_summary(tmp_path, object_name) for object_name in aggregate.OBJECTS]
    payload = json.loads(summaries[0].read_text())
    for variant in aggregate.PROMOTION_CANDIDATES:
        payload["mechanism"]["variants"][variant]["shift_reduction"] = 0.02
        payload["mechanism"]["variants"][variant]["shifted_normal_inflation"] = 0.98
    summaries[0].write_text(json.dumps(payload), encoding="utf-8")

    result = aggregate.aggregate("mechanism", summaries, tmp_path / "agg")

    assert result["verdict"] == "MECHANISM_PASS"
    assert result["variant_gates"]["V2_raw_pair_nll1"]["improved_object_count"] == 4


def test_invalid_control_does_not_invalidate_valid_baseline_and_candidates(
    tmp_path: Path,
) -> None:
    summaries = [_summary(tmp_path, object_name) for object_name in aggregate.OBJECTS]
    for path in summaries:
        payload = json.loads(path.read_text())
        payload["mechanism"]["variants"]["V1_aug_nll"]["validity"][
            "finite_outputs"
        ] = False
        path.write_text(json.dumps(payload), encoding="utf-8")

    result = aggregate.aggregate("mechanism", summaries, tmp_path / "agg")

    assert result["verdict"] == "MECHANISM_PASS"
    assert result["baseline_validity_ok"] is True
    assert result["all_variant_validity_ok"] is False
    assert result["variant_gates"]["V1_aug_nll"]["validity_ok"] is False


def test_public_phase_is_forbidden(tmp_path: Path) -> None:
    summaries = [_summary(tmp_path, object_name) for object_name in aggregate.OBJECTS]

    with pytest.raises(aggregate.AggregateError, match="public phase is forbidden"):
        aggregate.aggregate("public", summaries, tmp_path / "agg")
