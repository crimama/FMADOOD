from __future__ import annotations

# ruff: noqa: TC003
import json
from pathlib import Path

from scripts import aggregate_shift_conditioned_patch_flow as aggregate


def _mechanism_summary(parent: Path, object_name: str, *, reduction: float = 0.10) -> Path:
    path = parent / f"{object_name}.json"
    payload = {
        "protocol_id": aggregate.PROTOCOL_ID,
        "phase": "mechanism",
        "object": object_name,
        "support_manifest_sha256": aggregate.FROZEN_SUPPORT_MANIFEST_SHA256,
        "checkpoint": str(parent / object_name / "checkpoint.pt"),
        "query_adaptation": False,
        "test_public_accessed": False,
        "access_ledger": {
            "test_public_enumerated": False,
            "test_private_opened": False,
        },
        "mechanism": {
            "validity": {
                "inverse_reconstruction_max_error": 1e-6,
                "logdet_cancellation_max_error": 1e-6,
                "e0_e1_latent_max_error": 1e-7,
                "e0_e1_logdet_max_error": 1e-7,
                "gradient_finite_nonzero": True,
                "deterministic_repeat": True,
                "descriptor_witness": True,
                "raw_euclidean_memory_scoring": True,
            },
            "held_shift": {
                "correct_reduction_fraction": reduction,
                "correct_beats_zero_shuffled": True,
            },
            "identity": {"loo_delta_e0_std": 0.02, "latent_std_ratio": 1.0},
            "defect_retention": {"gap_retention": 0.96},
            "shift_prediction": {"beats_zero": True, "loss": 0.5, "zero_loss": 1.0},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _public_summary(parent: Path, object_name: str) -> Path:
    path = parent / f"{object_name}_public.json"
    payload = {
        "protocol_id": aggregate.PROTOCOL_ID,
        "phase": "public",
        "object": object_name,
        "support_manifest_sha256": aggregate.FROZEN_SUPPORT_MANIFEST_SHA256,
        "query_adaptation": False,
        "test_public_accessed": True,
        "access_ledger": {"test_private_opened": False},
        "diagnostics": {
            "bad_image_gap_z_delta": 0.02,
            "shifted_good_p95_inflation_reduction": 0.08,
            "e0_e1_maps_equal": True,
            "finite_nonconstant_complete": True,
        },
        "metrics": {
            "E0": {"seg_AUROC_0.05": 0.70, "seg_F1": 0.40},
            "E1": {"seg_AUROC_0.05": 0.70, "seg_F1": 0.40},
            "E5": {"seg_AUROC_0.05": 0.701, "seg_F1": 0.39},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_mechanism_pass_writes_public_authorization_sentinel(tmp_path: Path) -> None:
    summaries = [_mechanism_summary(tmp_path, object_name) for object_name in aggregate.OBJECTS]
    payload = aggregate.aggregate("mechanism", summaries, tmp_path / "agg")

    assert payload["verdict"] == "MECHANISM_PASS"
    sentinel = tmp_path / "agg" / "mechanism_passed.txt"
    assert sentinel.is_file()
    assert json.loads(sentinel.read_text())["verdict"] == "MECHANISM_PASS"


def test_mechanism_retention_without_shift_claim_is_kill(tmp_path: Path) -> None:
    summaries = [
        _mechanism_summary(tmp_path, object_name, reduction=0.01)
        for object_name in aggregate.OBJECTS
    ]
    payload = aggregate.aggregate("mechanism", summaries, tmp_path / "agg")

    assert payload["verdict"] in {"KILL_FOR_CLAIM", "CONTINUE_DIAGNOSTIC"}
    assert not (tmp_path / "agg" / "mechanism_passed.txt").exists()


def test_public_aggregate_promising_gate(tmp_path: Path) -> None:
    summaries = [_public_summary(tmp_path, object_name) for object_name in aggregate.OBJECTS]
    payload = aggregate.aggregate("public", summaries, tmp_path / "agg")

    assert payload["verdict"] == "PROMISING_DIAGNOSTIC"
    assert payload["gates"]["e0_e1_parity_ok"] is True
    assert payload["gates"]["pauroc_no_harm_ok"] is True
