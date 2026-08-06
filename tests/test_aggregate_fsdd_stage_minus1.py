from __future__ import annotations

# ruff: noqa: ANN202, SLF001
import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from types import ModuleType


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_fsdd_stage_minus1.py"
    spec = importlib.util.spec_from_file_location("aggregate_fsdd_stage_minus1", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _variant(
    *,
    shift: float,
    pauc: float,
    f1: float,
    clean_fpr: float = 0.01,
) -> dict[str, object]:
    return {
        "threshold": 0.75,
        "threshold_quantile": 0.995,
        "clean_fpr": clean_fpr,
        "primary_shift_fpr": {"x1": clean_fpr + shift, "x2": clean_fpr + shift},
        "clean": {"pAUROC_0.05": pauc, "fixed_F1": f1, "oracle_F1": f1 + 0.01},
        "shifted": {"pAUROC_0.05": pauc - 0.005, "fixed_F1": f1 - 0.005, "oracle_F1": f1 + 0.005},
        "interior": {
            "clean": {"pAUROC_0.05": pauc, "fixed_F1": f1},
            "shifted": {"pAUROC_0.05": pauc - 0.005, "fixed_F1": f1 - 0.005},
        },
        "coverage": {"complete": True, "finite": True, "nonconstant": True, "aligned": True},
    }


def _normalized_records(module: ModuleType) -> list[dict[str, object]]:
    records = []
    for object_name in module.OBJECTS:
        for seed in module.SEEDS:
            for source_index in range(2):
                source = f"{object_name}/train/good/{source_index}.png"
                for family in module.FAMILIES:
                    for axis in module.AXES:
                        variants = {
                            variant: _variant(
                                shift=0.02 if variant == "treatment" else 0.04,
                                pauc=0.94 if variant == "treatment" else 0.92,
                                f1=0.84 if variant == "treatment" else 0.81,
                            )
                            for variant in module.VARIANTS
                        }
                        normalized = {
                            variant: module._variant_record(value, variant)
                            for variant, value in variants.items()
                        }
                        records.append(
                            {
                                "object": object_name,
                                "seed": seed,
                                "source": source,
                                "source_fold": source_index,
                                "family": family,
                                "axis": axis,
                                "variants": normalized,
                            },
                        )
    return records


def _payload(module: ModuleType) -> dict[str, object]:
    support = [
        {
            "relative_path": f"can/train/good/{index:02d}.png",
            "fold": index // 4,
        }
        for index in range(16)
    ]
    head_records = [
        {
            "source": item["relative_path"],
            "source_fold": item["fold"],
            "heldout_family": family,
            "heldout_axis": axis,
            "variants": {
                variant: _variant(
                    shift=0.02 if variant == "treatment" else 0.04,
                    pauc=0.94 if variant == "treatment" else 0.92,
                    f1=0.84 if variant == "treatment" else 0.81,
                )
                for variant in module.VARIANTS
            },
        }
        for item in support
        for family in module.FAMILIES
        for axis in module.AXES
    ]
    stimulus_audit = [
        {
            "kind": "feature_visible_defect",
            "source": item["relative_path"],
            "family": family,
            "interior_pixels": 8,
            "boundary_pixels": 4,
            "clean": {"valid": True},
            "shifted": {
                f"{axis}_x{scale}": {"valid": True}
                for axis in module.AXES
                for scale in (1, 2)
            },
        }
        for item in support
        for family in module.FAMILIES
    ]
    return {
        "schema": "fsdd-lodf-stage-minus1-object-v1",
        "protocol_id": module.PROTOCOL_ID,
        "object": "can",
        "experiment_seed": 0,
        "support_manifest_sha256": module.SUPPORT_SHA256,
        "support": support,
        "head_records": head_records,
        "stimulus_audit": stimulus_audit,
        "access_audit": {
            "test_public_accessed": False,
            "test_private_accessed": False,
            "ground_truth_accessed": False,
            "query_update": False,
            "query_transform": False,
            "tta": False,
            "forbidden_path_hits": [],
        },
    }


def test_strict_gate_passes_two_seed_positive_fixture() -> None:
    module = _module()
    records = _normalized_records(module)
    lcbs = {
        "shift_gain_vs_raw_state": 0.01,
        "shift_gain_vs_nonfactorial": 0.01,
        "clean_pAUROC_0.05_hard_null_margin": 0.01,
        "clean_fixed_F1_hard_null_margin": 0.01,
        "shifted_pAUROC_0.05_hard_null_margin": 0.01,
        "shifted_fixed_F1_hard_null_margin": 0.01,
    }
    audits = [
        {
            "support_ok": True,
            "access_ok": True,
            "feature_visibility_ok": True,
            "lodf_losf_ok": True,
            "maps_ok": True,
        },
    ]
    gates = module._gates(records, audits, lcbs)
    assert gates["strict"] is True
    assert gates["G7_ribbon_and_ribbon_excluded"] is True
    assert gates["soft"] is True
    json.dumps(module._json_safe(gates), allow_nan=False)


def test_one_seed_directional_failure_blocks_strict() -> None:
    module = _module()
    records = _normalized_records(module)
    for record in records:
        if record["seed"] == 1:
            record["variants"]["treatment"]["shift_fpr_excess"] = 0.05
    lcbs = dict.fromkeys(
        (
            "shift_gain_vs_raw_state",
            "shift_gain_vs_nonfactorial",
            "clean_pAUROC_0.05_hard_null_margin",
            "clean_fixed_F1_hard_null_margin",
            "shifted_pAUROC_0.05_hard_null_margin",
            "shifted_fixed_F1_hard_null_margin",
        ),
        0.01,
    )
    audits = [
        dict.fromkeys(
            ("support_ok", "access_ok", "feature_visibility_ok", "lodf_losf_ok", "maps_ok"),
            True,
        ),
    ]
    gates = module._gates(records, audits, lcbs)
    assert gates["seed_directional_gates"][1]["shift"] is False
    assert gates["strict"] is False


@pytest.mark.parametrize("subset", ["ribbon_only", "ribbon_excluded"])
def test_ribbon_subset_failure_blocks_strict(subset: str) -> None:
    module = _module()
    records = _normalized_records(module)
    for record in records:
        is_ribbon = record["family"] == module.RIBBON_FAMILY
        if (subset == "ribbon_only" and is_ribbon) or (
            subset == "ribbon_excluded" and not is_ribbon
        ):
            record["variants"]["treatment"]["shift_fpr_excess"] = 0.05
    lcbs = dict.fromkeys(
        (
            "shift_gain_vs_raw_state",
            "shift_gain_vs_nonfactorial",
            "clean_pAUROC_0.05_hard_null_margin",
            "clean_fixed_F1_hard_null_margin",
            "shifted_pAUROC_0.05_hard_null_margin",
            "shifted_fixed_F1_hard_null_margin",
        ),
        0.01,
    )
    audits = [
        dict.fromkeys(
            ("support_ok", "access_ok", "feature_visibility_ok", "lodf_losf_ok", "maps_ok"),
            True,
        ),
    ]
    gates = module._gates(records, audits, lcbs)
    assert gates["ribbon_subset_directional_gates"][0][subset]["shift"] is False
    assert gates["G7_ribbon_and_ribbon_excluded"] is False
    assert gates["strict"] is False


def test_hierarchical_bootstrap_is_deterministic_and_json_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(module, "BOOTSTRAPS", 12)
    records = _normalized_records(module)
    first = module._bootstrap(records)
    second = module._bootstrap(records)
    assert first == second
    safe = module._json_safe(
        {
            "ok": np.asarray([1], dtype=np.bool_)[0],
            "count": np.int64(2),
            "value": np.float32(0.5),
        },
    )
    assert safe == {"ok": True, "count": 2, "value": 0.5}
    json.dumps(safe, allow_nan=False)
    with pytest.raises(ValueError, match="non-finite"):
        module._json_safe(np.float64(np.nan))


def test_threshold_quantile_is_frozen() -> None:
    module = _module()
    value = _variant(shift=0.02, pauc=0.9, f1=0.8)
    value["threshold_quantile"] = 0.99
    with pytest.raises(RuntimeError, match=r"0\.995"):
        module._variant_record(value, "fixture")


def test_payload_requires_complete_lodf_losf_cross_product() -> None:
    module = _module()
    payload = _payload(module)
    records, audit = module._normalize_payload(payload)
    assert len(records) == 16 * len(module.FAMILIES) * len(module.AXES)
    assert all(audit.values())

    payload["head_records"] = payload["head_records"][:-1]
    _, incomplete_audit = module._normalize_payload(payload)
    assert incomplete_audit["lodf_losf_ok"] is False
