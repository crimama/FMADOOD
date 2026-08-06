from __future__ import annotations

# ruff: noqa: C901, PERF401, PLR0913
import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

if TYPE_CHECKING:
    from types import ModuleType


def _module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "aggregate_dinov2l_default_mechanism_stage_m1.py"
    )
    spec = importlib.util.spec_from_file_location("mechanism_aggregate", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scores(arm: str, condition: str) -> tuple[float, float, float]:
    tail = {
        "A0": 0.11,
        "A1": 0.12,
        "A2": 0.12,
        "A3": 0.11,
        "A4": 0.105,
        "A5": 0.105,
        "A6": 0.12,
        "A7": 0.12,
        "L5": 0.12,
        "L11": 0.12,
        "L17": 0.12,
        "L23": 0.12,
    }[arm]
    if condition in {"C", "D"}:
        tail = 0.10
    p_base = {
        "A0": 0.84,
        "A1": 0.80,
        "A2": 0.80,
        "A3": 0.84,
        "A4": 0.83,
        "A5": 0.79,
        "A6": 0.78,
        "A7": 0.76,
        "L5": 0.76,
        "L11": 0.77,
        "L17": 0.78,
        "L23": 0.78,
    }[arm]
    f_base = p_base - 0.10
    if condition == "SD":
        p_base -= 0.01 if arm in {"A0", "A3", "A4"} else 0.03
        f_base -= 0.01 if arm in {"A0", "A3", "A4"} else 0.03
    return tail, p_base, f_base


def _row(
    module: ModuleType,
    *,
    object_name: str,
    source: str,
    fold: int,
    arm: str,
    condition: str,
    axis: str | None = None,
    sign: int | None = None,
    magnitude: int | None = None,
    family: str | None = None,
) -> dict[str, object]:
    q995, pauc, f1 = _scores(arm, condition)
    defect = condition in {"D", "SD"}
    row: dict[str, object] = {
        "schema": module.ROW_SCHEMA,
        "protocol_id": module.PROTOCOL_ID,
        "object": object_name,
        "fold": fold,
        "relative_path": source,
        "path_sha256": "a" * 64,
        "stimulus_id": f"{source}/{condition}/{axis}/{sign}/{magnitude}/{family}",
        "arm": arm,
        "condition": condition,
        "shift_axis": axis,
        "shift_sign": sign,
        "shift_magnitude": magnitude,
        "defect": family,
        "score_q95": q995 - 0.02,
        "score_q99": q995 - 0.01,
        "score_q995": q995,
        "inside_median": 0.8 if defect else None,
        "outside_median": 0.2 if defect else None,
        "defect_contrast": 0.6 if defect else None,
        "seg_auroc_0_05": pauc if defect else None,
        "fixed_f1": f1 if defect else None,
        "nn_identity_retention": 0.9 if condition == "S" else None,
        "nn_margin_mean": 0.2 if condition == "S" else None,
        "pair_distance_spearman": 0.95 if condition == "S" else None,
        "effective_rank": 8.0 if condition == "S" else None,
        "latent_rms": 1.0 if condition == "S" else None,
        "evaluated_image_sha256": "b" * 64,
    }
    if defect:
        row.update(
            defect_image_sha256="c" * 64,
            mask_sha256="d" * 64,
            defect_seed=17,
            defect_feature_retry=0,
        )
    return row


def _rows(module: ModuleType) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for object_name in module.OBJECTS:
        for source_index in range(16):
            source = f"{object_name}/train/good/{source_index:02d}.png"
            fold = source_index // 4
            for arm in module.ALL_ARMS:
                rows.append(
                    _row(
                        module,
                        object_name=object_name,
                        source=source,
                        fold=fold,
                        arm=arm,
                        condition="C",
                    ),
                )
                for axis in module.AXES:
                    for sign in module.SIGNS:
                        for magnitude in module.MAGNITUDES:
                            rows.append(
                                _row(
                                    module,
                                    object_name=object_name,
                                    source=source,
                                    fold=fold,
                                    arm=arm,
                                    condition="S",
                                    axis=axis,
                                    sign=sign,
                                    magnitude=magnitude,
                                ),
                            )
                for family in module.FAMILIES:
                    rows.append(
                        _row(
                            module,
                            object_name=object_name,
                            source=source,
                            fold=fold,
                            arm=arm,
                            condition="D",
                            family=family,
                        ),
                    )
                    for axis in module.AXES:
                        for sign in module.SIGNS:
                            for magnitude in module.MAGNITUDES:
                                rows.append(
                                    _row(
                                        module,
                                        object_name=object_name,
                                        source=source,
                                        fold=fold,
                                        arm=arm,
                                        condition="SD",
                                        axis=axis,
                                        sign=sign,
                                        magnitude=magnitude,
                                        family=family,
                                    ),
                                )
    return rows


def _summaries(module: ModuleType) -> list[dict[str, object]]:
    expected = 16 * len(module.ALL_ARMS) * (
        1
        + len(module.AXES) * len(module.SIGNS) * len(module.MAGNITUDES)
        + len(module.FAMILIES)
        + len(module.AXES)
        * len(module.SIGNS)
        * len(module.MAGNITUDES)
        * len(module.FAMILIES)
    )
    return [
        {
            "schema": module.OBJECT_SCHEMA,
            "protocol_id": module.PROTOCOL_ID,
            "object": object_name,
            "mode": "support_only",
            "support_count": 16,
            "fold_counts": [4, 4, 4, 4],
            "row_count": expected,
            "expected_factorial_count": expected,
            "observed_factorial_count": expected,
            "arms": list(module.ALL_ARMS),
            "conditions": list(module.CONDITIONS),
            "m0_score_max_abs_error": 1e-7,
            "m0_score_atol": module.CUDA_CDIST_PARITY_ATOL,
            "m0_nn_mismatch_count": 0,
            "parity_complete": True,
            "finite_complete": True,
            "static_scoring_only": True,
            "test_public_accessed": False,
            "test_private_accessed": False,
            "tta_calls": 0,
            "query_update_calls": 0,
        }
        for object_name in module.OBJECTS
    ]


def test_exact_factorial_coverage_and_static_access_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(module, "OBJECTS", ("can",))
    rows = _rows(module)

    normalized = module.validate_rows(rows)
    audit = module.validate_object_summaries(_summaries(module), {"can": len(rows)})

    assert len(normalized) == 16 * 12 * 85
    assert audit["can"]["static_scoring_only"] is True
    assert audit["can"]["access_clean"] is True


def test_missing_shift_direction_fails_factorial_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(module, "OBJECTS", ("can",))
    rows = _rows(module)
    rows = [
        row
        for row in rows
        if not (
            row["arm"] == "A0"
            and row["condition"] == "SD"
            and row["shift_axis"] == "gamma"
            and row["shift_sign"] == -1
            and row["shift_magnitude"] == 2
            and row["defect"] == "gaussian_blur"
        )
    ]

    with pytest.raises(module.AuditError, match="factorial parity"):
        module.validate_rows(rows)


def test_nonfinite_metric_is_rejected() -> None:
    module = _module()
    row = _row(
        module,
        object_name="can",
        source="can/train/good/00.png",
        fold=0,
        arm="A0",
        condition="C",
    )
    row["score_q995"] = float("nan")

    with pytest.raises(module.AuditError, match="non-finite"):
        module.normalize_row(row)


def test_stimulus_custody_verifies_arm_identity_and_decoded_clean_rgb(tmp_path: Path) -> None:
    module = _module()
    relative = "can/train/good/00.png"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    rgb = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    Image.fromarray(rgb).save(path)
    expected_hash = module.hashlib.sha256(rgb.tobytes()).hexdigest()
    rows = [
        _row(
            module,
            object_name="can",
            source=relative,
            fold=0,
            arm=arm,
            condition="C",
        )
        for arm in ("A0", "A1")
    ]
    for row in rows:
        row["evaluated_image_sha256"] = expected_hash

    audit = module.audit_stimulus_custody(rows, data_root=tmp_path)

    assert audit["clean_sources"] == 1
    assert audit["clean_rgb_hash_verified"] is True
    rows[1]["evaluated_image_sha256"] = "e" * 64
    with pytest.raises(module.AuditError, match="differs across arms"):
        module.audit_stimulus_custody(rows, data_root=tmp_path)


def test_bootstrap_is_deterministic_and_recomputes_layer_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(module, "BOOTSTRAPS", 12)
    rng = np.random.default_rng(4)
    data = rng.normal(size=(8, 16, 4, 2, 2, 4, 12, len(module.OUTCOMES)))

    first, seed1 = module.bootstrap_contrasts(data)
    second, seed2 = module.bootstrap_contrasts(data)

    assert first == second
    assert seed1 == seed2
    assert any(row["contrast"] == "fusion_layer_envelope" for row in first)


def _classification_fixture(module: ModuleType) -> tuple[np.ndarray, list[dict[str, object]]]:
    data = np.zeros((8, 1, 1, 1, 1, 1, len(module.ALL_ARMS), len(module.OUTCOMES)))
    for arm in module.ALL_ARMS:
        ai = module.ALL_ARMS.index(arm)
        data[..., ai, module.OUTCOMES.index("pair_distance_spearman")] = 0.95
        data[..., ai, module.OUTCOMES.index("effective_rank")] = 8.0
        data[..., ai, module.OUTCOMES.index("latent_rms")] = 1.0
        data[..., ai, module.OUTCOMES.index("nn_identity_retention")] = 0.9
    contrast_names = {
        name for name, _, _ in module.CONTRASTS
    } | {"dvt_flow_interaction", "fusion_layer_envelope"}
    rows = [
        {
            "contrast": contrast,
            "outcome": outcome,
            "estimate": 0.0,
            "ci_lower_2_5": -0.001,
            "ci_upper_97_5": 0.001,
            "positive_objects": 0,
            "negative_objects": 0,
            "zero_objects": 8,
        }
        for contrast in sorted(contrast_names)
        for outcome in module.OUTCOMES
    ]
    return data, rows


def test_directional_density_and_fusion_labels_require_six_of_eight_objects() -> None:
    module = _module()
    data, rows = _classification_fixture(module)
    lookup = {(row["contrast"], row["outcome"]): row for row in rows}
    density = lookup[("density", "SD_pauroc")]
    density.update(estimate=0.02, ci_lower_2_5=0.01, ci_upper_97_5=0.03, positive_objects=5)
    for outcome, estimate in (
        ("D_pauroc", 0.01),
        ("SD_pauroc", 0.01),
        ("D_fixed_f1", 0.02),
        ("SD_fixed_f1", 0.02),
    ):
        fusion = lookup[("fusion_layer_envelope", outcome)]
        fusion.update(
            estimate=estimate,
            ci_lower_2_5=estimate / 2,
            ci_upper_97_5=estimate * 1.5,
            positive_objects=5,
        )

    mechanism = module.classify_mechanisms(data, rows)

    assert mechanism["density"]["label"] == "NO_MATERIAL_CONTRIBUTION"
    assert mechanism["fusion"]["label"] == "DOMINANT_LAYER_OR_DESTRUCTIVE_FUSION"


def test_partial_flow_signal_is_mixed_not_not_necessary() -> None:
    module = _module()
    data, rows = _classification_fixture(module)
    signal = next(
        row
        for row in rows
        if row["contrast"] == "flow_with_dvt" and row["outcome"] == "D_pauroc"
    )
    signal.update(
        estimate=0.01,
        ci_lower_2_5=0.005,
        ci_upper_97_5=0.015,
        positive_objects=8,
    )

    mechanism = module.classify_mechanisms(data, rows)

    assert mechanism["flow"]["label"] == "MIXED_OR_INCONCLUSIVE"


def test_common_alignment_counts_one_sided_neighbor_destabilization_as_failure() -> None:
    module = _module()
    data, rows = _classification_fixture(module)
    outcome = {name: module.OUTCOMES.index(name) for name in module.OUTCOMES}
    arm = {name: module.ALL_ARMS.index(name) for name in module.ALL_ARMS}
    data[..., arm["A1"], outcome["tail_inflation_abs"]] = 1.0
    data[..., arm["A0"], outcome["tail_inflation_abs"]] = 0.8
    data[..., arm["A2"], outcome["tail_inflation_abs"]] = 0.84
    data[..., arm["A3"], outcome["tail_inflation_abs"]] = 0.9
    data[..., arm["A0"], outcome["pauroc_retention"]] = 0.2
    data[..., arm["A2"], outcome["pauroc_retention"]] = 0.16
    data[..., arm["A3"], outcome["pauroc_retention"]] = 0.1
    data[..., arm["A2"], outcome["nn_identity_retention"]] = 0.5
    lookup = {(row["contrast"], row["outcome"]): row for row in rows}
    dvt_tail = lookup[("dvt_total", "tail_inflation_abs")]
    dvt_tail.update(
        estimate=-0.2,
        ci_lower_2_5=-0.3,
        ci_upper_97_5=-0.1,
        positive_objects=0,
        negative_objects=8,
        zero_objects=0,
    )

    mechanism = module.classify_mechanisms(data, rows)

    assert mechanism["dvt"]["label"] == "COMMON_COORDINATE_ALIGNMENT"


def test_aggregate_assigns_frozen_labels_and_writes_required_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(module, "OBJECTS", ("can",))
    monkeypatch.setattr(module, "BOOTSTRAPS", 12)
    summary, contrasts, per_object = module.aggregate_rows(_rows(module), _summaries(module))

    module.write_outputs(tmp_path, summary, contrasts, per_object)

    assert summary["mechanism"]["dvt"]["label"] in {
        "QUERY_DRIVEN",
        "SHIFT_CORRECTOR",
        "COMMON_COORDINATE_ALIGNMENT",
        "RANKING_ONLY_OR_MIXED",
    }
    assert summary["claim_verdict"] == "KILL_FOR_CLAIM"
    assert summary["public_shadow_authorized"] is False
    assert {path.name for path in tmp_path.iterdir()} == {
        "summary.json",
        "causal_contrasts.tsv",
        "per_object_metrics.tsv",
        "neighbor_stability.tsv",
        "geometry.tsv",
        "verdict.txt",
    }
    json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))


def test_parity_or_query_mutation_failure_blocks_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(module, "OBJECTS", ("can",))
    rows = _rows(module)
    summaries = _summaries(module)
    summaries[0]["query_update_calls"] = 1

    with pytest.raises(module.AuditError, match="query update"):
        module.aggregate_rows(rows, summaries)


def test_real_chunk_custody_excludes_controller_smoke(tmp_path: Path) -> None:
    module = _module()
    chunks = tmp_path / "chunks"
    groups = (
        ("gpu0", ("can", "rice")),
        ("gpu1", ("fabric", "walnuts")),
        ("gpu2", ("vial", "sheet_metal")),
        ("gpu3", ("fruit_jelly", "wallplugs")),
    )
    for name, objects in groups:
        root = chunks / name
        root.mkdir(parents=True)
        (root / "run_manifest.json").write_text(
            json.dumps(
                {
                    "schema": "dinov2l-default-mechanism-m1-run-v1",
                    "protocol_id": module.PROTOCOL_ID,
                    "mode": "real",
                    "objects": list(objects),
                    "support_manifest_sha256": module.SUPPORT_SHA256,
                    "m0_complete": True,
                    "factorial_complete": True,
                    "static_scoring_only": True,
                    "test_public_accessed": False,
                    "test_private_accessed": False,
                }
            ),
            encoding="utf-8",
        )
        (root / "access_ledger.json").write_text(
            json.dumps(
                {
                    "mode": "real",
                    "support_reads": 160,
                    "support_content_hash_reads": 128,
                    "support_image_reads": 32,
                    "test_public_reads": 0,
                    "test_private_reads": 0,
                    "query_update_calls": 0,
                    "tta_calls": 0,
                    "static_scoring_only": True,
                    "whitelist_complete": True,
                }
            ),
            encoding="utf-8",
        )
    smoke = tmp_path / "smoke" / "objects"
    smoke.mkdir(parents=True)
    (smoke / "can_summary.json").write_text("{}", encoding="utf-8")

    module._validate_chunk_audits(chunks)

    broken = chunks / "gpu0" / "access_ledger.json"
    payload = json.loads(broken.read_text(encoding="utf-8"))
    payload["tta_calls"] = 1
    broken.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(module.AuditError, match="tta_calls"):
        module._validate_chunk_audits(chunks)

    payload["tta_calls"] = 0
    payload["support_reads"] = 32
    broken.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(module.AuditError, match="total support reads"):
        module._validate_chunk_audits(chunks)
