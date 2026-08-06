from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile as tiff
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_dscr_density_condition_audit import (  # noqa: E402
    ARMS,
    CONDITIONS,
    METRICS,
    evaluate_condition_audit,
)

FLOAT16_DTYPE = np.dtype(np.float16)
FLOAT32_DTYPE = np.dtype(np.float32)


def _score(condition_index: int, split: str, arm: str, dtype: np.dtype) -> np.ndarray:
    base = np.linspace(0.05, 0.95, 16, dtype=np.float32).reshape(4, 4)
    if split == "bad":
        base = np.flip(base, axis=0).copy()
        base[:2, :2] += 0.4
    if arm == "D0" and condition_index > 0:
        base = np.flip(base, axis=1).copy()
    return base.astype(dtype)


def _write_query(  # noqa: PLR0913
    data_root: Path,
    cell_root: Path,
    object_name: str,
    arm: str,
    split: str,
    stem: str,
    condition_index: int,
    dtype: np.dtype = FLOAT32_DTYPE,
) -> None:
    source = data_root / object_name / "test_public" / split / f"{stem}.png"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"source")
    if split == "bad":
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[:2, :2] = 255
        mask_path = (
            data_root / object_name / "test_public" / "ground_truth" / "bad" / f"{stem}_mask.png"
        )
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(mask_path)
    map_path = (
        cell_root / "arms" / arm / "anomaly_maps" / object_name / "test" / split / f"{stem}.tiff"
    )
    map_path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(map_path, _score(condition_index, split, arm, dtype))


def _write_complete_cell(
    tmp_path: Path,
    *,
    dtype: np.dtype = FLOAT32_DTYPE,
    d0_prefix: str = "000",
) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    cell_root = tmp_path / "cell"
    for arm in ARMS:
        prefix = d0_prefix if arm == "D0" else "000"
        for condition_index, condition in enumerate(CONDITIONS):
            stem = f"{prefix}_{condition}"
            for split in ("good", "bad"):
                _write_query(
                    data_root,
                    cell_root,
                    "can",
                    arm,
                    split,
                    stem,
                    condition_index,
                    dtype,
                )
    return data_root, cell_root


def test_evaluate_condition_audit_reports_metrics_deltas_and_interactions(
    tmp_path: Path,
) -> None:
    data_root, cell_root = _write_complete_cell(tmp_path)

    payload = evaluate_condition_audit(data_root, cell_root, "can")

    assert payload["schema"] == "dscr-density-condition-audit-v1"
    assert payload["comparable"] is False
    assert payload["arm_id_parity"]["match"] is True
    assert len(payload["arm_id_parity"]["logical_ids"]) == 8
    assert payload["query_coverage"] == {
        "expected_logical_query_count_per_arm": 8,
        "observed_logical_query_count_per_arm": 8,
        "expected_total_arm_map_count": 16,
        "observed_total_arm_map_count": 16,
        "complete": True,
    }
    assert payload["precisions_evaluated"] == ["float16", "float32"]
    for condition in CONDITIONS:
        row = payload["conditions"][condition]
        for arm in ARMS:
            assert row["arms"][arm]["coverage"]["complete"] is True
            assert row["arms"][arm]["validity"]["float32_tiff_available"] is True
            assert set(row["arms"][arm]["metrics"]) == {"float16", "float32"}
        for precision in ("float16", "float32"):
            assert set(row["delta_vs_g0"][precision]) == set(METRICS)
            for metric in METRICS:
                expected = (
                    row["arms"]["D0"]["metrics"][precision][metric]
                    - row["arms"]["G0"]["metrics"][precision][metric]
                )
                assert row["delta_vs_g0"][precision][metric] == pytest.approx(expected)

    for precision in ("float16", "float32"):
        for metric in METRICS:
            observed = payload["interaction"][precision][metric]
            deltas = payload["conditions"]
            expected_shifted = np.mean(
                [
                    deltas[condition]["delta_vs_g0"][precision][metric]
                    for condition in CONDITIONS[1:]
                ],
            )
            assert observed["regular_delta"] == pytest.approx(
                deltas["regular"]["delta_vs_g0"][precision][metric],
            )
            assert observed["mean_shifted_delta"] == pytest.approx(expected_shifted)
            assert observed["value"] == pytest.approx(
                expected_shifted - observed["regular_delta"],
            )


def test_evaluate_condition_audit_omits_float32_metrics_when_tiffs_are_float16(
    tmp_path: Path,
) -> None:
    data_root, cell_root = _write_complete_cell(tmp_path, dtype=FLOAT16_DTYPE)

    payload = evaluate_condition_audit(data_root, cell_root, "can")

    assert payload["precisions_evaluated"] == ["float16"]
    assert payload["validity"]["float32_tiff_complete"] is False
    assert "float32" not in payload["interaction"]
    for condition in CONDITIONS:
        assert set(payload["conditions"][condition]["delta_vs_g0"]) == {"float16"}


def test_evaluate_condition_audit_rejects_missing_condition(tmp_path: Path) -> None:
    data_root, cell_root = _write_complete_cell(tmp_path)
    missing = (
        cell_root / "arms" / "G0" / "anomaly_maps" / "can" / "test" / "good" / "000_shift_2.tiff"
    )
    missing.unlink()

    with pytest.raises(RuntimeError, match="requires exactly one good and one bad"):
        evaluate_condition_audit(data_root, cell_root, "can")


def test_evaluate_condition_audit_rejects_extra_condition(tmp_path: Path) -> None:
    data_root, cell_root = _write_complete_cell(tmp_path)
    for split in ("good", "bad"):
        _write_query(
            data_root,
            cell_root,
            "can",
            "G0",
            split,
            "000_underexposed",
            4,
        )

    with pytest.raises(RuntimeError, match="unexpected G0/good condition 'underexposed'"):
        evaluate_condition_audit(data_root, cell_root, "can")


def test_evaluate_condition_audit_rejects_duplicate_condition(tmp_path: Path) -> None:
    data_root, cell_root = _write_complete_cell(tmp_path)
    for split in ("good", "bad"):
        _write_query(
            data_root,
            cell_root,
            "can",
            "G0",
            split,
            "001_regular",
            0,
        )

    with pytest.raises(RuntimeError, match=r"observed=\{'good': 2, 'bad': 2\}"):
        evaluate_condition_audit(data_root, cell_root, "can")


def test_evaluate_condition_audit_rejects_arm_id_mismatch(tmp_path: Path) -> None:
    data_root, cell_root = _write_complete_cell(tmp_path, d0_prefix="001")

    with pytest.raises(RuntimeError, match="identical logical query IDs"):
        evaluate_condition_audit(data_root, cell_root, "can")
