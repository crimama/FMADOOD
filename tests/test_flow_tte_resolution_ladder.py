from __future__ import annotations

from pathlib import Path

import numpy as np

from flow_tte_resolution_ladder import (
    StageResult,
    run_resolution_sequence,
    small_component_counts,
    token_grid_shape,
)


def test_resolution_token_grids_match_hplus_patch16() -> None:
    assert token_grid_shape(672) == (42, 42)
    assert token_grid_shape(896) == (56, 56)
    assert token_grid_shape(1120) == (70, 70)


def test_small_defect_uses_strict_native_one_672_token_area() -> None:
    # 84px shorter edge -> 2 native px/token -> strict area < 4.
    gt = np.zeros((84, 84), dtype=bool)
    gt[1, 1:4] = True  # area 3: small
    gt[10:12, 10:12] = True  # area 4: not small
    prediction = np.zeros_like(gt)
    prediction[1, 2] = True
    counts = small_component_counts(prediction, gt)
    assert counts["area_limit_native_px"] == 4.0
    assert counts["small_component_areas_native_px"] == [3]
    assert counts["hit"] == counts["total"] == 1
    assert counts["recall"] == 1.0


def test_resolution_sequence_writes_leaderboard_before_advancing(tmp_path: Path) -> None:
    calls: list[int] = []

    def execute(resolution: int) -> StageResult:
        if calls:
            assert (tmp_path / f"stage_{calls[-1]}.tsv").is_file()
        calls.append(resolution)
        leaderboard = tmp_path / f"stage_{resolution}.tsv"
        leaderboard.write_text("ok\n")
        return StageResult(resolution, True, leaderboard, {})

    results = run_resolution_sequence(execute)
    assert calls == [672, 896, 1120]
    assert [row.resolution for row in results] == calls


def test_resolution_sequence_stops_after_invalid_672(tmp_path: Path) -> None:
    calls: list[int] = []

    def execute(resolution: int) -> StageResult:
        calls.append(resolution)
        return StageResult(resolution, False, tmp_path / "absent.tsv", {})

    results = run_resolution_sequence(execute)
    assert calls == [672]
    assert not results[0].valid
