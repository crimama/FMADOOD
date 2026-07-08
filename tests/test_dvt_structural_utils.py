from __future__ import annotations

# pyright: reportMissingImports=false
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_ROOT = _REPO_ROOT / "scripts"
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

from flow_tte_dvt_structural_utils import (  # noqa: E402
    high_region_share,
    low_rank_energy_summary,
    normalize_minmax,
    safe_corrcoef,
    top_percent_mean,
)


def test_safe_corrcoef_returns_nan_for_constant_input() -> None:
    left = np.ones((2, 2), dtype=np.float32)
    right = np.arange(4, dtype=np.float32).reshape(2, 2)

    assert math.isnan(safe_corrcoef(left, right))


def test_high_region_share_counts_overlap_between_top_regions() -> None:
    values = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    reference = np.asarray([[3.0, 2.0], [1.0, 0.0]], dtype=np.float32)

    assert high_region_share(values, reference, 50.0) == 0.0


def test_low_rank_energy_summary_detects_rank_one_field() -> None:
    base = np.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], dtype=np.float32)
    field = base.reshape(2, 1, 3)

    summary = low_rank_energy_summary(field, ranks=(1, 2))

    assert summary["top1_energy_share"] > 0.999
    assert summary["effective_rank"] < 1.01


def test_top_percent_mean_uses_largest_values() -> None:
    values = np.arange(10, dtype=np.float32)

    assert top_percent_mean(values, 0.2) == 8.5


def test_normalize_minmax_handles_constant_arrays() -> None:
    values = np.full((2, 3), 7.0, dtype=np.float32)

    normalized = normalize_minmax(values)

    assert np.all(normalized == 0.0)
