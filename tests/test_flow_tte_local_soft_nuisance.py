from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/run_flow_tte_local_soft_nuisance.py"
SPEC = importlib.util.spec_from_file_location("local_soft_nuisance", PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_compact_orbit_selects_first_two_source_ids_per_split() -> None:
    test_images = {
        "good": [
            "/data/002_shift_1.png",
            "/data/001_shift_1.png",
            "/data/001_regular.png",
            "/data/003_regular.png",
            "/data/002_regular.png",
        ],
        "bad": [
            "/data/012_regular.png",
            "/data/011_regular.png",
            "/data/013_regular.png",
            "/data/011_shift_1.png",
            "/data/012_shift_1.png",
        ],
    }
    selected = MODULE.compact_orbit_items(test_images)
    stems = {path.stem for _split, path in selected}
    assert stems == {
        "001_regular", "001_shift_1", "002_regular", "002_shift_1",
        "011_regular", "011_shift_1", "012_regular", "012_shift_1",
    }


def test_local_soft_correction_removes_one_quarter_projected_residual() -> None:
    values = np.ones((4, 4, 2), dtype=np.float32)
    mean = np.zeros_like(values)
    basis = np.array([[1.0], [0.0]], dtype=np.float32)
    corrected = MODULE.correct_layer(values, mean, [basis] * 4, strength=0.25)
    np.testing.assert_allclose(corrected[..., 0], 0.75)
    np.testing.assert_allclose(corrected[..., 1], 1.0)


def test_randomized_basis_is_orthonormal_and_deterministic() -> None:
    rows = np.random.default_rng(4).standard_normal((200, 12)).astype(np.float32)
    first = MODULE.randomized_right_basis(rows, rank=2, seed=9)
    second = MODULE.randomized_right_basis(rows, rank=2, seed=9)
    np.testing.assert_allclose(first.T @ first, np.eye(2), atol=1e-5)
    np.testing.assert_allclose(first, second)
