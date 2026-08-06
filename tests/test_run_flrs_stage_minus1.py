from __future__ import annotations

# ruff: noqa: SLF001
import importlib.util
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from flow_tte.flrs import (
    fit_signature_envelope,
    local_relation_signature,
    radius1_relation_signature,
    random_neighbor_signature,
    score_robust_energy,
)


def _module():  # noqa: ANN202
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_flrs_stage_minus1.py"
    spec = importlib.util.spec_from_file_location("run_flrs_stage_minus1", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_torch_signatures_match_frozen_numpy_equations() -> None:
    module = _module()
    rng = np.random.default_rng(13)
    features = rng.normal(size=(7, 9, 12)).astype(np.float32)
    np.testing.assert_allclose(
        module._relation_signature_torch(features, device="cpu", kind="flrs"),
        local_relation_signature(features),
        atol=2e-6,
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        module._relation_signature_torch(features, device="cpu", kind="radius1_only"),
        radius1_relation_signature(features),
        atol=2e-6,
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        module._relation_signature_torch(features, device="cpu", kind="random_neighbor"),
        random_neighbor_signature(features),
        atol=2e-6,
        rtol=2e-6,
    )


def test_torch_envelope_and_energy_match_numpy_reference() -> None:
    module = _module()
    rng = np.random.default_rng(17)
    values = [rng.normal(size=(5, 6, 4)).astype(np.float32) for _ in range(3)]
    reference = fit_signature_envelope(values)
    accelerated = module._fit_envelope_torch(values, device="cpu")
    np.testing.assert_allclose(accelerated.center, reference.center, atol=1e-7)
    np.testing.assert_allclose(accelerated.scale, reference.scale, atol=1e-6)
    np.testing.assert_allclose(
        module._score_envelope_torch(values[0], accelerated, device="cpu"),
        score_robust_energy(values[0], reference),
        atol=2e-6,
        rtol=2e-6,
    )


def test_defect_retry_uses_generation_validity_only() -> None:
    module = _module()
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    generator = Mock(
        side_effect=[
            RuntimeError("sealed no-op"),
            (image.copy(), mask, {"seed": 7}),
        ],
    )
    module.make_synthetic_defect = generator
    _, _, metadata, retry = module._make_valid_defect(
        image,
        object_name="vial",
        relative_path="vial/train/good/000.png",
        kind="chromatic_blotch",
    )
    assert retry == 1
    assert metadata["seed"] == 7
    assert generator.call_count == 2
