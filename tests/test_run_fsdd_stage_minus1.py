from __future__ import annotations

# ruff: noqa: ANN001, ANN202, SLF001
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_fsdd_stage_minus1.py"
    spec = importlib.util.spec_from_file_location("run_fsdd_stage_minus1", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pool(module):
    sources = ("object/train/good/a.png", "object/train/good/b.png")
    clean = {}
    shifts = {}
    defects = {}
    masks = {}
    primary = {}
    grids = {}
    rgb = {}
    texture = {}
    for source_index, source in enumerate(sources):
        base = np.arange(25 * 4, dtype=np.float32).reshape(25, 4) / 100.0
        base = base + source_index
        clean[source] = base
        grids[source] = (5, 5)
        rgb[("clean", source)] = np.arange(25, dtype=np.float32)
        texture[("clean", source)] = np.arange(25, dtype=np.float32)
        for axis_index, axis in enumerate(module.AXES):
            for scale in (1, 2):
                name = f"{axis}_x{scale}"
                primary[(source, axis, scale)] = name
                shifts[(source, name)] = base + (axis_index + 1) * scale / 100.0
        for family_index, family in enumerate(module.FAMILIES):
            mask = np.zeros((5, 5), dtype=bool)
            mask[1:4, 1:4] = True
            masks[(source, family)] = mask
            defects[(source, family, "clean", 0)] = base + (family_index + 1) / 10.0
            for axis_index, axis in enumerate(module.AXES):
                for scale in (1, 2):
                    defects[(source, family, axis, scale)] = (
                        base + (family_index + 1) / 10.0 + (axis_index + 1) * scale / 100.0
                    )
    pool = module.StimulusPool(
        clean=clean,
        shifts=shifts,
        defects=defects,
        masks=masks,
        primary_shift_names=primary,
        compound_shift_names=dict.fromkeys(sources, ()),
        grids=grids,
        rgb_seams=rgb,
        rgb_textures=texture,
    )
    entries = tuple(
        SimpleNamespace(relative_path=source, fold=index) for index, source in enumerate(sources)
    )
    return pool, entries


def test_treatment_and_nonfactorial_share_exact_training_cells() -> None:
    module = _module()
    pool, entries = _pool(module)
    treatment = module.training_data(
        pool,
        entries,
        source_fold=1,
        heldout_family="high_frequency_ribbon",
        heldout_axis="gamma",
        variant="treatment",
        experiment_seed=0,
    )
    supervised = module.training_data(
        pool,
        entries,
        source_fold=1,
        heldout_family="high_frequency_ribbon",
        heldout_axis="gamma",
        variant="nonfactorial",
        experiment_seed=0,
    )
    np.testing.assert_array_equal(treatment[0], supervised[0])
    np.testing.assert_array_equal(treatment[1], supervised[1])
    assert treatment[2] is not None
    assert supervised[2] is None


def test_training_uses_both_shift_scales_and_excludes_heldout_axis() -> None:
    module = _module()
    pool, entries = _pool(module)
    baseline = module.training_data(
        pool,
        entries,
        source_fold=1,
        heldout_family="high_frequency_ribbon",
        heldout_axis="gamma",
        variant="nonfactorial",
        experiment_seed=0,
    )[0]
    defects = dict(pool.defects)
    source = entries[0].relative_path
    defects[(source, "cutpaste", "exposure", 1)] = np.full((25, 4), 99.0, dtype=np.float32)
    modified = module.StimulusPool(
        **{**pool.__dict__, "defects": defects},
    )
    changed = module.training_data(
        modified,
        entries,
        source_fold=1,
        heldout_family="high_frequency_ribbon",
        heldout_axis="gamma",
        variant="nonfactorial",
        experiment_seed=0,
    )[0]
    assert not np.array_equal(baseline, changed)

    defects[(source, "cutpaste", "gamma", 1)] = np.full((25, 4), 77.0, dtype=np.float32)
    excluded = module.StimulusPool(**{**pool.__dict__, "defects": defects})
    excluded_rows = module.training_data(
        excluded,
        entries,
        source_fold=1,
        heldout_family="high_frequency_ribbon",
        heldout_axis="gamma",
        variant="nonfactorial",
        experiment_seed=0,
    )[0]
    np.testing.assert_array_equal(changed, excluded_rows)


def test_visibility_audit_rejects_constant_feature_delta() -> None:
    module = _module()
    reference_image = np.zeros((8, 8, 3), dtype=np.uint8)
    stimulus_image = reference_image.copy()
    stimulus_image[2:6, 2:6] = 10
    reference = np.zeros((9, 4), dtype=np.float32)
    constant = np.ones((9, 4), dtype=np.float32)
    audit = module._visibility_audit(reference_image, stimulus_image, reference, constant)
    assert audit["valid"] is False

    varying = constant.copy()
    varying[:, 0] = np.arange(9, dtype=np.float32)
    audit = module._visibility_audit(reference_image, stimulus_image, reference, varying)
    assert audit["valid"] is True


def test_patch_mask_contract_is_two_dimensional_for_toroidal_null() -> None:
    module = _module()
    flat = np.zeros(25, dtype=bool)
    flat[[6, 7, 8, 11, 12, 13, 16, 17, 18]] = True
    grid_mask = flat.reshape(5, 5)
    interior, boundary = module._interior_boundary(grid_mask)
    assert interior.shape == (5, 5)
    assert boundary.shape == (5, 5)
    shuffled = module.toroidal_mask_shuffle(interior, seed=7)
    assert shuffled.shape == grid_mask.shape
    assert int(shuffled.sum()) == int(interior.sum())


def test_feature_visibility_uses_full_overlap_not_eroded_interior() -> None:
    module = _module()
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    stimulus = image.copy()
    stimulus[2:6, 2:6] = 10
    reference = np.zeros((9, 2), dtype=np.float32)
    changed = reference.copy()
    changed[0, 0] = 1.0
    changed[1, 0] = 2.0
    full_overlap = np.zeros(9, dtype=bool)
    full_overlap[:2] = True
    eroded_interior = np.zeros(9, dtype=bool)
    eroded_interior[0] = True
    assert (
        module._visibility_audit(
            image,
            stimulus,
            reference,
            changed,
            selection=full_overlap,
        )["valid"]
        is True
    )
    assert (
        module._visibility_audit(
            image,
            stimulus,
            reference,
            changed,
            selection=eroded_interior,
        )["valid"]
        is False
    )


def test_high_frequency_ribbon_has_patch_grid_interior() -> None:
    module = _module()
    image = np.full((140, 140, 3), 127, dtype=np.uint8)
    defect, native_mask, metadata = module._interior_capable_high_frequency_ribbon(
        image,
        (10, 10),
        {"endpoints": [[30, 20], [110, 120]], "kind": "scratch"},
    )
    patch_mask = module.mask_to_patch_grid(native_mask, (10, 10)).reshape(10, 10)
    interior, boundary = module._interior_boundary(patch_mask)
    assert metadata["fsdd_construct"] == "high_frequency_ribbon"
    assert metadata["fsdd_ribbon_thickness"] == 56
    assert np.any(defect != image)
    assert np.any(interior)
    assert np.any(boundary)


def test_rgb_texture_null_detects_checkerboard_interior() -> None:
    module = _module()
    image = np.indices((64, 64)).sum(axis=0) % 2
    image = np.repeat((image * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    texture = module._rgb_texture_map(image, (8, 8)).reshape(8, 8)
    assert np.all(texture[2:6, 2:6] > 0.0)


def test_fixed_threshold_controls_f1() -> None:
    module = _module()
    labels = np.asarray([0, 0, 1, 1], dtype=bool)
    scores = np.asarray([0.1, 0.4, 0.6, 0.9], dtype=np.float32)
    strict = module._defect_metric(labels, scores, threshold=0.8)
    relaxed = module._defect_metric(labels, scores, threshold=0.5)
    assert strict["fixed_F1"] < relaxed["fixed_F1"]
    assert strict["oracle_F1"] == relaxed["oracle_F1"]
