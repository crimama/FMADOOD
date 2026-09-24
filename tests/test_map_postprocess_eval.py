"""Correctness tests for the offline map post-processing harness."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from map_postprocess_eval import (  # noqa: E402
    TRANSFORMS,
    evaluate_object,
    identity,
    run_harness,
)
from fmad.evaluation.streaming import StreamingSegmentationAccumulator  # noqa: E402


class _FakeDataset:
    """Minimal stand-in for MVTecAD2Dataset.get_test_images."""

    def __init__(self, mapping: dict[str, dict[str, list[str]]]):
        self._mapping = mapping

    def get_test_images(self, object_name: str, split: str) -> dict[str, list[str]]:
        return self._mapping[object_name]


def _build_run_dir(tmp_path, object_name: str, samples: dict[str, dict[str, np.ndarray]]) -> Path:
    """samples: {anomaly_type: {filename: score_map}}."""
    run_dir = tmp_path / "run-fake"
    for anomaly_type, files in samples.items():
        directory = run_dir / "anomaly_maps" / object_name / anomaly_type
        directory.mkdir(parents=True, exist_ok=True)
        for filename, score_map in files.items():
            np.savez_compressed(directory / f"{filename}.npz", score_map=score_map.astype(np.float32))
    return run_dir


def _make_scene(seed: int, shape=(12, 10)):
    rng = np.random.default_rng(seed)
    score_map = rng.random(shape, dtype=np.float64).astype(np.float32)
    mask = rng.random(shape) > 0.7
    return score_map, mask


def _standard_fixture(tmp_path):
    object_name = "can"
    good_map, good_mask = _make_scene(0)
    bad_map, bad_mask = _make_scene(1)
    samples = {"good": {"000.png": good_map}, "bad": {"001.png": bad_map}}
    run_dir = _build_run_dir(tmp_path, object_name, samples)
    masks = {
        (object_name, "good", "000.png"): good_mask,
        (object_name, "bad", "001.png"): bad_mask,
    }
    dataset = _FakeDataset({object_name: {"good": ["/data/can/good/000.png"],
                                          "bad": ["/data/can/bad/001.png"]}})

    def mask_loader(obj, anomaly_type, image_path, shape):
        mask = masks[(obj, anomaly_type, Path(image_path).name)]
        assert mask.shape == shape
        return mask

    return run_dir, dataset, object_name, mask_loader, {"good": good_map, "bad": bad_map}, \
        {"good": good_mask, "bad": bad_mask}


def test_identity_through_harness_matches_direct_accumulator(tmp_path):
    run_dir, dataset, object_name, mask_loader, maps, masks = _standard_fixture(tmp_path)

    payload = run_harness(run_dir, dataset, [object_name], "test_public", mask_loader, identity, {})

    direct = StreamingSegmentationAccumulator(max_fpr=0.05)
    # Same enumeration order the harness uses: sorted by (filename, anomaly_type).
    direct.update([maps["good"], maps["bad"]], [masks["good"], masks["bad"]])
    expected = direct.compute()

    result = payload["objects"][object_name]
    assert result["seg_AUROC_0.05"] == pytest.approx(expected.seg_auroc_0_05)
    assert result["seg_AUPRO_0.05"] == pytest.approx(expected.seg_aupro_0_05)
    assert result["seg_F1_oracle"] == pytest.approx(expected.seg_f1)
    for key in ("seg_AUROC_0.05", "seg_AUPRO_0.05", "seg_F1_oracle"):
        assert payload["macro"][key] == pytest.approx(result[key])


def test_missing_map_raises(tmp_path):
    run_dir, dataset, object_name, mask_loader, _, _ = _standard_fixture(tmp_path)
    (run_dir / "anomaly_maps" / object_name / "bad" / "001.png.npz").unlink()

    with pytest.raises(FileNotFoundError):
        run_harness(run_dir, dataset, [object_name], "test_public", mask_loader, identity, {})


def test_shape_mismatch_raises(tmp_path):
    run_dir, dataset, object_name, _, _, _ = _standard_fixture(tmp_path)

    def bad_mask_loader(obj, anomaly_type, image_path, shape):
        return np.zeros((shape[0] + 1, shape[1]), dtype=bool)

    with pytest.raises(ValueError, match="mask shape"):
        run_harness(run_dir, dataset, [object_name], "test_public", bad_mask_loader, identity, {})


def test_no_queries_for_object_raises(tmp_path):
    run_dir, _, object_name, mask_loader, _, _ = _standard_fixture(tmp_path)
    empty_dataset = _FakeDataset({object_name: {}})
    with pytest.raises(ValueError, match="no test images"):
        run_harness(run_dir, empty_dataset, [object_name], "test_public", mask_loader, identity, {})


@pytest.mark.parametrize("name", ["identity", "local_contrast", "highpass", "blur", "open_close"])
def test_transform_shape_dtype_and_non_mutation(name):
    rng = np.random.default_rng(4)
    score_map = rng.random((20, 24), dtype=np.float64).astype(np.float32)
    original = score_map.copy()
    transform = TRANSFORMS[name]

    out = transform(score_map)

    assert out.shape == score_map.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(score_map, original)  # input never mutated


def test_region_quantile_is_a_clear_stub():
    with pytest.raises(NotImplementedError):
        TRANSFORMS["region_quantile"](np.zeros((4, 4), dtype=np.float32))


def test_evaluate_object_raises_on_transform_producing_wrong_shape(tmp_path):
    run_dir, dataset, object_name, mask_loader, _, _ = _standard_fixture(tmp_path)
    accumulator = StreamingSegmentationAccumulator(max_fpr=0.05)

    def bad_transform(score_map, **_):
        return score_map[:-1].astype(np.float32)

    queries = [("bad", "/data/can/bad/001.png"), ("good", "/data/can/good/000.png")]
    with pytest.raises(ValueError, match="changed shape"):
        evaluate_object(run_dir, object_name, queries, mask_loader, bad_transform, {}, accumulator)
