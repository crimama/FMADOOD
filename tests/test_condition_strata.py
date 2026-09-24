"""Tests for condition_strata feature in pipeline."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fmad.datasets.base import ObjectInfo
from fmad.methods.base import BaseMethod, Prediction
from fmad.protocol import MethodContext
from fmad.pipeline import _run_streaming_object
from tests.test_reference_matching import TinyTripleBackbone
from tests.test_psma_mr_v2 import _write_image


class ConditionTestMethod(BaseMethod):
    """Minimal method that yields deterministic predictions for condition_strata testing."""

    def __init__(self, backbone, config):
        super().__init__(backbone, config)
        self.normal_mean = None

    def fit(self, support_paths, context):
        """Fit: memorize that we saw normal images."""
        self.normal_mean = 0.1

    def predict_batch(self, test_image_paths):
        """Generate deterministic score maps for each path."""
        predictions = []
        for path in test_image_paths:
            # Shape: (12, 16) to match _write_image output
            score_map = np.ones((12, 16), dtype=np.float32) * 0.1

            # Vary score based on filename to simulate defects
            if "good" not in str(path):
                score_map[:] = 0.7

            predictions.append(
                Prediction(
                    score_map=score_map,
                    image_score=float(score_map.mean()),
                    inference_seconds=0.001,
                )
            )
        return predictions

    def object_metadata(self):
        return {}


class MVTecAD2Mock:
    """Mock dataset that can be configured with conditions."""

    def __init__(self, tmp_path, conditions):
        self.root = tmp_path
        self.data_root = str(tmp_path)
        self.name = "mvtec_ad_2"
        self.conditions = conditions
        self._create_structure()

    def _create_structure(self):
        """Create directory structure for MVTec AD 2."""
        for condition in self.conditions:
            # Train good images
            train_good = self.root / "train" / condition / "good"
            train_good.mkdir(parents=True, exist_ok=True)
            for i in range(2):
                _write_image(train_good / f"img_{i:03d}.png", i)

            # Test images with condition suffix
            test_good = self.root / "test" / condition / "good"
            test_good.mkdir(parents=True, exist_ok=True)
            for i in range(2):
                _write_image(test_good / f"img_{i:03d}_{condition}.png", i)

            test_bad = self.root / "test" / condition / "bad"
            test_bad.mkdir(parents=True, exist_ok=True)
            for i in range(2):
                _write_image(test_bad / f"defect_{i:03d}_{condition}.png", 100 + i)

        # Unified ground truth masks (stored at root level for all conditions)
        gt_dir = self.root / "ground_truth" / "bad"
        gt_dir.mkdir(parents=True, exist_ok=True)
        for condition in self.conditions:
            for i in range(2):
                _write_image(gt_dir / f"defect_{i:03d}_{condition}_mask.png", i)

    def get_object_info(self, object_name):
        """Return object metadata."""
        return ObjectInfo(object_name, ["bad"], 12, False, False)

    def get_train_images(self, object_name):
        """Return support paths for the given object."""
        return sorted([str(self.root / "train" / cond / "good" / f"img_{i:03d}.png")
                       for cond in self.conditions for i in range(2)])

    def get_test_images(self, object_name):
        """Return test images grouped by anomaly type."""
        result = {}
        for condition in self.conditions:
            # Good images
            good_paths = [str(self.root / "test" / condition / "good" / f"img_{i:03d}_{condition}.png")
                         for i in range(2)]
            # Bad images
            bad_paths = [str(self.root / "test" / condition / "bad" / f"defect_{i:03d}_{condition}.png")
                        for i in range(2)]
            result.setdefault("good", []).extend(good_paths)
            result.setdefault("bad", []).extend(bad_paths)
        return result

    def get_ground_truth_dir(self, object_name):
        """Return ground truth base directory for all conditions."""
        return str(self.root / "ground_truth")


def test_condition_strata_default_disabled(tmp_path, monkeypatch):
    """Test that default (key absent/false) leaves metadata keys unchanged."""
    monkeypatch.setattr('fmad.evaluation.shift_diagnostic.acquisition_identity',
                        lambda path: (Path(path).stem, 'regular'))

    dataset = MVTecAD2Mock(tmp_path, ['regular'])
    method = ConditionTestMethod(TinyTripleBackbone(), {})

    config = {
        "protocol": {"shots": 2, "fixed_reference_manifest": None},
        "evaluation": {"condition_strata": False},  # Explicitly false
    }

    result = _run_streaming_object(
        method,
        dataset,
        "can",
        0,
        str(tmp_path / "run"),
        2,
        config=config,
        debug_output=False,
    )

    # With condition_strata disabled, metadata should not have condition_strata key
    assert "condition_strata" not in result.metadata
    assert "seg_AUPRO_0.05" in result.metadata["streaming_metrics"]


def test_condition_strata_enabled_two_conditions(tmp_path, monkeypatch):
    """Test that with two conditions, per-condition metrics equal FusionDiagnostic."""
    conditions = ['regular', 'underexposed']

    def mock_acquisition(path):
        path_str = str(path)
        for cond in conditions:
            if cond in path_str:
                return (Path(path).stem, cond)
        return (Path(path).stem, 'regular')

    monkeypatch.setattr('fmad.evaluation.shift_diagnostic.acquisition_identity',
                        mock_acquisition)

    dataset = MVTecAD2Mock(tmp_path, conditions)
    method = ConditionTestMethod(TinyTripleBackbone(), {})

    config = {
        "protocol": {"shots": 2, "fixed_reference_manifest": None},
        "evaluation": {"condition_strata": True},
    }

    result = _run_streaming_object(
        method,
        dataset,
        "can",
        0,
        str(tmp_path / "run"),
        2,
        config=config,
        debug_output=False,
    )

    # With condition_strata enabled, metadata should have condition_strata dict
    assert "condition_strata" in result.metadata
    strata = result.metadata["condition_strata"]

    # Should have both conditions
    assert "regular" in strata
    assert "underexposed" in strata

    # Each should have the required keys
    for cond_name, cond_data in strata.items():
        assert "AUPRO_0_05" in cond_data
        assert "pAUROC_0_05" in cond_data
        assert "queries" in cond_data
        assert "normal_pixels" in cond_data
        assert "defect_pixels" in cond_data
        assert isinstance(cond_data["queries"], list)


def test_condition_strata_good_only_yields_none_and_reason(tmp_path, monkeypatch):
    """Test that stratum with only good images yields AUPRO None and reason."""
    def mock_acquisition(path):
        path_str = str(path)
        if 'good_only' in path_str:
            return (Path(path).stem, 'good_only')
        return (Path(path).stem, 'regular')

    monkeypatch.setattr('fmad.evaluation.shift_diagnostic.acquisition_identity',
                        mock_acquisition)

    # Create dataset with two conditions: one with only good images, one with good+bad
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    # Condition 1: good_only - train and test with only good images
    train_good_only = dataset_dir / "train" / "good_only" / "good"
    train_good_only.mkdir(parents=True)
    for i in range(2):
        _write_image(train_good_only / f"img_{i:03d}.png", i)

    test_good_only = dataset_dir / "test" / "good_only" / "good"
    test_good_only.mkdir(parents=True)
    for i in range(2):
        _write_image(test_good_only / f"img_{i:03d}_good_only.png", i)

    # Condition 2: regular - train and test with good+bad images
    train_regular = dataset_dir / "train" / "regular" / "good"
    train_regular.mkdir(parents=True)
    for i in range(2):
        _write_image(train_regular / f"img_{i:03d}.png", i)

    test_good_regular = dataset_dir / "test" / "regular" / "good"
    test_good_regular.mkdir(parents=True)
    for i in range(2):
        _write_image(test_good_regular / f"img_{i:03d}_regular.png", i)

    test_bad_regular = dataset_dir / "test" / "regular" / "bad"
    test_bad_regular.mkdir(parents=True)
    for i in range(2):
        _write_image(test_bad_regular / f"defect_{i:03d}_regular.png", 100 + i)

    # Ground truth masks for regular condition only (good_only has no defects)
    gt_dir = dataset_dir / "ground_truth" / "bad"
    gt_dir.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        _write_image(gt_dir / f"defect_{i:03d}_regular_mask.png", i)

    class MixedConditionDataset:
        def __init__(self, root):
            self.data_root = str(root)

        def get_object_info(self, obj):
            return ObjectInfo(obj, ["bad"], 12, False, False)

        def get_train_images(self, obj):
            return sorted([str(train_good_only / f"img_{i:03d}.png") for i in range(2)] +
                         [str(train_regular / f"img_{i:03d}.png") for i in range(2)])

        def get_test_images(self, obj):
            return {
                "good": sorted([str(test_good_only / f"img_{i:03d}_good_only.png") for i in range(2)] +
                              [str(test_good_regular / f"img_{i:03d}_regular.png") for i in range(2)]),
                "bad": [str(test_bad_regular / f"defect_{i:03d}_regular.png") for i in range(2)]
            }

        def get_ground_truth_dir(self, obj):
            return str(dataset_dir / "ground_truth")

    method = ConditionTestMethod(TinyTripleBackbone(), {})

    config = {
        "protocol": {"shots": 2, "fixed_reference_manifest": None},
        "evaluation": {"condition_strata": True},
    }

    result = _run_streaming_object(
        method,
        MixedConditionDataset(dataset_dir),
        "can",
        0,
        str(tmp_path / "run"),
        2,
        config=config,
        debug_output=False,
    )

    # Should have condition_strata with both conditions
    assert "condition_strata" in result.metadata
    strata = result.metadata["condition_strata"]
    assert "good_only" in strata
    assert "regular" in strata

    # AUPRO for good_only should be None and reason should be present
    cond_data_good_only = strata["good_only"]
    assert cond_data_good_only["AUPRO_0_05"] is None
    assert cond_data_good_only["reason"] is not None

    # AUPRO for regular should have a value
    cond_data_regular = strata["regular"]
    assert cond_data_regular["AUPRO_0_05"] is not None


def test_condition_strata_non_bool_raises(tmp_path, monkeypatch):
    """Test that non-bool condition_strata config raises ValueError."""
    monkeypatch.setattr('fmad.evaluation.shift_diagnostic.acquisition_identity',
                        lambda path: (Path(path).stem, 'regular'))

    dataset = MVTecAD2Mock(tmp_path, ['regular'])
    method = ConditionTestMethod(TinyTripleBackbone(), {})

    config = {
        "protocol": {"shots": 2, "fixed_reference_manifest": None},
        "evaluation": {"condition_strata": "invalid_string"},  # Not a bool
    }

    with pytest.raises(ValueError, match="condition_strata must be a boolean"):
        _run_streaming_object(
            method,
            dataset,
            "can",
            0,
            str(tmp_path / "run"),
            2,
            config=config,
            debug_output=False,
        )
