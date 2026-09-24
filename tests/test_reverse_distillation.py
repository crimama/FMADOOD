from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from fmad.methods.reverse_distillation import ReverseDistillation
from fmad.protocol import MethodContext
from tests.test_psma_mr_v2 import _write_image
from tests.test_reference_matching import TinyPairBackbone


def _training_log_sink(record: list):
    def sink(step, metrics):
        record.append((step, metrics))
    return sink


def _write_images(tmp_path, count: int) -> list[str]:
    return [_write_image(tmp_path / f"{i}.png", i * 17) for i in range(count)]


def test_constructor_rejects_invalid_config():
    backbone = TinyPairBackbone()
    with pytest.raises(ValueError, match="teacher"):
        ReverseDistillation(backbone, {"teacher": "bogus"})
    with pytest.raises(ValueError, match="steps"):
        ReverseDistillation(backbone, {"teacher": "wrn50", "steps": True})
    with pytest.raises(ValueError, match="steps"):
        ReverseDistillation(backbone, {"teacher": "wrn50", "steps": 0})
    with pytest.raises(ValueError, match="learning_rate"):
        ReverseDistillation(backbone, {"teacher": "wrn50", "learning_rate": 0.0})
    with pytest.raises(ValueError, match="batch_size"):
        ReverseDistillation(backbone, {"teacher": "wrn50", "batch_size": 1.5})
    with pytest.raises(ValueError, match="crop_size"):
        ReverseDistillation(backbone, {"teacher": "wrn50", "crop_size": 33})
    with pytest.raises(ValueError, match="crop_size"):
        ReverseDistillation(backbone, {"teacher": "wrn50", "crop_size": 16})
    with pytest.raises(ValueError, match="flips"):
        ReverseDistillation(backbone, {"teacher": "wrn50", "flips": 1})
    with pytest.raises(ValueError, match="feature_layers"):
        ReverseDistillation(backbone, {"teacher": "dinov2", "feature_layers": [11]})


def test_wrn50_arm_fit_and_predict(tmp_path):
    backbone = TinyPairBackbone()
    paths = _write_images(tmp_path, 3)
    context = MethodContext("can", 0, 96, False, False)
    method = ReverseDistillation(
        backbone,
        {
            "teacher": "wrn50",
            "teacher_weights": None,
            "steps": 3,
            "batch_size": 2,
            "crop_size": 64,
            "train_resolution": 96,
            "resolution": 96,
        },
    )
    log_records: list = []
    method._training_log_sink = _training_log_sink(log_records)
    method.fit(paths, context)

    metadata = method.object_metadata()
    assert metadata["teacher_weights"] is None
    assert metadata["steps_completed"] == 3
    assert metadata["trainable_parameters"] > 0
    assert metadata["bank_free"] is True
    assert metadata["synthetic_anomalies_used"] is False
    assert metadata["query_labels_used"] is False
    assert metadata["dense_maps_accumulated"] is False
    assert metadata["loss"] is not None
    assert len(log_records) >= 1

    with Image.open(paths[0]) as opened:
        image = np.asarray(opened.convert("RGB"))
    anomaly_map, score = method.predict(paths[0])
    assert anomaly_map.dtype == np.float32
    assert anomaly_map.shape == image.shape[:2]
    assert np.isfinite(anomaly_map).all()
    assert score == pytest.approx(float(anomaly_map.max()))


def test_wrn50_default_teacher_weights_path_is_the_cache_file():
    backbone = TinyPairBackbone()
    method = ReverseDistillation(backbone, {"teacher": "wrn50"})
    assert method.teacher_weights == "/model-cache/hub/checkpoints/wide_resnet50_2-95faca4d.pth"


def test_wrn50_missing_weights_file_raises_clear_error():
    backbone = TinyPairBackbone()
    method = ReverseDistillation(
        backbone, {"teacher": "wrn50", "teacher_weights": "/nonexistent/does_not_exist.pth"}
    )
    with pytest.raises(FileNotFoundError, match="teacher_weights"):
        method._build_teacher_wrn50()


def test_dinov2_arm_fit_and_predict(tmp_path):
    backbone = TinyPairBackbone()
    paths = _write_images(tmp_path, 3)
    context = MethodContext("can", 0, 8, False, False)
    method = ReverseDistillation(
        backbone,
        {
            "teacher": "dinov2",
            "feature_layers": [11, 23],
            "steps": 2,
            "batch_size": 2,
            "crop_size": 8,
            "resolution": 8,
        },
    )
    log_records: list = []
    method._training_log_sink = _training_log_sink(log_records)
    method.fit(paths, context)

    metadata = method.object_metadata()
    assert metadata["teacher"] == "dinov2"
    assert metadata["steps_completed"] == 2
    assert metadata["trainable_parameters"] > 0
    assert metadata["loss"] is not None

    with Image.open(paths[0]) as opened:
        image = np.asarray(opened.convert("RGB"))
    anomaly_map, score = method.predict(paths[0])
    assert anomaly_map.dtype == np.float32
    assert anomaly_map.shape == image.shape[:2]
    assert np.isfinite(anomaly_map).all()
    assert score == pytest.approx(float(anomaly_map.max()))


def test_wrn50_fit_is_deterministic_given_the_same_seed(tmp_path):
    backbone = TinyPairBackbone()
    paths = _write_images(tmp_path, 3)
    context = MethodContext("can", 0, 96, False, False)
    config = {
        "teacher": "wrn50",
        "teacher_weights": None,
        "steps": 3,
        "batch_size": 2,
        "crop_size": 64,
        "train_resolution": 96,
        "resolution": 96,
    }
    method_a = ReverseDistillation(backbone, config)
    method_a.fit(paths, context)
    method_b = ReverseDistillation(backbone, config)
    method_b.fit(paths, context)

    state_a = method_a.student.state_dict()
    state_b = method_b.student.state_dict()
    assert state_a.keys() == state_b.keys()
    for key in state_a:
        assert torch.equal(state_a[key], state_b[key])
