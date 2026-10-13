from __future__ import annotations

import numpy as np
import pytest
import torch

import fmad.backbones.dinov2 as dinov2_module
import fmad.backbones.wrapper as wrapper_module
from fmad.backbones.dinov2 import DINOv2Backbone
from fmad.backbones.wrapper import DINOv2Wrapper


class _MockDINOv2(torch.nn.Module):
    patch_size = 14

    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(torch.nn.Identity() for _ in range(24))
        self.weight = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
        self.autocast_states: list[bool] = []
        self.autocast_dtypes: list[torch.dtype] = []

    def get_intermediate_layers(
        self,
        image_batch: torch.Tensor,
        layers: list[int] | None = None,
        *,
        n: int | None = None,
        return_class_token: bool = False,
    ):
        device_type = image_batch.device.type
        autocast_enabled = torch.is_autocast_enabled(device_type)
        self.autocast_states.append(autocast_enabled)
        dtype = (
            torch.get_autocast_dtype(device_type)
            if autocast_enabled
            else torch.float32
        )
        self.autocast_dtypes.append(dtype)
        tokens = self.weight.to(device=image_batch.device, dtype=dtype).reshape(1, 1, -1)
        tokens = tokens.expand(1, 4, -1)
        if return_class_token:
            cls_token = self.weight.to(device=image_batch.device, dtype=dtype).reshape(1, -1)
            return [(tokens, cls_token)]
        return [tokens for _ in (layers or ())]


def _load_mock_model(monkeypatch: pytest.MonkeyPatch) -> _MockDINOv2:
    model = _MockDINOv2()
    monkeypatch.setattr(wrapper_module, "load_dinov2_model", lambda _name: model)
    return model


def test_default_feature_interface_stays_cpu_numpy_fp32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _load_mock_model(monkeypatch)
    wrapper = DINOv2Wrapper("dinov2_vitl14", "cpu")

    features = wrapper.extract_features(torch.zeros((3, 28, 28)))

    assert len(features) == 4
    assert all(isinstance(feature, np.ndarray) for feature in features)
    assert all(feature.shape == (4, 8) and feature.dtype == np.float32 for feature in features)
    assert model.autocast_states == [False]
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_opt_in_keeps_fp32_tensors_on_model_device_and_uses_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _load_mock_model(monkeypatch)
    initial_weight = model.weight.detach().clone()
    wrapper = DINOv2Wrapper(
        "dinov2_vitl14",
        "cpu",
        feature_output_device="model",
        inference_autocast_dtype="bf16",
    )

    features = wrapper.extract_features(torch.zeros((3, 28, 28)))
    cls_features = wrapper.extract_cls_features(torch.zeros((3, 28, 28)))

    assert len(features) == 4
    assert all(isinstance(feature, torch.Tensor) for feature in features)
    assert all(
        feature.shape == (4, 8)
        and feature.dtype == torch.float32
        and feature.device == model.weight.device
        for feature in features
    )
    assert cls_features.shape == (8,)
    assert cls_features.dtype == torch.float32
    assert cls_features.device == model.weight.device
    assert model.autocast_states == [True, True]
    assert model.autocast_dtypes == [torch.bfloat16, torch.bfloat16]
    assert torch.equal(model.weight.detach(), initial_weight)
    assert all(not parameter.requires_grad for parameter in model.parameters())

    monkeypatch.setattr(dinov2_module, "find_local_dinov2_repo", lambda: None)
    backbone = DINOv2Backbone(
        "dinov2_vitl14",
        "cpu",
        feature_output_device="model",
        inference_autocast_dtype="bf16",
    )
    backbone._validate_and_record(wrapper)
    assert backbone.provenance is not None
    assert backbone.provenance["feature_output_device"] == "model"
    assert backbone.provenance["feature_output_device_resolved"] == "cpu"
    assert backbone.provenance["feature_output_interface"] == "torch.Tensor"
    assert backbone.provenance["feature_output_dtype"] == "float32"
    assert backbone.provenance["inference_autocast_dtype"] == "bf16"
    assert backbone.provenance["model_parameter_dtypes"] == ["torch.float32"]
    assert backbone.provenance["weights_frozen"] is True


def test_background_mask_accepts_opt_in_tensor_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _load_mock_model(monkeypatch)
    wrapper = DINOv2Wrapper(
        "dinov2_vitl14", "cpu", feature_output_device="model"
    )
    features = torch.randn((16, 8))

    mask = wrapper.compute_background_mask(
        features, (4, 4), masking_type=False
    )

    assert mask.shape == (16,)
    assert mask.dtype == np.bool_


@pytest.mark.parametrize(
    ("device", "options", "message"),
    [
        ("cpu", {"feature_output_device": "gpu"}, "feature_output_device"),
        ("cpu", {"inference_autocast_dtype": "fp8"}, "inference_autocast_dtype"),
        ("cpu", {"inference_autocast_dtype": "fp16"}, "only supported for CUDA"),
        ("mps", {}, "device must be CPU or CUDA"),
        ("cuda:invalid", {}, "unsupported DINOv2 device"),
    ],
)
def test_backbone_validates_output_and_autocast_device_options(
    device: str,
    options: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DINOv2Backbone("dinov2_vitl14", device, **options)
