from __future__ import annotations

import pytest
import torch

from fmad.backbones.dinov2 import DINOv2Backbone
from src.backbones import DINOv2Wrapper


def test_backbone_import_installs_public_pytree_registration_compatibility():
    assert callable(torch.utils._pytree.register_pytree_node)


class _FakeDINOv2(torch.nn.Module):
    patch_size = 14

    def __init__(self, depth: int = 12) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(torch.nn.Identity() for _ in range(depth))
        self.requested_layers: list[int] = []

    def get_intermediate_layers(
        self, image_batch: torch.Tensor, layers: list[int],
    ) -> list[torch.Tensor]:
        self.requested_layers = layers
        return [torch.zeros((1, 4, 8), device=image_batch.device) for _ in layers]


def test_dinov2_wrapper_uses_requested_feature_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeDINOv2()
    monkeypatch.setattr(DINOv2Wrapper, "load_model", lambda _self: model)
    wrapper = DINOv2Wrapper(
        "dinov2_vitb14_reg",
        "cpu",
        smaller_edge_size=448,
        feature_layers=(2, 5, 8, 11),
    )

    features = wrapper.extract_features(torch.zeros((3, 28, 28)))

    assert model.requested_layers == [2, 5, 8, 11]
    assert len(features) == 4
    assert all(feature.shape == (4, 8) for feature in features)


def test_dinov2_wrapper_rejects_layers_outside_model_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeDINOv2(depth=12)
    monkeypatch.setattr(DINOv2Wrapper, "load_model", lambda _self: model)

    with pytest.raises(ValueError, match="model depth 12"):
        DINOv2Wrapper(
            "dinov2_vitb14_reg",
            "cpu",
            feature_layers=(5, 11, 17, 23),
        )


def test_dinov2_backbone_exposes_cls_as_context() -> None:
    backbone = DINOv2Backbone("dinov2_vitb14_reg", "cpu")
    expected = torch.arange(4, dtype=torch.float32)

    class StubWrapper:
        def extract_cls_features(self, _image: torch.Tensor) -> torch.Tensor:
            return expected

    backbone._wrapper = StubWrapper()
    actual = backbone.extract_context_features(torch.zeros(1), "cls")
    assert actual is expected
    with pytest.raises(ValueError, match="Unsupported DINOv2 context source"):
        backbone.extract_context_features(torch.zeros(1), "register")
