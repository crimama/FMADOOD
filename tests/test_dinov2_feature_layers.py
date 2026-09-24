from __future__ import annotations

import pytest
import torch

from fmad.backbones.dinov2 import DINOv2Backbone
from fmad.backbones.wrapper import DINOv2Wrapper
from fmad.backbones import loader
from fmad.backbones.loader import dinov2_repo_tree_sha256
from fmad.backbones.provenance import backbone_state_sha256


def test_backbone_import_installs_public_pytree_registration_compatibility():
    assert callable(torch.utils._pytree.register_pytree_node)


class _FakeDINOv2(torch.nn.Module):
    patch_size = 14

    def __init__(self, depth: int = 12) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(torch.nn.Identity() for _ in range(depth))
        self.requested_layers: list[int] = []

    def get_intermediate_layers(
        self,
        image_batch: torch.Tensor,
        layers: list[int],
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


def test_dinov2_small_model_default_layers_match_twelve_blocks() -> None:
    backbone = DINOv2Backbone("dinov2_vits14", "cpu")

    assert backbone.feature_layers == (2, 5, 8, 11)


def test_backbone_state_hash_is_stable_and_value_sensitive() -> None:
    first = torch.nn.Linear(2, 2, bias=False)
    second = torch.nn.Linear(2, 2, bias=False)
    second.load_state_dict(first.state_dict())

    assert backbone_state_sha256(first) == backbone_state_sha256(second)
    with torch.no_grad():
        second.weight[0, 0] += 1
    assert backbone_state_sha256(first) != backbone_state_sha256(second)


def test_repo_tree_hash_ignores_cache_and_bytecode(tmp_path) -> None:
    (tmp_path / "hubconf.py").write_text("model = 1\n")
    (tmp_path / "model.pyc").write_bytes(b"ignored")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "cached.py").write_text("ignored")
    first = dinov2_repo_tree_sha256(tmp_path)

    (tmp_path / "__pycache__" / "new.py").write_text("ignored")
    (tmp_path / "model.pyc").write_bytes(b"changed")
    assert dinov2_repo_tree_sha256(tmp_path) == first


def test_loader_propagates_hub_type_error(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("TORCH_HOME", str(tmp_path))
    monkeypatch.delenv("DINOV2_LOCAL_REPO", raising=False)
    monkeypatch.setattr(
        torch.hub, "load", lambda *args, **kwargs: (_ for _ in ()).throw(TypeError("model error"))
    )

    with pytest.raises(TypeError, match="model error"):
        loader.load_dinov2_model("dinov2_vitl14")


def test_strict_offline_checks_repository_and_checkpoint_before_hub(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("TORCH_HOME", str(tmp_path))
    monkeypatch.setenv("DINOV2_LOCAL_REPO", str(tmp_path / "repo"))
    (tmp_path / "repo").mkdir()
    called = False

    def fail_hub(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("hub load must not run without cached weights")

    monkeypatch.setattr(torch.hub, "load", fail_hub)
    with pytest.raises(FileNotFoundError, match="Strict offline"):
        loader.load_dinov2_model("dinov2_vitl14", strict_offline=True)
    assert not called


def test_offline_load_uses_local_state_without_hub_weight_download(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "hubconf.py").touch()
    checkpoint = tmp_path / "hub/checkpoints/dinov2_vitl14_pretrain.pth"
    checkpoint.parent.mkdir(parents=True)
    expected = torch.nn.Linear(2, 2)
    torch.save(expected.state_dict(), checkpoint)
    monkeypatch.setenv("TORCH_HOME", str(tmp_path))
    monkeypatch.setenv("DINOV2_LOCAL_REPO", str(repo))

    def load(*args, **kwargs):
        assert kwargs["source"] == "local"
        assert kwargs["pretrained"] is False
        return torch.nn.Linear(2, 2)

    monkeypatch.setattr(torch.hub, "load", load)
    observed = loader.load_dinov2_model("dinov2_vitl14", strict_offline=True)
    assert backbone_state_sha256(observed) == backbone_state_sha256(expected)
