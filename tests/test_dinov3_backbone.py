from __future__ import annotations

import pytest

from scripts.dinov3_backbone import (
    UnknownDINOv3ModelError,
    dinov3_hidden_state_index,
    dinov3_patch_token_start,
    is_dinov3_model_name,
    patch_torch_pytree_compat,
    resolve_dinov3_model_id,
)


def test_resolve_dinov3_model_id_uses_default_vitl16_mirror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DINOV3_MODEL_ID", raising=False)

    model_id = resolve_dinov3_model_id("dinov3_vitl16")

    assert model_id == "camenduru/dinov3-vitl16-pretrain-lvd1689m"


def test_resolve_dinov3_model_id_supports_superadd_vith16plus_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DINOV3_MODEL_ID", raising=False)

    model_id = resolve_dinov3_model_id("dinov3_vith16plus")

    assert model_id == "facebook/dinov3-vith16plus-pretrain-lvd1689m"


def test_resolve_dinov3_model_id_honors_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DINOV3_MODEL_ID", "local/dinov3")

    assert resolve_dinov3_model_id("dinov3_vitl16") == "local/dinov3"


def test_resolve_dinov3_model_id_rejects_unknown_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DINOV3_MODEL_ID", raising=False)

    with pytest.raises(UnknownDINOv3ModelError):
        resolve_dinov3_model_id("dinov3_unknown")


def test_dinov3_token_offsets_skip_cls_and_register_tokens() -> None:
    assert dinov3_hidden_state_index(23) == 24
    assert dinov3_patch_token_start(4) == 5


def test_dinov3_model_name_detection_allows_alias_or_hf_id() -> None:
    assert is_dinov3_model_name("dinov3_vitl16")
    assert is_dinov3_model_name("camenduru/dinov3-vitl16-pretrain-lvd1689m")
    assert not is_dinov3_model_name("dinov2_vitl14")


def test_pytree_compat_patch_is_idempotent() -> None:
    patch_torch_pytree_compat()
    patch_torch_pytree_compat()
