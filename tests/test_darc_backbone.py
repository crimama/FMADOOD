from __future__ import annotations

from types import SimpleNamespace
from typing import NamedTuple, Optional, Tuple

import pytest
import torch
from torch import nn

import flow_tte.darc_backbone as darc_backbone
from flow_tte.darc_backbone import DarcBackboneError, DINOv3EarlyExitAdapter


class _Config(NamedTuple):
    patch_size: int = 2
    num_register_tokens: int = 2


class _Output(NamedTuple):
    hidden_states: Tuple[torch.Tensor, ...]
    last_hidden_state: torch.Tensor


class _Embeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embeddings = nn.Conv2d(3, 2, kernel_size=2, stride=2, bias=False)
        self.seen_dtype: Optional[torch.dtype] = None

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del bool_masked_pos
        self.seen_dtype = pixel_values.dtype
        patches = self.patch_embeddings(pixel_values).flatten(2).transpose(1, 2)
        prefix = torch.zeros(
            (pixel_values.shape[0], 3, patches.shape[-1]),
            dtype=patches.dtype,
            device=patches.device,
        )
        return torch.cat((prefix, patches), dim=1)


class _Rope(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_dtype: Optional[torch.dtype] = None

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.seen_dtype = pixel_values.dtype
        marker = torch.ones((), device=pixel_values.device, dtype=pixel_values.dtype)
        return marker, marker


class _Layer(nn.Module):
    def __init__(self, increment: float) -> None:
        super().__init__()
        self.increment = increment
        self.calls = 0
        self.grad_enabled: Optional[bool] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        del attention_mask
        assert position_embeddings is not None
        self.calls += 1
        self.grad_enabled = torch.is_grad_enabled()
        return hidden_states + self.increment


class _MalformedOnceLayer(_Layer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        values = super().forward(hidden_states, attention_mask, position_embeddings)
        return values[:, 0] if self.calls == 1 else values


class DINOv3ViTModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.embeddings = _Embeddings()
        self.rope_embeddings = _Rope()
        self.layer = nn.ModuleList((_Layer(1.0), _Layer(2.0), _Layer(4.0)))
        self.norm = _FinalNorm()
        self.forward_calls = 0

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ) -> _Output:
        assert output_hidden_states
        assert return_dict
        self.forward_calls += 1
        values = pixel_values.to(self.embeddings.patch_embeddings.weight.dtype)
        hidden = self.embeddings(values)
        states = [hidden]
        position = self.rope_embeddings(values)
        for layer in self.layer:
            hidden = layer(hidden, position_embeddings=position)
            states.append(hidden)
        return _Output(hidden_states=tuple(states), last_hidden_state=self.norm(states[-1]))


class _FinalNorm(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        channel_scale = torch.tensor((10.0, 1.0), dtype=values.dtype, device=values.device)
        return values * channel_scale


class UnsupportedModel(DINOv3ViTModel):
    pass


@pytest.fixture(autouse=True)
def _audited_transformers_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(darc_backbone, "_installed_transformers_version", lambda: "4.56.2")


def test_extract_uses_exact_early_exit_and_skips_prefix_tokens() -> None:
    # Given: a 2x3 patch grid and a request ending before the final transformer block.
    model = DINOv3ViTModel()
    model.embeddings.patch_embeddings.weight.data.fill_(1.0)
    adapter = DINOv3EarlyExitAdapter(model, layers=(0, 1))
    pixels = torch.ones((1, 3, 5, 7), dtype=torch.float64, requires_grad=True)

    # When: features are extracted through the audited internal path.
    result = adapter.extract(pixels)

    # Then: execution stops at layer 1 and returns normalized patch-only grids.
    assert result.used_early_exit
    assert [layer.calls for layer in model.layer] == [1, 1, 0]
    assert result.layers == (0, 1)
    assert result.grids[0].shape == (1, 2, 3, 2)
    torch.testing.assert_close(
        torch.linalg.vector_norm(result.grids[0], dim=-1),
        torch.ones(1, 2, 3),
    )
    assert model.layer[0].grad_enabled is False
    assert model.layer[1].grad_enabled is False


def test_extract_applies_norm_only_to_requested_final_layer() -> None:
    # Given: requests for an intermediate and the final transformer layer.
    model = DINOv3ViTModel()
    model.embeddings.patch_embeddings.weight.data.fill_(1.0)
    adapter = DINOv3EarlyExitAdapter(model, layers=(1, 2))
    pixels = torch.ones((1, 3, 4, 4))

    # When: both layers are extracted.
    result = adapter.extract(pixels)

    # Then: the final output is normalized before patch slicing, unlike layer 1.
    intermediate_raw = torch.full((2,), 15.0)
    final_raw = torch.tensor((190.0, 19.0))
    torch.testing.assert_close(
        result.grids[0][0, 0, 0],
        torch.nn.functional.normalize(intermediate_raw, dim=0),
    )
    torch.testing.assert_close(
        result.grids[1][0, 0, 0],
        torch.nn.functional.normalize(final_raw, dim=0),
    )
    assert model.forward_calls == 0


def test_extract_can_preserve_raw_final_block_for_superadd_parity() -> None:
    # Given: a final-layer request configured for get_intermediate_layers(norm=False) parity.
    model = DINOv3ViTModel()
    model.embeddings.patch_embeddings.weight.data.fill_(1.0)
    adapter = DINOv3EarlyExitAdapter(model, layers=(2,), normalize_final=False)

    # When: the final block is extracted without final normalization.
    result = adapter.extract(torch.ones((1, 3, 4, 4)))

    # Then: patch normalization sees the raw block output, not model.norm output.
    raw = torch.full((2,), 19.0)
    torch.testing.assert_close(result.grids[0][0, 0, 0], torch.nn.functional.normalize(raw, dim=0))
    assert result.used_early_exit


def test_extract_can_preserve_raw_patch_magnitudes_for_superadd_parity() -> None:
    # Given: an intermediate-layer request with feature normalization disabled.
    model = DINOv3ViTModel()
    model.embeddings.patch_embeddings.weight.data.fill_(1.0)
    adapter = DINOv3EarlyExitAdapter(model, layers=(0,), normalize_features=False)

    # When: the patch grid is extracted.
    result = adapter.extract(torch.ones((1, 3, 4, 4)))

    # Then: the model's raw post-block values and default float32 dtype are retained.
    torch.testing.assert_close(result.grids[0][0, 0, 0], torch.full((2,), 13.0))
    assert result.grids[0].dtype == torch.float32


def test_extract_reports_realized_integer_geometry_and_output_dtype() -> None:
    # Given: input dimensions that are not divisible by the patch size.
    model = DINOv3ViTModel()
    adapter = DINOv3EarlyExitAdapter(model, layers=(0,), output_dtype=torch.float16)
    pixels = torch.ones((1, 3, 5, 7), dtype=torch.float64)

    # When: patch features are extracted.
    result = adapter.extract(pixels)

    # Then: metadata records the retained 4x6 region and model dtype is respected.
    assert result.geometry.input_size == (5, 7)
    assert result.geometry.realized_size == (4, 6)
    assert result.geometry.grid_size == (2, 3)
    assert result.geometry.patch_size == 2
    assert result.grids[0].dtype == torch.float16
    assert model.embeddings.seen_dtype == torch.float32
    assert model.rope_embeddings.seen_dtype == torch.float32


def test_extract_falls_back_to_public_forward_for_unknown_model_class() -> None:
    # Given: a structurally similar model whose class is outside the audited contract.
    model = UnsupportedModel()
    adapter = DINOv3EarlyExitAdapter(model, layers=(0, 2))

    # When: features are extracted.
    result = adapter.extract(torch.ones((1, 3, 4, 4)))

    # Then: the public full-forward path is used and returns both requested grids.
    assert not result.used_early_exit
    assert model.forward_calls == 1
    assert len(result.grids) == 2
    assert [layer.calls for layer in model.layer] == [1, 1, 1]


def test_extract_falls_back_when_transformers_version_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the expected model class under an unaudited Transformers version.
    monkeypatch.setattr(darc_backbone, "_installed_transformers_version", lambda: "4.57.0")
    model = DINOv3ViTModel()
    adapter = DINOv3EarlyExitAdapter(model, layers=(0,))

    # When: a feature is requested.
    result = adapter.extract(torch.ones((1, 3, 4, 4)))

    # Then: internals are bypassed in favor of the stable public API.
    assert not result.used_early_exit
    assert model.forward_calls == 1


def test_extract_falls_back_when_internal_state_shape_changes() -> None:
    # Given: an audited model whose first internal block violates the token-shape contract once.
    model = DINOv3ViTModel()
    model.layer[0] = _MalformedOnceLayer(1.0)
    adapter = DINOv3EarlyExitAdapter(model, layers=(0,))

    # When: extraction detects the incompatible internal state.
    result = adapter.extract(torch.ones((1, 3, 4, 4)))

    # Then: it retries through the public API and labels the chosen path truthfully.
    assert not result.used_early_exit
    assert model.forward_calls == 1


def test_public_fallback_rejects_unavailable_raw_final_layer() -> None:
    # Given: an unaudited model and a request for raw final-block parity.
    model = UnsupportedModel()
    adapter = DINOv3EarlyExitAdapter(model, layers=(2,), normalize_final=False)

    # When/Then: extraction refuses to mislabel the normalized public state as raw.
    with pytest.raises(DarcBackboneError, match="raw final-layer parity"):
        adapter.extract(torch.ones((1, 3, 4, 4)))


def test_pytree_compat_filters_metadata_unsupported_by_torch_21(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the private Torch 2.1 registration API used by the dsba3 container.
    calls: list[dict[str, object]] = []

    def private_register(
        typ: object,
        flatten_fn: object,
        unflatten_fn: object,
        *,
        to_dumpable_context: object = None,
        from_dumpable_context: object = None,
    ) -> None:
        calls.append(
            {
                "typ": typ,
                "flatten_fn": flatten_fn,
                "unflatten_fn": unflatten_fn,
                "to_dumpable_context": to_dumpable_context,
                "from_dumpable_context": from_dumpable_context,
            },
        )

    pytree = SimpleNamespace(_register_pytree_node=private_register)
    monkeypatch.setattr(darc_backbone, "import_module", lambda _name: pytree)

    # When: Transformers calls the shim with metadata introduced after Torch 2.1.
    darc_backbone.patch_torch_pytree_compat()
    pytree.register_pytree_node(
        int,
        "flatten",
        "unflatten",
        serialized_type_name="builtins.int",
        to_dumpable_context="dump",
    )

    # Then: unsupported metadata is filtered while supported callbacks are retained.
    assert calls == [
        {
            "typ": int,
            "flatten_fn": "flatten",
            "unflatten_fn": "unflatten",
            "to_dumpable_context": "dump",
            "from_dumpable_context": None,
        },
    ]
