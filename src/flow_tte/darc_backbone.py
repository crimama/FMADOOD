from __future__ import annotations

from importlib import import_module, metadata
from typing import TYPE_CHECKING, Final, NamedTuple, Protocol, final, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

import torch
from torch import nn

_AUDITED_TRANSFORMERS_VERSION: Final = "4.56.2"


class DarcBackboneError(RuntimeError): ...


class _CompatibilityError(RuntimeError): ...


@runtime_checkable
class _ConfigLike(Protocol):
    patch_size: int
    num_register_tokens: int


@runtime_checkable
class _PatchProjectionLike(Protocol):
    weight: torch.Tensor


@runtime_checkable
class _EmbeddingsLike(Protocol):
    patch_embeddings: _PatchProjectionLike

    def __call__(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


@runtime_checkable
class _EarlyExitModelLike(Protocol):
    config: _ConfigLike
    embeddings: _EmbeddingsLike
    rope_embeddings: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
    layer: Sequence[
        Callable[
            [torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor] | None],
            torch.Tensor,
        ]
    ]
    norm: Callable[[torch.Tensor], torch.Tensor]


@runtime_checkable
class _ForwardOutputLike(Protocol):
    hidden_states: tuple[torch.Tensor, ...]
    last_hidden_state: torch.Tensor


@runtime_checkable
class _PublicModelLike(Protocol):
    def __call__(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ) -> _ForwardOutputLike: ...


class FeatureGeometry(NamedTuple):
    input_size: tuple[int, int]
    realized_size: tuple[int, int]
    grid_size: tuple[int, int]
    patch_size: int


class DINOv3PatchFeatures(NamedTuple):
    layers: tuple[int, ...]
    grids: tuple[torch.Tensor, ...]
    geometry: FeatureGeometry
    used_early_exit: bool


def patch_torch_pytree_compat() -> None:
    pytree = import_module("torch.utils._pytree")
    namespace = vars(pytree)
    if "register_pytree_node" not in namespace and "_register_pytree_node" in namespace:
        private_register = namespace["_register_pytree_node"]

        def register_pytree_node(
            typ: type,
            flatten_fn: Callable[..., object],
            unflatten_fn: Callable[..., object],
            **kwargs: object,
        ) -> object:
            supported_kwargs = {
                name: kwargs[name]
                for name in ("to_dumpable_context", "from_dumpable_context")
                if name in kwargs
            }
            return private_register(typ, flatten_fn, unflatten_fn, **supported_kwargs)

        namespace["register_pytree_node"] = register_pytree_node


def _installed_transformers_version() -> str | None:
    try:
        return metadata.version("transformers")
    except metadata.PackageNotFoundError:
        return None


def _call_public(model: _PublicModelLike, pixels: torch.Tensor) -> _ForwardOutputLike:
    return model(pixels, output_hidden_states=True, return_dict=True)


@final
class DINOv3EarlyExitAdapter:
    """Extract patch grids while stopping after the deepest requested DINOv3 block."""

    def __init__(
        self,
        model: nn.Module,
        layers: Sequence[int],
        output_dtype: torch.dtype = torch.float32,
        *,
        normalize_final: bool = True,
        normalize_features: bool = True,
    ) -> None:
        requested = tuple(int(layer) for layer in layers)
        if not requested or min(requested) < 0 or len(set(requested)) != len(requested):
            message = "layers must be a non-empty sequence of unique, non-negative indices"
            raise DarcBackboneError(message)
        if not torch.empty((), dtype=output_dtype).is_floating_point():
            message = "output_dtype must be a floating-point torch dtype"
            raise DarcBackboneError(message)
        self._model = model
        self._layers = requested
        self._output_dtype = output_dtype
        self._normalize_final = normalize_final
        self._normalize_features = normalize_features

    @torch.inference_mode()
    def extract(self, pixel_values: torch.Tensor) -> DINOv3PatchFeatures:
        geometry, register_count = self._geometry(pixel_values)
        grids: tuple[torch.Tensor, ...] | None = None
        model = self._audited_model()
        if model is not None:
            try:
                states = self._run_early_exit(model, pixel_values)
                grids = tuple(self._patch_grid(state, geometry, register_count) for state in states)
            except (AttributeError, IndexError, TypeError, _CompatibilityError):
                grids = None
        used_early_exit = grids is not None
        if grids is None:
            states = self._run_public_forward(pixel_values)
            grids = tuple(self._patch_grid(state, geometry, register_count) for state in states)
        return DINOv3PatchFeatures(self._layers, grids, geometry, used_early_exit)

    def _geometry(self, pixels: torch.Tensor) -> tuple[FeatureGeometry, int]:
        if pixels.ndim != 4:
            message = f"pixel_values must have shape [B,C,H,W], got {tuple(pixels.shape)}"
            raise DarcBackboneError(message)
        config = getattr(self._model, "config", None)
        if not isinstance(config, _ConfigLike):
            message = "model.config must expose integer patch_size and num_register_tokens"
            raise DarcBackboneError(message)
        height, width = int(pixels.shape[-2]), int(pixels.shape[-1])
        patch_size = int(config.patch_size)
        if patch_size <= 0 or height < patch_size or width < patch_size:
            message = "input dimensions and model patch_size do not form a non-empty grid"
            raise DarcBackboneError(message)
        grid_size = (height // patch_size, width // patch_size)
        realized_size = (grid_size[0] * patch_size, grid_size[1] * patch_size)
        register_count = int(config.num_register_tokens)
        geometry = FeatureGeometry((height, width), realized_size, grid_size, patch_size)
        return geometry, register_count

    def _audited_model(self) -> _EarlyExitModelLike | None:
        if _installed_transformers_version() != _AUDITED_TRANSFORMERS_VERSION:
            return None
        if type(self._model).__name__ != "DINOv3ViTModel":
            return None
        if not isinstance(self._model, _EarlyExitModelLike):
            return None
        if max(self._layers) >= len(self._model.layer):
            message = "requested DINOv3 layer exceeds the model depth"
            raise DarcBackboneError(message)
        return self._model

    def _run_early_exit(
        self,
        model: _EarlyExitModelLike,
        pixels: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        values = pixels.to(dtype=model.embeddings.patch_embeddings.weight.dtype)
        hidden = model.embeddings(values, None)
        position = model.rope_embeddings(values)
        selected: dict[int, torch.Tensor] = {}
        final_index = len(model.layer) - 1
        for index, layer in enumerate(model.layer):
            hidden = layer(hidden, None, position)
            if index in self._layers:
                selected[index] = (
                    model.norm(hidden) if index == final_index and self._normalize_final else hidden
                )
            if index >= max(self._layers):
                break
        if len(selected) != len(self._layers):
            raise _CompatibilityError
        return tuple(selected[index] for index in self._layers)

    def _run_public_forward(self, pixels: torch.Tensor) -> tuple[torch.Tensor, ...]:
        output = _call_public(self._model, pixels)
        depth = len(output.hidden_states) - 1
        if not self._normalize_final and depth - 1 in self._layers:
            message = "raw final-layer parity requires the audited DINOv3 early-exit path"
            raise DarcBackboneError(message)
        if max(self._layers) >= depth:
            message = "requested DINOv3 layer exceeds public hidden-state depth"
            raise DarcBackboneError(message)
        return tuple(
            output.last_hidden_state
            if index == depth - 1 and self._normalize_final
            else output.hidden_states[index + 1]
            for index in self._layers
        )

    def _patch_grid(
        self,
        state: torch.Tensor,
        geometry: FeatureGeometry,
        register_count: int,
    ) -> torch.Tensor:
        patch_count = geometry.grid_size[0] * geometry.grid_size[1]
        start = 1 + max(0, register_count)
        if state.ndim != 3 or int(state.shape[1]) < start + patch_count:
            raise _CompatibilityError
        patches = state[:, start : start + patch_count]
        grid = patches.reshape(
            int(state.shape[0]),
            geometry.grid_size[0],
            geometry.grid_size[1],
            int(state.shape[-1]),
        ).to(dtype=torch.float32)
        if self._normalize_features:
            grid = torch.nn.functional.normalize(grid, p=2, dim=-1)
        return grid.to(dtype=self._output_dtype).detach()


patch_torch_pytree_compat()
