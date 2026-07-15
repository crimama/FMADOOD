from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

_DEFAULT_MODEL_IDS: Dict[str, str] = {
    "dinov3_vith16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "dinov3_vith16+": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "dinov3_vitb16": "camenduru/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_vitl16": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
}
_TRUE_ENV_VALUES = {"1", "true", "yes", "y", "on"}
ImageInput = Union[str, Path, np.ndarray, Image.Image]


class ImageTransform(Protocol):
    def __call__(self, img: Image.Image) -> torch.Tensor: ...


class UnknownDINOv3ModelError(RuntimeError):
    """Raised when the requested DINOv3 backbone name is not supported."""


def is_dinov3_model_name(model_name: str) -> bool:
    normalized = model_name.lower()
    return normalized.startswith("dinov3_") or "dinov3" in normalized


def resolve_dinov3_model_id(model_name: str) -> str:
    override = os.environ.get("DINOV3_MODEL_ID")
    if override:
        return override
    if "/" in model_name:
        return model_name
    if model_name in _DEFAULT_MODEL_IDS:
        return _DEFAULT_MODEL_IDS[model_name]
    supported = ", ".join(sorted(_DEFAULT_MODEL_IDS))
    message = f"Unknown DINOv3 model name: {model_name}. Supported aliases: {supported}."
    raise UnknownDINOv3ModelError(message)


def dinov3_hidden_state_index(layer_index: int) -> int:
    return layer_index + 1


def dinov3_patch_token_start(num_register_tokens: int) -> int:
    return 1 + max(0, int(num_register_tokens))


def env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_ENV_VALUES


def patch_torch_pytree_compat() -> None:
    pytree = torch.utils._pytree  # noqa: SLF001
    if not hasattr(pytree, "register_pytree_node") and hasattr(
        pytree,
        "_register_pytree_node",
    ):
        private_register = pytree._register_pytree_node  # noqa: SLF001

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

        pytree.register_pytree_node = register_pytree_node


class DINOv3Backbone:
    """HuggingFace DINOv3 ViT wrapper matching the FlowTTE backbone protocol."""

    def __init__(
        self,
        model_name: str,
        device: str,
        smaller_edge_size: int = 448,
        feature_layers: Sequence[int] = (5, 11, 17, 23),
    ) -> None:
        self.model_name = model_name
        self.model_id = resolve_dinov3_model_id(model_name)
        self.device = device
        self.smaller_edge_size = smaller_edge_size
        self.feature_layers = tuple(int(layer) for layer in feature_layers)
        self._model: Optional[torch.nn.Module] = None
        self._transform: Optional[ImageTransform] = None
        self._patch_size: Optional[int] = None
        self._num_register_tokens = 0

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        local_files_only = env_flag_enabled("FMAD_DINOV3_OFFLINE")
        patch_torch_pytree_compat()
        auto_model = import_module("transformers").AutoModel
        model = auto_model.from_pretrained(
            self.model_id,
            local_files_only=local_files_only,
        )
        model.eval()
        self._model = model.to(self.device)
        self._patch_size = int(model.config.patch_size)
        self._num_register_tokens = int(getattr(model.config, "num_register_tokens", 0))

    def _ensure_transform(self) -> ImageTransform:
        cached = self._transform
        if cached is not None:
            return cached
        from torchvision import transforms  # noqa: PLC0415

        composed = transforms.Compose(
            [
                transforms.Resize(
                    size=self.smaller_edge_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ],
        )

        def apply_transform(img: Image.Image) -> torch.Tensor:
            tensor = composed(img)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("DINOv3 transform did not return a tensor.")
            return tensor

        self._transform = apply_transform
        return apply_transform

    def set_resolution(self, smaller_edge_size: int) -> None:
        if smaller_edge_size == self.smaller_edge_size:
            return
        self.smaller_edge_size = smaller_edge_size
        self._transform = None

    def prepare_image(self, img: ImageInput) -> Tuple[torch.Tensor, Tuple[int, int]]:
        self._ensure_loaded()
        if isinstance(img, (str, Path)):
            pil_image = Image.open(str(img)).convert("RGB")
        elif isinstance(img, np.ndarray):
            pil_image = Image.fromarray(img).convert("RGB")
        else:
            pil_image = img.convert("RGB")
        image_tensor = self._ensure_transform()(pil_image)
        patch_size = self._require_patch_size()
        height, width = image_tensor.shape[1:]
        cropped_height = height - height % patch_size
        cropped_width = width - width % patch_size
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]
        return image_tensor, (cropped_height // patch_size, cropped_width // patch_size)

    def extract_features(self, image_tensor: torch.Tensor) -> List[np.ndarray]:
        hidden_states = self._hidden_states(image_tensor)
        start = dinov3_patch_token_start(self._num_register_tokens)
        return [
            hidden_states[dinov3_hidden_state_index(layer)].squeeze(0)[start:].cpu().numpy()
            for layer in self.feature_layers
        ]

    def extract_features_with_context_tokens(
        self,
        image_tensor: torch.Tensor,
        context_source: str,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        hidden_states = self._hidden_states(image_tensor)
        start = dinov3_patch_token_start(self._num_register_tokens)
        layer_features = [
            hidden_states[dinov3_hidden_state_index(layer)].squeeze(0)[start:].cpu().numpy()
            for layer in self.feature_layers
        ]
        context_tokens = self._context_tokens_from_hidden_states(hidden_states, context_source)
        return layer_features, context_tokens.numpy().astype(np.float32, copy=False)

    def extract_cls_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self._hidden_states(image_tensor)
        return hidden_states[-1].squeeze(0)[0].detach().cpu()

    def extract_context_features(
        self,
        image_tensor: torch.Tensor,
        context_source: str,
    ) -> torch.Tensor:
        hidden_states = self._hidden_states(image_tensor)
        tokens = hidden_states[-1].squeeze(0).detach().cpu()
        source = context_source.lower()
        cls = tokens[0]
        if source == "cls":
            return cls
        if source not in ("register", "cls_register"):
            message = f"Unsupported DINOv3 context source: {context_source}"
            raise ValueError(message)
        start = dinov3_patch_token_start(self._num_register_tokens)
        registers = tokens[1:start]
        if registers.numel() == 0:
            message = "DINOv3 register context requested, but model has no register tokens."
            raise RuntimeError(message)
        register_mean = registers.mean(dim=0)
        if source == "register":
            return register_mean
        return torch.cat([cls, register_mean], dim=0)

    def extract_context_token_features(
        self,
        image_tensor: torch.Tensor,
        context_source: str,
    ) -> torch.Tensor:
        hidden_states = self._hidden_states(image_tensor)
        return self._context_tokens_from_hidden_states(hidden_states, context_source)

    def _context_tokens_from_hidden_states(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        context_source: str,
    ) -> torch.Tensor:
        tokens = hidden_states[-1].squeeze(0).detach().cpu()
        source = context_source.lower()
        if source == "cls":
            return tokens[:1]
        if source not in ("register", "cls_register"):
            message = f"Unsupported DINOv3 context token source: {context_source}"
            raise ValueError(message)
        start = dinov3_patch_token_start(self._num_register_tokens)
        registers = tokens[1:start]
        if registers.numel() == 0:
            message = "DINOv3 register context requested, but model has no register tokens."
            raise RuntimeError(message)
        if source == "register":
            return registers
        return torch.cat([tokens[:1], registers], dim=0)

    def _hidden_states(self, image_tensor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        self._ensure_loaded()
        model = self._require_model()
        with torch.inference_mode():
            image_batch = image_tensor.unsqueeze(0).to(self.device)
            outputs = model(
                image_batch,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("DINOv3 model did not return hidden states.")
        return tuple(hidden_states)

    def _require_model(self) -> torch.nn.Module:
        self._ensure_loaded()
        if self._model is None:
            raise RuntimeError("DINOv3 model is unavailable before model load.")
        return self._model

    def _require_patch_size(self) -> int:
        self._ensure_loaded()
        if self._patch_size is None:
            raise RuntimeError("DINOv3 patch size is unavailable before model load.")
        return self._patch_size
