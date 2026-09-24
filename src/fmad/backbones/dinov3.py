"""DINOv3 feature backbone with lazy loading and multi-scale support."""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms

from fmad.backbones.base import BaseBackbone, _validate_resolution
from fmad.registry import BACKBONE_REGISTRY

_DEFAULT_MODEL_IDS: Dict[str, str] = {
    "dinov3_vitl16": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
    "dinov3_vith16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "dinov3_vith16+": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "dinov3_vitb16": "camenduru/dinov3-vitb16-pretrain-lvd1689m",
}
_TRUE_ENV_VALUES = {"1", "true", "yes", "y", "on"}


def patch_torch_pytree_compat() -> None:
    """Ensure compatibility between PyTorch pytree and transformers."""
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


def resolve_dinov3_model_id(model_name: str) -> str:
    override = os.environ.get("DINOV3_MODEL_ID")
    if override:
        return override
    if "/" in model_name:
        return model_name
    if model_name in _DEFAULT_MODEL_IDS:
        return _DEFAULT_MODEL_IDS[model_name]
    supported = ", ".join(sorted(_DEFAULT_MODEL_IDS))
    raise ValueError(f"Unknown DINOv3 model name: {model_name}. Supported: {supported}.")


@BACKBONE_REGISTRY.register("dinov3_vitl16")
@BACKBONE_REGISTRY.register("dinov3_vith16plus")
class DINOv3Backbone(BaseBackbone):
    """Lazy DINOv3 feature extractor for multi-scale reference matching."""

    def __init__(
        self,
        model_name: str,
        device: str,
        smaller_edge_size: int = 448,
        feature_layers: Optional[Sequence[int]] = None,
        half_precision: bool = False,
    ) -> None:
        super().__init__(model_name, device, smaller_edge_size)
        defaults = (7, 15, 23, 31) if "vith16plus" in model_name else (5, 11, 17, 23)
        self.feature_layers = tuple(feature_layers or defaults)
        self.half_precision = half_precision
        self.model_id = resolve_dinov3_model_id(model_name)
        self._model: Optional[torch.nn.Module] = None
        self._transform_cache: dict[int, transforms.Compose] = {}
        self._patch_size: Optional[int] = None
        self._num_register_tokens = 0

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        patch_torch_pytree_compat()
        auto_model = import_module("transformers").AutoModel
        offline = os.environ.get("FMAD_DINOV3_OFFLINE", "1").strip().lower() in _TRUE_ENV_VALUES
        try:
            model = auto_model.from_pretrained(self.model_id, local_files_only=offline)
        except Exception:
            # Fallback to online if local_files_only failed and not strictly forced
            model = auto_model.from_pretrained(self.model_id)

        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self._model = model.to(self.device)
        self._patch_size = int(getattr(model.config, "patch_size", 16))
        self._num_register_tokens = int(getattr(model.config, "num_register_tokens", 0))

    def _get_transform(self, resolution: int) -> transforms.Compose:
        resolution = _validate_resolution(resolution)
        transform = self._transform_cache.get(resolution)
        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        size=resolution,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                        antialias=True,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                ]
            )
            self._transform_cache[resolution] = transform
        return transform

    def set_resolution(self, smaller_edge_size: int) -> None:
        super().set_resolution(smaller_edge_size)

    def prepare_image(
        self, img: Union[str, Path, np.ndarray, Image.Image], *, resolution: Optional[int] = None
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        self._ensure_loaded()
        if isinstance(img, (str, Path)):
            pil_image = Image.open(str(img)).convert("RGB")
        elif isinstance(img, np.ndarray):
            pil_image = Image.fromarray(img).convert("RGB")
        elif isinstance(img, Image.Image):
            pil_image = img.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(img)}")

        target_res = (
            self.smaller_edge_size if resolution is None else _validate_resolution(resolution)
        )
        image_tensor = self._get_transform(target_res)(pil_image)
        patch_size = self._patch_size or 16
        height, width = image_tensor.shape[1:]
        cropped_h = height - (height % patch_size)
        cropped_w = width - (width % patch_size)
        image_tensor = image_tensor[:, :cropped_h, :cropped_w]
        grid_size = (cropped_h // patch_size, cropped_w // patch_size)
        return image_tensor, grid_size

    def extract_features(self, image_tensor: torch.Tensor) -> List[np.ndarray]:
        self._ensure_loaded()
        with torch.inference_mode():
            batch = image_tensor.unsqueeze(0).to(self.device)
            if self.half_precision:
                batch = batch.half()
            outputs = self._model(batch, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            # hidden_states[0] is embedding output, hidden_states[layer+1] is Layer layer output
            start_token = 1 + max(0, self._num_register_tokens)
            features = []
            for layer in self.feature_layers:
                layer_tokens = hidden_states[layer + 1].squeeze(0)[start_token:]
                features.append(layer_tokens.float().cpu().numpy())
            return features

    def extract_cls_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        self._ensure_loaded()
        with torch.inference_mode():
            batch = image_tensor.unsqueeze(0).to(self.device)
            if self.half_precision:
                batch = batch.half()
            outputs = self._model(batch, output_hidden_states=True)
            # CLS token is token index 0 of the last hidden state
            cls_token = outputs.hidden_states[-1].squeeze(0)[0].detach().cpu()
            return cls_token

    def compute_background_mask(
        self, features, grid_size, threshold: float = 1.0, masking_type=False, kernel_size=3, **kwargs
    ) -> np.ndarray:
        first_pc = PCA(n_components=1, svd_solver="randomized").fit_transform(
            features.astype(np.float32)
        )
        if masking_type:
            mask = first_pc.squeeze() > threshold
            masked, unmasked = features[mask], features[~mask]
            masked_var = np.median(np.var(masked, axis=0)) if masked.size else 0
            unmasked_var = np.median(np.var(unmasked, axis=0)) if unmasked.size else 0
            if masked_var < unmasked_var:
                mask = ~mask
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        else:
            mask = np.ones_like(first_pc, dtype=bool)
        return mask.squeeze()
