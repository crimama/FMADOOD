"""DINOv2 preprocessing and feature extraction wrapper."""

from __future__ import annotations

from contextlib import nullcontext
from inspect import getsource
from types import MethodType
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA

from torchvision import transforms

from fmad.backbones.base import _validate_resolution
from fmad.backbones.loader import load_dinov2_model


def _validate_inference_options(
    device: str,
    feature_output_device: str,
    inference_autocast_dtype: Optional[str],
) -> tuple[torch.device, str]:
    """Validate the opt-in DINOv2 output and autocast settings."""
    try:
        model_device = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"unsupported DINOv2 device: {device!r}") from exc
    if model_device.type not in {"cpu", "cuda"}:
        raise ValueError("DINOv2 device must be CPU or CUDA")

    if feature_output_device not in {"cpu", "model"}:
        raise ValueError("feature_output_device must be 'cpu' or 'model'")

    if inference_autocast_dtype is None:
        inference_autocast_dtype = "none"
    if inference_autocast_dtype not in {"none", "fp16", "bf16"}:
        raise ValueError("inference_autocast_dtype must be 'none', 'fp16', or 'bf16'")
    if inference_autocast_dtype != "none" and model_device.type not in {"cpu", "cuda"}:
        raise ValueError("DINOv2 inference autocast requires a CPU or CUDA device")
    if inference_autocast_dtype == "fp16" and model_device.type != "cuda":
        raise ValueError("fp16 inference autocast is only supported for CUDA DINOv2 devices")

    return model_device, inference_autocast_dtype


def _validate_execution_options(
    model_device: torch.device,
    inference_autocast_dtype: str,
    inference_attention_backend: str,
    inference_compile_mode: str,
) -> tuple[str, str]:
    """Validate opt-in execution paths without selecting a fallback."""
    if inference_attention_backend not in {"default", "sdpa_flash"}:
        raise ValueError(
            "inference_attention_backend must be 'default' or 'sdpa_flash'"
        )
    if inference_attention_backend == "sdpa_flash":
        if model_device.type != "cuda":
            raise ValueError("sdpa_flash attention requires a CUDA DINOv2 device")
        if inference_autocast_dtype not in {"fp16", "bf16"}:
            raise ValueError("sdpa_flash attention requires fp16 or bf16 autocast")
        if not callable(getattr(torch.backends.cuda, "sdp_kernel", None)):
            raise RuntimeError("sdpa_flash requires torch.backends.cuda.sdp_kernel")

    if inference_compile_mode not in {"none", "default"}:
        raise ValueError("inference_compile_mode must be 'none' or 'default'")
    if inference_compile_mode == "default" and not callable(getattr(torch, "compile", None)):
        raise RuntimeError("inference_compile_mode='default' requires torch.compile")

    return inference_attention_backend, inference_compile_mode


class DINOv2Wrapper:
    def __init__(
        self,
        model_name: str,
        device: str,
        smaller_edge_size: int = 224,
        half_precision: bool = False,
        feature_layers: Optional[Sequence[int]] = None,
        feature_output_device: str = "cpu",
        inference_autocast_dtype: Optional[str] = "none",
        inference_attention_backend: str = "default",
        inference_compile_mode: str = "none",
    ):
        defaults = (2, 5, 8, 11) if "_vits14" in model_name else (5, 11, 17, 23)
        self.feature_layers = tuple(feature_layers or defaults)
        self.device = device
        self._model_device, self.inference_autocast_dtype = _validate_inference_options(
            device, feature_output_device, inference_autocast_dtype
        )
        (
            self.inference_attention_backend,
            self.inference_compile_mode,
        ) = _validate_execution_options(
            self._model_device,
            self.inference_autocast_dtype,
            inference_attention_backend,
            inference_compile_mode,
        )
        self.feature_output_device = feature_output_device
        self.smaller_edge_size = _validate_resolution(smaller_edge_size)
        self.half_precision = half_precision
        self.model_name = model_name
        self._transform_cache: dict[int, transforms.Compose] = {}
        self.transform = self._get_transform(self.smaller_edge_size)
        self._get_intermediate_layers = None
        self._compiled_get_intermediate_layers = None
        self._compiled_get_cls_intermediate_layers = None
        self.compile_tensor_dispatch = None
        self.model = self.load_model()
        depth = len(getattr(self.model, "blocks", ()))
        if not self.feature_layers or min(self.feature_layers) < 0:
            raise ValueError("feature_layers must contain non-negative indices")
        if depth and max(self.feature_layers) >= depth:
            raise ValueError(f"feature layer index must be less than model depth {depth}")

    def load_model(self):
        model = load_dinov2_model(self.model_name)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        self.transform = self._get_transform(self.smaller_edge_size)
        model = model.to(self.device)
        if self.inference_attention_backend == "sdpa_flash":
            self._route_attention_through_sdpa(model)
        self._get_intermediate_layers = model.get_intermediate_layers
        if self.inference_compile_mode == "default":
            self._compile_intermediate_layer_calls(model)
        return model

    @staticmethod
    def _route_attention_through_sdpa(model) -> None:
        """Route DINOv2's memory-efficient attention modules through SDPA."""
        attention_modules = [
            module
            for module in model.modules()
            if all(hasattr(module, name) for name in ("qkv", "proj", "num_heads"))
        ]
        if not attention_modules:
            raise RuntimeError(
                "sdpa_flash was requested, but no supported DINOv2 attention modules were found"
            )

        routed_count = 0
        for module in attention_modules:
            module_type = type(module)
            sdpa_forward = None
            for base_type in module_type.__mro__:
                if base_type.__name__ != "Attention":
                    continue
                try:
                    source = getsource(base_type.forward)
                except (OSError, TypeError) as exc:
                    raise RuntimeError(
                        "could not verify the DINOv2 SDPA attention implementation"
                    ) from exc
                if "scaled_dot_product_attention" in source:
                    sdpa_forward = base_type.forward
                    break
            if sdpa_forward is None:
                raise RuntimeError(
                    f"sdpa_flash does not support attention module {module_type.__module__}."
                    f"{module_type.__name__}"
                )

            # MemEffAttention normally selects xFormers when installed. Bind the
            # verified parent Attention implementation on this instance only;
            # parameters and the model state_dict remain unchanged.
            module.forward = MethodType(sdpa_forward, module)
            routed_count += 1

        if routed_count != len(attention_modules):
            raise RuntimeError(
                "sdpa_flash could not route every DINOv2 attention module through SDPA"
            )

    def _compile_intermediate_layer_calls(self, model) -> None:
        """Compile fixed patch and CLS extraction calls with no graph fallback."""
        self.compile_tensor_dispatch = self._route_compile_tensor_forwards(model)
        get_intermediate_layers = model.get_intermediate_layers
        feature_layers = tuple(self.feature_layers)

        def extract_features(image_batch):
            return get_intermediate_layers(image_batch, feature_layers)

        def extract_cls(image_batch):
            return get_intermediate_layers(
                image_batch, n=1, return_class_token=True
            )

        try:
            self._compiled_get_intermediate_layers = torch.compile(
                extract_features,
                backend="inductor",
                fullgraph=True,
                dynamic=False,
            )
            self._compiled_get_cls_intermediate_layers = torch.compile(
                extract_cls,
                backend="inductor",
                fullgraph=True,
                dynamic=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "failed to configure strict DINOv2 torch.compile inference"
            ) from exc

    @staticmethod
    def _method_source(method, *, required: tuple[str, ...], description: str) -> str:
        try:
            source = getsource(method)
        except (OSError, TypeError) as exc:
            raise RuntimeError(
                f"could not verify the upstream {description} implementation for compile"
            ) from exc
        if any(fragment not in source for fragment in required):
            raise RuntimeError(
                f"unsupported upstream {description} implementation for compile tensor dispatch"
            )
        return source

    @classmethod
    def _route_compile_tensor_forwards(cls, model) -> dict:
        """Bind upstream parent methods for the tensor-only frozen inference path.

        DINOv2's NestedTensorBlock and, without xFormers, MemEffAttention use
        zero-argument ``super()`` fallbacks. Dynamo 2.1 can fail while guarding
        those calls. Binding the same parent methods on each instance preserves
        the tensor path while avoiding changes to parameters or module classes.
        """
        if model.training or any(module.training for module in model.modules()):
            raise RuntimeError("compile tensor dispatch requires an eval-mode DINOv2 model")
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("compile tensor dispatch requires frozen DINOv2 parameters")

        nested_blocks = []
        mem_eff_attention = []
        for module in model.modules():
            mro = type(module).__mro__
            if any(base.__name__ == "NestedTensorBlock" for base in mro):
                nested_blocks.append(module)
            if any(base.__name__ == "MemEffAttention" for base in mro):
                mem_eff_attention.append(module)

        if not nested_blocks:
            raise RuntimeError(
                "compile tensor dispatch found no supported DINOv2 NestedTensorBlock modules"
            )

        for module in nested_blocks:
            mro = type(module).__mro__
            block_type = next(
                (base for base in mro[1:] if base.__name__ == "Block"), None
            )
            if block_type is None or not callable(getattr(block_type, "forward", None)):
                raise RuntimeError("NestedTensorBlock has no supported parent Block.forward")
            cls._method_source(
                type(module).forward,
                required=(
                    "isinstance(x_or_x_list, Tensor)",
                    "return super().forward(x_or_x_list)",
                ),
                description="NestedTensorBlock.forward",
            )
            cls._method_source(
                block_type.forward,
                required=("def forward(self, x: Tensor)", "self.attn(self.norm1(x))"),
                description="Block.forward",
            )
            module.forward = MethodType(block_type.forward, module)

        attention_route = "upstream_xformers"
        routed_attention_count = 0
        if mem_eff_attention:
            mem_eff_type = next(
                base
                for base in type(mem_eff_attention[0]).__mro__
                if base.__name__ == "MemEffAttention"
            )
            attention_globals = getattr(mem_eff_type.forward, "__globals__", {})
            if "XFORMERS_AVAILABLE" not in attention_globals:
                raise RuntimeError(
                    "could not resolve upstream MemEffAttention XFORMERS_AVAILABLE state"
                )
            xformers_available = attention_globals["XFORMERS_AVAILABLE"]
            if not isinstance(xformers_available, bool):
                raise RuntimeError("unsupported non-boolean DINOv2 XFORMERS_AVAILABLE state")
            if not xformers_available:
                cls._method_source(
                    mem_eff_type.forward,
                    required=(
                        "if not XFORMERS_AVAILABLE",
                        "if attn_bias is not None",
                        "return super().forward(x)",
                    ),
                    description="MemEffAttention.forward",
                )
                attention_type = next(
                    (
                        base
                        for base in mem_eff_type.__mro__[1:]
                        if base.__name__ == "Attention"
                    ),
                    None,
                )
                if attention_type is None or not callable(
                    getattr(attention_type, "forward", None)
                ):
                    raise RuntimeError("MemEffAttention has no supported parent Attention.forward")
                cls._method_source(
                    attention_type.forward,
                    required=("def forward(self, x: Tensor", "scaled_dot_product_attention"),
                    description="Attention.forward",
                )
                for module in mem_eff_attention:
                    module.forward = MethodType(attention_type.forward, module)
                attention_route = "parent_attention_sdpa_no_xformers"
                routed_attention_count = len(mem_eff_attention)

        return {
            "policy": "compile_only_tensor_dispatch",
            "nested_tensor_block_forward": "parent_block_forward",
            "nested_tensor_block_count": len(nested_blocks),
            "attention_route": attention_route,
            "mem_eff_attention_routed_count": routed_attention_count,
            "no_fallback": True,
            "model_state_unchanged": True,
        }

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
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
            self._transform_cache[resolution] = transform
        return transform

    def set_resolution(self, smaller_edge_size: int) -> None:
        """Change the default preprocessing resolution without reloading DINOv2."""
        self.smaller_edge_size = _validate_resolution(smaller_edge_size)
        self.transform = self._get_transform(self.smaller_edge_size)

    def _autocast_context(self):
        if self.inference_autocast_dtype == "none":
            return nullcontext()
        dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.inference_autocast_dtype]
        return torch.autocast(device_type=self._model_device.type, dtype=dtype)

    def _attention_context(self):
        if self.inference_attention_backend == "default":
            return nullcontext()
        return torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=False,
        )

    def prepare_image(self, img, *, resolution: int | None = None):
        if isinstance(img, (str, bytes)):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        target_resolution = (
            self.smaller_edge_size if resolution is None else _validate_resolution(resolution)
        )
        image_tensor = self._get_transform(target_resolution)(img)
        height, width = image_tensor.shape[1:]
        patch_size = int(self.model.patch_size)
        cropped_width, cropped_height = width - width % patch_size, height - height % patch_size
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]
        return image_tensor, (cropped_height // patch_size, cropped_width // patch_size)

    def extract_features(self, image_tensor):
        with torch.inference_mode(), self._autocast_context(), self._attention_context():
            image_batch = image_tensor.unsqueeze(0).to(self.device)
            if self.half_precision:
                image_batch = image_batch.half()
            if self.inference_compile_mode == "default":
                tokens = self._compiled_get_intermediate_layers(image_batch)
            else:
                tokens = self._get_intermediate_layers(image_batch, list(self.feature_layers))
            features = [token.squeeze().float() for token in tokens]
            if self.feature_output_device == "model":
                return features
            return [feature.cpu().numpy() for feature in features]

    def extract_cls_features(self, image_tensor):
        with torch.inference_mode(), self._autocast_context(), self._attention_context():
            image_batch = image_tensor.unsqueeze(0).to(self.device)
            if self.half_precision:
                image_batch = image_batch.half()
            if self.inference_compile_mode == "default":
                outputs = self._compiled_get_cls_intermediate_layers(image_batch)
            else:
                outputs = self._get_intermediate_layers(
                    image_batch, n=1, return_class_token=True
                )
            _, cls_token = outputs[0]
        return cls_token.squeeze(0).float()

    def get_embedding_visualization(self, tokens, grid_size, resized_mask=None, normalize=True):
        pca = PCA(n_components=3, svd_solver="randomized")
        if resized_mask is not None:
            tokens = tokens[resized_mask]
        reduced = pca.fit_transform(tokens.astype(np.float32))
        if resized_mask is not None:
            restored = np.zeros((*resized_mask.shape, 3), dtype=reduced.dtype)
            restored[resized_mask] = reduced
            reduced = restored
        reduced = reduced.reshape((*grid_size, -1))
        if normalize:
            return (reduced - np.min(reduced)) / (np.max(reduced) - np.min(reduced))
        return reduced

    def compute_background_mask_from_image(self, image, threshold=10, masking_type=None):
        image_tensor, grid_size = self.prepare_image(image)
        return self.compute_background_mask(
            self.extract_features(image_tensor), grid_size, threshold, masking_type
        )

    def compute_background_mask(
        self, img_features, grid_size, threshold=1, masking_type=False, kernel_size=3
    ):
        if isinstance(img_features, torch.Tensor):
            img_features = img_features.detach().float().cpu().numpy()
        first_pc = PCA(n_components=1, svd_solver="randomized").fit_transform(
            img_features.astype(np.float32)
        )
        if masking_type:
            mask = first_pc.squeeze() > threshold
            masked, unmasked = img_features[mask], img_features[~mask]
            masked_variance = np.median(np.var(masked, axis=0)) if masked.size else 0
            unmasked_variance = np.median(np.var(unmasked, axis=0)) if unmasked.size else 0
            if masked_variance < unmasked_variance:
                mask = ~mask
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        else:
            mask = np.ones_like(first_pc, dtype=bool)
        return mask.squeeze()
