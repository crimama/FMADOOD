"""DINOv2 feature backbone with lazy loading and provenance validation."""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from fmad.backbones.base import BaseBackbone
from fmad.registry import BACKBONE_REGISTRY
from fmad.backbones.loader import find_local_dinov2_repo, dinov2_repo_tree_sha256
from fmad.backbones.provenance import backbone_state_sha256
from fmad.backbones.wrapper import (
    DINOv2Wrapper,
    _validate_execution_options,
    _validate_inference_options,
)


@BACKBONE_REGISTRY.register("dinov2_vitl14")
@BACKBONE_REGISTRY.register("dinov2_vitl14_reg")
@BACKBONE_REGISTRY.register("dinov2_vits14")
class DINOv2Backbone(BaseBackbone):
    """Lazy DINOv2 feature extractor with cache and model-state provenance."""

    def __init__(
        self,
        model_name: str,
        device: str,
        smaller_edge_size: int = 448,
        feature_layers: Optional[Sequence[int]] = None,
        expected_repo_tree_sha256: Optional[str] = None,
        expected_state_sha256: Optional[str] = None,
        feature_output_device: str = "cpu",
        inference_autocast_dtype: Optional[str] = "none",
        inference_attention_backend: str = "default",
        inference_compile_mode: str = "none",
    ) -> None:
        super().__init__(model_name, device, smaller_edge_size)
        model_device, self.inference_autocast_dtype = _validate_inference_options(
            device, feature_output_device, inference_autocast_dtype
        )
        (
            self.inference_attention_backend,
            self.inference_compile_mode,
        ) = _validate_execution_options(
            model_device,
            self.inference_autocast_dtype,
            inference_attention_backend,
            inference_compile_mode,
        )
        self.feature_output_device = feature_output_device
        defaults = (2, 5, 8, 11) if "_vits14" in model_name else (5, 11, 17, 23)
        self.feature_layers = tuple(feature_layers or defaults)
        self.expected_repo_tree_sha256 = expected_repo_tree_sha256
        self.expected_state_sha256 = expected_state_sha256
        self._wrapper = None
        self.provenance = None
        self.backbone_repo_tree_sha256 = None
        self.backbone_state_sha256 = None

    def _validate_and_record(self, wrapper) -> None:
        repo = find_local_dinov2_repo()
        repo_hash = dinov2_repo_tree_sha256(repo) if repo is not None else None
        if (
            self.expected_repo_tree_sha256 is not None
            and repo_hash != self.expected_repo_tree_sha256
        ):
            raise RuntimeError("DINOv2 repository tree hash mismatch")
        state_hash = backbone_state_sha256(wrapper.model)
        if self.expected_state_sha256 is not None and state_hash != self.expected_state_sha256:
            raise RuntimeError("DINOv2 backbone state hash mismatch")
        parameter_dtypes = sorted({str(parameter.dtype) for parameter in wrapper.model.parameters()})
        self.provenance = {
            "repo_tree_sha256": repo_hash,
            "state_sha256": state_hash,
            "model_class": wrapper.model.__class__.__name__,
            "patch_size": int(getattr(wrapper.model, "patch_size", -1)),
            "depth": len(getattr(wrapper.model, "blocks", ())),
            "model_parameter_dtypes": parameter_dtypes,
            "weights_frozen": all(
                not parameter.requires_grad for parameter in wrapper.model.parameters()
            ),
            "feature_output_device": self.feature_output_device,
            "feature_output_device_resolved": (
                "cpu" if self.feature_output_device == "cpu" else str(self.device)
            ),
            "feature_output_interface": (
                "numpy.ndarray"
                if self.feature_output_device == "cpu"
                else "torch.Tensor"
            ),
            "feature_output_dtype": "float32",
            "inference_autocast_dtype": self.inference_autocast_dtype,
            "inference_attention_backend_requested": self.inference_attention_backend,
            "inference_attention_backend_resolved": (
                "pytorch_sdpa_flash_only"
                if self.inference_attention_backend == "sdpa_flash"
                else "dinov2_default"
            ),
            "inference_attention_fallback_policy": (
                "disabled"
                if self.inference_attention_backend == "sdpa_flash"
                else "upstream_default"
            ),
            "inference_compile_mode": self.inference_compile_mode,
            "inference_compile_tensor_dispatch": (
                dict(wrapper.compile_tensor_dispatch)
                if wrapper.compile_tensor_dispatch is not None else None
            ),
            "inference_compile_backend": (
                "inductor" if self.inference_compile_mode == "default" else None
            ),
            "inference_compile_fullgraph": (
                True if self.inference_compile_mode == "default" else None
            ),
            "inference_compile_dynamic": (
                False if self.inference_compile_mode == "default" else None
            ),
            "inference_compile_fallback_policy": (
                "disabled" if self.inference_compile_mode == "default" else "not_applicable"
            ),
            "inference_compile_first_call_behavior": (
                "compile_on_first_inference"
                if self.inference_compile_mode == "default"
                else "not_applicable"
            ),
        }
        self.backbone_repo_tree_sha256 = repo_hash
        self.backbone_state_sha256 = state_hash

    def _validate_repo_before_load(self) -> None:
        if self.expected_repo_tree_sha256 is None:
            return
        repo = find_local_dinov2_repo()
        if repo is None or dinov2_repo_tree_sha256(repo) != self.expected_repo_tree_sha256:
            raise RuntimeError("DINOv2 repository tree hash mismatch")

    def _ensure_loaded(self) -> None:
        if self._wrapper is None:
            self._validate_repo_before_load()
            self._wrapper = DINOv2Wrapper(
                self.model_name,
                self.device,
                self.smaller_edge_size,
                feature_layers=self.feature_layers,
                feature_output_device=self.feature_output_device,
                inference_autocast_dtype=self.inference_autocast_dtype,
                inference_attention_backend=self.inference_attention_backend,
                inference_compile_mode=self.inference_compile_mode,
            )
            self._validate_and_record(self._wrapper)

    def set_resolution(self, smaller_edge_size: int) -> None:
        previous_resolution = self.smaller_edge_size
        super().set_resolution(smaller_edge_size)
        if self.smaller_edge_size != previous_resolution and self._wrapper is not None:
            setter = getattr(self._wrapper, "set_resolution", None)
            if callable(setter):
                setter(self.smaller_edge_size)

    def warmup(self, image_path: str, iters: int = 25) -> None:
        """CUDA warmup with a sample image."""
        self._ensure_loaded()
        for _ in range(iters):
            img_tensor, _ = self._wrapper.prepare_image(image_path)
            self._wrapper.extract_features(img_tensor)

    def prepare_image(self, img, *, resolution: int | None = None) -> Tuple:
        self._ensure_loaded()
        return self._wrapper.prepare_image(img, resolution=resolution)

    def extract_features(self, image_tensor) -> List[np.ndarray | torch.Tensor]:
        self._ensure_loaded()
        return self._wrapper.extract_features(image_tensor)

    def extract_cls_features(self, image_tensor) -> torch.Tensor:
        self._ensure_loaded()
        return self._wrapper.extract_cls_features(image_tensor)

    def extract_context_features(
        self,
        image_tensor: torch.Tensor,
        context_source: str,
    ) -> torch.Tensor:
        if context_source != "cls":
            raise ValueError(f"Unsupported DINOv2 context source: {context_source}")
        return self.extract_cls_features(image_tensor)

    def compute_background_mask(
        self, features, grid_size, threshold: float = 1.0, masking_type=False, **kwargs
    ) -> np.ndarray:
        self._ensure_loaded()
        return self._wrapper.compute_background_mask(
            features, grid_size, threshold=threshold, masking_type=masking_type, **kwargs
        )
