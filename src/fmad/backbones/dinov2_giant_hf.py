"""Offline-first Hugging Face DINOv2-with-registers-Giant backbone.

The class is isolated from the Torch Hub DINOv2 loader because SubspaceAD's
official checkpoint is published as a Hugging Face safetensors model.
Transformers is imported lazily so other backbones do not require it.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from fmad.backbones.base import BaseBackbone, _validate_resolution
from fmad.registry import BACKBONE_REGISTRY


_MODEL_ID = "facebook/dinov2-with-registers-giant"
_MODEL_REVISION = "d45b6a06115df245b3b06b5104f84d5174fbda7b"
_WEIGHT_SHA256 = "c03832d44691e99b62ae28c4dfa2f134853a3614b3756c94f525109cce5a5051"
_WEIGHT_NAME = "model.safetensors"
_LAYERS = tuple(range(22, 29))


def _version_tuple(version: str, component_count: int = 3) -> tuple[int, ...]:
    components = re.match(r"^(\d+(?:\.\d+){0,2})", version)
    if components is None:
        raise RuntimeError(f"Could not parse package version: {version}")
    return tuple(int(part) for part in components.group(1).split(".")) + (0,) * (component_count - len(components.group(1).split(".")))


def _require_runtime_packages() -> None:
    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "SubspaceAD requires transformers>=4.48.0,<5.0 and "
            "safetensors>=0.4.3. Install these in the model runtime."
        ) from error
    transformers_version_tuple = _version_tuple(transformers_version)
    if transformers_version_tuple < (4, 48, 0):
        raise RuntimeError(
            "SubspaceAD requires transformers>=4.48.0,<5.0; found "
            f"{transformers_version}"
        )
    if transformers_version_tuple >= (5, 0, 0):
        raise RuntimeError(
            "SubspaceAD currently supports transformers>=4.48.0,<5.0; found "
            f"{transformers_version}"
        )

    try:
        safetensors_version = importlib.metadata.version("safetensors")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "SubspaceAD requires safetensors>=0.4.3 to load the pinned checkpoint."
        ) from error
    if _version_tuple(safetensors_version) < (0, 4, 3):
        raise RuntimeError(
            "SubspaceAD requires safetensors>=0.4.3; found " f"{safetensors_version}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@BACKBONE_REGISTRY.register("dinov2_vitg14_hf")
class DINOv2GiantHFBackbone(BaseBackbone):
    """DINOv2-G/14 with registers loaded from a pinned, local HF model folder.

    SubspaceAD consumes patch tokens from zero-based blocks 22 through 28.
    Attention matrices are intentionally not returned: its selected method
    configuration uses no saliency masking, so disabling them preserves the
    requested hidden-state features while avoiding a very large attention
    tensor allocation at 672-pixel input resolution.
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        smaller_edge_size: int = 672,
        feature_layers: Optional[Sequence[int]] = None,
        model_dir: str = "/model-cache/hub/hf_dinov2_with_registers_giant_d45b6a06",
        model_id: str = _MODEL_ID,
        model_revision: str = _MODEL_REVISION,
        expected_weight_sha256: str = _WEIGHT_SHA256,
    ) -> None:
        super().__init__(model_name, device, smaller_edge_size)
        self.feature_layers = tuple(feature_layers or _LAYERS)
        if self.feature_layers != _LAYERS:
            raise ValueError("SubspaceAD requires DINOv2-G/14 blocks 22 through 28")
        if model_id != _MODEL_ID or model_revision != _MODEL_REVISION:
            raise ValueError("SubspaceAD requires its pinned official DINOv2-G/14 model")
        if expected_weight_sha256 != _WEIGHT_SHA256:
            raise ValueError("SubspaceAD checkpoint SHA-256 does not match the pinned model")

        self.model_id = model_id
        self.model_revision = model_revision
        self.model_dir = Path(model_dir).expanduser()
        if not self.model_dir.is_absolute():
            raise ValueError("backbone_config.model_dir must be an absolute local path")
        self.expected_weight_sha256 = expected_weight_sha256
        self._model: Optional[torch.nn.Module] = None
        self._transform_cache: dict[int, transforms.Compose] = {}
        self.provenance: Optional[dict] = None
        self.backbone_state_sha256: Optional[str] = None

    def _get_transform(self, resolution: int) -> transforms.Compose:
        resolution = _validate_resolution(resolution)
        transform = self._transform_cache.get(resolution)
        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        (resolution, resolution),
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

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.model_dir.is_dir():
            raise FileNotFoundError(
                f"Pinned DINOv2-G/14 model directory is unavailable: {self.model_dir}"
            )
        config_path = self.model_dir / "config.json"
        weight_path = self.model_dir / _WEIGHT_NAME
        if not config_path.is_file() or not weight_path.is_file():
            raise FileNotFoundError(
                "Pinned DINOv2-G/14 model directory must contain config.json and "
                f"{_WEIGHT_NAME}: {self.model_dir}"
            )

        actual_sha256 = _sha256(weight_path)
        if actual_sha256 != self.expected_weight_sha256:
            raise RuntimeError(
                "DINOv2-G/14 checkpoint SHA-256 mismatch: "
                f"expected {self.expected_weight_sha256}, found {actual_sha256}"
            )

        _require_runtime_packages()
        try:
            from transformers import AutoModel
        except ImportError as error:
            raise RuntimeError(
                "SubspaceAD requires transformers>=4.48.0,<5.0 and "
                "safetensors>=0.4.3."
            ) from error

        model = AutoModel.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
            use_safetensors=True,
        )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model = model.to(self.device)

        config = model.config
        if (
            getattr(config, "model_type", None) != "dinov2_with_registers"
            or int(getattr(config, "hidden_size", 0)) != 1536
            or int(getattr(config, "num_hidden_layers", 0)) != 40
            or int(getattr(config, "patch_size", 0)) != 14
            or int(getattr(config, "num_register_tokens", 0)) != 4
        ):
            raise RuntimeError("Loaded checkpoint is not the expected DINOv2-G/14 model")

        self._model = model
        self.backbone_state_sha256 = actual_sha256
        self.provenance = {
            "model_id": self.model_id,
            "revision": self.model_revision,
            "checkpoint_file": _WEIGHT_NAME,
            "checkpoint_sha256": actual_sha256,
            "config_sha256": _sha256(config_path),
            "patch_size": int(config.patch_size),
            "depth": int(config.num_hidden_layers),
            "hidden_size": int(config.hidden_size),
            "num_register_tokens": int(config.num_register_tokens),
            "feature_layers": list(self.feature_layers),
            "output_attentions": False,
        }

    def prepare_image(
        self,
        img: Union[str, Path, np.ndarray, Image.Image],
        *,
        resolution: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if isinstance(img, (str, Path)):
            with Image.open(str(img)) as opened:
                image = opened.convert("RGB")
        elif isinstance(img, np.ndarray):
            image = Image.fromarray(img).convert("RGB")
        elif isinstance(img, Image.Image):
            image = img.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(img)}")

        target_resolution = (
            self.smaller_edge_size
            if resolution is None
            else _validate_resolution(resolution)
        )
        if target_resolution % 14:
            raise ValueError("DINOv2-G/14 input resolution must be divisible by 14")
        image_tensor = self._get_transform(target_resolution)(image)
        grid_size = (target_resolution // 14, target_resolution // 14)
        return image_tensor, grid_size

    def extract_features(self, image_tensor: torch.Tensor) -> List[np.ndarray]:
        self._ensure_loaded()
        assert self._model is not None
        if image_tensor.ndim != 3:
            raise ValueError("DINOv2-G/14 expects a single [C,H,W] image tensor")

        with torch.inference_mode():
            outputs = self._model(
                pixel_values=image_tensor.unsqueeze(0).to(self.device),
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Transformers did not return requested hidden states")
            start_token = 1 + int(self._model.config.num_register_tokens)
            features = []
            for layer in self.feature_layers:
                tokens = hidden_states[layer + 1][0, start_token:]
                features.append(np.ascontiguousarray(tokens.float().cpu().numpy()))
            del outputs, hidden_states
            return features

    def extract_cls_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("SubspaceAD uses patch features and does not read CLS tokens")

    def compute_background_mask(
        self,
        features,
        grid_size,
        threshold: float = 1.0,
        masking_type=False,
        **kwargs,
    ) -> np.ndarray:
        del features, threshold, masking_type, kwargs
        return np.ones(grid_size, dtype=bool)
