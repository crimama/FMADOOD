"""Bank-free reverse-distillation anomaly detection (pilot "RD-P1").

A frozen teacher (either a torchvision ``wide_resnet50_2`` or the shared
DINOv2 backbone) provides multi-level features on the support images.  A
small student network -- trained from scratch, per object, inside ``fit`` --
is asked to reconstruct those teacher features from a shared bottleneck.  At
inference time the anomaly map is the (upsampled, summed) per-level cosine
distance between teacher and student features; no support bank is retained.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fmad.methods.PSMA_MR import _apply_photometric, _read_rgb, _sample_photometric_transform
from fmad.methods.base import BaseMethod
from fmad.registry import METHOD_REGISTRY

_DEFAULT_TEACHER_WEIGHTS = "/model-cache/hub/checkpoints/wide_resnet50_2-95faca4d.pth"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _number(name: str, value: Any, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _integer(name: str, value: Any, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _multiple_of_32(name: str, value: Any) -> int:
    value = _integer(name, value, minimum=32)
    if value % 32 != 0:
        raise ValueError(f"{name} must be a multiple of 32")
    return value


class _ConvBlock(nn.Module):
    """conv3x3 -> BN -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class _UpBlock(nn.Module):
    """Nearest-upsample x2 followed by a conv block, used by the wrn50 decoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = _ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.block(x)


class _ReconstructionHead(nn.Module):
    """conv3x3-BN-ReLU then a 1x1 projection to the teacher channel width."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.refine = _ConvBlock(in_channels, in_channels)
        self.project = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.refine(x))


class _StudentUpsampling(nn.Module):
    """RD4AD-style student for the wrn50 arm (three strided teacher levels)."""

    def __init__(self, level_channels: tuple[int, int, int], bottleneck_channels: int) -> None:
        super().__init__()
        total_in = sum(level_channels)
        self.bottleneck = nn.Sequential(
            _ConvBlock(total_in, bottleneck_channels),
            _ConvBlock(bottleneck_channels, bottleneck_channels),
        )
        self.up1 = _UpBlock(bottleneck_channels, bottleneck_channels // 2)
        self.up2 = _UpBlock(bottleneck_channels // 2, bottleneck_channels // 4)
        self.head3 = _ReconstructionHead(bottleneck_channels, level_channels[2])
        self.head2 = _ReconstructionHead(bottleneck_channels // 2, level_channels[1])
        self.head1 = _ReconstructionHead(bottleneck_channels // 4, level_channels[0])

    def forward(self, level1: torch.Tensor, level2: torch.Tensor, level3: torch.Tensor):
        target = level3.shape[-2:]
        fused = torch.cat(
            [F.interpolate(level1, size=target, mode="bilinear", align_corners=False),
             F.interpolate(level2, size=target, mode="bilinear", align_corners=False),
             level3],
            dim=1,
        )
        bottleneck = self.bottleneck(fused)
        out3 = self.head3(bottleneck)
        feat2 = self.up1(bottleneck)
        out2 = self.head2(feat2)
        feat1 = self.up2(feat2)
        out1 = self.head1(feat1)
        return out1, out2, out3


class _StudentEqualStride(nn.Module):
    """Student for the dinov2 arm: two equal-stride teacher levels."""

    def __init__(self, level_channels: tuple[int, int], bottleneck_channels: int) -> None:
        super().__init__()
        total_in = sum(level_channels)
        self.bottleneck = nn.Sequential(
            _ConvBlock(total_in, bottleneck_channels),
            _ConvBlock(bottleneck_channels, bottleneck_channels),
        )
        self.head_shallow = _ReconstructionHead(bottleneck_channels, level_channels[0])
        self.head_deep = _ReconstructionHead(bottleneck_channels, level_channels[1])

    def forward(self, shallow: torch.Tensor, deep: torch.Tensor):
        fused = torch.cat([shallow, deep], dim=1)
        bottleneck = self.bottleneck(fused)
        return self.head_shallow(bottleneck), self.head_deep(bottleneck)


class _WRN50Teacher(nn.Module):
    """Frozen wide_resnet50_2 truncated to layer1/layer2/layer3 outputs."""

    def __init__(self, weights_path: str | None) -> None:
        super().__init__()
        import torchvision

        self.weights_path = weights_path
        model = torchvision.models.wide_resnet50_2(weights=None)
        if weights_path is not None:
            if not Path(weights_path).is_file():
                raise FileNotFoundError(
                    f"reverse_distillation teacher_weights not found: {weights_path}. "
                    "The container is offline; provide a local checkpoint file."
                )
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()
        mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        x = (image - self.mean) / self.std
        x = self.stem(x)
        level1 = self.layer1(x)
        level2 = self.layer2(level1)
        level3 = self.layer3(level2)
        return level1, level2, level3


def _cosine_distance_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    student_flat = student.flatten(2).transpose(1, 2)
    teacher_flat = teacher.flatten(2).transpose(1, 2)
    cosine = F.cosine_similarity(student_flat, teacher_flat, dim=-1, eps=1e-8)
    return (1.0 - cosine).mean()


def _resize_smaller_edge(image: np.ndarray, target: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = target / min(height, width)
    new_height, new_width = max(1, round(height * scale)), max(1, round(width * scale))
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(tensor, size=(new_height, new_width), mode="bilinear",
                            align_corners=False)
    return (resized[0].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)


def _crop_to_multiple(image: np.ndarray, multiple: int) -> np.ndarray:
    height, width = image.shape[:2]
    new_height = max(multiple, (height // multiple) * multiple)
    new_width = max(multiple, (width // multiple) * multiple)
    return np.ascontiguousarray(image[:new_height, :new_width])


def _to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


@METHOD_REGISTRY.register("reverse_distillation")
class ReverseDistillation(BaseMethod):
    """Bank-free reverse-distillation anomaly detector (RD-P1)."""

    def __init__(self, backbone, config: dict):
        super().__init__(backbone, config)
        self.teacher_arm = str(config.get("teacher", "wrn50")).lower()
        if self.teacher_arm not in {"wrn50", "dinov2"}:
            raise ValueError("reverse_distillation teacher must be 'wrn50' or 'dinov2'")

        self.bottleneck_channels = _integer(
            "bottleneck_channels", config.get("bottleneck_channels", 512)
        )
        self.steps = _integer("steps", config.get("steps", 1500))
        self.learning_rate = _number("learning_rate", config.get("learning_rate", 5e-3),
                                     minimum=0.0)
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        self.batch_size = _integer("batch_size", config.get("batch_size", 8))
        self.photometric_strength = _number(
            "photometric_strength", config.get("photometric_strength", 1.0), minimum=0.0
        )
        self.flips = bool(config.get("flips", True))
        if not isinstance(config.get("flips", True), bool):
            raise ValueError("flips must be a boolean")
        self.log_every = 50

        if self.teacher_arm == "wrn50":
            self.crop_size = _multiple_of_32("crop_size", config.get("crop_size", 512))
            self.train_resolution = _multiple_of_32(
                "train_resolution", config.get("train_resolution", 1024)
            )
            self.resolution = _multiple_of_32("resolution", config.get("resolution", 1024))
            teacher_weights = config.get("teacher_weights", _DEFAULT_TEACHER_WEIGHTS)
            if teacher_weights is not None and not isinstance(teacher_weights, str):
                raise ValueError("teacher_weights must be null or a string path")
            self.teacher_weights = teacher_weights
            self.feature_strides = [4, 8, 16]
            self.level_channels = (256, 512, 1024)
        else:
            self.crop_size = _integer("crop_size", config.get("crop_size", 448))
            self.resolution = _integer("resolution", config.get("resolution", 1120))
            self.train_resolution = self.resolution
            self.teacher_weights = None
            raw_layers = config.get("feature_layers", [11, 23])
            if (not isinstance(raw_layers, (list, tuple)) or len(raw_layers) != 2
                    or any(isinstance(v, bool) or not isinstance(v, int) for v in raw_layers)):
                raise ValueError("feature_layers must be a pair of integers for the dinov2 arm")
            self.feature_layers = tuple(int(v) for v in raw_layers)
            self.feature_strides = ["equal", "equal"]
            self.level_channels = None  # inferred from the backbone at fit time

        self.device = torch.device(
            getattr(backbone, "device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.teacher: nn.Module | None = None
        self.student: nn.Module | None = None
        self.steps_completed = 0
        self.final_loss: float | None = None
        self.trainable_parameters = 0

    # -- teacher feature extraction -------------------------------------------------

    def _build_teacher_wrn50(self) -> None:
        self.teacher = _WRN50Teacher(self.teacher_weights).to(self.device)
        self.student = _StudentUpsampling(self.level_channels, self.bottleneck_channels)
        self.student = self.student.to(self.device)

    def _build_teacher_dinov2(self) -> None:
        self.teacher = None  # the shared backbone plays the teacher role
        self.student = None  # channel widths are only known after one extraction

    def _wrn50_teacher_features(self, image_tensor: torch.Tensor):
        use_autocast = self.device.type == "cuda"
        with torch.no_grad():
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = self.teacher(image_tensor)
                outputs = tuple(o.float() for o in outputs)
            else:
                outputs = self.teacher(image_tensor)
        return outputs

    def _dinov2_teacher_features(self, image: np.ndarray, resolution: int):
        tensor, grid = self.backbone.prepare_image(image, resolution=resolution)
        with torch.no_grad():
            outputs = self.backbone.extract_features(tensor)
        declared = tuple(getattr(self.backbone, "feature_layers", self.feature_layers))
        by_layer = {layer: output for layer, output in zip(declared, outputs)}
        if any(layer not in by_layer for layer in self.feature_layers):
            raise ValueError(
                f"backbone did not return required layers {list(self.feature_layers)}"
            )
        maps = []
        height, width = grid
        for layer in self.feature_layers:
            matrix = torch.as_tensor(by_layer[layer], dtype=torch.float32).detach().cpu()
            while matrix.ndim > 2 and matrix.shape[0] == 1:
                matrix = matrix.squeeze(0)
            if matrix.ndim != 2:
                matrix = matrix.reshape(-1, matrix.shape[-1])
            if matrix.shape[0] != height * width and matrix.shape[1] == height * width:
                matrix = matrix.T.contiguous()
            channels = matrix.shape[1]
            grid_map = matrix.reshape(height, width, channels).permute(2, 0, 1).unsqueeze(0)
            maps.append(grid_map.to(self.device))
        return tuple(maps), (height, width)

    # -- data sampling ---------------------------------------------------------------

    def _sample_crop(
        self, image: np.ndarray, rng: np.random.Generator, resolution: int, crop_size: int
    ) -> np.ndarray:
        resized = _resize_smaller_edge(image, resolution)
        height, width = resized.shape[:2]
        effective_crop = min(crop_size, height, width)
        top = int(rng.integers(0, height - effective_crop + 1))
        left = int(rng.integers(0, width - effective_crop + 1))
        crop = resized[top:top + effective_crop, left:left + effective_crop]
        transform = _sample_photometric_transform(rng, self.photometric_strength)
        crop = _apply_photometric(crop, transform)
        if self.flips:
            if rng.random() < 0.5:
                crop = np.ascontiguousarray(crop[:, ::-1])
            if rng.random() < 0.5:
                crop = np.ascontiguousarray(crop[::-1, :])
        return _crop_to_multiple(crop, 32 if self.teacher_arm == "wrn50" else 14)

    # -- fit ----------------------------------------------------------------------

    def fit(self, support_paths: list[str], context) -> None:
        if not support_paths:
            raise ValueError("reverse_distillation requires at least one support image")
        torch.manual_seed(context.seed)
        rng = np.random.default_rng(context.seed)
        images = [_read_rgb(path) for path in support_paths]

        if self.teacher_arm == "wrn50":
            self._build_teacher_wrn50()
        else:
            self._build_teacher_dinov2()
            # Probe channel widths once to build the student lazily.
            probe_maps, _ = self._dinov2_teacher_features(images[0], self.resolution)
            self.level_channels = (int(probe_maps[0].shape[1]), int(probe_maps[1].shape[1]))
            self.student = _StudentEqualStride(self.level_channels, self.bottleneck_channels)
            self.student = self.student.to(self.device)

        self.trainable_parameters = sum(
            p.numel() for p in self.student.parameters() if p.requires_grad
        )
        optimizer = torch.optim.Adam(
            self.student.parameters(), lr=self.learning_rate, betas=(0.5, 0.999)
        )

        running_loss = None
        for step in range(1, self.steps + 1):
            self.student.train()
            crops = []
            for _ in range(self.batch_size):
                source_index = int(rng.integers(0, len(images)))
                crop = self._sample_crop(images[source_index], rng, self.train_resolution,
                                         self.crop_size)
                crops.append(crop)

            optimizer.zero_grad(set_to_none=True)
            total_loss = torch.zeros((), device=self.device)
            for crop in crops:
                if self.teacher_arm == "wrn50":
                    tensor = _to_tensor(crop, self.device)
                    teacher_levels = self._wrn50_teacher_features(tensor)
                else:
                    # The crop was cut at inference scale; pass its own size so the
                    # backbone does not rescale it and train/test strides stay equal.
                    teacher_levels, _ = self._dinov2_teacher_features(crop, int(min(crop.shape[:2])))

                student_levels = self.student(*teacher_levels)

                loss = torch.zeros((), device=self.device)
                for student_map, teacher_map in zip(student_levels, teacher_levels):
                    loss = loss + _cosine_distance_loss(student_map, teacher_map)
                total_loss = total_loss + loss

            total_loss = total_loss / len(crops)
            total_loss.backward()
            optimizer.step()

            loss_value = float(total_loss.detach().cpu())
            running_loss = loss_value
            if step % self.log_every == 0 or step == self.steps:
                self.log_training(step=step, loss=loss_value)
            self.steps_completed = step

        self.final_loss = running_loss
        self.student.eval()

    # -- predict --------------------------------------------------------------------

    def predict(self, test_image_path: str) -> tuple[np.ndarray, float]:
        image = _read_rgb(test_image_path)
        original_shape = image.shape[:2]
        multiple = 32 if self.teacher_arm == "wrn50" else 14
        resized = _crop_to_multiple(_resize_smaller_edge(image, self.resolution), multiple)
        resized_size = resized.shape[:2]

        self.student.eval()
        with torch.no_grad():
            if self.teacher_arm == "wrn50":
                tensor = _to_tensor(resized, self.device)
                teacher_levels = self._wrn50_teacher_features(tensor)
                student_levels = self.student(*teacher_levels)
            else:
                teacher_levels, _ = self._dinov2_teacher_features(resized, self.resolution)
                student_levels = self.student(*teacher_levels)

            total = torch.zeros((1, 1, *resized_size), device=self.device)
            for student_map, teacher_map in zip(student_levels, teacher_levels):
                cosine = F.cosine_similarity(student_map, teacher_map, dim=1, eps=1e-8)
                distance = (1.0 - cosine).unsqueeze(1)
                upsampled = F.interpolate(distance, size=resized_size, mode="bilinear",
                                          align_corners=False)
                total = total + upsampled

            final = F.interpolate(total, size=original_shape, mode="bilinear",
                                  align_corners=False)[0, 0]
        result = final.detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(result).all():
            raise RuntimeError("nonfinite reverse_distillation prediction")
        return result, float(result.max())

    # -- metadata -------------------------------------------------------------------

    def object_metadata(self) -> dict:
        return {
            "teacher": self.teacher_arm,
            "teacher_weights": self.teacher_weights,
            "feature_strides": list(self.feature_strides),
            "bottleneck_channels": self.bottleneck_channels,
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "crop_size": self.crop_size,
            "train_resolution": self.train_resolution,
            "resolution": self.resolution,
            "photometric_strength": self.photometric_strength,
            "flips": self.flips,
            "trainable_parameters": self.trainable_parameters,
            "steps_completed": self.steps_completed,
            "loss": self.final_loss,
            "bank_free": True,
            "synthetic_anomalies_used": False,
            "query_labels_used": False,
            "dense_maps_accumulated": False,
            "score_definition": "sum_levels_one_minus_cosine_teacher_student_upsampled",
        }
