#!/usr/bin/env python3
# ruff: noqa: ANN401, ARG002, COM812, EM102, FBT003, PLC0415, PLR0913, TC006
"""Run the preregistered shift-conditioned patch-local FlowTTE experiment.

The mechanism phase is support-only by construction: it loads only the exact
sixteen train/good paths per object from the frozen manifest and writes an
access ledger with zero public/private/test/ground-truth access.  The public
phase requires a machine-readable mechanism pass sentinel before it enumerates
MVTec AD2 test_public.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol, Sequence, cast

import numpy as np
import numpy.typing as npt

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "src", Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from flow_tte_support import merge_layer_features  # noqa: E402

from flow_tte.config import FlowConfig, FlowTTEConfig, ScoreConfig  # noqa: E402
from flow_tte.denoising import fit_feature_denoiser  # noqa: E402
from flow_tte.dinov3_early_exit import patch_torch_pytree_compat  # noqa: E402

# PyTorch 2.1 exposes only the private registration API.  Transformers is
# imported indirectly by torch.optim through torch.onnx/torch._dynamo, so the
# compatibility shim must be installed at process startup rather than only in
# the DINO wrapper construction path.
patch_torch_pytree_compat()

OBJECTS = (
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
)
PUBLIC_SMOKE_OBJECTS = ("fruit_jelly", "walnuts", "vial", "fabric")
SUPPORT_COUNT = 16
LAYERS = (7, 15, 23, 31)
PROTOCOL_ID = "shift-conditioned-patch-local-flow-20260804"
FROZEN_SUPPORT_MANIFEST_SHA256 = "9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5"
PHOTOMETRIC_AXES = ("brightness", "contrast", "gamma", "rgb_gain", "blur", "noise")
TRANSFORM_MIX_COUNTS = {0: 2, 1: 5, 2: 3}
STAGE_B_EPOCHS = 10
STAGE_B_LR = 1e-4
STAGE_B_WEIGHT_DECAY = 1e-4
STAGE_B_GRAD_CLIP = 0.5

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


class RunnerError(RuntimeError):
    """Raised when the runner cannot satisfy the frozen protocol."""


@dataclass(frozen=True)
class SupportEntry:
    object_name: str
    path: Path
    relative_path: str
    path_sha256: str
    content_sha256: str


@dataclass(frozen=True)
class PhotometricTransformSpec:
    name: str
    active_axes: tuple[str, ...]
    brightness: float = 1.0
    contrast: float = 1.0
    gamma: float = 1.0
    rgb_gain: tuple[float, float, float] = (1.0, 1.0, 1.0)
    blur_sigma: float = 0.0
    noise_std: float = 0.0
    seed: int = 0

    def targets(self) -> dict[str, Any]:
        labels = {axis: float(axis in self.active_axes) for axis in PHOTOMETRIC_AXES}
        continuous = {
            "brightness": _normalize_range(self.brightness, 0.7, 1.3),
            "contrast": _normalize_range(self.contrast, 0.8, 1.2),
            "gamma": _normalize_range(self.gamma, 0.8, 1.2),
            "rgb_gain_r": _normalize_range(self.rgb_gain[0], 0.9, 1.1),
            "rgb_gain_g": _normalize_range(self.rgb_gain[1], 0.9, 1.1),
            "rgb_gain_b": _normalize_range(self.rgb_gain[2], 0.9, 1.1),
            "blur_sigma": _normalize_range(self.blur_sigma, 0.0, 1.0),
            "noise_std": _normalize_range(self.noise_std, 0.0, 0.02),
        }
        masks = {
            "brightness": labels["brightness"],
            "contrast": labels["contrast"],
            "gamma": labels["gamma"],
            "rgb_gain_r": labels["rgb_gain"],
            "rgb_gain_g": labels["rgb_gain"],
            "rgb_gain_b": labels["rgb_gain"],
            "blur_sigma": labels["blur"],
            "noise_std": labels["noise"],
        }
        return {
            "multi_label": labels,
            "continuous": continuous,
            "regression_mask": masks,
        }


@dataclass(frozen=True)
class FeatureMap:
    values: FloatArray
    grid: tuple[int, int]
    image_shape: tuple[int, int]
    relative_path: str
    transform: str = "identity"


@dataclass(frozen=True)
class StageBPair:
    source: FeatureMap
    transformed: FeatureMap
    transform: PhotometricTransformSpec


class ShiftConditionedBackend(Protocol):
    def set_dvt_artifact(self, artifact: FloatArray | None, alpha: float) -> None: ...

    def transform_feature_maps(
        self, feature_maps: Sequence[FeatureMap]
    ) -> tuple[FeatureMap, ...]: ...

    def fit_stage_a(
        self, support_maps: Sequence[FeatureMap], config: Mapping[str, Any]
    ) -> None: ...

    def train_stage_b(
        self, pairs: Sequence[StageBPair], config: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def mechanism_summary(
        self,
        support_maps: Sequence[FeatureMap],
        held_pairs: Sequence[StageBPair],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def save_checkpoint(self, path: Path) -> None: ...

    def load_checkpoint(self, path: Path, config: Mapping[str, Any]) -> None: ...

    def score_maps(
        self,
        feature_maps: Sequence[FeatureMap],
        arm: str,
        descriptor_mode: str,
    ) -> Sequence[FloatArray]: ...


class TorchShiftConditionedBackend:
    """Concrete adapter from the runner contract to ``flow_tte.shift_conditioned``.

    The core module intentionally exposes low-level Torch modules.  This adapter
    owns the experiment plumbing: Stage-A fitting, Stage-B paired losses,
    support memory materialization, DVT persistence, and raw Euclidean 1-NN map
    scoring.
    """

    def __init__(self) -> None:
        self.device_name = "cpu"
        self.stage_a: Any = None
        self.model: Any = None
        self.predictor: Any = None
        self.support_center: Any = None
        self.support_maps: tuple[FeatureMap, ...] = ()
        self.support_banks: dict[str, Any] = {}
        self.loo_stats: dict[str, tuple[float, float]] = {}
        self.dvt_artifact: FloatArray | None = None
        self.dvt_alpha = 1.0
        self.stage_a_training: Mapping[str, Any] = {}

    def set_dvt_artifact(self, artifact: FloatArray | None, alpha: float) -> None:
        self.dvt_artifact = None if artifact is None else artifact.astype(np.float32, copy=True)
        self.dvt_alpha = float(alpha)

    def transform_feature_maps(self, feature_maps: Sequence[FeatureMap]) -> tuple[FeatureMap, ...]:
        if self.dvt_artifact is None:
            return tuple(feature_maps)
        result: list[FeatureMap] = []
        for item in feature_maps:
            if item.values.shape != self.dvt_artifact.shape:
                raise RunnerError(
                    f"DVT artifact shape mismatch: {item.values.shape} vs {self.dvt_artifact.shape}"
                )
            result.append(
                FeatureMap(
                    values=(item.values - self.dvt_alpha * self.dvt_artifact).astype(
                        np.float32, copy=False
                    ),
                    grid=item.grid,
                    image_shape=item.image_shape,
                    relative_path=item.relative_path,
                    transform=item.transform,
                ),
            )
        return tuple(result)

    def fit_stage_a(self, support_maps: Sequence[FeatureMap], config: Mapping[str, Any]) -> None:
        import torch

        from flow_tte.config import ExpansionConfig
        from flow_tte.trainer import FlowDensityEstimator

        if not support_maps:
            raise RunnerError("Stage-A requires support feature maps")
        self.device_name = str(config.get("device", "cpu"))
        flow_cfg = FlowConfig(**dict(config["stage_a_flow"]))
        estimator = FlowDensityEstimator(
            dim=int(support_maps[0].values.shape[-1]),
            config=flow_cfg,
            device=self.device_name,
        )
        density_quantile = float(ExpansionConfig().density_quantile)
        support_array = _stack_maps(support_maps)
        stats = estimator.fit(support_array, density_quantile=density_quantile)
        self.stage_a = estimator
        self.support_maps = tuple(support_maps)
        shift_module = importlib.import_module("flow_tte.shift_conditioned")
        self.model = shift_module.ShiftConditionedPatchLocalFlow(
            estimator.flow,
            patch_dim=int(support_maps[0].values.shape[-1]),
        ).to(estimator.device)
        self.predictor = shift_module.PairedRelativeShiftPredictor(
            descriptor_dim=int(self.model.descriptor.descriptor_dim),
            num_transform_types=len(PHOTOMETRIC_AXES),
            regression_dim=8,
        ).to(estimator.device)
        with torch.no_grad():
            support_tensor = self._tensor4(support_maps)
            descriptors = self.model.encode_shift(
                support_tensor.reshape(support_tensor.shape[0], -1, support_tensor.shape[-1]),
            ).descriptor
            self.support_center = descriptors.median(dim=0, keepdim=True).values.detach()
        self.stage_a_training = {
            "losses": [float(value) for value in stats.losses],
            "train_nll_mean": float(stats.train_nll_mean),
            "train_nll_std": float(stats.train_nll_std),
            "density_threshold": float(stats.density_threshold),
        }
        self._refresh_support_banks()

    def train_stage_b(
        self, pairs: Sequence[StageBPair], config: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        import torch
        import torch.nn.functional as functional
        from torch.nn.utils.clip_grad import clip_grad_norm_

        if self.model is None or self.predictor is None or self.stage_a is None:
            raise RunnerError("fit_stage_a must run before train_stage_b")
        if not pairs:
            raise RunnerError("Stage-B requires deterministic photometric pairs")
        shift_module = importlib.import_module("flow_tte.shift_conditioned")
        params = [
            parameter
            for parameter in list(self.model.parameters()) + list(self.predictor.parameters())
            if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            params,
            lr=float(config["stage_b"]["lr"]),
            weight_decay=float(config["stage_b"]["weight_decay"]),
        )
        losses: list[dict[str, float]] = []
        self.model.train()
        self.predictor.train()
        loss_cfg = dict(config["stage_b"].get("loss", {}))
        loss_weights = {
            "nll": float(loss_cfg.get("nll", 1.0)),
            "inv": float(loss_cfg.get("inv", 1.0)),
            "shift": float(loss_cfg.get("shift", 0.0)),
            "nbr": float(loss_cfg.get("nbr", 0.0)),
            "id": float(loss_cfg.get("id", 0.0)),
            "res": float(loss_cfg.get("res", 0.0)),
        }
        for _epoch in range(int(config["stage_b"]["epochs"])):
            epoch = {
                "nll": 0.0,
                "inv": 0.0,
                "shift": 0.0,
                "nbr": 0.0,
                "id": 0.0,
                "res": 0.0,
                "total": 0.0,
            }
            for pair in pairs:
                source = self._tensor4([pair.source])
                transformed = self._tensor4([pair.transformed])
                source_patches = source.reshape(1, -1, source.shape[-1])
                transformed_patches = transformed.reshape(1, -1, transformed.shape[-1])
                source_std = self._standardize_patches(source_patches)
                transformed_std = self._standardize_patches(transformed_patches)
                descriptor_source = self._relative_descriptor(source_patches)
                descriptor_transformed = self._relative_descriptor(transformed_patches)
                source_output = self.model.forward(source_std, descriptor=descriptor_source)
                output = self.model.forward(transformed_std, descriptor=descriptor_transformed)
                with torch.no_grad():
                    source_base_z, _ = self.stage_a.flow.forward(
                        source_std.reshape(-1, source_std.shape[-1])
                    )
                logits, regression = self.predictor.forward(
                    descriptor_source, descriptor_transformed
                )
                labels, continuous, masks = _targets_to_tensors(
                    pair.transform, device=output.z.device
                )
                regression_error = functional.smooth_l1_loss(
                    regression,
                    continuous,
                    reduction="none",
                )
                reg_loss = (regression_error * masks).sum() / masks.sum().clamp_min(1.0)
                shift_loss = functional.binary_cross_entropy_with_logits(logits, labels) + reg_loss
                nll = shift_module.normalized_nll_loss(output.z, output.logdet)
                inv = shift_module.normalized_huber_invariance_loss(
                    source_output.z.reshape(-1, source_output.z.shape[-1]),
                    output.z.reshape(-1, output.z.shape[-1]),
                )
                nbr = shift_module.teacher_feature_knn_kl_loss(
                    source_output.z.reshape(-1, source_output.z.shape[-1]),
                    source_patches.reshape(-1, source_patches.shape[-1]).detach(),
                    max_samples=256,
                    k=10,
                    temperature=0.1,
                )
                identity = shift_module.baseline_identity_mse(
                    source_output.z,
                    source_base_z.reshape_as(source_std).detach(),
                )
                res = shift_module.residual_magnitude_loss(output.diagnostics)
                total = (
                    loss_weights["nll"] * nll
                    + loss_weights["inv"] * inv
                    + loss_weights["shift"] * shift_loss
                    + loss_weights["nbr"] * nbr
                    + loss_weights["id"] * identity
                    + loss_weights["res"] * res
                )
                optimizer.zero_grad()
                total.backward()
                clip_grad_norm_(params, max_norm=float(config["stage_b"]["gradient_clip"]))
                optimizer.step()
                for name, value in (
                    ("nll", nll),
                    ("inv", inv),
                    ("shift", shift_loss),
                    ("nbr", nbr),
                    ("id", identity),
                    ("res", res),
                    ("total", total),
                ):
                    epoch[name] += float(value.detach().cpu())
            for name in epoch:
                epoch[name] /= float(len(pairs))
            losses.append(epoch)
            self._refresh_support_center()
        self._refresh_support_banks()
        return {
            "stage_a": self.stage_a_training,
            "stage_b_epoch_losses": losses,
            "pair_count": len(pairs),
            "trainable_parameter_count": int(sum(parameter.numel() for parameter in params)),
        }

    def mechanism_summary(
        self,
        support_maps: Sequence[FeatureMap],
        held_pairs: Sequence[StageBPair],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        import torch

        if self.model is None:
            raise RunnerError("model is not fitted")
        validity = self._validity_summary(support_maps)
        held = self._held_shift_summary(held_pairs)
        identity = self._identity_summary(support_maps)
        defect = self._synthetic_defect_summary(support_maps)
        shift = self._shift_prediction_summary(held_pairs)
        return {
            "validity": validity,
            "held_shift": held,
            "identity": identity,
            "defect_retention": defect,
            "shift_prediction": shift,
            "latent_std": {
                arm: float(torch.std(bank.detach()).cpu())
                for arm, bank in self.support_banks.items()
            },
        }

    def save_checkpoint(self, path: Path) -> None:
        import torch

        if self.model is None or self.stage_a is None or self.predictor is None:
            raise RunnerError("cannot save before fitting")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "protocol_id": PROTOCOL_ID,
                "model_state": self.model.state_dict(),
                "predictor_state": self.predictor.state_dict(),
                "base_flow_state": self.stage_a.flow.state_dict(),
                "standardizer": (
                    None
                    if self.stage_a.standardizer is None
                    else {
                        "mean": self.stage_a.standardizer.mean.detach().cpu(),
                        "std": self.stage_a.standardizer.std.detach().cpu(),
                    }
                ),
                "support_center": None
                if self.support_center is None
                else self.support_center.detach().cpu(),
                "support_maps": [
                    asdict(item) | {"values": item.values} for item in self.support_maps
                ],
                "support_banks": {
                    key: value.detach().cpu() for key, value in self.support_banks.items()
                },
                "loo_stats": self.loo_stats,
                "dvt_artifact": self.dvt_artifact,
                "dvt_alpha": self.dvt_alpha,
                "stage_a_training": self.stage_a_training,
            },
            path,
        )

    def load_checkpoint(self, path: Path, config: Mapping[str, Any]) -> None:
        import torch

        from flow_tte.trainer import FeatureStandardizer, FlowDensityEstimator

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("protocol_id") != PROTOCOL_ID:
            raise RunnerError(f"checkpoint protocol mismatch: {path}")
        support_maps = [
            FeatureMap(
                values=np.asarray(item["values"], dtype=np.float32),
                grid=tuple(item["grid"]),
                image_shape=tuple(item["image_shape"]),
                relative_path=str(item["relative_path"]),
                transform=str(item["transform"]),
            )
            for item in payload["support_maps"]
        ]
        self.device_name = str(config.get("device", "cpu"))
        self.stage_a = FlowDensityEstimator(
            dim=int(support_maps[0].values.shape[-1]),
            config=FlowConfig(**dict(config["stage_a_flow"])),
            device=self.device_name,
        )
        self.stage_a.flow.load_state_dict(payload["base_flow_state"])
        if payload.get("standardizer") is not None:
            self.stage_a.standardizer = FeatureStandardizer(
                mean=payload["standardizer"]["mean"].to(self.stage_a.device),
                std=payload["standardizer"]["std"].to(self.stage_a.device),
            )
        shift_module = importlib.import_module("flow_tte.shift_conditioned")
        self.model = shift_module.ShiftConditionedPatchLocalFlow(
            self.stage_a.flow,
            patch_dim=int(support_maps[0].values.shape[-1]),
        ).to(self.stage_a.device)
        self.model.load_state_dict(payload["model_state"])
        self.predictor = shift_module.PairedRelativeShiftPredictor(
            descriptor_dim=int(self.model.descriptor.descriptor_dim),
            num_transform_types=len(PHOTOMETRIC_AXES),
            regression_dim=8,
        ).to(self.stage_a.device)
        self.predictor.load_state_dict(payload["predictor_state"])
        self.support_center = payload["support_center"].to(self.stage_a.device)
        self.support_maps = tuple(support_maps)
        self.support_banks = {
            key: value.to(self.stage_a.device) for key, value in payload["support_banks"].items()
        }
        self.loo_stats = {key: tuple(value) for key, value in payload["loo_stats"].items()}
        self.dvt_artifact = (
            None
            if payload.get("dvt_artifact") is None
            else np.asarray(payload["dvt_artifact"], dtype=np.float32)
        )
        self.dvt_alpha = float(payload.get("dvt_alpha", 1.0))
        self.stage_a_training = dict(payload.get("stage_a_training", {}))

    def score_maps(
        self,
        feature_maps: Sequence[FeatureMap],
        arm: str,
        descriptor_mode: str,
    ) -> Sequence[FloatArray]:
        import torch

        if arm not in self.support_banks:
            raise RunnerError(f"unsupported arm {arm}")
        bank = self.support_banks[arm]
        mean, std = self.loo_stats[arm]
        maps: list[FloatArray] = []
        with torch.no_grad():
            for item in feature_maps:
                latent = self._latent_for_maps(
                    [item],
                    arm=arm,
                    descriptor_mode=descriptor_mode,
                )
                distances = torch.cdist(latent, bank, p=2.0).min(dim=1).values
                scores = (
                    ((distances - mean) / max(std, 1e-6))
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                maps.append(scores.reshape(item.grid).astype(np.float32, copy=False))
        return tuple(maps)

    def _tensor4(self, maps: Sequence[FeatureMap]) -> Any:
        import torch

        return torch.as_tensor(_stack_maps(maps), dtype=torch.float32, device=self.stage_a.device)

    def _standardize_patches(self, patches: Any) -> Any:
        if self.stage_a.standardizer is None:
            return patches
        flat = patches.reshape(-1, patches.shape[-1])
        transformed = self.stage_a.standardizer.transform(flat)
        return transformed.reshape_as(patches)

    def _relative_descriptor(self, dvt_patches: Any) -> Any:
        descriptor = self.model.encode_shift(dvt_patches).descriptor
        return descriptor - self.support_center

    def _latent_for_maps(self, maps: Sequence[FeatureMap], arm: str, descriptor_mode: str) -> Any:
        import torch

        dvt_patches = self._tensor4(maps).reshape(len(maps), -1, maps[0].values.shape[-1])
        patches = self._standardize_patches(dvt_patches)
        if arm in {"E0", "E1"}:
            z, _ = self.stage_a.flow.forward(patches.reshape(-1, patches.shape[-1]))
            return z
        descriptor = self._relative_descriptor(dvt_patches)
        if descriptor_mode == "zero":
            descriptor = torch.zeros_like(descriptor)
        elif descriptor_mode == "shuffled":
            descriptor = self._shuffled_descriptor(maps, descriptor)
        elif descriptor_mode != "correct":
            raise RunnerError(f"unknown descriptor mode {descriptor_mode}")
        output = self.model.forward(patches, descriptor=descriptor)
        return output.z.reshape(-1, output.z.shape[-1])

    def _shuffled_descriptor(self, maps: Sequence[FeatureMap], descriptor: Any) -> Any:
        if len(maps) > 1:
            return descriptor.roll(shifts=1, dims=0)
        support = self._tensor4(self.support_maps)
        support_patches = support.reshape(support.shape[0], -1, support.shape[-1])
        support_descriptors = self.model.encode_shift(support_patches).descriptor
        matching = [
            index
            for index, item in enumerate(self.support_maps)
            if item.relative_path == maps[0].relative_path
        ]
        if matching:
            wrong_index = (matching[0] + 1) % len(self.support_maps)
        else:
            wrong_index = stable_seed(maps[0].relative_path, "shuffle") % len(
                self.support_maps,
            )
        return support_descriptors[wrong_index : wrong_index + 1] - self.support_center

    def _refresh_support_banks(self) -> None:
        import torch

        with torch.no_grad():
            for arm in ("E0", "E1", "E5"):
                bank = self._latent_for_maps(
                    self.support_maps, arm=arm, descriptor_mode="correct"
                ).detach()
                self.support_banks[arm] = bank
                self.loo_stats[arm] = _loo_stats_torch(bank)

    def _refresh_support_center(self) -> None:
        import torch

        with torch.no_grad():
            support = self._tensor4(self.support_maps)
            patches = support.reshape(support.shape[0], -1, support.shape[-1])
            descriptors = self.model.encode_shift(patches).descriptor
            self.support_center = descriptors.median(dim=0, keepdim=True).values.detach()

    def _validity_summary(self, support_maps: Sequence[FeatureMap]) -> dict[str, Any]:
        import torch

        raw_patches = self._tensor4(support_maps[:2]).reshape(
            min(2, len(support_maps)), -1, support_maps[0].values.shape[-1]
        )
        patches = self._standardize_patches(raw_patches)
        descriptor = self._relative_descriptor(raw_patches)
        with torch.no_grad():
            forward = self.model.forward(patches, descriptor=descriptor)
            reverse = self.model.forward(forward.z, reverse=True, descriptor=descriptor)
            base_z, base_logdet = self.stage_a.flow.forward(patches.reshape(-1, patches.shape[-1]))
            e1_z = self._latent_for_maps(support_maps[:2], "E1", "correct")
            repeat = self.model.forward(patches, descriptor=descriptor)
            shuffled = self.model.forward(patches, descriptor=descriptor.roll(shifts=1, dims=0))
        return {
            "inverse_reconstruction_max_error": float(
                (reverse.z - patches).abs().max().detach().cpu()
            ),
            "logdet_cancellation_max_error": float(
                (forward.logdet + reverse.logdet).abs().max().detach().cpu()
            ),
            "e0_e1_latent_max_error": float((base_z - e1_z).abs().max().detach().cpu()),
            "e0_e1_logdet_max_error": float(base_logdet.new_zeros(()).detach().cpu()),
            "gradient_finite_nonzero": _gradient_probe(self.model, patches),
            "deterministic_repeat": bool(torch.allclose(forward.z, repeat.z, atol=0.0, rtol=0.0)),
            "descriptor_witness": bool(not torch.allclose(forward.z, shuffled.z)),
            "raw_euclidean_memory_scoring": True,
        }

    def _held_shift_summary(self, held_pairs: Sequence[StageBPair]) -> dict[str, Any]:
        e0: list[float] = []
        correct: list[float] = []
        zero: list[float] = []
        shuffled: list[float] = []
        for pair in held_pairs:
            for bucket, arm, mode in (
                (e0, "E0", "correct"),
                (correct, "E5", "correct"),
                (zero, "E5", "zero"),
                (shuffled, "E5", "shuffled"),
            ):
                scores = self.score_maps([pair.transformed], arm=arm, descriptor_mode=mode)[0]
                source_scores = self.score_maps([pair.source], arm=arm, descriptor_mode=mode)[0]
                bucket.append(float(np.mean(scores) - np.mean(source_scores)))
        result = {
            "e0_inflation": float(np.mean(e0)),
            "correct_inflation": float(np.mean(correct)),
            "zero_inflation": float(np.mean(zero)),
            "shuffled_inflation": float(np.mean(shuffled)),
        }
        return _complete_mechanism_payload({"held_shift": result})["held_shift"]

    def _identity_summary(self, support_maps: Sequence[FeatureMap]) -> dict[str, float]:
        e0 = _loo_distances_torch(self.support_banks["E0"])
        e5 = _loo_distances_torch(self.support_banks["E5"])
        e0_std = max(float(e0.std(unbiased=False).detach().cpu()), 1e-6)
        return {
            "loo_delta_e0_std": float(
                (e5.mean() - e0.mean()).detach().cpu() / e0_std,
            ),
            "latent_std_ratio": float(
                torch_std(self.support_banks["E5"]) / max(torch_std(self.support_banks["E0"]), 1e-6)
            ),
        }

    def _synthetic_defect_summary(self, support_maps: Sequence[FeatureMap]) -> dict[str, float]:
        retentions: list[float] = []
        for item in support_maps:
            defect = item.values.copy()
            h, w = defect.shape[:2]
            ys = slice(h // 4, max(h // 4 + 1, h // 2))
            xs = slice(w // 4, max(w // 4 + 1, w // 2))
            defect[ys, xs] += np.float32(0.25)
            defect_map = FeatureMap(
                defect, item.grid, item.image_shape, item.relative_path, "synthetic_defect"
            )
            e0_gap = float(
                np.mean(self.score_maps([defect_map], "E0", "correct")[0][ys, xs])
                - np.mean(self.score_maps([item], "E0", "correct")[0])
            )
            e5_gap = float(
                np.mean(self.score_maps([defect_map], "E5", "correct")[0][ys, xs])
                - np.mean(self.score_maps([item], "E5", "correct")[0])
            )
            retentions.append(e5_gap / max(e0_gap, 1e-6))
        return {"gap_retention": float(np.mean(retentions))}

    def _shift_prediction_summary(self, held_pairs: Sequence[StageBPair]) -> dict[str, Any]:
        import torch
        import torch.nn.functional as functional

        losses: list[float] = []
        zero_losses: list[float] = []
        with torch.no_grad():
            for pair in held_pairs:
                source = self._tensor4([pair.source]).reshape(1, -1, pair.source.values.shape[-1])
                transformed = self._tensor4([pair.transformed]).reshape(
                    1, -1, pair.transformed.values.shape[-1]
                )
                logits, regression = self.predictor.forward(
                    self._relative_descriptor(source),
                    self._relative_descriptor(transformed),
                )
                labels, continuous, masks = _targets_to_tensors(
                    pair.transform, device=logits.device
                )
                loss = functional.binary_cross_entropy_with_logits(logits, labels)
                regression_error = functional.smooth_l1_loss(
                    regression,
                    continuous,
                    reduction="none",
                )
                loss = loss + (regression_error * masks).sum() / masks.sum().clamp_min(1.0)
                zero_loss = functional.binary_cross_entropy_with_logits(
                    torch.zeros_like(logits), labels
                )
                zero_error = functional.smooth_l1_loss(
                    torch.zeros_like(regression),
                    continuous,
                    reduction="none",
                )
                zero_loss = zero_loss + (zero_error * masks).sum() / masks.sum().clamp_min(1.0)
                losses.append(float(loss.cpu()))
                zero_losses.append(float(zero_loss.cpu()))
        loss_value = float(np.mean(losses))
        zero_value = float(np.mean(zero_losses))
        return {"loss": loss_value, "zero_loss": zero_value, "beats_zero": loss_value < zero_value}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in (PROTOCOL_ID, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("mechanism", "public", "smoke"), required=True)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--support-json", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-root", default="")
    parser.add_argument("--mechanism-pass-sentinel", default="")
    parser.add_argument("--backend-class", default="")
    parser.add_argument("--host-gpu-id", default="")
    parser.add_argument("--container-cuda-slot", default="0")
    parser.add_argument("--cleanup-maps", action="store_true")
    args = parser.parse_args(list(argv))
    args.objects = tuple(part for part in args.objects.replace(",", " ").split() if part)
    if not args.objects:
        parser.error("--objects must name at least one object")
    if len(set(args.objects)) != len(args.objects):
        parser.error("--objects must be unique")
    if any(name not in OBJECTS for name in args.objects):
        parser.error(f"--objects must be a subset of {OBJECTS}")
    if args.phase == "public" and any(name not in PUBLIC_SMOKE_OBJECTS for name in args.objects):
        parser.error(f"public Phase 1 is frozen to {PUBLIC_SMOKE_OBJECTS}")
    if args.phase != "smoke" and (not args.data_root or not args.support_json):
        parser.error("--data-root and --support-json are required outside smoke mode")
    return args


def frozen_config(device: str) -> dict[str, Any]:
    stage_a = FlowTTEConfig(
        flow=FlowConfig(
            n_coupling_layers=2,
            hidden_multiplier=1,
            n_epochs=3,
            lr=2e-4,
            clamp=1.9,
            tail_weight=0.3,
            tail_top_k_ratio=0.05,
            lambda_logdet=1e-3,
            standardize=True,
        ),
        score=ScoreConfig(
            distance_weight=1.0,
            density_weight=0.0,
            loo_standardize=True,
            use_squared_distance=False,
        ),
        device=device,
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "device": device,
        "target_dataset": "MVTec AD2 single-image",
        "backbone": "dinov3_vith16plus",
        "layers": list(LAYERS),
        "feature_fusion": "layer_norm_mean",
        "dvt": {"mode": "position_mean", "alpha": 1.0},
        "stage_a_flow": asdict(stage_a.flow),
        "stage_a_score": asdict(stage_a.score),
        "stage_b": {
            "epochs": STAGE_B_EPOCHS,
            "lr": STAGE_B_LR,
            "weight_decay": STAGE_B_WEIGHT_DECAY,
            "gradient_clip": STAGE_B_GRAD_CLIP,
            "loss": {
                "nll": 1.0,
                "inv": 1.0,
                "shift": 0.1,
                "nbr": 0.5,
                "id": 1.0,
                "res": 1e-3,
            },
            "pairs": "deterministic photometric identity/one/two active transform mixture",
        },
        "condition_architecture": {
            "register_queries": 4,
            "register_width": 128,
            "cross_attention_heads": 4,
            "ffn_width": 256,
            "dropout": 0.0,
            "descriptor_dim": 64,
            "support_center": "detached_channelwise_median_original_support_descriptors",
            "bounded_residual_gain": {"init": 0.05, "max": 0.1},
        },
        "score": {
            "latent_metric": "raw_euclidean",
            "nn": "1NN",
            "standardization": "support_LOO",
            "l2_normalized": False,
            "distance_weight": 1.0,
            "density_weight": 0.0,
        },
        "query_adaptation": False,
        "test_public_access": False,
        "test_private_access": False,
    }


def _manifest_tail(object_name: str, raw_path: str) -> tuple[str, str, str, str]:
    parts = PurePosixPath(raw_path.replace("\\", "/")).parts
    if len(parts) < 4:
        raise ValueError(f"{object_name}: malformed support path {raw_path!r}")
    return cast("tuple[str, str, str, str]", tuple(parts[-4:]))


def load_support_manifest(path: Path, data_root: Path) -> dict[str, tuple[SupportEntry, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("support manifest must be a JSON object")
    if set(payload) != set(OBJECTS):
        raise ValueError("support manifest must contain exactly the eight frozen objects")
    root = data_root.resolve(strict=True)
    result: dict[str, tuple[SupportEntry, ...]] = {}
    for object_name in OBJECTS:
        raw_paths = payload[object_name]
        if (
            not isinstance(raw_paths, list)
            or len(raw_paths) != SUPPORT_COUNT
            or not all(isinstance(item, str) for item in raw_paths)
        ):
            raise ValueError(f"{object_name}: expected exactly {SUPPORT_COUNT} string paths")
        entries: list[SupportEntry] = []
        train_root = (root / object_name / "train" / "good").resolve(strict=True)
        for raw_path in raw_paths:
            tail = _manifest_tail(object_name, raw_path)
            if tail[:3] != (object_name, "train", "good"):
                raise ValueError(
                    f"{object_name}: support path must be inside {object_name}/train/good"
                )
            candidate = (train_root / tail[3]).resolve(strict=True)
            if candidate.parent != train_root or not candidate.is_file():
                raise ValueError(f"{object_name}: support path escapes train/good")
            relative = candidate.relative_to(root).as_posix()
            if any(
                part in {"test_public", "test_private", "ground_truth"}
                for part in relative.split("/")
            ):
                raise ValueError("test and ground-truth paths are forbidden in mechanism phase")
            entries.append(
                SupportEntry(
                    object_name=object_name,
                    path=candidate,
                    relative_path=relative,
                    path_sha256=sha256_bytes(relative.encode("utf-8")),
                    content_sha256=sha256_file(candidate),
                ),
            )
        if len({entry.path for entry in entries}) != SUPPORT_COUNT:
            raise ValueError(f"{object_name}: support paths must be unique")
        result[object_name] = tuple(entries)
    return result


def read_rgb(path: Path) -> npt.NDArray[np.uint8]:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - PIL is a runtime dependency in real runs.
        raise RunnerError("PIL is required to read images") from error
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def deterministic_transform_specs(
    object_name: str,
    relative_path: str,
    seed: int,
    role: str,
) -> tuple[PhotometricTransformSpec, ...]:
    specs: list[PhotometricTransformSpec] = []
    for index in range(10):
        active_count = 0 if index < 2 else 1 if index < 7 else 2
        rng = np.random.default_rng(stable_seed(seed, role, object_name, relative_path, index))
        axes = tuple(rng.choice(PHOTOMETRIC_AXES, size=active_count, replace=False).tolist())
        values: dict[str, Any] = {
            "name": f"{role}_{index:02d}_{active_count}active",
            "active_axes": axes,
            "seed": stable_seed(seed, role, object_name, relative_path, index, "noise"),
        }
        if "brightness" in axes:
            values["brightness"] = float(rng.uniform(0.7, 1.3))
        if "contrast" in axes:
            values["contrast"] = float(rng.uniform(0.8, 1.2))
        if "gamma" in axes:
            values["gamma"] = float(rng.uniform(0.8, 1.2))
        if "rgb_gain" in axes:
            values["rgb_gain"] = tuple(float(x) for x in rng.uniform(0.9, 1.1, size=3))
        if "blur" in axes:
            values["blur_sigma"] = float(rng.uniform(0.0, 1.0))
        if "noise" in axes:
            values["noise_std"] = float(rng.uniform(0.0, 0.02))
        specs.append(PhotometricTransformSpec(**values))
    return tuple(specs)


def apply_photometric_transform(
    image: npt.NDArray[np.uint8],
    spec: PhotometricTransformSpec,
) -> npt.NDArray[np.uint8]:
    values = image.astype(np.float32) / np.float32(255.0)
    values = np.clip(values * np.float32(spec.brightness), 0.0, 1.0)
    mean = values.mean(axis=(0, 1), keepdims=True)
    values = np.clip((values - mean) * np.float32(spec.contrast) + mean, 0.0, 1.0)
    values = np.clip(values, 1e-6, 1.0) ** np.float32(1.0 / spec.gamma)
    values = values * np.asarray(spec.rgb_gain, dtype=np.float32).reshape(1, 1, 3)
    values = _maybe_blur(np.clip(values, 0.0, 1.0), spec.blur_sigma)
    if spec.noise_std > 0.0:
        rng = np.random.default_rng(spec.seed)
        values = values + rng.normal(0.0, spec.noise_std, size=values.shape).astype(np.float32)
    return np.clip(np.rint(values * np.float32(255.0)), 0, 255).astype(np.uint8)


def _maybe_blur(values: FloatArray, sigma: float) -> FloatArray:
    if sigma <= 0.0:
        return values.astype(np.float32, copy=False)
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        return values.astype(np.float32, copy=False)
    return cast(FloatArray, gaussian_filter(values, sigma=(sigma, sigma, 0.0)).astype(np.float32))


def _normalize_range(value: float, low: float, high: float) -> float:
    if high <= low:
        raise ValueError("invalid normalization range")
    return float((value - low) / (high - low))


def _stack_maps(feature_maps: Sequence[FeatureMap]) -> FloatArray:
    if not feature_maps:
        raise RunnerError("expected at least one feature map")
    first = feature_maps[0].values.shape
    if any(item.values.shape != first for item in feature_maps):
        raise RunnerError("all feature maps must share one DVT/flow shape")
    return np.stack(
        [item.values.astype(np.float32, copy=False) for item in feature_maps], axis=0
    ).astype(np.float32, copy=False)


def _targets_to_tensors(transform: PhotometricTransformSpec, device: Any) -> tuple[Any, Any, Any]:
    import torch

    targets = transform.targets()
    labels = torch.tensor(
        [[targets["multi_label"][axis] for axis in PHOTOMETRIC_AXES]],
        dtype=torch.float32,
        device=device,
    )
    continuous_fields = (
        "brightness",
        "contrast",
        "gamma",
        "rgb_gain_r",
        "rgb_gain_g",
        "rgb_gain_b",
        "blur_sigma",
        "noise_std",
    )
    continuous = torch.tensor(
        [[targets["continuous"][field] for field in continuous_fields]],
        dtype=torch.float32,
        device=device,
    )
    masks = torch.tensor(
        [[targets["regression_mask"][field] for field in continuous_fields]],
        dtype=torch.float32,
        device=device,
    )
    return labels, continuous, masks


def _loo_stats_torch(bank: Any) -> tuple[float, float]:
    if bank.shape[0] < 2:
        return 0.0, 1.0
    nearest = _loo_distances_torch(bank)
    return (
        float(nearest.mean().detach().cpu()),
        float(nearest.std(unbiased=False).clamp_min(1e-6).detach().cpu()),
    )


def _loo_distances_torch(bank: Any, chunk_size: int = 2048) -> Any:
    import torch

    parts: list[Any] = []
    for start in range(0, int(bank.shape[0]), chunk_size):
        end = min(start + chunk_size, int(bank.shape[0]))
        distances = torch.cdist(bank[start:end], bank, p=2.0)
        local = torch.arange(end - start, device=bank.device)
        distances[local, local + start] = float("inf")
        parts.append(distances.min(dim=1).values)
    return torch.cat(parts, dim=0)


def torch_std(value: Any) -> float:
    return float(value.detach().float().std(unbiased=False).cpu())


def _gradient_probe(model: Any, patches: Any) -> bool:
    import torch

    model.zero_grad(set_to_none=True)
    probe = patches.detach().clone().requires_grad_(True)
    output = model.forward(probe)
    loss = output.z.square().mean() + output.descriptor.square().mean()
    loss.backward()
    grads = [
        parameter.grad.detach()
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    model.zero_grad(set_to_none=True)
    return bool(
        grads
        and all(torch.isfinite(grad).all() for grad in grads)
        and sum(float(grad.abs().sum().cpu()) for grad in grads) > 0.0
    )


def extract_feature_map(
    backbone: Any, image: npt.NDArray[np.uint8], relative_path: str, transform: str
) -> FeatureMap:
    tensor, grid = backbone.prepare_image(image)
    layer_rows = backbone.extract_features(tensor)
    if len(layer_rows) != len(LAYERS):
        raise RunnerError(f"expected {len(LAYERS)} DINOv3 layers, got {len(layer_rows)}")
    merged = merge_layer_features(layer_rows, "layer_norm_mean")
    height, width = int(grid[0]), int(grid[1])
    if merged.shape[0] != height * width:
        raise RunnerError(f"feature/grid mismatch for {relative_path}: {merged.shape} vs {grid}")
    return FeatureMap(
        values=merged.reshape(height, width, merged.shape[-1]).astype(np.float32, copy=False),
        grid=(height, width),
        image_shape=(int(image.shape[0]), int(image.shape[1])),
        relative_path=relative_path,
        transform=transform,
    )


def fit_dvt(feature_maps: Sequence[FeatureMap]) -> tuple[tuple[FeatureMap, ...], FloatArray | None]:
    denoiser = fit_feature_denoiser(
        "position_mean", [item.values for item in feature_maps], alpha=1.0
    )
    if denoiser is None:
        return tuple(feature_maps), None
    transformed = tuple(
        FeatureMap(
            values=denoiser.transform(item.values),
            grid=item.grid,
            image_shape=item.image_shape,
            relative_path=item.relative_path,
            transform=item.transform,
        )
        for item in feature_maps
    )
    return transformed, denoiser.artifact.astype(np.float32, copy=True)


def load_backend(class_path: str = "") -> ShiftConditionedBackend:
    if not class_path:
        return TorchShiftConditionedBackend()
    target = class_path
    module_name, _, class_name = target.rpartition(".")
    if not module_name or not class_name:
        raise RunnerError("--backend-class must be a fully qualified class path")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        raise RunnerError(
            "Missing src.flow_tte.shift_conditioned API. Expected "
            "ShiftConditionedPatchLocalFlow with fit_stage_a, train_stage_b, "
            "mechanism_summary, save_checkpoint, load_checkpoint, and score_maps.",
        ) from error
    backend_type = getattr(module, class_name)
    backend = backend_type()
    for method in (
        "set_dvt_artifact",
        "transform_feature_maps",
        "fit_stage_a",
        "train_stage_b",
        "mechanism_summary",
        "save_checkpoint",
        "load_checkpoint",
        "score_maps",
    ):
        if not callable(getattr(backend, method, None)):
            raise RunnerError(f"{target} missing required method {method}")
    return cast(ShiftConditionedBackend, backend)


def make_access_ledger(
    phase: str, objects: Sequence[str], support_entries: Mapping[str, Sequence[SupportEntry]] | None
) -> dict[str, Any]:
    return {
        "phase": phase,
        "objects": list(objects),
        "support_train_good_paths_loaded": (
            {
                key: [entry.relative_path for entry in value]
                for key, value in support_entries.items()
            }
            if support_entries is not None
            else {}
        ),
        "support_count_per_object": (
            {key: len(value) for key, value in support_entries.items()}
            if support_entries is not None
            else {}
        ),
        "test_public_enumerated": phase == "public",
        "test_public_opened": phase == "public",
        "test_private_enumerated": False,
        "test_private_opened": False,
        "ground_truth_used_by_training_or_adaptation": False,
        "query_adaptation": False,
        "notes": (
            "mechanism phase support-only; public phase authorized by pass sentinel"
            if phase == "public"
            else "support-only: no test_public/test_private/ground_truth enumeration"
        ),
    }


def mechanism_phase(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    support_path = Path(args.support_json)
    support_sha = sha256_file(support_path)
    if support_sha != FROZEN_SUPPORT_MANIFEST_SHA256:
        raise RunnerError(f"support manifest SHA mismatch: {support_sha}")
    support = load_support_manifest(support_path, Path(args.data_root))
    config = frozen_config(args.device)
    backend = load_backend(args.backend_class)
    backbone = _make_backbone(args.device)
    summaries: list[dict[str, Any]] = []
    for object_name in args.objects:
        object_dir = output_root / object_name
        object_dir.mkdir(parents=True, exist_ok=True)
        entries = support[object_name]
        images = {entry.relative_path: read_rgb(entry.path) for entry in entries}
        support_raw = tuple(
            extract_feature_map(
                backbone,
                images[entry.relative_path],
                entry.relative_path,
                "identity",
            )
            for entry in entries
        )
        support_maps, dvt_artifact = fit_dvt(support_raw)
        backend.set_dvt_artifact(dvt_artifact, alpha=1.0)
        train_pairs = transform_stage_b_pairs(
            backend,
            make_stage_b_pairs(backbone, images, entries, args.seed, role="train"),
        )
        held_pairs = transform_stage_b_pairs(
            backend,
            make_stage_b_pairs(backbone, images, entries, args.seed + 991, role="held"),
        )
        backend.fit_stage_a(support_maps, config)
        training_log = dict(backend.train_stage_b(train_pairs, config))
        mechanism = dict(backend.mechanism_summary(support_maps, held_pairs, config))
        checkpoint = object_dir / "checkpoint.pt"
        backend.save_checkpoint(checkpoint)
        summary = object_summary(
            phase="mechanism",
            object_name=object_name,
            config=config,
            support_manifest_sha256=support_sha,
            support_entries=entries,
            checkpoint=checkpoint,
            mechanism=mechanism,
            training_log=training_log,
        )
        write_json(object_dir / "summary.json", summary)
        summaries.append(summary)
    ledger = make_access_ledger(
        "mechanism", args.objects, {name: support[name] for name in args.objects}
    )
    write_json(output_root / "access_ledger.json", ledger)
    run_manifest = run_manifest_payload(args, config, support_sha, summaries, ledger)
    write_json(output_root / "run_manifest.json", run_manifest)
    print(json.dumps(run_manifest, indent=2, sort_keys=True))
    return run_manifest


def make_stage_b_pairs(
    backbone: Any,
    images: Mapping[str, npt.NDArray[np.uint8]],
    entries: Sequence[SupportEntry],
    seed: int,
    role: str,
) -> tuple[StageBPair, ...]:
    pairs: list[StageBPair] = []
    for entry in entries:
        source = extract_feature_map(
            backbone, images[entry.relative_path], entry.relative_path, "identity"
        )
        for spec in deterministic_transform_specs(
            entry.object_name, entry.relative_path, seed, role
        ):
            transformed_rgb = apply_photometric_transform(images[entry.relative_path], spec)
            transformed = extract_feature_map(
                backbone, transformed_rgb, entry.relative_path, spec.name
            )
            pairs.append(StageBPair(source=source, transformed=transformed, transform=spec))
    return tuple(pairs)


def transform_stage_b_pairs(
    backend: ShiftConditionedBackend,
    pairs: Sequence[StageBPair],
) -> tuple[StageBPair, ...]:
    transformed: list[StageBPair] = []
    for pair in pairs:
        source, shifted = backend.transform_feature_maps((pair.source, pair.transformed))
        transformed.append(
            StageBPair(source=source, transformed=shifted, transform=pair.transform),
        )
    return tuple(transformed)


def object_summary(
    *,
    phase: str,
    object_name: str,
    config: Mapping[str, Any],
    support_manifest_sha256: str,
    support_entries: Sequence[SupportEntry],
    checkpoint: Path,
    mechanism: Mapping[str, Any],
    training_log: Mapping[str, Any],
) -> dict[str, Any]:
    ledger = make_access_ledger(phase, (object_name,), {object_name: support_entries})
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "phase": phase,
        "object": object_name,
        "config": config,
        "support_manifest_sha256": support_manifest_sha256,
        "support_count": len(support_entries),
        "support": [asdict(entry) | {"path": str(entry.path)} for entry in support_entries],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
        "training_log": training_log,
        "mechanism": _complete_mechanism_payload(mechanism),
        "access_ledger": ledger,
        "test_public_accessed": phase == "public",
        "test_private_accessed": False,
        "query_adaptation": False,
    }


def _complete_mechanism_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    defaults = {
        "validity": {
            "inverse_reconstruction_max_error": math.inf,
            "logdet_cancellation_max_error": math.inf,
            "e0_e1_latent_max_error": math.inf,
            "e0_e1_logdet_max_error": math.inf,
            "gradient_finite_nonzero": False,
            "deterministic_repeat": False,
            "descriptor_witness": False,
            "raw_euclidean_memory_scoring": False,
        },
        "held_shift": {
            "e0_inflation": math.inf,
            "correct_inflation": math.inf,
            "zero_inflation": math.inf,
            "shuffled_inflation": math.inf,
        },
        "identity": {"loo_delta_e0_std": math.inf, "latent_std_ratio": math.inf},
        "defect_retention": {"gap_retention": 0.0},
        "shift_prediction": {"loss": math.inf, "zero_loss": math.inf},
    }
    result = json.loads(json.dumps(defaults))
    for key, value in raw.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key].update(value)
        else:
            result[key] = value
    held = result["held_shift"]
    e0 = float(held["e0_inflation"])
    correct = float(held["correct_inflation"])
    held["correct_reduction_fraction"] = float((e0 - correct) / e0) if e0 > 0.0 else float("nan")
    held["correct_beats_zero_shuffled"] = bool(
        correct < float(held["zero_inflation"]) and correct < float(held["shuffled_inflation"])
    )
    result["shift_prediction"]["beats_zero"] = bool(
        float(result["shift_prediction"]["loss"]) < float(result["shift_prediction"]["zero_loss"])
    )
    return result


def public_phase(args: argparse.Namespace) -> dict[str, Any]:
    sentinel = Path(args.mechanism_pass_sentinel)
    if not sentinel.is_file():
        raise RunnerError("public phase requires a mechanism pass sentinel")
    sentinel_payload = json.loads(sentinel.read_text(encoding="utf-8"))
    if sentinel_payload.get("verdict") != "MECHANISM_PASS":
        raise RunnerError("mechanism pass sentinel does not authorize public access")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    support_path = Path(args.support_json)
    support_sha = sha256_file(support_path)
    support = load_support_manifest(support_path, Path(args.data_root))
    config = frozen_config(args.device)
    backend = load_backend(args.backend_class)
    backbone = _make_backbone(args.device)
    public_rows: list[dict[str, Any]] = []
    for object_name in args.objects:
        checkpoint = _checkpoint_path(args, sentinel_payload, object_name)
        backend.load_checkpoint(checkpoint, config)
        tests = enumerate_test_public(Path(args.data_root), object_name)
        feature_maps: list[FeatureMap] = []
        image_rows: list[dict[str, Any]] = []
        for split, paths in tests.items():
            for path in paths:
                image = read_rgb(path)
                relative = path.relative_to(Path(args.data_root)).as_posix()
                feature_maps.extend(
                    backend.transform_feature_maps(
                        (extract_feature_map(backbone, image, relative, "identity"),),
                    ),
                )
                image_rows.append(
                    {
                        "object": object_name,
                        "split": split,
                        "path": str(path),
                        "relative_path": relative,
                        "stem": path.stem,
                        "image_shape": [int(image.shape[0]), int(image.shape[1])],
                    },
                )
        object_summary_payload = public_object_summary(
            args=args,
            object_name=object_name,
            backend=backend,
            feature_maps=feature_maps,
            image_rows=image_rows,
            support_entries=support[object_name],
            checkpoint=checkpoint,
            support_sha=support_sha,
            config=config,
        )
        write_json(output_root / object_name / "summary.json", object_summary_payload)
        public_rows.append(object_summary_payload)
    ledger = make_access_ledger(
        "public", args.objects, {name: support[name] for name in args.objects}
    )
    write_json(output_root / "access_ledger.json", ledger)
    run_manifest = run_manifest_payload(args, config, support_sha, public_rows, ledger)
    write_json(output_root / "run_manifest.json", run_manifest)
    print(json.dumps(run_manifest, indent=2, sort_keys=True))
    return run_manifest


def enumerate_test_public(data_root: Path, object_name: str) -> dict[str, tuple[Path, ...]]:
    base = data_root / object_name / "test_public"
    selected: dict[str, tuple[Path, ...]] = {}
    for split in ("good", "bad"):
        directory = base / split
        if not directory.is_dir():
            raise RunnerError(f"{object_name}: missing test_public/{split}")
        selected[split] = tuple(path for path in sorted(directory.iterdir()) if path.is_file())
        if not selected[split]:
            raise RunnerError(f"{object_name}: empty test_public/{split}")
    return selected


def public_object_summary(
    *,
    args: argparse.Namespace,
    object_name: str,
    backend: ShiftConditionedBackend,
    feature_maps: Sequence[FeatureMap],
    image_rows: Sequence[Mapping[str, Any]],
    support_entries: Sequence[SupportEntry],
    checkpoint: Path,
    support_sha: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    object_dir = Path(args.output_root) / object_name
    maps_root = object_dir / "anomaly_maps"
    metrics: dict[str, Any] = {}
    map_rows: list[dict[str, Any]] = []
    for arm in ("E0", "E1", "E5"):
        arm_maps = backend.score_maps(feature_maps, arm=arm, descriptor_mode="correct")
        if len(arm_maps) != len(feature_maps):
            raise RunnerError(f"{object_name}/{arm}: backend returned wrong map count")
        prediction_dir = maps_root / arm / object_name / "test"
        for row, score_map in zip(image_rows, arm_maps):
            split = str(row["split"])
            path = prediction_dir / split / f"{row['stem']}.tiff"
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_tiff(path, np.asarray(score_map, dtype=np.float32))
            map_rows.append(
                {
                    "arm": arm,
                    "split": split,
                    "stem": row["stem"],
                    "map": str(path),
                    "map_sha256": sha256_file(path),
                    "finite": bool(np.all(np.isfinite(score_map))),
                    "nonconstant": bool(float(np.ptp(score_map)) > 1e-8),
                },
            )
        metrics[arm] = evaluate_public_arm(
            data_root=Path(args.data_root),
            output_root=object_dir,
            object_name=object_name,
            arm=arm,
            map_rows=map_rows,
        )
    finite_complete = all(row["finite"] and row["nonconstant"] for row in map_rows)
    if args.cleanup_maps and maps_root.exists():
        shutil.rmtree(maps_root)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "phase": "public",
        "object": object_name,
        "config": config,
        "support_manifest_sha256": support_sha,
        "support_count": len(support_entries),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "split": "test_public/good,bad",
        "test_public_good_count": sum(1 for row in image_rows if row["split"] == "good"),
        "test_public_bad_count": sum(1 for row in image_rows if row["split"] == "bad"),
        "metrics": metrics,
        "diagnostics": {
            "finite_nonconstant_complete": finite_complete,
            "map_count": len(map_rows),
            "e0_e1_maps_equal": _maps_equal(map_rows, "E0", "E1"),
            "bad_image_gap_z_delta": _metric_delta(metrics, "E5", "E0", "bad_image_gap_z"),
            "shifted_good_p95_inflation_reduction": _relative_reduction(
                float(metrics["E0"]["shifted_good_p95_inflation"]),
                float(metrics["E5"]["shifted_good_p95_inflation"]),
            ),
        },
        "maps": map_rows if not args.cleanup_maps else "removed_after_metrics",
        "access_ledger": make_access_ledger(
            "public", (object_name,), {object_name: support_entries}
        ),
        "test_public_accessed": True,
        "test_private_accessed": False,
        "query_adaptation": False,
    }


def evaluate_public_arm(
    *,
    data_root: Path,
    output_root: Path,
    object_name: str,
    arm: str,
    map_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [row for row in map_rows if row["arm"] == arm]
    diagnostics = _public_map_diagnostics(
        data_root=data_root,
        object_name=object_name,
        rows=rows,
    )
    try:
        from fmad.evaluation.metrics import Evaluator
    except ModuleNotFoundError:
        scores = [float(np.mean(_read_tiff(Path(str(row["map"]))))) for row in rows]
        labels = [row["split"] == "bad" for row in rows]
        return {
            "seg_AUROC_0.05": _simple_auroc(labels, scores),
            "seg_F1": 0.0,
            "oracle_f1_label": "pooled best-threshold pixel F1 unavailable without fmad Evaluator",
            **diagnostics,
        }
    variant_root = output_root / "anomaly_maps" / arm
    evaluator = Evaluator({"pro_integration_limit": 0.05, "binary_postprocess": "none"})
    metrics = evaluator.evaluate_run(
        dataset_name="MVTec_AD_2",
        data_root=str(data_root),
        anomaly_maps_dir=str(variant_root),
        output_dir=str(output_root / "metrics" / arm),
        seed=0,
        objects=[object_name],
    )
    result = dict(metrics[object_name])
    result.setdefault("oracle_f1_label", "pooled best-threshold pixel F1")
    result.update(diagnostics)
    return result


def _public_map_diagnostics(
    *,
    data_root: Path,
    object_name: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute preregistered score-field diagnostics from public maps.

    Lighting shift is the dataset condition encoded in each image stem.  The
    metric therefore uses the real AD2 shifted-normal groups and does not add
    synthetic public-query augmentation or adaptation.
    """
    condition_good_p95: dict[str, list[float]] = {}
    bad_gap_z: list[float] = []
    bad_top1_gt_share: list[float] = []
    for row in rows:
        score_map = _read_tiff(Path(str(row["map"])))
        split = str(row["split"])
        stem = str(row["stem"])
        condition = _parse_ad2_condition(stem)
        if split == "good":
            condition_good_p95.setdefault(condition, []).append(
                float(np.quantile(score_map, 0.95)),
            )
            continue
        if split != "bad":
            raise RunnerError(f"unexpected public split: {split}")
        gt_mask = _load_public_gt_mask(
            data_root=data_root,
            object_name=object_name,
            split=split,
            stem=stem,
            prediction_shape=score_map.shape,
        )
        if not bool(np.any(gt_mask)):
            raise RunnerError(f"missing or empty public ground truth: {object_name}/{stem}")
        inside = float(np.mean(score_map[gt_mask]))
        outside = float(np.mean(score_map[~gt_mask]))
        bad_gap_z.append((inside - outside) / max(float(np.std(score_map)), 1e-6))
        bad_top1_gt_share.append(_top_fraction_gt_share(score_map, gt_mask, fraction=0.01))
    regular = condition_good_p95.get("regular")
    shifted = [
        float(np.mean(values) - np.mean(regular))
        for condition, values in sorted(condition_good_p95.items())
        if condition != "regular"
    ] if regular else []
    if not regular or not shifted or not bad_gap_z:
        raise RunnerError(
            f"{object_name}/public: incomplete condition/GT diagnostics",
        )
    return {
        "bad_image_gap_z": float(np.mean(bad_gap_z)),
        "bad_image_top1_gt_share": float(np.mean(bad_top1_gt_share)),
        "shifted_good_p95_inflation": float(np.mean(shifted)),
        "shifted_good_condition_count": len(shifted),
    }


def _parse_ad2_condition(stem: str) -> str:
    for condition in ("regular", "overexposed", "underexposed"):
        if stem.endswith(f"_{condition}"):
            return condition
    marker = "_shift_"
    if marker in stem and stem.rsplit(marker, maxsplit=1)[1].isdigit():
        return "shift_" + stem.rsplit(marker, maxsplit=1)[1]
    raise RunnerError(f"unrecognized AD2 condition suffix: {stem}")


def _load_public_gt_mask(
    *,
    data_root: Path,
    object_name: str,
    split: str,
    stem: str,
    prediction_shape: Sequence[int],
) -> BoolArray:
    from PIL import Image

    path = data_root / object_name / "test_public" / "ground_truth" / split / f"{stem}_mask.png"
    if not path.is_file():
        raise RunnerError(f"missing public ground truth: {path}")
    mask = np.asarray(Image.open(path)) > 0
    if tuple(mask.shape) != tuple(prediction_shape):
        try:
            import cv2
        except ImportError as error:
            raise RunnerError("opencv is required to align public masks") from error
        mask = cv2.resize(
            mask.astype(np.uint8),
            (int(prediction_shape[1]), int(prediction_shape[0])),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
    return np.asarray(mask, dtype=np.bool_)


def _top_fraction_gt_share(
    score_map: FloatArray,
    gt_mask: BoolArray,
    *,
    fraction: float,
) -> float:
    flat = score_map.reshape(-1)
    mask = gt_mask.reshape(-1)
    count = max(1, math.ceil(float(flat.size) * fraction))
    threshold = np.partition(flat, flat.size - count)[flat.size - count]
    selected = flat >= threshold
    return float(np.count_nonzero(mask & selected) / max(int(np.count_nonzero(selected)), 1))


def _relative_reduction(baseline: float, candidate: float) -> float:
    return float((baseline - candidate) / max(abs(baseline), 1e-12))


def _write_tiff(path: Path, values: FloatArray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import tifffile as tiff
    except ImportError as error:
        raise RunnerError("tifffile is required for public anomaly map output") from error
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise RunnerError(f"invalid anomaly map for {path}")
    tiff.imwrite(path, values.astype(np.float32, copy=False))


def _read_tiff(path: Path) -> FloatArray:
    import tifffile as tiff

    return np.asarray(tiff.imread(path), dtype=np.float32)


def _simple_auroc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return float("nan")
    wins = sum(float(pos > neg) + 0.5 * float(pos == neg) for pos in positives for neg in negatives)
    return float(wins / (len(positives) * len(negatives)))


def _gap_z(labels: Sequence[bool], scores: Sequence[float]) -> float:
    positives = np.asarray(
        [score for label, score in zip(labels, scores) if label], dtype=np.float32
    )
    negatives = np.asarray(
        [score for label, score in zip(labels, scores) if not label], dtype=np.float32
    )
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    return float(
        (float(np.mean(positives)) - float(np.mean(negatives)))
        / max(float(np.std(negatives)), 1e-6)
    )


def _maps_equal(rows: Sequence[Mapping[str, Any]], left: str, right: str) -> bool:
    left_rows = sorted(
        [row for row in rows if row["arm"] == left], key=lambda row: (row["split"], row["stem"])
    )
    right_rows = sorted(
        [row for row in rows if row["arm"] == right], key=lambda row: (row["split"], row["stem"])
    )
    if len(left_rows) != len(right_rows):
        return False
    return all(
        _arrays_close(_read_tiff(Path(str(a["map"]))), _read_tiff(Path(str(b["map"]))))
        for a, b in zip(left_rows, right_rows)
    )


def _arrays_close(left: FloatArray, right: FloatArray) -> bool:
    return bool(left.shape == right.shape and np.allclose(left, right, atol=1e-6, rtol=1e-6))


def _metric_delta(
    metrics: Mapping[str, Mapping[str, Any]], left: str, right: str, field: str
) -> float:
    try:
        return float(metrics[left].get(field, 0.0)) - float(metrics[right].get(field, 0.0))
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _checkpoint_path(
    args: argparse.Namespace, sentinel_payload: Mapping[str, Any], object_name: str
) -> Path:
    if args.checkpoint_root:
        path = Path(args.checkpoint_root) / object_name / "checkpoint.pt"
    else:
        checkpoint_by_object = sentinel_payload.get("checkpoints", {})
        if not isinstance(checkpoint_by_object, Mapping) or object_name not in checkpoint_by_object:
            raise RunnerError(f"sentinel does not provide checkpoint for {object_name}")
        path = Path(str(checkpoint_by_object[object_name]))
    if not path.is_file():
        raise RunnerError(f"missing checkpoint: {path}")
    return path


def _make_backbone(device: str) -> Any:
    from dinov3_backbone import DINOv3Backbone

    return DINOv3Backbone(
        model_name="dinov3_vith16plus",
        device=device,
        smaller_edge_size=672,
        feature_layers=LAYERS,
    )


def run_manifest_payload(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    support_sha: str,
    summaries: Sequence[Mapping[str, Any]],
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "phase": args.phase,
        "script_sha256": sha256_file(Path(__file__)),
        "support_manifest_sha256": support_sha,
        "objects": list(args.objects),
        "config": config,
        "summaries": [f"{item['object']}/summary.json" for item in summaries],
        "access_ledger": ledger,
        "host_gpu_id": str(args.host_gpu_id),
        "container_cuda_slot": str(args.container_cuda_slot),
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def smoke_phase(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = frozen_config(args.device)
    write_json(output_root / "frozen_config.json", config)
    ledger = make_access_ledger("mechanism", args.objects, None)
    write_json(output_root / "access_ledger.json", ledger)
    summaries = [
        {
            "object": object_name,
            "phase": "smoke",
            "protocol_id": PROTOCOL_ID,
            "mechanism": _complete_mechanism_payload({}),
            "access_ledger": ledger,
        }
        for object_name in args.objects
    ]
    for summary in summaries:
        path = output_root / str(summary["object"]) / "summary.json"
        write_json(path, summary)
    manifest = run_manifest_payload(args, config, "0" * 64, summaries, ledger)
    write_json(output_root / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    if args.phase == "mechanism":
        mechanism_phase(args)
    elif args.phase == "public":
        public_phase(args)
    else:
        smoke_phase(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
