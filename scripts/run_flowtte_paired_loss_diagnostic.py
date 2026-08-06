#!/usr/bin/env python3
# ruff: noqa: ANN401, ARG002, C901, COM812, E501, EM102, FBT003, PERF401, PLC0415, PLR0913, TC006
"""Run the preregistered FlowTTE paired-loss support-only diagnostic.

This runner deliberately returns to Baseline1, the strongest retained
patch-local MLP FlowTTE method, and changes only the Stage-B loss.  The
mechanism phase reads
only the frozen train/good support set, writes an access ledger that forbids
test/public/private/GT access, and records compact per-epoch rows for live
inspection.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
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
STAGE_MINUS1_OBJECTS = ("fruit_jelly", "walnuts", "vial", "fabric")
SUPPORT_COUNT = 16
LAYERS = (7, 15, 23, 31)
PROTOCOL_ID = "flowtte-paired-loss-diagnostic-20260804"
BASELINE_METHOD = "Baseline1"
# Kept stable because completed v4 artifacts are checksum-bound to this arm ID.
BASELINE_ARM_ID = "V0_default"
FROZEN_SUPPORT_MANIFEST_SHA256 = "9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5"
PHOTOMETRIC_AXES = ("brightness", "contrast", "gamma", "rgb_gain", "blur", "noise")
STAGE_B_EPOCHS = 5
STAGE_B_LR = 1e-4
STAGE_B_WEIGHT_DECAY = 1e-4
STAGE_B_GRAD_CLIP = 0.5
VARIANTS = (
    "V0_default",
    "V1_aug_nll",
    "V2_raw_pair_nll1",
    "V3_raw_pair_nll01",
    "C1_normalized_pair",
    "C2_random_pair",
)
PROMOTION_CANDIDATES = ("V2_raw_pair_nll1", "V3_raw_pair_nll01")

FloatArray = npt.NDArray[np.float32]


class RunnerError(RuntimeError):
    """Raised when the frozen support-only protocol cannot be satisfied."""


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


class PairedLossBackend(Protocol):
    def set_dvt_artifact(self, artifact: FloatArray | None, alpha: float) -> None: ...

    def transform_feature_maps(self, feature_maps: Sequence[FeatureMap]) -> tuple[FeatureMap, ...]: ...

    def fit_stage_a(self, support_maps: Sequence[FeatureMap], config: Mapping[str, Any]) -> None: ...

    def train_stage_b(
        self,
        pairs: Sequence[StageBPair],
        held_pairs: Sequence[StageBPair],
        config: Mapping[str, Any],
        progress: "ProgressWriter | None" = None,
        object_name: str = "",
    ) -> Mapping[str, Any]: ...

    def mechanism_summary(
        self,
        support_maps: Sequence[FeatureMap],
        held_pairs: Sequence[StageBPair],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def save_checkpoint(self, path: Path) -> None: ...


class TorchPairedLossBackend:
    """Baseline-flow-only backend for paired-loss diagnostics."""

    def __init__(self) -> None:
        self.device_name = "cpu"
        self.stage_a: Any = None
        self.stage_a_training: dict[str, Any] = {}
        self.support_maps: tuple[FeatureMap, ...] = ()
        self.flows: dict[str, Any] = {}
        self.banks: dict[str, Any] = {}
        self.loo_stats: dict[str, tuple[float, float]] = {}
        self.v0_defect_gaps: list[float] | None = None
        self.final_metrics: dict[str, dict[str, Any]] = {}
        self.dvt_artifact: FloatArray | None = None
        self.dvt_alpha = 1.0

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
                        np.float32,
                        copy=False,
                    ),
                    grid=item.grid,
                    image_shape=item.image_shape,
                    relative_path=item.relative_path,
                    transform=item.transform,
                )
            )
        return tuple(result)

    def fit_stage_a(self, support_maps: Sequence[FeatureMap], config: Mapping[str, Any]) -> None:
        from flow_tte.config import ExpansionConfig
        from flow_tte.trainer import FlowDensityEstimator

        if not support_maps:
            raise RunnerError("Stage-A requires support feature maps")
        self.device_name = str(config.get("device", "cpu"))
        estimator = FlowDensityEstimator(
            dim=int(support_maps[0].values.shape[-1]),
            config=FlowConfig(**dict(config["stage_a_flow"])),
            device=self.device_name,
        )
        stats = estimator.fit(
            _stack_maps(support_maps),
            density_quantile=float(ExpansionConfig().density_quantile),
        )
        self.stage_a = estimator
        self.support_maps = tuple(support_maps)
        self.final_metrics = {}
        self.stage_a_training = {
            "losses": [float(value) for value in stats.losses],
            "train_nll_mean": float(stats.train_nll_mean),
            "train_nll_std": float(stats.train_nll_std),
            "density_threshold": float(stats.density_threshold),
        }
        self._initialize_variant_flows()
        self._initialize_variant_banks()

    def train_stage_b(
        self,
        pairs: Sequence[StageBPair],
        held_pairs: Sequence[StageBPair],
        config: Mapping[str, Any],
        progress: "ProgressWriter | None" = None,
        object_name: str = "",
    ) -> Mapping[str, Any]:
        import torch
        from torch.nn.utils.clip_grad import clip_grad_norm_

        if self.stage_a is None:
            raise RunnerError("fit_stage_a must run before Stage-B")
        if not pairs:
            raise RunnerError("Stage-B requires deterministic photometric pairs")
        stage_b = dict(config["stage_b"])
        logs: dict[str, list[dict[str, float]]] = {variant: [] for variant in VARIANTS}
        for variant in VARIANTS:
            flow = self.flows[variant]
            if variant == BASELINE_ARM_ID:
                row = self._variant_metrics(variant, held_pairs)
                self.final_metrics[variant] = dict(row)
                row.update({"epoch": 0.0, "total": 0.0, "nll": 0.0, "raw_pair": 0.0, "stage": "final"})
                logs[variant].append(row)
                if progress is not None:
                    progress.write(object_name, variant, 0, row)
                continue
            optimizer = torch.optim.AdamW(
                flow.parameters(),
                lr=float(stage_b["lr"]),
                weight_decay=float(stage_b["weight_decay"]),
            )
            for epoch in range(1, int(stage_b["epochs"]) + 1):
                flow.train()
                accum = {
                    "total": 0.0,
                    "nll": 0.0,
                    "raw_pair": 0.0,
                    "norm_pair": 0.0,
                    "identity": 0.0,
                    "norm": 0.0,
                    "logdet_reg": 0.0,
                }
                order = deterministic_pair_order(len(pairs), stable_seed(PROTOCOL_ID, variant, epoch))
                for pair_index in order:
                    pair = pairs[pair_index]
                    source = self._standardized_flat([pair.source])
                    transformed = self._standardized_flat([pair.transformed])
                    with torch.no_grad():
                        identity_target, _ = self.flows[BASELINE_ARM_ID].forward(source)
                        pair_target = identity_target
                        if variant == "C2_random_pair":
                            pair_target = self._wrong_clean_image_target(pair.source)
                    source_z, source_logdet = flow.forward(source)
                    shifted_z, shifted_logdet = flow.forward(transformed)
                    losses = self._loss_terms(
                        variant,
                        source_z,
                        source_logdet,
                        shifted_z,
                        shifted_logdet,
                        pair_target.detach(),
                        identity_target.detach(),
                    )
                    optimizer.zero_grad()
                    losses["total"].backward()
                    clip_grad_norm_(flow.parameters(), max_norm=float(stage_b["gradient_clip"]))
                    optimizer.step()
                    for name, value in losses.items():
                        accum[name] += float(value.detach().cpu())
                for name in accum:
                    accum[name] /= float(len(pairs))
                row = {
                    "epoch": float(epoch),
                    "stage": "epoch",
                    **accum,
                    "held_raw_pair_distance": self._held_raw_pair_distance(
                        variant,
                        held_pairs,
                        max_pairs=32,
                    ),
                }
                logs[variant].append(row)
                if progress is not None:
                    progress.write(object_name, variant, epoch, row)
            self._refresh_banks((variant,))
            row = self._variant_metrics(variant, held_pairs)
            self.final_metrics[variant] = dict(row)
            row.update({"epoch": float(stage_b["epochs"]), "stage": "final"})
            logs[variant].append(row)
            if progress is not None:
                progress.write(object_name, variant, int(stage_b["epochs"]), row)
        return {
            "stage_a": self.stage_a_training,
            "stage_b_epoch_losses": logs,
            "pair_count": len(pairs),
            "variants": list(VARIANTS),
        }

    def mechanism_summary(
        self,
        support_maps: Sequence[FeatureMap],
        held_pairs: Sequence[StageBPair],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return {
            "variants": {
                variant: self._variant_full_summary(variant, held_pairs)
                for variant in VARIANTS
            },
            "validity": {
                variant: self._validity_summary(variant, support_maps)
                for variant in VARIANTS
            },
        }

    def save_checkpoint(self, path: Path) -> None:
        import torch

        if self.stage_a is None:
            raise RunnerError("cannot save before fitting")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "protocol_id": PROTOCOL_ID,
                "base_flow_state": self.stage_a.flow.state_dict(),
                "variant_flow_states": {
                    variant: flow.state_dict() for variant, flow in self.flows.items()
                },
                "standardizer": None
                if self.stage_a.standardizer is None
                else {
                    "mean": self.stage_a.standardizer.mean.detach().cpu(),
                    "std": self.stage_a.standardizer.std.detach().cpu(),
                },
                "support_maps": [asdict(item) | {"values": item.values} for item in self.support_maps],
                "banks": {key: value.detach().cpu() for key, value in self.banks.items()},
                "loo_stats": self.loo_stats,
                "dvt_artifact": self.dvt_artifact,
                "dvt_alpha": self.dvt_alpha,
                "stage_a_training": self.stage_a_training,
            },
            path,
        )

    def score_maps(self, feature_maps: Sequence[FeatureMap], variant: str) -> tuple[FloatArray, ...]:
        import torch

        if variant not in self.banks:
            raise RunnerError(f"unknown variant {variant}")
        bank = self.banks[variant]
        mean, std = self.loo_stats[variant]
        maps: list[FloatArray] = []
        with torch.no_grad():
            for item in feature_maps:
                latent = self._latent_for_maps([item], variant)
                distances = torch.cdist(latent, bank, p=2.0).min(dim=1).values
                scores = ((distances - mean) / max(std, 1e-6)).cpu().numpy().astype(np.float32)
                maps.append(scores.reshape(item.grid).astype(np.float32, copy=False))
        return tuple(maps)

    def _initialize_variant_flows(self) -> None:
        if self.stage_a is None:
            raise RunnerError("Stage-A is not fitted")
        self.flows = {}
        for variant in VARIANTS:
            flow = copy.deepcopy(self.stage_a.flow).to(self.stage_a.device)
            for parameter in flow.parameters():
                parameter.requires_grad_(variant != BASELINE_ARM_ID)
            self.flows[variant] = flow

    def _loss_terms(
        self,
        variant: str,
        source_z: Any,
        source_logdet: Any,
        shifted_z: Any,
        shifted_logdet: Any,
        target: Any,
        identity_target: Any,
    ) -> dict[str, Any]:
        from flow_tte.losses import tail_aware_nll

        zero = shifted_z.new_zeros(())
        dim = float(source_z.shape[-1])
        clean_nll = tail_aware_nll(source_z, source_logdet, tail_weight=0.3, tail_top_k_ratio=0.05)
        shifted_nll = tail_aware_nll(
            shifted_z,
            shifted_logdet,
            tail_weight=0.3,
            tail_top_k_ratio=0.05,
        )
        logdet_reg = 0.5 * (source_logdet.square().mean() + shifted_logdet.square().mean())
        normalized_logdet_reg = logdet_reg / (dim * dim)
        nll = 0.5 * (clean_nll + shifted_nll) / dim + 0.02 * normalized_logdet_reg
        raw_pair = raw_pair_distance_loss(shifted_z, target)
        norm_pair = normalized_pair_huber_loss(shifted_z, target)
        identity = clean_identity_loss(source_z, identity_target)
        norm = latent_norm_preservation_loss(shifted_z, target)
        if variant == "V1_aug_nll":
            total = nll
        elif variant == "V2_raw_pair_nll1":
            total = raw_pair + identity + 0.1 * norm + nll
        elif variant == "V3_raw_pair_nll01":
            total = raw_pair + identity + 0.1 * norm + 0.1 * nll
        elif variant == "C1_normalized_pair":
            total = norm_pair + identity + 0.1 * norm + 0.1 * nll
        elif variant == "C2_random_pair":
            total = raw_pair + identity + 0.1 * norm + 0.1 * nll
        else:
            total = zero
        return {
            "total": total,
            "nll": nll,
            "raw_pair": raw_pair if variant != "C1_normalized_pair" else zero,
            "norm_pair": norm_pair if variant == "C1_normalized_pair" else zero,
            "identity": identity if variant not in {"V1_aug_nll", BASELINE_ARM_ID} else zero,
            "norm": norm if variant not in {"V1_aug_nll", BASELINE_ARM_ID} else zero,
            "logdet_reg": normalized_logdet_reg,
        }

    def _variant_metrics(self, variant: str, held_pairs: Sequence[StageBPair]) -> dict[str, float]:
        baseline_metrics = self.final_metrics.get(BASELINE_ARM_ID)
        baseline = (
            float(baseline_metrics["shifted_normal_inflation"])
            if baseline_metrics is not None
            else self._held_shift_inflation(BASELINE_ARM_ID, held_pairs)
        )
        current = self._held_shift_inflation(variant, held_pairs)
        clean_delta, std_ratio = self._clean_identity_metrics(variant)
        validity = self._validity_summary(variant, self.support_maps)
        metrics = {
            "held_raw_pair_distance": self._held_raw_pair_distance(variant, held_pairs),
            "shifted_normal_inflation": current,
            "shift_reduction": _relative_reduction(baseline, current),
            "loo_delta": clean_delta,
            "std_ratio": std_ratio,
            "effective_rank_ratio": self._effective_rank_ratio(variant),
            "neighbor_retention": self._neighbor_retention(variant),
            "defect_retention": self._defect_retention(variant),
            "validity_passed": float(_validity_passed(validity)),
        }
        metrics["gate"] = float(self._variant_gate_from_metrics(metrics, validity))
        return metrics

    def _variant_full_summary(self, variant: str, held_pairs: Sequence[StageBPair]) -> dict[str, Any]:
        if variant in self.final_metrics:
            cached = dict(self.final_metrics[variant])
            validity = self._validity_summary(variant, self.support_maps)
            return {
                **cached,
                "gate_passed": bool(cached["gate"]),
                "validity_passed": bool(cached["validity_passed"]),
                "validity": validity,
            }
        metrics = self._variant_metrics(variant, held_pairs)
        validity = self._validity_summary(variant, self.support_maps)
        return {
            **metrics,
            "gate_passed": bool(metrics["gate"]),
            "validity_passed": bool(metrics["validity_passed"]),
            "validity": validity,
        }

    def _held_shift_inflation(self, variant: str, held_pairs: Sequence[StageBPair]) -> float:
        import torch

        bank = self.banks[variant]
        loo_mean, std = self.loo_stats[variant]
        values: list[float] = []
        with torch.no_grad():
            for pair in held_pairs:
                latent = self._latent_for_maps([pair.transformed], variant)
                distances = torch.cdist(latent, bank, p=2.0).min(dim=1).values
                values.append(
                    support_self_relative_inflation(
                        float(distances.mean().detach().cpu()),
                        loo_mean,
                        std,
                    )
                )
        return float(np.mean(values)) if values else float("inf")

    def _held_raw_pair_distance(
        self,
        variant: str,
        held_pairs: Sequence[StageBPair],
        max_pairs: int | None = None,
    ) -> float:
        import torch

        values: list[float] = []
        selected_pairs = tuple(held_pairs)
        if max_pairs is not None and len(selected_pairs) > max_pairs:
            rng = np.random.default_rng(stable_seed(PROTOCOL_ID, "held_raw_pair_sample", variant))
            indices = rng.choice(len(selected_pairs), size=max_pairs, replace=False)
            selected_pairs = tuple(selected_pairs[int(index)] for index in sorted(indices))
        with torch.no_grad():
            for pair in selected_pairs:
                source = self._standardized_flat([pair.source])
                transformed = self._standardized_flat([pair.transformed])
                target, _ = self.flows[BASELINE_ARM_ID].forward(source)
                shifted, _ = self.flows[variant].forward(transformed)
                values.append(float(raw_pair_distance_loss(shifted, target).cpu()))
        return float(np.mean(values)) if values else float("inf")

    def _clean_identity_metrics(self, variant: str) -> tuple[float, float]:
        import torch

        baseline_loo = _loo_distances_torch(self.banks[BASELINE_ARM_ID])
        current_loo = _loo_distances_torch(self.banks[variant])
        baseline_std = max(float(baseline_loo.std(unbiased=False).detach().cpu()), 1e-6)
        loo_delta = float((current_loo.mean() - baseline_loo.mean()).detach().cpu() / baseline_std)
        std_ratio = float(
            torch.std(self.banks[variant].float(), unbiased=False).detach().cpu()
            / max(float(torch.std(self.banks[BASELINE_ARM_ID].float(), unbiased=False).detach().cpu()), 1e-6)
        )
        return loo_delta, std_ratio

    def _effective_rank_ratio(self, variant: str) -> float:
        base, candidate = self._sampled_bank_pair(variant)
        return float(
            effective_rank(candidate)
            / max(float(effective_rank(base)), 1e-6)
        )

    def _neighbor_retention(self, variant: str) -> float:
        base, candidate = self._sampled_bank_pair(variant)
        return float(neighbor_identity_retention(base, candidate, k=10, max_samples=2048))

    def _defect_retention(self, variant: str) -> float:
        base_gaps = self._baseline_defect_gaps()
        candidate_gaps = self._defect_gaps_for(variant)
        return float(
            np.mean(
                [
                    candidate / max(base, 1e-6)
                    for candidate, base in zip(candidate_gaps, base_gaps)
                ]
            )
        )

    def _baseline_defect_gaps(self) -> list[float]:
        if self.v0_defect_gaps is None:
            self.v0_defect_gaps = self._defect_gaps_for(BASELINE_ARM_ID)
        return list(self.v0_defect_gaps)

    def _defect_gaps_for(self, variant: str) -> list[float]:
        import torch

        bank = self.banks[variant]
        _, std = self.loo_stats[variant]
        gaps: list[float] = []
        for item in self.support_maps:
            defect = item.values.copy()
            h, w = defect.shape[:2]
            ys = slice(h // 4, max(h // 4 + 1, h // 2))
            xs = slice(w // 4, max(w // 4 + 1, w // 2))
            defect[ys, xs] += np.float32(0.25)
            defect_map = FeatureMap(
                defect, item.grid, item.image_shape, item.relative_path, "synthetic_defect"
            )
            with torch.no_grad():
                latent = self._latent_for_maps([defect_map], variant)
                distances = torch.cdist(latent, bank, p=2.0).min(dim=1).values.reshape(item.grid)
            gaps.append(float(distances[ys, xs].mean().detach().cpu()) / max(std, 1e-6))
        return gaps

    def _variant_gate_from_metrics(
        self,
        metrics: Mapping[str, float],
        validity: Mapping[str, Any],
    ) -> bool:
        return bool(
            abs(metrics["loo_delta"]) <= 0.05
            and 0.8 <= metrics["std_ratio"] <= 1.25
            and metrics["effective_rank_ratio"] >= 0.90
            and metrics["neighbor_retention"] >= 0.90
            and metrics["defect_retention"] >= 0.90
            and _validity_passed(validity)
        )

    def _validity_summary(self, variant: str, support_maps: Sequence[FeatureMap]) -> dict[str, Any]:
        import torch

        flow = self.flows[variant]
        patches = self._standardized_flat(support_maps[:2])
        with torch.no_grad():
            z, logdet = flow.forward(patches)
            recon, reverse_logdet = flow.forward(z, reverse=True)
            repeat, _ = flow.forward(patches)
        gradient_ok = _gradient_probe(flow, patches)
        return {
            "inverse_reconstruction_max_error": float((recon - patches).abs().max().detach().cpu()),
            "logdet_cancellation_max_error": float(
                (logdet + reverse_logdet).abs().max().detach().cpu()
            ),
            "finite_gradients": gradient_ok,
            "deterministic_repeat": bool(torch.allclose(z, repeat, atol=0.0, rtol=0.0)),
            "finite_outputs": bool(torch.isfinite(z).all() and torch.isfinite(logdet).all()),
            "raw_euclidean_memory_scoring": True,
        }

    def _initialize_variant_banks(self) -> None:
        base = self._latent_for_maps(self.support_maps, BASELINE_ARM_ID).detach()
        base_loo = _loo_stats_torch(base)
        self.banks = dict.fromkeys(VARIANTS, base)
        self.loo_stats = dict.fromkeys(VARIANTS, base_loo)
        self.v0_defect_gaps = None

    def _refresh_banks(self, variants: Sequence[str]) -> None:
        for variant in variants:
            self.banks[variant] = self._latent_for_maps(self.support_maps, variant).detach()
            self.loo_stats[variant] = _loo_stats_torch(self.banks[variant])

    def _sampled_bank_pair(self, variant: str, max_rows: int = 2048) -> tuple[Any, Any]:
        import torch

        base = self.banks[BASELINE_ARM_ID]
        candidate = self.banks[variant]
        if base.shape[0] <= max_rows:
            return base, candidate
        generator = torch.Generator(device=base.device)
        generator.manual_seed(stable_seed(PROTOCOL_ID, "bank_sample"))
        indices = torch.randperm(base.shape[0], generator=generator, device=base.device)[:max_rows]
        return base[indices], candidate[indices]

    def _wrong_clean_image_target(self, source: FeatureMap) -> Any:
        matching = [
            index
            for index, item in enumerate(self.support_maps)
            if item.relative_path == source.relative_path
        ]
        if not matching:
            raise RunnerError(f"C2 source not found in support maps: {source.relative_path}")
        wrong_index = (matching[0] + 1) % len(self.support_maps)
        wrong = self._standardized_flat([self.support_maps[wrong_index]])
        target, _ = self.flows[BASELINE_ARM_ID].forward(wrong)
        return target

    def _latent_for_maps(self, maps: Sequence[FeatureMap], variant: str) -> Any:
        import torch

        with torch.no_grad():
            z, _ = self.flows[variant].forward(self._standardized_flat(maps))
        return z

    def _standardized_flat(self, maps: Sequence[FeatureMap]) -> Any:
        import torch

        if self.stage_a is None:
            raise RunnerError("Stage-A is not fitted")
        tensor = torch.as_tensor(_stack_maps(maps), dtype=torch.float32, device=self.stage_a.device)
        flat = tensor.reshape(-1, tensor.shape[-1])
        if self.stage_a.standardizer is None:
            return flat
        return self.stage_a.standardizer.transform(flat)


def raw_pair_distance_loss(left: Any, right: Any) -> Any:
    try:
        from flow_tte.paired_loss import raw_pair_distance_loss as imported

        return imported(left, right)
    except ModuleNotFoundError:
        return (left - right).norm(dim=-1).mean() / math.sqrt(float(left.shape[-1]))


def normalized_pair_huber_loss(left: Any, right: Any) -> Any:
    try:
        from flow_tte.paired_loss import normalized_pair_huber_loss as imported

        return imported(left, right)
    except ModuleNotFoundError:
        import torch.nn.functional as functional

        left_norm = functional.normalize(left, dim=-1)
        right_norm = functional.normalize(right, dim=-1)
        return functional.smooth_l1_loss(left_norm, right_norm)


def clean_identity_loss(left: Any, right: Any) -> Any:
    try:
        from flow_tte.paired_loss import clean_identity_loss as imported

        return imported(left, right)
    except ModuleNotFoundError:
        return raw_pair_distance_loss(left, right)


def latent_norm_preservation_loss(left: Any, right: Any) -> Any:
    try:
        from flow_tte.paired_loss import latent_norm_preservation_loss as imported

        return imported(left, right)
    except ModuleNotFoundError:
        return (left.norm(dim=-1) - right.norm(dim=-1)).abs().mean() / math.sqrt(float(left.shape[-1]))


def effective_rank(features: Any) -> float:
    try:
        from flow_tte.paired_loss import effective_rank as imported

        return float(imported(features))
    except ModuleNotFoundError:
        import torch

        centered = features.float() - features.float().mean(dim=0, keepdim=True)
        if min(centered.shape) < 2:
            return 1.0
        singular = torch.linalg.svdvals(centered)
        probs = singular / singular.sum().clamp_min(1e-12)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
        return float(torch.exp(entropy).detach().cpu())


def neighbor_identity_retention(
    reference: Any,
    candidate: Any,
    k: int = 10,
    max_samples: int = 2048,
) -> float:
    try:
        from flow_tte.paired_loss import neighbor_identity_retention as imported

        return float(imported(reference, candidate, k=k, max_samples=max_samples))
    except ModuleNotFoundError:
        import torch

        if reference.shape[0] <= 1:
            return 1.0
        if reference.shape[0] > max_samples:
            indices = torch.linspace(
                0,
                reference.shape[0] - 1,
                steps=max_samples,
                device=reference.device,
            ).round().long().unique()
            reference = reference[indices]
            candidate = candidate[indices]
        k_eff = min(k, reference.shape[0] - 1)
        ref_dist = torch.cdist(reference.float(), reference.float(), p=2.0)
        cand_dist = torch.cdist(candidate.float(), candidate.float(), p=2.0)
        diag = torch.arange(reference.shape[0], device=reference.device)
        ref_dist[diag, diag] = float("inf")
        cand_dist[diag, diag] = float("inf")
        ref_idx = torch.topk(ref_dist, k_eff, largest=False).indices
        cand_idx = torch.topk(cand_dist, k_eff, largest=False).indices
        overlaps = []
        for left, right in zip(ref_idx, cand_idx):
            overlaps.append(len(set(left.detach().cpu().tolist()) & set(right.detach().cpu().tolist())) / k_eff)
        return float(np.mean(overlaps))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def deterministic_pair_order(length: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(item) for item in rng.permutation(length)]


def _validity_passed(validity: Mapping[str, Any]) -> bool:
    return bool(
        float(validity["inverse_reconstruction_max_error"]) <= 1e-5
        and float(validity["logdet_cancellation_max_error"]) <= 2e-4
        and validity["finite_gradients"]
        and validity["deterministic_repeat"]
        and validity["finite_outputs"]
        and validity["raw_euclidean_memory_scoring"]
    )


def _gradient_probe(flow: Any, patches: Any) -> bool:
    import torch

    flow.zero_grad(set_to_none=True)
    probe = patches.detach().clone().requires_grad_(True)
    z, logdet = flow.forward(probe)
    loss = z.square().mean() + logdet.square().mean()
    loss.backward()
    input_gradient = probe.grad
    grads = [
        parameter.grad.detach()
        for parameter in flow.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    trainable_parameters = [parameter for parameter in flow.parameters() if parameter.requires_grad]
    flow.zero_grad(set_to_none=True)
    input_ok = bool(
        input_gradient is not None
        and torch.isfinite(input_gradient).all()
        and float(input_gradient.abs().sum().detach().cpu()) > 0.0
    )
    parameter_ok = bool(
        not trainable_parameters
        or (
            grads
            and all(torch.isfinite(grad).all() for grad in grads)
            and sum(float(grad.abs().sum().cpu()) for grad in grads) > 0.0
        )
    )
    return input_ok and parameter_ok


def _relative_reduction(baseline: float, candidate: float) -> float:
    return float((baseline - candidate) / max(abs(baseline), 1e-12))


def support_self_relative_inflation(
    shifted_distance_mean: float,
    loo_mean: float,
    loo_std: float,
) -> float:
    """LOO-standardized shifted score increase over the clean self-match score."""
    scale = max(float(loo_std), 1e-6)
    shifted_score = (float(shifted_distance_mean) - float(loo_mean)) / scale
    clean_self_score = -float(loo_mean) / scale
    return shifted_score - clean_self_score


def _loo_stats_torch(bank: Any) -> tuple[float, float]:
    nearest = _loo_distances_torch(bank)
    return (
        float(nearest.mean().detach().cpu()),
        float(nearest.std(unbiased=False).clamp_min(1e-6).detach().cpu()),
    )


def _loo_distances_torch(bank: Any, chunk_size: int = 2048) -> Any:
    import torch

    if bank.shape[0] < 2:
        return torch.ones(bank.shape[0], device=bank.device, dtype=bank.dtype)
    parts: list[Any] = []
    for start in range(0, int(bank.shape[0]), chunk_size):
        end = min(start + chunk_size, int(bank.shape[0]))
        distances = torch.cdist(bank[start:end], bank, p=2.0)
        local = torch.arange(end - start, device=bank.device)
        distances[local, local + start] = float("inf")
        parts.append(distances.min(dim=1).values)
    return torch.cat(parts, dim=0)


def _stack_maps(feature_maps: Sequence[FeatureMap]) -> FloatArray:
    if not feature_maps:
        raise RunnerError("expected at least one feature map")
    first = feature_maps[0].values.shape
    if any(item.values.shape != first for item in feature_maps):
        raise RunnerError("all feature maps must share one DVT/flow shape")
    return np.stack([item.values.astype(np.float32, copy=False) for item in feature_maps], axis=0)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("mechanism", "smoke"), required=True)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--support-json", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend-class", default="")
    parser.add_argument("--host-gpu-id", default="")
    parser.add_argument("--container-cuda-slot", default="0")
    args = parser.parse_args(list(argv))
    args.objects = tuple(part for part in args.objects.replace(",", " ").split() if part)
    if not args.objects:
        parser.error("--objects must name at least one object")
    if len(set(args.objects)) != len(args.objects):
        parser.error("--objects must be unique")
    if any(name not in OBJECTS for name in args.objects):
        parser.error(f"--objects must be a subset of {OBJECTS}")
    if args.phase == "mechanism" and (not args.data_root or not args.support_json):
        parser.error("--data-root and --support-json are required for mechanism phase")
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
            lambda_logdet=2e-2,
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
        "baseline_method": BASELINE_METHOD,
        "baseline_arm_id": BASELINE_ARM_ID,
        "device": device,
        "target_dataset": "MVTec AD2 single-image",
        "objects": list(STAGE_MINUS1_OBJECTS),
        "backbone": "dinov3_vith16plus",
        "layers": list(LAYERS),
        "feature_fusion": "layer_norm_mean",
        "native_size": 672,
        "dvt": {"mode": "position_mean", "alpha": 1.0},
        "support_preprocessing": {
            "deterministic_brightness_seed": 0,
            "brightness_range": [0.8, 1.2],
            "factor_formula": "np.random.default_rng(index).uniform(0.8, 1.2)",
        },
        "stage_a_flow": asdict(stage_a.flow),
        "stage_a_score": asdict(stage_a.score),
        "stage_b": {
            "epochs": STAGE_B_EPOCHS,
            "lr": STAGE_B_LR,
            "weight_decay": STAGE_B_WEIGHT_DECAY,
            "gradient_clip": STAGE_B_GRAD_CLIP,
            "variants": list(VARIANTS),
            "losses": {
                "V0_default": "Baseline1 core; no Stage-B update",
                "V1_aug_nll": "clean + augmented NLL/logdet",
                "V2_raw_pair_nll1": "raw pair + clean identity + norm + NLL weight 1.0",
                "V3_raw_pair_nll01": "raw pair + clean identity + norm + NLL weight 0.1",
                "C1_normalized_pair": "normalized Huber pair + identity + NLL 0.1",
                "C2_random_pair": "raw pair to deterministic wrong clean target",
            },
        },
        "diagnostic_sampling": {
            "effective_rank_neighbor_max_bank_rows": 2048,
            "seed": "sha256(protocol_id, bank_sample, variant)",
            "logdet_cancellation_tolerance": 2e-4,
            "inverse_reconstruction_tolerance": 1e-5,
        },
        "score": {
            "latent_metric": "raw_euclidean",
            "nn": "1NN",
            "standardization": "support_LOO",
            "density_weight": 0.0,
        },
        "register_film_transformer": False,
        "query_adaptation": False,
        "test_public_access": False,
        "test_private_access": False,
    }


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
                raise ValueError(f"{object_name}: support path must be inside train/good")
            candidate = (train_root / tail[3]).resolve(strict=True)
            if candidate.parent != train_root or not candidate.is_file():
                raise ValueError(f"{object_name}: support path escapes train/good")
            relative = candidate.relative_to(root).as_posix()
            if any(part in {"test_public", "test_private", "ground_truth"} for part in relative.split("/")):
                raise ValueError("test and ground-truth paths are forbidden")
            entries.append(
                SupportEntry(
                    object_name=object_name,
                    path=candidate,
                    relative_path=relative,
                    path_sha256=sha256_bytes(relative.encode("utf-8")),
                    content_sha256=sha256_file(candidate),
                )
            )
        result[object_name] = tuple(entries)
    return result


def _manifest_tail(object_name: str, raw_path: str) -> tuple[str, str, str, str]:
    parts = PurePosixPath(raw_path.replace("\\", "/")).parts
    if len(parts) < 4:
        raise ValueError(f"{object_name}: malformed support path {raw_path!r}")
    return cast("tuple[str, str, str, str]", tuple(parts[-4:]))


def read_rgb(path: Path) -> npt.NDArray[np.uint8]:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def deterministic_support_brightness(index: int, seed: int = 0) -> float:
    rng = np.random.default_rng(seed + index)
    return float(rng.uniform(0.8, 1.2))


def deterministic_transform_specs(
    object_name: str,
    relative_path: str,
    seed: int,
    role: str,
) -> tuple[PhotometricTransformSpec, ...]:
    specs: list[PhotometricTransformSpec] = []
    for index in range(10):
        active_count = 0 if index < 2 else 1 if index < 7 else 2
        rng = np.random.default_rng(stable_seed(PROTOCOL_ID, seed, role, object_name, relative_path, index))
        axes = tuple(rng.choice(PHOTOMETRIC_AXES, size=active_count, replace=False).tolist())
        values: dict[str, Any] = {
            "name": f"{role}_{index:02d}_{active_count}active",
            "active_axes": axes,
            "seed": stable_seed(PROTOCOL_ID, seed, role, object_name, relative_path, index, "noise"),
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


def extract_feature_map(
    backbone: Any,
    image: npt.NDArray[np.uint8],
    relative_path: str,
    transform: str,
) -> FeatureMap:
    tensor, grid = backbone.prepare_image(image)
    layer_rows = backbone.extract_features(tensor)
    if len(layer_rows) != len(LAYERS):
        raise RunnerError(f"expected {len(LAYERS)} DINOv3 layers, got {len(layer_rows)}")
    merged = merge_layer_features(layer_rows, "layer_norm_mean")
    height, width = int(grid[0]), int(grid[1])
    return FeatureMap(
        values=merged.reshape(height, width, merged.shape[-1]).astype(np.float32, copy=False),
        grid=(height, width),
        image_shape=(int(image.shape[0]), int(image.shape[1])),
        relative_path=relative_path,
        transform=transform,
    )


def fit_dvt(feature_maps: Sequence[FeatureMap]) -> tuple[tuple[FeatureMap, ...], FloatArray | None]:
    denoiser = fit_feature_denoiser(
        "position_mean",
        [item.values for item in feature_maps],
        alpha=1.0,
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


def load_backend(class_path: str = "") -> PairedLossBackend:
    if not class_path:
        return TorchPairedLossBackend()
    module_name, _, class_name = class_path.rpartition(".")
    if not module_name or not class_name:
        raise RunnerError("--backend-class must be fully qualified")
    module = importlib.import_module(module_name)
    backend = getattr(module, class_name)()
    for method in (
        "set_dvt_artifact",
        "transform_feature_maps",
        "fit_stage_a",
        "train_stage_b",
        "mechanism_summary",
        "save_checkpoint",
    ):
        if not callable(getattr(backend, method, None)):
            raise RunnerError(f"{class_path} missing required method {method}")
    return cast(PairedLossBackend, backend)


class ProgressWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, object_name: str, variant: str, epoch: int, row: Mapping[str, Any]) -> None:
        payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_id": PROTOCOL_ID,
            "object": object_name,
            "variant": variant,
            "epoch": epoch,
            "metrics": _jsonable(row),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        print(compact_progress_row(object_name, variant, epoch, row), flush=True)


def compact_progress_row(
    object_name: str,
    variant: str,
    epoch: int,
    row: Mapping[str, Any],
) -> str:
    if row.get("stage") == "epoch":
        fields = (
            object_name,
            variant,
            f"ep={epoch}",
            "stage=epoch",
            f"total={float(row.get('total', 0.0)):.6f}",
            f"nll={float(row.get('nll', 0.0)):.6f}",
            f"raw_pair={float(row.get('raw_pair', 0.0)):.6f}",
            f"identity={float(row.get('identity', 0.0)):.6f}",
            f"held_pair={float(row.get('held_raw_pair_distance', 0.0)):.6f}",
        )
    else:
        fields = (
            object_name,
            variant,
            f"ep={epoch}",
            "stage=final",
            f"shift_reduction={float(row.get('shift_reduction', 0.0)):.4f}",
            f"loo_delta={float(row.get('loo_delta', 0.0)):.4f}",
            f"std_ratio={float(row.get('std_ratio', 0.0)):.4f}",
            f"effective_rank_ratio={float(row.get('effective_rank_ratio', 0.0)):.4f}",
            f"neighbor_retention={float(row.get('neighbor_retention', 0.0)):.4f}",
            f"defect_retention={float(row.get('defect_retention', 0.0)):.4f}",
            f"gate={bool(row.get('gate', False))}",
        )
    return " | ".join(fields)


def make_stage_b_pairs(
    backbone: Any,
    images: Mapping[str, npt.NDArray[np.uint8]],
    source_maps: Mapping[str, FeatureMap],
    entries: Sequence[SupportEntry],
    seed: int,
    role: str,
) -> tuple[StageBPair, ...]:
    pairs: list[StageBPair] = []
    for entry in entries:
        source = source_maps[entry.relative_path]
        for spec in deterministic_transform_specs(entry.object_name, entry.relative_path, seed, role):
            transformed_rgb = apply_photometric_transform(images[entry.relative_path], spec)
            transformed = extract_feature_map(backbone, transformed_rgb, entry.relative_path, spec.name)
            pairs.append(StageBPair(source=source, transformed=transformed, transform=spec))
    return tuple(pairs)


def transform_stage_b_pairs(
    backend: PairedLossBackend,
    pairs: Sequence[StageBPair],
) -> tuple[StageBPair, ...]:
    transformed: list[StageBPair] = []
    for pair in pairs:
        source, shifted = backend.transform_feature_maps((pair.source, pair.transformed))
        transformed.append(StageBPair(source=source, transformed=shifted, transform=pair.transform))
    return tuple(transformed)


def mechanism_phase(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    support_path = Path(args.support_json)
    support_sha = sha256_file(support_path)
    if support_sha != FROZEN_SUPPORT_MANIFEST_SHA256:
        raise RunnerError(f"support manifest SHA mismatch: {support_sha}")
    support = load_support_manifest(support_path, Path(args.data_root))
    config = frozen_config(args.device)
    backbone = _make_backbone(args.device)
    progress = ProgressWriter(output_root / "progress.jsonl")
    summaries: list[dict[str, Any]] = []
    for object_name in args.objects:
        object_dir = output_root / object_name
        object_dir.mkdir(parents=True, exist_ok=True)
        backend = load_backend(args.backend_class)
        entries = support[object_name]
        images = {}
        for index, entry in enumerate(entries):
            image = read_rgb(entry.path)
            brightness = deterministic_support_brightness(index, seed=0)
            images[entry.relative_path] = apply_photometric_transform(
                image,
                PhotometricTransformSpec(
                    name="support_brightness",
                    active_axes=("brightness",),
                    brightness=brightness,
                ),
            )
        support_raw = tuple(
            extract_feature_map(
                backbone,
                images[entry.relative_path],
                entry.relative_path,
                "support_brightness_identity",
            )
            for entry in entries
        )
        source_map_by_relative = {item.relative_path: item for item in support_raw}
        support_maps, dvt_artifact = fit_dvt(support_raw)
        backend.set_dvt_artifact(dvt_artifact, alpha=1.0)
        train_pairs = transform_stage_b_pairs(
            backend,
            make_stage_b_pairs(backbone, images, source_map_by_relative, entries, args.seed, role="train"),
        )
        held_pairs = transform_stage_b_pairs(
            backend,
            make_stage_b_pairs(backbone, images, source_map_by_relative, entries, args.seed + 991, role="held"),
        )
        backend.fit_stage_a(support_maps, config)
        training_log = dict(
            backend.train_stage_b(
                train_pairs,
                held_pairs,
                config,
                progress=progress,
                object_name=object_name,
            )
        )
        mechanism = dict(backend.mechanism_summary(support_maps, held_pairs, config))
        checkpoint = object_dir / "checkpoint.pt"
        backend.save_checkpoint(checkpoint)
        summary = object_summary(
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
        write_json(output_root / "aggregate" / "partial_verdict.json", aggregate_payload(summaries))
    ledger = make_access_ledger(args.objects, {name: support[name] for name in args.objects})
    write_json(output_root / "access_ledger.json", ledger)
    manifest = run_manifest_payload(args, config, support_sha, summaries, ledger)
    write_json(output_root / "run_manifest.json", manifest)
    aggregate = aggregate_payload(summaries)
    write_json(output_root / "aggregate" / "verdict.json", aggregate)
    write_checksums(output_root)
    print(json.dumps({"run_manifest": manifest, "aggregate": aggregate}, indent=2, sort_keys=True))
    return manifest


def object_summary(
    *,
    object_name: str,
    config: Mapping[str, Any],
    support_manifest_sha256: str,
    support_entries: Sequence[SupportEntry],
    checkpoint: Path,
    mechanism: Mapping[str, Any],
    training_log: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "phase": "mechanism",
        "object": object_name,
        "config": _jsonable(config),
        "support_manifest_sha256": support_manifest_sha256,
        "support_count": len(support_entries),
        "support": [asdict(entry) | {"path": str(entry.path)} for entry in support_entries],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
        "training_log": _jsonable(training_log),
        "mechanism": _jsonable(mechanism),
        "access_ledger": make_access_ledger((object_name,), {object_name: support_entries}),
        "test_public_accessed": False,
        "test_private_accessed": False,
        "ground_truth_accessed": False,
        "query_adaptation": False,
    }


def aggregate_payload(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    objects = [str(item["object"]) for item in summaries]
    variants = {variant: [] for variant in VARIANTS}
    for summary in summaries:
        for variant in VARIANTS:
            variants[variant].append(dict(summary["mechanism"]["variants"][variant]))
    baseline_by_object = {
        str(summary["object"]): float(
            summary["mechanism"]["variants"][BASELINE_ARM_ID]["shifted_normal_inflation"]
        )
        for summary in summaries
    }
    variant_metrics: dict[str, Any] = {}
    for variant, rows in variants.items():
        wins = 0
        reductions = []
        defect = []
        gates = []
        object_rows = {}
        for summary, row in zip(summaries, rows):
            object_name = str(summary["object"])
            reduction = float(row["shift_reduction"])
            reductions.append(reduction)
            defect.append(float(row["defect_retention"]))
            gates.append(bool(row["gate_passed"]))
            wins += int(float(row["shifted_normal_inflation"]) < baseline_by_object[object_name])
            object_rows[object_name] = row
        variant_metrics[variant] = {
            "macro_shift_reduction": float(np.mean(reductions)) if reductions else float("nan"),
            "improved_object_count": wins,
            "macro_defect_retention": float(np.mean(defect)) if defect else float("nan"),
            "all_object_gate": bool(gates and all(gates)),
            "objects": object_rows,
        }
    candidate_passes = {
        variant: _candidate_macro_gate(variant_metrics[variant], objects)
        for variant in PROMOTION_CANDIDATES
    }
    c2_pass = _candidate_macro_gate(variant_metrics["C2_random_pair"], objects)
    if c2_pass:
        verdict = "KILL_FOR_CLAIM"
    elif any(candidate_passes.values()):
        verdict = "MECHANISM_PASS"
    else:
        verdict = "KILL_FOR_CLAIM"
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "objects_completed": objects,
        "required_objects": list(STAGE_MINUS1_OBJECTS),
        "variant_metrics": _jsonable(variant_metrics),
        "candidate_passes": candidate_passes,
        "c2_random_pair_pass": c2_pass,
        "verdict": verdict,
        "all8_support_authorized": verdict == "MECHANISM_PASS",
        "public_access_authorized": False,
    }


def _candidate_macro_gate(row: Mapping[str, Any], objects: Sequence[str]) -> bool:
    if len(objects) != 4:
        return False
    object_rows = cast(Mapping[str, Mapping[str, Any]], row["objects"])
    return bool(
        float(row["macro_shift_reduction"]) >= 0.10
        and int(row["improved_object_count"]) >= 3
        and float(row["macro_defect_retention"]) >= 0.97
        and all(abs(float(object_rows[obj]["loo_delta"])) <= 0.05 for obj in objects)
        and all(0.8 <= float(object_rows[obj]["std_ratio"]) <= 1.25 for obj in objects)
        and all(float(object_rows[obj]["effective_rank_ratio"]) >= 0.90 for obj in objects)
        and all(float(object_rows[obj]["neighbor_retention"]) >= 0.90 for obj in objects)
        and all(float(object_rows[obj]["defect_retention"]) >= 0.90 for obj in objects)
        and all(bool(object_rows[obj]["validity_passed"]) for obj in objects)
    )


def make_access_ledger(
    objects: Sequence[str],
    support_entries: Mapping[str, Sequence[SupportEntry]] | None,
) -> dict[str, Any]:
    return {
        "phase": "mechanism",
        "objects": list(objects),
        "support_train_good_paths_loaded": (
            {key: [entry.relative_path for entry in value] for key, value in support_entries.items()}
            if support_entries is not None
            else {}
        ),
        "support_count_per_object": (
            {key: len(value) for key, value in support_entries.items()}
            if support_entries is not None
            else {}
        ),
        "test_public_enumerated": False,
        "test_public_opened": False,
        "test_private_enumerated": False,
        "test_private_opened": False,
        "ground_truth_enumerated": False,
        "ground_truth_opened": False,
        "ground_truth_used_by_training_or_adaptation": False,
        "query_adaptation": False,
        "notes": "support-only Stage -1: no test_public/test_private/ground_truth enumeration",
    }


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
        "summaries": [f"{item['object']}/summary.json" for item in summaries],
        "config": _jsonable(config),
        "access_ledger": ledger,
        "host_gpu_id": str(args.host_gpu_id),
        "container_cuda_slot": str(args.container_cuda_slot),
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_checksums(output_root: Path) -> None:
    rows = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            rows.append(f"{sha256_file(path)}  {path.relative_to(output_root).as_posix()}")
    (output_root / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return value


def smoke_phase(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = frozen_config(args.device)
    ledger = make_access_ledger(args.objects, None)
    summaries = [
        {
            "object": object_name,
            "phase": "smoke",
            "protocol_id": PROTOCOL_ID,
            "mechanism": {
                "variants": {
                    variant: {
                        "shifted_normal_inflation": 0.0,
                        "shift_reduction": 0.0,
                        "loo_delta": 0.0,
                        "std_ratio": 1.0,
                        "effective_rank_ratio": 1.0,
                        "neighbor_retention": 1.0,
                        "defect_retention": 1.0,
                        "gate_passed": False,
                        "validity": {
                            "finite_outputs": True,
                            "deterministic_repeat": True,
                        },
                    }
                    for variant in VARIANTS
                },
                "validity": {},
            },
            "access_ledger": ledger,
        }
        for object_name in args.objects
    ]
    for summary in summaries:
        write_json(output_root / str(summary["object"]) / "summary.json", summary)
    write_json(output_root / "access_ledger.json", ledger)
    write_json(output_root / "aggregate" / "verdict.json", aggregate_payload(summaries))
    manifest = run_manifest_payload(args, config, "0" * 64, summaries, ledger)
    write_json(output_root / "run_manifest.json", manifest)
    write_checksums(output_root)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _make_backbone(device: str) -> Any:
    from dinov3_backbone import DINOv3Backbone

    return DINOv3Backbone(
        model_name="dinov3_vith16plus",
        device=device,
        smaller_edge_size=672,
        feature_layers=LAYERS,
    )


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    if args.phase == "mechanism":
        mechanism_phase(args)
    else:
        smoke_phase(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
