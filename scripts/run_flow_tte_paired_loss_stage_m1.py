#!/usr/bin/env python3
"""Run support-only baseline MLP Flow loss diagnostics for MVTec AD2.

This runner deliberately removes register/FiLM/Transformer branches.  Every
variant starts from the same clean-support MLP Flow baseline and changes only
the fine-tuning objective on deterministic photometric support pairs.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_shift_conditioned_patch_flow as shift_runner  # noqa: E402
import torch  # noqa: E402
from flow_tte.config import ExpansionConfig, FlowConfig, FlowTTEConfig, ScoreConfig  # noqa: E402
from flow_tte.losses import patch_nll, tail_aware_nll  # noqa: E402
from flow_tte.paired_loss import (  # noqa: E402
    deterministic_neighbor_retention,
    effective_rank_ratio,
    identity_anchor_loss,
    latent_norm_preservation_loss,
    normalized_huber_pair_loss,
    raw_paired_canonicalization_loss,
)
from flow_tte.trainer import FeatureStandardizer, FlowDensityEstimator  # noqa: E402

OBJECTS = shift_runner.OBJECTS
SMOKE_OBJECTS = ("fruit_jelly", "walnuts", "vial", "fabric")
PROTOCOL_ID = "flowtte-paired-loss-stage-m1-20260804"
FROZEN_SUPPORT_MANIFEST_SHA256 = shift_runner.FROZEN_SUPPORT_MANIFEST_SHA256
SUPPORT_COUNT = shift_runner.SUPPORT_COUNT
LAYERS = shift_runner.LAYERS
FeatureMap = shift_runner.FeatureMap
StageBPair = shift_runner.StageBPair


VARIANTS: dict[str, dict[str, Any]] = {
    "V0_default": {"train": False, "nll": 1.0, "raw_pair": 0.0, "norm_pair": 0.0, "id": 0.0, "norm": 0.0, "random_pair": False},
    "V1_aug_nll": {"train": True, "nll": 1.0, "raw_pair": 0.0, "norm_pair": 0.0, "id": 0.0, "norm": 0.0, "random_pair": False},
    "V2_raw_pair_nll1": {"train": True, "nll": 1.0, "raw_pair": 1.0, "norm_pair": 0.0, "id": 1.0, "norm": 0.1, "random_pair": False},
    "V3_raw_pair_nll01": {"train": True, "nll": 0.1, "raw_pair": 1.0, "norm_pair": 0.0, "id": 1.0, "norm": 0.1, "random_pair": False},
    "C1_normalized_pair_control": {"train": True, "nll": 0.1, "raw_pair": 0.0, "norm_pair": 1.0, "id": 1.0, "norm": 0.1, "random_pair": False},
    "C2_random_pair_control": {"train": True, "nll": 0.1, "raw_pair": 1.0, "norm_pair": 0.0, "id": 1.0, "norm": 0.1, "random_pair": True},
}


class RunnerError(RuntimeError):
    pass


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "mechanism"), required=True)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--support-json", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage-b-epochs", type=int, default=5)
    parser.add_argument("--stage-b-lr", type=float, default=1e-4)
    parser.add_argument("--host-gpu-id", default="")
    parser.add_argument("--container-cuda-slot", default="0")
    args = parser.parse_args(list(argv))
    args.objects = tuple(part for part in args.objects.replace(",", " ").split() if part)
    if not args.objects:
        parser.error("--objects must not be empty")
    if any(item not in OBJECTS for item in args.objects):
        parser.error(f"--objects must be a subset of {OBJECTS}")
    if args.phase != "smoke" and (not args.data_root or not args.support_json):
        parser.error("--data-root and --support-json are required outside smoke")
    return args


def frozen_config(device: str, *, stage_b_epochs: int = 5, stage_b_lr: float = 1e-4) -> dict[str, Any]:
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
        "target_dataset": "MVTec AD2 single-image",
        "data_root": "/home/hunim/Volume/DATA/mvtec_ad_2",
        "objects": list(SMOKE_OBJECTS),
        "support_count": SUPPORT_COUNT,
        "reference_policy": "same frozen SuperAD-16 support manifest",
        "backbone": "dinov3_vith16plus",
        "layers": list(LAYERS),
        "feature_fusion": "layer_norm_mean",
        "dvt": {"mode": "position_mean", "alpha": 1.0},
        "architecture": "patch-local MLP Flow only; no register, no FiLM, no Transformer",
        "stage_a_flow": asdict(stage_a.flow),
        "stage_a_score": asdict(stage_a.score),
        "stage_b": {
            "epochs": int(stage_b_epochs),
            "lr": float(stage_b_lr),
            "weight_decay": 1e-4,
            "gradient_clip": 0.5,
            "pairs": "deterministic photometric support pairs; train/held seeds split",
            "variants": VARIANTS,
        },
        "primary_support_metrics": [
            "held_shift_inflation",
            "shift_reduction_vs_V0",
            "clean_identity_delta_e0_std",
            "latent_std_ratio",
            "effective_rank_ratio",
            "neighbor_retention",
            "synthetic_defect_retention",
        ],
        "score": {"latent_metric": "raw_euclidean", "nn": "1NN", "density_weight": 0.0},
        "query_adaptation": False,
        "test_public_access": False,
        "test_private_access": False,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def smoke_phase(args: argparse.Namespace) -> dict[str, Any]:
    config = frozen_config(args.device, stage_b_epochs=args.stage_b_epochs, stage_b_lr=args.stage_b_lr)
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "phase": "smoke",
        "objects": list(args.objects),
        "config": config,
        "access_ledger": make_access_ledger("smoke", args.objects, None),
    }
    write_json(root / "frozen_config.json", config)
    write_json(root / "run_manifest.json", payload)
    write_json(root / "access_ledger.json", payload["access_ledger"])
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def mechanism_phase(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    support_json = Path(args.support_json)
    support_sha = shift_runner.sha256_file(support_json)
    if support_sha != FROZEN_SUPPORT_MANIFEST_SHA256:
        raise RunnerError(f"support manifest SHA mismatch: {support_sha}")
    support = shift_runner.load_support_manifest(support_json, Path(args.data_root))
    config = frozen_config(args.device, stage_b_epochs=args.stage_b_epochs, stage_b_lr=args.stage_b_lr)
    backbone = shift_runner._make_backbone(args.device)
    summaries: list[dict[str, Any]] = []
    for object_name in args.objects:
        summary = run_object(
            object_name=object_name,
            root=root / object_name,
            entries=support[object_name],
            backbone=backbone,
            config=config,
            seed=args.seed,
            support_sha=support_sha,
        )
        summaries.append(summary)
    ledger = make_access_ledger("mechanism", args.objects, {key: support[key] for key in args.objects})
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "phase": "mechanism",
        "objects": list(args.objects),
        "config": config,
        "support_manifest_sha256": support_sha,
        "summaries": [f"{item['object']}/summary.json" for item in summaries],
        "access_ledger": ledger,
    }
    write_json(root / "access_ledger.json", ledger)
    write_json(root / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def run_object(
    *,
    object_name: str,
    root: Path,
    entries: Sequence[Any],
    backbone: Any,
    config: Mapping[str, Any],
    seed: int,
    support_sha: str,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    images = {entry.relative_path: shift_runner.read_rgb(entry.path) for entry in entries}
    support_raw = tuple(
        shift_runner.extract_feature_map(
            backbone,
            images[entry.relative_path],
            entry.relative_path,
            "identity",
        )
        for entry in entries
    )
    support_maps, dvt_artifact = shift_runner.fit_dvt(support_raw)
    train_pairs = transform_pairs(
        dvt_artifact,
        shift_runner.make_stage_b_pairs(backbone, images, entries, seed, role="train"),
    )
    held_pairs = transform_pairs(
        dvt_artifact,
        shift_runner.make_stage_b_pairs(backbone, images, entries, seed + 991, role="held"),
    )
    stage_a = fit_stage_a(support_maps, config)
    teacher = teacher_cache(stage_a, support_maps)
    variants: dict[str, Any] = {}
    fitted: dict[str, FlowDensityEstimator] = {}
    for name, spec in VARIANTS.items():
        estimator = clone_estimator(stage_a, config)
        training_log = {"stage_b_epoch_losses": [], "trained": False}
        if spec["train"]:
            training_log = train_variant(estimator, support_maps, train_pairs, teacher, spec, config)
        metrics = evaluate_variant(name, estimator, stage_a, support_maps, held_pairs, teacher)
        variants[name] = {"spec": spec, "training_log": training_log, "metrics": metrics}
        fitted[name] = estimator
    variants = add_relative_metrics(variants)
    for name, payload in variants.items():
        row = {
            "object": object_name,
            "variant": name,
            "shift_reduction": payload["metrics"].get("shift_reduction_vs_V0"),
            "loo_delta": payload["metrics"].get("clean_identity_delta_e0_std"),
            "std_ratio": payload["metrics"].get("latent_std_ratio"),
            "rank_ratio": payload["metrics"].get("effective_rank_ratio"),
            "neighbor_retention": payload["metrics"].get("neighbor_retention"),
            "defect_retention": payload["metrics"].get("synthetic_defect_retention"),
            "pass": payload["metrics"].get("object_gate_pass"),
        }
        print("PROGRESS " + json.dumps(row, sort_keys=True), flush=True)
    checkpoint = root / "checkpoint.pt"
    save_checkpoint(checkpoint, fitted, stage_a, support_maps, dvt_artifact, teacher, config)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "phase": "mechanism",
        "object": object_name,
        "config": config,
        "support_manifest_sha256": support_sha,
        "support_count": len(entries),
        "support": [asdict(entry) | {"path": str(entry.path)} for entry in entries],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": shift_runner.sha256_file(checkpoint),
        "stage_a_training": {
            "losses": [float(value) for value in stage_a.stage_a_stats.losses],
            "train_nll_mean": float(stage_a.train_nll_mean),
            "train_nll_std": float(stage_a.train_nll_std),
        },
        "variants": variants,
        "access_ledger": make_access_ledger("mechanism", (object_name,), {object_name: entries}),
    }
    write_json(root / "summary.json", summary)
    return summary


def transform_pairs(
    dvt_artifact: npt.NDArray[np.float32] | None,
    pairs: Sequence[StageBPair],
) -> tuple[StageBPair, ...]:
    if dvt_artifact is None:
        return tuple(pairs)
    backend = shift_runner.TorchShiftConditionedBackend()
    backend.set_dvt_artifact(dvt_artifact, alpha=1.0)
    return shift_runner.transform_stage_b_pairs(backend, pairs)


def fit_stage_a(support_maps: Sequence[FeatureMap], config: Mapping[str, Any]) -> FlowDensityEstimator:
    estimator = FlowDensityEstimator(
        dim=int(support_maps[0].values.shape[-1]),
        config=FlowConfig(**dict(config["stage_a_flow"])),
        device=str(config.get("device", "cpu")),
    )
    stats = estimator.fit(
        shift_runner._stack_maps(support_maps),
        density_quantile=float(ExpansionConfig().density_quantile),
    )
    estimator.stage_a_stats = stats  # type: ignore[attr-defined]
    return estimator


def clone_estimator(
    source: FlowDensityEstimator,
    config: Mapping[str, Any],
) -> FlowDensityEstimator:
    estimator = FlowDensityEstimator(
        dim=int(source.flow.blocks[0].mask.shape[-1]),
        config=FlowConfig(**dict(config["stage_a_flow"])),
        device=str(config.get("device", "cpu")),
    )
    estimator.flow.load_state_dict(copy.deepcopy(source.flow.state_dict()))
    if source.standardizer is not None:
        estimator.standardizer = FeatureStandardizer(
            mean=source.standardizer.mean.detach().clone().to(estimator.device),
            std=source.standardizer.std.detach().clone().to(estimator.device),
        )
    estimator.train_nll_mean = source.train_nll_mean
    estimator.train_nll_std = source.train_nll_std
    estimator.density_threshold = source.density_threshold
    estimator.spatial_shape = source.spatial_shape
    return estimator


def teacher_cache(
    estimator: FlowDensityEstimator,
    support_maps: Sequence[FeatureMap],
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for item in support_maps:
            z, _ = forward_map(estimator, item)
            result[item.relative_path] = z.detach()
    return result


def train_variant(
    estimator: FlowDensityEstimator,
    support_maps: Sequence[FeatureMap],
    pairs: Sequence[StageBPair],
    teacher: Mapping[str, torch.Tensor],
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    del support_maps
    optimizer = torch.optim.AdamW(
        estimator.flow.parameters(),
        lr=float(config["stage_b"]["lr"]),
        weight_decay=float(config["stage_b"]["weight_decay"]),
    )
    losses: list[dict[str, float]] = []
    paths = sorted(teacher)
    target_by_path = dict(teacher)
    if spec.get("random_pair"):
        rolled = paths[1:] + paths[:1]
        target_by_path = {path: teacher[wrong] for path, wrong in zip(paths, rolled)}
    estimator.flow.train()
    flow_cfg = FlowConfig(**dict(config["stage_a_flow"]))
    for _epoch in range(int(config["stage_b"]["epochs"])):
        epoch = {"nll": 0.0, "raw_pair": 0.0, "norm_pair": 0.0, "id": 0.0, "norm": 0.0, "total": 0.0}
        for pair in pairs:
            source_x = standardized_map_tensor(estimator, pair.source)
            shifted_x = standardized_map_tensor(estimator, pair.transformed)
            clean_z, clean_logdet = estimator.flow.forward(source_x)
            shifted_z, shifted_logdet = estimator.flow.forward(shifted_x)
            clean_target = teacher[pair.source.relative_path]
            shifted_target = target_by_path[pair.source.relative_path]
            nll = tail_aware_nll(
                torch.stack([clean_z, shifted_z], dim=0),
                torch.stack([clean_logdet, shifted_logdet], dim=0),
                tail_weight=flow_cfg.tail_weight,
                tail_top_k_ratio=flow_cfg.tail_top_k_ratio,
            )
            raw_pair = 0.5 * (
                raw_paired_canonicalization_loss(clean_z, clean_target)
                + raw_paired_canonicalization_loss(shifted_z, shifted_target)
            )
            norm_pair = 0.5 * (
                normalized_huber_pair_loss(clean_z, clean_target)
                + normalized_huber_pair_loss(shifted_z, shifted_target)
            )
            identity = identity_anchor_loss(clean_z, clean_target)
            norm = latent_norm_preservation_loss(shifted_z, shifted_target)
            logdet_reg = 0.5 * (clean_logdet.square().mean() + shifted_logdet.square().mean())
            total = (
                float(spec["nll"]) * nll
                + float(spec["raw_pair"]) * raw_pair
                + float(spec["norm_pair"]) * norm_pair
                + float(spec["id"]) * identity
                + float(spec["norm"]) * norm
                + flow_cfg.lambda_logdet * logdet_reg
            )
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                estimator.flow.parameters(),
                max_norm=float(config["stage_b"]["gradient_clip"]),
            )
            optimizer.step()
            for key, value in (
                ("nll", nll),
                ("raw_pair", raw_pair),
                ("norm_pair", norm_pair),
                ("id", identity),
                ("norm", norm),
                ("total", total),
            ):
                epoch[key] += float(value.detach().cpu())
        for key in epoch:
            epoch[key] /= float(len(pairs))
        losses.append(epoch)
    return {"stage_b_epoch_losses": losses, "trained": True, "pair_count": len(pairs)}


def evaluate_variant(
    name: str,
    estimator: FlowDensityEstimator,
    baseline: FlowDensityEstimator,
    support_maps: Sequence[FeatureMap],
    held_pairs: Sequence[StageBPair],
    teacher: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    del name
    support_z = latent_for_maps(estimator, support_maps)
    baseline_z = latent_for_maps(baseline, support_maps)
    loo = loo_stats(support_z)
    base_loo = loo_stats(baseline_z)
    held_inflation: list[float] = []
    held_pair_distance: list[float] = []
    with torch.no_grad():
        for pair in held_pairs:
            source_score = score_map(estimator, pair.source, support_z, loo)
            shifted_score = score_map(estimator, pair.transformed, support_z, loo)
            held_inflation.append(float(shifted_score.mean() - source_score.mean()))
            shifted_z, _ = forward_map(estimator, pair.transformed)
            held_pair_distance.append(
                float(raw_paired_canonicalization_loss(shifted_z, teacher[pair.source.relative_path]).cpu())
            )
    baseline_loo_dist = loo_distances(baseline_z)
    denom = max(float(baseline_loo_dist.std(unbiased=False).detach().cpu()), 1e-6)
    support_gap = torch.linalg.vector_norm(support_z - baseline_z.detach(), dim=-1)
    return {
        "held_shift_inflation": float(np.mean(held_inflation)),
        "held_pair_target_distance": float(np.mean(held_pair_distance)),
        "clean_identity_delta_e0_std": float(support_gap.mean().detach().cpu() / denom),
        "latent_std_ratio": float(
            support_z.detach().std(unbiased=False).cpu()
            / baseline_z.detach().std(unbiased=False).clamp_min(1e-6).cpu()
        ),
        "effective_rank_ratio": float(effective_rank_ratio(support_z, baseline_z).detach().cpu()),
        "neighbor_retention": float(
            deterministic_neighbor_retention(support_z, baseline_z, k=10, max_samples=512)
            .detach()
            .cpu()
        ),
        "synthetic_defect_retention": synthetic_defect_retention(
            estimator,
            baseline,
            support_maps,
            support_z,
            baseline_z,
            loo,
            base_loo,
        ),
    }


def add_relative_metrics(variants: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(variants))
    base = float(result["V0_default"]["metrics"]["held_shift_inflation"])
    for name, payload in result.items():
        metrics = payload["metrics"]
        inflation = float(metrics["held_shift_inflation"])
        metrics["shift_reduction_vs_V0"] = (base - inflation) / max(abs(base), 1e-6)
        metrics["improves_shift_vs_V0"] = inflation < base
        metrics["object_gate_pass"] = bool(
            name != "V0_default"
            and metrics["shift_reduction_vs_V0"] >= 0.10
            and metrics["clean_identity_delta_e0_std"] <= 0.05
            and 0.8 <= metrics["latent_std_ratio"] <= 1.25
            and metrics["effective_rank_ratio"] >= 0.8
            and metrics["neighbor_retention"] >= 0.9
            and metrics["synthetic_defect_retention"] >= 0.9
        )
    return result


def standardized_map_tensor(estimator: FlowDensityEstimator, item: FeatureMap) -> torch.Tensor:
    x = torch.as_tensor(
        item.values.reshape(-1, item.values.shape[-1]),
        dtype=torch.float32,
        device=estimator.device,
    )
    if estimator.standardizer is not None:
        x = estimator.standardizer.transform(x)
    return x


def forward_map(estimator: FlowDensityEstimator, item: FeatureMap) -> tuple[torch.Tensor, torch.Tensor]:
    return estimator.flow.forward(standardized_map_tensor(estimator, item))


def latent_for_maps(
    estimator: FlowDensityEstimator,
    maps: Sequence[FeatureMap],
) -> torch.Tensor:
    parts = []
    with torch.no_grad():
        for item in maps:
            z, _ = forward_map(estimator, item)
            parts.append(z.detach())
    return torch.cat(parts, dim=0)


def loo_distances(bank: torch.Tensor, *, chunk_size: int = 2048) -> torch.Tensor:
    parts = []
    for start in range(0, int(bank.shape[0]), chunk_size):
        end = min(start + chunk_size, int(bank.shape[0]))
        distances = torch.cdist(bank[start:end], bank, p=2.0)
        local = torch.arange(end - start, device=bank.device)
        distances[local, local + start] = float("inf")
        parts.append(distances.min(dim=1).values)
    return torch.cat(parts, dim=0)


def loo_stats(bank: torch.Tensor) -> tuple[float, float]:
    distances = loo_distances(bank)
    return (
        float(distances.mean().detach().cpu()),
        float(distances.std(unbiased=False).clamp_min(1e-6).detach().cpu()),
    )


def score_map(
    estimator: FlowDensityEstimator,
    item: FeatureMap,
    bank: torch.Tensor,
    stats: tuple[float, float],
) -> torch.Tensor:
    z, _ = forward_map(estimator, item)
    distances = torch.cdist(z, bank, p=2.0).min(dim=1).values
    mean, std = stats
    return (distances - mean) / max(std, 1e-6)


def synthetic_defect_retention(
    estimator: FlowDensityEstimator,
    baseline: FlowDensityEstimator,
    support_maps: Sequence[FeatureMap],
    support_bank: torch.Tensor,
    baseline_bank: torch.Tensor,
    stats: tuple[float, float],
    baseline_stats: tuple[float, float],
) -> float:
    retentions: list[float] = []
    for item in support_maps:
        defect = item.values.copy()
        h, w = defect.shape[:2]
        ys = slice(h // 4, max(h // 4 + 1, h // 2))
        xs = slice(w // 4, max(w // 4 + 1, w // 2))
        defect[ys, xs] += np.float32(0.25)
        defect_map = FeatureMap(defect, item.grid, item.image_shape, item.relative_path, "synthetic_defect")
        base_source = score_map(baseline, item, baseline_bank, baseline_stats).reshape(item.grid)
        base_defect = score_map(baseline, defect_map, baseline_bank, baseline_stats).reshape(item.grid)
        cand_source = score_map(estimator, item, support_bank, stats).reshape(item.grid)
        cand_defect = score_map(estimator, defect_map, support_bank, stats).reshape(item.grid)
        base_gap = float(base_defect[ys, xs].mean() - base_source.mean())
        cand_gap = float(cand_defect[ys, xs].mean() - cand_source.mean())
        retentions.append(cand_gap / max(base_gap, 1e-6))
    return float(np.mean(retentions))


def save_checkpoint(
    path: Path,
    fitted: Mapping[str, FlowDensityEstimator],
    stage_a: FlowDensityEstimator,
    support_maps: Sequence[FeatureMap],
    dvt_artifact: npt.NDArray[np.float32] | None,
    teacher: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol_id": PROTOCOL_ID,
            "config": dict(config),
            "variant_flow_states": {
                key: estimator.flow.state_dict() for key, estimator in fitted.items()
            },
            "stage_a_flow_state": stage_a.flow.state_dict(),
            "standardizer": None
            if stage_a.standardizer is None
            else {
                "mean": stage_a.standardizer.mean.detach().cpu(),
                "std": stage_a.standardizer.std.detach().cpu(),
            },
            "support_maps": [asdict(item) | {"values": item.values} for item in support_maps],
            "dvt_artifact": dvt_artifact,
            "teacher": {key: value.detach().cpu() for key, value in teacher.items()},
        },
        path,
    )


def make_access_ledger(
    phase: str,
    objects: Sequence[str],
    support_entries: Mapping[str, Sequence[Any]] | None,
) -> dict[str, Any]:
    return {
        "phase": phase,
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
        "ground_truth_used_by_training_or_adaptation": False,
        "query_adaptation": False,
        "notes": "support-only loss diagnostic; public/GT access forbidden",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.phase == "smoke":
        smoke_phase(args)
    else:
        mechanism_phase(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
