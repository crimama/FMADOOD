# ruff: noqa: COM812, SLF001, TC003

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import run_flowtte_paired_loss_diagnostic as runner


def _maps(n: int = 4, h: int = 2, w: int = 2, c: int = 4) -> tuple[runner.FeatureMap, ...]:
    maps = []
    base = np.linspace(-0.5, 0.5, h * w * c, dtype=np.float32).reshape(h, w, c)
    for index in range(n):
        values = base + np.float32(index * 0.2)
        maps.append(
            runner.FeatureMap(
                values=values,
                grid=(h, w),
                image_shape=(h * 16, w * 16),
                relative_path=f"fruit_jelly/train/good/{index:03d}.png",
                transform="identity",
            )
        )
    return tuple(maps)


def _pairs(maps: tuple[runner.FeatureMap, ...]) -> tuple[runner.StageBPair, ...]:
    pairs = []
    for index, source in enumerate(maps):
        shifted = runner.FeatureMap(
            values=(source.values + np.float32(0.03 + index * 0.005)).astype(np.float32),
            grid=source.grid,
            image_shape=source.image_shape,
            relative_path=source.relative_path,
            transform="brightness",
        )
        pairs.append(
            runner.StageBPair(
                source=source,
                transformed=shifted,
                transform=runner.PhotometricTransformSpec(
                    name=f"shift_{index}",
                    active_axes=("brightness",),
                    brightness=1.1,
                ),
            )
        )
    return tuple(pairs)


def _config() -> dict[str, object]:
    config = runner.frozen_config("cpu")
    config["stage_a_flow"]["n_epochs"] = 1
    config["stage_a_flow"]["batch_size"] = 1024
    config["stage_b"]["epochs"] = 1
    return config


def test_stage_a_clones_all_arms_exactly_and_freezes_v0() -> None:
    backend = runner.TorchPairedLossBackend()
    backend.fit_stage_a(_maps(), _config())

    reference = backend.flows[runner.BASELINE_ARM_ID].state_dict()
    for variant in runner.VARIANTS:
        for key, value in backend.flows[variant].state_dict().items():
            assert torch.equal(value, reference[key])
    assert all(
        not parameter.requires_grad
        for parameter in backend.flows[runner.BASELINE_ARM_ID].parameters()
    )
    validity = backend._validity_summary(runner.BASELINE_ARM_ID, _maps())
    assert validity["finite_gradients"] is True


def test_arms_are_independent_module_copies() -> None:
    backend = runner.TorchPairedLossBackend()
    backend.fit_stage_a(_maps(), _config())

    first_v2 = next(backend.flows["V2_raw_pair_nll1"].parameters()).detach().clone()
    with torch.no_grad():
        next(backend.flows["V1_aug_nll"].parameters()).add_(1.0)

    assert not torch.equal(
        next(backend.flows["V1_aug_nll"].parameters()).detach(),
        first_v2,
    )
    assert torch.equal(next(backend.flows["V2_raw_pair_nll1"].parameters()).detach(), first_v2)


def test_c2_uses_next_clean_image_same_patch_target_not_patch_roll() -> None:
    maps = _maps()
    backend = runner.TorchPairedLossBackend()
    backend.fit_stage_a(maps, _config())

    target = backend._wrong_clean_image_target(maps[0])
    expected, _ = backend.flows[runner.BASELINE_ARM_ID].forward(
        backend._standardized_flat([maps[1]])
    )
    wrong_roll, _ = backend.flows[runner.BASELINE_ARM_ID].forward(
        backend._standardized_flat([maps[0]])
    )
    wrong_roll = wrong_roll.roll(shifts=max(1, wrong_roll.shape[0] // 7), dims=0)

    assert torch.equal(target, expected)
    assert not torch.equal(target, wrong_roll)


def test_frozen_config_names_baseline1_without_breaking_legacy_arm_id() -> None:
    config = runner.frozen_config("cpu")

    assert config["baseline_method"] == "Baseline1"
    assert config["baseline_arm_id"] == "V0_default"
    assert config["stage_b"]["variants"][0] == "V0_default"


def test_support_brightness_matches_frozen_index_seed_formula() -> None:
    expected = [float(np.random.default_rng(index).uniform(0.8, 1.2)) for index in range(4)]

    actual = [runner.deterministic_support_brightness(index) for index in range(4)]

    assert actual == pytest.approx(expected)


def test_neighbor_retention_forwards_shared_sample_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, int] = {}

    def fake_retention(
        reference: torch.Tensor,
        candidate: torch.Tensor,
        *,
        k: int,
        max_samples: int,
    ) -> torch.Tensor:
        del reference, candidate, k
        observed["max_samples"] = max_samples
        return torch.tensor(1.0)

    monkeypatch.setattr("flow_tte.paired_loss.neighbor_identity_retention", fake_retention)
    value = runner.neighbor_identity_retention(torch.eye(4), torch.eye(4), max_samples=2048)

    assert value == 1.0
    assert observed["max_samples"] == 2048


def test_shift_inflation_is_score_increase_over_clean_self_match() -> None:
    inflation = runner.support_self_relative_inflation(
        shifted_distance_mean=2.0,
        loo_mean=10.0,
        loo_std=4.0,
    )

    assert inflation == pytest.approx(0.5)


def test_training_progress_and_summary_schema(tmp_path: Path) -> None:
    maps = _maps()
    pairs = _pairs(maps)
    config = _config()
    backend = runner.TorchPairedLossBackend()
    backend.fit_stage_a(maps, config)
    progress = runner.ProgressWriter(tmp_path / "progress.jsonl")

    log = backend.train_stage_b(
        pairs,
        pairs,
        config,
        progress=progress,
        object_name="fruit_jelly",
    )
    summary = backend.mechanism_summary(maps, pairs, config)
    rows = [
        json.loads(line)
        for line in (tmp_path / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert log["pair_count"] == len(pairs)
    assert rows
    assert {"object", "variant", "epoch", "metrics"} <= set(rows[0])
    for variant in runner.VARIANTS:
        payload = summary["variants"][variant]
        assert {
            "held_raw_pair_distance",
            "shifted_normal_inflation",
            "shift_reduction",
            "loo_delta",
            "std_ratio",
            "effective_rank_ratio",
            "neighbor_retention",
            "defect_retention",
            "gate_passed",
            "validity",
        } <= set(payload)


def test_access_ledger_is_support_only() -> None:
    ledger = runner.make_access_ledger(("fruit_jelly",), None)

    assert ledger["test_public_enumerated"] is False
    assert ledger["test_public_opened"] is False
    assert ledger["test_private_enumerated"] is False
    assert ledger["test_private_opened"] is False
    assert ledger["ground_truth_enumerated"] is False
    assert ledger["ground_truth_opened"] is False
    assert ledger["query_adaptation"] is False
