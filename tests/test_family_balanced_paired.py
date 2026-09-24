"""FBPA controls: immutable episodes, exact off, one averaged optimizer step."""
from collections import Counter
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.nn import functional as F
import yaml

from fmad.methods import register_methods
from fmad.methods.bcma import BoundedMetric
from fmad.methods.family_balanced_paired import FamilyBalancedPaired, episode_schedule
from fmad.methods.tail_contrast import ResidualAdapter
from fmad.methods.transfer_update import TransferUpdate, _FunctionalScorer
from fmad.protocol import MethodContext
from fmad.registry import METHOD_REGISTRY
from tests.test_psma_mr_v2 import TinyPSMAV2Backbone, _write_image
from tests.test_transfer_update import config


def tiny_config(**overrides):
    return {**config(), "steps": 64, "cross_environment_meta": False, **overrides}


@pytest.mark.parametrize("balance,pairing", [(False, False), (True, False), (True, True)])
def test_identical_deck_with_declared_packing(balance, pairing):
    schedule = episode_schedule(balance, pairing)
    assert len(schedule) == 64 and all(len(batch) == 2 for batch in schedule)
    assert Counter(t for batch in schedule for t in batch) == Counter(
        (s, r, f) for s in range(16) for r in range(4) for f in range(2))
    assert all((a[2] != b[2]) == balance for a, b in schedule)
    assert all((a[0] == b[0]) == pairing for a, b in schedule)
    assert all(a[0] // 2 == b[0] // 2 for a, b in schedule)
    if not balance:
        assert [a[2] for a, _ in schedule] == [f for f in [0, 1, 0, 1] for _ in range(16)]


@pytest.mark.parametrize("kwargs", [
    {"episode_enabled": 1}, {"family_balance": "yes"}, {"source_pairing": 1},
    {"family_balance": False, "source_pairing": True}, {"episode_repeats": 2},
    {"episode_repeats": True}, {"episode_seed": -1}, {"episode_seed": True},
    {"reference_policy": "current_batch"}, {"steps": 63},
    {"cross_environment_meta": True}, {"transfer_update": False}, {"meta_outer_weight": .5},
])
def test_invalid_or_confounded_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        FamilyBalancedPaired(TinyPSMAV2Backbone(), tiny_config(**kwargs))


@pytest.mark.parametrize("force_weak", [False, True])
def test_disabled_exactly_reproduces_l3c(tmp_path, monkeypatch, force_weak):
    paths = [_write_image(tmp_path / f"{i}.png", 17 * i) for i in range(4)]
    cfg = tiny_config(steps=2, episode_enabled=False)
    parent = TransferUpdate(TinyPSMAV2Backbone(), cfg)
    disabled = FamilyBalancedPaired(TinyPSMAV2Backbone(), cfg)
    for model in (parent, disabled):
        if force_weak:
            monkeypatch.setattr(model, "_teacher_quantile",
                                lambda image, banks, q, mask=None: .1 if mask is None else .2)
        model.fit(paths, MethodContext("can", 0, 8, False, False))
        if force_weak:
            assert all(r["boundary_strength"] == r["outer_boundary_strength"] == .25
                       for r in model.history)
    assert parent.history == disabled.history
    assert parent.parameter_count == disabled.parameter_count
    for module in ("metric", "adapter"):
        for a, b in zip(getattr(parent, module).parameters(), getattr(disabled, module).parameters()):
            assert torch.equal(a, b)
    np.testing.assert_array_equal(parent.predict(paths[0])[0], disabled.predict(paths[0])[0])


@pytest.fixture(scope="module")
def fitted_variants(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fbpa")
    paths = [_write_image(tmp / f"{i}.png", 13 * i) for i in range(16)]
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    models = []
    try:
        for balance, pairing in [(False, False), (True, False), (True, True)]:
            model = FamilyBalancedPaired(TinyPSMAV2Backbone(),
                                        tiny_config(family_balance=balance, source_pairing=pairing))
            before_backbone = model.backbone.model.weight.detach().clone()
            seen = []
            original_loss = model._episode_loss
            original_step = torch.optim.AdamW.step
            original_clip = torch.nn.utils.clip_grad_norm_
            updates, clips = [], []

            def inspect(scorer, parameters, episode, banks, sampler, gain=None):
                # Two episode losses must see bit-identical parameters per update.
                snapshot = torch.cat([p.detach().flatten().clone() for p in parameters.values()])
                if len(seen) % 2:
                    assert torch.equal(seen[-1], snapshot)
                seen.append(snapshot)
                return original_loss(scorer, parameters, episode, banks, sampler, gain)

            model._episode_loss = inspect

            def step_once(optimizer, *args, **kwargs):
                assert len(seen) == 2 * (len(updates) + 1)
                assert len(clips) == len(updates) + 1
                updates.append(len(seen))
                return original_step(optimizer, *args, **kwargs)

            def clip_once(*args, **kwargs):
                assert len(seen) == 2 * (len(clips) + 1)
                clips.append(len(seen))
                return original_clip(*args, **kwargs)

            with patch.object(torch.optim.AdamW, "step", step_once), patch.object(
                    torch.nn.utils, "clip_grad_norm_", clip_once):
                model.fit(paths, MethodContext("can", 0, 8, False, False))
            model._episode_loss = original_loss
            assert len(updates) == len(clips) == 64
            assert len(seen) == 128 and not torch.equal(seen[0], seen[-1])
            assert torch.equal(model.backbone.model.weight, before_backbone)
            assert all(p.grad is None for p in model.backbone.model.parameters())
            assert not model.records
            prediction = model.predict(paths[0])[0]
            assert np.isfinite(prediction).all() and np.ptp(prediction) > 0
            models.append(model)
        yield models
    finally:
        torch.set_num_threads(original_threads)


def test_full_fit_deck_input_rng_bank_and_output_contract(fitted_variants):
    digests = []
    for model in fitted_variants:
        assert len(model.history) == 64
        assert all(row["episodes_consumed"] == 2 * (step + 1)
                   for step, row in enumerate(model.history))
        assert all(np.isfinite(row["loss"]) and np.isfinite(row["gradient_norm"])
                   for row in model.history)
        meta = model.object_metadata()
        assert meta["episode_count"] == 128 and meta["steps_completed"] == 64
        assert meta["training_bank_exclusion"] == "fixed_pair_excluded"
        assert meta["backbone_frozen"] and meta["cross_environment_meta"] is False
        assert meta["training_view_extract_calls"] == 64 * (6 if model.source_pairing else 8) * 2
        assert meta["feature_extract_calls_fit"] > meta["training_view_extract_calls"]
        assert model._fit_extract_calls is None
        assert meta["peak_cuda_fit_allocated_bytes"] is None
        assert all(row["unique_views"] == (6 if model.source_pairing else 8)
                   for row in model.history)
        for e in meta["episode_manifest"]:
            assert e["excluded_sources"] == [e["source"] // 2 * 2, e["source"] // 2 * 2 + 1]
            twin = next(t for t in meta["episode_manifest"] if t["source"] == e["source"]
                        and t["repeat"] == e["repeat"] and t["family_index"] != e["family_index"])
            assert e["shift_seed"] == twin["shift_seed"]
            assert e["view_sha256"][:2] == twin["view_sha256"][:2]
            assert e["rank_seed"] != twin["rank_seed"]
        # This rejects persistent tensor/map/graph state in scalar metadata/history.
        json.dumps(meta)
        json.dumps(model.history)
        digests.append(meta["episode_manifest_sha256"])
    assert len(set(digests)) == 1
    assert all(m.episode_manifest == fitted_variants[0].episode_manifest for m in fitted_variants)
    assert len({m.parameter_count for m in fitted_variants}) == 1


def test_combined_gradient_equals_one_mean_loss_and_shared_clean_counting():
    torch.manual_seed(8)
    model = FamilyBalancedPaired(TinyPSMAV2Backbone(), tiny_config())
    model.metric = BoundedMetric(8, True, .2, 4).double()
    model.adapter = ResidualAdapter(8, 2).double()
    scorer = _FunctionalScorer(model)
    parameters = dict(scorer.named_parameters())
    banks = {8: (F.normalize(torch.randn(9, 8, dtype=torch.float64), dim=-1), None, [2] * 9)}

    def view():
        return {8: (F.normalize(torch.randn(4, 8, dtype=torch.float64), dim=-1), (2, 2))}

    clean, shifted = view(), view()
    episodes = [dict(cached=[clean, shifted, view(), view()], teacher=torch.full((2, 2), .1),
                     mask=np.array([[True, False], [False, True]]), shape=(2, 2), budget=.0012)
                for _ in range(2)]

    def loss(index):
        return model._episode_loss(scorer, parameters, episodes[index], banks,
                                   torch.Generator().manual_seed(7 + index))[0]

    mean_loss = (loss(0) + loss(1)) / 2
    expected = torch.autograd.grad(mean_loss, tuple(parameters.values()))
    for index in range(2):
        (loss(index) / 2).backward()
    for p, grad in zip(parameters.values(), expected):
        torch.testing.assert_close(p.grad, grad, atol=1e-12, rtol=1e-10)
    clean_score = model.fused(clean, banks, (2, 2))
    teacher = episodes[0]["teacher"]
    from fmad.methods.relational_tail import relational_score_loss
    preserve = (clean_score - teacher).square().mean() + .25 * relational_score_loss(clean_score, teacher)
    logs = [model._episode_loss(scorer, parameters, e, banks,
                                torch.Generator().manual_seed(7 + i))[1]
            for i, e in enumerate(episodes)]
    assert logs[0]["P"] == logs[1]["P"] == pytest.approx(float(preserve.detach()))
    assert logs[0]["shift_loss"] == logs[1]["shift_loss"]


def test_support_contract_rejects_incomplete_or_duplicate_contents(tmp_path):
    model = FamilyBalancedPaired(TinyPSMAV2Backbone(), tiny_config())
    context = MethodContext("can", 0, 8, False, False)
    with pytest.raises(ValueError, match="exactly16"):
        model.fit([], context)
    paths = [_write_image(tmp_path / f"{i}.png", 0) for i in range(16)]
    with pytest.raises(ValueError, match="distinct support contents"):
        model.fit(paths, context)


def test_registry_configs_and_component_only_differences():
    register_methods("family_balanced_paired")
    assert METHOD_REGISTRY.get("family_balanced_paired") is FamilyBalancedPaired
    root = Path(__file__).parents[1] / "configs/methods"
    baseline = yaml.safe_load((root / "transfer_update_L3C.yaml").read_text())
    added = {"episode_enabled", "family_balance", "source_pairing", "episode_repeats",
             "episode_seed", "reference_policy"}
    configs = {}
    for label in ("L3C", "M0", "M1", "M2"):
        candidate = yaml.safe_load((root / f"family_balanced_paired_{label}.yaml").read_text())
        configs[label] = candidate
        assert {k: v for k, v in candidate.items() if k not in {"method", "method_config"}} == {
            k: v for k, v in baseline.items() if k not in {"method", "method_config"}}
        assert {k: v for k, v in candidate["method_config"].items() if k not in added} == baseline["method_config"]
        assert candidate["output"] == {"save_anomaly_maps": False, "save_weights": False}
    for left, right, switch in [("M0", "M1", "family_balance"), ("M1", "M2", "source_pairing")]:
        assert configs[left]["method_config"][switch] is False
        assert configs[right]["method_config"][switch] is True
        a, b = [yaml.safe_load((root / f"family_balanced_paired_{s}.yaml").read_text())
                for s in (left, right)]
        a["method_config"].pop(switch)
        b["method_config"].pop(switch)
        assert a == b
