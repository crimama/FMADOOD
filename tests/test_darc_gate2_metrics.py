from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import numpy as np
import pytest

from flow_tte.darc_gate2_metrics import (
    AD1_OBJECTS,
    Gate2Config,
    Gate2FailureCode,
    GroupResidual,
    InvalidGateInput,
    SourceAudit,
    SourceMetric,
    decide_gate2,
    higher_p999,
    paired_bootstrap_lower,
)


class _PopulationConfig(NamedTuple):
    d_ap: float = 0.1
    d_component_recall: float = 0.1
    l1_responses: Tuple[float, float] = (1.0, 1.0)
    r1_responses: Tuple[float, float] = (0.9, 0.9)
    l0_p999: float = 1.0
    l1_p999: float = 0.8


def _population(
    config: Optional[_PopulationConfig] = None,
) -> Tuple[Tuple[SourceMetric, ...], Tuple[GroupResidual, ...]]:
    active = config if config is not None else _PopulationConfig()
    sources = []
    groups = []
    for object_name in AD1_OBJECTS:
        for seed in (0, 1, 2):
            source_ids = tuple(f"{object_name}/train/good/{index:03}.png" for index in range(16))
            folds = tuple(index // 4 for index in range(16))
            population_sha256 = f"tokens-{object_name}-{seed}"
            groups.append(
                GroupResidual(
                    object_name=object_name,
                    seed=seed,
                    source_ids=source_ids,
                    fold_indices=folds,
                    l0_p999=active.l0_p999,
                    l1_p999=active.l1_p999,
                    l0_population_sha256=population_sha256,
                    l1_population_sha256=population_sha256,
                    l0_residual_sha256=f"l0-residual-{object_name}-{seed}",
                    l1_residual_sha256=f"l1-residual-{object_name}-{seed}",
                ),
            )
            for source_id, fold_index in zip(source_ids, folds):
                audit = SourceAudit(
                    population_sha256=f"pixels-{source_id}",
                    support_sha256=f"supports-{source_id}",
                    fallback_sha256=f"fallback-{source_id}",
                    mask_sha256=f"masks-{source_id}",
                )
                sources.append(
                    SourceMetric(
                        object_name=object_name,
                        seed=seed,
                        fold_index=fold_index,
                        source_id=source_id,
                        d_ap=active.d_ap,
                        d_component_recall=active.d_component_recall,
                        l1_responses=active.l1_responses,
                        r1_responses=active.r1_responses,
                        broad_pauroc_delta=0.01,
                        l0_audit=audit,
                        l1_audit=audit,
                        r1_audit=audit,
                    ),
                )
    return tuple(sources), tuple(groups)


def test_gate2_passes_exact_residual_and_retention_boundaries() -> None:
    # Given
    sources, groups = _population()

    # When
    decision = decide_gate2(sources, groups, Gate2Config(bootstrap_replicates=64))

    # Then
    assert decision.status == "PASS"
    assert decision.passed
    assert decision.residual_pass
    assert decision.retention_pass
    assert decision.residual_ratio == pytest.approx(0.8)
    assert decision.retention_ratio == pytest.approx(0.9)
    assert decision.failure_codes == ()


@pytest.mark.parametrize("field", ["d_ap", "d_component_recall"])
def test_zero_bootstrap_lower_bound_fails_signal(field: str) -> None:
    # Given
    sources, groups = _population()
    changed = tuple(row._replace(**{field: 0.0}) for row in sources)

    # When
    decision = decide_gate2(changed, groups, Gate2Config(bootstrap_replicates=64))

    # Then
    assert decision.status == "FAIL"
    assert not decision.passed
    expected = (
        Gate2FailureCode.SIGNAL_AP if field == "d_ap" else Gate2FailureCode.SIGNAL_COMPONENT_RECALL
    )
    assert expected in decision.failure_codes


def test_negative_profile_responses_are_retained_in_registered_aggregation() -> None:
    # Given
    sources, groups = _population(
        _PopulationConfig(l1_responses=(-10.0, 12.0), r1_responses=(-10.0, 11.8)),
    )

    # When
    decision = decide_gate2(sources, groups, Gate2Config(bootstrap_replicates=64))

    # Then
    assert decision.s_l1 == pytest.approx(1.0)
    assert decision.s_r1 == pytest.approx(0.9)
    assert decision.retention_pass


def test_nonpositive_ratio_denominators_are_json_safe_and_fail_scientifically() -> None:
    # Given
    sources, groups = _population(
        _PopulationConfig(l1_responses=(0.0, 0.0), l0_p999=0.0, l1_p999=0.0),
    )

    # When
    decision = decide_gate2(sources, groups, Gate2Config(bootstrap_replicates=8))

    # Then
    assert decision.residual_ratio is None
    assert decision.retention_ratio is None
    assert decision.to_manifest()["residual_ratio"] is None
    assert not decision.passed


def test_group_profile_source_macro_aggregation_is_unrounded_float64() -> None:
    # Given
    sources, groups = _population()
    changed = list(sources)
    changed[0] = changed[0]._replace(
        l1_responses=(-1.0, 5.0),
        r1_responses=(-3.0, 7.0),
        broad_pauroc_delta=-0.5,
    )

    # When
    decision = decide_gate2(changed, groups, Gate2Config(bootstrap_replicates=64))

    # Then
    expected_response = (719.0 + 2.0) / 720.0
    assert decision.s_l1 == pytest.approx(expected_response)
    assert decision.s_r1 == pytest.approx((719.0 * 0.9 + 2.0) / 720.0)
    assert decision.broad_pauroc_delta == pytest.approx((719.0 * 0.01 - 0.5) / 720.0)


@pytest.mark.parametrize("case", ["719", "duplicate", "44_groups", "nonfinite", "audit"])
def test_incomplete_duplicate_nonfinite_or_unpaired_input_is_invalid(case: str) -> None:
    # Given
    sources, groups = _population()
    changed_sources = list(sources)
    changed_groups = list(groups)
    if case == "719":
        changed_sources.pop()
    elif case == "duplicate":
        changed_sources[-1] = changed_sources[0]
    elif case == "44_groups":
        changed_groups.pop()
    elif case == "nonfinite":
        changed_sources[0] = changed_sources[0]._replace(d_ap=float("nan"))
    else:
        audit = changed_sources[0].l1_audit
        changed_audit = SourceAudit(
            audit.population_sha256,
            "different",
            audit.fallback_sha256,
            audit.mask_sha256,
        )
        changed_sources[0] = changed_sources[0]._replace(
            l1_audit=changed_audit,
        )

    # When / Then
    with pytest.raises(InvalidGateInput, match="INVALID_GATE_INPUT"):
        decide_gate2(changed_sources, changed_groups, Gate2Config(bootstrap_replicates=8))


@pytest.mark.parametrize("case", ["fold", "membership", "object"])
def test_wrong_fold_source_membership_or_object_set_is_invalid(case: str) -> None:
    # Given
    sources, groups = _population()
    changed_sources = list(sources)
    if case == "fold":
        changed_sources[0] = changed_sources[0]._replace(fold_index=1)
    elif case == "membership":
        changed_sources[0] = changed_sources[0]._replace(
            source_id="unregistered.png",
        )
    else:
        changed_sources[0] = changed_sources[0]._replace(object_name="unknown")

    # When / Then
    with pytest.raises(InvalidGateInput):
        decide_gate2(changed_sources, groups, Gate2Config(bootstrap_replicates=8))


def test_shared_stratified_bootstrap_is_deterministic_and_linear() -> None:
    # Given
    config = Gate2Config(bootstrap_replicates=257, bootstrap_seed=20260710)
    d_ap = np.arange(720, dtype=np.float64).reshape(45, 16) / 1000.0
    d_component = 1.0 - d_ap
    generator = np.random.Generator(np.random.PCG64(config.bootstrap_seed))
    indices = generator.integers(0, 16, size=(config.bootstrap_replicates, 45, 16))
    groups = np.arange(45)[None, :, None]
    expected_ap = np.quantile(d_ap[groups, indices].mean(axis=(1, 2)), 0.025)
    expected_component = np.quantile(d_component[groups, indices].mean(axis=(1, 2)), 0.025)

    # When
    first = paired_bootstrap_lower(d_ap, d_component, config)
    second = paired_bootstrap_lower(d_ap, d_component, config)

    # Then
    assert first == second
    assert first.d_ap == expected_ap
    assert first.d_component_recall == expected_component


def test_raw_residual_p999_uses_higher_empirical_quantile() -> None:
    # Given
    residuals = np.arange(1001, dtype=np.float64)

    # When
    result = higher_p999(residuals)

    # Then
    assert result == 999.0


def test_source_and_group_manifests_expose_frozen_audit_schema() -> None:
    # Given
    sources, groups = _population()

    # When
    source_manifest = sources[0].to_manifest()
    group_manifest = groups[0].to_manifest()

    # Then
    assert set(source_manifest) == {
        "object",
        "seed",
        "fold",
        "source",
        "d_ap",
        "d_component_recall",
        "l1_responses",
        "r1_responses",
        "broad_pauroc_delta",
        "l0_audit",
        "l1_audit",
        "r1_audit",
    }
    assert group_manifest["quantile_method"] == "higher"
    assert group_manifest["l0_residual_sha256"]
    assert group_manifest["l1_residual_sha256"]
