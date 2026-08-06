#!/usr/bin/env python3
"""Dependency-free executable smoke checks for the WTCF core."""

from __future__ import annotations

import numpy as np
import torch

from flow_tte.config import FlowConfig
from flow_tte.trainer import FlowDensityEstimator
from flow_tte.wtcf import (
    WTCFConfig,
    discover_normal_witnesses,
    make_image_disjoint_folds,
    refine_wtcf,
    soft_nearest_distance,
    upper_cvar,
)


def main() -> int:
    folds = make_image_disjoint_folds(np.arange(4), fold_count=4, seed=3)
    if set(folds.values()) != {0, 1, 2, 3}:
        raise AssertionError("image-disjoint folds are incomplete")
    query = torch.tensor([[0.0, 0.0], [4.0, 0.0]])
    bank = torch.tensor([[1.0, 0.0], [8.0, 0.0]])
    soft = soft_nearest_distance(query, bank, tau=1e-4)
    hard = torch.cdist(query, bank).min(dim=1).values
    torch.testing.assert_close(soft, hard, atol=1e-5, rtol=0.0)
    if float(upper_cvar(torch.tensor([0.1, 0.2, 5.0, 6.0]), 0.5)) != 5.5:
        raise AssertionError("upper CVaR smoke mismatch")

    repeated = {
        source: np.asarray([[float(source), 0.0], [float(source), 100.0]], dtype=np.float32)
        for source in range(4)
    }
    witness = discover_normal_witnesses(
        repeated,
        {source: source for source in repeated},
        config=WTCFConfig(
            steps=1,
            witness_rows_per_source=2,
            witness_candidates_per_pair=2,
            witness_cosine=0.99,
            witness_min_source_pairs=3,
            witness_min_fold_pairs=3,
            max_witnesses=2,
            unsupported_probes=1,
            audit_rows=8,
        ),
    )
    if witness.accepted_count != 1:
        raise AssertionError("witness discovery smoke mismatch")

    rng = np.random.default_rng(17)
    clean = {
        source: rng.normal(loc=source * 0.05, scale=0.5, size=(8, 4)).astype(np.float32)
        for source in range(4)
    }
    features = np.concatenate(list(clean.values()))
    estimator = FlowDensityEstimator(
        dim=4,
        config=FlowConfig(
            n_coupling_layers=2,
            hidden_multiplier=1,
            n_epochs=1,
            lr=1e-3,
            batch_size=16,
            seed=7,
            standardize=True,
        ),
        device="cpu",
    )
    estimator.fit(features, density_quantile=0.9)
    result = refine_wtcf(
        estimator,
        clean,
        mode="wtcf",
        config=WTCFConfig(
            steps=2,
            query_rows=4,
            memory_rows=8,
            witness_rows_per_source=4,
            witness_candidates_per_pair=2,
            witness_min_source_pairs=2,
            witness_min_fold_pairs=2,
            max_witnesses=2,
            unsupported_probes=2,
            witness_target=1.10,
            rollback_interval=1,
            audit_rows=8,
            pair_spearman_floor=0.0,
            neighbor_retention_floor=0.0,
            geometry_ratio_floor=0.1,
            geometry_ratio_ceiling=10.0,
            logdet_drift_std_ceiling=100.0,
        ),
        seed=5,
    )
    if not result.finite:
        raise AssertionError("WTCF smoke produced a non-finite result")
    if not result.final_geometry.passed:
        raise AssertionError("WTCF smoke failed geometry")
    if not result.final_directional.passed:
        raise AssertionError("WTCF smoke failed directional audit")
    print("WTCF_CORE_SMOKE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
