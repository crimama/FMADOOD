# ruff: noqa: COM812, SLF001, TC003
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from flow_tte.semantic_atlas_metric import SemanticAtlasConfig, fit_semantic_atlas
from scripts import run_flowtte_semantic_atlas_metric as runner


def test_frozen_config_matches_preregistration() -> None:
    config = runner.frozen_config("cpu")
    assert config["baseline_method"] == "Baseline1-core"
    assert config["atlas"] == {
        "n_clusters": 8,
        "beta": 0.25,
        "semantic_tangent_rank": 4,
        "min_cluster_sources": 6,
        "min_cluster_rows": 512,
        "orthogonal_residual_min_ratio": 0.25,
        "neighbor_k": 10,
        "seed": 0,
    }
    assert config["density_weight"] == 0.0
    assert config["test_public_access"] is False


def test_source_folds_are_disjoint_and_cover_sixteen() -> None:
    entries = [type("Entry", (), {"path_sha256": f"{idx:064x}"})() for idx in range(16)]
    folds = runner._folds(entries)
    assert [len(fold) for fold in folds] == [4, 4, 4, 4]
    assert sorted(item for fold in folds for item in fold) == list(range(16))


def test_score_all_identity_and_soft_metric() -> None:
    rng = np.random.default_rng(8)
    clean = rng.normal(size=(80, 6)).astype(np.float32)
    shifted = clean.copy()
    shifted[:, 5] += 0.4
    sources = [f"s{idx % 8}" for idx in range(80)]
    atlas = fit_semantic_atlas(
        clean,
        shifted,
        sources,
        SemanticAtlasConfig(
            n_clusters=1, min_cluster_rows=20, min_cluster_sources=4, semantic_tangent_rank=0
        ),
    )
    scores = runner.score_all(
        shifted, clean, {"A0_baseline1_core": None, "A1_semantic_atlas_orth": atlas}, chunk_size=16
    )
    assert np.isfinite(scores["A0_baseline1_core"]).all()
    assert scores["A1_semantic_atlas_orth"].mean() < scores["A0_baseline1_core"].mean()


def test_smoke_writes_manifest(tmp_path: Path) -> None:
    assert (
        runner.main(
            [
                "--phase",
                "smoke",
                "--output-root",
                str(tmp_path),
                "--objects",
                "fruit_jelly",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    payload = json.loads((tmp_path / "run_manifest.json").read_text())
    assert payload["finite"] is True
    assert payload["public_access_authorized"] is False
