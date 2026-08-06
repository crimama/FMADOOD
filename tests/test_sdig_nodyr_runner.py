from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile as tiff

from scripts.run_sdig_nodyr_ad2 import (
    OBJECTS,
    condition_from_stem,
    load_support_manifest,
    materialize_variant_maps,
    parse_args,
    structured_support_permutation,
)


def test_condition_from_stem_preserves_multitoken_shift_name() -> None:
    assert condition_from_stem("000_regular") == "regular"
    assert condition_from_stem("014_shift_3") == "shift_3"
    assert condition_from_stem("unexpected") == "unknown"


def test_parse_args_freezes_layers_and_source_orbit() -> None:
    args = parse_args([
        "--data-root", "/data",
        "--output-root", "/out",
        "--support-json", "/support.json",
        "--objects", "can",
        "--host-gpu-id", "0",
        "--source-ids", "000,004",
    ])
    assert args.layers == (15, 23)
    assert args.source_ids == ("000", "004")


def test_structured_shuffle_changes_image_and_spatial_pairing_deterministically() -> None:
    first = structured_support_permutation([(2, 3)] * 4, "can", "cpu")
    second = structured_support_permutation([(2, 3)] * 4, "can", "cpu")

    assert first.shape == (24,)
    assert first.tolist() == second.tolist()
    assert sorted(first.tolist()) == list(range(24))
    assert all(source // 6 != target // 6 for source, target in enumerate(first.tolist()))


def test_structured_shuffle_rejects_mixed_grids() -> None:
    with pytest.raises(ValueError, match="equal support grids"):
        structured_support_permutation([(2, 3), (3, 2)], "can", "cpu")


def test_load_support_manifest_rebases_and_requires_exactly_sixteen(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    payload: dict[str, list[str]] = {}
    for object_name in OBJECTS:
        support_dir = data_root / object_name / "train" / "good"
        support_dir.mkdir(parents=True)
        payload[object_name] = []
        for index in range(16):
            path = support_dir / f"{index:03d}_regular.png"
            path.touch()
            payload[object_name].append(f"/old/root/{object_name}/train/good/{path.name}")
    manifest = tmp_path / "support.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    actual = load_support_manifest(manifest, data_root)

    assert all(len(paths) == 16 for paths in actual.values())
    assert actual["can"][0] == data_root / "can" / "train" / "good" / "000_regular.png"

    payload["can"] = payload["can"][:-1]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 16"):
        load_support_manifest(manifest, data_root)


def test_materialize_variant_maps_round_trips_compact_patch_scores(tmp_path: Path) -> None:
    cache = tmp_path / "cache.npz"
    patch_scores = np.arange(6, dtype=np.float32)
    np.savez_compressed(cache, nodyr=patch_scores)
    records = [{
        "score_cache": str(cache),
        "grid_height": 2,
        "grid_width": 3,
        "image_height": 4,
        "image_width": 6,
        "split": "good",
        "stem": "000_regular",
    }]

    materialize_variant_maps(tmp_path, "can", records, "nodyr")

    path = Path(records[0]["map_nodyr"])
    actual = tiff.imread(path)
    assert actual.shape == (4, 6)
    assert actual.dtype == np.float32
    assert np.all(np.isfinite(actual))
    assert records[0]["map_sha256_nodyr"]
