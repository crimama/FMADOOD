from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

from flow_tte.darc_gate1_artifacts import write_completion, write_json, write_jsonl
from flow_tte.darc_gate1_runtime import (
    Gate1RuntimeConfig,
    cue_record,
    prepare_gate1_run,
)
from flow_tte.darc_synthetic import THIN_W1_L32, insert_line_cue


def _config(tmp_path: Path) -> Gate1RuntimeConfig:
    data_root = tmp_path / "data"
    normal_root = data_root / "bottle" / "train" / "good"
    normal_root.mkdir(parents=True)
    for index in range(16):
        (normal_root / f"{index:03d}.png").write_bytes(f"normal-{index}".encode())
    return Gate1RuntimeConfig(
        data_root=data_root,
        output_root=tmp_path / "output",
        object_name="bottle",
        device="cpu",
        seeds=(0,),
        code_config_sha256="code",
    )


def _write_seed_artifacts(config: Gate1RuntimeConfig) -> None:
    prepared = prepare_gate1_run(config)
    seed = prepared.pending[0]
    root = seed.expectation.seed_root
    write_json(root / "source_metrics.json", {"rows": []})
    write_json(root / "metrics.json", {"source_count": 16})
    write_json(root / "bootstrap_inputs.json", {"rows": []})
    write_jsonl(root / "samples.jsonl", [])
    write_completion(seed.expectation)


def test_prepare_resume_accepts_only_matching_selection_and_provenance(tmp_path: Path) -> None:
    # Given
    config = _config(tmp_path)
    _write_seed_artifacts(config)
    selection = config.output_root / "selections" / "bottle" / "seed=0.json"
    inode = selection.stat().st_ino

    # When
    prepared = prepare_gate1_run(config)

    # Then: an exact completion is reused without rewriting its selection manifest.
    assert prepared.pending == ()
    assert selection.stat().st_ino == inode


def test_prepare_resume_rejects_stale_selection_completion(tmp_path: Path) -> None:
    # Given
    config = _config(tmp_path)
    _write_seed_artifacts(config)
    selection = config.output_root / "selections" / "bottle" / "seed=0.json"
    write_json(selection, {"split_inventory_sha256": "stale"})

    # When
    prepared = prepare_gate1_run(config)

    # Then
    assert len(prepared.pending) == 1
    assert prepared.pending[0].expectation.provenance.selection_sha256 != "stale"


def test_cue_record_persists_reconstructable_raster_provenance() -> None:
    # Given
    image = np.full((128, 128, 3), 255, dtype=np.uint8)
    cue = insert_line_cue(image, THIN_W1_L32, seed=17)

    # When
    record = cue_record(cue, ("support-a", "support-b"))

    # Then
    assert record["metadata"]["seed"] == 17
    assert record["metadata"]["start_xy"] == list(cue.metadata.start_xy)
    assert record["metadata"]["end_xy"] == list(cue.metadata.end_xy)
    assert record["metadata"]["color_rgb"] == list(cue.metadata.color_rgb)
    assert len(record["mask_sha256"]) == 64
    assert record["mask_pixel_count"] == int(np.count_nonzero(cue.mask))
