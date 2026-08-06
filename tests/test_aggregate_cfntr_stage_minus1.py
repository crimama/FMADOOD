from __future__ import annotations

# Test-fixture construction is intentionally explicit so parity rows remain auditable.
# ruff: noqa: PERF401
import csv
import hashlib
import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from scripts.aggregate_cfntr_stage_minus1 import (
    BASIS_BOOTSTRAPS,
    DEFECTS,
    FROZEN_SUPPORT_MANIFEST_SHA256,
    LAYERS,
    OBJECTS,
    PROTOCOL_ID,
    RANDOM_CAPTURE_FIELDS,
    RANDOM_DRIFT_FIELDS,
    SHIFT_FLOAT_FIELDS,
    AuditError,
    _expected_config,
    aggregate,
    assigned_defect_shift,
    main,
    stable_seed,
    validation_names,
)
from scripts.run_cfntr_stage_minus1 import _frozen_config as runner_frozen_config

if TYPE_CHECKING:
    from pathlib import Path


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _support_rows(object_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    relatives = [f"{object_name}/train/good/{index:03d}_regular.png" for index in range(16)]
    hashes = {relative: _sha_bytes(relative.encode()) for relative in relatives}
    ordered = sorted(relatives, key=lambda relative: (hashes[relative], relative))
    folds = {relative: index % 4 for index, relative in enumerate(ordered)}
    for relative in relatives:
        rows.append(
            {
                "object_name": object_name,
                "path": f"/data/mvtec_ad_2/{relative}",
                "relative_path": relative,
                "path_sha256": hashes[relative],
                "content_sha256": _sha_bytes(f"content:{relative}".encode()),
                "fold": folds[relative],
            },
        )
    return rows


def _write_object(parent: Path, object_name: str) -> Path:
    support = _support_rows(object_name)
    support_path = parent / f"{object_name}_support.json"
    support_path.write_text(json.dumps(support, indent=2) + "\n", encoding="utf-8")
    shift_rows: list[dict[str, object]] = []
    defect_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    basis_rows: list[dict[str, object]] = []
    packed_masks: dict[str, np.ndarray] = {}
    for entry in support:
        relative = str(entry["relative_path"])
        fold = int(entry["fold"])
        for layer in LAYERS:
            for transform in validation_names(object_name, relative):
                row: dict[str, object] = {
                    "object": object_name,
                    "relative_path": relative,
                    "fold": fold,
                    "layer": layer,
                    "transform": transform,
                }
                values = {
                    "physical_energy_capture": 0.80,
                    "clean_pca_energy_capture": 0.30,
                    "random_energy_capture_q95": 0.11425,
                    "raw_p95": 1.0,
                    "physical_p95": 0.25,
                    "clean_pca_p95": 0.80,
                    "random_p95_q05": 0.8715,
                    "physical_drift_reduction": 0.75,
                    "clean_pca_drift_reduction": 0.20,
                    "random_drift_reduction_q95": 0.1285,
                }
                values.update(
                    {
                        field: 0.10 + 0.001 * index
                        for index, field in enumerate(RANDOM_CAPTURE_FIELDS)
                    },
                )
                values.update(
                    {
                        field: 0.10 + 0.002 * index
                        for index, field in enumerate(RANDOM_DRIFT_FIELDS)
                    },
                )
                for field in SHIFT_FLOAT_FIELDS:
                    row[field] = values[field]
                shift_rows.append(row)
            for defect in DEFECTS:
                mask = np.zeros(16, dtype=np.uint8)
                mask[(int(entry["fold"]) + DEFECTS.index(defect)) % mask.size] = 1
                mask_hash = _sha_bytes(mask.tobytes())
                mask_key = f"{str(entry['path_sha256'])[:16]}__{defect}"
                packed_masks[mask_key] = np.packbits(mask)
                for acquisition in ("clean", "shifted"):
                    defect_rows.append(
                        {
                            "object": object_name,
                            "relative_path": relative,
                            "fold": fold,
                            "layer": layer,
                            "defect": defect,
                            "acquisition": acquisition,
                            "assigned_shift": (
                                "none"
                                if acquisition == "clean"
                                else assigned_defect_shift(object_name, relative, defect)
                            ),
                            "defect_seed": stable_seed(object_name, relative, defect, "defect"),
                            "defect_image_sha256": _sha_bytes(
                                f"defect:{relative}:{defect}".encode(),
                            ),
                            "evaluated_image_sha256": _sha_bytes(
                                f"evaluated:{relative}:{defect}:{acquisition}".encode(),
                            ),
                            "mask_key": mask_key,
                            "mask_height": 4,
                            "mask_width": 4,
                            "mask_sha256": mask_hash,
                            "raw_seg_AUROC_0.05": 0.90,
                            "physical_seg_AUROC_0.05": 0.899,
                            "delta_seg_AUROC_0.05": -0.001,
                            "raw_seg_F1": 0.80,
                            "physical_seg_F1": 0.798,
                            "delta_seg_F1": -0.002,
                            "raw_anomaly_energy": 2.0,
                            "physical_anomaly_energy": 1.9,
                            "anomaly_energy_retention": 0.95,
                        },
                    )
    for layer in LAYERS:
        for replicate in range(BASIS_BOOTSTRAPS):
            stability_rows.append(
                {
                    "object": object_name,
                    "layer": layer,
                    "replicate": replicate,
                    "stability": 0.95,
                },
            )
        for fold in range(4):
            basis_rows.append(
                {
                    "object": object_name,
                    "fold": fold,
                    "layer": layer,
                    "physical_sha256": _sha_bytes(
                        f"physical:{object_name}:{fold}:{layer}".encode(),
                    ),
                    "clean_pca_sha256": _sha_bytes(f"pca:{object_name}:{fold}:{layer}".encode()),
                    "random_sha256": _sha_bytes(f"random:{object_name}:{fold}:{layer}".encode()),
                },
            )
    artifact_paths = {
        "shifts": parent / f"{object_name}_shifts.tsv",
        "stability": parent / f"{object_name}_stability.tsv",
        "defects": parent / f"{object_name}_defects.tsv",
        "bases": parent / f"{object_name}_basis_hashes.tsv",
        "support": support_path,
        "masks": parent / f"{object_name}_synthetic_masks.npz",
    }
    _write_tsv(artifact_paths["shifts"], shift_rows)
    _write_tsv(artifact_paths["stability"], stability_rows)
    _write_tsv(artifact_paths["defects"], defect_rows)
    _write_tsv(artifact_paths["bases"], basis_rows)
    np.savez_compressed(artifact_paths["masks"], **packed_masks)
    total_maps = 17_024
    summary = {
        "protocol_id": PROTOCOL_ID,
        "maximum_verdict": "PROMISING_DIAGNOSTIC",
        "object": object_name,
        "population": "train/good exact support-16 only",
        "test_public_accessed": False,
        "test_private_accessed": False,
        "support_count": 16,
        "fold_counts": [4, 4, 4, 4],
        "resolution": 448 if object_name == "sheet_metal" else 672,
        "backbone": "dinov2_vitl14",
        "layers": list(LAYERS),
        "rank": 4,
        "g5_ingredients": {
            "complete": True,
            "finite_value_fraction": 1.0,
            "total_maps": total_maps,
            "finite_maps": total_maps,
            "nonconstant_maps": total_maps,
            "valid_maps": total_maps,
            "expected_total_maps": total_maps,
            "parity_complete": True,
            "shift_row_count": 384,
            "defect_row_count": 512,
            "expected_arms_per_case": 19,
            "observed_arms_per_case": 19,
            "expected_shift_row_count": 384,
            "expected_defect_row_count": 512,
            "expected_shift_map_count": 7296,
            "expected_defect_map_count": 9728,
            "expected_retained_mask_count": 64,
            "retained_mask_count": 64,
        },
        "artifacts": {name: str(path) for name, path in artifact_paths.items()},
        "artifact_sha256": {name: _sha_file(path) for name, path in artifact_paths.items()},
        "host_gpu_id": str(OBJECTS.index(object_name) // 2),
        "container_cuda_slot": "0",
    }
    summary_path = parent / f"{object_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary_path


def _full_run(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    for shard_index in range(4):
        parent = root / "chunks" / f"gpu{shard_index}"
        parent.mkdir(parents=True)
        config_path = parent / "frozen_config.json"
        config_path.write_text(
            json.dumps(_expected_config(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        objects = list(OBJECTS[2 * shard_index : 2 * shard_index + 2])
        summaries = [_write_object(parent, object_name) for object_name in objects]
        manifest = {
            "protocol_id": PROTOCOL_ID,
            "objects": objects,
            "test_public_accessed": False,
            "test_private_accessed": False,
            "runner_sha256": "a" * 64,
            "core_sha256": "b" * 64,
            "config_sha256": _sha_file(config_path),
            "support_manifest_sha256": FROZEN_SUPPORT_MANIFEST_SHA256,
            "summaries": [path.name for path in summaries],
            "summary_sha256": {
                object_name: _sha_file(parent / f"{object_name}_summary.json")
                for object_name in objects
            },
            "host_gpu_id": str(shard_index),
            "container_cuda_slot": "0",
        }
        (parent / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
    return root


def _refresh_summary_hash(root: Path, object_name: str) -> None:
    summary_path = next(root.rglob(f"{object_name}_summary.json"))
    manifest_path = summary_path.parent / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["summary_sha256"][object_name] = _sha_file(summary_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def test_all_eight_aggregation_recomputes_seed_envelope_and_passes(tmp_path: Path) -> None:
    root = _full_run(tmp_path)

    payload = aggregate(root, tmp_path / "aggregate", bootstrap_replicates=31)

    assert payload["verdict"] == "PROMISING_DIAGNOSTIC"
    assert all(payload["gates"][name]["pass"] for name in ("G1", "G2", "G3", "G4", "G5"))
    assert payload["gates"]["G1"]["random_seed_macro_q95"] == pytest.approx(0.11425)
    assert payload["gates"]["G1"]["margin_lcb95"] == pytest.approx(0.50)
    assert payload["gates"]["G3"]["physical_lcb95"] == pytest.approx(0.75)
    assert payload["gates"]["G4"]["minimum_object_delta_seg_F1"] == pytest.approx(-0.002)
    assert (tmp_path / "aggregate" / "summary.json").is_file()
    assert len((tmp_path / "aggregate" / "bootstrap.tsv").read_text().splitlines()) == 32


def test_aggregator_frozen_config_matches_runner_contract() -> None:
    assert _expected_config() == runner_frozen_config()


def test_public_access_flag_invalidates_run_before_statistics(tmp_path: Path) -> None:
    root = _full_run(tmp_path)
    summary_path = next(root.rglob("can_summary.json"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["test_public_accessed"] = True
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _refresh_summary_hash(root, "can")

    with pytest.raises(AuditError, match="public access"):
        aggregate(root, tmp_path / "aggregate", bootstrap_replicates=3)


def test_cli_emits_invalid_verdict_for_g5_hash_failure(tmp_path: Path) -> None:
    root = _full_run(tmp_path)
    shift_path = next(root.rglob("can_shifts.tsv"))
    shift_path.write_text(shift_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    output = tmp_path / "invalid"

    status = main(["--run-root", str(root), "--output-root", str(output)])

    assert status == 2
    assert (output / "verdict.txt").read_text(encoding="utf-8") == "INVALID_RUN\n"
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["gates"]["G5"]["pass"] is False
