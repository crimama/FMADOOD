from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.aggregate_same16_baselines import METHODS, OBJECTS, aggregate, method_run_dir


def test_aggregate_same16_baselines_validates_and_means(tmp_path: Path) -> None:
    support = {
        object_name: [f"/data/{object_name}/train/good/{index:03d}.png" for index in range(16)]
        for object_name in OBJECTS
    }
    support_path = tmp_path / "support.json"
    support_path.write_text(json.dumps(support), encoding="utf-8")
    support_sha256 = hashlib.sha256(support_path.read_bytes()).hexdigest()

    for method_index, method in enumerate(METHODS):
        for object_name in OBJECTS:
            run_dir = method_run_dir(tmp_path, method, object_name)
            run_dir.mkdir(parents=True)
            metric = 0.6 + 0.1 * method_index
            row = {
                "seg_AUROC": metric,
                "seg_AUROC_0.05": metric,
                "seg_F1_raw": metric,
                "seg_F1": metric,
                "seg_AUPRO_0.05": metric,
                "coverage": {
                    "prediction_map_count": 2,
                    "ground_truth_map_count": 2,
                    "shape_match_count": 2,
                    "nonconstant_prediction_map_count": 2,
                    "anomalous_region_count": 1,
                    "non_finite_prediction_count": 0,
                },
            }
            (run_dir / "metrics.json").write_text(
                json.dumps({object_name: row}),
                encoding="utf-8",
            )
            if method == "basic":
                manifest = {
                    "support_manifest_sha256": support_sha256,
                    "object_diagnostics": [
                        {
                            "selected_support_paths": support[object_name],
                        }
                    ],
                }
            else:
                manifest = {
                    "objects": {
                        object_name: {
                            "reference_manifest_sha256": support_sha256,
                            "selected_reference_paths": support[object_name],
                        },
                    },
                }
            (run_dir / "run_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

    result = aggregate(
        tmp_path,
        support_path,
        expected_support_sha256=support_sha256,
    )

    assert result["comparable"] is True
    assert result["means"]["basic"]["seg_AUROC_0.05"] == 0.6
    assert result["basic_minus_baseline"]["superad"]["seg_F1_raw"] == pytest.approx(-0.1)


def test_aggregate_rejects_same_basename_from_wrong_support_split(tmp_path: Path) -> None:
    support = {
        object_name: [f"/data/{object_name}/train/good/{index:03d}.png" for index in range(16)]
        for object_name in OBJECTS
    }
    support_path = tmp_path / "support.json"
    support_path.write_text(json.dumps(support), encoding="utf-8")
    support_sha256 = hashlib.sha256(support_path.read_bytes()).hexdigest()

    for method in METHODS:
        for object_name in OBJECTS:
            run_dir = method_run_dir(tmp_path, method, object_name)
            run_dir.mkdir(parents=True)
            row = dict.fromkeys(
                (
                    "seg_AUROC",
                    "seg_AUROC_0.05",
                    "seg_F1_raw",
                    "seg_F1",
                    "seg_AUPRO_0.05",
                ),
                0.6,
            )
            row["coverage"] = {
                "prediction_map_count": 2,
                "ground_truth_map_count": 2,
                "shape_match_count": 2,
                "nonconstant_prediction_map_count": 2,
                "anomalous_region_count": 1,
                "non_finite_prediction_count": 0,
            }
            (run_dir / "metrics.json").write_text(
                json.dumps({object_name: row}),
                encoding="utf-8",
            )
            paths = support[object_name]
            if method == "basic" and object_name == "can":
                paths = [paths[0].replace("/train/good/", "/test_public/good/"), *paths[1:]]
            key = "selected_support_paths" if method == "basic" else "selected_reference_paths"
            if method == "basic":
                manifest = {
                    "support_manifest_sha256": support_sha256,
                    "object_diagnostics": [{key: paths}],
                }
            else:
                manifest = {
                    "objects": {
                        object_name: {
                            "reference_manifest_sha256": support_sha256,
                            key: paths,
                        },
                    },
                }
            (run_dir / "run_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

    with pytest.raises(ValueError, match="outside can/train/good"):
        aggregate(
            tmp_path,
            support_path,
            expected_support_sha256=support_sha256,
        )
