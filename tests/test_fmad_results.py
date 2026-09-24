import csv
import json
from pathlib import Path

import pytest

from fmad.methods.base import ObjectResult
from fmad.results import ResultsManager


def _result(name="widget"):
    return ObjectResult(
        name,
        {},
        {"q": 0.123456789},
        0.25,
        {"q": 0.125},
        {
            "support_identity_sha256": "abc",
            "prediction_diagnostics": {"q": 1},
            "inference_time_insight": "retained",
        },
    )


def test_runs_are_unique_and_do_not_overwrite(tmp_path):
    config = {"method": "m", "seed": 3, "nested": {"b": 2, "a": 1}}
    first = ResultsManager(str(tmp_path), "m", "d", "b", 3, config=config)
    second = ResultsManager(str(tmp_path), "m", "d", "b", 3, config=config)
    assert first.run_dir != second.run_dir
    assert first.run_dir.startswith(str(tmp_path / "m" / "d" / "b" / "seed=3"))
    assert first.config_hash == second.config_hash


def test_manifest_has_code_and_environment_provenance(tmp_path):
    manager = ResultsManager(str(tmp_path), "m", "d", "b", 0, config={})
    manager.save_run_manifest([_result()])
    manifest = json.loads((Path(manager.run_dir) / "run_manifest.json").read_text())
    assert manifest["code_sha256"]
    assert manifest["environment"]["packages"]["numpy"]
    assert manifest["environment"]["packages"]["Pillow"]
    assert "faiss-gpu-cu11" in manifest["environment"]["packages"]


def test_summary_preserves_precision_skips_aggregates_and_defaults_missing(tmp_path):
    manager = ResultsManager(
        str(tmp_path), "m", "d", "b", 0, config={"evaluation": {"metric_contract": "c"}}
    )
    manager.append_summary(
        {"widget": {"seg_AUROC_0.05": 0.12345678901234567, "seg_F1_oracle": 0.2}, "mean": 0.99},
        [_result()],
    )
    with open(manager.summary_path, newline="") as stream:
        row = next(csv.DictReader(stream, delimiter="\t"))
    assert row["seg_AUROC_0.05"] == repr(0.12345678901234567)
    assert row["seg_AUPRO_0.05"] == ""
    assert row["metric_contract"] == "c"


def test_debug_outputs_and_manifest_filtering(tmp_path):
    result = _result()
    manager = ResultsManager(str(tmp_path), "m", "d", "b", 0, config={})
    manager.save_object_times([result])
    manager.save_run_manifest([result])
    run_dir = Path(manager.run_dir)
    assert not (run_dir / "time_measurements.csv").exists()
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert "support_identity_sha256" in manifest["objects"]["widget"]
    assert "prediction_diagnostics" not in manifest["objects"]["widget"]
    assert "inference_time_insight" in manifest["objects"]["widget"]

    debug = ResultsManager(str(tmp_path), "m", "d", "b", 0, config={"debug_output": True})
    debug.save_object_times([result])
    assert (Path(debug.run_dir) / "time_measurements.csv").exists()


def test_config_and_summary_contracts_fail_closed(tmp_path):
    manager = ResultsManager(str(tmp_path), "m", "d", "b", 0, config={"x": 1})
    with pytest.raises(ValueError, match="constructor configuration"):
        manager.save_config({"x": 2})
    manager.summary_path = str(tmp_path / "summary.tsv")
    Path(manager.summary_path).write_text("wrong\n")
    with pytest.raises(ValueError, match="header"):
        manager.append_summary({"widget": {}}, [_result()])
