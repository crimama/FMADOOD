import json

import pytest
import yaml

from scripts.experiments import import_completed_runs


def _config(tmp_path, name, **method):
    path = tmp_path / "configs" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"method": "m", "method_config": method}))
    return path


def _run(root, name, obj, value, data_root="/data/clean", **method):
    run_dir = root / name
    run_dir.mkdir(parents=True)
    config = {"method": "m", "data_root": data_root, "device": "cuda:0",
              "method_config": {"k": 1, **method}}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
    (run_dir / "run_manifest.json").write_text(json.dumps({"objects": {obj: {}}}))
    (run_dir / "metrics.json").write_text(json.dumps({obj: {"AUPRO_0_05": value}}))
    return run_dir


def _import(tmp_path, sources, configs, labels, output="out"):
    return import_completed_runs(tmp_path, tmp_path / output, sources, configs, labels, "test_label_use_declared")


def test_matches_declared_subset_and_copies_each_run(tmp_path):
    src = tmp_path / "raw"
    _run(src, "a", "bottle", 0.5, mode="gated")
    _run(src, "b", "cable", 0.6, mode="gated")
    _run(src, "c", "bottle", 0.4, mode="off")
    configs = [_config(tmp_path, "gated", mode="gated"), _config(tmp_path, "off", mode="off")]
    state = _import(tmp_path, [src], configs, ["gated", "off"])
    assert state["state"] == "partial"  # off/cable missing
    assert {(e["label"], e["object"]) for e in state["completed"]} == {
        ("gated", "bottle"), ("gated", "cable"), ("off", "bottle")}
    copied = sorted(p.parent.name for p in (tmp_path / "out").rglob("run_manifest.json"))
    assert copied == ["a", "b", "c"]


def test_identical_duplicate_dropped_conflicting_duplicate_rejected(tmp_path):
    src = tmp_path / "raw"
    _run(src, "a", "bottle", 0.5, mode="gated")
    _run(src, "b", "bottle", 0.5, mode="gated")
    configs = [_config(tmp_path, "gated", mode="gated")]
    state = _import(tmp_path, [src], configs, ["gated"])
    assert state["state"] == "complete" and state["duplicates_dropped"] == ["raw/b"]
    _run(src, "c", "bottle", 0.7, mode="gated")
    with pytest.raises(ValueError, match="conflicting duplicate"):
        _import(tmp_path, [src], configs, ["gated"], output="out2")


def test_rejects_unmatched_ambiguous_and_mixed_data_roots(tmp_path):
    src = tmp_path / "raw"
    _run(src, "a", "bottle", 0.5, mode="gated")
    with pytest.raises(ValueError, match="matches 0"):
        _import(tmp_path, [src], [_config(tmp_path, "x", mode="x")], ["x"])
    loose = [_config(tmp_path, "l1"), _config(tmp_path, "l2")]
    with pytest.raises(ValueError, match="matches 2"):
        _import(tmp_path, [src], loose, ["l1", "l2"], output="o2")
    _run(src, "b", "cable", 0.5, data_root="/data/noise", mode="gated")
    with pytest.raises(ValueError, match="data roots"):
        _import(tmp_path, [src], [_config(tmp_path, "gated", mode="gated")], ["gated"], output="o3")


def test_rerun_is_noop_and_foreign_output_rejected(tmp_path):
    src = tmp_path / "raw"
    _run(src, "a", "bottle", 0.5, mode="gated")
    configs = [_config(tmp_path, "gated", mode="gated")]
    _import(tmp_path, [src], configs, ["gated"])
    status = tmp_path / "out" / "status.json"
    before = (status.read_bytes(), status.stat().st_mtime_ns)
    _import(tmp_path, [src], configs, ["gated"])
    assert (status.read_bytes(), status.stat().st_mtime_ns) == before
    with pytest.raises(ValueError, match="not owned"):
        _import(tmp_path, [src], configs, ["other"])


def test_most_specific_config_wins_and_host_prefix_ignored(tmp_path):
    src = tmp_path / "raw"
    _run(src, "base", "bottle", 0.5, data_root="/home/a/Volume/DATA/MVTecAD", mode="gated")
    _run(src, "guard", "bottle", 0.6, data_root="/home/b/Volume/DATA/MVTecAD", mode="gated", guard=True)
    configs = [_config(tmp_path, "base", mode="gated"), _config(tmp_path, "guard", mode="gated", guard=True)]
    state = _import(tmp_path, [src], configs, ["base", "guard"])
    assert state["state"] == "complete" and state["data_root"] == "MVTecAD"
    assert {e["label"]: e["source_run"] for e in state["completed"]} == {"base": "raw/base", "guard": "raw/guard"}
