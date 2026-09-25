import json
import hashlib

import pytest
import yaml

from scripts.experiments import import_completed_runs


def _config(tmp_path, name, objects=None, **method):
    path = tmp_path / "configs" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {"method": "m", "method_config": method}
    if objects is not None:
        config["dataset_config"] = {"objects": objects}
    path.write_text(yaml.safe_dump(config))
    return path


def _run(root, name, obj, value, data_root="/data/clean", config_extra=None,
         manifest_extra=None, object_extra=None, **method):
    run_dir = root / name
    run_dir.mkdir(parents=True)
    config = {"method": "m", "data_root": data_root, "device": "cuda:0",
              "method_config": {"k": 1, **method}}
    config.update(config_extra or {})
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
    (run_dir / "run_manifest.json").write_text(json.dumps({"objects": {obj: object_extra or {}},
                                                            **(manifest_extra or {})}))
    (run_dir / "metrics.json").write_text(json.dumps({obj: {"AUPRO_0_05": value}}))
    return run_dir


def _import(tmp_path, sources, configs, labels, output="out", objects=("bottle",)):
    return import_completed_runs(tmp_path, tmp_path / output, sources, configs, labels,
                                 "test_label_use_declared", objects)


def test_matches_declared_subset_and_copies_each_run(tmp_path):
    src = tmp_path / "raw"
    _run(src, "a", "bottle", 0.5, mode="gated")
    _run(src, "b", "cable", 0.6, mode="gated")
    _run(src, "c", "bottle", 0.4, mode="off")
    configs = [_config(tmp_path, "gated", mode="gated"), _config(tmp_path, "off", mode="off")]
    state = _import(tmp_path, [src], configs, ["gated", "off"], objects=("bottle", "cable"))
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
        _import(tmp_path, [src], [_config(tmp_path, "gated", mode="gated")], ["gated"], output="o3",
                objects=("bottle", "cable"))


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


def test_declared_scope_keeps_missing_object_partial(tmp_path):
    src = tmp_path / "raw"
    _run(src, "a", "bottle", 0.5, config_extra={"dataset_config": {"objects": ["bottle", "cable"]}})
    configs = [_config(tmp_path, "m", objects=["bottle", "cable"])]
    state = _import(tmp_path, [src], configs, ["m"], objects=None)
    assert state["state"] == "partial"
    assert state["plan"]["objects"] == ["bottle", "cable"]
    assert len(state["completed"]) == 1


def test_new_import_requires_scope_if_config_omits_it(tmp_path):
    src = tmp_path / "raw"
    _run(src, "a", "bottle", 0.5)
    with pytest.raises(ValueError, match="scope is ambiguous"):
        _import(tmp_path, [src], [_config(tmp_path, "m")], ["m"], objects=None)
    assert not (tmp_path / "out").exists()


def test_owned_import_preserves_scope_on_rerun(tmp_path):
    src = tmp_path / "raw"
    _run(src, "a", "bottle", 0.5)
    config = _config(tmp_path, "m")
    _import(tmp_path, [src], [config], ["m"], objects=("bottle", "cable"))
    status = tmp_path / "out" / "status.json"
    before = (status.read_bytes(), status.stat().st_mtime_ns)
    state = _import(tmp_path, [src], [config], ["m"], objects=None)
    assert state["state"] == "partial" and state["plan"]["objects"] == ["bottle", "cable"]
    assert (status.read_bytes(), status.stat().st_mtime_ns) == before

    owner = tmp_path / "out" / "import_sources.json"
    legacy = json.loads(owner.read_text())
    del legacy["objects"]
    owner.write_text(json.dumps(legacy))
    legacy_status = json.loads(status.read_text())
    legacy_status["import_sources"] = legacy
    status.write_text(json.dumps(legacy_status, indent=2) + "\n")
    before = (status.read_bytes(), status.stat().st_mtime_ns)
    state = _import(tmp_path, [src], [config], ["m"], objects=None)
    assert state["plan"]["objects"] == ["bottle", "cable"]
    assert (status.read_bytes(), status.stat().st_mtime_ns) == before


@pytest.mark.parametrize("change", [
    {"config_extra": {"seed": 1}},
    {"config_extra": {"protocol": {"shots": 2}}},
    {"config_extra": {"evaluation": {"mode": "other"}}},
    {"manifest_extra": {"code_sha256": "different"}},
    {"object_extra": {"protocol": {"support_content_sha256": "different"}}},
])
def test_equal_metrics_with_different_provenance_conflicts(tmp_path, change):
    src = tmp_path / "raw"
    _run(src, "a", "bottle", 0.5)
    _run(src, "b", "bottle", 0.5, **change)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        _import(tmp_path, [src], [_config(tmp_path, "m")], ["m"])


def test_rerun_identity_ignores_run_id_and_timing(tmp_path):
    src = tmp_path / "raw"
    _run(src, "a", "bottle", 0.5,
         manifest_extra={"run_id": "first", "code_sha256": "same"},
         object_extra={"protocol": {"support_content_sha256": "same"}, "elapsed": 1})
    _run(src, "b", "bottle", 0.5,
         manifest_extra={"run_id": "second", "code_sha256": "same"},
         object_extra={"protocol": {"support_content_sha256": "same"}, "elapsed": 2})
    state = _import(tmp_path, [src], [_config(tmp_path, "m")], ["m"])
    assert state["duplicates_dropped"] == ["raw/b"]

def test_real_config_hash_changes_with_output_dir_but_rerun_deduplicates(tmp_path):
    src = tmp_path / "raw"
    first = _run(src, "a", "bottle", 0.5, config_extra={"output_dir": "/runs/first", "seed": 0})
    second = _run(src, "b", "bottle", 0.5, config_extra={"output_dir": "/runs/second", "seed": 0})
    hashes = []
    for run_dir in (first, second):
        config = yaml.safe_load((run_dir / "config.yaml").read_text())
        # Exact ResultsManager hash algorithm; importing fmad requires torch,
        # which is not installed in the collector's lightweight test env.
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False)
        config_hash = hashlib.sha256(canonical.encode()).hexdigest()
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        manifest.update(config_hash=config_hash, run_id=run_dir.name, code_sha256="same")
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
        hashes.append(config_hash)
    assert hashes[0] != hashes[1]
    state = _import(tmp_path, [src], [_config(tmp_path, "m")], ["m"])
    assert state["duplicates_dropped"] == ["raw/b"]

    # The raw hash is ignored only because the normalized saved config and
    # object provenance are compared directly.
    changed = _run(src, "c", "bottle", 0.5,
                   config_extra={"output_dir": "/runs/third", "seed": 1})
    changed_config = yaml.safe_load((changed / "config.yaml").read_text())
    canonical = json.dumps(changed_config, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False)
    changed_manifest = json.loads((changed / "run_manifest.json").read_text())
    changed_manifest.update(config_hash=hashlib.sha256(canonical.encode()).hexdigest(),
                            run_id="third", code_sha256="same")
    (changed / "run_manifest.json").write_text(json.dumps(changed_manifest))
    with pytest.raises(ValueError, match="conflicting duplicate"):
        _import(tmp_path, [src], [_config(tmp_path, "m")], ["m"], output="changed_seed")
