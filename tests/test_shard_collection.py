import json
from pathlib import Path

import pytest
import yaml

from scripts.experiments import combine_completed_shards


def shard(root, name, obj):
    output = root / name
    output.mkdir()
    state = dict(state="complete", started=1, ended=2, split="seen_development",
                 plan=dict(labels=["A", "B"], objects=[obj]),
                 completed=[dict(label=label, object=obj) for label in ("A", "B")])
    (output / "status.json").write_text(json.dumps(state))
    for label in ("A", "B"):
        run = output / label / f"run-{obj}"
        run.mkdir(parents=True)
        (run / "metrics.json").write_text(json.dumps({obj: {"score": .3}}))
        (run / "run_manifest.json").write_text(json.dumps(dict(objects={obj: {}}, run_id=obj)))
        (run / "config.yaml").write_text(yaml.safe_dump(dict(method=label, seed=0,
            dataset_config=dict(objects=[obj], preprocess="fixed"))))
        (output / label / "summary.tsv").write_text(f"object\tscore\n{obj}\t0.3\n")
    return output


def test_combine_preserves_evidence_and_is_noop(tmp_path):
    a, b = shard(tmp_path, "one", "can"), shard(tmp_path, "two", "fabric")
    original_manifest = next(a.rglob("run_manifest.json"))
    original_manifest.write_bytes(original_manifest.read_bytes() + b"\r\n")
    out = tmp_path / "combined"
    combine_completed_shards(tmp_path, out, [a, b])
    state = json.loads((out / "status.json").read_text())
    assert state["plan"] == dict(labels=["A", "B"], objects=["can", "fabric"])
    assert state["source_shards"] == ["one", "two"]
    assert len(state["completed"]) == 4
    assert (out / "A/summary.tsv").read_text() == "object\tscore\ncan\t0.3\nfabric\t0.3\n"
    for index, source in enumerate((a, b)):
        for label in ("A", "B"):
            for path in (source / label).rglob("run_manifest.json"):
                target = out / label / f"shard_{index}" / path.relative_to(source / label)
                assert path.read_bytes() == target.read_bytes()
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.rglob("*") if p.is_file()}
    combine_completed_shards(tmp_path, out, [a, b])
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}


@pytest.mark.parametrize("fault", ["overlap", "running", "missing", "config", "split", "duplicate", "labels"])
def test_reject_invalid_shards_without_output(tmp_path, fault):
    a, b = shard(tmp_path, "one", "can"), shard(tmp_path, "two", "can" if fault == "overlap" else "fabric")
    p = b / "status.json"
    state = json.loads(p.read_text())
    if fault == "running": state["state"] = "running"
    if fault == "split": state["split"] = "other"
    if fault == "duplicate": state["completed"].append(state["completed"][0])
    if fault == "labels": state["plan"]["labels"] = ["B", "A"]
    if fault == "missing": next(b.rglob("metrics.json")).unlink()
    if fault == "config":
        c = next(b.rglob("config.yaml"))
        config = yaml.safe_load(c.read_text());config["seed"] = 1
        c.write_text(yaml.safe_dump(config))
    p.write_text(json.dumps(state))
    out = tmp_path / "combined"
    with pytest.raises((ValueError, FileNotFoundError)):
        combine_completed_shards(tmp_path, out, [a, b])
    assert not out.exists()


def test_unowned_output_is_preserved(tmp_path):
    a = shard(tmp_path, "one", "can")
    out = tmp_path / "combined";out.mkdir()
    marker = out / "manual.txt";marker.write_text("preserve")
    with pytest.raises(ValueError, match="not owned"):
        combine_completed_shards(tmp_path, out, [a])
    assert marker.read_text() == "preserve"


def test_recover_stopped_partial_with_complementary_stages(tmp_path):
    import shutil
    partial, recovery = shard(tmp_path, "partial", "can"), shard(tmp_path, "recovery", "can")
    state = json.loads((partial / "status.json").read_text())
    state.update(state="invalid_run", completed=state["completed"][:1], error="OOM")
    (partial / "status.json").write_text(json.dumps(state))
    shutil.rmtree(partial / "B")
    state = json.loads((recovery / "status.json").read_text())
    state["plan"]["labels"] = ["B"]
    state["completed"] = state["completed"][1:]
    (recovery / "status.json").write_text(json.dumps(state))
    shutil.rmtree(recovery / "A")
    out = tmp_path / "combined"
    with pytest.raises(ValueError):
        combine_completed_shards(tmp_path, out, [partial, recovery])
    combine_completed_shards(tmp_path, out, [partial, recovery], recover_partial=True)
    state = json.loads((out / "status.json").read_text())
    assert state["state"] == "complete" and len(state["completed"]) == 2
    assert state["source_states"][0]["error"] == "OOM"
    with pytest.raises(ValueError, match="do not complete"):
        combine_completed_shards(tmp_path, tmp_path / "incomplete", [partial], recover_partial=True)
